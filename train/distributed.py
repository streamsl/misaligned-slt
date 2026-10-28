"""Multi-GPU data-parallel training (torchrun). One module; every trainer inherits it through `run_epoch_loop`.

WHY NOT `DistributedDataParallel`. DDP synchronises gradients from hooks it installs around `Module.forward`,
but no trainer here calls `forward`: each stage's `step_fn` calls `model.forward_loss(...)` (SLT) or the head's
own signature. Wrapping in DDP and calling `.module.forward_loss(...)` runs a completely UNSYNCHRONISED training
job that looks fine — every rank silently optimises its own shard. A wrapper module whose `forward` is the loss
would fix that, but this stack's loss graph is CONDITIONAL (the translation term is routed per window mode, the
covered and critic text groups depend on each window's prediction), so ranks disagree about which parameters 
received gradients and DDP then needs `find_unused_parameters=True` — extra graph traversal every step plus its 
own failure modes. Explicit averaging after `backward()` is correct for ANY call pattern and any graph: parameters
with no gradient this step contribute an explicit zero, which is exactly their mathematical contribution.

`batch_size` in every config stays the GLOBAL batch: each rank loads its part of every global batch of 1 process
(data.loader.RankSlice). Inside a training step (`train_step`) everything that reads whole batch reads the global one: 
train-mode BatchNorm statistics (backbones.stgcn_block.masked_batch_norm), the BIO loss (train.losses.bio_nll_dice_loss) 
and the stage-2 text loss's token mean (train.slt.training_loss). So N ranks compute the losses, the averaged gradients 
and the running statistics of 1 process on same global batch, up to float rounding (tests/test_multi_gpu.py), and same 
config describes same optimisation on 1 GPU and on 8, with no LR rescaling. Each rank's gradient is world_size times 
the part of the global gradient that its rows give, so their mean (`average_gradients`) is the global gradient.

Exceptions: dropout, the DLM's OPUT masks and the stage-2 GT-crop jitter draw from each rank's own RNG stream, and a last
global batch with fewer items than ranks is dropped (data.loader.RankSlice). The Moryossef release model reads its zero padding,
so its trainer pads each rank's part to the global batch's longest chunk (moryossef26.trainer.global_batch_padding).
The BatchNorm backward sums are collectives too, so every rank must backpropagate into every train-mode BatchNorm of 
the step, in 1 order: autograd runs them in reverse creation order, and the stage-2 loss keeps a zero-valued path into 
the translator's window features on a rank with no text row (models.streaming_slt forward_loss). A loss that can leave 
a train-mode encoder without gradient on one rank (a gated arm with lambda_bio 0 would; train.slt refuses one) hangs the step.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import timedelta
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from utils import pick_device

_IN_STEP = False

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def is_main() -> bool: # Rank 0 owns every SIDE EFFECT: checkpoint writes, wandb, history.csv, progress bars, stdout summaries.
    return rank() == 0

def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))

def cleanup() -> None:
    if is_distributed(): dist.destroy_process_group()

def barrier() -> None:
    if is_distributed(): dist.barrier()

def per_rank_batch_size(global_batch: int) -> int:
    """This rank's batch of a GLOBAL dev batch whose items the ranks split (the offline monitor): ceil(global / world size).
    A dev batch sets memory and speed only, so a world size that does not divide it rounds up. Training sets split each global
    batch in data.loader.RankSlice instead."""
    return -(-int(global_batch) // world_size())

def broadcast_state(module: torch.nn.Module) -> None:
    """Copy rank 0's parameters and buffers to every rank once, as DDP does when it wraps a model. `average_gradients` keeps
    the ranks equal only if they START equal, and some inits can differ per rank: a random draw (the DLM [MASK] row) whenever
    the ranks' generators differ, or a fit on each rank's own GPU (the one-stream stitch, train/slt.py)."""
    if not is_distributed(): return
    for tensor in [*module.parameters(), *module.buffers()]: dist.broadcast(tensor.data, src=0)

def broadcast_object(obj):
    # Rank 0's picklable `obj` on every rank. The dev numbers: every rank then takes the same best-epoch and early-stop decisions.
    if not is_distributed(): return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]

def all_sum(tensor: torch.Tensor) -> torch.Tensor: # In place: the sum over ranks (no gradient). 1 process: unchanged.
    if is_distributed(): dist.all_reduce(tensor)
    return tensor

def in_step() -> bool:
    # True inside `train_step` under torchrun: train-mode BatchNorm statistics and the losses then read the global batch.
    return _IN_STEP

@contextmanager
def train_step():
    # The loop wraps each training step's forward in it (train.helpers.run_epoch_loop). Every rank runs the same collectives in it:
    # each train-mode BatchNorm call and each BIO loss of a step on every rank, in the same order. 1 process: a no-op.
    global _IN_STEP
    _IN_STEP = is_distributed()
    try: yield
    finally: _IN_STEP = False

def merge_dicts(local: dict) -> dict:
    # Union of every rank's dict (disjoint keys, picklable values), returned on every rank.
    if not is_distributed(): return local
    gathered: list[dict] = [None] * world_size()  # type: ignore[list-item]
    dist.all_gather_object(gathered, local)
    return {k: v for part in gathered for k, v in part.items()}


def init_distributed() -> torch.device | None:
    """Join the torchrun process group if launched under one; return this rank's device (else None).

    Idempotent and safe to call unconditionally: without torchrun's env vars this is a no-op and every trainer
    runs exactly as it did single-process.
    """
    if is_distributed(): return torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ: return None
    if int(os.environ["WORLD_SIZE"]) <= 1: return None
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # A rank waits inside a collective while rank 0 writes latest.pt (weights and optimizer moments) and model.pt, and while the
    # other ranks finish their share of a dev monitor; NCCL's default bound (10 min) can end a healthy run there.
    # ponytail: 1 fixed bound; a true hang now surfaces after an hour.
    dist.init_process_group(backend=backend, timeout=timedelta(hours=1))
    # pick_device also sets the TF32 matmul precision of a 1-process run (train.py calls it only without torchrun).
    device = pick_device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(device)
    print(f"[dist] rank {rank()}/{world_size()} on {device} ({backend})", flush=True)
    return device


def average_gradients(params) -> None:
    """Average gradients across ranks in ONE flat all-reduce.

    Parameters whose gradient is None this step (a conditional branch that did not fire on THIS rank) get a
    materialised zero first: every rank must reduce the identical tensor list in the identical order, and zero is
    the correct contribution from a rank whose batch did not exercise that branch.
    """
    if not is_distributed(): return
    params = [p for p in params if p.requires_grad]
    if not params: return
    # None on every rank must stay None: AdamW skips these parameters, including decay and moments.
    present = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=params[0].device)
    dist.all_reduce(present, op=dist.ReduceOp.MAX)
    grads = []
    for p, active in zip(params, present.tolist()):
        if not active: continue
        if p.grad is None: p.grad = torch.zeros_like(p)
        grads.append(p.grad)
        
    if not grads: return
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size())
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset:offset + n].view_as(g))
        offset += n


def reduce_metrics(metrics: dict[str, float]) -> dict[str, float]:
    # Mean a metrics dict (the epoch means of the train logs) across ranks, so every rank logs the same numbers.
    if not is_distributed() or not metrics: return metrics
    # Key sets are NOT identical across ranks: text_*/oput_mode*/val_translation_* only exist on ranks whose batches 
    # contained that mode. Reducing per-rank sorted vectors would silently pair different keys (or hang on length 
    # mismatch), so gather the union first and mean each key over the ranks that produced it.
    gathered: list[list[str]] = [None] * world_size()  # type: ignore[list-item]
    dist.all_gather_object(gathered, sorted(metrics))
    keys = sorted(set().union(*gathered))
    device = torch.device(f"cuda:{local_rank()}") if torch.cuda.is_available() else torch.device("cpu")
    vals = torch.tensor([float(metrics.get(k, 0.0)) for k in keys], dtype=torch.float64, device=device)
    counts = torch.tensor([float(k in metrics) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return {k: float(v / c) for k, v, c in zip(keys, vals.tolist(), counts.tolist()) if c > 0}


class _GatherRows(torch.autograd.Function):
    # Every rank's rows (one shape) on every rank. Every rank computes the same loss of them, so the gradient of its own rows is
    # the same on every rank, and their sum over ranks, the convention that `average_gradients` divides, is world_size times it.
    @staticmethod
    def forward(ctx, x):
        parts = [torch.empty_like(x) for _ in range(world_size())]
        dist.all_gather(parts, x.contiguous())
        ctx.rows = x.shape[0]
        return torch.cat(parts)

    @staticmethod
    def backward(ctx, grad):
        return grad[rank() * ctx.rows:(rank() + 1) * ctx.rows] * world_size()


def max_shape(tensor: torch.Tensor) -> list[int]:
    # In `train_step` under torchrun: the largest size of each dim of `tensor` over the ranks. Else its own shape.
    if not in_step(): return list(tensor.shape)
    shape = torch.tensor(tensor.shape, device=tensor.device)
    dist.all_reduce(shape, op=dist.ReduceOp.MAX)
    return shape.tolist()

def gather_rows(values: torch.Tensor, targets: torch.Tensor, fill: int) -> tuple[torch.Tensor, torch.Tensor]:
    # In `train_step` under torchrun: (B, T, ...) `values` and their (B, T) `targets` of every rank, each rank's part padded to
    #  longest part (targets with `fill`, which the loss ignores), so a loss of them is the loss of global batch. Else as given.
    if not in_step(): return values, targets
    rows, frames = max_shape(targets)
    grow = (0, frames - targets.shape[1], 0, rows - targets.shape[0])
    targets = F.pad(targets, grow, value=fill)
    parts = [torch.empty_like(targets) for _ in range(world_size())]
    dist.all_gather(parts, targets)
    return _GatherRows.apply(F.pad(values, (0, 0) * (values.dim() - 2) + grow)), torch.cat(parts)
