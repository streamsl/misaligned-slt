"""Multi-GPU data-parallel training (torchrun). One module; every trainer inherits it through `run_epoch_loop`.

WHY NOT `DistributedDataParallel`. DDP synchronises gradients from hooks it installs around `Module.forward`,
but no trainer here calls `forward`: each stage's `step_fn` calls `model.forward_loss(...)` (SLT) or the head's
own signature. Wrapping in DDP and calling `.module.forward_loss(...)` runs a completely UNSYNCHRONISED training
job that looks fine — every rank silently optimises its own shard. A wrapper module whose `forward` is the loss
would fix that, but this stack's loss graph is CONDITIONAL (the translation term is routed per window mode, the
gate/CB terms fire only for the modes that carry them), so ranks disagree about which parameters received
gradients and DDP then needs `find_unused_parameters=True` — extra graph traversal every step plus its own
failure modes. Explicit averaging after `backward()` is correct for ANY call pattern and any graph: parameters
with no gradient this step contribute an explicit zero, which is exactly their mathematical contribution.

`batch_size` in every config stays the GLOBAL batch and is split across ranks (`per_rank_batch_size`). The same
config therefore describes the same optimisation on 1 GPU and on 8, which is what keeps a calibrated recipe
(e.g. Uni-Sign's effective 32 in baseline_train.yaml) reproducible across machines and needs no LR rescaling.
"""
from __future__ import annotations
import os
import torch
import torch.distributed as dist


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

def init_distributed() -> torch.device | None:
    """Join the torchrun process group if launched under one; return this rank's device (else None).

    Idempotent and safe to call unconditionally: without torchrun's env vars this is a no-op and every trainer
    runs exactly as it did single-process.
    """
    if is_distributed(): return torch.device(f"cuda:{local_rank()}")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ: return None
    if int(os.environ["WORLD_SIZE"]) <= 1: return None
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    device = torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(device)
    print(f"[dist] rank {rank()}/{world_size()} on {device} ({backend})", flush=True)
    return device

def cleanup() -> None:
    if is_distributed(): dist.destroy_process_group()

def barrier() -> None:
    if is_distributed(): dist.barrier()

def per_rank_batch_size(global_batch: int) -> int:
    """Split a config's GLOBAL batch across ranks. Refuses a non-divisible split rather than silently changing
    the effective batch (which would make a multi-GPU run a different experiment from its single-GPU config)."""
    w = world_size()
    if w == 1: return int(global_batch)
    if int(global_batch) % w: raise SystemExit(
        f"batch_size {global_batch} is the GLOBAL batch and must divide the world size {w} "
        f"(configs/*.yaml batch_size). Use a batch_size divisible by {w}, or launch with a different --nproc-per-node."
    )
    return int(global_batch) // w


def average_gradients(params) -> None:
    """Average gradients across ranks in ONE flat all-reduce.

    Parameters whose gradient is None this step (a conditional branch that did not fire on THIS rank) get a
    materialised zero first: every rank must reduce the identical tensor list in the identical order, and zero is
    the correct contribution from a rank whose batch did not exercise that branch.
    """
    if not is_distributed(): return
    grads = []
    for p in params:
        if not p.requires_grad: continue
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
    """Mean a metrics dict across ranks so EVERY rank sees identical dev numbers.

    Load-bearing, not cosmetic: best-checkpoint selection, early stopping and LR-plateau decisions read these
    values. If ranks reduced only their own shard they would disagree about which epoch is best and when to
    stop — the ranks would diverge mid-run. Keys are sorted so the reduced vector is order-identical everywhere.
    """
    if not is_distributed() or not metrics: return metrics
    # Key sets are NOT identical across ranks: cb_*/oput_mode*/val_translation_* only exist on ranks whose batches contained 
    # that mode. Reducing per-rank sorted vectors would silently pair different keys (or hang on length mismatch), so gather 
    # the union first and mean each key over the ranks that produced it.
    gathered: list[list[str]] = [None] * world_size()  # type: ignore[list-item]
    dist.all_gather_object(gathered, sorted(metrics))
    keys = sorted(set().union(*gathered))
    device = torch.device(f"cuda:{local_rank()}") if torch.cuda.is_available() else torch.device("cpu")
    vals = torch.tensor([float(metrics.get(k, 0.0)) for k in keys], dtype=torch.float64, device=device)
    counts = torch.tensor([float(k in metrics) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return {k: float(v / c) for k, v, c in zip(keys, vals.tolist(), counts.tolist()) if c > 0}
