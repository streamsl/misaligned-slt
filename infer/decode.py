"""Block-diffusion decoding with DMax's soft parallel decoding (SPD) and self-revision.

`block_diffusion_decode` runs 1 cold-start decode under fixed conditioning. Blocks are decoded left to right. Inside a block, 
the masked slots are unmasked by DMax's rule (longest confident prefix, the leftmost mask as fallback), every committed slot 
of the block is re-predicted on each pass, and a soft mask/token embedding mixture carries the pass's belief into next forward. 
A row's block is final once no mask remains and every active slot is settled, or a pass changes nothing; the caller's `finalize` 
hook then extends its KV cache by that block. No state crosses streaming strides.

Reference: DMax dInfer `ThresholdParallelDecoder.decode_uniform` + `get_transfer_index_uniform` (parallel_strategy.py) and 
`BlockDiffusionRunner.decode_uniform` (generate_uniform.py). Greedy (temperature 0) only: dInfer's Gumbel sampling path isn't 
reproduced, and every dInfer evaluation runs temperature 0. 5 deviations are marked inline, each for its own reason: 
row-independent stopping (DMax stops a block batch-globally), a -inf MASK logit (a progress guarantee dInfer's uniform path 
lacks), a revision-pass budget in place of dInfer's whole-block step cap, padding from 1st EOS, and a separate cache-append pass. 
The last 3 change what the decode costs; docs/implementation_notes.md records each with its measurement.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# DMax Breakflag: a block is final once every active slot's max-prob reaches this. 0.8 is where label smoothing 0.2
# (configs/dlm.yaml) puts the per-token loss MINIMISER, not a ceiling — exceeding it costs 0.044 nats against a
# 2.99-nat floor — so this test does fire, and how often is a property of the trained model. It is a live part of the
# pass budget, not a dead constant: measured 466 passes against 488 with it disabled on the tiny fixture at block 16.
SETTLE_CONFIDENCE = 0.9


@dataclass
class DecodeResult:
    sequences: torch.Tensor
    confidence: torch.Tensor
    forwards: int       # denoising passes over the batch
    cache_appends: int  # hard passes that append a finished block to the KV cache (no LM head); 1 per non-final block


def longest_confident_prefix_mask(confidence: torch.Tensor, mask_index: torch.Tensor, threshold: float) -> torch.Tensor:
    # DMax dInfer parallel_strategy.py get_transfer_index_uniform (steps 2-5): the masked slots before the first
    # masked slot below `threshold`; the leftmost mask when there is none, so every pass commits at least 1 slot.
    mask_index = mask_index.bool()
    low = mask_index & (confidence < float(threshold))
    after_failure = torch.cumsum(low.long(), dim=1) > 0
    candidates = mask_index & ~after_failure
    first_mask = (torch.cumsum(mask_index.long(), dim=1) == 1) & mask_index
    return torch.where(candidates.any(dim=1, keepdim=True), candidates, first_mask)


def spd_hybrid_embeddings(
    embedding_layer: nn.Embedding, token_ids: torch.Tensor, logits: torch.Tensor, active_mask: torch.Tensor,
    mask_token_id: int, top_k: int = 1, renormalize: bool = True, eps: float = 1e-6,
) -> torch.Tensor:
    """DMax SPD embeddings for one block (parallel_strategy.py decode_uniform, soft-embedding part). Active non-mask
    slots get sum_k p_k e(y_k) + (1 - sum_k p_k) e(MASK), rescaled to the probability-weighted target norm (DMax Eq. 10); 
    every other slot keeps its hard embedding. DMax patches its previous embeddings in place and hard-refreshes changed slots; 
    rebuilding from the current ids gives the same tensor because here a committed slot never re-masks (the -inf MASK logit 
    below removes the re-mask case dInfer's hard refresh handles).
    """
    base = embedding_layer(token_ids).clone()
    active = active_mask.bool() & (token_ids != int(mask_token_id))
    if not active.any(): return base

    probs = F.softmax(logits.float(), dim=-1)
    k = min(int(top_k), probs.shape[-1])
    if k == 1: topk_probs, topk_indices = probs.max(dim=-1, keepdim=True)
    else: topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
    residual = torch.clamp(1.0 - topk_probs.sum(dim=-1, keepdim=True), min=0.0)

    topk_embeds = embedding_layer(topk_indices)
    mask_embed = embedding_layer(torch.full((1,), int(mask_token_id), dtype=torch.long, device=token_ids.device)).view(1, 1, -1)
    mixed = (topk_embeds * topk_probs.unsqueeze(-1)).sum(dim=2) + mask_embed * residual
    if renormalize:
        current_norm = torch.linalg.vector_norm(mixed, dim=-1, keepdim=True)
        expected_topk_norm = (torch.linalg.vector_norm(topk_embeds, dim=-1) * topk_probs).sum(dim=-1, keepdim=True)
        target_norm = expected_topk_norm + torch.linalg.vector_norm(mask_embed, dim=-1, keepdim=True) * residual
        mixed = mixed * (target_norm / (current_norm + eps))
    base[active] = mixed.to(base.dtype)[active]
    return base


def block_diffusion_decode(
    forward: Callable[[torch.Tensor, torch.Tensor | None, int, int], torch.Tensor],
    finalize: Callable[[torch.Tensor, int, int], None] | None, embedding_layer: nn.Embedding, initial_token_ids: torch.Tensor, 
    mask_token_id: int, block_size: int, threshold: float, eos_token_id: int | None = None, pad_token_id: int | None = None,
    spd_top_k: int = 1, spd_renormalize: bool = True,
) -> DecodeResult:
    """Cold-start block decode under fixed conditioning.

    `forward(ids, soft, lo, hi)` returns logits of slots [lo, hi) given the final prefix [0, lo); `soft` is SPD embedding of 
    those slots or None for hard ids. `finalize(ids, lo, hi)` is called once block [lo, hi) is final and later block follows, 
    so a cached `forward` can extend its prefix. Slots past `hi` never enter a forward: under block-causal attention they 
    cannot influence the block.

    Per block and per row: commit the longest run of masked slots whose max-prob clears `threshold` (at least 1), re-predict 
    every committed slot of the block, and stop when no mask remains and every active slot's max-prob reaches SETTLE_CONFIDENCE 
    or the pass changed nothing. At most 1 revision-only pass per active slot follows last commit. A row that has stopped is 
    frozen while its batch-mates finish, so a row decodes exactly as it would alone. EOS is an ordinary slot until its block is 
    final; the row then ends and the rest of its canvas is padding.
    """
    ids = initial_token_ids.clone()
    batch, length = ids.shape
    mask_id, block = int(mask_token_id), int(block_size)
    pad_id = int(eos_token_id if pad_token_id is None else pad_token_id) if eos_token_id is not None else pad_token_id
    confidence = torch.zeros_like(ids, dtype=torch.float32)
    confidence[ids != mask_id] = 1.0
    finished = torch.zeros(batch, dtype=torch.bool, device=ids.device)
    forwards = appends = 0

    for lo in range(0, length, block):
        hi = min(lo + block, length)
        active = ids[:, lo:hi] == mask_id  # the block's generated slots; a finished row has none
        if active.any():
            soft = None
            done = ~active.any(dim=1)
            revisions = torch.zeros(batch, dtype=torch.long, device=ids.device)
            for _ in range(2 * (hi - lo)):  # tight: >= 1 commit per pass and <= 1 revision pass per slot, so a
                # block of 16 can reach exactly 32 passes. Any third stop condition must re-derive this bound.
                logits = forward(ids, soft, lo, hi).to(torch.float32, copy=True)

                # DMax's threshold decoder keeps a MASK-predicting slot masked for another pass (get_transfer_index_threshold, 
                # rm_mask); its uniform decoder has no guard and can strand a MASK. A -inf MASK logit commits the runner-up 
                # instead, so every pass makes progress. It also drops MASK from the softmax, so every max-prob below 
                # (commit test, settle test, recorded confidence) is DMax's divided by 1 - p(MASK) and SPD residual is 
                # (DMax's - p(MASK)) / (1 - p(MASK)); the MASK head row starts at 0 and is never a target.
                logits[..., mask_id] = torch.finfo(logits.dtype).min
                forwards += 1
                maxp, x0 = logits.softmax(dim=-1).max(dim=-1)  # greedy x0 and its probability (_get_prob_stats, T = 0)
                slots = ids[:, lo:hi]
                masked = slots == mask_id
                had_mask = masked.any(dim=1)

                # decode_uniform: update_mask = high_conf_index | (active_index & ~mask_index). DMax reduces its
                # stopping tests over the whole batch (Breakflag after select_undecoded, a no-op without writeback),
                # so a row keeps being re-predicted while a batch-mate needs passes; here a stopped row is frozen.
                update = (longest_confident_prefix_mask(maxp, masked, threshold) | (active & ~masked)) & ~done.unsqueeze(1)
                changed = update & (x0 != slots)
                slots = torch.where(update, x0, slots)
                ids[:, lo:hi] = slots
                confidence[:, lo:hi] = torch.where(update, maxp, confidence[:, lo:hi])

                # decode_uniform's Breakflag per row: every active slot at max-prob >= SETTLE_CONFIDENCE, or nothing changed. 
                # DMax tests it before checking for masks; with a threshold above 0.9 that could end a block with a mask left, 
                # so here a row must be mask-free first.
                no_mask = ~(slots == mask_id).any(dim=1)
                settled = ((maxp >= SETTLE_CONFIDENCE) | ~active).all(dim=1)
                stable = ~changed.any(dim=1)

                # DMax caps whole loop at block_length passes (generate_uniform.py decode_uniform, `while step < block_length`) 
                # with 32-slot blocks. At 1 commit per pass the commits alone exhaust that cap and no pass is left for revision, 
                # so the cap here is 1 revision-only pass per active slot (<= 2 x active slots per block, the loop bound above) 
                # — DOUBLE dInfer's ceiling, which a pass-count comparison against a published DMax number has to account for.
                revisions += (~had_mask & ~done).long()
                done |= no_mask & (stable | settled | (revisions >= active.sum(dim=1)))
                if bool(done.all()): break
                soft = spd_hybrid_embeddings(
                    embedding_layer, slots, logits, active & (slots != mask_id) & ~done.unsqueeze(1), mask_id,
                    top_k=spd_top_k, renormalize=spd_renormalize,
                )
            if eos_token_id is not None:
                # DMax early stop (decode_uniform end: `orig_x[eos_idx, block_loc.end:] = eos_id`): a final block holding EOS 
                # ends its row. Padding starts at 1st EOS rather than at block end as callers read sequence up to its 1st EOS.
                is_eos = (ids[:, lo:hi] == int(eos_token_id)) & active
                for row in is_eos.any(dim=1).nonzero(as_tuple=False).flatten().tolist():
                    first = lo + int(is_eos[row].nonzero(as_tuple=False)[0])
                    if pad_id is not None: ids[row, first + 1:] = pad_id
                    confidence[row, first + 1:] = 1.0
                    finished[row] = True
                    
        if hi >= length or bool(finished.all()): break
        if finalize is not None:
            finalize(ids, lo, hi)
            appends += 1
    return DecodeResult(sequences=ids, confidence=confidence, forwards=forwards, cache_appends=appends)
