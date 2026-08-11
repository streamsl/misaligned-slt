"""SPD + DCD within-decode machinery (DMax §3.2 / DCD §4–5). `spd_dcd_decode` runs one cold-start decode 
under fixed conditioning: SPD carries the renormalized soft mask/token embedding state across steps, 
DCD's sliding window selects commits. No state crosses streaming strides."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecodeStep:
    token_ids: torch.Tensor
    confidence: torch.Tensor
    selected: torch.Tensor
    predicted: torch.Tensor | None = None  # raw per-position prediction (DMax self-revision)


@dataclass
class SPDDecodeResult:
    sequences: torch.Tensor
    confidence: torch.Tensor
    steps: int
    last_logits: torch.Tensor | None = None
    commit_logits: torch.Tensor | None = None


def sample_tokens(
    logits: torch.Tensor, temperature: float = 0.0,
    top_k: int | None = None, top_p: float | None = None,
    margin_confidence: bool = False, neg_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token confidence and sampled/argmax token ids.

    Order matches DCD sample_tokens: temperature, then top-p, then top-k, so cutoffs are computed on the tempered
    distribution. Identical at temperature=0 with no filters (our defaults).
    """
    if temperature and temperature > 0: logits = logits / float(temperature)
    if top_p is not None and float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove_sorted = cumulative > float(top_p)
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(logits, dtype=torch.bool)
        remove.scatter_(-1, sorted_indices, remove_sorted)
        logits = logits.masked_fill(remove, torch.finfo(logits.dtype).min)

    if top_k is not None:
        kth = torch.topk(logits, min(int(top_k), logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, torch.finfo(logits.dtype).min)

    probs = F.softmax(logits, dim=-1)
    if temperature and temperature > 0:
        token_ids = torch.distributions.Categorical(probs=probs).sample()
        confidence = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    else: confidence, token_ids = probs.max(dim=-1)

    if margin_confidence:
        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        if top2.shape[-1] == 1: confidence = top2[..., 0]
        else: confidence = top2[..., 0] - top2[..., 1]

    if neg_entropy: confidence = torch.sum(probs * torch.log(probs.clamp_min(1e-10)), dim=-1)
    return confidence, token_ids


def dcd_decode_num(
    confidence: torch.Tensor, candidates: torch.Tensor,
    algo: Literal["threshold", "fixed"] = "threshold", algo_param: int | float = 0.9,
) -> torch.Tensor: # DCD token count with the paper/code at-least-one fallback.
    candidates = candidates.to(dtype=torch.bool, device=confidence.device)
    if algo.endswith("fixed"):
        fixed = torch.full((confidence.shape[0],), int(algo_param), dtype=torch.long, device=confidence.device)
        return torch.minimum(candidates.sum(dim=1), fixed)
    if algo.endswith("threshold"):
        above = candidates & (confidence >= float(algo_param))
        return torch.maximum(above.sum(dim=1), candidates.any(dim=1).long())
    raise ValueError(f"Unsupported DCD decode algorithm: {algo}")


def dcd_select_indices(
    confidence: torch.Tensor,
    candidates: torch.Tensor,
    num_decode: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]: # Select highest-confidence candidate indices per batch row.
    masked_conf = torch.where(candidates.bool(), confidence, torch.full_like(confidence, -torch.inf))
    confs: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []

    for row in range(masked_conf.shape[0]):
        k = int(num_decode[row].item())
        if k <= 0:
            confs.append(masked_conf.new_zeros((0,)))
            indices.append(torch.empty((0,), dtype=torch.long, device=masked_conf.device))
            continue

        top = torch.topk(masked_conf[row], k)
        confs.append(top.values)
        indices.append(top.indices)
    return confs, indices


def longest_confident_prefix_mask(confidence: torch.Tensor, mask_index: torch.Tensor, threshold: float) -> torch.Tensor:
    # DMax longest-contiguous-prefix promotion with leftmost-mask fallback.
    mask_index = mask_index.bool()
    is_low_conf = mask_index & (confidence < float(threshold))
    after_first_failure = torch.cumsum(is_low_conf.long(), dim=1) > 0
    candidates = mask_index & (~after_first_failure)
    has_selection = candidates.any(dim=1, keepdim=True)
    first_mask = (torch.cumsum(mask_index.long(), dim=1) == 1) & mask_index
    return torch.where(has_selection, candidates, first_mask)


def spd_hybrid_embeddings(
    embedding_layer: nn.Embedding, token_ids: torch.Tensor,
    logits: torch.Tensor, active_mask: torch.Tensor, mask_token_id: int,
    top_k: int = 1, renormalize: bool = True, eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DMax SPD hybrid embeddings with Eq. 10 norm restoration.

    Active non-mask positions receive a soft state: sum_k p_k e(y_k) + (1 - sum_k p_k) e(MASK)
    rescaled to the probability-weighted target norm. Other positions use hard token embeddings.
    """
    base = embedding_layer(token_ids).clone()
    active = active_mask.bool() & (token_ids != int(mask_token_id))
    if not active.any():
        probs = F.softmax(logits.float(), dim=-1)
        return base, probs.max(dim=-1).values

    probs = F.softmax(logits.float(), dim=-1)
    k = min(int(top_k), probs.shape[-1])
    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
    residual = torch.clamp(1.0 - topk_probs.sum(dim=-1, keepdim=True), min=0.0)

    topk_embeds = embedding_layer(topk_indices)
    mask_ids = torch.full((1,), int(mask_token_id), dtype=torch.long, device=token_ids.device)
    mask_embed = embedding_layer(mask_ids).view(1, 1, -1)

    mixed = (topk_embeds * topk_probs.unsqueeze(-1)).sum(dim=2) + mask_embed * residual
    if renormalize:
        current_norm = torch.linalg.vector_norm(mixed, dim=-1, keepdim=True)
        topk_norms = torch.linalg.vector_norm(topk_embeds, dim=-1)
        expected_topk_norm = (topk_norms * topk_probs).sum(dim=-1, keepdim=True)
        mask_norm = torch.linalg.vector_norm(mask_embed, dim=-1, keepdim=True)
        target_norm = expected_topk_norm + mask_norm * residual
        mixed = mixed * (target_norm / (current_norm + eps))
    base[active] = mixed.to(base.dtype)[active]
    return base, topk_probs[..., 0]


def dcd_threshold_step(
    logits: torch.Tensor, token_ids: torch.Tensor, candidate_mask: torch.Tensor,
    threshold: float, temperature: float = 0.0, top_k: int | None = None, top_p: float | None = None,
    decode_algo: str = "threshold", decode_param: int | float | None = None,
) -> DecodeStep: # One DCD selection step over an existing masked-text window.
    confidence, predicted = sample_tokens(
        logits, temperature=temperature, top_k=top_k, top_p=top_p,
        margin_confidence=decode_algo.startswith("topk_margin"),
        neg_entropy=decode_algo.startswith("entropy"),
    )
    algo_param = threshold if decode_param is None else decode_param
    selected = torch.zeros_like(candidate_mask, dtype=torch.bool)

    if decode_algo in {"dmax_prefix", "longest_prefix", "prefix_threshold"}:
        selected = longest_confident_prefix_mask(confidence, candidate_mask, float(algo_param))
    else:
        num_decode = dcd_decode_num(confidence, candidate_mask, decode_algo, algo_param)
        _, selected_indices = dcd_select_indices(confidence, candidate_mask, num_decode)
        for row, indices in enumerate(selected_indices): selected[row, indices] = True
    updated = torch.where(selected, predicted, token_ids)
    return DecodeStep(token_ids=updated, confidence=confidence, selected=selected, predicted=predicted)


def _truncate_rows_after_eos(
    token_ids: torch.Tensor, confidence_out: torch.Tensor,
    eos_pos: torch.Tensor, changed_mask: torch.Tensor,
    eos_token_id: int | None, pad_id: int | None,
) -> torch.Tensor: # Scan changed positions for a (possibly new, earlier) EOS and pad after it.
    if eos_token_id is None or not changed_mask.any(): return eos_pos
    for row in range(token_ids.shape[0]):
        changed_indices = changed_mask[row].nonzero(as_tuple=False).flatten()
        if changed_indices.numel() == 0: continue
        eos_hits = changed_indices[token_ids[row, changed_indices] == int(eos_token_id)]
        if eos_hits.numel() == 0: continue

        first_eos = int(eos_hits.min().item())
        previous_eos = int(eos_pos[row].item())
        eos_pos[row] = min(previous_eos, first_eos)
        if first_eos + 1 < previous_eos and pad_id is not None:
            after = slice(first_eos + 1, previous_eos)
            token_ids[row, after] = int(pad_id)
            confidence_out[row, after] = 1.0
    return eos_pos


def spd_dcd_decode(
    logits_fn, embedding_layer: nn.Embedding, initial_token_ids: torch.Tensor, mask_token_id: int, 
    steps: int, threshold: float, eos_token_id: int | None = None, pad_token_id: int | None = None, 
    temperature: float = 0.0, top_k: int = 1, spd_renormalize: bool = True, spd_revision: bool = True,
    window_length: int | None = None, max_window_length: int | None = None, window_type: str = "sliding",
    decode_algo: str = "threshold", decode_param: int | float | None = None, sample_top_k: int | None = None,
    top_p: float | None = None, cache_type: str = "none", block_size: int | None = None, 
    settle_confidence: float = 0.9, fill_leftover_masks: bool = True,
) -> SPDDecodeResult:
    """Cold-start SPD + DCD decode under a fixed-conditioning logits function.

    `logits_fn(token_ids, soft_embeds)` must hold its visual conditioning fixed across iterations; `soft_embeds` is 
    SPD state local to this call. DCD sliding window: decode confident masked positions in [window_left, window_right), 
    advance the left edge past decoded positions, extend the right edge by the remaining masks.

    Termination is DYNAMIC, as in DMax/DCD (`window_*_decode` loops `while (tokens == mask).any()`), bounded only
    by a safety cap. `steps` (a.k.a. diffusion_steps) is NOT the commit budget — it bounds only the settle phase.

    `block_size` clips the window to `window_left`'s attention block: under block-causal (BD3LM) attention a deferred 
    token never sees later blocks, so waiting on them is informationless (DCD `window_causal_decode`, DMax per-block 
    SPD). DCD's Dynamic Block Extension is not ported — this mBART-BD3LM decoder trains at a fixed block size, so 
    variable-size expansion would be a separate experiment. `None` only for fully bidirectional decoders or ablations.

    Once all masks commit, `spd_revision` spends `steps` on DMax settle passes (`_settle`): re-argmax committed tokens 
    until stable or all confidences reach `settle_confidence` (dInfer parallel_strategy.py decode_uniform Breakflag: 0.9).

    Settle ORDER follows DMax: settle PER BLOCK to convergence BEFORE advancing, so block b+1 is denoised against a 
    SETTLED block b — `_settle` at each block crossing plus a final settle. Per-block settles are bounded by the block 
    size (DMax's `while step < block_length`), the final one by `steps`.
    """
    cache_type = str(cache_type)
    cache_aware = bool(getattr(logits_fn, "supports_dcd_cache", False))
    if cache_type != "none" and not cache_aware:
        raise NotImplementedError(
            "This logits_fn does not expose DCD KV-cache support; use cache_type='none' or a cache-aware decoder."
        )
    if window_type not in {"sliding", "static"}: raise ValueError(f"Unsupported DCD window_type: {window_type}")
    block = int(block_size) if block_size else None

    def _suppress_mask(logits: torch.Tensor) -> torch.Tensor:
        # Never let [MASK] win argmax/selection (DMax rm_mask, parallel_strategy.get_transfer_index_threshold:
        # `mask_index & (x0 != mask_id)`): a high-confidence MASK wastes a decode slot (the write keeps it masked)
        # and pollutes the confidence the commit gate reads. In-place is safe — logits are fresh per call.
        # Divergence: DMax keeps MASK in the softmax DENOMINATOR, -inf drops it, so our confidence is fractionally
        # higher — excluded mass ~0 at the trained vocab (1732, MASK never a CE target), visible only at toy vocabs.
        logits[..., int(mask_token_id)] = torch.finfo(logits.dtype).min
        return logits

    def _block_end(left: int) -> int: # End of the attention block containing `left` (exclusive).
        if block is None: return 1 << 30
        return (left // block + 1) * block

    token_ids = initial_token_ids.clone()
    batch, full_length = token_ids.shape
    device = token_ids.device
    prompt_length = 0
    for pos in range(full_length):
        if (token_ids[:, pos] == int(mask_token_id)).any(): break
        prompt_length += 1
    if prompt_length >= full_length:
        return SPDDecodeResult(sequences=token_ids, confidence=torch.ones_like(token_ids, dtype=torch.float32), steps=0)

    # DCD has no step budget (decode_algorithm.py window_*_decode): the threshold rule commits >=1 token/step, so
    # a decode ends in <= (#generated slots) forwards and capping by `steps` would force-fill the tail of any
    # sequence needing more. Cap scales with BATCH: the window tracks the min first-mask over rows, so >=1 commit
    # is guaranteed GLOBALLY, not per row — desynchronized rows approach batch * (#generated slots).
    commit_cap = 2 * batch * (full_length - prompt_length) + int(window_length or full_length) + 1
    win_len = int(window_length or (full_length - prompt_length))
    win_len = max(1, min(win_len, full_length - prompt_length))
    max_win = int(max_window_length or full_length)
    max_win = max(win_len, max_win)
    pad_id = int(eos_token_id if pad_token_id is None else pad_token_id) if eos_token_id is not None else pad_token_id

    confidence_out = torch.zeros_like(token_ids, dtype=torch.float32)
    confidence_out[token_ids != int(mask_token_id)] = 1.0
    soft_embeds, used_steps = None, 0
    last_logits, commit_logits = None, None
    window_left = prompt_length
    window_right = min(full_length, prompt_length + win_len)
    eos_pos = torch.full((batch,), full_length, dtype=torch.long, device=device)
    try:
        logits_fn_params = inspect.signature(logits_fn).parameters
        accepts_window = len(logits_fn_params) >= 3
    except (TypeError, ValueError): accepts_window = cache_aware

    generated_region = torch.arange(full_length, device=device).unsqueeze(0) >= prompt_length
    positions_row = torch.arange(full_length, device=device).unsqueeze(0)
    _block_start = (lambda p: (p // block) * block) if block is not None else (lambda p: prompt_length)

    def _full_forward(ids: torch.Tensor, embeds: torch.Tensor | None) -> torch.Tensor:
        if cache_type != "none" and accepts_window: return logits_fn(ids, embeds, None)
        return logits_fn(ids, embeds)

    def _settle(budget: int, soft: torch.Tensor | None, lo: int, hi: int) -> None:
        """DMax `decode_uniform` Breakflag revision of [lo, hi) to self-consistency (stop when all active max-probs
        clear settle_confidence=0.9, or nothing changes). Only committed (non-mask, pre-EOS) tokens in [lo, hi) are
        revisable — DMax never re-enters a FINISHED block (`decode_uniform` writes only `x[:, block_start:block_end]`): 
        later blocks were committed against it."""
        nonlocal token_ids, commit_logits, eos_pos, last_logits, used_steps
        if not spd_revision: return
        for _ in range(max(1, int(budget))):
            revisable = (token_ids != int(mask_token_id)) & generated_region
            revisable &= (positions_row >= int(lo)) & (positions_row < int(hi))
            revisable &= positions_row < eos_pos.unsqueeze(1)
            if pad_id is not None: revisable &= token_ids != int(pad_id)
            if not revisable.any(): return
            full_logits = _suppress_mask(_full_forward(token_ids, soft))
            last_logits = full_logits
            conf, pred = sample_tokens(full_logits, temperature=temperature, top_k=sample_top_k, top_p=top_p)
            changed = revisable & (pred != token_ids)
            token_ids = torch.where(revisable, pred, token_ids)
            confidence_out[revisable] = conf[revisable]
            if changed.any():
                if commit_logits is None: commit_logits = torch.zeros_like(full_logits)
                commit_logits = torch.where(changed.unsqueeze(-1), full_logits, commit_logits)
                eos_pos = _truncate_rows_after_eos(token_ids, confidence_out, eos_pos, changed, eos_token_id, pad_id)
                if soft is not None:
                    # `inputs_embeds` REPLACES ids, so hard-refresh revised positions (DMax parallel_strategy
                    # does this every iteration) or later passes re-score pre-settle tokens and multi-pass
                    # settling collapses to one effective pass.
                    soft = torch.where(changed.unsqueeze(-1), embedding_layer(token_ids), soft)
            used_steps += 1
            # Breakflag on the ARGMAX max-prob (DMax `max_probs >= 0.9`), not the sampled token's prob: 
            # identical at temperature=0, but at temperature>0 the sampled prob would gate on noise.
            maxp = full_logits.softmax(dim=-1).max(dim=-1).values
            if not changed.any() or bool((maxp[revisable] >= float(settle_confidence)).all().item()): return

    settled_block_start = _block_start(prompt_length)  # blocks already settled-before-advance
    for step in range(commit_cap):
        generated_region = torch.arange(full_length, device=device).unsqueeze(0) >= prompt_length
        if not ((token_ids == int(mask_token_id)) & generated_region).any(): break

        if window_type == "sliding":
            while window_left < full_length and (token_ids[:, window_left] != int(mask_token_id)).all():
                window_left += 1

        # Settle-before-advance: once the window enters a NEW block, the block(s) it left are fully committed —
        # settle them now, so the block about to be committed (and any EOS in it, which eos_pos freezes
        # irreversibly) conditions on a settled prefix, not a half-decoded one.
        if block is not None and _block_start(window_left) > settled_block_start:
            # Only the block(s) just left: earlier ones are frozen, the new one is incomplete. HARD embeds (None).
            _settle(block, None, settled_block_start, _block_start(window_left))
            settled_block_start = _block_start(window_left)
            # Settle may have revised the prefix while the soft state still embeds the PRE-settle one, so rebuild
            # it HARD (DMax resets embeddings each block). Rebuild rather than None: the prefix-cache window
            # forward needs inputs_embeds, and its native embedding has no [MASK] row.
            soft_embeds = embedding_layer(token_ids)

        if window_left >= full_length: break
        window_right = max(window_right, window_left + 1)
        # Clip to window_left's block (DCD window_causal_decode clips to block_right likewise).
        window_right = min(window_right, full_length, int(eos_pos.max().item()), window_left + max_win, _block_end(window_left))
        if window_right <= window_left: break
        if cache_type != "none" and accepts_window: full_logits = logits_fn(token_ids, soft_embeds, (window_left, window_right))
        else: full_logits = logits_fn(token_ids, soft_embeds)
        full_logits = _suppress_mask(full_logits)

        last_logits = full_logits
        logits = full_logits[:, window_left:window_right]
        candidate = token_ids[:, window_left:window_right] == int(mask_token_id)
        if not candidate.any():
            # Static-window jump (DCD static advance); sliding mode never lands here (left-edge advance stops at
            # the 1st mask). Without it a block-clipped static window stalls on a finished block.
            window_left = window_right
            window_right = min(full_length, int(eos_pos.max().item()), window_left + win_len, _block_end(window_left))
            if window_right <= window_left: break
            continue

        decoded = dcd_threshold_step(
            logits=logits, token_ids=token_ids[:, window_left:window_right],
            candidate_mask=candidate, threshold=threshold,
            temperature=temperature, top_k=sample_top_k, top_p=top_p,
            decode_algo=decode_algo, decode_param=decode_param,
        )
        newly_selected = decoded.selected & candidate
        selected_global = torch.zeros_like(token_ids, dtype=torch.bool)
        selected_global[:, window_left:window_right] = newly_selected

        if commit_logits is None: commit_logits = torch.zeros_like(full_logits)
        commit_logits = torch.where(selected_global.unsqueeze(-1), full_logits, commit_logits)
        token_ids[:, window_left:window_right] = decoded.token_ids
        confidence_window = confidence_out[:, window_left:window_right]
        confidence_window[newly_selected] = decoded.confidence[newly_selected]

        if spd_revision and decoded.predicted is not None:
            # DMax self-revision (decode_uniform: update_mask = high_conf | (active & ~mask)): refresh every committed token 
            # in the window with this step's prediction, so later siblings overturn early errors — the recovery OPUT's L_pred 
            # trains. Frozen: prefix left of window_left (DCD deferred commitment), pad fills, anything at/after EOS.
            window_tokens = token_ids[:, window_left:window_right]
            revisable = (~candidate) & (window_tokens != int(mask_token_id))
            if pad_id is not None: revisable &= window_tokens != int(pad_id)

            positions = torch.arange(window_left, window_right, device=device).unsqueeze(0)
            revisable &= positions < eos_pos.unsqueeze(1)
            if revisable.any():
                revised_changed = revisable & (decoded.predicted != window_tokens)
                token_ids[:, window_left:window_right] = torch.where(revisable, decoded.predicted, window_tokens)
                confidence_window[revisable] = decoded.confidence[revisable]
                rev_global = torch.zeros_like(token_ids, dtype=torch.bool)
                rev_global[:, window_left:window_right] = revisable
                commit_logits = torch.where(rev_global.unsqueeze(-1), full_logits, commit_logits)
                # Revised positions join the EOS scan: a revision into EOS truncates like a fresh EOS commit.
                changed_global = torch.zeros_like(token_ids, dtype=torch.bool)
                changed_global[:, window_left:window_right] = revised_changed
                selected_global = selected_global | changed_global
                newly_selected = newly_selected | revised_changed

        if eos_token_id is not None and newly_selected.any():
            for row in range(batch):
                selected_indices = selected_global[row].nonzero(as_tuple=False).flatten()
                if selected_indices.numel() == 0: continue
                selected_tokens = token_ids[row, selected_indices]
                eos_selected = selected_indices[selected_tokens == int(eos_token_id)]

                if eos_selected.numel() == 0: continue
                first_eos = int(eos_selected.min().item())
                previous_eos = int(eos_pos[row].item())
                eos_pos[row] = min(eos_pos[row], first_eos)

                if first_eos + 1 < previous_eos and pad_id is not None:
                    after = slice(first_eos + 1, previous_eos)
                    token_ids[row, after] = int(pad_id)
                    confidence_out[row, after] = 1.0

        if window_type == "sliding":
            old_window_right = window_right
            remaining_masks = (token_ids[:, window_left:window_right] == int(mask_token_id)).sum(dim=1)
            window_right = min(
                full_length, int(eos_pos.max().item()),
                window_left + max_win, window_right + win_len - int(remaining_masks.max().item()),
            )
            if window_right < old_window_right: window_right = old_window_right
        elif (token_ids[:, window_left:window_right] != int(mask_token_id)).all():
            window_left = window_right
            window_right = min(full_length, int(eos_pos.max().item()), window_left + win_len)

        active = (token_ids != int(mask_token_id)) & generated_region
        # DMax scopes SPD soft state to current block (decode_uniform builds soft embeds for the block only). Mapped to DCD: 
        # only [window_left, window_right) stays soft, the committed prefix is hard — deferred commitment, in all cache modes.
        window_active = torch.zeros_like(active)
        window_active[:, window_left:window_right] = True
        active = active & window_active
        if pad_id is not None: active = active & (token_ids != int(pad_id))
        soft_embeds, _ = spd_hybrid_embeddings(
            embedding_layer=embedding_layer, token_ids=token_ids,
            logits=full_logits, active_mask=active, mask_token_id=mask_token_id,
            top_k=top_k, renormalize=spd_renormalize,
        )
        used_steps += 1  # accumulate: `= step + 1` wiped the settle passes _settle() counts at block crossings

    # Masks left: fill in 1 forced pass so no [MASK] id reaches the tokenizer or commit gate.
    leftover = (token_ids == int(mask_token_id)) & generated_region
    if fill_leftover_masks and leftover.any():
        # Unreachable normally (the cap covers the batch worst case); if it fires, the decode was force-terminated early.
        print(f"[decode] WARNING: commit loop hit its safety cap with {int(leftover.sum())} masked slots left; "
              f"force-filling in one pass (premature termination — investigate confidence/threshold settings)", flush=True)
        full_logits = _suppress_mask(_full_forward(token_ids, soft_embeds))
        last_logits = full_logits
        conf, pred = sample_tokens(full_logits, temperature=temperature, top_k=sample_top_k, top_p=top_p)
        token_ids = torch.where(leftover, pred, token_ids)
        confidence_out[leftover] = conf[leftover]

        if commit_logits is None: commit_logits = torch.zeros_like(full_logits)
        commit_logits = torch.where(leftover.unsqueeze(-1), full_logits, commit_logits)
        eos_pos = _truncate_rows_after_eos(token_ids, confidence_out, eos_pos, leftover, eos_token_id, pad_id)
        used_steps += 1

    # Last block only — earlier ones settled at their boundary crossings and are frozen. `steps` bounds 
    # ONLY this phase, counted separately so a long commit never starves it; converges in 1-3 passes.
    if not ((token_ids == int(mask_token_id)) & generated_region).any():
        _settle(max(1, int(steps)), soft_embeds, settled_block_start, full_length)

    return SPDDecodeResult(
        sequences=token_ids, confidence=confidence_out, steps=used_steps, 
        last_logits=last_logits, commit_logits=commit_logits,
    )
