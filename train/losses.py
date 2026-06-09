"""Stage-2 loss functions: the BIO Dice+CE term (Moryossef recipe) and the §6.3
confidence-bound term for right-truncated windows. OPUT lives in `models.dlm_decoder`."""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from data.windowing import BIO


@dataclass
class ConfidenceBoundStats:
    loss: torch.Tensor
    active_positions: torch.Tensor
    active_count: torch.Tensor
    trunc_tokens: torch.Tensor
    trunc_confidence: torch.Tensor


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    # Plain CE over valid positions; no class weights.
    if logits.ndim != targets.ndim + 1: raise ValueError(f"logits shape {tuple(logits.shape)} does not match targets {tuple(targets.shape)}")
    token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape_as(targets)
    if valid_mask is None: return token_loss.mean()
    mask = valid_mask.to(dtype=token_loss.dtype, device=token_loss.device)
    return (token_loss * mask).sum() / mask.sum().clamp(min=1.0)


def binary_sign_dice_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = BIO["UNK"], eps: float = 1e-6) -> torch.Tensor:
    # Moryossef-style Dice over signing vs non-signing, ignoring UNK padding.
    if logits.shape[:2] != targets.shape: raise ValueError(f"logits shape {tuple(logits.shape)} does not match targets {tuple(targets.shape)}")
    valid = targets != ignore_index
    if not valid.any(): return logits.sum() * 0.0
    probs = logits.softmax(dim=-1)
    pred_sign = (probs[..., BIO["B"]] + probs[..., BIO["I"]]) * valid.to(probs.dtype)
    gold_sign = (targets >= BIO["B"]).to(probs.dtype) * valid.to(probs.dtype)
    numerator = 2.0 * (pred_sign * gold_sign).sum()
    denominator = pred_sign.sum() + gold_sign.sum() + eps
    return 1.0 - numerator / denominator


def bio_nll_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = BIO["UNK"],
    dice_weight: float = 1.5, ce_weight: float = 1.0,
) -> torch.Tensor: # BIO loss used by default: unweighted CE plus weighted Dice.
    # The CE term is intentionally not class-weighted. Padding/UNK positions are ignored by both terms and must never be converted to O.
    valid = targets != ignore_index
    if not valid.any(): return logits.sum() * 0.0
    ce = masked_cross_entropy(logits, targets.clamp_min(0), valid) if ce_weight else logits.sum() * 0.0
    dice = binary_sign_dice_loss(logits, targets, ignore_index=ignore_index)
    return ce_weight * ce + dice_weight * dice


def confidence_bound_loss(
    trunc_logits: torch.Tensor, full_tokens: torch.Tensor, reference_tokens: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None, trunc_tokens: torch.Tensor | None = None, trunc_confidence: torch.Tensor | None = None,
    tau_cb: float = 0.75, verified_full_evidence_gate: bool = True, enabled: bool = True, pad_token_id: int | None = None,
) -> ConfidenceBoundStats:
    """Confidence-bound loss for right-truncated Mode 2a windows.

    With `verified_full_evidence_gate=True`, a truncated position is penalized
    only when the full-evidence decode is itself verified by the reference:

        (f_i == r_i) and (max p_trunc_i > tau_cb) and (argmax p_trunc_i != f_i)

    This preserves P1: the right-truncated visual input never receives a partial text label. 
    The reference is used only to decide whether the full-evidence self-target is trustworthy at this slot.
    """
    if not enabled:
        zero = trunc_logits.sum() * 0.0
        probs = trunc_logits.softmax(dim=-1)
        conf, pred = probs.max(dim=-1)
        active = torch.zeros_like(pred, dtype=torch.bool)
        return ConfidenceBoundStats(zero, active, active.sum(), pred, conf)

    seq_len = min(trunc_logits.shape[1], full_tokens.shape[1])
    logits = trunc_logits[:, :seq_len]
    full = full_tokens[:, :seq_len].to(device=logits.device)
    ref = reference_tokens[:, :seq_len].to(device=logits.device) if reference_tokens is not None else None

    probs = logits.softmax(dim=-1)
    logits_conf, logits_pred = probs.max(dim=-1)
    if trunc_tokens is not None: trunc_pred = trunc_tokens[:, :seq_len].to(device=logits.device)
    else: trunc_pred = logits_pred
    if trunc_confidence is not None: trunc_conf = trunc_confidence[:, :seq_len].to(device=logits.device)
    else: trunc_conf = logits_conf

    active = trunc_conf > float(tau_cb)
    active = active & (trunc_pred != full)
    if verified_full_evidence_gate:
        if ref is None: raise ValueError("reference_tokens is required when verified_full_evidence_gate=True")
        active = active & (full == ref)

    if valid_mask is not None: active = active & valid_mask[:, :seq_len].to(device=logits.device, dtype=torch.bool)
    if pad_token_id is not None:
        active = active & (full != int(pad_token_id))
        if ref is not None: active = active & (ref != int(pad_token_id))

    if not active.any(): loss = logits.sum() * 0.0
    else:
        token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), full.reshape(-1), reduction="none").reshape_as(full)
        loss = (token_loss * active.to(token_loss.dtype)).sum() / active.sum().clamp(min=1)

    return ConfidenceBoundStats(
        loss=loss, active_positions=active, active_count=active.sum(),
        trunc_tokens=trunc_pred, trunc_confidence=trunc_conf,
    )
