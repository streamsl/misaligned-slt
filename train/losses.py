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


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    # CE over valid positions, optionally per-class weighted. Normalized by valid-frame count
    # (not by the sum of class weights) so the scale stays comparable to the unweighted loss.
    if logits.ndim != targets.ndim + 1: raise ValueError(f"logits shape {tuple(logits.shape)} does not match targets {tuple(targets.shape)}")
    weight = None
    if class_weights is not None: weight = class_weights.to(dtype=logits.dtype, device=logits.device)
    token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), weight=weight, reduction="none").reshape_as(targets)
    
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


def bio_class_weight_tensor(class_weights: dict | list | None) -> torch.Tensor | None:
    """Build a length-4 BIO class-weight tensor (indexed UNK/O/B/I) from config.

    Accepts a {"O":..,"B":..,"I":..} dict or a 4-element list. UNK is forced to 0 (ignored).
    Returns None when no weights are given (→ plain unweighted CE, Moryossef's default recipe).
    """
    if not class_weights: return None
    if isinstance(class_weights, dict):
        w = [0.0, float(class_weights.get("O", 1.0)), float(class_weights.get("B", 1.0)), float(class_weights.get("I", 1.0))]
    else:
        w = [float(x) for x in class_weights]
        if len(w) != 4: raise ValueError(f"BIO class_weights list must have 4 entries (UNK,O,B,I); got {w}")
        w[BIO["UNK"]] = 0.0
    return torch.tensor(w, dtype=torch.float32)


def bio_nll_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = BIO["UNK"],
    dice_weight: float = 1.5, ce_weight: float = 1.0, class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """BIO loss: CE + weighted binary-signing Dice. Padding/UNK is ignored by both terms, never relabelled O.

    `class_weights` (length-4 UNK/O/B/I tensor) upweights the rare boundary/gap classes in the CE term. Moryossef 2026 
    used UNWEIGHTED CE because their joint *sign* head gave dense `B` supervision; we dropped the sign head (YouTube-SL-25 
    has no sign spans) and measured `B`=1.1% / `O`=11% of frames (corrected per-video-fps timeline), where unweighted CE + 
    binary Dice (which gives no B-vs-I signal at all) collapses to predicting all-`I`. Upweighting `B`/`O` is the standard 
    rare-class remedy and a justified, data-driven deviation. None ⇒ Moryossef's exact recipe.
    """
    valid = targets != ignore_index
    if not valid.any(): return logits.sum() * 0.0
    ce = masked_cross_entropy(logits, targets.clamp_min(0), valid, class_weights=class_weights) if ce_weight else logits.sum() * 0.0
    dice = binary_sign_dice_loss(logits, targets, ignore_index=ignore_index)
    return ce_weight * ce + dice_weight * dice


def confidence_bound_gate(
    full_tokens: torch.Tensor, trunc_tokens: torch.Tensor, trunc_confidence: torch.Tensor,
    reference_tokens: torch.Tensor | None = None, valid_mask: torch.Tensor | None = None,
    tau_cb: float = 0.75, verified_full_evidence_gate: bool = True, pad_token_id: int | None = None,
) -> torch.Tensor:
    """§6.3 active-slot gate, decoupled from the CE so the caller can re-mask the
    gated slots before the grad-bearing forward: (π_i > τ) & (t_i != f_i)
    [& (f_i == r_i) when the verified gate is on], minus padding/invalid slots."""
    active = trunc_confidence > float(tau_cb)
    active = active & (trunc_tokens != full_tokens)
    if verified_full_evidence_gate:
        if reference_tokens is None: raise ValueError("reference_tokens is required when verified_full_evidence_gate=True")
        active = active & (full_tokens == reference_tokens)

    if valid_mask is not None: active = active & valid_mask.to(device=active.device, dtype=torch.bool)
    if pad_token_id is not None:
        active = active & (full_tokens != int(pad_token_id))
        if reference_tokens is not None: active = active & (reference_tokens != int(pad_token_id))
        # TRUNC pads too: the decoder back-fills every slot after a committed EOS with pad @ FABRICATED
        # confidence 1.0 (infer/decode.py bookkeeping — π_j was never computed there). A right-truncated decode
        # legitimately ends earlier than the full-evidence one, so without this the whole post-EOS tail passes
        # (conf 1.0 > τ) & (pad != f) & (f == r) and receives dense CE toward the reference continuation — the
        # exact partial-target-on-truncated-input supervision P1 forbids ("uncertainty below τ_cb is free"
        # bypassed by a confidence the model never expressed). The early-EOS slot ITSELF keeps its real commit
        # confidence and stays eligible — confidently ending where full evidence continues IS a P1 error.
        active = active & (trunc_tokens != int(pad_token_id))
    return active


def confidence_bound_loss(
    trunc_logits: torch.Tensor, full_tokens: torch.Tensor, reference_tokens: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None, trunc_tokens: torch.Tensor | None = None, trunc_confidence: torch.Tensor | None = None,
    tau_cb: float = 0.75, verified_full_evidence_gate: bool = True, enabled: bool = True, pad_token_id: int | None = None,
    active_mask: torch.Tensor | None = None,
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

    if active_mask is not None: active = active_mask[:, :seq_len].to(device=logits.device, dtype=torch.bool)
    else: active = confidence_bound_gate(
        full_tokens=full, trunc_tokens=trunc_pred, trunc_confidence=trunc_conf,
        reference_tokens=ref, valid_mask=valid_mask[:, :seq_len] if valid_mask is not None else None,
        tau_cb=tau_cb, verified_full_evidence_gate=verified_full_evidence_gate, pad_token_id=pad_token_id,
    )
    if not active.any(): loss = logits.sum() * 0.0
    else:
        token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), full.reshape(-1), reduction="none").reshape_as(full)
        # Normalize by the VALID reference slots, not the gated slots: spec §6.3 writes L_cb as a position sum in the same form 
        # as OPUT's (which is per-valid-token). A per-ACTIVE-slot mean is sparsity-invariant — 1 gated slot in the batch would 
        # carry the same gradient magnitude as a fully-gated batch, giving Mode-2a windows (~9% of the batch) the majority of 
        # the translation gradient (the epoch-4 loss shock).
        denom = (valid_mask[:, :seq_len].to(device=token_loss.device, dtype=token_loss.dtype).sum()
                 if valid_mask is not None else token_loss.new_tensor(float(active.numel())))
        loss = (token_loss * active.to(token_loss.dtype)).sum() / denom.clamp(min=1)
    return ConfidenceBoundStats(
        loss=loss, active_positions=active, active_count=active.sum(),
        trunc_tokens=trunc_pred, trunc_confidence=trunc_conf,
    )
