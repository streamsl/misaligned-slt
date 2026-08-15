# Stage-2 losses: BIO Dice+CE (Moryossef recipe) and confidence-bound term for right-truncated windows. 
# OPUT lives in models/dmax.py (the model's `.dlm_decoder` attribute).
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from data.windowing import BIO, make_bio_labels


@dataclass
class ConfidenceBoundStats:
    loss: torch.Tensor
    active_positions: torch.Tensor
    active_count: torch.Tensor
    trunc_tokens: torch.Tensor
    trunc_confidence: torch.Tensor


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor | None = None, class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    # CE over valid positions, optionally per-class weighted. Normalized by valid-frame count 
    # (not by summed class weights) so the scale stays comparable to the unweighted loss.
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


def bio_label_counts(records, trusted_gap_s: float | None = None) -> list[int]:
    """Corpus BIO label histogram (UNK/O/B/I) over each record's full timeline, for `balanced` class weights.

    Calls the real labeller on a uniform per-video timeline rather than re-deriving prevalence, so UNK/trusted-gap
    handling can never drift from training. No poses are read.
    """
    counts = np.zeros(4, dtype=np.int64)
    for rec in records:
        fps = float(rec.pose.fps); duration = float(rec.pose.duration_s)
        n = max(1, int(round(duration * fps)))
        kw = {} if trusted_gap_s is None else {"trusted_gap_s": trusted_gap_s}
        labels = make_bio_labels(np.arange(n) / fps, rec.sentences, 0.0, duration, video_duration_s=duration, **kw)
        counts += np.bincount(np.asarray(labels), minlength=4)[:4]
    return [int(c) for c in counts]


def resolve_bio_class_weights(cfg: dict, records, trusted_gap_s: float | None = None) -> None:
    """Replace a `bio_class_weights: balanced` config entry with the concrete 4-list measured on `records`.

    Resolved once at setup, in place, so every consumer sees the same numbers and the run's saved config records
    the exact weights used — a corpus-derived weight vector is not reproducible from the string alone.
    """
    if str(cfg.get("bio_class_weights") or "").lower() != "balanced": return
    counts = bio_label_counts(records, trusted_gap_s=trusted_gap_s)
    cfg["bio_class_weights"] = balanced_bio_class_weights(counts)
    print(f"[bio] balanced class weights from {len(records)} train videos: UNK/O/B/I counts {counts} -> "
          f"{[round(w, 4) for w in cfg['bio_class_weights']]}", flush=True)


def balanced_bio_class_weights(label_counts) -> list[float]:
    """Inverse-sqrt-frequency BIO weights from MEASURED label counts, normalised to mean weight 1 over valid frames.

    Derived per corpus rather than pinned to one dataset's numbers: `B` is 1 frame per sentence, so its share is set by that corpus's 
    sentence rate and cannot be a constant. Inverse-sqrt, not inverse-frequency: at a sub-1% `B` share the latter asks for a ~100x weight, 
    which over-segments (Moryossef's CNN ablations) — sqrt is the standard dense-segmentation compromise. Mean-1 normalisation keeps the 
    CE scale unweighted-comparable so `lambda_bio` carries over.
    """
    counts = np.asarray(label_counts, dtype=np.float64)
    if counts.shape != (4,): raise ValueError(f"label_counts must have 4 entries (UNK,O,B,I); got {counts.tolist()}")
    valid = counts.copy(); valid[BIO["UNK"]] = 0.0
    total = valid.sum()
    if total <= 0 or (valid > 0).sum() < 2: raise ValueError(f"degenerate BIO label counts {counts.tolist()}")
    w = np.zeros(4)
    present = valid > 0
    w[present] = 1.0 / np.sqrt(valid[present] / total)
    w *= total / float((w * valid).sum())  # mean weight 1 per valid frame
    w[BIO["UNK"]] = 0.0
    return [float(x) for x in w]


def bio_class_weight_tensor(class_weights: dict | list | str | None, label_counts=None) -> torch.Tensor | None:
    """Length-4 BIO class-weight tensor (indexed UNK/O/B/I) from a {"O","B","I"} dict, a 4-element list, or
    "balanced" (derived from `label_counts` — see `balanced_bio_class_weights`; portable across corpora).
    UNK forced to 0 (ignored). None when no weights given → unweighted CE, Moryossef's default recipe."""
    if not class_weights: return None
    if isinstance(class_weights, str):
        if class_weights.lower() != "balanced": raise ValueError(f"Unknown bio_class_weights: {class_weights!r}")
        if label_counts is None: raise ValueError("bio_class_weights: balanced needs label_counts from the train split")
        w = balanced_bio_class_weights(label_counts)
    elif isinstance(class_weights, dict):
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
    """BIO loss: CE + weighted binary-signing Dice. Padding/UNK ignored by both terms, never relabelled O.

    `class_weights` (length-4 UNK/O/B/I) upweights the rare boundary/gap classes in CE. Moryossef 2026 used UNWEIGHTED CE — their joint 
    *sign* head gave dense `B` supervision. Without that head (YouTube-SL-25 has no sign spans), `B`=1.1% / `O`=11% of frames (corrected 
    per-video-fps timeline) and unweighted CE + binary Dice (no B-vs-I signal) collapses to all-`I`. None ⇒ Moryossef's exact recipe.
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
    """Active-slot gate, decoupled from the CE so the caller can re-mask gated slots before the grad-bearing
    forward: (π_i > τ) & (t_i != f_i) [& (f_i == r_i) with the verified gate on], minus padding/invalid slots."""
    active = trunc_confidence > float(tau_cb)
    active = active & (trunc_tokens != full_tokens)
    if verified_full_evidence_gate:
        if reference_tokens is None: raise ValueError("reference_tokens is required when verified_full_evidence_gate=True")
        active = active & (full_tokens == reference_tokens)

    if valid_mask is not None: active = active & valid_mask.to(device=active.device, dtype=torch.bool)
    if pad_token_id is not None:
        active = active & (full_tokens != int(pad_token_id))
        if reference_tokens is not None: active = active & (reference_tokens != int(pad_token_id))
        # TRUNC pads too: after a committed EOS the decoder back-fills every slot with pad @ FABRICATED confidence 1.0 (infer/decode.py 
        # bookkeeping — π_j was never computed). A truncated decode legitimately ends earlier than the full-evidence one, so without this 
        # the post-EOS tail passes the gate and gets dense CE toward the reference continuation — the partial-target-on-truncated-input 
        # supervision P1 forbids. The early-EOS slot ITSELF keeps its real commit confidence and stays eligible: confidently ending where 
        # full evidence continues IS a P1 error.
        active = active & (trunc_tokens != int(pad_token_id))
    return active


def confidence_bound_loss(
    trunc_logits: torch.Tensor, full_tokens: torch.Tensor, reference_tokens: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None, trunc_tokens: torch.Tensor | None = None, trunc_confidence: torch.Tensor | None = None,
    tau_cb: float = 0.75, verified_full_evidence_gate: bool = True, enabled: bool = True, pad_token_id: int | None = None,
    active_mask: torch.Tensor | None = None,
) -> ConfidenceBoundStats:
    """Confidence-bound loss for right-truncated Mode 2a windows: CE toward the full-evidence tokens on the slots `confidence_bound_gate` 
    marks active. With the verified gate on, the active-slot target ALGEBRAICALLY equals the reference token (gate requires f==r), so this 
    is verified error-triggered reference CE — a direct signal. What the self-decode teacher adds is the MASK (the slot is achievable from 
    full evidence at current capacity, and the truncated view is confidently wrong there) and the on-policy decoded context; neither can 
    inject a wrong label — teacher failure yields silence (no active slots), never corruption. Monitor cb_active_count for a dead gate.
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

    # Only the AR arm needs pi/argmax from the grad-bearing logits; the DLM arm supplies both from its decode.
    # Computing them unconditionally put a full-vocab (B,L,V) softmax inside the autograd region every step.
    if trunc_tokens is None or trunc_confidence is None:
        probs = logits.softmax(dim=-1)
        logits_conf, logits_pred = probs.max(dim=-1)
    trunc_pred = trunc_tokens[:, :seq_len].to(device=logits.device) if trunc_tokens is not None else logits_pred
    trunc_conf = trunc_confidence[:, :seq_len].to(device=logits.device) if trunc_confidence is not None else logits_conf

    if active_mask is not None: active = active_mask[:, :seq_len].to(device=logits.device, dtype=torch.bool)
    else: active = confidence_bound_gate(
        full_tokens=full, trunc_tokens=trunc_pred, trunc_confidence=trunc_conf,
        reference_tokens=ref, valid_mask=valid_mask[:, :seq_len] if valid_mask is not None else None,
        tau_cb=tau_cb, verified_full_evidence_gate=verified_full_evidence_gate, pad_token_id=pad_token_id,
    )
    if not active.any(): loss = logits.sum() * 0.0
    else:
        token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), full.reshape(-1), reduction="none").reshape_as(full)
        # Normalize by VALID reference slots, not gated slots: L_cb as a position sum in OPUT's form (per-valid-token). 
        # A per-ACTIVE-slot mean is sparsity-invariant — 1 gated slot would carry the same gradient magnitude as a 
        # fully-gated batch, giving Mode-2a windows most of the translation gradient.
        denom = (valid_mask[:, :seq_len].to(device=token_loss.device, dtype=token_loss.dtype).sum()
                 if valid_mask is not None else token_loss.new_tensor(float(active.numel())))
        loss = (token_loss * active.to(token_loss.dtype)).sum() / denom.clamp(min=1)
    return ConfidenceBoundStats(
        loss=loss, active_positions=active, active_count=active.sum(),
        trunc_tokens=trunc_pred, trunc_confidence=trunc_conf,
    )
