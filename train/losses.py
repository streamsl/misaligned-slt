# Stage losses: BIO Dice+CE (S1 and stage 2 alike) and the confidence-bound term for right-truncated windows.
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
    active_count: torch.Tensor


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor | None = None, 
    class_weights: torch.Tensor | None = None, label_smoothing: float = 0.0,
) -> torch.Tensor:
    # CE over valid positions, optionally per-class weighted. Normalized by valid-frame count 
    # (not by summed class weights) so the scale stays comparable to the unweighted loss.
    if logits.ndim != targets.ndim + 1: raise ValueError(f"logits shape {tuple(logits.shape)} does not match targets {tuple(targets.shape)}")
    weight = None
    if class_weights is not None: weight = class_weights.to(dtype=logits.dtype, device=logits.device)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), weight=weight, 
        reduction="none", label_smoothing=float(label_smoothing)
    ).reshape_as(targets)
    
    if valid_mask is None: return token_loss.mean()
    mask = valid_mask.to(dtype=token_loss.dtype, device=token_loss.device)
    return (token_loss * mask).sum() / mask.sum().clamp(min=1.0)


def binary_sign_dice_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = BIO["UNK"], eps: float = 1e-6) -> torch.Tensor:
    # Dice over caption-unit foreground (B/I) versus outside (O), ignoring unknown frames.
    if logits.shape[:2] != targets.shape: raise ValueError(f"logits shape {tuple(logits.shape)} does not match targets {tuple(targets.shape)}")
    valid = targets != ignore_index
    if not valid.any(): return logits.sum() * 0.0
    probs = logits.softmax(dim=-1)
    pred_sign = (probs[..., BIO["B"]] + probs[..., BIO["I"]]) * valid.to(probs.dtype)
    gold_sign = (targets >= BIO["B"]).to(probs.dtype) * valid.to(probs.dtype)
    numerator = 2.0 * (pred_sign * gold_sign).sum()
    denominator = pred_sign.sum() + gold_sign.sum() + eps
    return 1.0 - numerator / denominator


def bio_label_counts(records) -> list[int]:
    """Corpus BIO label histogram (UNK/O/B/I) over each record's full timeline, for `balanced` class weights.

    Calls the real labeller on a uniform per-video timeline rather than re-deriving prevalence, so UNK/trusted-gap 
    handling can never drift from training. No poses are read.
    """
    counts = np.zeros(4, dtype=np.int64)
    for rec in records:
        fps = float(rec.pose.fps); duration = float(rec.pose.duration_s)
        n = max(1, int(round(duration * fps)))
        labels = make_bio_labels(np.arange(n) / fps, rec.sentences, 0.0, duration, video_duration_s=duration)
        counts += np.bincount(np.asarray(labels), minlength=4)[:4]
    return [int(c) for c in counts]


def resolve_bio_class_weights(cfg: dict, records) -> None:
    """Replace a `bio_class_weights: balanced` config entry with the concrete 4-list measured on `records`.

    Resolved once at setup, in place, so every consumer sees the same numbers and the run's saved config records
    the exact weights used — a corpus-derived weight vector is not reproducible from the string alone.
    """
    if str(cfg.get("bio_class_weights") or "").lower() != "balanced": return
    counts = bio_label_counts(records)
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
    *sign* head gave dense `B` supervision. Without that head (YouTube-SL-25 has no sign spans), `B` is ~1 frame per caption unit (under
    1% of frames) and unweighted CE + binary Dice (no B-vs-I signal) collapses to all-`I`. None ⇒ unweighted CE; Dice term on the phrase 
    head is our adaptation (their Dice sits on the sign head).
    """
    valid = targets != ignore_index
    if not valid.any(): return logits.sum() * 0.0
    ce = masked_cross_entropy(logits, targets.clamp_min(0), valid, class_weights=class_weights) if ce_weight else logits.sum() * 0.0
    dice = binary_sign_dice_loss(logits, targets, ignore_index=ignore_index)
    return ce_weight * ce + dice_weight * dice


def confidence_bound_gate(
    full_tokens: torch.Tensor, trunc_tokens: torch.Tensor, trunc_confidence: torch.Tensor, reference_tokens: torch.Tensor, 
    valid_mask: torch.Tensor | None = None, tau_cb: float = 0.75, pad_token_id: int | None = None,
) -> torch.Tensor:
    """Active-slot gate, decoupled from the CE so the caller can re-mask gated slots before the grad-bearing
    forward: (π_i > τ) & (t_i != f_i) & (f_i == r_i), minus padding/invalid slots. The verified term (f_i == r_i)
    keeps only slots where the full-evidence decode is right, so a wrong teacher token never marks a slot."""
    active = trunc_confidence > float(tau_cb)
    active = active & (trunc_tokens != full_tokens) & (full_tokens == reference_tokens)

    if valid_mask is not None: active = active & valid_mask.to(device=active.device, dtype=torch.bool)
    if pad_token_id is not None:
        active = active & (full_tokens != int(pad_token_id)) & (reference_tokens != int(pad_token_id))
        # TRUNC pads too: after a committed EOS the decoder back-fills every slot with pad @ FABRICATED confidence 1.0 (infer/decode.py 
        # bookkeeping — π_j was never computed). A truncated decode legitimately ends earlier than the full-evidence one, so without 
        # this the post-EOS tail passes the gate and its fabricated-confidence pad slots get a loss they never earned. The early-EOS 
        # slot ITSELF keeps its real commit confidence and stays eligible: confidently ending where full evidence continues is exactly 
        # the error the unlikelihood term lowers.
        active = active & (trunc_tokens != int(pad_token_id))
    return active


def confidence_bound_loss(
    trunc_logits: torch.Tensor, trunc_tokens: torch.Tensor, active_mask: torch.Tensor, valid_mask: torch.Tensor,
) -> ConfidenceBoundStats:
    """Confidence bound for right-truncated (Mode 2a) windows: UNLIKELIHOOD on the truncated decode's OWN token at the slots
    `confidence_bound_gate` marks active (the truncated decode is confident, and it differs from a full-evidence decode that
    equals the reference). L = -log(1 - p(t)) lowers the probability of that wrong token and names no target token (unlikelihood 
    training, Welleck et al., ICLR 2020). The reference only decides WHERE the truncated view is confidently wrong, never WHAT 
    it should say there, so the truncated input gets no text target (P1): a CE toward the full-evidence token would teach words 
    whose evidence can lie past the cut. The push fades as p(t) falls, and a slot leaves the gate once p(t) <= tau_cb. The
    effect is per-token confidence at 1st confidently wrong slot. The FSM commit has no confidence condition: confidence is 
    a reported score that feeds the reveal policies (infer/stability.py).
    `trunc_tokens` and `active_mask` cover 1st seq_len slots of `trunc_logits`; `valid_mask` marks the valid reference slots.
    """
    seq_len = trunc_tokens.shape[1]
    logits = trunc_logits[:, :seq_len]
    if not active_mask.any(): loss = logits.sum() * 0.0
    else: # -log(1 - p(t)) = logsumexp(all) - logsumexp(all but t): stable even as p(t) -> 1, and exact under bf16 autocast.
        others = logits.scatter(-1, trunc_tokens.unsqueeze(-1), float("-inf"))
        token_loss = torch.logsumexp(logits.float(), dim=-1) - torch.logsumexp(others.float(), dim=-1)
        # Normalize by VALID reference slots, not gated slots: L_cb as a position sum in OPUT's form (per-valid-token).
        # A per-ACTIVE-slot mean is sparsity-invariant — 1 gated slot would carry the same gradient magnitude as a
        # fully-gated batch, giving Mode-2a windows most of the translation gradient.
        denom = valid_mask[:, :seq_len].to(device=token_loss.device, dtype=token_loss.dtype).sum()
        loss = (token_loss * active_mask.to(token_loss.dtype)).sum() / denom.clamp(min=1)
    return ConfidenceBoundStats(loss=loss, active_count=active_mask.sum())
