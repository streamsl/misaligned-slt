from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field

import torch
from data.windowing import BIO


def _span_opens(tag: int, prev_tag: int | None, active: bool) -> bool:
    """A span OPENS on a predicted `B`, or on an O→I transition mid-buffer.

    Mirrors the prediction decode used everywhere else (`metrics.signing_runs_with_b_splits`): the model rarely 
    wins argmax with `B` (one B frame per sentence; 68% of caption boundaries have no visual pause), and the first 
    signing frame after a gap is a sentence start by construction. One deliberate exception: signing at BUFFER START 
    without a `B` (prev_tag is None) does NOT open a span — Mode-2b training labels a left-truncated sentence as `I` 
    without `B` precisely so the model can signal "this sentence started before the buffer; do not translate it" 
    (spec §5.2b). Opening there would translate fragments whose head was discarded.
    """
    if tag == BIO["B"]: return True
    return (not active) and tag == BIO["I"] and prev_tag == BIO["O"]


def first_closing_o_index(bio_tags: torch.Tensor | list[int]) -> int | None:
    """Index of the first frame that closes an active span: a closing O, or a new B.

    A new B closes the previous span (back-to-back sentences have no O gap; closing only on O would silently drop 
    the 1st sentence, while the training-side target rule — windowing.first_complete_span — needs only the sentence 
    end inside the window; Hard Rule §1.4.8 requires the same rule here). Span opening follows `_span_opens`.
    """
    if not isinstance(bio_tags, torch.Tensor): bio_tags = torch.as_tensor(bio_tags)
    active, prev = False, None
    for idx, tag in enumerate(bio_tags.tolist()):
        if active and tag in (BIO["O"], BIO["B"]): return idx
        if _span_opens(tag, prev, active): active = True
        prev = tag
    return None


def bio_complete_spans(bio_tags: torch.Tensor | list[int]) -> list[tuple[int, int]]:
    # Complete predicted spans as [start_idx, closing_idx]; opening per `_span_opens`
    # (B anywhere, or mid-buffer O→I), closing on O or on the next span's B.
    if not isinstance(bio_tags, torch.Tensor): bio_tags = torch.as_tensor(bio_tags)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    prev: int | None = None
    for idx, tag in enumerate(bio_tags.tolist()):
        if start is not None and tag in (BIO["O"], BIO["B"]):
            spans.append((start, idx))
            start = None
        if _span_opens(tag, prev, start is not None): start = idx
        prev = tag
    return spans


def first_complete_bio_span(bio_tags: torch.Tensor | list[int]) -> tuple[int, int] | None:
    spans = bio_complete_spans(bio_tags)
    return spans[0] if spans else None


@dataclass
class BoundaryHistory:
    hysteresis_strides: int = 2
    delta_enc_frames: int = 3
    values: deque[int | None] = field(default_factory=deque)

    def push(self, closing_index: int | None) -> None:
        self.values.append(closing_index)
        while len(self.values) > self.hysteresis_strides:
            self.values.popleft()

    def stable(self) -> bool:
        if len(self.values) < self.hysteresis_strides: return False
        if any(v is None for v in self.values): return False
        vals = [int(v) for v in self.values if v is not None]
        return max(vals) - min(vals) <= self.delta_enc_frames

    def latest(self) -> int | None:
        return self.values[-1] if self.values else None


@dataclass
class CommitDecision:
    boundary_stable: bool
    translation_confident: bool
    closing_index: int | None

    @property
    def should_commit(self) -> bool:
        return self.boundary_stable and self.translation_confident and self.closing_index is not None


class CommitGate:
    """Two-signal commit gate (spec §7.3); both signals must hold to emit.

    1. **Boundary-stable** — the predicted closing index (closing O, or the B opening the next sentence) has moved ≤ `delta_enc_frames` 
       over the last `hysteresis_strides` strides (a hysteresis filter on `BoundaryHistory`, so a single stride's vote never commits).
    2. **Translation-hardened** — every output token's DCD confidence ≥ `token_confidence_tau` at the *current* stride 
       (forward-looking, not an edit-distance to previous strides — that would reward the warm-started state the design forbids).

    Frozen constants, not dev-tuned (`configs/inference.yaml`). `reset()` after a commit clears the hysteresis history.
    """
    def __init__(self, delta_enc_frames: int = 3, hysteresis_strides: int = 2, token_confidence_tau: float = 0.75):
        self.history = BoundaryHistory(hysteresis_strides=int(hysteresis_strides), delta_enc_frames=int(delta_enc_frames))
        self.token_confidence_tau = float(token_confidence_tau)

    def update(self, phrase_bio_tags: torch.Tensor | list[int], token_confidence: torch.Tensor | None = None) -> CommitDecision:
        closing = first_closing_o_index(phrase_bio_tags)
        self.history.push(closing)
        if token_confidence is None or token_confidence.numel() == 0: trans_ok = False
        else: trans_ok = bool((token_confidence >= self.token_confidence_tau).all().item())
        return CommitDecision(
            boundary_stable=self.history.stable(), 
            translation_confident=trans_ok, 
            closing_index=self.history.latest()
        )

    def reset(self) -> None:
        self.history.values.clear()
