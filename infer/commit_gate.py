from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field

import torch
from data.windowing import BIO


def first_closing_o_index(bio_tags: torch.Tensor | list[int]) -> int | None: 
    # Return the first O following an active B/I span.
    if not isinstance(bio_tags, torch.Tensor): bio_tags = torch.as_tensor(bio_tags)
    active = False
    for idx, tag in enumerate(bio_tags.tolist()):
        if tag == BIO["B"]: active = True
        elif active and tag == BIO["O"]: return idx
    return None


def bio_complete_spans(bio_tags: torch.Tensor | list[int]) -> list[tuple[int, int]]:
    # Return complete predicted spans as [start_idx, closing_o_idx].
    if not isinstance(bio_tags, torch.Tensor): bio_tags = torch.as_tensor(bio_tags)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, tag in enumerate(bio_tags.tolist()):
        if tag == BIO["B"]: start = idx
        elif start is not None and tag == BIO["O"]:
            spans.append((start, idx))
            start = None
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

    1. **Boundary-stable** — the predicted closing-O index has moved ≤ `delta_enc_frames` over the last 
       `hysteresis_strides` strides (a hysteresis filter on `BoundaryHistory`, so a single stride's vote never commits).
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
