from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field

import torch
from data.windowing import BIO


def _span_opens(tag: int, prev_tag: int | None, active: bool) -> bool:
    """A span OPENS on a predicted `B`, or O→I mid-buffer.

    Mirrors `metrics.signing_runs_with_b_splits`: `B` rarely wins argmax (one B per sentence; 68% of caption boundaries
    have no visual pause), so the first signing frame after a gap counts as a start. Buffer-start signing without `B`
    (prev_tag None) does NOT open: Mode-2b labels a left-truncated sentence `I`-without-`B` ("started before the buffer,
    don't translate"), which also makes the overlap cut safe — its ≤2δ leftover arrives as buffer-start `I`.
    """
    if tag == BIO["B"]: return True
    return (not active) and tag == BIO["I"] and prev_tag == BIO["O"]


def first_terminator_index(bio_tags: torch.Tensor | list[int]) -> int | None:
    """First frame TERMINATING an active span: an `O`, or a new `B`.

    Back-to-back sentences have no `O` gap, so `B` must terminate too (BIO-standard; why a `B` class 
    exists over binary I/O — Moryossef 2023/2026). `O`-only would bridge such a pair until buffer cap.
    `windowing.first_complete_span` encodes the same rule on GT timestamps; keep in sync.
    """
    if not isinstance(bio_tags, torch.Tensor): bio_tags = torch.as_tensor(bio_tags)
    active, prev = False, None
    for idx, tag in enumerate(bio_tags.tolist()):
        if active and tag in (BIO["O"], BIO["B"]): return idx
        if _span_opens(tag, prev, active): active = True
        prev = tag
    return None


def bio_complete_spans(bio_tags: torch.Tensor | list[int]) -> list[tuple[int, int]]:
    # Complete predicted spans as [start_idx, terminator_idx]; open per `_span_opens`, terminate on O or next B.
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


def select_target_span(
    bio_tags: torch.Tensor | list[int], min_span_frames: int = 0, skip_term_before: int = 0,
) -> tuple[int, int] | None:
    """First complete span ≥ `min_span_frames` (Λ_min) terminating at/after `skip_term_before` — the FSM's target.

    `skip_term_before` is the commit frontier χ in frames (= frames committed = index of the first uncommitted frame).
    A span terminating at or before χ is already emitted — an overlap-cut leftover or a stale re-detection — so skip
    it. The bound is `term <= χ`, not `<`: the cut geometry (`last_commit_t = event.end_s - δ/fps`) makes equality the
    common case. This commit-log check is the whole no-re-emission guarantee, replacing the "Λ_min > 2δ" coupling that
    was infeasible on short-sentence corpora (asf: 2δ > p10 sentence length). Spans STRADDLING χ stay selectable:
    δ-overlap stops a late terminator estimate eating the successor's onset, and their ≤δ committed prefix is
    attention-floored by Ω's χ term.

    Λ_min is a duration noise floor: spans shorter than δ are unresolvable from boundary evidence (a 1-frame flicker
    passes stability hysteresis). See StreamingSLTRunner for how it is derived.
    """
    for start, term in bio_complete_spans(bio_tags):
        if term <= int(skip_term_before): continue  # term == χ: all content frames (< term) are committed
        if term - start >= int(min_span_frames): return (start, term)
    return None


def open_span_start(bio_tags: torch.Tensor | list[int]) -> int | None:
    """Start of a TERMINATOR-LESS span running to the buffer edge (Mode-2a right-truncation, or a buffer-cap forced
    commit); None if the buffer ends outside a span.

    Same rule as `bio_complete_spans`, but returns the FINAL still-open span — the anchor the membership gate needs on
    the forced/open path (docs/membership_gate.md §2.8): Ω anchored here gives γ≡γ_s with no right cliff (Ω≈0 for the
    all-I interior), while frame 0 would sweep the opening B and floor the span the gate should open. A buffer-start
    I-run (left-truncated leftover) never opens → None."""
    if not isinstance(bio_tags, torch.Tensor): bio_tags = torch.as_tensor(bio_tags)
    start: int | None = None
    prev: int | None = None
    for idx, tag in enumerate(bio_tags.tolist()):
        if start is not None and tag in (BIO["O"], BIO["B"]): start = None
        if _span_opens(tag, prev, start is not None): start = idx
        prev = tag
    return start


@dataclass
class BoundaryHistory:
    """Terminator hysteresis for ONE candidate span.

    Entries are (start_idx, terminator_idx). Buffer indices are stable between commits (it only grows at the right
    edge), so `start_idx` identifies the span across strides; a start moving by more than `delta_enc_frames` is a
    different span and clears the history — mixing spans makes the stability test meaningless.
    """
    hysteresis_strides: int = 3
    delta_enc_frames: int = 3
    values: deque[tuple[int, int] | None] = field(default_factory=deque)

    def push(self, span: tuple[int, int] | None) -> None:
        if span is not None:
            last = next((v for v in reversed(self.values) if v is not None), None)
            if last is not None and abs(int(span[0]) - int(last[0])) > self.delta_enc_frames:
                self.values.clear()  # target identity changed
        self.values.append(span)
        while len(self.values) > self.hysteresis_strides:
            self.values.popleft()

    def stable(self) -> bool:
        if len(self.values) < self.hysteresis_strides: return False
        if any(v is None for v in self.values): return False
        terms = [int(v[1]) for v in self.values if v is not None]
        return max(terms) - min(terms) <= self.delta_enc_frames

    def latest(self) -> tuple[int, int] | None:
        return self.values[-1] if self.values else None


@dataclass
class CommitDecision:
    boundary_stable: bool
    translation_confident: bool
    terminator_index: int | None

    @property
    def should_commit(self) -> bool:
        return self.boundary_stable and self.translation_confident and self.terminator_index is not None


class CommitGate:
    """Two-signal commit gate; both must hold to emit.

    1. **Boundary-stable** — terminator moved ≤ `delta_enc_frames` over the last `hysteresis_strides` strides for the
       SAME span (identity by start index; `BoundaryHistory` clears on a target change, so one stride's vote — or one
       inherited from another span — never commits).
    2. **Translation-hardened** — MEAN per-token confidence of the *current* stride's decode ≥ `token_confidence_tau`.
       Not edit-distance to previous strides: that rewards the warm-started state the design forbids. Mean, not
       `all(≥τ)`: function words sit at 0.2–0.5, so `all(≥0.75)` never commits on the AR arm (mean ~0.4), zeroing its
       §9.3 streaming recall while the DLM arm's DCD-committed tokens pass — the arms become incomparable.

    τ must be CALIBRATED to the model's clean-input confidence (like δ_enc / buffer_cap), see configs/inference.yaml.
    It floors low-confidence junk only; "confidently wrong" truncated decodes stay high-confidence and are Ω's job.
    """
    def __init__(self, delta_enc_frames: int = 3, hysteresis_strides: int = 3, token_confidence_tau: float = 0.75):
        self.history = BoundaryHistory(hysteresis_strides=int(hysteresis_strides), delta_enc_frames=int(delta_enc_frames))
        self.token_confidence_tau = float(token_confidence_tau)

    def update(self, span: tuple[int, int] | None, token_confidence: torch.Tensor | None = None) -> CommitDecision:
        # `span` must be the one the caller decoded, never recomputed here — the gate scores exactly what is emitted.
        self.history.push(span)
        if token_confidence is None or token_confidence.numel() == 0: trans_ok = False
        else: trans_ok = bool((token_confidence.float().mean() >= self.token_confidence_tau).item())
        latest = self.history.latest()
        return CommitDecision(
            boundary_stable=self.history.stable(), translation_confident=trans_ok,
            terminator_index=None if latest is None else int(latest[1]),
        )

    def reset(self) -> None:
        self.history.values.clear()
