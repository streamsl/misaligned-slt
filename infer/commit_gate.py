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
    without `B` precisely so the model can signal "this sentence started before the buffer; do not translate it". 
    Opening there would translate fragments whose head was discarded. This is also what makes the post-commit overlap 
    cut safe: ≤2δ leftover tail of previous sentence shows up as buffer-start `I` and is never selected as a target.
    """
    if tag == BIO["B"]: return True
    return (not active) and tag == BIO["I"] and prev_tag == BIO["O"]


def first_terminator_index(bio_tags: torch.Tensor | list[int]) -> int | None:
    """Index of the first frame that TERMINATES an active span: an `O`, or a new `B`.

    A sign sentence can start immediately after another (back-to-back, no `O` gap), so the terminator of a span is 
    the first following `O` **or** `B` — the BIO-standard rule and the whole reason a `B` class exists over binary 
    I/O tagging (Moryossef 2023/2026). Terminating only on `O` would never close the first sentence of a back-to-back 
    pair and FSM would bridge them until buffer cap. The training-side target rule (windowing.first_complete_span) 
    encodes the same semantics on GT timestamps (Same rule at train & inference). Span opening follows `_span_opens`.
    """
    if not isinstance(bio_tags, torch.Tensor): bio_tags = torch.as_tensor(bio_tags)
    active, prev = False, None
    for idx, tag in enumerate(bio_tags.tolist()):
        if active and tag in (BIO["O"], BIO["B"]): return idx
        if _span_opens(tag, prev, active): active = True
        prev = tag
    return None


def bio_complete_spans(bio_tags: torch.Tensor | list[int]) -> list[tuple[int, int]]:
    # Complete predicted spans as [start_idx, terminator_idx]; opening per `_span_opens`
    # (B anywhere, or mid-buffer O→I), terminating on O or on the next span's B.
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


def select_target_span(bio_tags: torch.Tensor | list[int], min_span_frames: int = 0) -> tuple[int, int] | None:
    """First complete span at least `min_span_frames` long (Λ_min) — the FSM's translation target.

    Λ_min filters phantom micro-spans: a spurious `B` inside the post-commit leftover (or a 1-frame flicker) forms
    a terminated span that is otherwise commit-eligible — a static spurious span trivially passes the boundary-
    stability hysteresis. Derive Λ_min from the dev sentence-length distribution (a low percentile, in encoder
    frames) and keep Λ_min > 2δ so the ≤2δ overlap-cut leftover can never qualify even if mislabelled `B`.
    """
    for start, term in bio_complete_spans(bio_tags):
        if term - start >= int(min_span_frames): return (start, term)
    return None


def first_complete_bio_span(bio_tags: torch.Tensor | list[int]) -> tuple[int, int] | None:
    # First complete span with NO minimum length (cf. select_target_span, which applies Λ_min). Test helper.
    spans = bio_complete_spans(bio_tags)
    return spans[0] if spans else None


def open_span_start(bio_tags: torch.Tensor | list[int]) -> int | None:
    """Start index of a TERMINATOR-LESS span that runs to the buffer edge (Mode-2a right-truncation, or a
    buffer-cap forced commit), or None if the buffer ends outside a span.

    Same open/terminate rule as `bio_complete_spans` (`_span_opens`: B anywhere or mid-buffer O→I; terminate on
    O or the next B), but returns the start of the FINAL still-open span instead of the completed ones. This is
    the anchor the membership gate needs for the forced/open path (docs/membership_gate.md §2.8): anchoring Ω at
    this s gives γ≡γ_s with no right cliff (Ω≈0 for the all-I interior), whereas anchoring at frame 0 would sweep
    the opening B and floor the very span the gate is meant to open. A buffer-start I-run (left-truncated leftover)
    never opens, so it correctly yields None (that span is not translated)."""
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

    Entries are (start_idx, terminator_idx). Buffer frame indices are stable between commits (the buffer only
    grows at the right edge), so `start_idx` identifies the span across strides; when the selected span's start
    moves by more than `delta_enc_frames` the history is for a DIFFERENT span and is cleared — mixing terminator
    estimates of different spans would make the stability test meaningless.
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
    """Two-signal commit gate; both signals must hold to emit.

    1. **Boundary-stable** — the SELECTED target span's terminator (first `O`, or the `B` opening the next sentence) has moved 
       ≤ `delta_enc_frames` over last `hysteresis_strides` strides, for the SAME span (identity by start index; `BoundaryHistory` 
       clears on a target change, so a single stride's vote — or a vote inherited from a different span — never commits).
    2. **Translation-hardened** — the decode's MEAN per-token confidence ≥ `token_confidence_tau` at the *current* stride
       (forward-looking, not an edit-distance to previous strides — that would reward the warm-started state the design forbids).
       Mean, not `all(≥τ)`: a single low-confidence subword (a function word / morpheme legitimately sits at 0.2–0.5) must not
       veto an otherwise-confident sentence. An `all()` floor is only satisfiable when τ sits below the clean per-token MINIMUM,
       which for a label-smoothed decoder is near-zero — so `all(≥0.75)` is effectively "never commit" on the AR arm (whose
       honest teacher-forced confidence means ~0.4), silently zeroing §9.3 AR streaming recall. The DLM arm's DCD-committed
       tokens sit high, so it was unaffected — the mismatch made the two arms incomparable, which is exactly what §9.3 must not do.

    τ must be CALIBRATED to the deployed model's clean-input confidence (like δ_enc / buffer_cap are measured), NOT hand-set;
    see configs/inference.yaml. The gate is a floor against low-confidence junk — it does NOT catch "confidently wrong"
    truncated decodes (those keep high confidence); the membership gate Ω is the mechanism for that.
    """
    def __init__(self, delta_enc_frames: int = 3, hysteresis_strides: int = 3, token_confidence_tau: float = 0.75):
        self.history = BoundaryHistory(hysteresis_strides=int(hysteresis_strides), delta_enc_frames=int(delta_enc_frames))
        self.token_confidence_tau = float(token_confidence_tau)

    def update(self, span: tuple[int, int] | None, token_confidence: torch.Tensor | None = None) -> CommitDecision:
        # `span` is the target the caller selected (select_target_span) and decoded — the gate scores exactly the
        # span being emitted, never a recomputed one that could disagree with the decode.
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
