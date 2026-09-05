"""Window primitives: BIO labels from GT boundaries, plus the first-complete-span rule shared by Mode-3 training and
streaming inference. UNK is the padding/ignore class — padding is never labelled O."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np

BIO = {"UNK": 0, "O": 1, "B": 2, "I": 3}
BIO_IGNORE_INDEX = BIO["UNK"]
ModeName = Literal["mode1", "mode2", "mode3", "mode4"]
Mode2Subcase = Literal["right", "left", "both"]

# Uncaptioned stretches longer than this carry no trustworthy non-signing evidence (see make_bio_labels). 

TRUSTED_GAP_S = 8.0 # Module-level default. Only moryossef26/trainer.py reads an override (data.yaml
# subtitles.trusted_gap_s, absent by default); every other call site takes this constant.

@dataclass(frozen=True)
class SentenceSpan:
    video_id: str
    start_s: float
    end_s: float
    text: str
    # False = a QUARANTINED region: real sentences whose internal boundaries can't be located in time (punctuated chains longer 
    # than the buffer, see loader.reconstruct_sentences). Frames get UNK (no BIO supervision), the span is never a translation 
    # anchor, and eval treats it as an ignore region. Labeled wrongly is worse than not labeled: same rule as trusted-gap UNK.
    reliable: bool = True

    @property
    def duration_s(self) -> float:
        return float(self.end_s - self.start_s)

@dataclass(frozen=True)
class WindowSpec:
    video_id: str
    start_s: float
    end_s: float
    mode: ModeName
    anchor_index: int | None = None
    subcase: Mode2Subcase | None = None

@dataclass
class WindowSample:
    spec: WindowSpec
    poses: np.ndarray
    timestamps_s: np.ndarray
    bio_labels: np.ndarray
    frame_mask: np.ndarray
    spans: tuple[SentenceSpan, ...]
    translation_target: SentenceSpan | None
    anchor_span: SentenceSpan | None = None
    full_evidence_spec: WindowSpec | None = None
    # χ (membership gate, docs/membership_gate.md §2.7): frames of LEFT-TRUNCATED predecessors — a sentence whose B
    # precedes the window edge is one the FSM already committed (the left edge mimics the post-commit cut). Sampler
    # bookkeeping, not a model belief; at inference the FSM supplies it from its commit log.
    commit_mask: np.ndarray | None = None
    # Every complete, reliable, >= Lambda_min sentence inside the window (time order): the multi-sentence target pool of a
    # Mode-1/3 window; empty for the other modes (P1: no text for a sentence the window does not show whole).
    candidate_sentences: tuple[SentenceSpan, ...] = ()


def untrusted_o_intervals(spans: tuple[SentenceSpan, ...], duration_s: float, trusted_gap_s: float) -> list[tuple[float, float]]:
    # Uncaptioned stretches (incl. video head/tail) LONGER than trusted_gap_s.
    bounds: list[tuple[float, float]] = []
    prev = 0.0
    for span in sorted(spans, key=lambda s: s.start_s):
        if span.start_s > prev: bounds.append((prev, span.start_s))
        prev = max(prev, span.end_s)
    if duration_s > prev: bounds.append((prev, duration_s))
    return [(a, b) for a, b in bounds if (b - a) > float(trusted_gap_s)]


def make_bio_labels(
    frame_times_s: np.ndarray, spans: tuple[SentenceSpan, ...],
    window_start_s: float, window_end_s: float, frame_mask: np.ndarray | None = None,
    trusted_gap_s: float | None = TRUSTED_GAP_S, video_duration_s: float | None = None,
) -> np.ndarray:
    """Build BIO labels from GT boundaries; padding is caller-masked as UNK.

    `O` is supervised only where caption absence evidences a real pause: uncaptioned stretches up to `trusted_gap_s`. Longer ones 
    (intros/outros/credits, 42% of uncaptioned time on Auslan) may contain uncaptioned signing — `O` there teaches "signing → O" 
    on the frames most like signing, so they get UNK (no loss). `trusted_gap_s=None` labels everything O.
    """
    labels = np.full((len(frame_times_s),), BIO["O"], dtype=np.int64)

    if trusted_gap_s is not None and len(spans):
        duration = float(video_duration_s) if video_duration_s is not None \
                                           else max(float(window_end_s), max(span.end_s for span in spans))
        for a, b in untrusted_o_intervals(spans, duration, trusted_gap_s):
            labels[(frame_times_s >= a) & (frame_times_s < b)] = BIO["UNK"]

    spans_overlapping_window = [span for span in spans if span.end_s > window_start_s and span.start_s < window_end_s]
    for span in spans_overlapping_window:
        in_span = (frame_times_s >= span.start_s) & (frame_times_s < span.end_s)
        if not in_span.any(): continue
        # Quarantined region: interior sentence boundaries cannot be time-located, so they are UNK (never I: I would
        # assert "one phrase" across a multi-sentence span, teaching under-segmentation; never O: that is "signing->O").
        # The ONSET is exempt — it is a real cue timestamp — so it keeps its B below.
        if not getattr(span, "reliable", True):
            # Its INTERIOR boundaries are unlocatable, but its ONSET is not: a quarantined chain begins at a cue
            # timestamp where Punkt placed a sentence boundary — the same evidence that validates a reliable merged
            # chain's outer bounds. Dropping it would discard ~12% of all `B` supervision (the rarest, most heavily
            # weighted class) and bias the `balanced` class-weight fit, which is derived from this labeller.
            labels[in_span] = BIO["UNK"]
            first_q = int(np.argmax(in_span))
            if span.start_s >= window_start_s: labels[first_q] = BIO["B"]   # same guard as the reliable branch
            continue

        first = int(np.argmax(in_span))
        if span.start_s >= window_start_s:  # frame_times_s[first] >= span.start_s by construction of in_span
            labels[first] = BIO["B"]
            labels[in_span & (np.arange(len(labels)) != first)] = BIO["I"]
        else: labels[in_span] = BIO["I"]

    if frame_mask is not None:
        labels = labels.copy()
        labels[~frame_mask.astype(bool)] = BIO["UNK"]
    return labels


def first_complete_span(
    spans: tuple[SentenceSpan, ...],
    window_start_s: float, window_end_s: float,
    min_tail_s: float = 1e-6, min_span_s: float = 0.0,
) -> SentenceSpan | None:
    """Earliest span whose B and TERMINATOR are inside the same window (first-complete-span).

    Complete = end lies ≥ `min_tail_s` inside the window, i.e. the window holds the frame AFTER the span's last — the frame 
    carrying terminator label (`O` on a gap, or next sentence's `B` when back-to-back; adjacent sentences have no closing `O`, 
    and requiring one would misread a completed anchor as right-truncated). Must stay in sync with its label-space twin 
    `infer.commit_gate.bio_complete_spans` (terminate on O-or-B) so training (GT) and inference (predicted) select alike.

    `min_span_s` = Λ_min in seconds. The deployed gate's `select_target_span` skips complete spans shorter than
    `span_selection.min_span_frames`, so the TRAINING rule must too: without the same floor a window supervises a
    sentence the gate never anchors, silently conditioning the decoder on the wrong sentence's Ω mask.
    """
    for span in sorted(spans, key=lambda s: (s.start_s, s.end_s)):
        if not getattr(span, "reliable", True): continue     # quarantined: never a translation target
        if span.end_s - span.start_s < min_span_s: continue  # Λ_min: never a deployable commit target
        has_b = span.start_s >= window_start_s
        has_terminator = span.end_s + min_tail_s <= window_end_s
        if has_b and has_terminator: return span
    return None

def classify_anchor_visibility(span: SentenceSpan, start_s: float, end_s: float) -> str:
    has_start = span.start_s >= start_s and span.start_s < end_s
    has_end = span.end_s > start_s and span.end_s < end_s
    if has_start and has_end: return "complete"
    if has_start and not has_end: return "right"
    if not has_start and has_end: return "left"
    if span.start_s < start_s and span.end_s > end_s: return "both"
    return "outside"

def count_complete_spans(
    spans: tuple[SentenceSpan, ...],
    window_start_s: float, window_end_s: float,
    min_tail_s: float = 1e-6, min_span_s: float = 0.0,
) -> int:
    # Same terminator semantics AND Λ_min floor as first_complete_span — the sampler's mode-relabel counts with
    # this and targets with that; a floor on only one leaves a sub-Λ_min-only window a nominal mode1 with
    # target=None (silently unsupervised).
    return sum(
        1 for span in spans if getattr(span, "reliable", True) and span.end_s - span.start_s >= min_span_s
        and span.start_s >= window_start_s and span.end_s + min_tail_s <= window_end_s
    )
