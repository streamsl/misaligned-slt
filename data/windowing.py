"""Window primitives: BIO labels from GT boundaries, plus the first-complete-span rule shared by Mode-3 training and
streaming inference. UNK is the padding/ignore class — padding is never labelled O."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np

BIO = {"UNK": 0, "O": 1, "B": 2, "I": 3}
BIO_IGNORE_INDEX = BIO["UNK"]
# Names the labelling rule; `annotation_fingerprint` adds the content hash. Both travel with every checkpoint and
# artifact, and the name is what a POOLED S1 can be checked against when its fingerprint covers other languages.
ANNOTATION_PROTOCOL = "caption_units"
ModeName = Literal["mode1", "mode2", "mode3", "mode4"]
Mode2Subcase = Literal["right", "left", "both"]

# Uncaptioned stretches longer than this carry no trustworthy non-signing evidence (see make_bio_labels). 
TRUSTED_GAP_S = 8.0 # 1 constant for every labeller and for the class-weight counts.

@dataclass(frozen=True)
class SentenceSpan:
    video_id: str
    start_s: float
    end_s: float
    text: str
    # A timestamp-supported caption unit, possibly containing several linguistic sentences.
    # False excludes a unit from targets and gold, and labels every frame of it UNK: unsupported coverage, or a unit
    # that crosses a clean-segment edge (data.loader.split_clean_segments). It still occupies the timeline.
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
    translation_target: SentenceSpan | None
    anchor_span: SentenceSpan | None = None
    full_evidence_spec: WindowSpec | None = None
    # χ (membership gate, docs/membership_gate.md §2.4 (committed prefix) / §2.5 (log eps floor)): frames of LEFT-TRUNCATED
    # predecessors — a sentence whose B precedes the window edge is one the FSM already committed (the left edge mimics the
    # post-commit cut). Sampler bookkeeping, not a model belief; at inference the FSM supplies it from its commit log.
    commit_mask: np.ndarray | None = None
    # Every complete, reliable, >= Lambda_min sentence inside the window (time order): the multi-sentence target pool 
    # of a Mode-1/3 window; empty for the other modes (P1: no text for a sentence the window does not show whole).
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
    window_start_s: float, window_end_s: float, video_duration_s: float | None = None,
) -> np.ndarray:
    """Label caption-unit membership on the supplied frame grid.

    Short uncaptioned gaps receive O as a heuristic; caption absence doesn't prove non-signing. Gaps longer than TRUSTED_GAP_S receive 
    UNK. Padding remains UNK. A unit can contain several linguistic sentences and internal pauses.
    """
    labels = np.full((len(frame_times_s),), BIO["O"], dtype=np.int64)

    # A clean segment with no unit (data.loader.split_clean_segments) is 1 uncaptioned stretch: UNK when longer than the gap.
    duration = float(video_duration_s) if video_duration_s is not None else max([float(window_end_s), *(span.end_s for span in spans)])
    for a, b in untrusted_o_intervals(spans, duration, TRUSTED_GAP_S):
        labels[(frame_times_s >= a) & (frame_times_s < b)] = BIO["UNK"]

    spans_overlapping_window = [span for span in spans if span.end_s > window_start_s and span.start_s < window_end_s]
    for span in spans_overlapping_window:
        in_span = (frame_times_s >= span.start_s) & (frame_times_s < span.end_s)
        if not in_span.any(): continue
        # Quarantined unit (unsupported coverage past the pose stream, or a unit crossing a clean-segment edge): UNK on
        # every frame, its onset included (never I: that asserts "one phrase" over a span no label describes; never O:
        # that is "signing->O"; never an onset B: a 1-frame B run would be a gold span the label metrics score).
        if not getattr(span, "reliable", True):
            labels[in_span] = BIO["UNK"]
            continue

        first = int(np.argmax(in_span))
        if span.start_s >= window_start_s:  # frame_times_s[first] >= span.start_s by construction of in_span
            labels[first] = BIO["B"]
            labels[in_span & (np.arange(len(labels)) != first)] = BIO["I"]
        else: labels[in_span] = BIO["I"]
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

    `min_span_s` = Λ_min in seconds. The deployed gate's `candidate_spans` skips complete spans shorter than
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
    # Same terminator semantics AND Λ_min floor as first_complete_span — the sampler's mode-relabel counts with this and targets
    # with that; a floor on only 1 leaves a sub-Λ_min-only window a nominal mode1 with target=None (silently unsupervised).
    return sum(
        1 for span in spans if getattr(span, "reliable", True) and span.end_s - span.start_s >= min_span_s
        and span.start_s >= window_start_s and span.end_s + min_tail_s <= window_end_s
    )
