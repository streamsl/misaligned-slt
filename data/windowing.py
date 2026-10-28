"""Window primitives: BIO labels from GT boundaries, plus the complete-span count of the sampler's Mode-3 draws.
UNK is the padding/ignore class — padding is never labelled O."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from utils import UNIT_SEPARATOR

BIO = {"UNK": 0, "O": 1, "B": 2, "I": 3}
# Names the labelling rule; `annotation_fingerprint` adds the content hash. Both travel with every checkpoint and
# artifact, and the name is what a POOLED S1 can be checked against when its fingerprint covers other languages.
ANNOTATION_PROTOCOL = "caption_units"
ModeName = Literal["mode1", "mode2", "mode3", "mode4"]
Mode2Subcase = Literal["right", "left", "both"]
InputName = Literal["chunks", "windows", "restart_chunks"]   # data.chunks INPUT_TYPES

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
    # A designed window, or (a stage-2 arm's train row, data/sampler.py spec_for) a restart chunk or a chunk at its anchor.
    input: InputName = "windows"

@dataclass
class WindowSample:
    spec: WindowSpec
    poses: np.ndarray
    timestamps_s: np.ndarray
    bio_labels: np.ndarray
    translation_target: SentenceSpan | None
    # The rate of the window's frame grid, on which Λ_min and δ become frame counts (train/slt.py): the record's fps, 
    # or the realized rate of the kept frames of an augmented train window (data/sampler.py materialize).
    fps: float
    # χ (membership gate, docs/membership_gate.md §2.4 (committed prefix) / §2.5 (log eps floor)): frames of LEFT-TRUNCATED
    # predecessors — a sentence whose B precedes the window edge is one the FSM already committed (the left edge mimics the
    # post-commit cut). Sampler bookkeeping, not a model belief; at inference the FSM supplies it from its commit log.
    commit_mask: np.ndarray | None = None
    # Every complete, reliable, >= Lambda_min sentence inside the window (time order): the multi-sentence target pool 
    # of a Mode-1/3 window; empty for the other modes (P1: no text for a sentence the window does not show whole).
    candidate_sentences: tuple[SentenceSpan, ...] = ()
    # The record's next unit starts more than δ after the text target's last unit (or there is none): the only rows on which the
    # text loss may train the segmentation branch, through where the span from that last unit's onset ends (models.streaming_slt).
    target_pause_after: bool = False
    # Window-relative times of the text target's B frame and terminator frame (data/sampler.py materialize); None without a target.
    target_span_s: tuple[float, float] | None = None
    # Window-relative time of the B frame of the target's LAST unit (its first unit's for a 1-unit target); None without a target.
    target_last_start_s: float | None = None


def joined_target(run) -> SentenceSpan:
    # 1 unit is its own target; a run is one span from its first onset to its last end, its texts joined by UNIT_SEPARATOR.
    if len(run) == 1: return run[0]
    return replace(run[0], end_s=run[-1].end_s, text=f" {UNIT_SEPARATOR} ".join(u.text for u in run))

def join_runs(units, sentences, gap_s: float, max_span_s: float | None = None) -> list[tuple[SentenceSpan, ...]]:
    """Group time-ordered complete units into runs that one text target may cover: each next unit starts at most `gap_s` (δ on
    the record's frame grid) after the previous one ends, no other unit of `sentences` (reliable or not) lies between them, and
    the whole run spans at most `max_span_s` (when given). Most caption boundaries inside a run show no pause in the poses, so
    only the text can place them; a run never crosses a longer gap, which a pause marks and which may hold uncaptioned signing."""
    runs: list[list[SentenceSpan]] = []
    for unit in units:
        if runs:
            prev, first = runs[-1][-1], runs[-1][0]
            between = any(u is not prev and u is not unit and u.start_s < unit.start_s and u.end_s > prev.end_s for u in sentences)
            fits = max_span_s is None or unit.end_s - first.start_s <= max_span_s
            if unit.start_s - prev.end_s <= gap_s and not between and fits: runs[-1].append(unit); continue
        runs.append([unit])
    return [tuple(r) for r in runs]

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
    frame_times_s: np.ndarray, spans: tuple[SentenceSpan, ...], window_start_s: float, window_end_s: float, 
    video_duration_s: float | None = None, untrusted: tuple[tuple[float, float], ...] | None = None,
) -> np.ndarray:
    """Label caption-unit membership on the supplied frame grid.

    Short uncaptioned gaps receive O as a heuristic; caption absence doesn't prove non-signing. Gaps longer than TRUSTED_GAP_S receive 
    UNK. Padding remains UNK. A unit can contain several linguistic sentences and internal pauses.
    """
    labels = np.full((len(frame_times_s),), BIO["O"], dtype=np.int64)

    # A clean segment with no unit (data.loader.split_clean_segments) is 1 uncaptioned stretch: UNK when longer than the gap.
    # `untrusted` (VideoRecord.untrusted_s) carries a cut segment's source-time gaps; an uncut record's gaps are found here.
    duration = float(video_duration_s) if video_duration_s is not None else max([float(window_end_s), *(span.end_s for span in spans)])
    for a, b in (untrusted if untrusted is not None else untrusted_o_intervals(spans, duration, TRUSTED_GAP_S)):
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
    # Same terminator semantics AND Λ_min floor as the target rule in WindowSampler.materialize: _mode3_spec accepts a 
    # draw by this count, and a floor on only 1 of the 2 accepts a Mode-3 window whose second unit materialize never selects.
    return sum(
        1 for span in spans if getattr(span, "reliable", True) and span.end_s - span.start_s >= min_span_s
        and span.start_s >= window_start_s and span.end_s + min_tail_s <= window_end_s
    )
