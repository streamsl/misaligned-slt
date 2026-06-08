"""Window primitives: BIO label construction from GT boundaries and the
first-complete-span rule shared by Mode-3 training and streaming inference. UNK is
the padding/ignore class — padding is never labelled O (Hard Rule §1.4.1)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np

BIO = {"UNK": 0, "O": 1, "B": 2, "I": 3}
BIO_IGNORE_INDEX = BIO["UNK"]
ModeName = Literal["mode1", "mode2", "mode3", "mode4"]
Mode2Subcase = Literal["right", "left", "both"]


@dataclass(frozen=True)
class SentenceSpan:
    video_id: str
    start_s: float
    end_s: float
    text: str

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


def make_bio_labels(
    frame_times_s: np.ndarray, spans: tuple[SentenceSpan, ...],
    window_start_s: float, window_end_s: float,
    frame_mask: np.ndarray | None = None,
) -> np.ndarray: # Build BIO labels from GT boundaries; padding is caller-masked as UNK.
    labels = np.full((len(frame_times_s),), BIO["O"], dtype=np.int64)
    spans_overlapping_window = [span for span in spans if span.end_s > window_start_s and span.start_s < window_end_s]

    for span in spans_overlapping_window:
        in_span = (frame_times_s >= span.start_s) & (frame_times_s < span.end_s)
        if not in_span.any(): continue

        first = int(np.argmax(in_span))
        if span.start_s >= window_start_s and frame_times_s[first] >= span.start_s:
            labels[first] = BIO["B"]
            labels[in_span & (np.arange(len(labels)) != first)] = BIO["I"]
        else: labels[in_span] = BIO["I"]

    if frame_mask is not None:
        labels = labels.copy()
        labels[~frame_mask.astype(bool)] = BIO["UNK"]
    return labels


def first_complete_span(
    spans: tuple[SentenceSpan, ...],
    window_start_s: float,
    window_end_s: float,
    min_o_after_s: float = 1e-6,
) -> SentenceSpan | None: # Earliest span whose B and closing O are inside the same window.
    for span in sorted(spans, key=lambda s: (s.start_s, s.end_s)):
        has_b = span.start_s >= window_start_s
        has_closing_o = span.end_s + min_o_after_s <= window_end_s
        if has_b and has_closing_o: return span
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
    window_start_s: float,
    window_end_s: float,
    min_o_after_s: float = 1e-6,
) -> int:
    return sum(
        1 for span in spans 
        if span.start_s >= window_start_s and span.end_s + min_o_after_s <= window_end_s
    )
