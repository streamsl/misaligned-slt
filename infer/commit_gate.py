from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field

import torch
from data.windowing import BIO


def _span_opens(tag: int, prev_tag: int | None, active: bool) -> bool:
    """A span OPENS on a predicted `B`, or O→I mid-buffer.

    Mirrors `metrics.signing_runs_with_b_splits`: `B` rarely wins argmax (one B per sentence; most adjacent captions 
    chain with no gap have no visual pause), so 1st signing frame after a gap counts as a start. Buffer-start signing 
    without `B` (prev_tag None) doesn't open: Mode-2b labels a left-truncated sentence `I`-without-`B` ("started before 
    the buffer, don't translate"), which also makes the overlap cut safe — its ≤2δ leftover arrives as buffer-start `I`.
    """
    if tag == BIO["B"]: return True
    return (not active) and tag == BIO["I"] and prev_tag == BIO["O"]


def bio_complete_spans(bio_tags: torch.Tensor | list[int]) -> list[tuple[int, int]]:
    # Complete predicted spans as [start_idx, terminator_idx]; open per `_span_opens`, terminate on O, next B, or UNK.
    # UNK closes like O — the same rule the deployed decode applies by remapping UNK->O before the FSM (infer/stream.py) 
    # and that `metrics.signing_runs_with_b_splits` uses. It matters only in LABEL domain, where UNK marks quarantined region: 
    # without it, reliable sentence abutting a quarantine has no terminator, so its label-domain span swallows the quarantine 
    # and the gate's GT anchor stops matching `first_complete_span`'s time-domain target — the two are required to agree.
    if not isinstance(bio_tags, torch.Tensor): bio_tags = torch.as_tensor(bio_tags)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    prev: int | None = None
    for idx, tag in enumerate(bio_tags.tolist()):
        if start is not None and tag in (BIO["O"], BIO["B"], BIO["UNK"]):
            spans.append((start, idx))
            start = None
        if _span_opens(tag, prev, start is not None): start = idx
        prev = tag
    return spans


def candidate_spans(bio_tags: torch.Tensor | list[int], min_span_frames: int = 0, skip_term_before: int = 0):
    """Complete spans ≥ `min_span_frames` (Λ_min) terminating after `skip_term_before`, in order. The first is the target of
    the FSM and of the membership gate's Ω anchor (`next(candidate_spans(...), None)`); every one gets a hysteresis vote each
    stride (`CommitGate.vote`).

    `skip_term_before` is the commit frontier χ in frames (= frames committed = index of 1st uncommitted frame). A span 
    terminating at or before χ is already emitted — an overlap-cut leftover or a stale re-detection — so skip it. The bound 
    is `term <= χ`, not `<`: the cut geometry (`last_commit_t = event.end_s - δ/fps`) makes equality the common case. This 
    commit-log check is the whole no-re-emission guarantee, so Λ_min and δ need no coupling ("Λ_min > 2δ" rule is infeasible 
    on short-sentence corpora: on asf 2δ > p10 sentence length). Spans STRADDLING χ stay selectable: δ-overlap stops a late 
    terminator estimate eating the successor's onset, and their ≤δ committed prefix is attention-floored by Ω's χ term.

    Λ_min is a LABEL-domain floor chosen to reject brief flicker. A genuine unit shorter than the floor cannot be
    emitted exactly; report that exclusion when comparing localization scores.
    """
    for start, term in bio_complete_spans(bio_tags):
        if term <= int(skip_term_before): continue  # term == χ: all content frames (< term) are committed
        if term - start >= int(min_span_frames): yield (start, term)


def open_span_start(bio_tags: torch.Tensor | list[int]) -> int | None:
    """Start of a TERMINATOR-LESS span running to the buffer edge (Mode-2a right-truncation, or a buffer-cap 
    forced commit); None if the buffer ends outside a span.

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


@dataclass(eq=False)  # identity, not value: two histories with equal votes are still two spans
class BoundaryHistory:
    """Terminator hysteresis for ONE candidate span: its last `hysteresis_strides` votes, (start, terminator) 
    in absolute stream frames, or None for a stride that did not see it."""
    hysteresis_strides: int = 3
    delta_enc_frames: int = 12
    values: deque[tuple[int, int] | None] = field(default_factory=deque)

    def push(self, span: tuple[int, int] | None) -> None:
        self.values.append(span)
        while len(self.values) > self.hysteresis_strides: self.values.popleft()

    def last_seen(self) -> tuple[int, int] | None:
        return next((v for v in reversed(self.values) if v is not None), None)

    def holds(self, span: tuple[int, int]) -> bool:
        # K votes, none missing, and the terminator (this observation included) moved ≤ δ over them.
        if len(self.values) < self.hysteresis_strides or any(v is None for v in self.values): return False
        terms = [int(v[1]) for v in self.values] + [int(span[1])]
        return max(terms) - min(terms) <= self.delta_enc_frames


class CommitGate:
    """Boundary-stability commit gate: a span commits when its terminator moved ≤ `delta_enc_frames` over the last
    `hysteresis_strides` strides.

    1 history per candidate span, keyed by absolute stream frames (buffer indices shift at every commit). Identity is 
    the start within δ of the history's last seen start; a start moving further is a different span with its own history, 
    so 1 stride's vote, or 1 inherited from another span, never commits. A commit retires only the histories it emitted:
    a later span keeps the votes it earned while it waited, so back-to-back units commit at their own stable time. Token 
    confidences are not a commit signal; events still carry them (infer/stability.py reveal policies).
    """
    def __init__(self, delta_enc_frames: int = 12, hysteresis_strides: int = 3):
        self.delta_enc_frames, self.hysteresis_strides = int(delta_enc_frames), int(hysteresis_strides)
        self.histories: list[BoundaryHistory] = []

    def _match(self, span: tuple[int, int], pool: list[BoundaryHistory]) -> BoundaryHistory | None:
        best, best_gap = None, None
        for history in pool:
            last = history.last_seen()
            gap = None if last is None else abs(int(span[0]) - int(last[0]))
            if gap is not None and gap <= self.delta_enc_frames and (best_gap is None or gap < best_gap): 
                best, best_gap = history, gap
        return best

    def vote(self, spans: list[tuple[int, int]]) -> None:
        # 1 call = 1 stride: each candidate span pushes 1 vote into its own history; a history whose span is absent gets None.
        pool = list(self.histories)
        for span in spans:
            history = self._match(span, pool)
            if history is None:
                history = BoundaryHistory(self.hysteresis_strides, self.delta_enc_frames)
                self.histories.append(history)
            else: pool.remove(history)
            history.push(span)
        for history in pool: history.push(None)
        self.histories = [h for h in self.histories if h.last_seen() is not None]  # unseen for K strides: gone

    def stable(self, span: tuple[int, int]) -> bool:
        # Reads votes only: a pass that adds none (cap, EOF drain, post-commit re-check) commits only stability already earned.
        history = self._match(span, self.histories)
        return history is not None and history.holds(span)

    def retire(self, span: tuple[int, int]) -> None:
        # Drop committed span's own history (matched as `stable` matches it: a vote-free pass can see its terminator up to 
        # δ earlier than the votes) and any history ending at or before its terminator; later spans keep their votes.
        own = self._match(span, self.histories)
        self.histories = [h for h in self.histories if h is not own and int(h.last_seen()[1]) > int(span[1])]

    def reset(self) -> None:
        self.histories.clear()
