"""Sentence-count split: a committed span whose translation holds k >= 2 sentences becomes k events, cut in proportion to the
sentences' token counts. The text decides how many; length decides where (coarse); Lambda_min keeps every piece deployable."""
from __future__ import annotations
from typing import Callable
from data.batch import SENTENCE_SEP

_SEP_CHAR = SENTENCE_SEP.strip()

def split_sentences(text: str | None) -> list[str]:
    return [p.strip() for p in (text or "").split(_SEP_CHAR) if p.strip()]

def clean_text(text: str | None) -> str | None:
    # Scoring never sees the separator: a system that emits it without splitting is scored on the plain words.
    return None if text is None else " ".join(text.replace(_SEP_CHAR, " ").split())

def split_span_by_text(
    start_s: float, end_s: float, text: str | None, token_len: Callable[[str], int], min_span_s: float
) -> list[tuple[float, float, str]]:
    """[(start, end, sentence)] for one translated span. One piece when the text holds one sentence; k pieces cut in proportion to
    the sentences' token counts when it holds k >= 2; if any piece would fall under Lambda_min the span stays whole."""
    pieces = split_sentences(text)
    if len(pieces) < 2: return [(float(start_s), float(end_s), " ".join(pieces))]
    weights = [max(1, int(token_len(p))) for p in pieces]
    total, dur = float(sum(weights)), float(end_s) - float(start_s)
    cuts, acc = [float(start_s)], float(start_s)
    for w in weights: acc += dur * w / total; cuts.append(acc)
    cuts[-1] = float(end_s)
    out = [(cuts[i], cuts[i + 1], pieces[i]) for i in range(len(pieces))]
    if any(e - s < float(min_span_s) - 1e-9 for s, e, _ in out): return [(float(start_s), float(end_s), " ".join(pieces))]
    return out
