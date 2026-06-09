from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable
from data.windowing import BIO
import torch


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_s - self.start_s))


def temporal_iou(a: Segment, b: Segment) -> float:
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    union = a.duration_s + b.duration_s - inter
    return inter / union if union > 0 else 0.0


def match_segments(predicted: list[Segment], gold: list[Segment], threshold: float = 0.1) -> list[tuple[int, int, float]]:
    scored = [] # Greedy one-to-one tIoU matching.
    for pi, pred in enumerate(predicted):
        for gi, gt in enumerate(gold):
            score = temporal_iou(pred, gt)
            if score >= threshold: scored.append((score, pi, gi))

    scored.sort(reverse=True)
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, pi, gi in scored:
        if pi in used_pred or gi in used_gold: continue
        used_pred.add(pi)
        used_gold.add(gi)
        matches.append((pi, gi, score))
    return matches


def segmentation_prf(predicted: list[Segment], gold: list[Segment], tiou_threshold: float = 0.1) -> dict[str, float]:
    matches = match_segments(predicted, gold, threshold=tiou_threshold)
    tp = len(matches)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "matches": float(tp)}


def bio_tags_to_segments(tags: torch.Tensor | list[int], timestamps_s: torch.Tensor | list[float]) -> list[Segment]:
    if not isinstance(tags, torch.Tensor): tags = torch.as_tensor(tags)
    if not isinstance(timestamps_s, torch.Tensor): timestamps_s = torch.as_tensor(timestamps_s, dtype=torch.float32)
    tags_list = tags.tolist()
    times = timestamps_s.tolist()
    segments: list[Segment] = []
    start: float | None = None
    for idx, tag in enumerate(tags_list):
        if tag == BIO["B"]:
            if start is not None: segments.append(Segment(start, times[idx]))
            start = times[idx]
        elif tag == BIO["O"] and start is not None:
            segments.append(Segment(start, times[idx]))
            start = None
    if start is not None and times:
        step = (times[-1] - times[-2]) if len(times) > 1 else 0.04 # Assume 25 FPS if only 1 timestamp is present, to give segments with some duration.
        segments.append(Segment(start, times[-1] + step))
    return segments


@lru_cache(maxsize=8)
def _load_evaluate_metric(name: str):
    try:
        import evaluate
        return evaluate.load(name)
    except Exception: return None


def _char_split_cjk(text: str) -> str: # Space CJK characters for whitespace-tokenizing metrics such as CIDEr.
    out: list[str] = []
    for ch in text:
        is_cjk = "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf"
        if is_cjk: # '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿'
            if out and out[-1] != " ": out.append(" ")
            out.append(ch)
            out.append(" ")
        else: out.append(ch)
    return "".join(out).strip()


def compute_text_metrics(
    predictions: list[str], references: list[str],
    sacrebleu_tokenize: str = "13a", bleurt_checkpoint: str | None = None,
) -> dict[str, float]: # Compute translation metrics with optional backends loaded lazily.
    if not predictions: return {"bleu4": 0.0, "bleurt": 0.0, "rougeL": 0.0, "cider": 0.0, "meteor": 0.0, "chrf": 0.0}
    refs_nested = [[ref] for ref in references]
    cjk_predictions = [_char_split_cjk(pred) for pred in predictions]
    cjk_references = [_char_split_cjk(ref) for ref in references]
    cjk_refs_nested = [[ref] for ref in cjk_references]
    scores = {"bleu4": 0.0, "bleurt": 0.0, "rougeL": 0.0, "cider": 0.0, "meteor": 0.0, "chrf": 0.0}

    bleu = _load_evaluate_metric("sacrebleu")
    if bleu is not None:
        try: 
            scores["bleu4"] = float(bleu.compute(predictions=predictions, references=refs_nested, tokenize=sacrebleu_tokenize)["score"])
        except Exception: pass

    rouge = _load_evaluate_metric("rouge")
    if rouge is not None:
        try:
            rouge_kwargs = {"tokenizer": lambda text: text.split()} if cjk_predictions != predictions else {}
            scores["rougeL"] = float(rouge.compute(predictions=cjk_predictions, references=cjk_references, **rouge_kwargs)["rougeL"])
        except Exception: pass

    cider = _load_evaluate_metric("sunhill/cider")
    if cider is not None:
        try: scores["cider"] = float(cider.compute(predictions=cjk_predictions, references=cjk_refs_nested)["cider_score"])
        except Exception: pass

    meteor = _load_evaluate_metric("meteor")
    if meteor is not None:
        try: scores["meteor"] = float(meteor.compute(predictions=cjk_predictions, references=cjk_references)["meteor"])
        except Exception: pass

    chrf = _load_evaluate_metric("chrf")
    if chrf is not None:
        try: scores["chrf"] = float(chrf.compute(predictions=predictions, references=refs_nested)["score"])
        except Exception: pass

    if bleurt_checkpoint:
        try:
            from bleurt.score import BleurtScorer
            bleurt_scores = BleurtScorer(bleurt_checkpoint).score(candidates=predictions, references=references)
            scores["bleurt"] = float(sum(bleurt_scores) / max(1, len(bleurt_scores)))
        except Exception: pass
    return scores
