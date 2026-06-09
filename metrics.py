from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable
from data.windowing import BIO
import torch


def bio_frame_metrics(logits: torch.Tensor, labels: torch.Tensor, prefix: str = "bio") -> dict[str, float]:
    # Frame-level BIO precision/recall/F1 over signing frames, ignoring UNK.
    pred = logits.argmax(dim=-1)
    valid = labels != BIO["UNK"]
    gold_pos = valid & ((labels == BIO["B"]) | (labels == BIO["I"]))
    pred_pos = valid & ((pred == BIO["B"]) | (pred == BIO["I"]))
    tp = (gold_pos & pred_pos).sum().float()
    precision = tp / pred_pos.sum().clamp(min=1)
    recall = tp / gold_pos.sum().clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    acc = ((pred == labels) & valid).sum().float() / valid.sum().clamp(min=1)
    return {
        f"{prefix}_precision": float(precision.detach().cpu().item()),
        f"{prefix}_recall": float(recall.detach().cpu().item()),
        f"{prefix}_f1": float(f1.detach().cpu().item()),
        f"{prefix}_frame_acc": float(acc.detach().cpu().item()),
        f"{prefix}_valid_frames": float(valid.sum().detach().cpu().item()),
    }


def bio_labels_to_segments(bio: torch.Tensor) -> list[dict]:
    # BIO label tensor -> [{start,end}] frame segments (Moryossef metrics.py).
    labels = bio.detach().cpu().numpy()
    segments, seg_start = [], None
    for j, label in enumerate(labels):
        if label == BIO["B"]:
            if seg_start is not None: segments.append({"start": seg_start, "end": j - 1})
            seg_start = j
        elif label == BIO["O"] and seg_start is not None:
            segments.append({"start": seg_start, "end": j - 1}); seg_start = None
    if seg_start is not None: segments.append({"start": seg_start, "end": len(labels) - 1})
    return segments


def likeliest_segments(logits: torch.Tensor) -> list[dict]:
    # Argmax decode -> contiguous B/I runs (Moryossef likeliest_probs_to_segments).
    preds = logits.detach().cpu().argmax(dim=-1).numpy()
    segments, seg_start = [], None
    for i, p in enumerate(preds):
        if p in (BIO["B"], BIO["I"]):
            if seg_start is None: seg_start = i
        elif seg_start is not None:
            segments.append({"start": seg_start, "end": i - 1}); seg_start = None
    if seg_start is not None: segments.append({"start": seg_start, "end": len(preds) - 1})
    return segments


def _segment_iou_frames(pred: list[dict], gold: list[dict], max_len: int) -> float:
    import numpy as _np
    pv, gv = _np.zeros(max_len), _np.zeros(max_len)
    for s in pred: pv[s["start"]:s["end"] + 1] = 1
    for s in gold: gv[s["start"]:s["end"] + 1] = 1
    inter = _np.logical_and(pv, gv).sum(); union = _np.logical_or(pv, gv).sum()
    if union == 0: return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def _segment_recall(segments: list[dict], gold: list[dict], allowed_shift: int = 17) -> float:
    # Moryossef _segment_recall: a gold segment is hit if any pred overlaps it within +/-allowed_shift frames (~0.68s @25fps).
    if not gold: return 1.0 if not segments else 0.0
    hit = 0
    for sg in gold:
        start, end = sg["start"] - allowed_shift, sg["end"] + allowed_shift
        if any(s["start"] <= end and s["end"] >= start for s in segments): hit += 1
    return hit / len(gold)


def _segment_f1(pred: list[dict], gold: list[dict]) -> float:
    if not gold or not pred: return 1.0 if len(pred) == len(gold) else 0.0
    precision = _segment_recall(gold, pred); recall = _segment_recall(pred, gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _macro_frame_f1(pred: torch.Tensor, gold: torch.Tensor, classes=(BIO["O"], BIO["B"], BIO["I"])) -> float:
    f1s = []
    for c in classes:
        tp = ((pred == c) & (gold == c)).sum().float()
        fp = ((pred == c) & (gold != c)).sum().float()
        fn = ((pred != c) & (gold == c)).sum().float()
        denom = (2 * tp + fp + fn)
        if denom > 0: f1s.append(float((2 * tp / denom).item()))
    return sum(f1s) / len(f1s) if f1s else 0.0


def moryossef_segment_metrics(logits: torch.Tensor, labels: torch.Tensor, prefix: str = "phrase") -> dict[str, float]:
    """Per-item Moryossef segmentation metrics (frame macro-F1, frame-IoU, segment-F1).

    Faithful to segmentation/.../metrics.py + evaluate.py: for each sequence, mask out UNK, argmax-decode predicted segments,  
    build gold segments, and average frame-F1 / segment-IoU / segment-F1 over items. Phrase level only (no sign head).
    """
    frame_f1s, ious, seg_f1s = [], [], []
    for i in range(labels.shape[0]):
        gold = labels[i]
        valid = gold != BIO["UNK"]
        n = int(valid.sum())
        if n == 0: continue

        gold_v = gold[:n]
        logit_v = logits[i, :n]
        frame_f1s.append(_macro_frame_f1(logit_v.argmax(dim=-1), gold_v))

        pred_segs = likeliest_segments(logit_v)
        gold_segs = bio_labels_to_segments(gold_v)
        ious.append(_segment_iou_frames(pred_segs, gold_segs, n))
        seg_f1s.append(_segment_f1(pred_segs, gold_segs))

    avg = lambda xs: float(sum(xs) / len(xs)) if xs else 0.0
    return {f"{prefix}_frame_f1": avg(frame_f1s), f"{prefix}_seg_iou": avg(ious), f"{prefix}_seg_f1": avg(seg_f1s)}

    
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
    predictions: list[str], references: list[str], sacrebleu_tokenize: str = "13a", 
    bleurt_checkpoint: str | None = None, prefix: str = "translation"
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
    return {f"{prefix}_{key}": float(value) for key, value in scores.items()}


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor, prefix: str = "translation") -> dict[str, float]:
    # Teacher-forced token accuracy, ignoring -100 labels.
    valid = labels != -100
    if logits.shape[1] != labels.shape[1]:
        labels = labels[:, : logits.shape[1]]
        valid = valid[:, : logits.shape[1]]
    pred = logits.argmax(dim=-1)
    correct = ((pred == labels) & valid).sum().float()
    total = valid.sum().clamp(min=1)
    return {
        f"{prefix}_token_acc": float((correct / total).detach().cpu().item()),
        f"{prefix}_tokens": float(valid.sum().detach().cpu().item()),
    }