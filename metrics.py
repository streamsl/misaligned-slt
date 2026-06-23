"""Metrics — TWO families, by domain and consumer. They are not interchangeable:

There is ONE segment metric DEFINITION — `segmentation_prf` (greedy one-to-one tIoU-matched P/R/F1) — used by
BOTH the training monitor and the final eval. tIoU is scale-invariant, so feeding the SAME segments in frame
units (monitor) or seconds (eval) scores identically. This does NOT make the monitor and the RQ2 eval equal:
they score DIFFERENT inputs — the monitor decodes the BIO head's raw (ungated) argmax over sampler dev WINDOWS
and averages per-window F1 (macro); RQ2 `--stream` scores commit-GATED FSM events over whole VIDEOS against GT
sentences as corpus precision/recall (micro). So `val_phrase_tiou_f1` (BIO-head quality) and the streaming seg
F1 (end-to-end, after the commit gate) legitimately differ. The only other segmentation signal is a per-FRAME
score (different granularity, not redundant).

FRAME-DOMAIN entry points (consume BIO logits/labels; the BIO-head TRAINING monitor):
  bio_frame_metrics          per-frame B/I-vs-O precision/recall/F1/accuracy.
  moryossef_segment_metrics  per-item `*_frame_f1` (macro O/B/I frame F1) + `*_tiou_f1`/`*_seg_precision`/
                             `*_seg_recall` from `segmentation_prf` on frame-unit segments (the collapse-proof
                             monitor; an all-`I`/all-`O` collapse cannot game one-to-one matching). The looser
                             overlap seg-F1 and frame-IoU flavors were removed (used for no decision).
  Used by: train/stage2.py (in-model BIO head), moryossef26/{trainer,infer}.py (the standalone segmenter).
  Segment decoders feeding it: `bio_labels_to_segments` (gold), `signing_runs_with_b_splits` (prediction),
  `likeliest_segments` (parity) — all three are one parameterized core, `_bio_runs`.

TIME-DOMAIN entry points (consume Segment(start_s, end_s) spans in seconds; the FINAL DVC evaluation + Analysis A):
  Segment / temporal_iou / match_segments  greedy one-to-one tIoU matching.
  segmentation_prf                         the SAME P/R/F1 the monitor uses, on seconds spans.
  Used by: eval.py (RQ2 tIoU brackets), analyze.py (Analysis A pred-vs-GT matching).

TEXT: compute_text_metrics (BLEU-4/ROUGE-L/METEOR/CIDEr/BLEURT) + token_accuracy.
"""
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


def _bio_runs(tags, *, split_on_b: bool, open_on_i: bool, close_on_unk: bool) -> list[dict]:
    """Unified BIO -> [{start,end}] frame-segment decoder. The three public decoders are parameterizations:

      bio_labels_to_segments     split_on_b=True,  open_on_i=False, close_on_unk=False  (gold, B-required)
      signing_runs_with_b_splits split_on_b=True,  open_on_i=True,  close_on_unk=True   (prediction/inference)
      likeliest_segments         split_on_b=False, open_on_i=True,  close_on_unk=True   (pure run, parity ref)

    split_on_b: an interior B closes the open segment and opens a new one (back-to-back sentences with no O gap);
    when False, B only opens if nothing is open (B behaves like I). open_on_i: an I with nothing open starts a
    segment (first signing frame after a gap = a sentence start). close_on_unk: UNK closes like O; gold decoding
    keeps UNK non-closing (gold is pre-sliced to valid frames).
    """
    if isinstance(tags, torch.Tensor): tags = tags.detach().cpu().tolist()
    segments: list[dict] = []
    start: int | None = None
    for i, tag in enumerate(tags):
        if tag == BIO["B"]:
            if split_on_b:
                if start is not None: segments.append({"start": start, "end": i - 1})
                start = i
            elif start is None: start = i
        elif tag == BIO["I"]:
            if open_on_i and start is None: start = i
        elif (tag == BIO["O"] or (tag == BIO["UNK"] and close_on_unk)) and start is not None:
            segments.append({"start": start, "end": i - 1}); start = None
    if start is not None: segments.append({"start": start, "end": len(tags) - 1})
    return segments


def bio_labels_to_segments(bio: torch.Tensor) -> list[dict]:
    # GOLD decode (Moryossef metrics.py): B-required. Turns GT BIO labels into reference segments.
    return _bio_runs(bio, split_on_b=True, open_on_i=False, close_on_unk=False)

def likeliest_segments(logits: torch.Tensor) -> list[dict]:
    # Parity reference: Moryossef's raw argmax run decode (interior B does not split). Takes LOGITS.
    return _bio_runs(logits.detach().cpu().argmax(dim=-1), split_on_b=False, open_on_i=True, close_on_unk=True)

def signing_runs_with_b_splits(tags: torch.Tensor | list[int]) -> list[dict]:
    """PREDICTION/inference decode: contiguous signing runs, split at interior `B` (== infer.bio_tags_to_segments).

    Moryossef's prediction decode (`likeliest_probs_to_segments`) never requires a predicted `B` — a segment is
    any contiguous B/I run; requiring `B` to OPEN is fatal here (`B` is ~1% of frames, 68% of caption boundaries
    have no visual pause), so a model that detects signing but never wins argmax with `B` yields zero segments.
    Opening on the O→signing transition loses nothing (first signing frame after a gap IS a sentence start);
    interior `B`s are honoured as splits, feeding the Analysis-A over/under-segmentation taxonomy.
    """
    return _bio_runs(tags, split_on_b=True, open_on_i=True, close_on_unk=True)

def _frame_segments_to_seconds(segs: list[dict]) -> list["Segment"]:
    # Frame-index segments -> Segment objects (end is exclusive: a 1-frame segment spans [start, start+1)).
    # tIoU is scale-invariant, so matching in frame units gives the SAME number as matching in seconds.
    return [Segment(float(s["start"]), float(s["end"]) + 1.0) for s in segs]


def _macro_frame_f1(pred: torch.Tensor, gold: torch.Tensor, classes=(BIO["O"], BIO["B"], BIO["I"])) -> float:
    f1s = []
    for c in classes:
        tp = ((pred == c) & (gold == c)).sum().float()
        fp = ((pred == c) & (gold != c)).sum().float()
        fn = ((pred != c) & (gold == c)).sum().float()
        denom = (2 * tp + fp + fn)
        if denom > 0: f1s.append(float((2 * tp / denom).item()))
    return sum(f1s) / len(f1s) if f1s else 0.0


def segmentation_prf(predicted: list[Segment], gold: list[Segment], tiou_threshold: float = 0.1) -> dict[str, float]:
    """One-to-one tIoU-matched precision/recall/F1/matches — THE canonical segment metric (RQ2 + the BIO-head
    monitor via `moryossef_segment_metrics`). Unit-agnostic: pass seconds Segments (eval) or frame-unit Segments
    (monitor); tIoU is scale-invariant. Nothing predicted AND nothing gold = perfect (nothing to find, none emitted)."""
    if not predicted and not gold: return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matches": 0.0}
    matches = match_segments(predicted, gold, threshold=tiou_threshold)
    tp = len(matches)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "matches": float(tp)}


def moryossef_segment_metrics(
    logits: torch.Tensor, labels: torch.Tensor, prefix: str = "phrase",
    decode: str = "runs_bsplit", tiou_threshold: float = 0.5,
) -> dict[str, float]:
    """Per-item BIO-head training monitor: a per-FRAME score and the ONE segment score used everywhere.

    Returns two complementary granularities, nothing redundant:
    - `{prefix}_frame_f1`: macro F1 over the O/B/I frame classes (frame classification quality).
    - `{prefix}_tiou_f1` / `_seg_precision` / `_seg_recall`: one-to-one tIoU-matched segment P/R/F1 via the
      SAME `segmentation_prf` that scores RQ2 (here on frame-unit segments; tIoU is scale-invariant, so this
      is identical to the seconds-domain number). This is the collapse-proof early-stopping monitor — neither
      the all-`I` nor all-`O` collapse can game one-to-one matching. The previous overlap-tolerance `seg_f1`
      and frame-IoU `seg_iou` flavors were looser, collapse-foolable, and used for no decision; removed.

    `decode` picks the predicted-segment rule (`runs_bsplit` = inference decode, default; `bio` = B-required;
    `likeliest` = raw run). `tiou_threshold` (default 0.5) is the monitor's match threshold.
    """
    frame_f1s, tiou_f1s, precisions, recalls = [], [], [], []
    for i in range(labels.shape[0]):
        gold = labels[i]
        valid = gold != BIO["UNK"]
        n = int(valid.sum())
        if n == 0: continue

        gold_v = gold[:n]
        logit_v = logits[i, :n]
        pred_tags = logit_v.argmax(dim=-1)
        frame_f1s.append(_macro_frame_f1(pred_tags, gold_v))

        if decode == "likeliest": pred_segs = likeliest_segments(logit_v)
        elif decode == "bio": pred_segs = bio_labels_to_segments(pred_tags)
        else: pred_segs = signing_runs_with_b_splits(pred_tags)
        prf = segmentation_prf(
            _frame_segments_to_seconds(pred_segs), _frame_segments_to_seconds(bio_labels_to_segments(gold_v)),
            tiou_threshold=tiou_threshold,
        )
        tiou_f1s.append(prf["f1"]); precisions.append(prf["precision"]); recalls.append(prf["recall"])

    avg = lambda xs: float(sum(xs) / len(xs)) if xs else 0.0
    return {
        f"{prefix}_frame_f1": avg(frame_f1s), f"{prefix}_tiou_f1": avg(tiou_f1s),
        f"{prefix}_seg_precision": avg(precisions), f"{prefix}_seg_recall": avg(recalls),
    }

    
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


# Full-width CJK punctuation -> ASCII. mT5 decodes Chinese text but emits ASCII '?'/',' for some marks while the
# references carry full-width '？'/'，' (and '。' etc.), so an un-normalized char-BLEU penalizes a 1-char punctuation
# mismatch on nearly every sentence. Uni-Sign's eval (fine_tuning.py:285) does exactly this for '，'/'？' on the refs;
# we normalize BOTH sides over the common marks so the score reflects content, not punctuation encoding.
_CJK_PUNCT_MAP = {
    '￥': '$', '％': '%', '＃': '#', '＠': '@', '，': ',', '。': '.', '？': '?', '！': '!', '、': ',', '；': ';', '：': ':',
    '（': '(', '）': ')', '【': '[', '】': ']', '《': '<', '》': '>', '「': '"', '」': '"', '『': '"', '』': '"', 
    '“': '"', '”': '"', '‘': "'", '’': "'", '—': '-', '–': '-', '·': '.', '…': '...', '　': ' ', '﹏': '_', '～': '~', 
}
_CJK_PUNCT_TABLE = {ord(k): v for k, v in _CJK_PUNCT_MAP.items()}


def compute_text_metrics(
    predictions: list[str], references: list[str], sacrebleu_tokenize: str = "13a",
    bleurt_checkpoint: str | None = "/tmp/BLEURT-20", prefix: str = "translation"
) -> dict[str, float]: # Compute translation metrics with optional backends loaded lazily.
    scores = {"bleu4": 0.0, "bleurt": 0.0, "rougeL": 0.0, "cider": 0.0, "meteor": 0.0}
    if not predictions: return {f"{prefix}_{key}": value for key, value in scores.items()}
    # CJK detection from the (gold) references; if present, normalize full-width punctuation on BOTH sides so
    # the ASCII-punct model output matches the full-width refs (else char-BLEU loses ~1 n-gram/sentence).
    is_cjk = any(_char_split_cjk(ref) != ref for ref in references)
    if is_cjk:
        predictions = [pred.translate(_CJK_PUNCT_TABLE) for pred in predictions]
        references = [ref.translate(_CJK_PUNCT_TABLE) for ref in references]
        
    refs_nested = [[ref] for ref in references]
    cjk_predictions = [_char_split_cjk(pred) for pred in predictions]
    cjk_references = [_char_split_cjk(ref) for ref in references]
    cjk_refs_nested = [[ref] for ref in cjk_references]

    bleu = _load_evaluate_metric("sacrebleu")
    if bleu is not None:
        # Chinese/CJK has no word spaces, so sacrebleu's default '13a' (European) tokenizer collapses BLEU to
        # ~0 even on correct translations (n-grams never match). Switch to sacrebleu's CJK char tokenizer 'zh'
        # whenever the references are CJK — this is the CSL-Daily / Uni-Sign convention, so BLEU is comparable
        # to their reported 25.x. cjk_references != references is the same CJK signal used for ROUGE/CIDEr below.
        try: scores["bleu4"] = float(bleu.compute(
                predictions=predictions, references=refs_nested, 
                tokenize="zh" if is_cjk else sacrebleu_tokenize
            )["score"])
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
