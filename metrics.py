"""Metrics — frame-domain (BIO-head training monitor) and time-domain (final eval).

`segmentation_prf` (greedy one-to-one tIoU-matched P/R/F1) is THE segment metric for both; tIoU is scale-invariant,
so frame units and seconds score identically. Numbers differ only because inputs do: the stage-2 monitor
(train/slt.py) = raw (ungated) BIO argmax on sampler dev WINDOWS, macro per window (`val_phrase_tiou_f1`); the S1
monitor (train/bio_pretrain.py) scores the DEPLOYED duration-decoded tags whenever inference.yaml `duration_decode`
is on for the language, so the two `val_phrase_tiou_f1` series are NOT comparable; RQ2 `--stream` = commit-GATED FSM
events on whole VIDEOS vs GT sentences, corpus-micro.

FRAME-DOMAIN (BIO logits/labels) — bio_frame_metrics, moryossef_segment_metrics; used by train/slt.py, 
train/bio_pretrain.py, moryossef26/, eval.py (FSM-internal BIO diagnostic), analyze.py (tune-decode). ONE decode 
rule for BOTH prediction and gold (default `signing_runs_with_b_splits`; why there); decoders parameterize `_bio_runs`.

TIME-DOMAIN (Segment(start_s, end_s) seconds) — Segment/temporal_iou/match_segments/segmentation_prf; used by
eval.py (RQ2 tIoU brackets), analyze.py (Analysis A pred-vs-GT matching).

TEXT: compute_text_metrics (BLEU-4/ROUGE-L/METEOR/CIDEr/BLEURT).
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
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


def bio_frame_metrics(logits: torch.Tensor, labels: torch.Tensor, prefix: str = "bio") -> dict[str, float]:
    # Frame-level BIO P/R/F1 over signing frames, ignoring UNK.
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
    """Unified BIO -> [{start,end}] frame-segment decoder. Public decoders parameterize it
    (split_on_b, open_on_i, close_on_unk):

      bio_labels_to_segments      T F F   B-required; full-annotation gold
      signing_runs_with_b_splits  T T T   inference rule; monitor BOTH sides
      decode="likeliest"          F T T   pure run, parity ref

    split_on_b: interior B closes and reopens (back-to-back sentences, no O gap); else B opens only if nothing open.
    open_on_i: I with nothing open opens (sentence start after a gap; headless left-truncated fragment).
    close_on_unk: UNK closes like O; B-required gold keeps it non-closing (full annotation has no interior UNK).
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
    # GOLD decode (Moryossef metrics.py): B-required.
    return _bio_runs(bio, split_on_b=True, open_on_i=False, close_on_unk=False)

def signing_runs_with_b_splits(tags: torch.Tensor | list[int]) -> list[dict]:
    """PREDICTION/inference decode: signing runs split at interior `B` (== moryossef26.infer.bio_tags_to_segments).

    Requiring a predicted `B` to OPEN is fatal (`B` is ~1% of frames, and most adjacent captions chain with no gap): 
    a signing-detecting model that never argmaxes `B` yields zero segments — Moryossef's `likeliest_probs_to_segments` 
    doesn't require one either. Interior `B`s split, feeding the Analysis-A over/under-segmentation taxonomy.
    """
    return _bio_runs(tags, split_on_b=True, open_on_i=True, close_on_unk=True)

def _frame_segments_to_seconds(segs: list[dict]) -> list["Segment"]:
    # Frame indices -> Segments; end is exclusive (a 1-frame segment spans [start, start+1)).
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
    # One-to-one tIoU-matched precision/recall/F1/matches — THE canonical segment metric (RQ2 + BIO-head monitor).
    # Unit-agnostic (tIoU is scale-invariant): seconds or frame-unit Segments. Nothing predicted AND nothing gold = perfect.
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
    """Per-item BIO-head training monitor: one per-FRAME score, one segment score.

    `{prefix}_frame_f1`: macro F1 over O/B/I frame classes.
    `{prefix}_tiou_f1`/`_seg_precision`/`_seg_recall`: `segmentation_prf` (the RQ2 metric) on frame-unit segments.
    Collapse-proof for early stopping — an all-`I`/all-`O` collapse can't game one-to-one matching (the looser
    overlap `seg_f1` / frame-IoU `seg_iou` flavors could; removed).
    `decode` applies to BOTH sides (`runs_bsplit` = inference, default; `bio` = B-required; `likeliest` = raw run).
    """
    # B-required gold fits Moryossef's full annotations (every onset visible) but breaks on OUR misaligned windows:
    # make_bio_labels tags a left-truncated span as a HEADLESS I-run, so gold emits NO segment where a PERFECT
    # tagger's run decode emits one — an unavoidable FP capping precision far below 1. Symmetric decode fixes it.
    if decode == "likeliest": decode_fn = lambda t: _bio_runs(t, split_on_b=False, open_on_i=True, close_on_unk=True)
    elif decode == "bio": decode_fn = bio_labels_to_segments
    else: decode_fn = signing_runs_with_b_splits

    frame_f1s, tiou_f1s, precisions, recalls = [], [], [], []
    n_matches = n_pred = n_gold = 0  # raw counts for a caller's micro pooling; the scores below stay per-item macro
    for i in range(labels.shape[0]):
        gold = labels[i]
        valid = gold != BIO["UNK"]
        n_valid = int(valid.sum())
        if n_valid == 0: continue

        # Trim TRAILING padding (collators pad with UNK on the right); keep interior UNK (untrusted gaps).
        last = int(torch.nonzero(valid).max().item()) + 1
        gold_v = gold[:last]
        pred_tags = logits[i, :last].argmax(dim=-1)
        interior_unk = gold_v == BIO["UNK"]
        if bool(interior_unk.any()):
            # No reliable label in untrusted gaps: mask BOTH sides so `close_on_unk` splits runs identically.
            pred_tags = torch.where(interior_unk, torch.full_like(pred_tags, BIO["UNK"]), pred_tags)
        frame_f1s.append(_macro_frame_f1(pred_tags[~interior_unk], gold_v[~interior_unk]))

        pred_segs = _frame_segments_to_seconds(decode_fn(pred_tags))
        gold_segs = _frame_segments_to_seconds(decode_fn(gold_v))
        prf = segmentation_prf(pred_segs, gold_segs, tiou_threshold=tiou_threshold)
        tiou_f1s.append(prf["f1"]); precisions.append(prf["precision"]); recalls.append(prf["recall"])
        n_matches += int(prf["matches"]); n_pred += len(pred_segs); n_gold += len(gold_segs)

    avg = lambda xs: float(sum(xs) / len(xs)) if xs else 0.0
    return {
        f"{prefix}_frame_f1": avg(frame_f1s), f"{prefix}_tiou_f1": avg(tiou_f1s),
        f"{prefix}_seg_precision": avg(precisions), f"{prefix}_seg_recall": avg(recalls),
        f"{prefix}_n_matches": n_matches, f"{prefix}_n_pred": n_pred, f"{prefix}_n_gold": n_gold,
    }


@lru_cache(maxsize=8)
def _load_evaluate_metric(name: str):
    try:
        import evaluate
        return evaluate.load(name)
    except Exception as e:
        print(f"[metrics] WARNING: metric backend {name!r} unavailable ({type(e).__name__}: {e}); "
              f"its column will read 0.0 — do NOT report that cell.", flush=True)
        return None


# Full-width CJK punctuation -> ASCII. mT5 emits ASCII '?'/',' where refs carry '？'/'，', so un-normalized 
# char-BLEU penalizes a 1-char mismatch on nearly every sentence. Uni-Sign (fine_tuning.py:285) normalizes 
# '，'/'？' on refs only; we cover the common marks on BOTH sides.
_CJK_PUNCT_TABLE = str.maketrans({
    '￥': '$', '％': '%', '＃': '#', '＠': '@', '，': ',', '。': '.', '？': '?', '！': '!', '、': ',', '；': ';', '：': ':',
    '（': '(', '）': ')', '【': '[', '】': ']', '《': '<', '》': '>', '「': '"', '」': '"', '『': '"', '』': '"', 
    '“': '"', '”': '"', '‘': "'", '’': "'", '—': '-', '–': '-', '·': '.', '…': '...', '　': ' ', '﹏': '_', '～': '~', 
})
def _char_split_cjk(text: str) -> str: # Space CJK chars for whitespace-tokenizing metrics such as CIDEr.
    out: list[str] = []
    for ch in text:
        is_cjk = "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf"
        if is_cjk:
            if out and out[-1] != " ": out.append(" ")
            out.append(ch)
            out.append(" ")
        else: out.append(ch)
    return "".join(out).strip()


def _rouge_l(hyps: list[str], refs: list[str]) -> float:
    """ROUGE-L f-score (mean over sentences) via pltrdy `rouge` — Uni-Sign's package (`['rouge-l']['f']`,
    whitespace-tokenized over the char-split strings), matching SLRT_metrics.translation_performance. HF
    `evaluate`'s "rouge" is Google's rouge_score, whose ROUGE-L F differs (~1 pt on CJK) and is NOT comparable to
    their 0.55 — fallback only. Empty hyp/ref scores 0 (pltrdy raises)."""
    try:
        from rouge import Rouge as _PltRouge
        scorer = _PltRouge()
        fs = []
        for h, r in zip(hyps, refs):
            if not h.strip() or not r.strip(): fs.append(0.0)
            else: fs.append(float(scorer.get_scores(h, r)[0]["rouge-l"]["f"]))
        return float(sum(fs) / len(fs)) if fs else 0.0
    except Exception:
        rouge = _load_evaluate_metric("rouge")
        if rouge is None: return 0.0
        try: return float(rouge.compute(predictions=hyps, references=refs, tokenizer=lambda t: t.split())["rougeL"])
        except Exception: return 0.0


def compute_text_metrics(
    predictions: list[str], references: list[str], sacrebleu_tokenize: str = "13a",
    bleurt_checkpoint: str | None = "/tmp/BLEURT-20", prefix: str = "translation"
) -> dict[str, float]:
    scores = {"bleu4": 0.0, "bleurt": 0.0, "rougeL": 0.0, "cider": 0.0, "meteor": 0.0}
    if not predictions: return {f"{prefix}_{key}": value for key, value in scores.items()}
    # Preprocess EXACTLY like Uni-Sign's eval (fine_tuning.py:284-288 + SLRT_metrics.translation_performance) for
    # paper-comparable numbers. A full punctuation map on BOTH sides DEPRESSES ROUGE-L by ~0.03 (LCS sees a
    # different sequence). BLEURT scores RAW text.
    is_cjk = any(_char_split_cjk(ref) != ref for ref in references)

    def _proc(s: str, is_ref: bool) -> str:
        if not is_cjk: return s                                  # word-level for non-CJK (Uni-Sign level='word')
        s = s.replace(" ", "").replace("\n", "")
        if is_ref: s = s.replace("，", ",").replace("？", "?")    # Uni-Sign's asymmetric ref-only normalization
        return " ".join(list(s))                                 # char-split (Uni-Sign level='char')

    pred_proc = [_proc(p, False) for p in predictions]
    ref_proc = [_proc(r, True) for r in references]
    ref_proc_nested = [[r] for r in ref_proc]

    bleu = _load_evaluate_metric("sacrebleu")
    if bleu is not None:
        # '13a' over the char-split strings = Uni-Sign's `sableu`: char-level BLEU for CJK, word BLEU otherwise.
        try: scores["bleu4"] = float(bleu.compute(
                predictions=pred_proc, references=ref_proc_nested, tokenize=sacrebleu_tokenize
            )["score"])
        except Exception: pass
    scores["rougeL"] = _rouge_l(pred_proc, ref_proc)

    cider = _load_evaluate_metric("sunhill/cider")
    if cider is not None:
        try: scores["cider"] = float(cider.compute(predictions=pred_proc, references=ref_proc_nested)["cider_score"])
        except Exception: pass

    meteor = _load_evaluate_metric("meteor")
    if meteor is not None:
        try: scores["meteor"] = float(meteor.compute(predictions=pred_proc, references=ref_proc)["meteor"])
        except Exception: pass

    if bleurt_checkpoint:
        try:
            from bleurt.score import BleurtScorer
            bleurt_scores = BleurtScorer(bleurt_checkpoint).score(candidates=predictions, references=references)
            scores["bleurt"] = float(sum(bleurt_scores) / max(1, len(bleurt_scores)))
        except Exception: pass
    return {f"{prefix}_{key}": float(value) for key, value in scores.items()}
