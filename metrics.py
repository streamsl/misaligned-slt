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

TIME-DOMAIN (Segment(start_s, end_s) seconds) — Segment/temporal_iou/match_segments/segmentation_prf; used by eval.py 
(RQ2 tIoU brackets), analyze.py (segmenter-error analysis pred-vs-GT matching).

TEXT: compute_text_metrics (BLEU-4/ROUGE-L/METEOR/BLEURT).
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from data.windowing import BIO
from sacrebleu import sentence_bleu
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
    # Predicted-B rate: B is <1% of frames, so an unweighted loss can drive the class to never fire while
    # precision/recall/accuracy all stay high (they score signing-vs-not, which B and I share). That collapse
    # shipped undetected once. Gold rate alongside it, so a near-zero value is readable without another run.
    pred_b = (valid & (pred == BIO["B"])).sum().float() / valid.sum().clamp(min=1)
    gold_b = (valid & (labels == BIO["B"])).sum().float() / valid.sum().clamp(min=1)
    return {
        f"{prefix}_precision": float(precision.detach().cpu().item()),
        f"{prefix}_recall": float(recall.detach().cpu().item()),
        f"{prefix}_f1": float(f1.detach().cpu().item()),
        f"{prefix}_frame_acc": float(acc.detach().cpu().item()),
        f"{prefix}_valid_frames": float(valid.sum().detach().cpu().item()),
        f"{prefix}_pred_b_rate": float(pred_b.detach().cpu().item()),
        f"{prefix}_gold_b_rate": float(gold_b.detach().cpu().item()),
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
    doesn't require one either. Interior `B`s split, feeding the segmenter-error over/under-segmentation taxonomy.
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
# One metric set for BOTH modes so RQ1 and RQ2 tables share columns. CIDEr is excluded: it needs a corpus
# document frequency, so it has no per-pair form for the RQ2 fusion, and reporting it in one table only
# would make the two incomparable.
_TEXT_KEYS = ("bleu4", "bleurt", "rougeL", "meteor")


def _char_split_cjk(text: str) -> str: # Space CJK chars for whitespace-tokenizing metrics.
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


def _bleurt_scores(hyps: list[str], refs: list[str], checkpoint: str | None) -> list[float]:
    # Per-example BLEURT on RAW text (BleurtScorer.score is inherently per-example). No checkpoint / failure -> zeros.
    if not checkpoint or not hyps: return [0.0] * len(hyps)
    try:
        from bleurt.score import BleurtScorer
        return [float(x) for x in BleurtScorer(checkpoint).score(candidates=list(hyps), references=list(refs))]
    except Exception: return [0.0] * len(hyps)


def _uni_sign_preprocess(predictions: list[str], references: list[str]) -> tuple[list[str], list[str], bool]:
    """Uni-Sign eval preprocessing (fine_tuning.py:284-288 + SLRT_metrics): word-level for non-CJK, char-split
    for CJK with the asymmetric ref-only punctuation map. Returned so corpus and per-pair scorers preprocess
    identically. `is_cjk` is detected over the references so scoring level never drifts within one call."""
    is_cjk = any(_char_split_cjk(ref) != ref for ref in references)

    def _proc(s: str, is_ref: bool) -> str:
        if not is_cjk: return s
        s = s.replace(" ", "").replace("\n", "")
        if is_ref: s = s.replace("，", ",").replace("？", "?")
        return " ".join(list(s))

    return [_proc(p, False) for p in predictions], [_proc(r, True) for r in references], is_cjk


def _corpus_metric(name: str, predictions, references, *, key: str, **kw) -> float:
    # Load an `evaluate` metric and score it, returning 0.0 if the metric is unavailable or the compute fails.
    metric = _load_evaluate_metric(name)
    if metric is None: return 0.0
    try: return float(metric.compute(predictions=predictions, references=references, **kw)[key])
    except Exception: return 0.0


def _sentence_text_scores(
    hyps: list[str], refs: list[str], sacrebleu_tokenize: str = "13a", bleurt_checkpoint: str | None = "/tmp/BLEURT-20",
) -> list[dict[str, float]]:
    """Per-pair sentence scores {bleu4(sentence), rougeL, meteor, bleurt} — the primitive the RQ2 fusion sums. BLEU
    here is smoothed sentence-BLEU: corpus BLEU pools across pairs and cannot be split per pair."""
    if not hyps: return []
    pred_proc, ref_proc, _ = _uni_sign_preprocess(hyps, refs)
    bleu = [float(sentence_bleu(h, [r], tokenize=sacrebleu_tokenize).score) for h, r in zip(pred_proc, ref_proc)]
    bleurt = _bleurt_scores(hyps, refs, bleurt_checkpoint)
    rouge = [_rouge_l([h], [r]) for h, r in zip(pred_proc, ref_proc)]
    meteor = _load_evaluate_metric("meteor")
    met = [
        float(meteor.compute(predictions=[h], references=[r])["meteor"]) for h, r in zip(pred_proc, ref_proc)
    ] if meteor is not None else [0.0] * len(hyps)
    return [dict(zip(_TEXT_KEYS, vals)) for vals in zip(bleu, bleurt, rouge, met)]


def compute_text_metrics(
    predictions: list[str], references: list[str], *, localization_aware: bool = False,
    n_pred: int | None = None, n_gold: int | None = None, memo: dict[tuple[str, str], dict[str, float]] | None = None,
    sacrebleu_tokenize: str = "13a", bleurt_checkpoint: str | None = "/tmp/BLEURT-20", prefix: str | None = None,
) -> dict[str, float]:
    """Translation-quality metrics, in 2 modes that are NOT comparable and are therefore named differently.

    2 modes answer different questions, so they get different key prefixes (`translation_*` vs `soda_*`). Sharing 1 name 
    invites exactly 1 mistake: reading a fused RQ2 cell as if it were translation quality & concluding the model collapsed. 
    The relationship is  soda_X ~= (mean per-pair X) * segmentation.f1, so a fused cell divided by the f1 reported beside 
    it recovers the per-pair quality — no extra metric needed.

    localization_aware=False (default — RQ1 and GT-span controls): CORPUS BLEU-4/ROUGE-L/METEOR/BLEURT over the whole set. 
    Paper-comparable (Uni-Sign reports corpus BLEU). ROUGE-L/METEOR/BLEURT are per-sentence means; BLEU-4 pools across the 
    set, so it is computed corpus-level.

    localization_aware=True (RQ2 dense/streaming), prefix `soda`: SODA F1 (Fujita et al. 2020) over MATCHED (pred, gold) 
    pairs. Per-pair sentence scores are summed, then precision = Σ/n_pred, recall = Σ/n_gold, F1 = 2PR/(P+R) — charging
    spurious predictions AND missed gold, so a spammy or under-generating method cannot inflate the score by scoring only 
    the subset it localizes. Sentence-BLEU (corpus BLEU does not split per pair). Needs n_pred/n_gold; `memo` caches 
    per-pair scores across tIoU thresholds (BLEURT is a model forward).
    """
    prefix = prefix if prefix is not None else ("soda" if localization_aware else "translation")
    if localization_aware:
        if n_pred is None or n_gold is None: 
            raise ValueError("localization_aware=True needs n_pred and n_gold (SODA count normalisation)")
        
        pairs = list(zip(predictions, references))
        if memo is None: per_pair = _sentence_text_scores(predictions, references, sacrebleu_tokenize, bleurt_checkpoint)
        else:
            uncached = [p for p in pairs if p not in memo]
            if uncached:
                scored = _sentence_text_scores(
                    [h for h, _ in uncached], [r for _, r in uncached], sacrebleu_tokenize, bleurt_checkpoint
                )
                for p, sc in zip(uncached, scored): memo[p] = sc
            per_pair = [memo[p] for p in pairs]

        out: dict[str, float] = {}
        for k in _TEXT_KEYS:
            s = float(sum(p[k] for p in per_pair))
            p_ = s / n_pred if n_pred else 0.0
            r_ = s / n_gold if n_gold else 0.0
            out[f"{prefix}_{k}"] = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0
        return out

    if not predictions: return {f"{prefix}_{k}": 0.0 for k in _TEXT_KEYS}
    # BLEURT scores RAW text; the rest use the Uni-Sign preprocessing (paper-comparable) — the ref-only punctuation
    # map avoids depressing ROUGE-L, whose LCS would otherwise see a different sequence.
    pred_proc, ref_proc, _ = _uni_sign_preprocess(predictions, references)
    ref_nested = [[r] for r in ref_proc]
    bleurt = _bleurt_scores(predictions, references, bleurt_checkpoint)
    out = {
        "bleu4": _corpus_metric("sacrebleu", pred_proc, ref_nested, key="score", tokenize=sacrebleu_tokenize),
        "bleurt": float(sum(bleurt) / len(bleurt)) if bleurt else 0.0,
        "rougeL": _rouge_l(pred_proc, ref_proc),
        "meteor": _corpus_metric("meteor", pred_proc, ref_proc, key="meteor"),
    }
    return {f"{prefix}_{k}": float(out[k]) for k in _TEXT_KEYS}
