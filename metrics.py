"""Metrics — frame-domain (BIO-head training monitor) and time-domain (final eval).

`segmentation_prf` provides one-to-one tIoU-matched precision, recall and F1.
S1 and stage-2 monitors score legal BIO paths on sampler windows. Streaming RQ2 scores committed events
on full videos. Whole-video segmenter evaluation also reports its spans through the RQ2 scoring boundary.
Frame diagnostics and span diagnostics answer different questions; training monitors are not final results.

FRAME-DOMAIN: bio_frame_metrics and moryossef_segment_metrics, with a shared span parser for predictions and gold.

TIME-DOMAIN (Segment(start_s, end_s) seconds) — Segment/temporal_iou/match_segments/segmentation_prf; used by eval.py 
(RQ2 tIoU brackets), report.py and reporting/ (outcome and protocol tables).

TEXT: compute_text_metrics (BLEU-4/ROUGE-L/METEOR/BLEURT).
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from data.windowing import BIO
from sacrebleu import sentence_bleu
from sacrebleu.metrics import BLEU
import numpy as np
import random
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
    # precision/recall/accuracy all stay high (they score signing-vs-not, which B and I share). Gold rate
    # alongside it, so a near-zero value is readable without another run.
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


def signing_runs_with_b_splits(tags: torch.Tensor | list[int]) -> list[dict]:
    """PREDICTION/inference decode: signing runs split at interior `B` (== moryossef26.infer.bio_tags_to_segments).

    Requiring a predicted `B` to OPEN is fatal (`B` is ~1% of frames, and most adjacent captions chain with no gap): 
    a signing-detecting model that never argmaxes `B` yields zero segments — Moryossef's `likeliest_probs_to_segments` 
    doesn't require one either. Interior `B`s split back-to-back sentences. `I` with nothing open opens (sentence start 
    after a gap; headless left-truncated fragment). `O` and `UNK` close. Returns [{start, end}] frame segments.
    """
    if isinstance(tags, torch.Tensor): tags = tags.detach().cpu().tolist()
    segments: list[dict] = []
    start: int | None = None
    for i, tag in enumerate(tags):
        if tag == BIO["B"]:
            if start is not None: segments.append({"start": start, "end": i - 1})
            start = i
        elif tag == BIO["I"]:
            if start is None: start = i
        elif tag in (BIO["O"], BIO["UNK"]) and start is not None:
            segments.append({"start": start, "end": i - 1}); start = None
    if start is not None: segments.append({"start": start, "end": len(tags) - 1})
    return segments

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


@dataclass
class CompleteSpanMetrics:
    # Pool complete-span counts across windows; fragments and empty true negatives earn no F1 credit.
    n_matches: int = 0
    n_pred: int = 0
    n_gold: int = 0

    def update(self, pred_tags, labels, lengths, min_span_frames=1, tiou_threshold=.5):
        from infer.commit_gate import bio_complete_spans
        for pred, gold, n in zip(pred_tags.detach().cpu(), labels.detach().cpu(), lengths.detach().cpu().tolist()):
            pred, gold = pred[:n], gold[:n]
            ignored = gold == BIO['UNK']
            # Parse predictions before using ignore labels: a GT ignore boundary must not create a predicted terminator.
            ps = [Segment(a, b) for a, b in bio_complete_spans(pred) if b-a >= min_span_frames and int(ignored[a:b].sum()) <= (b-a)/2]
            gs = [Segment(a, b) for a, b in bio_complete_spans(gold)]
            self.n_matches += len(match_segments(ps, gs, threshold=tiou_threshold))
            self.n_pred += len(ps); self.n_gold += len(gs)

    def compute(self, prefix='phrase'):
        p = self.n_matches / self.n_pred if self.n_pred else 0.
        r = self.n_matches / self.n_gold if self.n_gold else 0.
        return {f'{prefix}_tiou_f1': 2*p*r/(p+r) if p+r else 0., f'{prefix}_seg_precision': p, f'{prefix}_seg_recall': r, 
                f'{prefix}_n_matches': self.n_matches, f'{prefix}_n_pred': self.n_pred, f'{prefix}_n_gold': self.n_gold}


def moryossef_segment_metrics(
    logits: torch.Tensor, labels: torch.Tensor, prefix: str = "phrase", tiou_threshold: float = 0.5
) -> dict[str, float]:
    """External-protocol run overlap: 1 frame score and 1 segment score per item.

    `{prefix}_frame_f1`: macro F1 over O/B/I frame classes.
    `{prefix}_tiou_f1`/`_seg_precision`/`_seg_recall`: `segmentation_prf` (the RQ2 metric) on frame-unit segments.
    Clipped fragments can score well on short windows. Training monitors use CompleteSpanMetrics instead.
    Both sides decode with the inference rule (signing_runs_with_b_splits).
    Binary signing-frame overlap belongs to bio_frame_metrics; it does not test sentence boundaries.
    """
    # 1 decode on BOTH sides: make_bio_labels tags a left-truncated span as a HEADLESS I-run, so a B-required gold decode
    # emits NO segment where a perfect tagger's run decode emits one, an unavoidable FP capping precision far below 1.

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
        tags_i = logits[i].argmax(dim=-1)[:last]
        interior_unk = gold_v == BIO["UNK"]
        if bool(interior_unk.any()):
            # No reliable label in untrusted gaps: mask BOTH sides so UNK splits runs identically on both sides.
            tags_i = torch.where(interior_unk, torch.full_like(tags_i, BIO["UNK"]), tags_i)
        frame_f1s.append(_macro_frame_f1(tags_i[~interior_unk], gold_v[~interior_unk]))

        pred_segs = _frame_segments_to_seconds(signing_runs_with_b_splits(tags_i))
        gold_segs = _frame_segments_to_seconds(signing_runs_with_b_splits(gold_v))
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
_DVC_KEYS = (*_TEXT_KEYS, "cider")  # CIDEr exists only per video corpus (densevid), never per pair (SODA)
_DENSEVID_GARBAGE_SEED = 1337  # the original draws unseeded; we seed for reproducible tables (documented deviation)

def char_level_for_target(target_lang: str | None) -> bool:
    # Whether a target language is scored per CHARACTER (Uni-Sign `level='char'`) or per WORD.
    code = str(target_lang or "").split("_")[0].lower()
    return code in ("zh", "ja", "ko")  # languages scored per CHARACTER, by ISO prefix of `target_lang`

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


def _uni_sign_preprocess(
    predictions: list[str], references: list[str], char_level: bool | None = None,
) -> tuple[list[str], list[str], bool]:
    """Uni-Sign eval preprocessing (fine_tuning.py:284-288 + SLRT_metrics): word-level, or char-split with the
    full-width→ASCII punctuation fold when `char_level`. Shared by corpus & per-pair scorers so both preprocess identically.

    `char_level=None` falls back to detecting CJK in the references — kept only so a caller that genuinely has no
    language context still works on a Chinese corpus. Every caller that knows its target language should PASS the
    value (see `char_level_for_target`); the fallback is batch-dependent and must not decide a reported number.
    """
    is_cjk = bool(char_level) if char_level is not None else any(_char_split_cjk(ref) != ref for ref in references)

    def _proc(s: str, is_ref: bool) -> str:
        if not is_cjk: return s
        s = s.replace(" ", "").replace("\n", "")
        s = s.translate(_CJK_PUNCT_TABLE)
        return " ".join(list(s))

    return [_proc(p, False) for p in predictions], [_proc(r, True) for r in references], is_cjk


def _rouge_l(hyps: list[str], refs: list[str]) -> float:
    """ROUGE-L f-score (mean over sentences) via pltrdy `rouge` — Uni-Sign's package (`['rouge-l']['f']`, whitespace-tokenized 
    over char-split strings), matching SLRT_metrics.translation_performance. HF `evaluate`'s "rouge" is Google's rouge_score, 
    whose ROUGE-L F differs (~1 pt on CJK) and is NOT comparable to their 0.55. Empty hyp/ref scores 0 (pltrdy raises)."""
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


def _cider_corpus(hyps: list[str], refs: list[str]) -> float:
    try: # Official CIDEr-D (pycocoevalcap) over one video's pairs, as densevid_eval calls it; 0.0 with a warning if absent.
        from pycocoevalcap.cider.cider import Cider
    except Exception as e:
        print(f"[metrics] WARNING: pycocoevalcap unavailable ({type(e).__name__}); densevid_cider reads 0.0", flush=True)
        return 0.0
    score, _ = Cider().compute_score({i: [r] for i, r in enumerate(refs)}, {i: [h] for i, h in enumerate(hyps)})
    return float(score)


def _bleurt_scores(hyps: list[str], refs: list[str], checkpoint: str | None) -> list[float]:
    # Per-example BLEURT on RAW text (BleurtScorer.score is inherently per-example). No checkpoint / failure -> zeros.
    if not checkpoint or not hyps: return [0.0] * len(hyps)
    try:
        from bleurt.score import BleurtScorer
        return [float(x) for x in BleurtScorer(checkpoint).score(candidates=list(hyps), references=list(refs))]
    except Exception: return [0.0] * len(hyps)


def bleu_pair_counts(hyps: list[str], refs: list[str], char_level: bool | None = None) -> list[dict]:
    """Per-pair BLEU-4 ingredients under the shared preprocessing and tokenizer: hypothesis length, reference length,
    matched and total n-grams for n = 1..4. The unsmoothed corpus BLEU-4 of ANY set of pairs is `bleu_from_counts` of
    their rows, so a report can show exactly which counts a video or a corpus adds up."""
    scorer = BLEU(tokenize="13a")
    h, r, _ = _uni_sign_preprocess(list(hyps), list(refs), char_level)
    rows = []
    for hh, rr in zip(h, r):
        st = scorer.corpus_score([hh], [[rr]])
        rows.append({
            "sys_len": int(st.sys_len), "ref_len": int(st.ref_len), 
            "counts": [int(c) for c in st.counts], "totals": [int(t) for t in st.totals]
        })
    return rows


def bleu_from_counts(rows: list[dict]) -> float:
    """Corpus BLEU-4 from summed `bleu_pair_counts` rows, with NO smoothing: the same number as 1 video's `densevid_bleu4` (see 
    `densevid_text_metrics`), so reporting/protocol reproduces that column from its counts. The RQ1 corpus BLEU keeps sacrebleu's 
    default `exp` smoothing; the two differ only when 1 n-gram order has 0 matches in the whole set."""
    if not rows: return 0.0
    return float(BLEU.compute_bleu(
        correct=[sum(r["counts"][n] for r in rows) for n in range(4)], total=[sum(r["totals"][n] for r in rows) for n in range(4)],
        sys_len=sum(r["sys_len"] for r in rows), ref_len=sum(r["ref_len"] for r in rows), smooth_method="none",
    ).score)


def sentence_bleu_scores(hyps: list[str], refs: list[str], char_level: bool | None = None) -> list[float]:
    """Smoothed sentence BLEU per pair under the shared preprocessing — the BLEU column of `_sentence_text_scores` alone.
    The per-gold deployment score (reporting/outcomes.py localized_bleu4) needs only this column."""
    if not hyps: return []
    pred_proc, ref_proc, _ = _uni_sign_preprocess(hyps, refs, char_level)
    return [float(sentence_bleu(h, [r], tokenize="13a").score) for h, r in zip(pred_proc, ref_proc)]


def _corpus_metric(name: str, predictions, references, *, key: str, **kw) -> float:
    # Load an `evaluate` metric and score it, returning 0.0 if the metric is unavailable or the compute fails.
    metric = _load_evaluate_metric(name)
    if metric is None: return 0.0
    try: return float(metric.compute(predictions=predictions, references=references, **kw)[key])
    except Exception: return 0.0
    

def _sentence_text_scores(
    hyps: list[str], refs: list[str], sacrebleu_tokenize: str = "13a", 
    bleurt_checkpoint: str | None = "/tmp/BLEURT-20", char_level: bool | None = None,
) -> list[dict[str, float]]:
    """Per-pair sentence scores {bleu4(sentence), rougeL, meteor, bleurt} — the primitive the RQ2 fusion sums. BLEU here is 
    sentence-BLEU with sacrebleu's `exp` smoothing: corpus BLEU pools across pairs and cannot be split per pair, and 1 short 
    sentence often has no 4-gram match, so unsmoothed sentence BLEU would be 0 for most pairs. The densevid column is corpus 
    BLEU per video and is NOT smoothed (see `densevid_text_metrics`)."""
    if not hyps: return []
    pred_proc, ref_proc, _ = _uni_sign_preprocess(hyps, refs, char_level)
    bleu = [float(sentence_bleu(h, [r], tokenize=sacrebleu_tokenize).score) for h, r in zip(pred_proc, ref_proc)]
    bleurt = _bleurt_scores(hyps, refs, bleurt_checkpoint)
    rouge = [_rouge_l([h], [r]) for h, r in zip(pred_proc, ref_proc)]
    meteor = _load_evaluate_metric("meteor")
    met = [
        float(meteor.compute(predictions=[h], references=[r])["meteor"]) for h, r in zip(pred_proc, ref_proc)
    ] if meteor is not None else [0.0] * len(hyps)
    return [dict(zip(_TEXT_KEYS, vals)) for vals in zip(bleu, bleurt, rouge, met)]


def garbage_reference(rng: random.Random) -> str:
    # densevid_eval's reference for a prediction that overlaps no gold: a random lowercase string of 10..20 letters.
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(10, 20)))


def densevid_text_metrics(
    pairs_by_video: dict[str, list[tuple[str, str | None]]], *, char_level: bool | None = None, sacrebleu_tokenize: str = "13a",
    bleurt_checkpoint: str | None = "/tmp/BLEURT-20", prefix: str = "densevid", per_video: bool = False
) -> dict[str, float] | dict[str, dict[str, float]]:
    """DVC-style matching and aggregation (ranjaykrishna/densevid_eval), with the repository's text scorers.

    Faithful components, per tIoU threshold (the caller builds `pairs_by_video` for one threshold):
      * MANY-TO-MANY pairing — every (prediction, gt) pair with tIoU >= threshold is ONE single-reference evaluation instance 
        (a prediction overlapping k gold captions contributes k pairs);
      * GARBAGE baseline — a prediction with no overlapping gt is scored against a random lowercase string of length 10..20 
        (their `random_string(random.randint(10, 20))`; ref None in `pairs_by_video` marks these);
      * PER-VIDEO scoring then MEAN over gold videos — a gold video with no predictions scores 0;
      * the caller averages thresholds for the headline (our threshold_average does this for every text key).

    Deviations, both deliberate and documented: the garbage rng is SEEDED, and tokenization is our shared Uni-Sign preprocessing + 
    sacrebleu 13a instead of COCO BLEU with PTB tokenization. Shared text preprocessing does not make DVC and RQ1 corpus scores
    comparable: matching and aggregation differ. BLEU-4 is corpus BLEU over each video's pairs with NO smoothing, as COCO BLEU:
    a video with no 4-gram match scores 0 (COCO adds only a 1e-15 epsilon), where sacrebleu's default `exp` smoothing would give
    it credit. `bleu_from_counts` uses same setting, so its per-video number is this column. CIDEr-D is the official COCO scorer
    over the video's pairs (pycocoevalcap, IDF from that video's references — their Cider semantics); ROUGE-L/METEOR are per-pair 
    means (COCO semantics); BLEURT is not in the original toolkit and follows the same per-pair mean, zeros without a checkpoint.

    KNOWN PROPERTY, not a bug: the protocol is RECALL-BLIND — a missed gold caption never enters any pair, so it is uncharged. 
    It is the headline (continuity with the previous paper); the SODA fusion is reported beside it because it charges misses.
    """
    rng = random.Random(_DENSEVID_GARBAGE_SEED)
    out: dict[str, dict[str, float]] = {}
    for vid in sorted(pairs_by_video):
        pairs = pairs_by_video[vid]
        if not pairs:  # gold video with no predictions: original assigns 0 across scorers
            out[vid] = {k: 0.0 for k in _DVC_KEYS}
            continue
        hyps = [h for h, _ in pairs]
        refs = [r if r is not None else garbage_reference(rng) for _, r in pairs]
        hyp_p, ref_p, _ = _uni_sign_preprocess(hyps, refs, char_level)
        row = {
            "bleu4": _corpus_metric(
                "sacrebleu", hyp_p, [[r] for r in ref_p], key="score", tokenize=sacrebleu_tokenize, smooth_method="none"
            ),
            "rougeL": float(np.mean([_rouge_l([h], [r]) for h, r in zip(hyp_p, ref_p)])),
            "bleurt": float(np.mean(_bleurt_scores(hyps, refs, bleurt_checkpoint))),
            "cider": _cider_corpus(hyp_p, ref_p),
        }
        meteor = _load_evaluate_metric("meteor")
        row["meteor"] = float(np.mean([
            meteor.compute(predictions=[h], references=[r])["meteor"] for h, r in zip(hyp_p, ref_p)
        ])) if meteor is not None else 0.0
        out[vid] = row
    if per_video: return out  # 1 row per gold video, the rows the mean below is taken over
    if not out: return {f"{prefix}_{k}": 0.0 for k in _DVC_KEYS}
    return {f"{prefix}_{k}": float(np.mean([r[k] for r in out.values()])) for k in _DVC_KEYS}


def compute_text_metrics(
    predictions: list[str], references: list[str], *, localization_aware: bool = False,
    n_pred: int | None = None, n_gold: int | None = None, memo: dict[tuple[str, str], dict[str, float]] | None = None,
    sacrebleu_tokenize: str = "13a", bleurt_checkpoint: str | None = "/tmp/BLEURT-20", 
    char_level: bool | None = None, prefix: str | None = None
) -> dict[str, float]:
    """Translation-quality metrics, in 2 modes that are NOT comparable and are therefore named differently.

    2 modes answer different questions, so they get different key prefixes (`translation_*` vs `soda_*`). Sharing 1 name 
    invites exactly 1 mistake: reading a fused RQ2 cell as if it were translation quality & concluding the model collapsed. 
    The relationship is  soda_X ~= (mean per-pair X) * segmentation.f1, so a fused cell divided by the f1 reported beside 
    it recovers the per-pair quality — no extra metric needed.

    localization_aware=False (default — RQ1 and generated dev captions): CORPUS BLEU-4/ROUGE-L/METEOR/BLEURT over the whole set.
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
        if memo is None: per_pair = _sentence_text_scores(predictions, references, sacrebleu_tokenize, bleurt_checkpoint, char_level)
        else:
            uncached = [p for p in pairs if p not in memo]
            if uncached:
                scored = _sentence_text_scores(
                    [h for h, _ in uncached], [r for _, r in uncached], sacrebleu_tokenize, bleurt_checkpoint, char_level
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
    pred_proc, ref_proc, _ = _uni_sign_preprocess(predictions, references, char_level)
    ref_nested = [[r] for r in ref_proc]
    bleurt = _bleurt_scores(predictions, references, bleurt_checkpoint)
    out = {
        "bleu4": _corpus_metric("sacrebleu", pred_proc, ref_nested, key="score", tokenize=sacrebleu_tokenize),
        "bleurt": float(sum(bleurt) / len(bleurt)) if bleurt else 0.0,
        "rougeL": _rouge_l(pred_proc, ref_proc),
        "meteor": _corpus_metric("meteor", pred_proc, ref_proc, key="meteor"),
    }
    return {f"{prefix}_{k}": float(out[k]) for k in _TEXT_KEYS}
