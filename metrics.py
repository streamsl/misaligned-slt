"""Every metric calculation of the repository. `evaluation/` only builds the inputs (gold units and chunks from the records, 
the scoreable events of a file) and runs these; `report.py` and `reporting/` only read them.

SEGMENTS (Segment(start_s, end_s) seconds): temporal_iou, match_segments, segmentation_prf.
BIO TAGS (training and segmenter diagnostics): bio_frame_metrics, CompleteSpanMetrics, moryossef_segment_metrics.
TEXT (TEXT_KEYS: BLEU-4, BLEURT, ROUGE-L, METEOR, CIDEr-D; SODA_KEYS without CIDEr-D): pair_text_scores (per pair), 
text_scores (1 set of pairs), sentence_bleu_scores, the BLEU count rows (bleu_pair_counts, bleu_from_counts) and the
shared preprocessing (_uni_sign_preprocess) at the scoring level that each corpus declares (text_level: per character or per word,
and lowercase).

2 aggregations of matched pairs: densevid_text_metrics (the DVC of densevid_eval: per gold video, garbage references for 
unmatched predictions, the mean over gold videos) and soda_scores (the SODA-style text F1, which charges missed gold).

MAIN SCORES of a row, all on 1 pairing, the edge chunk match: edge_chunk_scores (edge chunk F1, and the DVC and the SODA-style text F1 
of its pairs). tIoU SCORES of a row, on the tIoU pairing: evaluate_predicted_events (the sentence tIoU F1 and the DVC of densevid_eval).

PER UNIT: uncaptioned_units, unit_holders, held_unit_latency, long_form_scores.
REPORT READOUTS (report.py results only, no `_scores.json` key): caption_shape, cue_edge_share, edge_chunk_classes, cap_cut_captions.
ERROR TYPES of the edge chunk match (report.py results and progress, analyze.py --stage segmenter-errors): edge_error_types.
DISPLAY REPLAY (`--stability`): score_policy, accumulate_policy, merged_policy, laal.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Any
import os
import random

import torch
import numpy as np
from sacrebleu import corpus_bleu, sentence_bleu
from sacrebleu.metrics import BLEU
from data.windowing import BIO
from infer.stability import Track, lcp, displays
try: from pycocoevalcap.cider.cider import Cider
except Exception: Cider = None   # CIDEr-D reads 0.0 without pycocoevalcap (requirements.txt lists it; `_cider_corpus` warns)


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


def match_segments(predicted: list[Segment], gold: list[Segment], threshold: float) -> list[tuple[int, int, float]]:
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
    """Signing runs split at interior `B`: the run rule of `moryossef_segment_metrics` (Moryossef-protocol frame and run scores 
    that --segmenter-eval and the online `fsm_bio` rows print beside the main scores) and of the `report.py` debug view, applied to the 
    tags and the labels alike. It is not the rule of an emitted span, nor of `val_phrase_*` (CompleteSpanMetrics): those use
    infer.commit_gate.bio_complete_spans, which opens no headless run and emits no run without a terminator (P1 at inference).

    Requiring predicted `B` to OPEN is fatal (`B` is ~1% of frames, and most adjacent captions chain without gap): signing-detecting 
    model that never argmaxes `B` yields 0 segments — Moryossef's `likeliest_probs_to_segments` doesn't require one either. Interior 
    `B`s split back-to-back sentences. `I` with nothing open opens (sentence start after a gap; headless left-truncated fragment). 
    `O` and `UNK` close. A run open at the end is kept. Returns [{start, end}] frame segments.
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


def macro_frame_f1(pred: torch.Tensor, gold: torch.Tensor, classes=(BIO["O"], BIO["B"], BIO["I"])) -> float:
    f1s = []
    for c in classes:
        tp = ((pred == c) & (gold == c)).sum().float()
        fp = ((pred == c) & (gold != c)).sum().float()
        fn = ((pred != c) & (gold == c)).sum().float()
        denom = (2 * tp + fp + fn)
        if denom > 0: f1s.append(float((2 * tp / denom).item()))
    return sum(f1s) / len(f1s) if f1s else 0.0


def segmentation_prf(predicted: list[Segment], gold: list[Segment], tiou_threshold: float) -> dict[str, float]:
    # One-to-one tIoU-matched precision/recall/F1/matches: the sentence tIoU F1 (a tIoU score of a row).
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
    `{prefix}_tiou_f1`/`_seg_precision`/`_seg_recall`: `segmentation_prf` (the sentence tIoU F1 of the rows) on frame-unit segments.
    Clipped fragments can score well on short windows. Training monitors use CompleteSpanMetrics instead.
    Both sides decode with the Moryossef-protocol run rule (signing_runs_with_b_splits), not the emitted-span rule.
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
        frame_f1s.append(macro_frame_f1(tags_i[~interior_unk], gold_v[~interior_unk]))

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
def char_level_for_target(target_lang: str | None) -> bool:
    # Whether a target language is scored per CHARACTER (Uni-Sign `level='char'`) or per WORD.
    code = str(target_lang or "").split("_")[0].lower()
    return code in ("zh", "ja", "ko")  # languages scored per CHARACTER, by ISO prefix of `target_lang`

@dataclass(frozen=True)
class TextLevel:
    """How the texts of 1 corpus are scored, declared by its data.yaml entry (`text_level`), never inferred from the texts: per
    character (`char`) or per word, and with `lowercase` the hypothesis and the reference both in lowercase. Every text score reads
    it in `_uni_sign_preprocess` (and BLEURT's raw text in `pair_text_scores`); the scorers take it as their `char_level` argument."""
    char: bool = False
    lowercase: bool = False

def text_level(target_lang: str | None, lowercase_references: bool | None = False) -> TextLevel:
    # The scoring level of 1 corpus: per character for a zh/ja/ko target (char_level_for_target), and lowercase on both sides when
    # its data.yaml entry sets lowercase_references (its captions carry no capitals), so every system is scored alike on it.
    return TextLevel(char=char_level_for_target(target_lang), lowercase=bool(lowercase_references))

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
    predictions: list[str], references: list[str], char_level: TextLevel | bool | None = None,
) -> tuple[list[str], list[str], TextLevel]:
    """Uni-Sign eval preprocessing (fine_tuning.py:284-288 + SLRT_metrics): word-level, or char-split with the
    full-width→ASCII punctuation fold at a per-character level, after both sides are lowercased when the level says so.
    Shared by corpus & per-pair scorers so both preprocess identically. Returns the level it applied.

    `char_level` is the corpus's `TextLevel` (`text_level`); a bool is a cased level (True: per character). `None` falls back to
    detecting CJK in the references — kept only so a caller that genuinely has no language context still works on a Chinese corpus.
    Every caller that knows its corpus should PASS the level; the fallback is batch-dependent and must not decide a reported number.
    """
    level = char_level if isinstance(char_level, TextLevel) else TextLevel(
        char=bool(char_level) if char_level is not None else any(_char_split_cjk(ref) != ref for ref in references))

    def _proc(s: str, is_ref: bool) -> str:
        if level.lowercase: s = s.lower()
        if not level.char: return s
        s = s.replace(" ", "").replace("\n", "")
        s = s.translate(_CJK_PUNCT_TABLE)
        return " ".join(list(s))

    return [_proc(p, False) for p in predictions], [_proc(r, True) for r in references], level


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
    `densevid_text_metrics`), so reporting/protocol reproduces that column from its counts. Pooled corpus BLEU-4 (the boundary sweep, `text_scores`
    default) keeps sacrebleu's `exp` smoothing. The two differ only when 1 n-gram order has 0 matches in the whole set."""
    if not rows: return 0.0
    return float(BLEU.compute_bleu(
        correct=[sum(r["counts"][n] for r in rows) for n in range(4)], total=[sum(r["totals"][n] for r in rows) for n in range(4)],
        sys_len=sum(r["sys_len"] for r in rows), ref_len=sum(r["ref_len"] for r in rows), smooth_method="none",
    ).score)

# ---------------------------------------------------------------- text scores

# 1 metric set for every text table (the boundary sweep, the edge chunk DVC, the tIoU DVC), in this column order.
TEXT_KEYS = ("bleu4", "bleurt", "rougeL", "meteor", "cider")
# SODA-style F1 sums per-pair scores. CIDEr-D needs a corpus document frequency, so it has no per-pair score and no SODA column.
SODA_KEYS = ("bleu4", "bleurt", "rougeL", "meteor")

@lru_cache(maxsize=None)
def _warn_once(message: str) -> None:   # a scorer that cannot run reads 0.0; say so once per run, not once per call
    print(f"[metrics] WARNING: {message}; its column reads 0.0 — do NOT report that cell.", flush=True)

def _cider_corpus(hyps: list[str], refs: list[str]) -> float: # Official CIDEr-D (pycocoevalcap) over 1 set of preprocessed pairs
    # IDF from that set's references, as densevid_eval calls it per video.
    if not hyps: return 0.0
    if Cider is None: _warn_once("CIDEr-D needs pycocoevalcap"); return 0.0
    return float(Cider().compute_score({i: [r] for i, r in enumerate(refs)}, {i: [h] for i, h in enumerate(hyps)})[0])

@lru_cache(maxsize=None)
def _bleurt_scorer(checkpoint: str):
    # 1 load per checkpoint and run: BLEURT-20 is a 2 GB TensorFlow model, and the DVC scores call BLEURT once per video.
    # TensorFlow then takes GPU memory as it needs it, not all free memory at its first op, so it shares the GPU with torch.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    from bleurt.score import BleurtScorer
    return BleurtScorer(checkpoint)

def _bleurt_scores(hyps: list[str], refs: list[str], checkpoint: str | None) -> list[float]:
    # Per-example BLEURT on RAW text (BleurtScorer.score is inherently per-example). Switched off (None) -> zeros.
    # A missing checkpoint or a failure -> zeros with one warning, as CIDEr-D warns.
    if not checkpoint or not hyps: return [0.0] * len(hyps)
    try:
        return [float(x) for x in _bleurt_scorer(checkpoint).score(candidates=list(hyps), references=list(refs))]
    except Exception as exc:
        _warn_once(f"BLEURT could not run with checkpoint {checkpoint} ({type(exc).__name__})")
        return [0.0] * len(hyps)

def sentence_bleu_scores(hyps: list[str], refs: list[str], char_level: bool | None = None) -> list[float]:
    """Sentence BLEU-4 per pair (sacrebleu 13a, `exp` smoothing: 1 short sentence often has no 4-gram match, so unsmoothed 
    sentence BLEU would be 0 for most pairs), under the shared preprocessing: the BLEU column of `pair_text_scores`."""
    if not hyps: return []
    h, r, _ = _uni_sign_preprocess(hyps, refs, char_level)
    return [float(sentence_bleu(a, [b], tokenize="13a").score) for a, b in zip(h, r)]


def pair_text_scores(
    hyps: list[str], refs: list[str], char_level: bool | None = None, bleurt_checkpoint: str | None = "/tmp/BLEURT-20",
    keys: tuple[str, ...] = SODA_KEYS,
) -> list[dict[str, float]]:
    """The per-pair scores of each (hypothesis, reference) pair, 1 dict per pair with `keys` (SODA_KEYS or a part of it), 
    under the shared preprocessing: sentence BLEU-4 (`sentence_bleu_scores`), BLEURT on the raw text (0 without a checkpoint), 
    ROUGE-L and METEOR. The SODA-style F1 sums them (`soda_scores`); `text_scores` averages BLEURT, ROUGE-L and METEOR."""
    if not hyps: return []
    h, r, level = _uni_sign_preprocess(hyps, refs, char_level)
    columns = {}
    if "bleu4" in keys: columns["bleu4"] = sentence_bleu_scores(hyps, refs, char_level)
    raw = ([t.lower() for t in hyps], [t.lower() for t in refs]) if level.lowercase else (hyps, refs)   # BLEURT reads the raw text
    if "bleurt" in keys: columns["bleurt"] = _bleurt_scores(*raw, bleurt_checkpoint)
    if "rougeL" in keys: columns["rougeL"] = [_rouge_l([a], [b]) for a, b in zip(h, r)]
    if "meteor" in keys:
        meteor = _load_evaluate_metric("meteor")
        try: columns["meteor"] = [
            meteor.compute(predictions=[a], references=[b])["meteor"] for a, b in zip(h, r)
        ] if meteor is not None else [0.0] * len(h)
        except Exception as err: _warn_once(f"METEOR failed ({type(err).__name__}: {err})"); columns["meteor"] = [0.0] * len(h)
    return [{k: float(columns[k][i]) for k in keys} for i in range(len(hyps))]


def text_scores(
    hyps: list[str], refs: list[str], char_level: bool | None = None, bleurt_checkpoint: str | None = "/tmp/BLEURT-20",
    keys: tuple[str, ...] = TEXT_KEYS, smooth_method: str = "exp",
) -> dict[str, float]:
    """The `keys` (TEXT_KEYS) of 1 set of pairs, under the shared preprocessing. BLEU-4 is corpus BLEU over the set (n-gram counts
    pooled, sacrebleu 13a): `smooth_method` "exp" is sacrebleu's default, the corpus BLEU-4 that Uni-Sign reports (the boundary sweep and every 
    pooled BLEU-4). `densevid_text_metrics` passes "none", as COCO BLEU in densevid_eval. CIDEr-D is the corpus score of the set 
    (`_cider_corpus`). BLEURT, ROUGE-L and METEOR are the means of their per-pair scores (`pair_text_scores`). A caller that needs 
    only BLEU-4 passes keys=("bleu4",) and computes nothing else."""
    if not hyps: return {k: 0.0 for k in keys}
    h, r, _ = _uni_sign_preprocess(hyps, refs, char_level)
    means = tuple(k for k in keys if k in ("bleurt", "rougeL", "meteor"))
    per_pair = pair_text_scores(hyps, refs, char_level, bleurt_checkpoint, keys=means) if means else []
    out = {k: float(np.mean([p[k] for p in per_pair])) for k in means}
    if "bleu4" in keys: out["bleu4"] = float(corpus_bleu(h, [r], tokenize="13a", smooth_method=smooth_method).score)
    if "cider" in keys: out["cider"] = _cider_corpus(h, r)
    return {k: out[k] for k in keys}


def soda_scores(
    pairs: dict[str, list[tuple[str, str, int]]], n_captions: dict[str, int], n_units: dict[str, int], char_level: bool | None = None,
    bleurt_checkpoint: str | None = "/tmp/BLEURT-20", keys: tuple[str, ...] = SODA_KEYS, memo: dict | None = None,
) -> dict[str, float]:
    """The SODA-style text F1 (Fujita et al. 2020) of matched pairs, per `keys` (SODA_KEYS or a part of it). `pairs` = {video:
    [(caption text, gold text, gold units the pair holds)]}; `n_captions` and `n_units` = {video: count}. Per video: precision = 
    summed pair scores / captions, recall = summed (pair score x the units of the pair) / units, F1 = 2PR / (P + R), 0 for a video 
    with no pair; then the mean F1 over the videos with captions or units. With 1 unit/pair this is SODA's sum over the matched gold. 
    A caption that holds a run of k units credits each of k units, as the edge chunk F1 does. Per-pair scores (`pair_text_scores`) 
    are cached in `memo` ((caption, gold) -> scores), so the taus of 1 row score each pair once."""
    memo = {} if memo is None else memo
    todo = sorted({(h, g) for v in pairs for h, g, _ in pairs[v]} - memo.keys())
    for pair, score in zip(todo, pair_text_scores([h for h, _ in todo], [g for _, g in todo], char_level, bleurt_checkpoint, keys)):
        memo[pair] = score
    videos = sorted(v for v in set(n_captions) | set(n_units) if n_captions.get(v) or n_units.get(v))
    out = {}
    for k in keys:
        f1 = []
        for v in videos:
            rows = pairs.get(v, [])
            p = sum(memo[(h, g)][k] for h, g, _ in rows) / n_captions[v] if n_captions.get(v) else 0.0
            r = sum(memo[(h, g)][k] * u for h, g, u in rows) / n_units[v] if n_units.get(v) else 0.0
            f1.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
        out[k] = float(np.mean(f1)) if f1 else 0.0
    return out

# ---------------------------------------------------------------- tIoU scores of a row: sentence tIoU F1 and the DVC of densevid_eval

DENSEVID_GARBAGE_SEED = 1337  # the original draws unseeded; we seed for reproducible tables (documented deviation)

def garbage_reference(rng: random.Random) -> str:
    # densevid_eval's reference for a prediction that overlaps no gold: a random lowercase string of 10..20 letters.
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(10, 20)))

def densevid_pairs(pred_events, gold_events, threshold: float) -> list[tuple[str, str | None]]:
    """DVC pairs of 1 video at 1 threshold: every (prediction, gold) pair with tIoU >= threshold, and (prediction, None)
    for a prediction that overlaps no gold (None = garbage reference, drawn by the scorer)."""
    out = []
    for ev in pred_events:
        if ev.text is None: continue
        over = [gt.text for gt in gold_events if temporal_iou(ev.segment, gt.segment) >= float(threshold) and gt.text is not None]
        out.extend((ev.text, r) for r in over) if over else out.append((ev.text, None))
    return out

def densevid_text_metrics(
    pairs_by_video: dict[str, list[tuple[str, str | None]]], *, char_level: bool | None = None,
    bleurt_checkpoint: str | None = "/tmp/BLEURT-20", prefix: str = "densevid", 
    per_video: bool = False, keys: tuple[str, ...] = TEXT_KEYS,
) -> dict[str, float] | dict[str, dict[str, float]]:
    """The DVC aggregation of densevid_eval (ranjaykrishna/densevid_eval) over given pairs, with the repository's text scores
    (`text_scores`, `keys` of TEXT_KEYS). `pairs_by_video` = {gold video: [(prediction text, gold text, or None for a prediction 
    with no gold pair)]}, for 1 threshold. The caller makes the pairs (`densevid_pairs` for the tIoU pairing, `edge_chunk_scores` 
    for the edge chunk pairing). A GARBAGE reference for a prediction with no gold pair (a random lowercase string of length 10..20; 
    ref None marks these). PER-VIDEO scoring, then the MEAN over gold videos (a gold video with no predictions scores 0). Deliberate 
    deviations: the garbage rng is SEEDED, and tokenization is the shared Uni-Sign preprocessing + sacrebleu 13a, not COCO BLEU with 
    PTB tokenization. BLEU-4 is corpus BLEU over each video's pairs with NO smoothing, as COCO BLEU (`bleu_from_counts` gives the same 
    number). CIDEr-D takes its IDF from the video's references, as densevid_eval does. BLEURT, ROUGE-L and METEOR are per-pair means. 
    The aggregation is RECALL-BLIND: a gold unit with no pair adds nothing; the SODA-style F1 (`soda_scores`) charges it."""
    rng = random.Random(DENSEVID_GARBAGE_SEED)
    out: dict[str, dict[str, float]] = {}
    for vid in sorted(pairs_by_video):
        pairs = pairs_by_video[vid]
        refs = [r if r is not None else garbage_reference(rng) for _, r in pairs]
        out[vid] = text_scores([h for h, _ in pairs], refs, char_level, bleurt_checkpoint, keys=keys, smooth_method="none")
    if per_video: return out  # 1 row per gold video, the rows the mean below is taken over
    return {f"{prefix}_{k}": float(np.mean([row[k] for row in out.values()])) if out else 0.0 for k in keys}


def evaluate_predicted_events(
    predicted: dict[str, list], gold: dict[str, list], thresholds: list[float], char_level: bool = False, densevid: bool = False,
) -> dict[str, Any]:
    """tIoU scores of a row (the main scores are `edge_chunk_scores`). Per tIoU threshold: predictions and gold are matched 
    1-to-1 by tIoU (`match_segments`), and the segmentation block reports that sentence-level P/R/F1. With densevid=True the text 
    block holds the DVC columns of densevid_eval: many-to-many tIoU pairs (`densevid_pairs`), garbage references for unmatched 
    predictions, per-video corpus scores averaged over gold videos (`densevid_text_metrics`). They do not charge missed gold. 
    `threshold_average` averages the threshold grid; emission latency and matched tIoU are reported separately."""
    results = []
    all_video_ids = sorted(set(predicted) | set(gold))
    for threshold in thresholds:
        dv_pairs: dict[str, list[tuple[str, str | None]]] = {}
        vid_prec, vid_rec, latencies, best_tious = [], [], [], []
        n_pairs = n_unmatched_pred = n_missed_gold = total_pred = total_gold = 0
        for video_id in all_video_ids:
            pred_events, gold_events = predicted.get(video_id, []), gold.get(video_id, [])
            if not pred_events and not gold_events: continue
            total_pred += len(pred_events); total_gold += len(gold_events)

            # SEGMENTATION: sentence tIoU F1 — greedy 1-to-1 tIoU matching, shared verbatim with segmenter-eval and the BIO monitor. 
            # densevid's own proposal P/R is coverage-based (many-to-many), which lets duplicate spans all "match" 1 gold and keeps 
            # precision at 1.0; 1-to-1 charges the duplicate, so it is the defensible number for a segmentation claim.
            pred_segs, gold_segs = [ev.segment for ev in pred_events], [gt.segment for gt in gold_events]
            if densevid and gold_events: # DVC pairs (densevid_text_metrics documents the protocol and its recall-blindness)
                dv_pairs[video_id] = densevid_pairs(pred_events, gold_events, threshold)

            prf = segmentation_prf(pred_segs, gold_segs, tiou_threshold=float(threshold))
            if pred_events: vid_prec.append(prf["precision"])
            if gold_events: vid_rec.append(prf["recall"])
            one_to_one = match_segments(pred_segs, gold_segs, threshold=float(threshold))

            # Counts come from SAME 1-to-1 matching as P/R, so they never contradict it: a merged span covering 2 gold sentences 
            # matches 1 and leaves the other missed — under many-to-many bookkeeping it will read "0 missed" against that same recall.
            n_missed_gold += len(gold_events) - len(one_to_one)
            for pi, gi, iou in one_to_one:
                best_tious.append(float(iou))
                if pred_events[pi].commit_time_s is not None: # Emission latency.
                    latencies.append(float(pred_events[pi].commit_time_s) - float(gold_events[gi].end_s))

            n_pairs += sum(1 for pi, gi, _ in one_to_one if pred_events[pi].text is not None and gold_events[gi].text is not None)
            n_unmatched_pred += len(pred_events) - len(one_to_one)

        text_metrics = densevid_text_metrics(dv_pairs, char_level=char_level) if densevid else {}
        precision = float(np.mean(vid_prec)) if vid_prec else 0.0
        recall = float(np.mean(vid_rec)) if vid_rec else 0.0
        
        latency_block, tiou_block = {}, {}
        if latencies:
            arr = np.sort(np.asarray(latencies, dtype=np.float64))
            latency_block = {
                "median_latency_s": float(np.median(arr)),
                "p90_latency_s": float(arr[min(len(arr) - 1, int(round(0.9 * (len(arr) - 1))))]),
                "n": len(arr)
            }
        if best_tious:
            arr = np.asarray(best_tious, dtype=np.float64)
            tiou_block = {
                "min": float(np.min(arr)), "p25": float(np.percentile(arr, 25)),
                "median": float(np.median(arr)), "p75": float(np.percentile(arr, 75)),
                "max": float(np.max(arr))
            }
        results.append({
            "tiou_threshold": float(threshold),
            "segmentation": {
                "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            },
            "text_metrics": text_metrics,
            # scored_pairs: the 1-to-1 pairs with text; unmatched_predictions support `precision`, missed_gold `recall`.
            "scored_pairs": n_pairs, "unmatched_predictions": n_unmatched_pred, "missed_gold": n_missed_gold,
            "total_predictions": total_pred, "total_gold": total_gold,
            "mean_matched_tiou": float(np.mean(best_tious)) if best_tious else 0.0,
            "matched_tiou_distribution": tiou_block, "emission_latency": latency_block,
        })
    keys = results[0]["text_metrics"] if results else {}
    summary = {k: float(np.mean([r["text_metrics"][k] for r in results])) for k in keys}
    summary["segmentation_f1"] = float(np.mean([r["segmentation"]["f1"] for r in results]))
    return {"thresholds": results, "threshold_average": summary}

# ---------------------------------------------------------------- main scores of a row: the edge chunk match, its F1 and its text scores

def _edge_items(chunks: list[list]) -> list[tuple[float, float, str, range]]:
    # The admissible gold items of 1 video: every run of consecutive units inside 1 chunk, as (start of its first unit, 
    # end of its last, their texts joined, the ids of its units). A chunk of n units gives n (n + 1) / 2 items.
    items, first = [], 0
    for chunk in chunks:
        items += [(
            chunk[i].start_s, chunk[j].end_s, " ".join(u.text for u in chunk[i:j + 1]), range(first + i, first + j + 1)
        ) for i in range(len(chunk)) for j in range(i, len(chunk))]
        first += len(chunk)
    return items

def _edge_matches(events: list, items: list, tau: float) -> list[tuple[int, int]]:
    # (event, item) pairs: an event matches an item only when its start AND its end are within tau of the item's. Greedy 1-to-1
    # by the summed edge error, ties by event then item order; each event is used once, and an item is taken only when none of its
    # units is used yet. No tIoU floor. It scans all events x items of a video; vectorise it if long test chunks make it slow.
    cand = sorted(
        (abs(e.start_s - it[0]) + abs(e.end_s - it[1]), pi, gi) 
        for pi, e in enumerate(events) for gi, it in enumerate(items) 
        if abs(e.start_s - it[0]) <= tau and abs(e.end_s - it[1]) <= tau
    )
    used_events, used_units, pairs = set(), set(), []
    for _, pi, gi in cand:
        if pi in used_events or used_units.intersection(items[gi][3]): continue
        used_events.add(pi); used_units.update(items[gi][3]); pairs.append((pi, gi))
    return pairs

def _table_prf(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per column (an edge tau) of a (videos, columns, 4) table of precision, has events, recall, has gold: precision averaged over
    the videos with events, recall over the videos with gold, and their F1. The headline F1 avg is the mean of the F1s; it is not a
    mean over videos."""
    has_p, has_g = t[:, :, 1], t[:, :, 3]
    p = (t[:, :, 0] * has_p).sum(0) / np.maximum(has_p.sum(0), 1)
    r = (t[:, :, 2] * has_g).sum(0) / np.maximum(has_g.sum(0), 1)
    return p, r, np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)

def edge_chunk_table(pred: dict, chunks: dict, taus, vids: list[str]) -> tuple[np.ndarray, list[dict[str, dict[int, tuple[str, int]]]]]:
    """The edge chunk match of `pred` on `vids`: a (videos, taus, 4) table, the form `_table_prf` reads: precision (matched 
    captions / captions), has captions, recall (units inside matched items / units), has units; and per tau the matches as 
    {video: {index of the caption in pred[video]: (joined text of its item, units in the item)}}."""
    t, matches = np.zeros((len(vids), len(taus), 4)), [{} for _ in taus]
    for i, v in enumerate(vids):
        items, evs, n_units = _edge_items(chunks.get(v, [])), pred.get(v, []), sum(map(len, chunks.get(v, [])))
        for j, tau in enumerate(taus):
            matched = _edge_matches(evs, items, float(tau))
            covered = sum(len(items[gi][3]) for _, gi in matched)   # matched items share no unit
            t[i, j] = (
                len(matched) / len(evs) if evs else 0.0, float(bool(evs)),
                covered / n_units if n_units else 0.0, float(bool(n_units))
            )
            matches[j][v] = {pi: (items[gi][2], len(items[gi][3])) for pi, gi in matched}
    return t, matches

def edge_chunk_scores(
    pred: dict, chunks: dict, taus, char_level: bool, 
    bleurt_checkpoint: str | None = "/tmp/BLEURT-20", keys: tuple[str, ...] = TEXT_KEYS,
) -> dict:
    """The main scores of a row. All read the edge chunk match. A chunk is a run of reliable units that no pause separates
    (`evaluation.predictions.gold_chunks`). A caption may hold any run of consecutive units inside 1 chunk (an item), so a merge 
    inside a chunk is correct when both of its outer edges are within tau. An edge more than tau from every run edge (past a pause, 
    or inside a unit) matches nothing. A caption matches an item when both its edges are within tau of the item's (`_edge_matches`). 
    Its gold text is the joined text of the item's units. Per tau:
      * the edge chunk F1 (where): precision = matched captions / captions per video, averaged over the videos with captions. 
        recall = units inside matched items / units per video, averaged over the videos with units; F1 of the two;
      * `dvc` (what): the DVC aggregation of densevid_eval on these pairs (`densevid_text_metrics`, the `keys` of TEXT_KEYS): 
        per gold video, each caption against the gold text of its match, a caption with no match against a garbage reference; 
        the mean over gold videos;
      * `soda` (what, missed units charged): the SODA-style text F1 on the same pairs (`soda_scores`, the `keys` of SODA_KEYS).
        `f1_avg`, `dvc_avg` and `soda_avg` average all taus (eval.yaml scoring.edge_taus_s). The text blocks are None for a row with 
        no caption text (segmentation only).
    """
    vids = sorted(v for v in set(pred) | set(chunks) if pred.get(v) or chunks.get(v))
    t, matches = edge_chunk_table(pred, chunks, taus, vids)
    p, r, f1 = _table_prf(t)
    has_text = any(e.text is not None for v in vids for e in pred.get(v, []))
    n_captions = {v: len(pred.get(v, [])) for v in vids}
    n_units = {v: sum(map(len, chunks.get(v, []))) for v in vids}
    soda_keys, memo, rows = tuple(k for k in keys if k in SODA_KEYS), {}, []
    for j, tau in enumerate(taus):
        row = {
            "tau_s": float(tau), "precision": float(p[j]), "recall": float(r[j]), 
            "f1": float(f1[j]), "dvc": None, "soda": None,
            "matched_captions": sum(map(len, matches[j].values())), 
            "matched_units": sum(k for m in matches[j].values() for _, k in m.values())
        }
        if has_text:
            dvc = densevid_text_metrics({v: [
                (e.text, matches[j][v][pi][0] if pi in matches[j][v] else None) 
                for pi, e in enumerate(pred.get(v, [])) if e.text is not None
            ] for v in vids if n_units[v]}, char_level=char_level, bleurt_checkpoint=bleurt_checkpoint, keys=keys)
            row["dvc"] = {k: dvc[f"densevid_{k}"] for k in keys}
            row["soda"] = soda_scores({v: [
                (pred[v][pi].text, g, k) for pi, (g, k) in matches[j][v].items() if pred[v][pi].text is not None
            ] for v in vids}, n_captions, n_units, char_level, bleurt_checkpoint, soda_keys, memo)
        rows.append(row)
    over_taus = lambda name: {
        k: float(np.mean([x[name][k] for x in rows])) for k in rows[0][name]
    } if rows and has_text else None
    return {
        "f1_avg": float(f1.mean()), "precision_avg": float(p.mean()), "recall_avg": float(r.mean()),
        "dvc_avg": over_taus("dvc"), "soda_avg": over_taus("soda"), "taus": rows, "captions": sum(n_captions.values()), 
        "units": sum(n_units.values()), "chunks": sum(len(chunks.get(v, [])) for v in vids),
    }

# ---------------------------------------------------------------- scores per reliable gold unit

def holds(e, g, cover: float) -> bool:   # the event holds at least `cover` of the unit's duration
    return min(e.end_s, g.end_s) - max(e.start_s, g.start_s) >= cover * max(g.end_s - g.start_s, 1e-6)

def uncaptioned_units(pred: dict, gold: dict) -> dict:
    """Reliable gold units that no scored caption overlaps at all (no shared time), beside the edge chunk score in every row. 
    Edge score counts such a unit as a miss, same as a unit under a wrong caption. This count separates the two. It includes the 
    cost of P1 at inference: at the cap an open span with a decoded start that the cap cannot keep gets 1 event for its closed 
    part only, so the rest of it, signing with no decoded start and an open span at the stream end get no caption."""
    units = [(vid, g) for vid, gl in gold.items() for g in gl]
    count = sum(1 for vid, g in units if not any(e.end_s > g.start_s and e.start_s < g.end_s for e in pred.get(vid, [])))
    return {"units": len(units), "uncaptioned": count, "share": count / len(units) if units else 0.0}

def unit_holders(gold: dict, pred: dict, cover: float = 0.5) -> dict:
    """{video: [per reliable gold unit, the event that holds at least `cover` of it and was emitted first (commit time, 
    else order), or None]}: the one event-to-unit rule of the held-unit latency, the span classes and the commit-only latency."""
    out = {}
    for vid, units in gold.items():
        evs = pred.get(vid, [])
        order = sorted(
            range(len(evs)), 
            key=lambda i: (evs[i].commit_time_s if evs[i].commit_time_s is not None else float("inf"), i)
        )
        out[vid] = [next((evs[i] for i in order if holds(evs[i], g, cover)), None) for g in units]
    return out

def held_unit_latency(gold: dict, pred: dict, cover: float = 0.5) -> dict | None:
    """Latency per reliable gold unit: commit time of 1st event that holds at least `cover` of the unit, minus unit's end. 
    The headline latency uses only 1-to-1 matched pairs, so a unit inside a merged event (it waits for the whole span) drops 
    out of it. Here that unit counts, with the merged event's commit time. Units that no event holds are counted as not covered, 
    never as 0 latency. None when the events carry no commit times (offline rows)."""
    if not any(e.commit_time_s is not None for evs in pred.values() for e in evs): return None
    lat = np.concatenate([v for v in _unit_latencies(gold, pred, cover).values()] or [np.zeros(0)])
    n, arr = len(lat), np.sort(lat[np.isfinite(lat)])
    pick = lambda q: round(float(arr[min(len(arr) - 1, int(round(q * (len(arr) - 1))))]), 3) if len(arr) else float("nan")
    return {"units": n, "covered": len(arr), "covered_share": round(len(arr) / max(n, 1), 4), "p90_s": pick(0.9), 
            "median_s": round(float(np.median(arr)), 3) if len(arr) else float("nan"),
            "mean_s": round(float(arr.mean()), 3) if len(arr) else float("nan")}

def _unit_latencies(gold: dict, pred: dict, cover: float) -> dict:
    # {video: per-unit latency (commit of its holder minus its end), +inf when no event with a commit time holds it}
    timed = {vid: [e for e in evs if e.commit_time_s is not None] for vid, evs in pred.items()}
    return {vid: np.array([np.inf if h is None else h.commit_time_s - g.end_s for g, h in zip(units, holders)], dtype=float)
            for vid, units in gold.items() for holders in [unit_holders({vid: units}, timed, cover)[vid]]}

def long_form_scores(gold: dict, pred: dict, unk: dict, char_level: bool, cover: float = 0.5) -> dict:
    """Long-form BLEU-4: per source video, event texts in commit order (ties by start) against reliable gold texts in time order, scored 
    as 1 corpus of documents. Event at least half inside an UNK stretch (unk = {video: [(start, end)]}: uncaptioned gaps and quarantined 
    units) has no reference there and is dropped first, and the dropped share is reported. LAAL and AL per gold unit (laal): every word 
    of the unit's holder (1st event holding at least `cover` of it) appears at that event's commit time, since committed text never 
    changes (NE = 0). A mean per video, then over videos. Units that no event holds are reported as the uncovered share."""
    hyps, refs, dropped, total, laal_v, al_v, covered, n_units = [], [], 0, 0, [], [], 0, 0
    for vid, units in gold.items():
        evs = pred.get(vid, []); total += len(evs)
        keep = [e for e in evs if sum(
            max(0.0, min(e.end_s, b) - max(e.start_s, a)) for a, b in unk.get(vid, [])
        ) < 0.5 * max(e.end_s - e.start_s, 1e-6)]
        dropped += len(evs) - len(keep)
        keep.sort(key=lambda e: (e.commit_time_s if e.commit_time_s is not None else e.end_s, e.start_s))
        hyps.append(" ".join(e.text for e in keep if e.text)); refs.append(" ".join(g.text for g in units if g.text))
        lv, av = [], []

        for g, h in zip(units, unit_holders({vid: units}, {vid: keep}, cover)[vid]):
            n_units += 1
            if h is None or h.commit_time_s is None or not h.text: continue
            covered += 1
            words, ref_words = [h.commit_time_s] * len(h.text.split()), len((g.text or "").split())
            lv.append(laal(words, g.start_s, g.end_s, ref_words)); av.append(laal(words, g.start_s, g.end_s, 0))
        if lv: laal_v.append(float(np.mean(lv))); al_v.append(float(np.mean(av)))

    bleu = text_scores(hyps, refs, char_level, keys=("bleu4",))["bleu4"] if hyps else 0.0
    return {"long_form_bleu4": round(float(bleu), 2), "dropped_share": round(dropped / max(total, 1), 4),
            "laal_s": round(float(np.mean(laal_v)), 3) if laal_v else float("nan"),
            "al_s": round(float(np.mean(al_v)), 3) if al_v else float("nan"),
            "covered_share": round(covered / max(n_units, 1), 4), "videos": len(hyps)}

# ---------------------------------------------------------------- report readouts (report.py results; no `_scores.json` key)

def caption_shape(pred: dict, gold: dict, cover: float = 0.5) -> dict:
    """The shape of the scored captions `pred`: their count, their durations (end - start, s: p10, p50, p90, max, mean), and k, 
    the reliable units of the same video that a caption holds (at least `cover` of each, the rule of `unit_holders`), as the 
    shares of k = 0, 1, 2, 3 and 4+ and the mean k. Gold units passed as `pred` give the durations of the units themselves."""
    rows = [(e.end_s - e.start_s, sum(holds(e, g, cover) for g in gold.get(vid, []))) for vid, evs in pred.items() for e in evs]
    d, k = np.array([r[0] for r in rows], dtype=float), np.array([r[1] for r in rows], dtype=int)
    stat = lambda f: float(f(d)) if len(d) else float("nan")
    share = lambda m: float(m.mean()) if len(k) else float("nan")
    return {"captions": len(rows),
            "duration_s": {"p10": stat(lambda x: np.percentile(x, 10)), "p50": stat(np.median), 
                           "p90": stat(lambda x: np.percentile(x, 90)), "max": stat(np.max), "mean": stat(np.mean)},
            "units_held": {**{str(i): share(k == i) for i in range(4)}, "4+": share(k >= 4), "mean": share(k)}}

def _union_inside(intervals, lo: float, hi: float) -> float:   # the length of the union of `intervals` inside [lo, hi]
    total, end = 0.0, lo
    for a, b in sorted(intervals):
        a, b = max(a, end), min(b, hi)
        if b > a: total += b - a; end = b
    return total

def cue_edge_share(pred: dict, gold: dict, breaks: dict, taus) -> dict:
    """Do the captions cut inside a unit where the captioner broke the display? `breaks` = {video: [(end of display cue k, start 
    of cue k + 1)]} (`evaluation.predictions.cue_breaks`). A junction is inner to a reliable unit when it lies strictly inside it, 
    so a unit of n display cues holds n - 1 of them. Per tau: an inside edge is a start/end of a caption that lies in a reliable 
    unit and more than tau from both of its edges (it can match no run at tau). `at_cue_break` = the share of inside edges within 
    tau of an inner junction of that unit. `chance` = mean over same edges of |N ∩ I| / |I|, I = (unit start + tau, unit end - tau) 
    and N = the union of [cue end - tau, next cue start + tau] over the unit's inner junctions. A share above chance means the 
    captions cut inside units where the display breaks. Also: the units, the units with an inner junction and inner junctions."""
    # 1e-6 s: a junction at a unit edge stays outside it after the round trip of the times through a cut stream's clock
    inner = {vid: [[
        (min(x, y), max(x, y)) for x, y in breaks.get(vid, []) if g.start_s + 1e-6 < min(x, y) and max(x, y) < g.end_s - 1e-6
    ] for g in units] for vid, units in gold.items()}
    rows = []
    for tau in map(float, taus):
        n = hits = 0; chance = 0.0
        for vid, units in gold.items():
            for t in (x for e in pred.get(vid, []) for x in (e.start_s, e.end_s)):
                for g, junctions in zip(units, inner[vid]):
                    lo, hi = g.start_s + tau, g.end_s - tau
                    if not lo < t < hi: continue
                    n += 1
                    hits += any(a - tau <= t <= b + tau for a, b in junctions)
                    chance += _union_inside([(a - tau, b + tau) for a, b in junctions], lo, hi) / (hi - lo)
        rows.append({"tau_s": tau, "inside_edges": n, "at_cue_break": hits / n if n else 0.0, "chance": chance / n if n else 0.0})
    held = [j for units in inner.values() for j in units]
    return {"taus": rows, "units": len(held), "units_with_break": sum(1 for j in held if j), "breaks": sum(map(len, held))}

def edge_chunk_classes(pred: dict, chunks: dict, classes: dict, taus) -> dict:
    """The edge chunk match per class of chunk. `classes` = {video: 1 class per chunk of chunks[video], in its order}
    (`evaluation.predictions.chunk_classes`). Per class: its reliable units, the matched share (units inside the runs 
    that the match pairs / the units, pooled over videos, then the mean over taus) and the uncaptioned units (no caption 
    overlaps them, the rule of `uncaptioned_units`, so the classes sum to its count)."""
    out: dict[str, dict] = {}
    for vid, video_chunks in chunks.items():
        evs, items = pred.get(vid, []), _edge_items(video_chunks)
        units = [(c, u) for c, chunk in zip(classes[vid], video_chunks) for u in chunk]
        matched = [{i for _, gi in _edge_matches(evs, items, float(tau)) for i in items[gi][3]} for tau in taus]
        for i, (c, u) in enumerate(units):
            row = out.setdefault(c, {"units": 0, "matched": 0.0, "uncaptioned": 0})
            row["units"] += 1
            row["matched"] += sum(i in m for m in matched) / len(taus)
            row["uncaptioned"] += not any(e.end_s > u.start_s and e.start_s < u.end_s for e in evs)
    return {c: {
        "units": r["units"], "matched_share": r["matched"] / r["units"], "uncaptioned": r["uncaptioned"]
    } for c, r in out.items()}

# The error types of the edge chunk match, in the order of their rules (`edge_error_types`).
EDGE_CAPTION_TYPES = ("matched", "joined", "cut", "bridged", "offset", "phantom")
EDGE_UNIT_TYPES = ("matched", "joined", "cut", "bridged", "offset", "missed")

def edge_error_types(pred: dict, chunks: dict, taus, cover: float = 0.5) -> dict:
    """The error types of the edge chunk match (the one error table of `analyze.py --stage segmenter-errors` and `report.py`).
    A caption holds a unit when it covers at least `cover` of the unit (`holds`). An edge cuts a unit when it lies inside the unit
    and more than tau from both of the unit's edges, so it can match no run edge at tau. Per tau, each caption and each reliable
    unit takes the first type whose rule fits:
      captions: matched (it matches a run of 1 unit), joined (it matches a run of 2 or more units of 1 chunk), cut (an edge cuts a
        unit), bridged (it holds units of 2 or more chunks), offset (it holds units of 1 chunk, but an edge is more than tau from
        the run edges, or another caption took the match), phantom (it holds no unit and cuts none);
      units: matched and joined (inside a matched run of 1 unit, of 2 or more units), cut (a caption edge cuts it), bridged (a
        caption holds it and a unit of another chunk), offset (a caption holds it), missed (no caption holds it or cuts it;
        every unit of `uncaptioned_units` is missed).
    A share is a count of 1 video over its captions (units), averaged over the videos with captions (units), as `_table_prf`
    averages. So at each tau the matched and joined caption shares sum to the edge chunk precision, and the unit shares to its
    recall. `*_avg` is the mean over taus. Per tau, `cut_positions` holds each cutting edge as its place in the unit,
    (edge - unit start) / unit duration. Per video: `held`, the units each caption holds; `pairs`, each caption that holds
    exactly 1 unit that no other caption holds, as (caption, unit, edge offsets start - unit start and end - unit end, tIoU), with
    no tau; and per tau the type of each caption and of each unit, and `runs`, the pairs of the main score as (caption, the units
    of its run). A unit is its index in the video's units in chunk order (the ids of `_edge_items`)."""
    taus = [float(t) for t in taus]
    vids = sorted(v for v in set(pred) | set(chunks) if pred.get(v) or chunks.get(v))
    cap = np.zeros((len(vids), len(taus), len(EDGE_CAPTION_TYPES)))
    unit = np.zeros((len(vids), len(taus), len(EDGE_UNIT_TYPES)))
    has_c, has_u, positions, per_video = np.zeros(len(vids)), np.zeros(len(vids)), [[] for _ in taus], {}
    for i, v in enumerate(vids):
        evs, video_chunks = pred.get(v, []), chunks.get(v, [])
        units = [(c, u) for c, chunk in enumerate(video_chunks) for u in chunk]   # the unit ids of `_edge_items`
        items = _edge_items(video_chunks)
        held = [[k for k, (_, u) in enumerate(units) if holds(e, u, cover)] for e in evs]
        bridging = [len({units[k][0] for k in h}) > 1 for h in held]
        holders = [[pi for pi, h in enumerate(held) if k in h] for k in range(len(units))]
        has_c[i], has_u[i] = float(bool(evs)), float(bool(units))
        out = per_video[v] = {"held": [len(h) for h in held], "taus": [], "pairs": [
            (pi, h[0], e.start_s - units[h[0]][1].start_s, e.end_s - units[h[0]][1].end_s,
             temporal_iou(Segment(e.start_s, e.end_s), Segment(units[h[0]][1].start_s, units[h[0]][1].end_s)))
            for pi, (e, h) in enumerate(zip(evs, held)) if len(h) == 1 and holders[h[0]] == [pi]]}
        for j, tau in enumerate(taus):
            pairs = _edge_matches(evs, items, tau)   # the pairs of the main score
            run = {pi: len(items[gi][3]) for pi, gi in pairs}
            in_run = {k: len(items[gi][3]) for _, gi in pairs for k in items[gi][3]}
            cuts = [[(k, (t - u.start_s) / (u.end_s - u.start_s)) for t in (e.start_s, e.end_s)
                     for k, (_, u) in enumerate(units) if u.start_s + tau < t < u.end_s - tau] for e in evs]
            positions[j] += [x for c in cuts for _, x in c]
            cut_units = {k for c in cuts for k, _ in c}
            types_c = [("matched" if run[pi] == 1 else "joined") if pi in run else "cut" if cuts[pi] else
                       ("bridged" if bridging[pi] else "offset") if held[pi] else "phantom" for pi in range(len(evs))]
            types_u = [("matched" if in_run[k] == 1 else "joined") if k in in_run else "cut" if k in cut_units else
                       "bridged" if any(bridging[pi] for pi in holders[k]) else "offset" if holders[k] else "missed"
                       for k in range(len(units))]
            for x, name in enumerate(EDGE_CAPTION_TYPES): cap[i, j, x] = types_c.count(name) / len(evs) if evs else 0.0
            for x, name in enumerate(EDGE_UNIT_TYPES): unit[i, j, x] = types_u.count(name) / len(units) if units else 0.0
            out["taus"].append({"captions": types_c, "units": types_u, "runs": [(pi, list(items[gi][3])) for pi, gi in pairs]})
    mean_over = lambda t, has: (t * has[:, None, None]).sum(0) / max(has.sum(), 1.0)
    c_share, u_share = mean_over(cap, has_c), mean_over(unit, has_u)
    count = lambda name, side, j: sum(r["taus"][j][side].count(name) for r in per_video.values())
    return {
        "taus": [{"tau_s": tau, "captions": dict(zip(EDGE_CAPTION_TYPES, map(float, c_share[j]))),
                  "units": dict(zip(EDGE_UNIT_TYPES, map(float, u_share[j]))),
                  "caption_counts": {name: count(name, "captions", j) for name in EDGE_CAPTION_TYPES},
                  "unit_counts": {name: count(name, "units", j) for name in EDGE_UNIT_TYPES},
                  "cut_positions": positions[j]} for j, tau in enumerate(taus)],
        "captions_avg": dict(zip(EDGE_CAPTION_TYPES, map(float, c_share.mean(0)))) if taus else {},
        "units_avg": dict(zip(EDGE_UNIT_TYPES, map(float, u_share.mean(0)))) if taus else {},
        "captions": int(sum(len(pred.get(v, [])) for v in vids)), "units": int(sum(sum(map(len, chunks.get(v, []))) for v in vids)),
        "per_video": per_video,
    }


def cap_cut_captions(pred: dict, chunks: dict, gold: dict, taus, char_level: bool, hours: float, cover: float = 0.5) -> dict:
    """The scored captions flagged `cap_cut` (closed part of an open span that the cap (inference.yaml buffer_cap_s) cut, the declared 
    exception to P1 at inference), apart from the rest: their count, the count per hour of `hours` (the pose time of scored streams), 
    reliable units they hold (at least `cover` of each), per tau the cap-cut captions that the edge chunk match pairs (matched among 
    all captions, as the main score pairs them) and units of their runs, and `without` / `uncaptioned_without`: `edge_chunk_scores` 
    and `uncaptioned_units` of `pred` with these captions removed (None when there are none: scores of all captions then hold)."""
    vids = sorted(v for v in set(pred) | set(chunks) if pred.get(v) or chunks.get(v))
    _, matches = edge_chunk_table(pred, chunks, taus, vids)
    cut = {v: [i for i, e in enumerate(pred.get(v, [])) if e.cap_cut] for v in vids}
    captions = sum(map(len, cut.values()))
    rest = {v: [e for e in evs if not e.cap_cut] for v, evs in pred.items()}
    return {"captions": captions, "per_hour": captions / hours,
            "units_held": sum(holds(pred[v][i], g, cover) for v in vids for i in cut[v] for g in gold.get(v, [])),
            "taus": [{
                "tau_s": float(tau), "matched": sum(i in matches[j][v] for v in vids for i in cut[v]),
                "matched_units": sum(matches[j][v][i][1] for v in vids for i in cut[v] if i in matches[j][v])
            } for j, tau in enumerate(taus)],
            "without": edge_chunk_scores(rest, chunks, taus, char_level) if captions else None,
            "uncaptioned_without": uncaptioned_units(rest, gold) if captions else None}

# ---------------------------------------------------------------- display replay scores (eval.py --online --stability)

# Rates rebuilt from summed counts when videos merge (numerator, denominator): exact, and a video with no committed track counts.
COUNT_RATES = {
    "phantom_reveal_rate": ("n_phantom_reveals", "n_tracks"), "normalized_erasure": ("n_erased_tokens", "n_final_tokens"), 
    "boundary_flip_rate": ("n_boundary_flips", "n_tracks"), "laal_s": ("laal_total_s", "n_laal_tracks")
}
SUMS = ("laal_total_s",)   # summed like the n_* counts, never averaged over n_tracks

def _units(tokens: list[int], separator_id: int) -> list[tuple[int, ...]]:
    units, unit = [], []
    for t in tokens:
        if t == separator_id: units.append(tuple(unit)); unit = []
        else: unit.append(t)
    return units + [tuple(unit)]

def _boundary_flip(previous: list[tuple[int, ...]], units: list[tuple[int, ...]]) -> bool:
    """A separator added or removed, or one moved: the units on BOTH sides of a separator changed. 
    A rewrite inside 1 unit changes only that unit, so it is an erasure, not a flip."""
    if len(previous) != len(units): return True
    return any(previous[j] != units[j] and previous[j + 1] != units[j + 1] for j in range(len(units) - 1))

def strip_separator(track: Track, separator_id: int | None) -> tuple[Track, int]:
    """(the track with separator removed from every hypothesis, its boundary flips): number of strides whose separators 
    changed from the previous stride's (`_boundary_flip`: 1 added or removed, or 1 moved). The separator is scored as a 
    space, as in dev BLEU, the boundary sweep and the rows, so a flip changes no displayed word; the flips are counted apart."""
    if separator_id is None: return track, 0
    hyps, flips, previous = [], 0, None
    for h in track.hyps:
        keep = h.token_ids != int(separator_id)
        units = _units([int(t) for t in h.token_ids.tolist()], int(separator_id))
        flips += int(previous is not None and _boundary_flip(previous, units))
        previous = units
        hyps.append(SimpleNamespace(
            commit_time_s=h.commit_time_s, span_start_s=h.span_start_s, committed=h.committed,
            token_ids=h.token_ids[keep], token_confidence=h.token_confidence[keep]
        ))
    return Track(span_start_s=track.span_start_s, hyps=hyps), flips

def erased_tokens(screens: list[tuple[float, list[int]]], committed: bool) -> int:
    # Tokens removed from screen from 1 stride to the next (Arivazhagan et al. 2020). 
    # Sentence that never commits leaves screen in full.
    erased, previous = 0, []
    for _, shown in screens:
        erased += len(previous) - lcp(previous, shown)
        previous = shown
    return erased + (0 if committed else len(previous))

def stabilization_times(screens: list[tuple[float, list[int]]], final: list[int]) -> list[float]:
    # Per final token: the time from which the screen shows it at its position on every later stride.
    times = [float("nan")] * len(final)
    for j, token in enumerate(final):
        for t, shown in screens:
            ok = j < len(shown) and shown[j] == token
            if ok and np.isnan(times[j]): times[j] = t
            elif not ok: times[j] = float("nan")
    return times

def laal(word_times_s: list[float], unit_start_s: float, unit_end_s: float, reference_words: int) -> float:
    """Length-adaptive average lagging (Papi et al. 2022) of 1 caption unit, in seconds. Word i appears at word_times_s[i]. 
    Delays count from unit's start. The source is the unit's own duration. The oracle rate uses the longer of the output 
    and the reference, so an over-long output is not rewarded. tau is the first word shown after the unit has ended."""
    delays = [float(t) - float(unit_start_s) for t in word_times_s]
    if not delays: return float("nan")
    duration = float(unit_end_s) - float(unit_start_s)
    rate = duration / max(len(delays), int(reference_words), 1)
    tau = next((i for i, d in enumerate(delays, 1) if d >= duration), len(delays))
    return float(sum(delays[i] - i * rate for i in range(tau)) / tau)

def score_policy(
    tracks: list[Track], policy, separator_id: int | None = None, units: dict | None = None, word_start=None, **kw
) -> dict:
    """Latency and prefix-correctness for one policy over all tracks.

    first_token_latency_s    when the FIRST token becomes visible, relative to the commit the FSM would have made: 
                             negative = the policy showed text EARLIER than commit_only (the point of the exercise).
    frozen_prefix_error      largest number of shown tokens that disagree with the sentence finally committed, on any screen 
                             before the commit, per most tokens shown before it — the price of revealing early. 0 for commit_only 
                             by construction. This is the FIRST-ORDER cost only: the traces come from unfrozen re-translation, 
                             so it cannot capture a frozen prefix degrading the continuation. That needs prefix-forced decoding.
    revealed_fraction        longest prefix of COMMITTED sentence that a screen showed before the commit, per its length, in [0, 1]. 
                             Wrong or surplus shown tokens add nothing here. frozen_prefix_error and vanished_track_rate charge them.
    contradicted_track_rate  fraction of sentences with >= 1 screen before the commit that disagrees with the committed sentence — 
                             the rewrite rate, and the decision number for prefix forcing. Text that disappears and returns unchanged 
                             is no contradiction; normalized_erasure counts it.
    phantom_reveal_rate      tracks that never commit (a target the FSM decoded on deferred strides, or an open span that vanished 
                             before its terminator appeared, and never emitted) on which the policy showed >= 1 token, per committed 
                             track: text on screen that system never emits. Such tracks add none to other fields, which describe 
                             committed sentences.
    normalized_erasure       tokens removed from screen (a changed shown token, a shorter hypothesis, a never-committed sentence 
                             leaving the screen) per committed final token, over all tracks (NE, Arivazhagan et al. 2020). 0 for 
                             commit_only: the deployed display is append-only.
    boundary_flip_rate       strides whose separators changed from previous stride's of same sentence (1 added, removed or moved), 
                             per committed track (the separator itself is dropped from every other field).
    laal_s                   length-adaptive average lagging in seconds, over committed tracks that `units` maps to gold units they 
                             hold ({track index: (start_s, end_s, reference words)}): word i appears when its tokens become stable 
                             on screen. `word_start` (token ids -> word-start flags) makes it word-level. W/o it each token is a word.
    """
    lat, err, revealed, contradicted, vanished = [], [], [], [], []
    phantoms = erased = final_tokens = flips = n_laal = 0
    laal_total = 0.0

    for index, tr in enumerate(tracks):
        if not tr.hyps: continue
        tr, track_flips = strip_separator(tr, separator_id)
        reveals = policy(tr, **kw)
        screens = displays(tr, reveals)
        erased += erased_tokens(screens, tr.committed)
        if not tr.committed:
            phantoms += int(any(shown for _, shown in screens))   # text that was on screen and is never emitted
            continue

        flips += track_flips   # a committed track counts under every policy, an empty sentence too, so all share 1 track set
        final_tokens += len(tr.final_tokens)
        unit = (units or {}).get(index)

        if unit is not None and tr.final_tokens:
            times = stabilization_times(screens, tr.final_tokens)
            starts = word_start(tr.final_tokens) if word_start is not None else [True] * len(times)
            words: list[float] = []
            for t, first in zip(times, starts):
                if first or not words: words.append(t)
                else: words[-1] = max(words[-1], t)   # a word is shown when its last token is stable
            laal_total += laal(words, *unit); n_laal += 1

        final = tr.final_tokens
        commit_t = tr.commit_time_s
        # Every screen BEFORE the commit, not only those at a reveal: an unforced re-translation can 
        # rewrite a shown token on a stride that reveals nothing new.
        before = [shown for t, shown in screens if t < commit_t]
        lat.append(next((t for t, shown in screens if shown), commit_t) - commit_t)  # 1st stride with text on screen
        k = max((len(shown) for shown in before), default=0)

        # The committed sentence's own prefix on screen (at most all of it): wrong or surplus shown tokens 
        # are text that must change or vanish, and frozen_prefix_error / vanished_track_rate charge them.
        revealed.append(max((lcp(shown, final) for shown in before), default=0) / max(1, len(final)))
        # Positions shown BEYOND the committed sentence are contradictions too: they were displayed and the final sentence does
        # not contain them (zip() alone would drop them for a hypothesis that shrinks between strides, as the DLM canvas does).
        wrong = max((
            sum(1 for a, b in zip(shown, final) if a != b) + max(0, len(shown) - len(final)) for shown in before
        ), default=0)
        err.append(wrong / max(1, k) if k else 0.0)
        contradicted.append(1.0 if wrong else 0.0)
        # Showing MORE tokens than the sentence finally has means displayed text later disappears — a distinct failure from a
        # token changing, and the signature of a reveal policy reading slots the decode never really scored (post-EOS padding
        # carries a fabricated confidence of 1.0). Should be ~0 on a trimmed trace.
        vanished.append(1.0 if k > len(final) else 0.0)
    return {
        "n_tracks": len(lat), "first_token_latency_s": float(np.mean(lat)) if lat else 0.0,
        "revealed_fraction": float(np.mean(revealed)) if revealed else 0.0, 
        "frozen_prefix_error": float(np.mean(err)) if err else 0.0,
        # Fraction of SENTENCES whose display would visibly rewrite at least once. This is the number that decides
        # whether prefix-forced decoding is worth building: near 0 means unforced re-translation is already stable.
        "contradicted_track_rate": float(np.mean(contradicted)) if contradicted else 0.0,
        "vanished_track_rate": float(np.mean(vanished)) if vanished else 0.0,
        "n_phantom_reveals": phantoms, "phantom_reveal_rate": phantoms / len(lat) if lat else 0.0,
        "n_erased_tokens": erased, "n_final_tokens": final_tokens, 
        "normalized_erasure": erased / final_tokens if final_tokens else 0.0,
        "n_boundary_flips": flips, "boundary_flip_rate": flips / len(lat) if lat else 0.0,
        "n_laal_tracks": n_laal, "laal_total_s": laal_total,
        "laal_s": laal_total / n_laal if n_laal else 0.0,   # read with n_laal_tracks, as every rate with its count
    }


def accumulate_policy(totals: dict, name: str, row: dict) -> None:
    """Fold 1 video's `score_policy` row into a running (sum, count) so the caller never holds every Track.

    The `n_*` fields are counts and add up. Every other field is an unweighted mean over that call's committed tracks, 
    so weighting by `n_tracks` and dividing at the end reproduces scoring the whole split at once exactly. COUNT_RATES 
    are rebuilt from their counts, so a video with phantoms but no committed track still counts.
    """
    acc = totals.setdefault(name, {})
    n = int(row["n_tracks"])
    for key, value in row.items():
        if key.startswith("n_"): acc[key] = acc.get(key, 0) + int(value)
        elif key in SUMS: acc[key] = acc.get(key, 0.0) + float(value)
        elif key not in COUNT_RATES: acc[key] = acc.get(key, 0.0) + float(value) * n


def merged_policy(totals: dict) -> dict:
    n = int(totals.get("n_tracks", 0))
    means = {k: (v / n if n else 0.0) for k, v in totals.items() if not k.startswith("n_") and k not in SUMS}
    counts = {k: (v if k in SUMS else int(v)) for k, v in totals.items() if k.startswith("n_") or k in SUMS}
    rates = {
        rate: (counts.get(num, 0) / counts[den] if counts.get(den) else 0.0) 
        for rate, (num, den) in COUNT_RATES.items()
    }
    return {**counts, **means, **rates}
