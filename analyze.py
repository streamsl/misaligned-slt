from __future__ import annotations
from dataclasses import asdict, dataclass
from statistics import median
from pathlib import Path
import json, argparse, math

import torch
import numpy as np
from tqdm import tqdm
from data.loader import load_language_records
from data.windowing import BIO, TRUSTED_GAP_S, make_bio_labels
from poses import load_pose_window

from models.checkpointing import load_model_checkpoint
from infer.duration_decode import duration_split_tags
from infer.commit_gate import first_terminator_index, select_target_span
from moryossef26.infer import _phrase_logits, _set_rope_chunk, duration_decode_params, fit_duration_prior, predict_phrase_segments
from eval import _build_eval_model, _load_segmenter, _translate_windows, load_prediction_file, save_prediction_file
from metrics import Segment, match_segments, moryossef_segment_metrics, compute_text_metrics
from utils import load_yaml, update_yaml_scalar, pick_device, checkpoint_dir, resolve_pretrained

# Analysis-A pred↔GT matching bar: deliberately LOW so near-misses count as matched pairs feeding the (Δ_head, Δ_tail) jitter CDF, 
# not as phantom/skip events. A high bar biases the DF toward near-zero offsets and miscalibrates the mode ratios. A corpus-level 
# analysis knob, not a segmenter setting, so it lives here; override per-run with --tiou-threshold.
ANALYSIS_A_MATCH_TIOU = 0.1

@dataclass(frozen=True)
class JitterSample:
    video_id: str
    pred_index: int
    gold_index: int
    tiou: float
    delta_head_s: float
    delta_tail_s: float

@dataclass(frozen=True)
class SegmenterErrorAnalysis:
    jitter_samples: list[JitterSample]
    mode_ratios: dict[str, float]
    event_counts: dict[str, int]
    matched_pairs: int
    regular_matches: int
    videos: int
    # Relative position in (0,1) of each spurious internal cut the segmenter placed inside an over-segmented GT sentence. 
    # This is what Mode 2 needs (where truncation lands), and it is NOT captured by the matched-pair (Δ_head, Δ_tail) CDF 
    # — those are one-to-one boundary noise. The Mode-2 window sampler draws its cut depth from this distribution; uniform 
    # is only a last-resort fallback when no over-segmentation was observed.
    overseg_cut_positions: list[float]

def _laplace_fit(values: list[float]) -> dict[str, float]:
    if not values: return {"loc": 0.0, "scale": 0.0}
    loc = float(median(values))
    scale = sum(abs(v - loc) for v in values) / max(1, len(values))
    return {"loc": loc, "scale": float(scale)}

def normalize_counts(counts: dict[str, int | float]) -> dict[str, float]:
    total = float(sum(max(0, v) for v in counts.values()))
    if total <= 0: return {"mode1": 1.0, "mode2": 0.0, "mode3": 0.0, "mode4": 0.0}
    return {key: float(max(0, value) / total) for key, value in counts.items()}

def mode_weights_from_events(counts: dict[str, int]) -> dict[str, float]:
    # Per streaming_slt_prompt.md §5.5: skip mass is split between truncated-window and multi-complete-window training cases.
    skipped_half = 0.5 * float(counts["skipped"])
    return {
        "mode1": float(counts["matched"]),
        "mode2": float(counts["oversegmentation"]) + skipped_half,
        "mode3": float(counts["undersegmentation"]) + skipped_half,
        "mode4": float(counts["phantom"]),
    }

def analyze_segmenter_errors(
    predicted: dict[str, list[Segment]], gold: dict[str, list[Segment]],
    durations: dict[str, float], tiou_threshold: float = 0.1,
) -> SegmenterErrorAnalysis:
    """Compute Analysis-A event counts and regular-match jitter samples.

    Boundary-jitter samples intentionally exclude over-segmented GT spans and under-segmenting predicted spans. 
    Those are separate window modes, not ordinary boundary noise around a one-to-one segment.
    """
    jitter_samples: list[JitterSample] = []
    overseg_cut_positions: list[float] = []
    counts = {"matched": 0, "oversegmentation": 0, "undersegmentation": 0, "skipped": 0, "phantom": 0}
    matched_pairs, regular_matches = 0, 0

    for video_id in sorted(set(gold) | set(predicted)):
        pred_segments = list(predicted.get(video_id, []))
        gold_segments = list(gold.get(video_id, []))
        matches = match_segments(pred_segments, gold_segments, threshold=tiou_threshold)
        matched_pairs += len(matches)
        matched_gold = {gold_idx for _, gold_idx, _ in matches}

        overlapping_pred_by_gold: dict[int, list[int]] = {}
        overlapping_gold_by_pred: dict[int, list[int]] = {}
        for pred_idx, pred in enumerate(pred_segments):
            for gold_idx, gt in enumerate(gold_segments):
                overlap = max(0.0, min(pred.end_s, gt.end_s) - max(pred.start_s, gt.start_s))
                if overlap > 0:
                    overlapping_pred_by_gold.setdefault(gold_idx, []).append(pred_idx)
                    overlapping_gold_by_pred.setdefault(pred_idx, []).append(gold_idx)

        overseg_gold = {gold_idx for gold_idx, pred_indices in overlapping_pred_by_gold.items() if len(pred_indices) >= 2}
        underseg_pred = {pred_idx for pred_idx, gold_indices in overlapping_gold_by_pred.items() if len(gold_indices) >= 2}
        underseg_gold = {gold_idx for pred_idx in underseg_pred for gold_idx in overlapping_gold_by_pred.get(pred_idx, [])}
        phantom_pred = {
            pred_idx for pred_idx, pred in enumerate(pred_segments) if pred_idx not in overlapping_gold_by_pred
            and pred.start_s >= 0.0 and pred.end_s <= float(durations.get(video_id, pred.end_s))
        }
        counts["oversegmentation"] += len(overseg_gold)
        counts["undersegmentation"] += len(underseg_pred)
        counts["phantom"] += len(phantom_pred)
        counts["skipped"] += len(set(range(len(gold_segments))) - matched_gold - overseg_gold - underseg_gold)

        # Record where each spurious internal cut fell, relative to the over-segmented GT span.
        for gold_idx in overseg_gold:
            gt = gold_segments[gold_idx]
            dur = gt.end_s - gt.start_s
            if dur <= 0: continue
            
            overlapping = sorted((pred_segments[pi] for pi in overlapping_pred_by_gold[gold_idx]), key=lambda s: s.start_s)
            for left, right in zip(overlapping, overlapping[1:]):
                cut = min(max(0.5 * (left.end_s + right.start_s), gt.start_s), gt.end_s)
                rel = (cut - gt.start_s) / dur
                if 0.0 < rel < 1.0: overseg_cut_positions.append(float(rel))

        for pred_idx, gold_idx, score in matches:
            if gold_idx in overseg_gold or pred_idx in underseg_pred: continue
            pred, gt = pred_segments[pred_idx], gold_segments[gold_idx]
            counts["matched"] += 1
            regular_matches += 1
            jitter_samples.append(JitterSample(
                video_id=video_id, pred_index=pred_idx, gold_index=gold_idx, tiou=float(score),
                delta_head_s=float(pred.start_s - gt.start_s), delta_tail_s=float(pred.end_s - gt.end_s),
            ))
    mode_counts = mode_weights_from_events(counts)
    return SegmenterErrorAnalysis(
        jitter_samples=jitter_samples, mode_ratios=normalize_counts(mode_counts), event_counts=counts,
        matched_pairs=matched_pairs, regular_matches=regular_matches, videos=len(set(gold) | set(predicted)),
        overseg_cut_positions=overseg_cut_positions,
    )


def write_analysis_a_outputs(analysis: SegmenterErrorAnalysis, output_dir: str | Path, language: str) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jitter_rows = [asdict(sample) for sample in analysis.jitter_samples]
    head = [row["delta_head_s"] for row in jitter_rows]
    tail = [row["delta_tail_s"] for row in jitter_rows]
    jitter_payload = {
        "language": language, "samples": jitter_rows,
        "laplace": {"head": _laplace_fit(head), "tail": _laplace_fit(tail)},
        # Mode-2 truncation-depth distribution (relative cut positions of over-segmentation events).
        # The window sampler reads this; if empty it falls back to a uniform interior cut.
        "overseg_cut_positions": analysis.overseg_cut_positions,
    }
    mode_payload = {
        "language": language,
        "mode_ratios": analysis.mode_ratios,
        "source_event_counts": analysis.event_counts,
        "source_weights": mode_weights_from_events(analysis.event_counts),
    }
    taxonomy_payload = {
        "language": language,
        "event_counts": analysis.event_counts,
        "matched_pairs": analysis.matched_pairs,
        "regular_matches": analysis.regular_matches,
        "videos": analysis.videos,
        "moryossef_2020_mapping": {
            "matched": "Started Pre/Post-Signing and Signing Underflow/Overflow",
            "oversegmentation": "Signing Undetected Incorrectly",
            "undersegmentation": "Bridged",
            "skipped": "Skipped",
            "phantom": "Signing Detected Incorrectly",
        },
    }
    paths = {
        "jitter": str(output_dir / f"a_jitter_{language}.json"),
        "mode_ratios": str(output_dir / f"a_mode_ratios_{language}.json"),
        "taxonomy": str(output_dir / f"a_error_taxonomy_{language}.json"),
    }
    Path(paths["jitter"]).write_text(json.dumps(jitter_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["mode_ratios"]).write_text(json.dumps(mode_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["taxonomy"]).write_text(json.dumps(taxonomy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def dataset_summary(args: argparse.Namespace) -> dict:
    cfg = load_yaml(args.data_config)
    records, splits = load_language_records(cfg, args.language, split=args.split)
    durations = [span.duration_s for rec in records for span in rec.sentences]
    # Λ_min derivation (inference.yaml span_selection): p1/2 of dev sentence durations × median fps — an order of
    # magnitude above the ≤2δ phantom scale with full margin to the shortest real sentence. Per-corpus, per-fps:
    # set inference.yaml min_span_frames (+ the dlm.yaml membership_gate mirror) from this when switching corpus.
    p1_s = float(np.percentile(durations, 1)) if durations else 0.0
    median_fps = float(np.median([float(rec.pose.fps) for rec in records])) if records else 0.0
    return {
        "language": args.language, "split": args.split or "all", "records": len(records),
        "sentences": len(durations), "split_sizes": {k: len(v) for k, v in splits.items()},
        "mean_sentence_s": sum(durations) / len(durations) if durations else 0.0,
        "max_sentence_s": max(durations) if durations else 0.0,
        "p1_sentence_s": p1_s, "median_fps": median_fps,
        "suggested_min_span_frames": int(math.ceil(p1_s / 2.0 * median_fps)) if durations else 0,
    }


def segmenter_infer(args: argparse.Namespace) -> dict:
    # Predicted phrase segments on a split — the upstream segmenter for Analysis A / Analysis B / the RQ2 cascade.
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run segmenter inference on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    model, device, velocity, rope_chunk_s, checkpoint = _load_segmenter(args)
    # Same per-arch decode defaults as eval --segmenter-eval: s1 -> our semi-Markov duration re-split (infer/duration_decode.py), external 
    # -> faithful Moryossef argmax. NB for Analysis A/B as a MEASUREMENT instrument you may want `--segmenter-decode duration` even with 
    # the external arch (measure the deployed decode's error modes, not the published baseline's); the flag exists for exactly that.
    dd = duration_decode_params(load_yaml(args.inference_config), args.language)  # per-language switch = source of truth
    decode = args.segmenter_decode or ("duration" if (args.segmenter_arch == "s1" and dd is not None) else "plain")
    duration_prior = None
    if decode == "duration":
        train_records, _ = load_language_records(data_cfg, args.language, split="train")
        duration_prior = fit_duration_prior(train_records, **(dd or {}))
    print(f"[segmenter-infer] {args.segmenter_arch} segmenter from {checkpoint} "
          f"(decode={'duration' if duration_prior else 'plain'})", flush=True)

    predictions = predict_phrase_segments(
        model, records, device=device, velocity=velocity, rope_chunk_s=rope_chunk_s, duration_prior=duration_prior,
    )
    output = Path(args.output or f"outputs/segmenter_predictions_{args.segmenter_arch}_{args.language}_{args.split}.json")
    save_prediction_file(predictions, output)
    return {
        "language": args.language, "split": args.split, "videos": len(records),
        "segmenter_arch": args.segmenter_arch, "checkpoint": checkpoint,
        "predicted_segments": sum(len(v) for v in predictions.values()), "output": str(output),
    }


def tune_decode(args: argparse.Namespace) -> dict:
    """Two-fold dev tuning of the semi-Markov decode pair (split_bias, snap_radius_s) — infer/duration_decode.py.

    F1(split_bias) is a knife edge, not a plateau (the bias is the Lagrange multiplier of a segment-count constraint against a 
    near-constant per-segment normalisation cost, so the DP's split count responds almost step-wise), and the two knobs interact 
    (a larger snap radius partially rescues an oversplitting bias — a marginal sweep at the wrong bias misleads about the radius). 
    Hence: joint FINE grid, predictions computed ONCE per video, decode-only re-sweep, per-fold macro F1@0.5, and a fold-consistent 
    selection (max of min(foldA, foldB)). Pin the selected pair in inference.yaml `duration_decode: {split_bias, snap_radius_s}` —
    a deliberate human step, never auto-applied, so eval runs stay tuning-free.
    """
    if args.split == "test": raise SystemExit("tune-decode is dev-only: tuning on test is test contamination")
    data_cfg = load_yaml(args.data_config)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    train_records, _ = load_language_records(data_cfg, args.language, split="train")
    prior = fit_duration_prior(train_records)
    if prior is None: raise SystemExit("too few train captions to fit a duration prior")
    model, device, velocity, rope_chunk_s, checkpoint = _load_segmenter(args)
    model.eval().to(device)
    print(f"[tune-decode] {args.segmenter_arch} segmenter from {checkpoint}; one forward per video, then decode-only sweep", flush=True)

    cached = []  # (video_id, tags, pB, gold, fps) — predictions computed once; the sweep is decode-only
    with torch.no_grad():
        for rec in records:
            poses, timestamps = load_pose_window(rec.pose, 0.0, rec.pose.duration_s, normalize=True)
            if poses.shape[0] == 0: continue
            _set_rope_chunk(model, rec, rope_chunk_s)
            logits = _phrase_logits(model, poses, timestamps, device, velocity)[0].float().cpu()
            gold = torch.as_tensor(np.asarray(make_bio_labels(
                timestamps, rec.sentences, 0.0, float(rec.pose.duration_s),
                trusted_gap_s=TRUSTED_GAP_S, video_duration_s=rec.pose.duration_s
            ))).long()
            pB = torch.softmax(logits, dim=-1)[:, BIO["B"]].numpy()  # snap evidence; matches duration_decode_tags
            cached.append((rec.video_id, logits.argmax(-1).numpy(), pB, gold, float(rec.pose.fps)))
    cached.sort(key=lambda c: c[0])
    folds = (cached[::2], cached[1::2])

    def fold_f1(fold, bias, radius):
        f1s = []
        for _, tags, pB, gold, fps in fold:
            t = duration_split_tags(tags, pB, fps, prior, split_bias=bias, snap_radius_s=radius)
            oh = torch.nn.functional.one_hot(torch.as_tensor(t).long(), num_classes=4).float().unsqueeze(0)
            f1s.append(moryossef_segment_metrics(oh, gold.unsqueeze(0), prefix="p", tiou_threshold=0.5)["p_tiou_f1"])
        return float(np.mean(f1s)) if f1s else 0.0

    grid_bias = [round(float(b), 2) for b in np.arange(2.5, 6.01, 0.25)]
    grid_radius = [0.0, 0.5, 1.0, 1.5]
    rows, best = [], None
    for bias in grid_bias:
        for radius in grid_radius:
            a, b = fold_f1(folds[0], bias, radius), fold_f1(folds[1], bias, radius)
            rows.append({"split_bias": bias, "snap_radius_s": radius, "foldA_f1@0.5": round(a, 4), "foldB_f1@0.5": round(b, 4)})
            key = (min(a, b), (a + b) / 2)  # fold-consistent first, mean as tie-break
            if best is None or key > best[0]: best = (key, rows[-1])
        print(f"[tune-decode] bias={bias:4.2f}: " + " ".join(
            f"r={r['snap_radius_s']:.1f}:{r['foldA_f1@0.5']:.3f}/{r['foldB_f1@0.5']:.3f}" for r in rows[-len(grid_radius):]
        ), flush=True)

    selected = dict(best[1])
    # Held-out estimate — the number to QUOTE: re-select using each fold alone, evaluate that pair on the other
    # fold, average. The selected pair's own cells are in-selection maxima and overstate the gain.
    heldout = []
    for sel_i, eval_i in ((0, 1), (1, 0)):
        key_col = f"fold{'AB'[sel_i]}_f1@0.5"
        by_sel = max(rows, key=lambda r: r[key_col])
        heldout.append(by_sel[f"fold{'AB'[eval_i]}_f1@0.5"])
        
    heldout_f1 = round(sum(heldout) / 2, 4)
    output = Path(args.output or f"outputs/tune_decode_{args.segmenter_arch}_{args.language}_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "language": args.language, "split": args.split, "segmenter_arch": args.segmenter_arch, "checkpoint": checkpoint,
        "videos": len(cached), "prior": {"mu_log_s": prior.mu_log_s, "sd_log_s": prior.sd_log_s, "cap_s": prior.cap_s},
        "selected": selected, "heldout_f1@0.5": heldout_f1, "heldout_per_fold": [round(h, 4) for h in heldout], "grid": rows,
        "pin_as": {"duration_decode": {"split_bias": selected["split_bias"], "snap_radius_s": selected["snap_radius_s"]}},
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[tune-decode] selected split_bias={selected['split_bias']} snap_radius_s={selected['snap_radius_s']} "
          f"(in-selection F1@0.5 {selected['foldA_f1@0.5']}/{selected['foldB_f1@0.5']}; HELD-OUT estimate {heldout_f1} — "
          f"quote the held-out number); pin the pair in inference.yaml duration_decode", flush=True)
    payload_out = dict(payload); payload_out["output"] = str(output)
    payload_out.pop("grid")  # the full grid lives in the JSON; keep the stage's stdout summary short
    return payload_out


def analysis_a(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test: raise SystemExit("Analysis A must run on dev; --allow-test only for smoke debugging")
    cfg = load_yaml(args.data_config)
    records, _ = load_language_records(cfg, args.language, split=args.split)
    predictions = load_prediction_file(args.predictions)  # the segmenter-infer output file
    gold_segments = {record.video_id: [Segment(span.start_s, span.end_s) for span in record.sentences] for record in records}
    durations = {record.video_id: float(record.pose.duration_s) for record in records}
    analysis = analyze_segmenter_errors(
        predicted=predictions, gold=gold_segments, durations=durations,
        tiou_threshold=float(args.tiou_threshold if args.tiou_threshold is not None else ANALYSIS_A_MATCH_TIOU),
    )
    paths = write_analysis_a_outputs(analysis, args.output_dir, args.language)
    return {
        "language": args.language, "split": args.split, "event_counts": analysis.event_counts, "mode_ratios": analysis.mode_ratios, 
        "matched_pairs": analysis.matched_pairs, "regular_matches": analysis.regular_matches, "outputs": paths,
    }


def _eval_model_for(method: str, args: argparse.Namespace, method_cfg: dict, device: torch.device):
    # Reuse eval.py's checkpoint-loading model builder (plain scalars — no fake namespace).
    data_cfg = load_yaml(args.data_config)
    return _build_eval_model(method, args.checkpoint, args.language, data_cfg, method_cfg, device)


def analysis_b(args: argparse.Namespace) -> dict:
    """Analysis B — the paper's MOTIVATING experiment: a CLEAN SLT model degrades under a REALISTIC upstream segmenter's boundary errors. 
    Runs on dev, before SLT training, and is method-INDEPENDENT (clean baseline is not our method). Produce clean-vs-realistic table.

    Realistic point: translate the windows the EXTERNAL segmenter actually cut (`--predictions`, the segmenter-infer output), with each 
    window's reference = the GT sentence it most overlaps. Clean point: translate the GT-trimmed windows. The controlled severity CURVE 
    is the SAME machinery as `eval.py --rq 1 --method baseline --split dev` — run that for the curve; this stage assembles realistic gap.
    """
    if args.split == "test" and not args.allow_test: raise SystemExit("Analysis B runs on dev; --allow-test only for smoke debugging")
    if not args.predictions: raise SystemExit("--predictions required: Moryossef segmenter's spans (analyze.py --stage segmenter-infer)")
    data_cfg = load_yaml(args.data_config)
    base_cfg = load_yaml(args.baseline_config, language=args.language)  # re-point ${language} paths like every other load
    inference_cfg = load_yaml(args.inference_config)
    device = pick_device(args.device)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    model, tokenizer = _eval_model_for("baseline", args, base_cfg, device)

    def _translate_all(spans_with_refs, desc):
        # Load (I/O-bound, tqdm'd) then decode in --batch-size chunks — the one-window-per-model-call version of
        # this stage was wall-clock-dominated by per-window beam decodes and per-window pose-file opens.
        items, refs = [], []
        for rec, start_s, end_s, ref in tqdm(spans_with_refs, desc=f"{desc}: load"):
            poses, ts = load_pose_window(rec.pose, start_s, end_s, normalize=True)
            if poses.shape[0] == 0: continue
            items.append((poses, ts, start_s)); refs.append(ref)
        preds = [t for t, _, _ in _translate_windows(
            model, tokenizer, "baseline", items, device, inference_cfg, base_cfg, batch_size=int(args.batch_size)
        )]
        return preds, refs

    clean_pred, clean_ref = _translate_all( # Clean point: GT-trimmed windows
        [(rec, float(s.start_s), float(s.end_s), s.text) for rec in records for s in rec.sentences], "Analysis B: clean (GT spans)"
    )

    # Realistic point: the external segmenter's predicted spans; reference = the max-overlap GT sentence.
    predicted = load_prediction_file(args.predictions)
    by_id = {r.video_id: r for r in records}
    real_spans, covered, phantoms = [], set(), 0
    for vid, spans in predicted.items():
        rec = by_id.get(vid)
        if rec is None: continue
        for span in spans:
            best, best_ov = None, 0.0
            for gt in rec.sentences:
                ov = max(0.0, min(span.end_s, gt.end_s) - max(span.start_s, gt.start_s))
                if ov > best_ov: best_ov, best = ov, gt
            if best is None:
                phantoms += 1; continue  # phantom span in a gap → no GT reference; excluded from the corpus score
            covered.add((vid, float(best.start_s)))
            real_spans.append((rec, float(span.start_s), float(span.end_s), best.text))

    real_pred, real_ref = _translate_all(real_spans, "Analysis B: realistic (segmenter spans)")
    clean = compute_text_metrics(clean_pred, clean_ref, prefix="clean") if clean_pred else {}
    realistic = compute_text_metrics(real_pred, real_ref, prefix="realistic") if real_pred else {}
    n_gold = sum(len(r.sentences) for r in records)
    payload = {
        "language": args.language, "split": args.split, "clean": clean, "realistic": realistic,
        "clean_windows": len(clean_pred), "realistic_windows": len(real_pred),
        "delta_bleu4": float(clean.get("clean_bleu4", 0.0) - realistic.get("realistic_bleu4", 0.0)),
        # Accounting visibility: the realistic point scores MATCHED windows only — GT sentences no window overlaps
        # cost nothing, phantom windows cost nothing, and several windows may share one reference. These counters
        # keep that visible (report gold_coverage next to delta_bleu4; a low coverage means the gap understates
        # the true segmentation damage). Kept matched-only deliberately: the recall-inclusive accounting lives in
        # RQ2's event scoring — Analysis B is the per-window conditional quality gap, not the system recall story.
        "gold_sentences_total": n_gold, "gold_sentences_covered": len(covered), "gold_coverage": round(len(covered) / max(n_gold, 1), 4),
        "phantom_windows_excluded": phantoms, "duplicate_reference_windows": len(real_pred) - len(covered),
        "note": "Controlled severity curve = eval.py --rq 1 --method baseline --split dev (no-robustness floor).",
    }
    output = Path(args.output or f"outputs/analysis_b_{args.language}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["output"] = str(output)
    return payload


def tail_benefit(args: argparse.Namespace) -> dict:
    """Tail-benefit curve sets BUFFER_CAP_S.

    Protocol: clean-trained translator (Analysis-B clean baseline), dev split, head fixed at true sentence start (Δ_head = 0), trailing 
    context Δ_tail swept upward; BLEU-4 per Δ_tail. The elbow is 1st grid point whose marginal BLEU per extra second drops below explicit 
    latency/quality coefficient (eval.yaml tail_benefit.latency_quality_coeff_bleu_per_s) — not hand-picked number or %-of-clean threshold.

    Buffer-cap semantics (spec ambiguity resolved, documented): the spec says "this elbow is the buffer cap", but the FSM buffer must hold 
    the WHOLE sentence plus the trailing context, so a 1–3 s tail elbow alone cannot be the cap. We persist buffer_cap_s = p99 sentence 
    duration + elbow (both raw values are in the JSON artifact for the paper).
    """
    if args.split == "test" and not args.allow_test: raise SystemExit("Tail-benefit runs on dev; --allow-test only for smoke debugging")
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    base_cfg = load_yaml(args.baseline_config, language=args.language)  # re-point ${language} paths like every other load
    inference_cfg = load_yaml(args.inference_config)
    tb_cfg = eval_cfg.get("tail_benefit", {})
    grid = [float(x) for x in tb_cfg.get("tail_grid_s", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])]
    coeff = float(tb_cfg.get("latency_quality_coeff_bleu_per_s", 0.5))

    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    device = pick_device(args.device)
    model, tokenizer = _eval_model_for("baseline", args, base_cfg, device)

    durations = [span.duration_s for rec in records for span in rec.sentences]
    p99_duration = float(np.percentile(durations, 99)) if durations else 0.0
    sentences = [(rec, span) for rec in records for span in rec.sentences]
    if args.num_sentences: sentences = sentences[: int(args.num_sentences)]

    # ONE pose read per sentence at the LONGEST tail, sliced per grid point in memory (normalization is
    # per-frame, so a slice of the normalized long window == the short window loaded directly), then decode in
    # --batch-size chunks. The previous shape — len(grid) separate pose-file reads AND len(grid) single-window
    # beam decodes per sentence — was wall-clock-dominated by both (hours on a network filesystem).
    max_tail = max(grid)
    per_tail: dict[float, dict[str, list]] = {dt: {"items": [], "references": [], "clamped": 0} for dt in grid}
    for rec, span in tqdm(sentences, desc="tail-benefit: load"):
        poses, timestamps = load_pose_window(rec.pose, span.start_s, min(rec.pose.duration_s, span.end_s + max_tail), normalize=True)
        if poses.shape[0] == 0: continue
        for dt in grid:
            end_s = min(rec.pose.duration_s, span.end_s + dt)
            n = int(np.searchsorted(timestamps, end_s, side="left"))
            if n == 0: continue
            per_tail[dt]["items"].append((poses[:n], timestamps[:n], float(span.start_s)))
            per_tail[dt]["references"].append(span.text)
            # Track end-of-video clamping: a clamped row got LESS trailing context than this grid point claims,
            # so a flat marginal-BLEU step at large dt can be "no more benefit" OR "no more video" — without this
            # fraction the elbow (and the persisted buffer_cap_s) can be an artifact of the corpus's tail margins.
            if end_s < span.end_s + dt - 1e-6: per_tail[dt]["clamped"] += 1

    curve = []
    for dt in grid:
        preds = [t for t, _, _ in _translate_windows(
            model, tokenizer, "baseline", per_tail[dt]["items"], device, inference_cfg, base_cfg, batch_size=int(args.batch_size)
        )]
        bleu = compute_text_metrics(preds, per_tail[dt]["references"])["translation_bleu4"]
        n = len(preds)
        clamped_fraction = per_tail[dt]["clamped"] / max(n, 1)
        curve.append({"delta_tail_s": dt, "bleu4": float(bleu), "n": n, "clamped_fraction": round(clamped_fraction, 4)})
        print(f"[tail-benefit] tail={dt:.1f}s BLEU4={bleu:.2f} (n={n}, clamped={clamped_fraction:.1%})", flush=True)

    heavy = [c for c in curve if c["clamped_fraction"] > 0.2]
    if heavy: print(f"[tail-benefit] WARNING: {len(heavy)} grid point(s) have >20% end-of-video clamping "
                    f"(from tail={heavy[0]['delta_tail_s']}s); the elbow may reflect missing video, not saturated benefit.", flush=True)

    elbow = grid[-1]
    for prev, cur in zip(curve, curve[1:]):
        gap_s = cur["delta_tail_s"] - prev["delta_tail_s"]
        marginal = (cur["bleu4"] - prev["bleu4"]) / gap_s if gap_s > 0 else 0.0
        if marginal < coeff:
            elbow = prev["delta_tail_s"]
            break

    buffer_cap_s = round(p99_duration + elbow, 2)
    payload = {
        "language": args.language, "split": args.split, "sentences": len(sentences),
        "latency_quality_coeff_bleu_per_s": coeff, "curve": curve,
        "elbow_tail_s": elbow, "p99_sentence_duration_s": p99_duration, "buffer_cap_s": buffer_cap_s,
    }
    output = Path(args.output or f"outputs/tail_benefit_{args.language}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_config and update_yaml_scalar(args.inference_config, ("buffer_cap_s",), buffer_cap_s):
        payload["config_updated"] = args.inference_config
    payload["output"] = str(output)
    return payload


def delta_enc(args: argparse.Namespace) -> dict:
    """BIO temporal noise floor sets the commit gate's delta_enc.

    Uses the S1 in-system BIO head (the DEPLOYED terminator estimator the FSM runs), NOT Moryossef analysis segmenter and NOT DLM checkpoint. 
    2 reasons it must be S1 head: (1) δ_enc calibrates commit gate's cut overlap against the noise of the head the FSM actually contains — 
    analysis segmenter never runs in FSM; (2) δ_enc is a gate-GEOMETRY constant needed to TRAIN the DLM (dlm.yaml δ is asserted == this), so 
    it must be measured from the head as it ENTERS stage 2 (checkpoints/bio_s1, produced by train-bio), before the DLM exists — reading it 
    off the DLM checkpoint is an ordering circularity.

    Runs that head twice per dev sentence window under small input perturbations and measures how far the predicted terminator index 
    (first_terminator_index — first O-or-B, the statistic the commit gate tracks) moves. 2 perturbation families: (a) drop the leading frame 
    — the stride-phase misalignment a growing buffer produces; (b) Gaussian keypoint noise sigma on x/y — pose-estimator jitter. delta_enc 
    = ceil(p90 over both families): boundary movement below the head's own noise floor must not block a commit.
    """
    if args.split == "test" and not args.allow_test: raise SystemExit("Delta-enc runs on dev; --allow-test only for smoke debugging")
    from train.bio_pretrain import build_bio_s1_model
    data_cfg = load_yaml(args.data_config)
    cfg = load_yaml(args.bio_config, language=args.language)  # ${language} in checkpoint.dir -> the right bio_s1 dir
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    device = pick_device(args.device)

    pretrained = resolve_pretrained(cfg, data_cfg, args.language, default="checkpoints/csl_daily_pose_only_slt.pth")
    model = build_bio_s1_model(cfg, pretrained_path=pretrained)
    checkpoint = args.checkpoint or str(Path(checkpoint_dir(cfg, default=f"checkpoints/bio_s1/{args.language}")) / "model.pt")
    load_model_checkpoint(model, checkpoint, strict=True)
    model.eval().to(device)
    print(f"[delta-enc] S1 BIO head from {checkpoint}", flush=True)
    sigma = float(args.noise_sigma)

    # δ_enc must be measured under the DEPLOYED terminator decode. When inference.yaml's duration_decode is on, the FSM duration-re-splits 
    # every buffer BEFORE reading terminators (infer/stream.py step()), so the raw argmax statistic is the wrong instrument: on back-to-back 
    # corpora the first argmax O-or-B jumps by WHOLE SENTENCES under a one-frame perturbation (the b2b terminator simply is not in the argmax 
    # tags), and p90 of that instability is the argmax decoder's pathology, not the head's noise floor, which forced min_span_frames past the 
    # buffer cap and broke FSM span selection outright. Under the deployed decode the terminator moves by DP/snap jitter instead.
    dd = duration_decode_params(load_yaml(args.inference_config), args.language)
    duration_prior = None
    if dd is not None:
        train_records, _ = load_language_records(data_cfg, args.language, split="train")
        duration_prior = fit_duration_prior(train_records, **dd)
    # The statistic must be the terminator the deployed commit gate TRACKS: the terminator of the SELECTED span (select_target_span with the 
    # deployed Λ_min), not first_terminator_index over raw tags — the latter includes terminators of phantom micro-spans (1-frame flickers) 
    # that Λ_min filters at deployment, so its perturbation jitter calibrates δ on spans the gate can never commit.
    min_span = int((load_yaml(args.inference_config).get("span_selection", {}) or {}).get("min_span_frames", 0))
    print(f"[delta-enc] terminator decode: {'duration (deployed)' if duration_prior else 'plain argmax'}; "
          f"Lambda_min={min_span} frames", flush=True)

    sentences = [(rec, span) for rec in records for span in rec.sentences]
    if args.num_sentences: sentences = sentences[: int(args.num_sentences)]
    rng = np.random.default_rng(int(args.seed))

    @torch.no_grad()
    def closing_indices(variants: list[tuple[np.ndarray, np.ndarray]], start_s: float, fps: float) -> list[int | None]:
        # ONE batched forward for all perturbation variants of a sentence (they differ by <=1 frame in length;
        # right-pad by repeating the last frame with frame_mask=False — the training collator's pad contract).
        T = max(p.shape[0] for p, _ in variants)
        poses = torch.stack([torch.cat([
            torch.as_tensor(p, dtype=torch.float32), torch.as_tensor(p[-1:], dtype=torch.float32).expand(T - p.shape[0], -1, -1)
        ]) for p, _ in variants]).to(device)
        ts = torch.stack([torch.nn.functional.pad(
            torch.as_tensor(t - start_s, dtype=torch.float32), (0, T - t.shape[0])
        ) for _, t in variants]).to(device)

        mask = torch.stack([torch.arange(T) < p.shape[0] for p, _ in variants]).to(device)
        logits = model(poses, mask, timestamps_s=ts).logits
        out: list[int | None] = []

        for i, (p, _) in enumerate(variants):
            n = int(p.shape[0])
            tags = logits[i, :n].argmax(dim=-1)
            if duration_prior is not None and n > 2:
                pB = torch.softmax(logits[i, :n].float(), dim=-1)[:, BIO["B"]].cpu().numpy()
                tags = torch.as_tensor(duration_split_tags(
                    tags.cpu().numpy(), pB, fps, duration_prior, mark_onsets=False, split_open_tail="survival"
                ))
            span = select_target_span(tags, min_span)
            out.append(int(span[1]) if span is not None else first_terminator_index(tags))
        return out

    shifts: dict[str, list[int]] = {"drop_first_frame": [], "keypoint_noise": []}
    for rec, span in tqdm(sentences, desc="delta-enc"):
        start_s = max(0.0, span.start_s - 1.0)
        end_s = min(rec.pose.duration_s, span.end_s + 1.0)
        poses, timestamps = load_pose_window(rec.pose, start_s, end_s, normalize=True)
        if poses.shape[0] < 3: continue
        fps = float(rec.pose.fps)
        noisy = poses.copy()
        noisy[..., :2] = noisy[..., :2] + rng.normal(0.0, sigma, size=noisy[..., :2].shape).astype(noisy.dtype)
        base, dropped, perturbed = closing_indices([(poses, timestamps), (poses[1:], timestamps[1:]), (noisy, timestamps)], start_s, fps)
        if base is None: continue
        # The dropped buffer's indices sit one frame earlier on the original timeline.
        if dropped is not None: shifts["drop_first_frame"].append(abs((dropped + 1) - base))
        if perturbed is not None: shifts["keypoint_noise"].append(abs(perturbed - base))

    stats = {family: {
        "n": len(values), "median": float(np.median(values)) if values else 0.0,
        "p90": float(np.percentile(values, 90)) if values else 0.0, "max": int(max(values)) if values else 0,
    } for family, values in shifts.items()}
    delta = int(math.ceil(max((s["p90"] for s in stats.values()), default=0.0))) or 1
    payload = {
        "language": args.language, "split": args.split, "noise_sigma": sigma,
        "sentences": len(sentences), "families": stats, "delta_enc_frames": delta,
    }
    output = Path(args.output or f"outputs/delta_enc_{args.language}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_config and update_yaml_scalar(args.inference_config, ("boundary_stability", "delta_enc_frames"), delta):
        payload["config_updated"] = args.inference_config
    payload["output"] = str(output)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Misaligned-SLT analysis utilities")
    parser.add_argument(
        "--stage", default="dataset-summary",
        choices=["dataset-summary", "segmenter-infer", "tune-decode", "analysis-a", "analysis-b", "tail-benefit", "delta-enc"],
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--segmenter-arch", default="moryossef", choices=["moryossef", "s1"],
        help="segmenter-infer backend: moryossef = the faithful Moryossef analysis segmenter (default), s1 = in-system BIO head (ablation)"
    )
    parser.add_argument("--moryossef-config", default="configs/moryossef26.yaml", help="Moryossef analysis-segmenter config")
    parser.add_argument(
        "--segmenter-decode", default=None, choices=["duration", "plain"],
        help="whole-video decode; default per arch: s1 -> duration (our semi-Markov re-split), moryossef -> plain (Moryossef argmax)"
    )
    parser.add_argument("--bio-config", default="configs/bio_pretrain.yaml", help="S1 (in-system head) config for --segmenter-arch s1")
    parser.add_argument("--slt-config", default="configs/dlm.yaml")
    parser.add_argument("--baseline-config", default="configs/baseline_eval.yaml")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--language", default=None)  # None -> data.yaml active_languages[0] (never a stale hardcode)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--tiou-threshold", type=float, default=None)
    parser.add_argument("--num-sentences", type=int, default=None)
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="windows per translate/forward batch in loop-decode paths (RQ2 rows, analysis-b, tail-benefit, delta-enc)"
    )
    parser.add_argument("--noise-sigma", type=float, default=0.005, help="delta-enc keypoint-noise std (normalized coords)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-config", action="store_true", help="Persist the measured constant into configs/inference.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-test", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.language is None: args.language = str(load_yaml(args.data_config).get("active_languages", ["csl"])[0])
    if args.stage == "dataset-summary": result = dataset_summary(args)
    elif args.stage == "segmenter-infer": result = segmenter_infer(args)
    elif args.stage == "tune-decode": result = tune_decode(args)
    elif args.stage == "analysis-a":
        if not args.predictions: raise SystemExit("--predictions is required for --stage analysis-a")
        result = analysis_a(args)
    elif args.stage == "analysis-b": result = analysis_b(args)
    elif args.stage == "tail-benefit": result = tail_benefit(args)
    elif args.stage == "delta-enc": result = delta_enc(args)
    else: raise ValueError(f"Unsupported stage: {args.stage}")
    print(json.dumps(result, indent=2, sort_keys=True))
