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
from infer.duration_decode import DEPLOYED_SEGMENTER_ARCH, decode_config_key, duration_split_tags
from infer.commit_gate import bio_complete_spans, select_target_span
from moryossef26.infer import _phrase_logits, _set_rope_chunk, duration_decode_params, fit_duration_prior, predict_phrase_segments

from eval import _build_eval_model, _load_segmenter, _translate_windows, load_prediction_file, save_prediction_file
from metrics import Segment, match_segments, moryossef_segment_metrics, compute_text_metrics
from utils import load_yaml, update_yaml_scalar, pick_device, checkpoint_dir, resolve_pretrained

# Low on purpose: near-misses feed the (Δ_head, Δ_tail) jitter CDF as matched pairs, not phantom/skip events;
# a high bar biases the CDF to zero. Override: --tiou-threshold.
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
    # Position in (0,1) of each spurious cut in an over-segmented GT sentence = Mode-2 truncation depth, 
    # not (Δ_head, Δ_tail) jitter. The Mode-2 window sampler draws cut depth from this; uniform if empty.
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
    skipped_half = 0.5 * float(counts["skipped"]) # Skip mass splits between truncated-window and multi-complete-window cases.
    return {
        "mode1": float(counts["matched"]),
        "mode2": float(counts["oversegmentation"]) + skipped_half,
        "mode3": float(counts["undersegmentation"]) + skipped_half,
        "mode4": float(counts["phantom"]),
    }

def analyze_segmenter_errors(
    predicted: dict[str, list[Segment]], gold: dict[str, list[Segment]],
    durations: dict[str, float], tiou_threshold: float = 0.1,
) -> SegmenterErrorAnalysis: # Analysis-A event counts and regular-match jitter samples.

    # Jitter excludes over-segmented GT and under-segmenting pred spans — separate window modes, not 1-to-1 boundary noise.
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
        counts["oversegmentation"] += sum(len(overlapping_pred_by_gold[g]) for g in overseg_gold)
        counts["undersegmentation"] += len(underseg_pred)
        counts["phantom"] += len(phantom_pred)
        counts["skipped"] += len(set(range(len(gold_segments))) - matched_gold - overseg_gold - underseg_gold)

        # Mode-2 cut depths.
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
    # Λ_min (inference.yaml span_selection) = p1/2 of dev durations × median fps, an order of magnitude above the ≤2δ phantom scale. 
    # Per-corpus/per-fps: on a corpus switch reset min_span_frames + its dlm.yaml membership_gate mirror.
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
    # Upstream segmenter for Analysis A/B and the RQ2 cascade.
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run segmenter inference on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    model, device, velocity, rope_chunk_s, checkpoint = _load_segmenter(args)
    # Decode defaults per arch, as in eval --segmenter-eval: s1 -> our semi-Markov re-split, moryossef -> faithful Moryossef argmax. 
    # `--segmenter-decode duration` on external measures the deployed decode's errors, not the baseline's.
    # arch-aware: each segmenter reads its own duration_decode_<arch> block, never the other's.
    dd = duration_decode_params(load_yaml(args.inference_config), args.language, arch=args.segmenter_arch)
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
    save_prediction_file(predictions, output, provenance={
        "segmenter_arch": args.segmenter_arch, "decode": "duration" if duration_prior else "plain",
        "decode_hparams": dd if duration_prior else None, "checkpoint": checkpoint,
        "language": args.language, "split": args.split,
    })
    return {
        "language": args.language, "split": args.split, "videos": len(records),
        "segmenter_arch": args.segmenter_arch, "checkpoint": checkpoint,
        "predicted_segments": sum(len(v) for v in predictions.values()), "output": str(output),
    }


def tune_decode(args: argparse.Namespace) -> dict:
    """Two-fold dev tuning of the semi-Markov decode pair (split_bias, snap_radius_s) — infer/duration_decode.py.

    F1(split_bias) is a knife edge (segment-count Lagrange multiplier against a near-constant per-segment cost, so the split count steps) 
    and the knobs interact, so marginal sweeps mislead: joint FINE grid, one forward/video, decode-only re-sweep, per-fold macro F1@0.5, 
    max(min(foldA, foldB)). Run ONCE PER SEGMENTER: each arch's posteriors are calibrated differently, so `--segmenter-arch <arch>` pins 
    inference.yaml `duration_decode_<arch>.<lang>` — s1's block is the DEPLOYED one (FSM, membership gate, RQ1/RQ2), moryossef's is read 
    only by the baseline's own analysis. Never auto-applied without --write-config, keeping eval tuning-free.
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

    cached = []  # (video_id, tags, pB, gold, fps)
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

    def fold_f1(fold, bias, radius, w_b):
        f1s = []
        for _, tags, pB, gold, fps in fold:
            t = duration_split_tags(tags, pB, fps, prior, split_bias=bias, snap_radius_s=radius, boundary_logit_weight=w_b)
            oh = torch.nn.functional.one_hot(torch.as_tensor(t).long(), num_classes=4).float().unsqueeze(0)
            f1s.append(moryossef_segment_metrics(oh, gold.unsqueeze(0), prefix="p", tiou_threshold=0.5)["p_tiou_f1"])
        return float(np.mean(f1s)) if f1s else 0.0

    # Emission weight shifts the count-optimal bias (its logits are negative off-boundary), so the joint
    # grid must cover higher bias than the w=0 sweep needed.
    grid_bias = [round(float(b), 2) for b in np.arange(2.5, 8.01, 0.25)]
    grid_radius = [0.0, 0.5, 1.0, 1.5]
    grid_weight = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows, best = [], None
    for w_b in grid_weight:
        for bias in grid_bias:
            for radius in grid_radius:
                a, b = fold_f1(folds[0], bias, radius, w_b), fold_f1(folds[1], bias, radius, w_b)
                rows.append({
                    "split_bias": bias, "snap_radius_s": radius, "boundary_logit_weight": w_b, 
                    "foldA_f1@0.5": round(a, 4), "foldB_f1@0.5": round(b, 4)
                })
                key = (min(a, b), (a + b) / 2)  # fold-consistent first, mean as tie-break
                if best is None or key > best[0]: best = (key, rows[-1])
            print(f"[tune-decode] w={w_b:.2f} bias={bias:4.2f}: " + " ".join(
                f"r={r['snap_radius_s']:.1f}:{r['foldA_f1@0.5']:.3f}/{r['foldB_f1@0.5']:.3f}" for r in rows[-len(grid_radius):]
            ), flush=True)

    selected = dict(best[1])
    # Held-out estimate, the number to QUOTE: re-select per fold, evaluate on the other, average. 
    # The selected pair's own cells are in-selection maxima, overstating the gain.
    heldout = []
    for sel_i, eval_i in ((0, 1), (1, 0)):
        key_col = f"fold{'AB'[sel_i]}_f1@0.5"
        by_sel = max(rows, key=lambda r: r[key_col])
        heldout.append(by_sel[f"fold{'AB'[eval_i]}_f1@0.5"])
        
    heldout_f1 = round(sum(heldout) / 2, 4)
    output = Path(args.output or f"outputs/tune_decode_{args.segmenter_arch}_{args.language}_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    block = decode_config_key(args.segmenter_arch)
    payload = {
        "language": args.language, "split": args.split, "segmenter_arch": args.segmenter_arch, "checkpoint": checkpoint,
        "videos": len(cached), "prior": {"mu_log_s": prior.mu_log_s, "sd_log_s": prior.sd_log_s, "cap_s": prior.cap_s},
        "selected": selected, "heldout_f1@0.5": heldout_f1, "heldout_per_fold": [round(h, 4) for h in heldout], "grid": rows,
        "pin_as": {block: {str(args.language): {
            "split_bias": selected["split_bias"], "snap_radius_s": selected["snap_radius_s"],
            "boundary_logit_weight": selected["boundary_logit_weight"]
        }}},
        "rope_eval_chunk_s": rope_chunk_s,  # s1 only; the trained context this tuning is valid at
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.write_config: # Written as a YAML flow mapping under <block>.<language>.
        # The baseline's triple lives in its own block: one shared entry would decode the baseline with the
        # triple tuned for our head (see duration_decode_params).
        flow = (f"{{split_bias: {selected['split_bias']}, snap_radius_s: {selected['snap_radius_s']}, "
                f"boundary_logit_weight: {selected['boundary_logit_weight']}}}")
        if update_yaml_scalar(args.inference_config, (block, str(args.language)), flow):
            payload["config_updated"] = args.inference_config
            print(f"[tune-decode] wrote {block}.{args.language}: {flow} to {args.inference_config}", flush=True)
        else:
            print(f"[tune-decode] could not find {block}.{args.language} in {args.inference_config}; "
                  f"add the row manually under {block}: {args.language}: {flow}", flush=True)
    print(f"[tune-decode] selected split_bias={selected['split_bias']} snap_radius_s={selected['snap_radius_s']} "
          f"boundary_logit_weight={selected['boundary_logit_weight']} (in-selection F1@0.5 "
          f"{selected['foldA_f1@0.5']}/{selected['foldB_f1@0.5']}; HELD-OUT estimate {heldout_f1} — "
          f"quote the held-out number); pin the triple in inference.yaml {block}.{args.language}", flush=True)
    payload_out = dict(payload); payload_out["output"] = str(output)
    payload_out.pop("grid")  # grid lives in the JSON; keep stdout short
    return payload_out


def _assert_predictions_match_pinned_decode(args: argparse.Namespace) -> None:
    """Refuse spans decoded with a triple that is no longer pinned.

    Analysis A sets stage-2's window-error distribution, so re-tuning the upstream segmenter silently invalidates
    it — and S1, which trains on that distribution. The predictions file stamps the triple it used; compare.
    """
    stamped = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    if not isinstance(stamped, dict) or "provenance" not in stamped: return  # unstamped legacy file
    prov = stamped["provenance"]
    arch = str(prov.get("segmenter_arch") or DEPLOYED_SEGMENTER_ARCH)
    used = prov.get("decode_hparams") if prov.get("decode") == "duration" else None
    pinned = duration_decode_params(load_yaml(args.inference_config), args.language, arch=arch)
    if used == pinned: return
    raise SystemExit(
        f"{args.predictions} was decoded with {used}, but {decode_config_key(arch)}.{args.language} now pins "
        f"{pinned}. Analysis A calibrates stage-2 training, so it must reflect the decode you report — re-run "
        f"`analyze.py --stage segmenter-infer --segmenter-arch {arch} --segmenter-decode duration`, then retrain S1."
    )


def analysis_a(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test: raise SystemExit("Analysis A must run on dev; --allow-test only for smoke debugging")
    cfg = load_yaml(args.data_config)
    records, _ = load_language_records(cfg, args.language, split=args.split)
    predictions = load_prediction_file(args.predictions)  # the segmenter-infer output file
    _assert_predictions_match_pinned_decode(args)
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
    # eval.py's checkpoint-loading builder (plain scalars, no fake namespace).
    data_cfg = load_yaml(args.data_config)
    return _build_eval_model(method, args.checkpoint, args.language, data_cfg, method_cfg, device)


def analysis_b(args: argparse.Namespace) -> dict:
    """Analysis B — the paper's MOTIVATING experiment: a clean SLT model degrades under a realistic segmenter's
    boundary errors. Dev split, before SLT training, method-independent.

    Realistic point: windows the external segmenter actually cut (`--predictions` from segmenter-infer), reference = max-overlap GT sentence. 
    Clean point: GT-trimmed windows. Controlled severity CURVE: `eval.py --rq 1 --method baseline --split dev`; here, only the realistic gap.
    """
    if args.split == "test" and not args.allow_test: raise SystemExit("Analysis B runs on dev; --allow-test only for smoke debugging")
    if not args.predictions: raise SystemExit("--predictions required: Moryossef segmenter's spans (analyze.py --stage segmenter-infer)")
    data_cfg = load_yaml(args.data_config)
    base_cfg = load_yaml(args.baseline_config, language=args.language)  # re-point ${language} paths
    inference_cfg = load_yaml(args.inference_config)
    device = pick_device(args.device)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    model, tokenizer = _eval_model_for("baseline", args, base_cfg, device)

    def _translate_all(spans_with_refs, desc):
        # Load (I/O-bound) then batch-decode: one window per model call was dominated by beam decodes and pose opens.
        items, refs, kept = [], [], []
        for rec, start_s, end_s, ref, key in tqdm(spans_with_refs, desc=f"{desc}: load"):
            poses, ts = load_pose_window(rec.pose, start_s, end_s, normalize=True)
            if poses.shape[0] == 0: continue
            items.append((poses, ts, start_s)); refs.append(ref); kept.append(key)
        preds = [t for t, _, _ in _translate_windows(
            model, tokenizer, "baseline", items, device, inference_cfg, base_cfg, batch_size=int(args.batch_size)
        )]
        # `kept` = the gold key of each SCORED window: coverage must count only spans that were actually
        # translated, not ones dropped here for zero pose frames.
        return preds, refs, kept

    clean_pred, clean_ref, _ = _translate_all(
        [(rec, float(s.start_s), float(s.end_s), s.text, (rec.video_id, float(s.start_s)))
         for rec in records for s in rec.sentences], "Analysis B: clean (GT spans)"
    )

    # Realistic point.
    predicted = load_prediction_file(args.predictions)
    by_id = {r.video_id: r for r in records}
    real_spans, phantoms = [], 0  # `covered` is derived AFTER translation (see below)
    for vid, spans in predicted.items():
        rec = by_id.get(vid)
        if rec is None: continue
        for span in spans:
            best, best_ov = None, 0.0
            for gt in rec.sentences:
                ov = max(0.0, min(span.end_s, gt.end_s) - max(span.start_s, gt.start_s))
                if ov > best_ov: best_ov, best = ov, gt
            if best is None:
                phantoms += 1; continue  # phantom span in a gap: no GT reference, excluded from the score
            real_spans.append((rec, float(span.start_s), float(span.end_s), best.text, (vid, float(best.start_s))))

    real_pred, real_ref, real_keys = _translate_all(real_spans, "Analysis B: realistic (segmenter spans)")
    covered = set(real_keys)
    clean = compute_text_metrics(clean_pred, clean_ref, prefix="clean") if clean_pred else {}
    realistic = compute_text_metrics(real_pred, real_ref, prefix="realistic") if real_pred else {}
    n_gold = sum(len(r.sentences) for r in records)
    payload = {
        "language": args.language, "split": args.split, "clean": clean, "realistic": realistic,
        "clean_windows": len(clean_pred), "realistic_windows": len(real_pred),
        "delta_bleu4": float(clean.get("clean_bleu4", 0.0) - realistic.get("realistic_bleu4", 0.0)),
        # Realistic scores MATCHED windows only: uncovered GT and phantom windows cost nothing, windows may share
        # a reference. Low gold_coverage => the gap understates the damage; recall-inclusive accounting is RQ2's.
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

    Clean-trained translator (Analysis-B clean baseline), dev, head at the true sentence start (Δ_head = 0),
    Δ_tail swept, BLEU-4 per point. Elbow = first grid point whose marginal BLEU/s drops below eval.yaml
    tail_benefit.latency_quality_coeff_bleu_per_s — not hand-picked, not %-of-clean.

    The spec calls the elbow the buffer cap, but the FSM buffer holds the WHOLE sentence plus trailing context, 
    so a 1–3 s elbow cannot be it: buffer_cap_s = p99 sentence duration + stride_s + delta/fps (raw terms in the JSON).
    """
    if args.split == "test" and not args.allow_test: raise SystemExit("Tail-benefit runs on dev; --allow-test only for smoke debugging")
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    base_cfg = load_yaml(args.baseline_config, language=args.language)
    inference_cfg = load_yaml(args.inference_config)
    tb_cfg = eval_cfg.get("tail_benefit", {})
    grid = [float(x) for x in tb_cfg.get("tail_grid_s", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])]
    coeff = float(tb_cfg.get("latency_quality_coeff_bleu_per_s", 0.5))

    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    device = pick_device(args.device)
    model, tokenizer = _eval_model_for("baseline", args, base_cfg, device)

    durations = [span.duration_s for rec in records for span in rec.sentences]
    p99_duration = float(np.percentile(durations, 99)) if durations else 0.0
    fps_hint = float(np.median([r.pose.fps for r in records])) if records else 24.0
    sentences = [(rec, span) for rec in records for span in rec.sentences]
    if args.num_sentences: sentences = sentences[: int(args.num_sentences)]

    # ONE pose read per sentence at the LONGEST tail, sliced per grid point (per-frame normalization, so a slice of the long window == the 
    # short load), then batch-decode: len(grid) reads + len(grid) single-window beam decodes per sentence took hours on a network filesystem.
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
            # Clamped = less tail than the grid point claims: a flat step at large dt can mean "no more video",
            # not "no more benefit". Without this fraction the elbow is a tail-margin artifact.
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

    # CAPACITY, not the elbow: the buffer must hold the longest realistic sentence (p99), plus one stride for its
    # terminator to appear, plus the delta commit tolerance. The elbow cannot size it — the curve's SIGN depends on
    # what the scoring model trained on (span-trained baseline_train: context hurts, elbow pins to 0; buffer-trained
    # arms: context helps), so it measures context appetite, not buffer requirement. The curve stays as a diagnostic.
    stride_s = float(inference_cfg.get("stride_s", 1.0))
    delta_s = float((inference_cfg.get("boundary_stability", {}) or {}).get("delta_enc_frames", 0)) / max(fps_hint, 1.0)
    buffer_cap_s = round(p99_duration + stride_s + delta_s, 2)
    payload = {
        "language": args.language, "split": args.split, "sentences": len(sentences),
        "latency_quality_coeff_bleu_per_s": coeff, "curve": curve,
        "elbow_tail_s": elbow, "p99_sentence_duration_s": p99_duration, "buffer_cap_s": buffer_cap_s,
        "cap_terms": {"p99_s": p99_duration, "stride_s": stride_s, "delta_enc_s": round(delta_s, 3)},
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

    Use the S1 in-system BIO head (checkpoints/bio_s1, from train-bio) — what the FSM runs — not the Moryossef
    analysis segmenter, not the DLM checkpoint: δ_enc calibrates the gate's cut overlap against that head's noise,
    dlm.yaml asserts δ == this value, and reading it off the DLM checkpoint is an ordering circularity. 
    Measure as the head ENTERS stage 2.

    Two forwards per dev sentence window; shift of the selected span's terminator index under (a) dropped leading
    frame = stride-phase misalignment from a growing buffer; (b) Gaussian keypoint noise sigma on x/y = pose
    jitter. delta_enc = ceil(p90 over both): movement below the head's own noise floor must not block a commit.
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

    # Measure δ_enc under the DEPLOYED decode: with inference.yaml duration_decode on the FSM re-splits every buffer BEFORE reading 
    # terminators (infer/stream.py step()). Raw argmax is the wrong instrument — on back-to-back corpora its first O-or-B jumps WHOLE 
    # SENTENCES per one-frame perturbation, whose p90 pushed min_span_frames past the buffer cap and broke FSM span selection.
    dd = duration_decode_params(load_yaml(args.inference_config), args.language)
    duration_prior = None
    if dd is not None:
        train_records, _ = load_language_records(data_cfg, args.language, split="train")
        duration_prior = fit_duration_prior(train_records, **dd)
    # Track the SELECTED span's terminator (select_target_span at the deployed Λ_min), not first_terminator_index over raw tags: 
    # raw tags carry phantom 1-frame micro-spans Λ_min filters at deployment, calibrating δ on spans the gate can never commit.
    min_span = int((load_yaml(args.inference_config).get("span_selection", {}) or {}).get("min_span_frames", 0))
    print(f"[delta-enc] terminator decode: {'duration (deployed)' if duration_prior else 'plain argmax'}; "
          f"Lambda_min={min_span} frames", flush=True)

    fps_hint = float(np.median([r.pose.fps for r in records])) if records else 24.0
    sentences = [(rec, span) for rec in records for span in rec.sentences]
    if args.num_sentences: sentences = sentences[: int(args.num_sentences)]
    rng = np.random.default_rng(int(args.seed))

    @torch.no_grad()
    def decode_variants(variants: list[tuple[np.ndarray, np.ndarray]], start_s: float, fps: float) -> list[tuple[int, int] | None]:
        """(terminator_index, span_count) per variant, or None where the deployed FSM would select nothing.

        Equal lengths batch, other lengths get their own forward — correctness, not speed: the Uni-Sign pose encoder IGNORES frame_mask 
        (parity only) and its ST-GCN temporal convs have a ±6-frame receptive field, so a pad frame leaks into last real frames and global 
        attention perturbs EVERY frame's logits (|Δlogit| up to 0.077, flipping argmax tags). Padding would bias the drop-first-frame 
        family alone — manufacturing the jitter δ measures.
        """
        out: list[tuple[int, int] | None] = [None] * len(variants)
        groups: dict[int, list[int]] = {}
        for i, (p, _) in enumerate(variants): groups.setdefault(int(p.shape[0]), []).append(i)
        for n, idxs in groups.items():
            poses = torch.stack([torch.as_tensor(variants[i][0], dtype=torch.float32) for i in idxs]).to(device)
            ts = torch.stack([torch.as_tensor(variants[i][1] - start_s, dtype=torch.float32) for i in idxs]).to(device)
            mask = torch.ones(poses.shape[:2], dtype=torch.bool, device=device)  # exact length: no padding at all
            logits = model(poses, mask, timestamps_s=ts).logits
            for j, i in enumerate(idxs):
                tags = logits[j].argmax(dim=-1)
                if duration_prior is not None and n > 2:
                    pB = torch.softmax(logits[j].float(), dim=-1)[:, BIO["B"]].cpu().numpy()
                    tags = torch.as_tensor(duration_split_tags(
                        tags.cpu().numpy(), pB, fps, duration_prior, mark_onsets=False, split_open_tail="survival"
                    ))
                span = select_target_span(tags, min_span)
                # No fallback to first_terminator_index: `span is None` = "no target this stride" (FSM waits), and
                # a fallback would calibrate δ on a terminator the gate never tracks.
                out[i] = (int(span[1]), len(bio_complete_spans(tags))) if span is not None else None
        return out

    shifts: dict[str, list[int]] = {"drop_first_frame": [], "keypoint_noise": []}
    flips: dict[str, int] = {"drop_first_frame": 0, "keypoint_noise": 0}
    for rec, span in tqdm(sentences, desc="delta-enc"):
        start_s = max(0.0, span.start_s - 1.0)
        end_s = min(rec.pose.duration_s, span.end_s + 1.0)
        poses, timestamps = load_pose_window(rec.pose, start_s, end_s, normalize=True)
        if poses.shape[0] < 3: continue
        fps = float(rec.pose.fps)
        noisy = poses.copy()
        noisy[..., :2] = noisy[..., :2] + rng.normal(0.0, sigma, size=noisy[..., :2].shape).astype(noisy.dtype)
        base, dropped, perturbed = decode_variants([(poses, timestamps), (poses[1:], timestamps[1:]), (noisy, timestamps)], start_s, fps)
        if base is None: continue
        base_term, base_k = base
        # δ covers CONTINUOUS jitter only: a changed SEGMENT COUNT = re-decided segmentation (terminator moves ~a whole sentence), the χ 
        # commit log's business. Detect by count, not magnitude — magnitude would clip genuine jitter on short-sentence corpora. 
        # Dropped-buffer indices sit 1 frame earlier.
        for fam, val, off in (("drop_first_frame", dropped, 1), ("keypoint_noise", perturbed, 0)):
            if val is None: continue
            term, k = val
            if k != base_k: flips[fam] += 1; continue
            shifts[fam].append(abs((term + off) - base_term))

    # Retained (same-count) pairs only; discarded re-decisions sit beside them, not folded in.
    stats = {family: {
        "n": len(values), "median": float(np.median(values)) if values else 0.0,
        "p90": float(np.percentile(values, 90)) if values else 0.0, "max": int(max(values)) if values else 0,
        "count_flips_excluded": flips[family],
    } for family, values in shifts.items()}
    delta = int(math.ceil(max((s["p90"] for s in stats.values()), default=0.0))) or 1
    for family, s in stats.items():
        total = s["n"] + s["count_flips_excluded"]
        if total and s["count_flips_excluded"] / total > 0.1: print(
            f"[delta-enc] WARNING: {family} discarded {s['count_flips_excluded']}/{total} pairs as segment-count "
            f"re-decisions — δ={delta} rests on {s['n']} samples; inspect before --write-config.", flush=True)
        if s["n"] == 0: print(f"[delta-enc] WARNING: {family} has NO usable pairs; its p90 contributes 0.", flush=True)
    payload = {
        "language": args.language, "split": args.split, "noise_sigma": sigma,
        "sentences": len(sentences), "families": stats, "delta_enc_frames": delta,
    }
    output = Path(args.output or f"outputs/delta_enc_{args.language}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # delta is not a standalone constant. On commit the buffer is cut at terminator-delta, so the next buffer opens
    # with a delta-frame leftover of the sentence just emitted; Lambda_min must exceed delta or that leftover is
    # selectable as a span (infer/stream.py's own rule and default). dlm.yaml's gate must equal both. Written
    # together so the four values cannot drift — a stale pair is exactly what leaves the FSM in an invalid geometry.
    lam = delta + 1
    p10_frames = int(np.percentile([s.duration_s for r in records for s in r.sentences], 10) * fps_hint) if records else 0
    if p10_frames and lam > p10_frames: print(
        f"[delta-enc] WARNING: Lambda_min={lam} exceeds the p10 sentence ({p10_frames} frames) — >10% of real "
        f"sentences become unselectable. delta is inflated by snap_radius_s; re-tune the decode before accepting.", flush=True
    )
    payload["min_span_frames"] = lam
    if args.write_config:
        written = [update_yaml_scalar(args.inference_config, ("boundary_stability", "delta_enc_frames"), delta),
                   update_yaml_scalar(args.inference_config, ("span_selection", "min_span_frames"), lam),
                   update_yaml_scalar(args.slt_config, ("membership_gate", "delta"), delta),
                   update_yaml_scalar(args.slt_config, ("membership_gate", "min_span_frames"), lam)]
        payload["config_updated"] = [c for c, ok in zip([args.inference_config] * 2 + [args.slt_config] * 2, written) if ok]
        # Report what update_yaml_scalar actually changed — an unconditional "wrote" here would mask a failed write.
        if payload["config_updated"]: print(f"[delta-enc] wrote delta_enc_frames={delta}, min_span_frames={lam} to "
                                            f"{', '.join(sorted(set(payload['config_updated'])))}", flush=True)
        else: print(f"[delta-enc] WARNING: --write-config changed nothing (keys missing or files unwritable): "
                    f"{args.inference_config}, {args.slt_config}", flush=True)
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
    parser.add_argument(
        "--write-config", action="store_true",
        help="Persist measured constants: tail-benefit -> buffer_cap_s; delta-enc -> delta_enc_frames + "
        "min_span_frames (both configs); tune-decode -> duration_decode_<arch>.<language>"
    )
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
