from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json, argparse
import numpy as np

from data.loader import ANNOTATION_PROTOCOL, annotation_fingerprint, load_language_records
from moryossef26.infer import predict_phrase_segments
from infer.duration_decode import DurationModel

from metrics import Segment, match_segments
from eval import (PredictionEvent, _drop_quarantined_predictions, _gold_events, _load_segmenter, 
                  load_prediction_file, require_annotation_match, save_prediction_file, to_source)
from utils import lambda_min_frames, load_yaml

# Low on purpose: a near-miss is a boundary offset (a matched pair), not a phantom or skipped event; a high bar would
# push near-misses into the error classes and bias the offset distribution towards zero. Override: --tiou-threshold.
SEGMENTER_ERROR_MATCH_TIOU = 0.1

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
    videos: int
    # Position in (0,1) of each spurious cut inside an over-segmented GT unit: where a segmenter truncates a unit
    # (the Mode-2 situation), reported beside the (Δ_head, Δ_tail) boundary offsets.
    overseg_cut_positions: list[float]

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
    durations: dict[str, float], material_overlap_s: float = 0.0, tiou_threshold: float = 0.1
) -> SegmenterErrorAnalysis: # Segmenter error counts and regular-match jitter samples.

    # `material_overlap_s`: a pred counts as covering a gold (for over/under-segmentation tests) only when their overlap exceeds this.
    # 0 = any-overlap, which DOUBLE-COUNTS small boundary offsets on back-to-back corpora: a span grazing its neighbour by a fraction of
    # a second is a boundary offset, yet any-overlap reclassifies the event as under-segmentation. Principled floor = Λ_min in seconds:
    # a fragment shorter than minimum selectable span can't form a 2nd window at deployment, so it can't make the event multi-sentence.

    # Jitter excludes over-segmented GT and under-segmenting pred spans — separate window modes, not 1-to-1 boundary noise.
    jitter_samples: list[JitterSample] = []
    overseg_cut_positions: list[float] = []
    counts = {"matched": 0, "oversegmentation": 0, "undersegmentation": 0, "skipped": 0, "phantom": 0}
    matched_pairs = 0

    for video_id in sorted(set(gold) | set(predicted)):
        pred_segments = list(predicted.get(video_id, []))
        gold_segments = list(gold.get(video_id, []))
        matches = match_segments(pred_segments, gold_segments, threshold=tiou_threshold)
        matched_pairs += len(matches)
        matched_pred = {pred_idx for pred_idx, _, _ in matches}
        matched_gold = {gold_idx for _, gold_idx, _ in matches}

        overlapping_pred_by_gold: dict[int, list[int]] = {}
        overlapping_gold_by_pred: dict[int, list[int]] = {}
        for pred_idx, pred in enumerate(pred_segments):
            for gold_idx, gt in enumerate(gold_segments):
                overlap = max(0.0, min(pred.end_s, gt.end_s) - max(pred.start_s, gt.start_s))
                if overlap > float(material_overlap_s):
                    overlapping_pred_by_gold.setdefault(gold_idx, []).append(pred_idx)
                    overlapping_gold_by_pred.setdefault(pred_idx, []).append(gold_idx)

        overseg_gold = {gold_idx for gold_idx, pred_indices in overlapping_pred_by_gold.items() if len(pred_indices) >= 2}
        underseg_pred = {pred_idx for pred_idx, gold_indices in overlapping_gold_by_pred.items() if len(gold_indices) >= 2}
        underseg_gold = {gold_idx for pred_idx in underseg_pred for gold_idx in overlapping_gold_by_pred.get(pred_idx, [])}
        # Pred-level categories are made MUTUALLY EXCLUSIVE, or the mode weights double-charge single events: matching uses tIoU with no 
        # absolute-overlap floor while the overlap maps use material_overlap_s, so a 1-to-1 matched pred whose raw overlap is under the 
        # floor would otherwise ALSO count as phantom, and a pred bridging 2 gold sentences (1 under-segmentation event) would otherwise 
        # ALSO be charged as an over-segmentation fragment of each over-segmented gold it touches.
        phantom_pred = {
            pred_idx for pred_idx, pred in enumerate(pred_segments) if pred_idx not in overlapping_gold_by_pred 
            and pred_idx not in matched_pred and pred.start_s >= 0.0 and pred.end_s <= float(durations.get(video_id, pred.end_s))
        }
        counts["oversegmentation"] += sum(sum(1 for pi in overlapping_pred_by_gold[g] if pi not in underseg_pred) for g in overseg_gold)
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
            jitter_samples.append(JitterSample(
                video_id=video_id, pred_index=pred_idx, gold_index=gold_idx, tiou=float(score),
                delta_head_s=float(pred.start_s - gt.start_s), delta_tail_s=float(pred.end_s - gt.end_s),
            ))
    mode_counts = mode_weights_from_events(counts)
    return SegmenterErrorAnalysis(
        jitter_samples=jitter_samples, mode_ratios=normalize_counts(mode_counts), event_counts=counts,
        matched_pairs=matched_pairs, videos=len(set(gold) | set(predicted)), overseg_cut_positions=overseg_cut_positions,
    )


def write_segmenter_error_outputs(
    analysis: SegmenterErrorAnalysis, output_dir: str | Path, language: str, arch: str, split: str = "dev"
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jitter_rows = [asdict(sample) for sample in analysis.jitter_samples]
    jitter_payload = {
        "language": language, "split": str(split), "segmenter_arch": arch, "samples": jitter_rows,
        "overseg_cut_positions": analysis.overseg_cut_positions,
    }
    mode_payload = {
        "language": language, "split": str(split), "segmenter_arch": arch,
        "mode_ratios": analysis.mode_ratios,
        "source_event_counts": analysis.event_counts,
        "source_weights": mode_weights_from_events(analysis.event_counts),
    }
    taxonomy_payload = {
        "language": language, "split": str(split), "segmenter_arch": arch,
        "event_counts": analysis.event_counts,
        "matched_pairs": analysis.matched_pairs,
        "videos": analysis.videos,
        "moryossef_2020_mapping": {
            "matched": "Started Pre/Post-Signing and Signing Underflow/Overflow",
            "oversegmentation": "Signing Undetected Incorrectly", "undersegmentation": "Bridged",
            "skipped": "Skipped", "phantom": "Signing Detected Incorrectly",
        },
    }
    paths = {
        "jitter": str(output_dir / f"a_jitter_{arch}_{language}_{split}.json"),
        "mode_ratios": str(output_dir / f"a_mode_ratios_{arch}_{language}_{split}.json"),
        "taxonomy": str(output_dir / f"a_error_taxonomy_{arch}_{language}_{split}.json"),
    }
    Path(paths["jitter"]).write_text(json.dumps(jitter_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["mode_ratios"]).write_text(json.dumps(mode_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["taxonomy"]).write_text(json.dumps(taxonomy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def dataset_summary(args: argparse.Namespace) -> dict:
    cfg = load_yaml(args.data_config)
    records, splits = load_language_records(cfg, args.language, split=args.split)
    durations = [span.duration_s for rec in records for span in rec.sentences if getattr(span, "reliable", True)]
    # Λ_min is a fixed label-domain floor (utils.LAMBDA_MIN_FRAMES, inference.yaml span_selection.min_span_frames) 
    # that no stage writes; train/slt._inject_gate_geometry copies it into the gate.
    p1_s = float(np.percentile(durations, 1)) if durations else 0.0
    median_fps = float(np.median([float(rec.pose.fps) for rec in records])) if records else 0.0
    return {
        "language": args.language, "split": args.split or "all", "records": len(records),
        "sentences": len(durations), "split_sizes": {k: len(v) for k, v in splits.items()},
        "mean_sentence_s": sum(durations) / len(durations) if durations else 0.0,
        "max_sentence_s": max(durations) if durations else 0.0,
        "p1_sentence_s": p1_s, "median_fps": median_fps,
    }


def segmenter_infer(args: argparse.Namespace) -> dict: # Upstream segmenter for the error taxonomy and the RQ2 cascade.
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run segmenter inference on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    model, device, velocity, rope_chunk_s, checkpoint = _load_segmenter(args)
    decode = args.segmenter_decode or "duration"
    duration = DurationModel.for_language(data_cfg, args.language) if decode == "duration" else None
    print(f"[segmenter-infer] {args.segmenter_arch} from {checkpoint} (decode={decode})", flush=True)

    segments = predict_phrase_segments(model, records, device=device, velocity=velocity, rope_chunk_s=rope_chunk_s, duration=duration)
    events = to_source({ # 1 decode/stream (clean segment); the span file is per source video in source time, like every artifact.
        vid: [PredictionEvent(video_id=vid, start_s=float(s.start_s), end_s=float(s.end_s)) for s in segs]
        for vid, segs in segments.items()
    }, records)
    predictions = {vid: [ev.segment for ev in evs] for vid, evs in events.items()}
    output = Path(args.output or f"outputs/segmenter_predictions_{args.segmenter_arch}_{args.language}_{args.split}.json")
    save_prediction_file(predictions, output, provenance={
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(records),
        "segmenter_arch": args.segmenter_arch, "decode": decode, "duration_model": duration.to_dict() if duration else None,
        "segmentation_decode": "semi_markov_viterbi" if duration else "bio_argmax",
        "pose_normalization": "chunk" if args.segmenter_arch == "s1" else "video",
        "checkpoint": checkpoint, "language": args.language, "split": args.split,
    })
    return {
        "language": args.language, "split": args.split, "videos": len(predictions), "segmenter_arch": args.segmenter_arch, 
        "checkpoint": checkpoint, "predicted_segments": sum(len(v) for v in predictions.values()), "output": str(output),
    }


def _assert_predictions_match_decode(args: argparse.Namespace) -> None:
    # The error taxonomy must describe the same decoder as the reported segmentation results.
    stamped = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    if not isinstance(stamped, dict) or "provenance" not in stamped: return
    prov = stamped["provenance"]
    arch = str(prov.get("segmenter_arch") or args.segmenter_arch)
    expected = args.segmenter_decode or prov.get("decode") or "duration"
    if expected not in {"plain", "duration"}: raise ValueError("Regenerate predictions with a current plain or duration decoder")
    if expected == "duration":
        fitted = DurationModel.for_language(load_yaml(args.data_config), args.language).to_dict()
        if prov.get("duration_model") != fitted: raise ValueError("Regenerate predictions with the train-fitted duration prior")
    if arch != args.segmenter_arch or prov.get("decode") != expected:
        raise ValueError(f"Regenerate {args.segmenter_arch} predictions with its {expected} BIO decoder")


def segmenter_errors(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test: raise SystemExit(
        "Segmenter-error analysis on test needs --allow-test (the reported taxonomy may "
        "describe the same split as the results tables; no training stage reads these files)."
    )
    cfg = load_yaml(args.data_config)
    records, _ = load_language_records(cfg, args.language, split=args.split)
    require_annotation_match(args.predictions, records, "--predictions")

    predictions = load_prediction_file(args.predictions)  # the segmenter-infer output file
    _assert_predictions_match_decode(args)
    # Ignore-region, prediction side (RQ2 rule): a segmenter span majority-inside a quarantined region would otherwise count as PHANTOM.
    predictions = _drop_quarantined_predictions(predictions, records)
    # Gold, spans and durations per source video in source time.
    gold_segments = {vid: [ev.segment for ev in evs] for vid, evs in _gold_events(records).items()}
    durations = {record.pose.video_id: sum(record.pose.frame_counts) / float(record.pose.fps) for record in records}

    # Convert the frame floor to this corpus's time base.
    median_fps = float(np.median([float(record.pose.fps) for record in records])) if records else 24.0
    lam_s = lambda_min_frames(load_yaml(args.inference_config)) / median_fps
    analysis = analyze_segmenter_errors(
        predicted=predictions, gold=gold_segments, durations=durations, material_overlap_s=lam_s,
        tiou_threshold=float(args.tiou_threshold if args.tiou_threshold is not None else SEGMENTER_ERROR_MATCH_TIOU)
    )
    print(f"[segmenter-errors] event taxonomy with material_overlap_s={lam_s:.3f}s (= span_selection.min_span_frames/{median_fps:g}fps): "
          f"a graze shorter than the minimum selectable span is boundary jitter, not a second sentence.", flush=True)
    paths = write_segmenter_error_outputs(analysis, args.output_dir, args.language, args.segmenter_arch, split=args.split)
    return {
        "language": args.language, "split": args.split, "segmenter_arch": args.segmenter_arch, "event_counts": analysis.event_counts, 
        "mode_ratios": analysis.mode_ratios, "matched_pairs": analysis.matched_pairs, "outputs": paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Misaligned-SLT analysis utilities")
    parser.add_argument("--stage", default="dataset-summary", choices=["dataset-summary", "segmenter-infer", "segmenter-errors"])
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--segmenter-arch", default="moryossef", choices=["moryossef", "s1"],
                        help="segmenter-infer backend: moryossef = external Moryossef segmenter, s1 = in-system BIO head")
    parser.add_argument("--moryossef-config", default="configs/moryossef26.yaml", 
                        help="Moryossef segmenter config (segmenter-infer, segmenter-errors, RQ2 cascade)")
    parser.add_argument("--segmenter-decode", choices=["plain", "duration"], default=None,
                        help="Segmenter-infer decoder (default duration: legal BIO + train-fitted duration prior, both segmenters)")
    parser.add_argument("--bio-config", default="configs/bio_pretrain.yaml", help="S1 (in-system head) config for --segmenter-arch s1")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--language", default=None)  # None -> data.yaml active_languages[0] (never a stale hardcode)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--tiou-threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-test", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.language is None: args.language = str(load_yaml(args.data_config).get("active_languages", ["asf"])[0])
    if args.stage == "dataset-summary": result = dataset_summary(args)
    elif args.stage == "segmenter-infer": result = segmenter_infer(args)
    elif args.stage == "segmenter-errors":
        if not args.predictions: raise SystemExit("--predictions is required for --stage segmenter-errors")
        result = segmenter_errors(args)
    else: raise ValueError(f"Unsupported stage: {args.stage}")
    print(json.dumps(result, indent=2, sort_keys=True))
