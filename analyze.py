from __future__ import annotations
from dataclasses import asdict, dataclass
from statistics import median
from pathlib import Path
import json, argparse

import torch
from data.loader import load_language_records
from moryossef26.infer import load_prediction_file, load_segmenter_for_inference, predict_phrase_segments, save_prediction_file
from moryossef26.trainer import build_segmenter
from metrics import Segment, match_segments, temporal_iou
from utils import load_yaml


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
    # Per streaming_slt_prompt.md §5.5: skip mass is split between
    # truncated-window and multi-complete-window training cases.
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

        overseg_gold = {
            gold_idx for gold_idx, pred_indices in overlapping_pred_by_gold.items()
            if len(pred_indices) >= 2
        }
        underseg_pred = {
            pred_idx for pred_idx, gold_indices in overlapping_gold_by_pred.items()
            if len(gold_indices) >= 2
        }
        underseg_gold = {
            gold_idx for pred_idx in underseg_pred
            for gold_idx in overlapping_gold_by_pred.get(pred_idx, [])
        }
        phantom_pred = {
            pred_idx for pred_idx, pred in enumerate(pred_segments)
            if pred_idx not in overlapping_gold_by_pred
            and pred.start_s >= 0.0 and pred.end_s <= float(durations.get(video_id, pred.end_s))
        }
        counts["oversegmentation"] += len(overseg_gold)
        counts["undersegmentation"] += len(underseg_pred)
        counts["phantom"] += len(phantom_pred)
        counts["skipped"] += len(set(range(len(gold_segments))) - matched_gold - overseg_gold - underseg_gold)

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
        matched_pairs=matched_pairs, regular_matches=regular_matches, videos=len(set(gold) | set(predicted))
    )


def write_analysis_a_outputs(analysis: SegmenterErrorAnalysis, output_dir: str | Path, language: str) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jitter_rows = [asdict(sample) for sample in analysis.jitter_samples]
    head = [row["delta_head_s"] for row in jitter_rows]
    tail = [row["delta_tail_s"] for row in jitter_rows]
    jitter_payload = {
        "language": language, "samples": jitter_rows,
        "laplace": {"head": _laplace_fit(head), "tail": _laplace_fit(tail)}
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
    return {
        "language": args.language, "split": args.split or "all", "records": len(records),
        "sentences": len(durations), "split_sizes": {k: len(v) for k, v in splits.items()},
        "mean_sentence_s": sum(durations) / len(durations) if durations else 0.0,
        "max_sentence_s": max(durations) if durations else 0.0,
    }


def segmenter_infer(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test: 
        raise SystemExit("Refusing to run segmenter inference on test without --allow-test")
    cfg = load_yaml(args.data_config)
    records, _ = load_language_records(cfg, args.language, split=args.split)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_segmenter(args.segmenter_config)
    load_segmenter_for_inference(args.checkpoint, model, device)
    seg_cfg = load_yaml(args.segmenter_config)
    predictions = predict_phrase_segments(model, records, device=device, velocity=bool(seg_cfg.get("velocity", True)))
    output = Path(args.output or f"outputs/segmenter_predictions_{args.language}_{args.split}.json")
    save_prediction_file(predictions, output)
    return {
        "language": args.language, "split": args.split, "videos": len(records),
        "predicted_segments": sum(len(v) for v in predictions.values()), "output": str(output),
    }


def analysis_a(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Analysis A must run on dev; use --allow-test only for smoke debugging")
    cfg = load_yaml(args.data_config)
    seg_cfg = load_yaml(args.segmenter_config)
    records, _ = load_language_records(cfg, args.language, split=args.split)
    predictions = load_prediction_file(args.predictions)
    gold_segments = {
        record.video_id: [Segment(span.start_s, span.end_s) for span in record.sentences]
        for record in records
    }
    durations = {record.video_id: float(record.pose.duration_s) for record in records}
    analysis = analyze_segmenter_errors(
        predicted=predictions, gold=gold_segments, durations=durations,
        tiou_threshold=float(args.tiou_threshold or seg_cfg.get("match_tiou_threshold", 0.1)),
    )
    paths = write_analysis_a_outputs(analysis, args.output_dir, args.language)
    return {
        "language": args.language, "split": args.split, "event_counts": analysis.event_counts, 
        "mode_ratios": analysis.mode_ratios, "matched_pairs": analysis.matched_pairs,
        "regular_matches": analysis.regular_matches, "outputs": paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Misaligned-SLT analysis utilities")
    parser.add_argument("--stage", default="dataset-summary", choices=["dataset-summary", "segmenter-infer", "analysis-a"])
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--segmenter-config", default="configs/segmenter.yaml")
    parser.add_argument("--language", default="asf")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--checkpoint", default="checkpoints/segmenter/asf/model.pt")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--tiou-threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-test", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.stage == "dataset-summary": result = dataset_summary(args)
    elif args.stage == "segmenter-infer": result = segmenter_infer(args)
    elif args.stage == "analysis-a":
        if not args.predictions: raise SystemExit("--predictions is required for --stage analysis-a")
        result = analysis_a(args)
    else: raise ValueError(f"Unsupported stage: {args.stage}")
    print(json.dumps(result, indent=2, sort_keys=True))
