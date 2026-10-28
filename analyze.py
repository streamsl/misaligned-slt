from __future__ import annotations
from pathlib import Path
import json, argparse, math
import numpy as np

from data.loader import ANNOTATION_PROTOCOL, annotation_fingerprint, load_language_records
from moryossef26.infer import predict_phrase_segments
from infer.duration_decode import DurationModel
from models.checkpointing import without_annotation, load_checkpoint_meta

from metrics import edge_error_types
from evaluation.boundary_sweep import run_boundary_sweep
from evaluation.predictions import (
    corpus_token, drop_quarantined_predictions, gold_chunks, load_prediction_file, PredictionEvent,
    save_prediction_file, to_source, warn_annotation_mismatch
)
from evaluation.systems import load_segmenter, source_language
from utils import lambda_min_s, load_yaml, output_dir

# The error types of Moryossef et al. (2020) that each type of the edge chunk match corresponds to (a name map only).
MORYOSSEF_2020_NAMES = {
    "matched": "Started Pre/Post-Signing and Signing Underflow/Overflow (within tau)",
    "offset": "Started Pre/Post-Signing and Signing Underflow/Overflow", "joined": "Bridged (inside 1 chunk, correct here)",
    "bridged": "Bridged", "cut": "Signing Undetected Incorrectly", "phantom": "Signing Detected Incorrectly", "missed": "Skipped",
}

def caption_modes(types: dict) -> list[dict[str, float]]:
    """The window mode that each caption type matches, per tau (`metrics.edge_error_types` output), as shares of all captions:
    mode 1 holds 1 unit (matched, and offset with 1 unit held), mode 2 cuts a unit (cut), mode 3 holds 2 or more units (joined,
    bridged, and offset with 2 or more units held), mode 4 holds no unit (phantom). A missed unit has no caption, so no mode."""
    mode_of = {"matched": "mode1", "joined": "mode3", "cut": "mode2", "bridged": "mode3", "phantom": "mode4"}   # offset: by the units held
    out = []
    for j, row in enumerate(types["taus"]):
        counts = {"mode1": 0, "mode2": 0, "mode3": 0, "mode4": 0}
        for v in types["per_video"].values():
            for kind, held in zip(v["taus"][j]["captions"], v["held"]):
                counts[mode_of.get(kind) or ("mode1" if held <= 1 else "mode3")] += 1
        total = sum(counts.values())
        out.append({"tau_s": row["tau_s"], **{k: n / total if total else 0.0 for k, n in counts.items()}})
    return out


def write_segmenter_error_outputs(
    types: dict, out_dir: str | Path, language: str, arch: str, split: str = "dev", source_language: str | None = None,
) -> dict[str, str]:
    """The 3 files of `--stage segmenter-errors`, from `metrics.edge_error_types`: the edge offsets of the 1-to-1 held pairs and
    the cut positions per tau (jitter_*), the implied mode mix per tau and its mean (mode_ratios_*), and the shares and counts of
    each error type per tau and their means, with the Moryossef 2020 names (error_types_*). A zero-shot segmenter (trained on
    `source_language`) names both corpora (evaluation.predictions.corpus_token)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    head = {"language": language, "source_language": source_language or language, "split": str(split), "segmenter_arch": arch}
    corpora = corpus_token(language, source_language)
    jitter_payload = {**head, "samples": [
        {"video_id": vid, "caption_index": pi, "unit_index": k, "delta_head_s": dh, "delta_tail_s": dt, "tiou": iou}
        for vid, v in sorted(types["per_video"].items()) for pi, k, dh, dt, iou in v["pairs"]
    ], "cut_positions": {f"{r['tau_s']:g}": r["cut_positions"] for r in types["taus"]}}
    modes = caption_modes(types)
    mode_payload = {**head, "taus": modes, "mode_ratios": {
        k: sum(m[k] for m in modes) / len(modes) if modes else 0.0 for k in ("mode1", "mode2", "mode3", "mode4")}}
    types_payload = {**head, "captions": types["captions"], "units": types["units"],
                     "taus": [{k: r[k] for k in ("tau_s", "captions", "units", "caption_counts", "unit_counts")} for r in types["taus"]],
                     "captions_avg": types["captions_avg"], "units_avg": types["units_avg"],
                     "moryossef_2020_names": MORYOSSEF_2020_NAMES}
    paths = {
        "jitter": str(out_dir / f"jitter_{arch}_{corpora}_{split}.json"),
        "mode_ratios": str(out_dir / f"mode_ratios_{arch}_{corpora}_{split}.json"),
        "error_types": str(out_dir / f"error_types_{arch}_{corpora}_{split}.json"),
    }
    Path(paths["jitter"]).write_text(json.dumps(jitter_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["mode_ratios"]).write_text(json.dumps(mode_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["error_types"]).write_text(json.dumps(types_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def dataset_summary(args: argparse.Namespace) -> dict:
    cfg = load_yaml(args.data_config)
    records, splits = load_language_records(cfg, args.language, split=args.split)
    durations = [span.duration_s for rec in records for span in rec.sentences if getattr(span, "reliable", True)]
    # Λ_min is a fixed label-domain floor (inference.yaml span_selection.min_span_s, read by utils.lambda_min_s)
    # that no stage writes; train/slt._inject_gate_geometry copies it into the gate.
    p1_s = float(np.percentile(durations, 1)) if durations else 0.0
    # The stored rate of each video, as a dataset table reports it: the reader gives every video on the poses.model_fps grid (rec.pose.fps).
    median_fps = float(np.median([float(rec.pose.source_fps or rec.pose.fps) for rec in records])) if records else 0.0
    return {
        "language": args.language, "split": args.split or "all", "records": len(records),
        "sentences": len(durations), "split_sizes": {k: len(v) for k, v in splits.items()},
        "mean_sentence_s": sum(durations) / len(durations) if durations else 0.0,
        "max_sentence_s": max(durations) if durations else 0.0,
        "p1_sentence_s": p1_s, "median_fps": median_fps,
    }


def segmenter_infer(args: argparse.Namespace) -> dict: # Upstream segmenter for the error types and the cascade rows.
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run segmenter inference on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    source = source_language(args)   # the segmenter's corpus (its stamps and duration prior); the streams are --language's
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    model, device, velocity, rope_chunk_s, checkpoint = load_segmenter(args)
    decode = args.segmenter_decode or "duration"
    duration = DurationModel.for_language(data_cfg, source) if decode == "duration" else None
    print(f"[segmenter-infer] {args.segmenter_arch} from {checkpoint} (decode={decode})", flush=True)

    min_span_s = lambda_min_s(load_yaml(args.inference_config))   # the --offline row's span rule: complete spans, Λ_min floor, no cap
    segments = predict_phrase_segments(model, records, device=device, velocity=velocity, rope_chunk_s=rope_chunk_s, duration=duration,
                                       min_span_s=min_span_s)
    events = to_source({ # 1 decode/stream (clean segment); the span file is per source video in source time, like every artifact.
        vid: [PredictionEvent(video_id=vid, start_s=float(s.start_s), end_s=float(s.end_s)) for s in segs]
        for vid, segs in segments.items()
    }, records)
    predictions = {vid: [ev.segment for ev in evs] for vid, evs in events.items()}
    output = Path(args.output) if args.output else \
             output_dir("segmenter") / f"segmenter_predictions_{args.segmenter_arch}_{corpus_token(args.language, source)}_{args.split}.json"
    save_prediction_file(predictions, output, provenance={
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(records),
        "segmenter_arch": args.segmenter_arch, "decode": decode, "duration_model": duration.to_dict() if duration else None,
        "segmentation_decode": "semi_markov_viterbi" if duration else "bio_argmax", "min_span_s": min_span_s,
        "pose_normalization": "chunk" if args.segmenter_arch == "s1" else "video",
        "checkpoint": checkpoint, "segmenter_inputs": load_checkpoint_meta(checkpoint).get("pretrain_inputs"),  # the segmenter's pretrain.inputs
        "language": args.language, "source_language": source, "split": args.split,
    })
    return {
        "language": args.language, "split": args.split, "videos": len(predictions), "segmenter_arch": args.segmenter_arch, 
        "checkpoint": checkpoint, "predicted_segments": sum(len(v) for v in predictions.values()), "output": str(output),
    }


def chance_spans(args: argparse.Namespace) -> dict:
    """The chance rows' span file, made by no model: every stream of the split tiled from its start with spans of the median duration
    of the corpus's reliable train units (train labels only), the last one clipped to the stream end; per source video in source time,
    in the span-file format of --stage segmenter-infer. `eval.py --predictions <file>` scores it as a segmentation-only row."""
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to write chance spans on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    train, _ = load_language_records(data_cfg, args.language, split="train")
    span_s = float(np.median([u.duration_s for r in train for u in r.sentences if getattr(u, "reliable", True)]))
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    # Spans per stream (clean segment) on its own clock; the rounding keeps a stream of exactly n spans from a float-noise extra one.
    tiles = {r.video_id: [PredictionEvent(r.video_id, k * span_s, min(float(r.pose.duration_s), (k + 1) * span_s))
                          for k in range(max(1, math.ceil(round(float(r.pose.duration_s) / span_s, 6))))] for r in records}
    predictions = {vid: [ev.segment for ev in evs] for vid, evs in to_source(tiles, records).items()}
    output = Path(args.output) if args.output else output_dir("segmenter") / f"segmenter_predictions_chance_{args.language}_{args.split}.json"
    save_prediction_file(predictions, output, provenance={
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(records),
        "segmenter_arch": "chance", "decode": "none", "span_s": span_s, "train_annotation_fingerprint": annotation_fingerprint(train),
        "language": args.language, "split": args.split,
    })
    return {"language": args.language, "split": args.split, "streams": len(records), "span_s": span_s,
            "predicted_segments": sum(len(v) for v in predictions.values()), "output": str(output)}


def _assert_predictions_match_decode(args: argparse.Namespace) -> None:
    # The error types must describe the same decoder as the reported segmentation results.
    stamped = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    if not isinstance(stamped, dict) or "provenance" not in stamped: return
    prov = stamped["provenance"]
    arch = str(prov.get("segmenter_arch") or args.segmenter_arch)
    expected = args.segmenter_decode or prov.get("decode") or "duration"
    if expected not in {"plain", "duration"}: raise ValueError("Regenerate predictions with a current plain or duration decoder")
    if prov.get("source_language", args.language) != source_language(args): raise SystemExit(
        f"{args.predictions} holds spans of a segmenter trained on {prov.get('source_language', args.language)!r}: pass that "
        f"corpus as --source-language, so its prior is checked and the error types are named after it."
    )
    if expected == "duration":
        fitted = DurationModel.for_language(load_yaml(args.data_config), source_language(args)).to_dict()
        if without_annotation(prov.get("duration_model")) != without_annotation(fitted):
            raise ValueError("Regenerate predictions with the train-fitted duration prior")
        if prov.get("duration_model") != fitted: print(   # same fitted numbers on other train labels: a fingerprint warning
            f"WARNING: {args.predictions} decoded under a duration prior fitted on other train labels; "
            f"regenerate it before you report it.", flush=True
        )
    if arch != args.segmenter_arch or prov.get("decode") != expected:
        raise ValueError(f"Regenerate {args.segmenter_arch} predictions with its {expected} BIO decoder")


def segmenter_errors(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test: raise SystemExit(
        "Segmenter-error analysis on test needs --allow-test (the reported error types may "
        "describe the same split as the results tables; no training stage reads these files)."
    )
    cfg = load_yaml(args.data_config)
    records, _ = load_language_records(cfg, args.language, split=args.split)
    warn_annotation_mismatch(args.predictions, records, "--predictions")

    predictions = load_prediction_file(args.predictions)  # the segmenter-infer output file
    _assert_predictions_match_decode(args)
    # The quarantine rule of every row: a span majority-inside a quarantined unit is not scored.
    predictions = drop_quarantined_predictions(predictions, records)
    # The gold chunks and the edge taus of the main score, per source video in source time.
    scoring_cfg = load_yaml(args.eval_config)["scoring"]
    taus = [float(t) for t in scoring_cfg["edge_taus_s"]]
    types = edge_error_types(predictions, gold_chunks(records, float(scoring_cfg["chunk_gap_s"])), taus)
    print(f"[segmenter-errors] error types of the edge chunk match (taus {', '.join(f'{t:g}' for t in taus)} s); a caption holds "
          f"a unit when it covers at least half of it.", flush=True)
    paths = write_segmenter_error_outputs(
        types, args.output_dir or output_dir("analysis"), args.language, args.segmenter_arch, split=args.split, source_language=source_language(args)
    )
    return {
        "language": args.language, "split": args.split, "segmenter_arch": args.segmenter_arch, "captions": types["captions"],
        "units": types["units"], "captions_avg": types["captions_avg"], "units_avg": types["units_avg"], "outputs": paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Misaligned-SLT analysis utilities")
    parser.add_argument("--stage", default="dataset-summary",
                        choices=["dataset-summary", "segmenter-infer", "segmenter-errors", "boundary-sweep", "chance-spans"])
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--segmenter-arch", default="moryossef", choices=["moryossef", "s1"],
                        help="segmenter-infer backend: moryossef = external Moryossef segmenter, s1 = in-system BIO head")
    parser.add_argument("--moryossef-config", default="configs/moryossef26.yaml", 
                        help="Moryossef segmenter config (segmenter-infer, segmenter-errors, the cascade rows)")
    parser.add_argument("--segmenter-decode", choices=["plain", "duration"], default=None,
                        help="Segmenter-infer decoder (default duration: legal BIO + train-fitted duration prior, both segmenters)")
    parser.add_argument("--bio-config", default="configs/bio_pretrain.yaml", help="S1 (in-system head) config for --segmenter-arch s1")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--language", default=None)  # None -> data.yaml active_languages[0] (never a stale hardcode)
    parser.add_argument("--source-language", default=None,
                        help="segmenter-infer, segmenter-errors, boundary-sweep: the corpus the model trained on (default --language), as "
                             "eval.py --source-language: its stamps, warm start and duration prior; the data stay --language's")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--checkpoint", default=None, help="segmenter-infer: the segmenter; boundary-sweep: the translator")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=None, help="segmenter-errors: the folder of its 3 files (default outputs/analysis)")
    # boundary-sweep: one translator on gold units whose edges move on a grid (evaluation/boundary_sweep.py; eval.yaml boundary_sweep)
    parser.add_argument("--method", default="dlm", choices=["baseline", "ar", "dlm"], help="boundary-sweep: the translator")
    parser.add_argument("--method-config", default=None)
    parser.add_argument("--num-beams", type=int, default=None, help="boundary-sweep: AR search override; the main comparisons are greedy")
    parser.add_argument("--severity-grid-s", default=None,
                        help="boundary-sweep: comma-separated signed grid (fractions of duration in relative mode, else seconds)")
    # Fall back to --severity-grid-s / eval.yaml. One shared grid's cross-product cannot express a truncation-only sweep.
    parser.add_argument("--severity-grid-head", default=None,
                        help="boundary-sweep: head-axis grid override (use --severity-grid-head=-0.3,0 for negative-leading values)")
    parser.add_argument("--severity-grid-tail", default=None,
                        help="boundary-sweep: tail-axis grid override (use --severity-grid-tail=-0.3,0 for negative-leading values)")
    parser.add_argument("--severity-mode", default=None, choices=["relative", "absolute"],  # None -> eval.yaml boundary_sweep.severity_mode
                        help="boundary-sweep: relative (fraction of sentence duration, default) or absolute seconds")
    parser.add_argument("--num-sentences", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-test", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    data_cfg = load_yaml(args.data_config)
    if args.language is None: args.language = str(data_cfg["active_languages"][0])
    if args.source_language is not None and (args.stage in ("dataset-summary", "chance-spans") or args.source_language not in data_cfg["languages"]):
        raise SystemExit(f"--source-language {args.source_language!r} must name a corpus of {args.data_config}, for a stage that reads a "
                         f"model (segmenter-infer, segmenter-errors, boundary-sweep)")
    if args.stage == "boundary-sweep":   # 1 row per (grid_head, grid_tail) point, a table
        print(run_boundary_sweep(args).to_string())
        raise SystemExit(0)
    if args.stage == "dataset-summary": result = dataset_summary(args)
    elif args.stage == "segmenter-infer": result = segmenter_infer(args)
    elif args.stage == "chance-spans": result = chance_spans(args)
    elif args.stage == "segmenter-errors":
        if not args.predictions: raise SystemExit("--predictions is required for --stage segmenter-errors")
        result = segmenter_errors(args)
    else: raise ValueError(f"Unsupported stage: {args.stage}")
    print(json.dumps(result, indent=2, sort_keys=True))
