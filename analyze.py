from __future__ import annotations
from pathlib import Path
import json, argparse
import numpy as np

from data.loader import load_language_records
from infer.duration_decode import DurationModel
from models.checkpointing import without_annotation

from metrics import edge_error_types
from evaluation.boundary_sweep import run_boundary_sweep
from evaluation.predictions import corpus_token, drop_quarantined_predictions, gold_chunks, load_prediction_file, warn_annotation_mismatch
from evaluation.systems import source_language
from utils import load_yaml, output_dir

# The error types of Moryossef et al. (2020) that each type of the edge chunk match corresponds to (a name map only).
MORYOSSEF_2020_NAMES = {
    "matched": "Started Pre/Post-Signing and Signing Underflow/Overflow (within tau)",
    "offset": "Started Pre/Post-Signing and Signing Underflow/Overflow", "joined": "Bridged (inside 1 chunk, correct here)",
    "bridged": "Bridged", "cut": "Signing Undetected Incorrectly", "phantom": "Signing Detected Incorrectly", "missed": "Skipped",
}

def caption_modes(types: dict) -> list[dict[str, float]]:
    """The window mode that each caption type matches, per cell (tau, t) of the grid of the main score (`metrics.edge_error_types`
    output, in the order of its `cells`), as shares of all captions: mode 1 holds 1 unit (matched, and offset with 1 unit held),
    mode 2 cuts a unit (cut), mode 3 holds 2 or more units (joined, bridged, and offset with 2 or more units held), mode 4 holds no
    unit (phantom). A missed unit has no caption, so no mode."""
    mode_of = {"matched": "mode1", "joined": "mode3", "cut": "mode2", "bridged": "mode3", "phantom": "mode4"}   # offset: by the units held
    out = []
    for j, cell in enumerate(types["cells"]):
        counts = {"mode1": 0, "mode2": 0, "mode3": 0, "mode4": 0}
        for v in types["per_video"].values():
            for kind, held in zip(v["cells"][j]["captions"], v["held"]):
                counts[mode_of.get(kind) or ("mode1" if held <= 1 else "mode3")] += 1
        total = sum(counts.values())
        out.append({"tau_s": cell["tau_s"], "tiou": cell["tiou"], **{k: n / total if total else 0.0 for k, n in counts.items()}})
    return out


def write_segmenter_error_outputs(
    types: dict, out_dir: str | Path, language: str, arch: str, decode: str, split: str = "dev", source_language: str | None = None,
) -> dict[str, str]:
    """The 3 files of `--stage segmenter-errors`, from `metrics.edge_error_types`: the edge offsets of the 1-to-1 held pairs and
    the cut positions per edge limit tau (jitter_*; edge offsets read no tIoU threshold), the implied mode mix per cell (tau, t) of
    the grid of the main score and its mean over the cells (mode_ratios_*), and the shares and counts of each error type per cell
    and their means over the cells, with the Moryossef 2020 names (error_types_*). Each name holds the segmenter and the decode of the
    spans, so the files of 2 decodes never overwrite each other. A zero-shot segmenter (trained on `source_language`) names both corpora
    (evaluation.predictions.corpus_token)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    head = {"language": language, "source_language": source_language or language, "split": str(split), "segmenter_arch": arch,
            "decode": decode}
    corpora = corpus_token(language, source_language)
    jitter_payload = {**head, "samples": [
        {"video_id": vid, "caption_index": pi, "unit_index": k, "delta_head_s": dh, "delta_tail_s": dt, "tiou": iou}
        for vid, v in sorted(types["per_video"].items()) for pi, k, dh, dt, iou in v["pairs"]
    ], "cut_positions": {f"{c['tau_s']:g}": c["cut_positions"] for c in types["cells"]}}   # the cut rule reads tau only: 1 list per tau
    modes = caption_modes(types)
    mode_payload = {**head, "cells": modes, "mode_ratios": {
        k: sum(m[k] for m in modes) / len(modes) if modes else 0.0 for k in ("mode1", "mode2", "mode3", "mode4")}}
    types_payload = {**head, "captions": types["captions"], "units": types["units"],
                     "cells": [{k: c[k] for k in ("tau_s", "tiou", "captions", "units", "caption_counts", "unit_counts")} for c in types["cells"]],
                     "captions_avg": types["captions_avg"], "units_avg": types["units_avg"],
                     "moryossef_2020_names": MORYOSSEF_2020_NAMES}
    paths = {
        "jitter": str(out_dir / f"jitter_{arch}_{decode}_{corpora}_{split}.json"),
        "mode_ratios": str(out_dir / f"mode_ratios_{arch}_{decode}_{corpora}_{split}.json"),
        "error_types": str(out_dir / f"error_types_{arch}_{decode}_{corpora}_{split}.json"),
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


def _span_source(args: argparse.Namespace) -> tuple[str, str]:
    """The segmenter and the decode of --predictions (the events file of a no-text row, eval.py --text none), checked before any record
    load. segmenter-errors decodes nothing: it reads the spans of the decode that the row which made the file used (eval.py --decode
    chooses it), and names its files after both. Returns (segmenter_arch, decode)."""
    stamped = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    prov = (stamped.get("provenance") if isinstance(stamped, dict) else None) or {}
    arch, decode = prov.get("segmenter_arch"), prov.get("decode")
    if not arch: raise SystemExit(f"{args.predictions} names no segmenter_arch: give the events file of a no-text row (eval.py --text none).")
    if decode not in {"argmax", "viterbi", "duration"}: raise ValueError(   # a name that no row writes now (plain)
        f"{args.predictions} decodes with {decode!r}: run its no-text row again (decode argmax, viterbi or duration)")
    if prov.get("source_language", args.language) != source_language(args): raise SystemExit(
        f"{args.predictions} holds spans of a segmenter trained on {prov.get('source_language', args.language)!r}: pass that "
        f"corpus as --source-language, so its prior is checked and the error types are named after it."
    )
    if decode == "duration":
        fitted = DurationModel.for_language(load_yaml(args.data_config), source_language(args)).to_dict()
        if without_annotation(prov.get("duration_model")) != without_annotation(fitted):
            raise ValueError("Regenerate predictions with the train-fitted duration prior")
        if prov.get("duration_model") != fitted: print(   # same fitted numbers on other train labels: a fingerprint warning
            f"WARNING: {args.predictions} decoded under a duration prior fitted on other train labels; "
            f"regenerate it before you report it.", flush=True
        )
    return arch, decode


def segmenter_errors(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test: raise SystemExit(
        "Segmenter-error analysis on test needs --allow-test (the reported error types may "
        "describe the same split as the results tables; no training stage reads these files)."
    )
    arch, decode = _span_source(args)
    cfg = load_yaml(args.data_config)
    records, _ = load_language_records(cfg, args.language, split=args.split)
    warn_annotation_mismatch(args.predictions, records, "--predictions")
    predictions = load_prediction_file(args.predictions)   # the spans of the events file of a no-text row
    # The quarantine rule of every row: a span majority-inside a quarantined unit is not scored.
    predictions = drop_quarantined_predictions(predictions, records)
    # The gold chunks and the grid of the main score (edge limits x tIoU thresholds), per source video in source time.
    scoring_cfg = load_yaml(args.eval_config)["scoring"]
    taus, thresholds = ([float(t) for t in scoring_cfg[key]] for key in ("edge_taus_s", "tiou_thresholds"))
    types = edge_error_types(predictions, gold_chunks(records, float(scoring_cfg["chunk_gap_s"])), taus, thresholds)
    print(f"[segmenter-errors] {arch} spans of the {decode} decode: error types of the counted pairs of the main score, per cell of the "
          f"grid (edge limits τ {', '.join(f'{t:g}' for t in taus)} s x tIoU {', '.join(f'{t:g}' for t in thresholds)}); a caption "
          f"holds a unit when it covers at least half of it.", flush=True)
    paths = write_segmenter_error_outputs(
        types, args.output_dir or output_dir("analysis"), args.language, arch, decode, split=args.split, source_language=source_language(args)
    )
    return {
        "language": args.language, "split": args.split, "segmenter_arch": arch, "decode": decode, "captions": types["captions"],
        "units": types["units"], "captions_avg": types["captions_avg"], "units_avg": types["units_avg"], "outputs": paths,
    }


def build_parser() -> argparse.ArgumentParser:
    # No abbreviated flags, as eval.py: a prefix never selects another flag.
    parser = argparse.ArgumentParser(description="Misaligned-SLT analysis utilities", allow_abbrev=False)
    parser.add_argument("--stage", default="dataset-summary", choices=["dataset-summary", "segmenter-errors", "boundary-sweep"])
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--language", default=None)  # None -> data.yaml active_languages[0] (never a stale hardcode)
    parser.add_argument("--source-language", default=None,
                        help="segmenter-errors, boundary-sweep: the corpus the model trained on (default --language), as "
                             "eval.py --source-language: its stamps, warm start and duration prior; the data stay --language's")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--predictions", default=None, help="segmenter-errors: the events file of a no-text row (eval.py --text none)")
    parser.add_argument("--output", default=None, help="boundary-sweep: the sweep file")
    parser.add_argument("--output-dir", default=None, help="segmenter-errors: the folder of its 3 files (default outputs/analysis)")
    # boundary-sweep: one translator on gold units whose edges move on a grid (evaluation/boundary_sweep.py; eval.yaml boundary_sweep)
    parser.add_argument("--text", default=None, choices=["pretrimmed", "ar", "dlm"],
                        help="boundary-sweep: the translator, as eval.py --text (pretrimmed: the pretrimmed translator; ar, dlm: that arm)")
    parser.add_argument("--pretrimmed-checkpoint", default=None,
                        help="boundary-sweep: the pretrimmed translator (default: data.yaml pretrained_slt)")
    parser.add_argument("--arm-checkpoint", default=None, help="boundary-sweep: the arm (default: the arm's checkpoint.dir)")
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
    if args.source_language is not None and (args.stage == "dataset-summary" or args.source_language not in data_cfg["languages"]):
        raise SystemExit(f"--source-language {args.source_language!r} must name a corpus of {args.data_config}, for a stage that reads a "
                         f"model (segmenter-errors, boundary-sweep)")
    if args.stage == "boundary-sweep":   # 1 row per (grid_head, grid_tail) point, a table
        print(run_boundary_sweep(args).to_string())
        raise SystemExit(0)
    if args.stage == "dataset-summary": result = dataset_summary(args)
    elif args.stage == "segmenter-errors":
        if not args.predictions: raise SystemExit("--predictions is required for --stage segmenter-errors")
        result = segmenter_errors(args)
    else: raise ValueError(f"Unsupported stage: {args.stage}")
    print(json.dumps(result, indent=2, sort_keys=True))
