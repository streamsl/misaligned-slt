"""The RQ2 entry point (eval.py --rq 2): run 1 row, write its events and scores, print the table.

run_rq2 runs 1 row: --online (streaming FSM, evaluation/rq2_online.py), --offline or --segments (evaluation/rq2_offline.py), or a saved events 
file (--predictions, scoring only). Events and scores go to outputs/rq2_online/ or outputs/rq2_offline/, by row's `when` (utils.output_dir).
"""
from __future__ import annotations
from pathlib import Path
import argparse, json
import pandas as pd

from data.loader import annotation_fingerprint, ANNOTATION_PROTOCOL, load_language_records
from metrics import char_level_for_target
from evaluation.predictions import (
    _gold_events, _write_events_json, load_event_predictions, rq2_output_stem, 
    scoreable_predictions, stamped_annotation, warn_annotation_mismatch
)
from evaluation.rq2_offline import run_cascade, run_offline
from evaluation.rq2_online import _online_segmenter_arch, run_online
from evaluation.systems import _decode_provenance, _method_config_path, _parse_grid, rq2_translator_token
from evaluation.scoring import _joined_threshold_average, edge_chunk_scores, evaluate_predicted_events, gold_chunks, uncaptioned_units
from utils import lambda_min_s, load_yaml, output_dir, target_language


def _format_threshold_table(t: "pd.DataFrame") -> "pd.DataFrame":
    t = t.copy()
    avg = t.mean(axis=1)
    out = {}
    for idx, row in t.iterrows():
        vals = [float(v) for v in row]
        if all(v.is_integer() for v in vals): out[idx] = [f"{int(v):d}" for v in vals] + [f"{avg[idx]:.1f}"]
        else: out[idx] = [f"{v:.4f}" for v in vals] + [f"{avg[idx]:.4f}"]
    return pd.DataFrame.from_dict(out, orient="index", columns=[*t.columns, "avg"])


def run_rq2(args: argparse.Namespace) -> "pd.DataFrame":
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run RQ2 on test without --allow-test")
    if not args.predictions and not args.online and not args.segments and not args.offline: raise SystemExit(
        "--rq 2 needs --online (the streaming FSM), --offline (final model self-segments offline), "
        "--segments (given-spans cascaded baselines), or --predictions (score an events JSON)"
    )
    # Only the online row (the streaming FSM) reads --no-translate; a cascade or offline row would translate and be named 'none'.
    if getattr(args, "no_translate", False) and not args.online: raise SystemExit("--no-translate applies to --online only.")
    if getattr(args, "shadow_open_spans", False) and (not args.online or getattr(args, "no_translate", False)):
        raise SystemExit("--shadow-open-spans applies to a translating --online run only.")
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    method_cfg = load_yaml(_method_config_path(args), language=args.language)
    if getattr(args, "num_beams", None): method_cfg.setdefault("validation", {})["num_beams"] = int(args.num_beams)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    thresholds = _parse_grid(args.tiou_thresholds, eval_cfg.get("rq2", {}).get("tiou_thresholds", [0.5, 0.7, 0.9]))
    provenance = {
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(records), "method": args.method, 
        "num_beams": int(method_cfg.get("validation", {}).get("num_beams", 1)) if args.method == "baseline" else 1,
        "language": args.language, "split": args.split, "checkpoint": getattr(args, "checkpoint", None),
        "segments": getattr(args, "segments", None), "gate": bool(method_cfg.get("membership_gate", {}).get("enabled", False)),
    }
    # Geometry describes event generation; a saved-event rescore depends only on events and annotations.
    score_cfg = {} if args.predictions else load_yaml(args.inference_config)
    stem, events_path = None, None   # set by whichever branch writes events; a --predictions re-score writes no new artifact
    online_arch = _online_segmenter_arch(args) if args.online else None
    if online_arch is not None:
        provenance.update(method="baseline", segmenter_arch=online_arch, gate=False, num_beams=1)
        provenance["min_span_s"] = lambda_min_s(score_cfg)
    # The decode stamp follows the EFFECTIVE translator (an online cascade translates with the clean AR baseline
    # whatever --method says); a saved-event rescore reads no model geometry, the events carry their own.
    if not args.predictions: provenance.update(_decode_provenance(provenance["method"], method_cfg))
    if not args.predictions and getattr(args, "offline", False):
        # Λ_min is the offline joint row's GENERATION floor (run_offline drops shorter proposals). Scoring applies no floor,
        # and the cascade rows take external spans and declare none, so the stamp records how the events were made.
        provenance["generation_min_span_s"] = lambda_min_s(score_cfg)
    # A deliberate beam override changes the search budget and cannot enter a main row delta.
    if args.method == "baseline" and int(provenance["num_beams"]) != 1: print(
        f"[rq2] WARNING: baseline decodes with num_beams={provenance['num_beams']} while the ar/dlm arms are greedy — "
        f"this row is not search-budget matched.", flush=True
    )
    if args.online:
        predicted = run_online(args)
        arm_file = run_online.last_arm_checkpoint if online_arch is None else None   # the joint arm actually loaded
        if online_arch is None: provenance["checkpoint"] = arm_file
        # The arm's own token (config stem, one-stream stamp) also names its spans, so its same-span controls get their own files.
        spans = online_arch or f"joint-{rq2_translator_token(argparse.Namespace(**{**vars(args), 'no_translate': False}), arm_file)}"
        decode = "plain" if getattr(args, "segmenter_decode", None) == "plain" else "duration"
        translator = "clean" if online_arch and not getattr(args, "no_translate", False) else rq2_translator_token(args, arm_file)
        provenance["segmenter_arch"], provenance["decode"] = spans, decode
        provenance["translate"] = not bool(getattr(args, "no_translate", False))
        provenance["pose_normalization"] = "buffer"
        provenance["duration_model"] = run_online.last_duration_model
        provenance["segmentation_decode"] = run_online.last_segmentation_decode
        provenance["velocity"] = run_online.last_velocity
        # The FSM's commit geometry as run (the live inference.yaml tolerance).
        provenance["delta_s"] = float(score_cfg["boundary_stability"]["delta_s"])
        provenance["online_geometry"] = {
            "buffer_cap_s": score_cfg.get("buffer_cap_s"), "stride_s": score_cfg.get("stride_s", 1.0),
            "boundary_stability": score_cfg.get("boundary_stability", {}), "min_span_s": lambda_min_s(score_cfg),
            "segmentation_decode": run_online.last_segmentation_decode,
        }
        if online_arch is not None:
            provenance["checkpoint"] = getattr(run_online, "last_checkpoint", None)
            provenance["streaming_adaptation"] = "shared_fsm"
            if provenance["translate"]:
                provenance["translation_checkpoint"] = run_online.last_translation_checkpoint
                provenance["translation_pose_normalization"] = "span"
        provenance["when"] = "online"
        provenance["compute"] = run_online.last_compute   # wall time per stride on this device; not a score
        provenance["model_annotation"] = getattr(run_online, "last_model_annotation", None)   # the translator's labels
        stem = rq2_output_stem("online", spans, decode, translator, args.language, args.split)
        events_path = output_dir("rq2_online") / f"{stem}.json"
        _write_events_json(predicted, events_path, provenance)
    elif args.offline:
        predicted = run_offline(args)
        provenance["checkpoint"] = run_offline.last_checkpoint
        provenance["caption_context"] = "full_proposal_with_context"
        provenance["pose_normalization"] = "chunk"
        provenance["translation_pose_normalization"] = "commit_box"   # run_offline: one body box over the estimated commit buffer
        provenance["model_annotation"] = getattr(run_offline, "last_model_annotation", None)
        arm = rq2_translator_token(args, run_offline.last_checkpoint)
        provenance["segmenter_arch"], provenance["decode"], provenance["when"] = f"joint-{arm}", "duration", "offline"
        stem = rq2_output_stem("offline", provenance["segmenter_arch"], "duration", arm, args.language, args.split)
        events_path = output_dir("rq2_offline") / f"{stem}.json"
        _write_events_json(predicted, events_path, provenance)
    elif args.segments:
        src = json.loads(Path(args.segments).read_text(encoding="utf-8")).get("provenance") or {}
        spans, decode = src.get("segmenter_arch"), src.get("decode")
        if not spans or not decode: raise SystemExit(
            f"--segments {args.segments} carries no segmenter_arch/decode provenance, so this row can't name itself. Span files are written by "
            f"`eval.py --emit-gold-segments` and `analyze.py --stage segmenter-infer`; RQ2 events file without both keys must be regenerated."
        )
        # Spans an FSM committed online keep the name `online`: run_cascade clamps each crop (and an arm's body box) to frames the buffer
        # held at commit time, so re-translating them moves the translator alone. The clean translator reads span-normalized crops, as the
        # online cascades do; an arm reads the body box of the estimated commit buffer. The events keep their commit times.
        when = "online" if src.get("when") == "online" else "offline"
        predicted = run_cascade(args)
        provenance["checkpoint"] = run_cascade.last_checkpoint
        provenance["translation_pose_normalization"] = "span" if args.method == "baseline" else "commit_box"
        provenance["model_annotation"] = getattr(run_cascade, "last_model_annotation", None)
        # Named after the run: the translator token reads the stamp of the file the cascade actually loaded.
        stem = rq2_output_stem(when, spans, decode, rq2_translator_token(args, run_cascade.last_checkpoint), args.language, args.split)
        events_path = output_dir(f"rq2_{when}") / f"{stem}.json"
        if events_path.resolve() == Path(args.segments).resolve(): raise SystemExit(
            f"--segments {args.segments} would be overwritten by its own row: the spans and the translator are both "
            f"this file's. The same-span control reads it with --method baseline."
        )
        provenance["segmenter_arch"], provenance["decode"], provenance["when"] = spans, decode, when
        # The events keep the span file's own annotation stamp, so a reader of this row also warns when those spans are stale.
        provenance["annotation_fingerprint"] = stamped_annotation(args.segments)
        _write_events_json(predicted, events_path, provenance)
    else:
        warn_annotation_mismatch(args.predictions, records, "--predictions")
        predicted = load_event_predictions(args.predictions)
        gold_ids = {r.pose.video_id for r in records}
        if predicted and not (set(predicted) & gold_ids): raise SystemExit(
            f"--predictions has {len(predicted)} video_ids, none in the {len(gold_ids)} '{args.split}' videos — "
            f"wrong split or wrong events file (score would be silently near-zero)."
        )
        # Partial mismatch (same handling as run_cascade): a predicted video absent from this split has no gold, so it would be scored as pure 
        # phantoms — charging per-video precision 0 and a 0-F1 text row for events that belong to ANOTHER split, not being flagged as wrong file.
        foreign = set(predicted) - gold_ids
        if foreign:
            print(f"[rq2] WARNING: {len(foreign)}/{len(predicted)} --predictions video_ids are absent from "
                  f"--split {args.split}; scoring only the {len(predicted) - len(foreign)} that match.", flush=True)
            predicted = {vid: evs for vid, evs in predicted.items() if vid in gold_ids}
        
    gold = _gold_events(records)
    char_level = char_level_for_target(target_language(data_cfg, args.language))
    densevid = not bool(getattr(args, "no_densevid", False))
    # The main localization score (the edge chunk score, eval.yaml rq2.chunk_gap_s and rq2.edge_taus_s) beside the sentence tIoU F1.
    chunks, taus = gold_chunks(records, float(eval_cfg["rq2"]["chunk_gap_s"])), [float(t) for t in eval_cfg["rq2"]["edge_taus_s"]]
    # Rows whose events hold split spans (rows 6 and 10): the joined score is printed beside the split headline (diagnostic).
    joined = _joined_threshold_average(predicted, records, gold, chunks, thresholds, taus, char_level, densevid)
    predicted = scoreable_predictions(predicted, records)
    scored = evaluate_predicted_events(predicted, gold, thresholds, char_level=char_level, densevid=densevid)
    edge = edge_chunk_scores(predicted, chunks, taus, char_level)
    uncaptioned = uncaptioned_units(predicted, gold)
    summary = scored.get("thresholds", [])
    if events_path is not None: events_path.with_name(f"{stem}_scores.json").write_text(json.dumps({
        "provenance": provenance, "edge_chunk": edge, "uncaptioned": uncaptioned, "thresholds": scored.get("thresholds", []),
         "threshold_average": scored.get("threshold_average", {}), **({"joined_threshold_average": joined} if joined is not None else {})
    }, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    print(f"[rq2] main localization score: edge chunk F1 avg {edge['f1_avg']:.4f} (tau " + " / ".join(f"{t:g}" for t in taus) + " s: " + 
          " / ".join(f"{r['f1']:.4f}" for r in edge["taus"]) + "); the table below gives the sentence tIoU F1 (segmentation.f1)", flush=True)
    print(f"[rq2] units with no caption: {uncaptioned['uncaptioned']} of {uncaptioned['units']} ({uncaptioned['share']:.1%}). "
          "No caption overlaps them", flush=True)
    if joined is not None:
        avg = scored.get("threshold_average", {})
        print(f"[rq2] unit split: edge chunk F1 avg {edge['f1_avg']:.4f} split vs {joined['edge_chunk']['f1_avg']:.4f} joined back "
              f"into decoded spans; sentence F1 avg {float(avg['segmentation_f1']):.4f} vs {float(joined['segmentation_f1']):.4f}. " +
              (f"DVC BLEU-4 avg {float(avg['densevid_bleu4']):.2f} vs {float(joined['densevid_bleu4']):.2f}" if densevid else "") +
              " (report.py results gives the paired bootstrap)", flush=True)
    summary = pd.json_normalize(summary, sep=".")  # one row per tIoU threshold
    summary.set_index("tiou_threshold", inplace=True)
    table = summary.T
    # Row order = reading order: segmentation, the densevid headline, then the SODA fusion beside it.
    rank = lambda k: (0 if k.startswith("segmentation.") else 1 if ".densevid_" in k else 2 if ".soda_" in k else 3, k)
    table = table.loc[sorted(table.index, key=rank)]
    if not bool(getattr(args, "no_densevid", False)):
        print("[rq2] headline = densevid_* rows, threshold average (densevid_eval matching and aggregation over "
              "the configured tIoU set); soda_* charges missed gold and is reported beside it.", flush=True)
    return _format_threshold_table(table)
