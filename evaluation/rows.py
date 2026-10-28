"""The row entry point (eval.py --online / --offline / --segments / --predictions): run 1 row, write its events and scores, print the table.

run_row runs 1 row: --online (streaming FSM, evaluation/online.py), --offline or --segments (evaluation/offline.py), or a saved events 
file (--predictions, scoring only). Events and scores go to outputs/online/ or outputs/offline/, by row's `when` (utils.output_dir).
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse, json
import pandas as pd

from data.loader import annotation_fingerprint, ANNOTATION_PROTOCOL, load_language_records
from metrics import SODA_KEYS, TEXT_KEYS, edge_chunk_scores, evaluate_predicted_events, text_level, uncaptioned_units
from evaluation.predictions import (
    corpus_token, gold_events, write_events_json, gold_chunks, load_event_predictions, output_stem,
    scoreable_predictions, stamped_annotation, warn_annotation_mismatch
)
from evaluation.offline import run_cascade, run_offline
from infer.commit_gate import cap_cut_window
from evaluation.online import online_segmenter_arch, run_online
from evaluation.systems import decode_provenance, method_config_path, parse_grid, source_language, translator_token, translator_decoder
from utils import cfg_get, lambda_min_s, load_yaml, output_dir, resolve_pretrained, target_language


def _format_threshold_table(t: "pd.DataFrame") -> "pd.DataFrame":
    t = t.copy()
    avg = t.mean(axis=1)
    out = {}
    for idx, row in t.iterrows():
        vals = [float(v) for v in row]
        if all(v.is_integer() for v in vals): out[idx] = [f"{int(v):d}" for v in vals] + [f"{avg[idx]:.1f}"]
        else: out[idx] = [f"{v:.4f}" for v in vals] + [f"{avg[idx]:.4f}"]
    return pd.DataFrame.from_dict(out, orient="index", columns=[*t.columns, "avg"])


def run_row(args: argparse.Namespace) -> "pd.DataFrame":
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run a row on test without --allow-test")
    if not args.predictions and not args.online and not args.segments and not args.offline: raise SystemExit(
        "A row needs --online (the streaming FSM), --offline (final model self-segments offline), "
        "--segments (given-spans cascaded baselines), or --predictions (score an events JSON)"
    )
    # Only the online row (the streaming FSM) reads --no-translate; a cascade or offline row would translate and be named 'none'.
    if getattr(args, "no_translate", False) and not args.online: raise SystemExit("--no-translate applies to --online only.")
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    # The models trained on `source` (--source-language, default --language); the data, gold and text scoring are --language's.
    source = source_language(args)
    method_cfg = load_yaml(method_config_path(args), language=source)
    if getattr(args, "num_beams", None): method_cfg.setdefault("validation", {})["num_beams"] = int(args.num_beams)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    thresholds = parse_grid(args.tiou_thresholds, eval_cfg["scoring"]["tiou_thresholds"])
    # The evaluated corpus's scoring level (per character or word; lowercase both sides when data.yaml sets lowercase_references).
    level = text_level(target_language(data_cfg, args.language), cfg_get(data_cfg, "languages", args.language, "lowercase_references"))
    provenance = {
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(records), "method": args.method,
        "num_beams": int(method_cfg["validation"]["num_beams"]) if args.method == "baseline" else 1,
        "language": args.language, "source_language": source, "split": args.split, "checkpoint": getattr(args, "checkpoint", None),
        "segments": getattr(args, "segments", None), "gate": bool(method_cfg["membership_gate"]["enabled"]), "text_level": asdict(level),
    }
    corpora = corpus_token(args.language, source)   # the corpus field of the row's name: both corpora for a zero-shot row
    # Geometry describes event generation; a saved-event rescore depends only on events and annotations.
    score_cfg = {} if args.predictions else load_yaml(args.inference_config)
    # The cap cut's window as the online rows run it, in buffer time: it follows from the cap, the stride and K
    # (infer.commit_gate.cap_cut_window). The offline rows have no buffer cap.
    cap_cut = dict(zip(("window_start_s", "window_end_s"), cap_cut_window(
        float(score_cfg["buffer_cap_s"]), float(score_cfg["stride_s"]), int(score_cfg["boundary_stability"]["hysteresis_strides"]))
    )) if not args.predictions and args.online else None
    stem, events_path = None, None   # set by whichever branch writes events; a --predictions re-score writes no new artifact
    online_arch = online_segmenter_arch(args) if args.online else None
    if online_arch is not None:
        provenance.update(method="baseline", segmenter_arch=online_arch, gate=False, num_beams=1)
        provenance["min_span_s"] = lambda_min_s(score_cfg)
    # The decode stamp follows the EFFECTIVE translator (an online cascade translates with a clean floor whatever --method says) and
    # its decoder: a clean floor's is the one its file stamps, the file build_eval_model loads (--checkpoint, else data.yaml
    # pretrained_slt). A saved-event rescore reads no model geometry, the events carry their own.
    if not args.predictions:
        translator_file = None if provenance["method"] != "baseline" or getattr(args, "no_translate", False) \
                          else args.checkpoint or resolve_pretrained(method_cfg, data_cfg, source, default="")
        provenance.update(decode_provenance(translator_decoder(provenance["method"], translator_file), method_cfg))
    if not args.predictions and getattr(args, "offline", False):
        # Λ_min is the offline joint row's GENERATION floor (run_offline drops shorter proposals). Scoring applies no floor,
        # and the cascade rows take external spans and declare none, so the stamp records how the events were made.
        provenance["generation_min_span_s"] = lambda_min_s(score_cfg)
    # A deliberate beam override changes the search budget and cannot enter a main row delta.
    if args.method == "baseline" and int(provenance["num_beams"]) != 1: print(
        f"[row] WARNING: baseline decodes with num_beams={provenance['num_beams']} while the ar/dlm arms are greedy — "
        f"this row is not search-budget matched.", flush=True
    )
    if args.online:
        predicted = run_online(args)
        arm_file = run_online.last_arm_checkpoint if online_arch is None else None   # the joint arm actually loaded
        if online_arch is None: provenance["checkpoint"] = arm_file
        # The arm's own token (config stem, one-stream stamp) also names its spans, so its same-span controls get their own files.
        spans = online_arch or f"joint-{translator_token(argparse.Namespace(**{**vars(args), 'no_translate': False}), arm_file)}"
        decode = "plain" if getattr(args, "segmenter_decode", None) == "plain" else "duration"
        # An online cascade is named after the clean floor it translates with (clean or clean-dlm).
        translator = translator_token(args, run_online.last_translation_checkpoint) \
                     if online_arch and not getattr(args, "no_translate", False) else translator_token(args, arm_file)
        provenance["segmenter_arch"], provenance["decode"] = spans, decode
        provenance["translate"] = not bool(getattr(args, "no_translate", False))
        provenance["pose_normalization"] = "buffer"
        provenance["duration_model"] = run_online.last_duration_model
        provenance["segmentation_decode"] = run_online.last_segmentation_decode
        provenance["velocity"] = run_online.last_velocity
        # The FSM's commit geometry as run (the live inference.yaml tolerance).
        provenance["delta_s"] = float(score_cfg["boundary_stability"]["delta_s"])
        provenance["online_geometry"] = {
            "buffer_cap_s": score_cfg["buffer_cap_s"], "stride_s": score_cfg["stride_s"],
            "boundary_stability": score_cfg["boundary_stability"], "min_span_s": lambda_min_s(score_cfg),
            "cap_cut": cap_cut,
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
        stem = output_stem("online", spans, decode, translator, corpora, args.split)
        events_path = output_dir("online") / f"{stem}.json"
        write_events_json(predicted, events_path, provenance)
    elif args.offline:
        predicted = run_offline(args)
        provenance["checkpoint"] = run_offline.last_checkpoint
        provenance["caption_context"] = "full_proposal_with_context"
        provenance["pose_normalization"] = "chunk"
        provenance["translation_pose_normalization"] = "commit_box"   # run_offline: one body box over the estimated commit buffer
        provenance["model_annotation"] = getattr(run_offline, "last_model_annotation", None)
        arm = translator_token(args, run_offline.last_checkpoint)
        provenance["segmenter_arch"], provenance["decode"], provenance["when"] = f"joint-{arm}", "duration", "offline"
        stem = output_stem("offline", provenance["segmenter_arch"], "duration", arm, corpora, args.split)
        events_path = output_dir("offline") / f"{stem}.json"
        write_events_json(predicted, events_path, provenance)
    elif args.segments:
        src = json.loads(Path(args.segments).read_text(encoding="utf-8")).get("provenance") or {}
        spans, decode = src.get("segmenter_arch"), src.get("decode")
        if not spans or not decode: raise SystemExit(
            f"--segments {args.segments} carries no segmenter_arch/decode provenance, so this row can't name itself. Span files are written by "
            f"`eval.py --emit-gold-segments` and `analyze.py --stage segmenter-infer`; an events file without both keys must be regenerated."
        )
        # Spans made by a model stamp the corpus it trained on (model-free gold and chance spans stamp none). The row's name carries one
        # source, so its spans and its translator must share it.
        if src.get("source_language") not in (None, source): raise SystemExit(
            f"--segments {args.segments} holds spans of models trained on {src['source_language']!r}, but this row's translator is "
            f"{source!r}'s (--source-language, default --language). Pass --source-language {src['source_language']}."
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
        stem = output_stem(when, spans, decode, translator_token(args, run_cascade.last_checkpoint), corpora, args.split)
        events_path = output_dir(when) / f"{stem}.json"
        if events_path.resolve() == Path(args.segments).resolve(): raise SystemExit(
            f"--segments {args.segments} would be overwritten by its own row: the spans and the translator are both "
            f"this file's. The same-span control reads it with --method baseline."
        )
        provenance["segmenter_arch"], provenance["decode"], provenance["when"] = spans, decode, when
        # The events keep the span file's own annotation stamp, so a reader of this row also warns when those spans are stale.
        provenance["annotation_fingerprint"] = stamped_annotation(args.segments)
        write_events_json(predicted, events_path, provenance)
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
            print(f"[row] WARNING: {len(foreign)}/{len(predicted)} --predictions video_ids are absent from "
                  f"--split {args.split}; scoring only the {len(predicted) - len(foreign)} that match.", flush=True)
            predicted = {vid: evs for vid, evs in predicted.items() if vid in gold_ids}
        
    gold = gold_events(records)
    densevid = not bool(getattr(args, "no_densevid", False))
    # The main scores read the edge chunk match (eval.yaml scoring.chunk_gap_s and scoring.edge_taus_s): its F1, and the DVC and the SODA-style
    # text F1 of its pairs. The tIoU scores read the tIoU pairing: the sentence tIoU F1 and the DVC of densevid_eval.
    chunks, taus = gold_chunks(records, float(eval_cfg["scoring"]["chunk_gap_s"])), [float(t) for t in eval_cfg["scoring"]["edge_taus_s"]]
    # Never cut (the method rule): each committed span is one event, one caption with its whole text.
    predicted = scoreable_predictions(predicted, records)
    scored = evaluate_predicted_events(predicted, gold, thresholds, char_level=level, densevid=densevid)
    edge = edge_chunk_scores(predicted, chunks, taus, level)
    uncaptioned = uncaptioned_units(predicted, gold)
    summary = scored.get("thresholds", [])
    if events_path is not None: events_path.with_name(f"{stem}_scores.json").write_text(json.dumps({
        "provenance": provenance, "main": {"edge_chunk": edge, "uncaptioned": uncaptioned},
        "tiou": {"thresholds": scored.get("thresholds", []), "threshold_average": scored.get("threshold_average", {})},
    }, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    if level.lowercase: print(f"[row] every text is scored in lowercase (data.yaml languages.{args.language}.lowercase_references)", flush=True)
    for line in main_score_lines(edge, uncaptioned, taus): print(f"[row] {line}", flush=True)
    summary = pd.json_normalize(summary, sep=".")  # one row per tIoU threshold
    summary.set_index("tiou_threshold", inplace=True)
    table = summary.T
    table = table.loc[sorted(table.index, key=lambda k: (0 if k.startswith("segmentation.") else 1 if ".densevid_" in k else 2, k))]
    print("[row] tIoU scores (the tIoU pairing, not the main scores): the table below gives the sentence tIoU F1 (segmentation.*)" +
          (" and the DVC of densevid_eval (text_metrics.densevid_*)" if densevid else ""), flush=True)
    return _format_threshold_table(table)


def main_score_lines(edge: dict, uncaptioned: dict, taus) -> list[str]:
    """The printed main scores of a row: the edge chunk F1 (where) with its precision and recall per tau, the DVC and the SODA-style
    text F1 of its pairs (what), and the units with no caption."""
    lines = [f"MAIN localization: edge chunk F1 avg {edge['f1_avg']:.4f} (tau " + " / ".join(f"{t:g}" for t in taus) + " s: " +
             " / ".join(f"{r['f1']:.4f}" for r in edge["taus"]) + f"; precision {edge['precision_avg']:.4f}, recall {edge['recall_avg']:.4f})",
             "MAIN localization precision / recall per tau: " + ", ".join(f"{r['tau_s']:g} s {r['precision']:.4f} / {r['recall']:.4f}" for r in edge["taus"])]
    for name, block, keys in (("edge chunk DVC", edge["dvc_avg"], TEXT_KEYS), ("SODA-style text F1 on the same pairs", edge["soda_avg"], SODA_KEYS)):
        if block: lines.append(f"MAIN {name}: " + ", ".join(f"{k} {block[k]:.4f}" for k in keys))
    lines.append(f"units with no caption: {uncaptioned['uncaptioned']} of {uncaptioned['units']} ({uncaptioned['share']:.1%}); "
                 "no caption overlaps them")
    return lines
