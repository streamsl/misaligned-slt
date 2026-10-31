"""The row entry point (eval.py): run 1 row, write its events and scores, return the table of its main score.

The flags of a row are the fields of its name, {when}_{spans}_{decode}_{text}_{corpus}_{split} (evaluation.predictions.output_stem):
--offline or --online, --spans (gold, moryossef, s1, ar, dlm, chance, or a span or events file), --decode (the decode of a standalone
segmenter: argmax, viterbi or duration) and --text (none, pretrimmed, ar or dlm). run_row cuts the spans: the gold units, the chance tiling, a
file, or one pass of a segmenter or an arm (run_offline, evaluation/online.py run_online). The pass also gives the text when the text is
its own (an arm's own translator; online, the pretrimmed translator at a segmenter's commits); the cascade (run_cascade) gives every other
text. --predictions scores a saved events file again. Events and scores go to outputs/online/ or outputs/offline/, by the row's `when`
(utils.output_dir), or to --output. Every row prints 1 report: the title line with the main score (the F1 of the tIoU x edge-limit
grid at edge tau avg and tIoU avg), the table that run_row returns (rows: each score at every edge tau, their avg and the reference row
"no limit"; columns: the tIoU thresholds and their avg), then its notes (`table.attrs["notes"]`, which eval.py prints below the table):
the units with no caption and where the secondary scores are (the keys densevid and frame of the _scores.json), which are never printed.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import argparse, json
import pandas as pd

from data.loader import annotation_fingerprint, ANNOTATION_PROTOCOL, load_language_records
from metrics import TEXT_NAME, edge_chunk_scores, evaluate_predicted_events, text_level, uncaptioned_units
from evaluation.predictions import (
    corpus_token, gold_events, write_events_json, gold_chunks, load_event_predictions, output_stem,
    scoreable_predictions, stamped_annotation, warn_annotation_mismatch
)
from evaluation.offline import chance_spans, run_cascade, run_offline
from infer.commit_gate import cap_cut_window
from models.checkpointing import load_checkpoint_meta
from evaluation.online import run_online
from evaluation.systems import (
    ARMS, SEGMENTERS, TEXT_METHODS, decode_provenance, method_config_path, parse_grid, source_language, translator_token,
    translator_decoder)
from utils import cfg_get, checkpoint_dir, lambda_min_s, load_yaml, output_dir, resolve_pretrained, target_language

SOURCES = ("gold", *SEGMENTERS, *ARMS, "chance")   # the span sources that --spans names; any other value is a span or events file


def _text_token(args: argparse.Namespace, source: str) -> str:
    # The `text` field of a cascade, named before any model loads from the file that build_eval_model loads: --pretrimmed-checkpoint, else
    # data.yaml pretrained_slt; --arm-checkpoint, else the arm's checkpoint.dir.
    if args.text == "none": return "none"
    method = TEXT_METHODS[args.text]
    method_cfg = load_yaml(method_config_path(args, method), language=source)
    file = (args.pretrimmed_checkpoint or resolve_pretrained(method_cfg, load_yaml(args.data_config), source, default="")) \
           if method == "baseline" else args.arm_checkpoint or checkpoint_dir(method_cfg, default="")
    return translator_token(args, method, file if file and Path(file).exists() else None)


def _saved_events(path: str, records: list, flag: str, split: str) -> dict:
    """The events of a saved file (--predictions, --spans <file>) on the videos of this split: a file with none of them is refused, and
    the events of another split's videos are dropped, because a video with no gold would score every event as a phantom."""
    warn_annotation_mismatch(path, records, flag)
    predicted, gold_ids = load_event_predictions(path), {r.pose.video_id for r in records}
    if predicted and not (set(predicted) & gold_ids): raise SystemExit(
        f"{flag} {path} has {len(predicted)} video_ids, none in the {len(gold_ids)} '{split}' videos — "
        f"wrong split or wrong events file (score would be silently near-zero)."
    )
    foreign = set(predicted) - gold_ids
    if foreign: print(f"[row] WARNING: {len(foreign)}/{len(predicted)} {flag} video_ids are absent from "
                      f"--split {split}; scoring only the {len(predicted) - len(foreign)} that match.", flush=True)
    return {vid: evs for vid, evs in predicted.items() if vid in gold_ids}


def _check(args: argparse.Namespace, seg_token: str | None) -> dict:
    """Every refusal of a row, before any record load. Returns the provenance of a --spans file ({} for a span source that --spans names)."""
    # A model file names a model that the row builds, else the row would ignore it and score another model.
    reads = {"segmenter_checkpoint": args.spans in SEGMENTERS, "arm_checkpoint": args.spans in ARMS or args.text in ARMS,
             "pretrimmed_checkpoint": args.text == "pretrimmed"}
    unread = [f"--{key.replace('_', '-')}" for key, read in reads.items() if getattr(args, key, None) and not read]
    if unread: raise SystemExit(
        f"{' and '.join(unread)}: this row builds no such model. The segmenter file goes with --spans moryossef or s1, an arm's file "
        f"with --spans or --text ar or dlm, and the pretrimmed translator's file with --text pretrimmed."
    )
    if args.predictions:
        if any((args.offline, args.online, args.spans, args.text, args.decode)): raise SystemExit(
            "--predictions scores a saved events file again as it is: drop --offline/--online, --spans, --text and --decode.")
        if args.output: raise SystemExit("--predictions scores a saved file again and writes no file: drop --output.")
        if getattr(args, "no_densevid", False): raise SystemExit(
            "--predictions prints the main scores only and computes no densevid score (the row's _scores.json holds them, key densevid): "
            "drop --no-densevid.")
        if args.source_language: raise SystemExit("--source-language names the corpus of the models that a row builds; --predictions builds none.")
        return {}
    if not args.spans or not args.text: raise SystemExit(
        f"A row needs --spans ({', '.join(SOURCES)}, or a span or events file) and --text (none, pretrimmed, ar or dlm), with --offline or "
        f"--online; --predictions scores a saved events file again. The boundary sweep is analyze.py --stage boundary-sweep."
    )
    source, src, when, path = source_language(args), {}, "online" if args.online else "offline", Path(args.spans)
    if args.spans not in SOURCES:
        if not path.is_file(): raise SystemExit(f"--spans {args.spans}: no such file, and not one of {', '.join(SOURCES)}.")
        src = json.loads(path.read_text(encoding="utf-8")).get("provenance") or {}
        if not src.get("segmenter_arch") or not src.get("decode"): raise SystemExit(
            f"--spans {args.spans} carries no segmenter_arch/decode provenance, so this row can't name itself. The events file of every "
            f"row (eval.py) names both: run the row that made these spans again."
        )
        if src["decode"] == "plain": raise SystemExit(
            f"--spans {args.spans} names the decode 'plain', which no row names now (offline it is argmax, online viterbi): run the row "
            f"that made these spans again."
        )
        # Spans an FSM committed keep the name `online`: run_cascade clamps each crop (and an arm's body box) to the frames the buffer
        # held at commit time, so translating them again moves the translator alone. The events keep their commit times.
        when, given = "online" if src.get("when") == "online" else "offline", "online" if args.online else "offline" if args.offline else None
        if given not in (None, when): raise SystemExit(f"--spans {args.spans} holds {when} spans (its provenance): pass --{when}, or neither.")
        if args.decode: raise SystemExit(f"--spans {args.spans} keeps the decode that made its spans ({src['decode']}): drop --decode.")
        # Spans made by a model stamp the corpus it trained on (model-free gold and chance spans stamp none). The row's name carries one
        # source, so its spans and its translator must share it.
        if src.get("source_language") not in (None, source): raise SystemExit(
            f"--spans {args.spans} holds spans of models trained on {src['source_language']!r}, but this row's models are {source!r}'s "
            f"(--source-language, default --language). Pass --source-language {src['source_language']}."
        )
    else:
        if not (args.offline or args.online): raise SystemExit(f"--spans {args.spans}: pass --offline (whole videos) or --online (the streaming FSM).")
        if args.online and args.spans in ("gold", "chance"): raise SystemExit(f"--spans {args.spans} is offline only: no FSM commits these spans.")
        if args.spans == "gold" and args.text == "none": raise SystemExit(
            "--spans gold --text none would score the gold units against themselves: give them a translator (--text pretrimmed, ar or dlm).")
        if args.decode and args.spans not in SEGMENTERS and not (args.spans in ARMS and args.decode == "duration"): raise SystemExit(
            f"--decode {args.decode}: the {args.spans} arm's head always decodes with its trained duration model, which its gate reads too."
            if args.spans in ARMS else f"--spans {args.spans} takes no --decode: no model decodes these spans."
        )
    if args.spans in ARMS and args.text in ARMS and args.text != args.spans: raise SystemExit(
        f"--spans {args.spans} --text {args.text}: the head of one arm with the translator of the other is no row. Each arm reads its own "
        f"spans (--spans {args.text} --text {args.text}), and the pretrimmed translator reads any arm's spans (--text pretrimmed)."
    )
    init = args.segmenter_init
    if init != "finetuned" and not (args.spans == "moryossef" and args.text == "none"): raise SystemExit(
        f"--segmenter-init {init} is an untrained control of the Moryossef segmenter: score its spans alone, with --spans moryossef "
        f"--text none. A row with text reads the fine-tuned segmenter."
    )
    # A stage-2 arm as --spans s1 (its head in the S1 layout, models.checkpointing.s1_layout_state) would write S1's rows. The rows of an
    # arm's head read the whole arm and carry the arm's name.
    arm = load_checkpoint_meta(args.segmenter_checkpoint) if args.spans == "s1" and args.segmenter_checkpoint else {}
    if arm.get("architecture") in ("two_stream_slt", "one_stream_slt"): raise SystemExit(
        f"--segmenter-checkpoint {args.segmenter_checkpoint} is a stage-2 arm: score its head with --spans {arm.get('decoder')} "
        f"--arm-checkpoint {args.segmenter_checkpoint}, whose rows carry the arm's name. As --spans s1 its spans would take the name of S1's rows."
    )
    fsm_text = args.online and ((args.spans in SEGMENTERS and args.text == "pretrimmed") or (args.spans in ARMS and args.text == args.spans))
    if args.stability and not fsm_text: raise SystemExit(
        "--stability records the decodes of the FSM: it needs an online row whose FSM translates (--spans moryossef or s1 with --text "
        "pretrimmed, or an arm with its own text)."
    )
    if args.source_language and args.text == "none" and args.spans not in SEGMENTERS + ARMS and not src.get("source_language"):
        raise SystemExit("--source-language names the corpus of the models that a row builds; this row builds none.")
    # --output names the events file (its _scores.json goes beside it). A zero-shot row and an untrained control keep their token in
    # it, so report.py never reads them as an in-corpus or a fine-tuned row.
    if args.output:
        missing = [t for t in (corpus_token(args.language, source) if source != args.language else None,
                               seg_token if init != "finetuned" else None) if t and t not in Path(args.output).name]
        if missing: raise SystemExit(
            f"--output {args.output}: put {' and '.join(map(repr, missing))} in its file name, so this row never takes the name of another."
        )
    # The main score is the F1 of the tIoU x edge-limit grid, so --tiou-thresholds moves it, and the row's name holds no grid: a row on
    # other thresholds than eval.yaml scoring.tiou_thresholds writes to --output, so the default name always holds the eval.yaml grid.
    if args.tiou_thresholds and not args.output:
        grid = load_yaml(args.eval_config)["scoring"]["tiou_thresholds"]
        if parse_grid(args.tiou_thresholds, grid) != parse_grid(None, grid): raise SystemExit(
            f"--tiou-thresholds {args.tiou_thresholds}: the main score is the F1 of the tIoU x edge-limit grid, so thresholds other than "
            f"eval.yaml scoring.tiou_thresholds ({', '.join(f'{float(t):g}' for t in grid)}) give another main score under the same row "
            f"name. Give --output, so this row never takes the name of the row on the eval.yaml grid."
        )
    if src:   # a file's spans: never an arm on the spans of its own head, never the file itself
        token = _text_token(args, source)
        stem = output_stem(when, src["segmenter_arch"], src["decode"], token, corpus_token(args.language, source), args.split)
        if src["segmenter_arch"] == f"joint-{token}": raise SystemExit(
            f"--spans {args.spans} holds the spans of this arm's own head: the arm on them is its own --{when} row ({stem}), not a "
            f"cascade. The same-span control reads them with --text pretrimmed."
        )
        if (Path(args.output) if args.output else output_dir(when) / f"{stem}.json").resolve() == path.resolve(): raise SystemExit(
            f"--spans {args.spans} would be overwritten by its own row: the spans and the text are both this file's."
        )
        if args.text == "none" and not args.output: raise SystemExit(
            f"--spans {args.spans} --text none: give --output. Its default name, {stem}, is the no-text row of the pass that made these "
            f"spans, and this row would replace it and lose its frame scores."
        )
    return src


def run_row(args: argparse.Namespace) -> "pd.DataFrame":
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run a row on test without --allow-test")
    # The spans field of a standalone segmenter's row: an untrained control adds its init (and seed), so it never takes the name of
    # the fine-tuned segmenter's row.
    init = args.segmenter_init
    seg_token = args.spans if init == "finetuned" else f"{args.spans}-{init}" + (str(args.seed) if init == "random" else "")
    src = _check(args, seg_token)
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    # The models trained on `source` (--source-language, default --language); the data, gold and text scoring are --language's.
    source = source_language(args)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    thresholds = parse_grid(args.tiou_thresholds, eval_cfg["scoring"]["tiou_thresholds"])
    # The evaluated corpus's scoring level (per character or word; lowercase both sides when data.yaml sets lowercase_references).
    level = text_level(target_language(data_cfg, args.language), cfg_get(data_cfg, "languages", args.language, "lowercase_references"))
    corpora = corpus_token(args.language, source)   # the corpus field of the row's name: both corpora for a zero-shot row
    events_path, frame = None, None   # set by a row that writes events; a --predictions re-score writes no new artifact
    if args.predictions: predicted = _saved_events(args.predictions, records, "--predictions", args.split)
    else:
        # The row's own model: its translator, else the arm that cuts its spans, else the pretrimmed translator (its method config is read).
        method = TEXT_METHODS.get(args.text) or (args.spans if args.spans in ARMS else "baseline")
        method_cfg = load_yaml(method_config_path(args, method), language=source)
        if getattr(args, "num_beams", None): method_cfg.setdefault("validation", {})["num_beams"] = int(args.num_beams)
        provenance = {
            "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(records), "method": method,
            "num_beams": int(method_cfg["validation"]["num_beams"]) if method == "baseline" else 1,
            "language": args.language, "source_language": source, "split": args.split, "checkpoint": None,
            "gate": bool(method_cfg["membership_gate"]["enabled"]), "text_level": asdict(level),
        }
        score_cfg = load_yaml(args.inference_config)
        # The decode stamp follows the row's translator and its decoder: a pretrimmed translator's is the one its file stamps, the file
        # that build_eval_model loads (--pretrimmed-checkpoint, else data.yaml pretrained_slt). A row with no text stamps its arm's.
        translator_file = (args.pretrimmed_checkpoint or resolve_pretrained(method_cfg, data_cfg, source, default="")) \
            if args.text == "pretrimmed" else None
        provenance.update(decode_provenance(translator_decoder(method, translator_file), method_cfg))
        # A deliberate beam override changes the search budget and cannot enter a main row delta.
        if args.text == "pretrimmed" and int(provenance["num_beams"]) != 1: print(
            f"[row] WARNING: the pretrimmed translator decodes with num_beams={provenance['num_beams']} while the ar/dlm arms are greedy — "
            f"this row is not search-budget matched.", flush=True
        )
        seg = args.spans in SEGMENTERS
        # The text that the span pass gives itself: an arm's own translator, or online the pretrimmed translator at a segmenter's commits (the
        # online cascade). The cascade translates the spans for every other text; its span pass reads no --method-config (that names the
        # config of the row's translator).
        native = (args.spans in ARMS and args.text == args.spans) or (args.online and seg and args.text == "pretrimmed")
        cascade = args.text != "none" and not native
        span_args = argparse.Namespace(**{**vars(args), "text": args.text if native else "none", **({"method_config": None} if cascade else {})})
        span_prov = {"segmenter_init": init, "min_span_s": lambda_min_s(score_cfg), "method": "baseline", "gate": False, "num_beams": 1,
                     **({"seed": int(args.seed)} if init != "finetuned" else {})} if seg else {}
        text = "none"   # the text field of a row with no text; a pass with its own text and the cascade name theirs
        if src:   # a file's spans; with no text, the spans alone, and the file's provenance describes them
            when, spans, decode = "online" if src.get("when") == "online" else "offline", src["segmenter_arch"], src["decode"]
            predicted = _saved_events(args.spans, records, "--spans", args.split)
            if not cascade: predicted = {vid: [replace(ev, text=None) for ev in evs] for vid, evs in predicted.items()}
            span_prov = src if cascade else {"segmenter_arch": spans, "decode": decode, "when": when, "translate": False, "span_source": src}
        elif args.spans == "gold":   # the reliable gold units, offline; --text none is refused
            when, spans, decode, predicted = "offline", "gold", "none", gold_events(records)
            span_prov = {"segmenter_arch": "gold", "decode": "none", "when": when}
        elif args.spans == "chance":   # model-free spans of the median train unit length (an appendix row)
            train, _ = load_language_records(data_cfg, args.language, split="train")
            (predicted, span_s), when, spans, decode = chance_spans(records, train), "offline", "chance", "none"
            span_prov = {"segmenter_arch": "chance", "decode": "none", "when": when, "translate": False, "span_s": span_s,
                         "train_annotation_fingerprint": annotation_fingerprint(train)}
        elif args.online:
            predicted = run_online(span_args)
            arm_file = None if seg else run_online.last_arm_checkpoint   # the joint arm actually loaded
            # The arm's own token (config stem, variant stamps) also names its spans, so its same-span controls get their own files.
            when, spans, decode = "online", seg_token if seg else f"joint-{translator_token(span_args, args.spans, arm_file)}", run_online.last_decode
            # The cap cut's window as the online rows run it, in buffer time: it follows from the cap, the stride and K
            # (infer.commit_gate.cap_cut_window). The offline rows have no buffer cap.
            cap_cut = dict(zip(("window_start_s", "window_end_s"), cap_cut_window(
                float(score_cfg["buffer_cap_s"]), float(score_cfg["stride_s"]), int(score_cfg["boundary_stability"]["hysteresis_strides"]))))
            span_prov.update({
                "checkpoint": run_online.last_checkpoint if seg else arm_file, "segmenter_arch": spans, "decode": decode, "translate": native,
                "pose_normalization": "buffer", "duration_model": run_online.last_duration_model,
                "segmentation_decode": run_online.last_segmentation_decode, "velocity": run_online.last_velocity,
                # The FSM's commit geometry as run (the live inference.yaml tolerance).
                "delta_s": float(score_cfg["boundary_stability"]["delta_s"]),
                "online_geometry": {
                    "buffer_cap_s": score_cfg["buffer_cap_s"], "stride_s": score_cfg["stride_s"],
                    "boundary_stability": score_cfg["boundary_stability"], "min_span_s": lambda_min_s(score_cfg), "cap_cut": cap_cut,
                    "segmentation_decode": run_online.last_segmentation_decode,
                },
                "when": "online",
                "compute": run_online.last_compute,   # wall time per stride on this device; not a score
                "model_annotation": getattr(run_online, "last_model_annotation", None),   # the translator's labels
            })
            if seg: span_prov["streaming_adaptation"] = "shared_fsm"
            if seg and native: span_prov.update(translation_checkpoint=run_online.last_translation_checkpoint, translation_pose_normalization="span")
            # An online cascade is named after the pretrimmed translator it translates with (pretrimmed or pretrimmed-dlm).
            text = translator_token(span_args, "baseline" if seg else args.spans,
                                    run_online.last_translation_checkpoint if seg and native else arm_file, translate=native)
            frame = run_online.last_frame
        else:
            predicted = run_offline(span_args)
            arm_file = None if seg else run_offline.last_checkpoint   # the arm actually loaded
            when, spans, decode = "offline", seg_token if seg else f"joint-{translator_token(span_args, args.spans, arm_file)}", run_offline.last_decode
            span_prov.update({
                "checkpoint": run_offline.last_checkpoint, "model_annotation": getattr(run_offline, "last_model_annotation", None),
                "segmenter_arch": spans, "decode": decode, "when": "offline", "translate": native,
                "duration_model": run_offline.last_duration_model, "segmentation_decode": run_offline.last_segmentation_decode,
                "velocity": run_offline.last_velocity,
                # The arm's head and S1 read the Uni-Sign body box of each head chunk; the Moryossef segmenter its release contract.
                "pose_normalization": "video" if args.spans == "moryossef" else "chunk",
                # Λ_min is the offline row's GENERATION floor (run_offline drops shorter proposals). Scoring applies no floor, and the
                # cascade takes the spans as given and declares none, so the stamp records how the spans were made.
                "generation_min_span_s": lambda_min_s(score_cfg),
            })
            if seg: span_prov["segmenter_inputs"] = run_offline.last_segmenter_inputs
            if native:
                span_prov["caption_context"] = "full_proposal_with_context"
                span_prov["translation_pose_normalization"] = "commit_box"   # run_offline: one body box over the estimated commit buffer
            text = translator_token(span_args, "baseline" if seg else args.spans, arm_file, translate=native)
            frame = run_offline.last_frame
        if cascade:   # the translator of --text on the spans: the events keep their intervals, commit times and flags
            predicted = run_cascade(args, predicted)
            provenance.update(
                checkpoint=run_cascade.last_checkpoint, model_annotation=getattr(run_cascade, "last_model_annotation", None),
                translation_pose_normalization="span" if method == "baseline" else "commit_box",
                segmenter_arch=spans, decode=decode, when=when, span_source=span_prov,
            )
            # Named after the run: the text token reads the stamp of the file the cascade actually loaded.
            text = translator_token(args, method, run_cascade.last_checkpoint)
        else: provenance.update(span_prov)
        if src:   # the events keep the file's own annotation stamp, so a reader of this row also warns when those spans are stale
            provenance.update(spans_file=args.spans, annotation_fingerprint=stamped_annotation(args.spans))
        stem = output_stem(when, spans, decode, text, corpora, args.split)
        events_path = Path(args.output) if args.output else output_dir(when) / f"{stem}.json"
        write_events_json(predicted, events_path, provenance)

    gold = gold_events(records)
    # The main scores (metrics.edge_chunk_scores): per video 1 tIoU match of the captions with the runs of units inside 1 chunk (eval.yaml
    # scoring.chunk_gap_s); a pair counts in the cell (tau, t) when its tIoU >= t (scoring.tiou_thresholds, or --tiou-thresholds) and both
    # its edges are within tau (scoring.edge_taus_s). Per cell the F1, and the DVC and the SODA-style text F1 of the pairs that count; the
    # main score is the F1 at tau avg and tIoU avg. The densevid scores read the one-caption-per-unit tIoU pairing of densevid_eval at the
    # same thresholds: the sentence tIoU F1 and its DVC.
    gap_s, taus = float(eval_cfg["scoring"]["chunk_gap_s"]), [float(t) for t in eval_cfg["scoring"]["edge_taus_s"]]
    # Never cut (the method rule): each committed span is one event, one caption with its whole text.
    predicted, chunks = scoreable_predictions(predicted, records), gold_chunks(records, gap_s)
    edge = edge_chunk_scores(predicted, chunks, taus, thresholds, level)
    uncaptioned = uncaptioned_units(predicted, gold)
    no_text = args.text == "none"
    if events_path is not None:   # a --predictions re-score writes no file, so it computes no densevid score
        densevid = not bool(getattr(args, "no_densevid", False)) and not no_text   # a no-text row has no text to pair
        # 1 layout for every row, the main scores first; densevid has the shape of the main block (by_tiou, avg), its dvc None with no
        # text or --no-densevid. A no-text row also keeps the frame and run scores of the Moryossef protocol, a mean over the streams with
        # gold signing, in 1 key set (metrics.frame_scores): the offline whole-video pass of a segmenter (run_offline) or the FSM's
        # stitched tags (run_online). None: not computed.
        scores = {"provenance": provenance, "main": {"edge_chunk": edge, "uncaptioned": uncaptioned},
                  "densevid": evaluate_predicted_events(predicted, gold, thresholds, char_level=level, densevid=densevid)}
        if no_text: scores["frame"] = frame
        scores_path = events_path.with_name(f"{events_path.stem}_scores.json")
        scores_path.write_text(json.dumps(scores, indent=2, default=float) + "\n", encoding="utf-8")
    # The screen report of every row: the main score first and alone (the title line and the table that run_row returns), then its notes;
    # the secondary scores are only named.
    if level.lowercase: print(f"[row] every text is scored in lowercase (data.yaml languages.{args.language}.lowercase_references)", flush=True)
    print(f"[row] MAIN score: F1 {edge['avg']['f1']:.4f} at edge tau avg and tIoU avg (edge chunk match: a caption and a run of units "
          f"inside 1 chunk pair 1 to 1 by tIoU; a pair counts at tIoU t when its tIoU >= t and at edge tau when both its edges are within "
          f"tau; an avg is the mean of its cells, and an avg F1 is the F1 of the mean precision and recall; edge tau 'no limit' is the tIoU "
          f"test alone, shown for reference and in no average" + (
          "; DVC = the DVC of densevid_eval on the pairs that count, SODA = the SODA-style text F1 on the same pairs"
          if edge["avg"]["dvc"] else "") + f"). {edge['captions']} captions, {edge['units']} reliable units in {edge['chunks']} chunks "
          f"(caption gaps up to {gap_s:g} s)", flush=True)
    table = _main_table(edge)
    table.attrs["notes"] = [
        f"[row] units with no caption: {uncaptioned['uncaptioned']} of {uncaptioned['units']} ({uncaptioned['share']:.1%}); no caption "
        f"overlaps them",
        f"[row] densevid scores (one caption per unit; secondary, not printed): key densevid of {scores_path}" if events_path is not None
        else f"[row] densevid scores (one caption per unit; secondary, not printed): a re-score writes no file; its row wrote them in "
        f"{Path(args.predictions).with_name(Path(args.predictions).stem + '_scores.json')} (key densevid, and frame for a no-text row)",
        *([f"[row] frame scores of the Moryossef protocol (secondary, not printed): key frame of {scores_path}" +
           ("" if frame else " (null: this pass computes none)")] if events_path is not None and no_text else []),
    ]
    return table


def _main_table(edge: dict) -> "pd.DataFrame":
    """A row's main scores on screen (metrics.edge_chunk_scores), the grid of edge tau x tIoU. Rows (score, edge tau (s)): precision,
    recall, F1, then, when the captions have text, the DVC (TEXT_KEYS) and the SODA-style text F1 (SODA_KEYS) of the pairs that count,
    each at every edge tau, their avg, and "no limit" (the tIoU test alone: a reference row in no average). Columns (tIoU): the
    thresholds and their avg. The main score is the F1 at edge tau avg and tIoU avg."""
    n = len(edge["tiou_thresholds"])
    lines = {f"{tau:g}": edge["cells"][i * n:(i + 1) * n] + [edge["by_tau"][i]] for i, tau in enumerate(edge["taus_s"])}   # tau-major cells
    lines.update({"avg": edge["by_tiou"] + [edge["avg"]], "no limit": edge["no_edge_limit"]["by_tiou"] + [edge["no_edge_limit"]["avg"]]})
    def values(r): return {"precision": r["precision"], "recall": r["recall"], "F1": r["f1"],
                           **{f"DVC {TEXT_NAME[k]}": v for k, v in (r["dvc"] or {}).items()},
                           **{f"SODA {TEXT_NAME[k]}": v for k, v in (r["soda"] or {}).items()}}
    scores = list(values(edge["avg"]))
    return pd.DataFrame(
        [[values(r)[s] for r in line] for s in scores for line in lines.values()],
        index=pd.MultiIndex.from_product([scores, list(lines)], names=["score", "edge tau (s)"]),
        columns=pd.Index([f"{t:g}" for t in edge["tiou_thresholds"]] + ["avg"], name="tIoU"),
    ).round(4)
