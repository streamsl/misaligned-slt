"""Candidate unit rules replayed on saved artifacts and scored against the deployed rule (error analysis; no deployed event changes).

--resplit cuts split spans of a events file again by a model-evidence rule of infer/unit_split.py and scores each version;
--separator-replay replays the text-side unit commit of infer/unit_split.py on saved open-span decodes and scores it against 
gold and the same run's deployed commits.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np

import torch
from data.loader import load_language_records
from infer.unit_split import attention_recut, fit_duration_prior, likelihood_recut, own_span_attention, separator_commits
from models.checkpointing import load_checkpoint_meta

from metrics import char_level_for_target
from evaluation.predictions import (
    _gold_events, _write_events_json, joined_events, load_event_predictions, scoreable_predictions, warn_annotation_mismatch
)
from evaluation.scoring import cut_errors, dvc_by_video, evaluate_predicted_events, headline_f1_bootstrap, paired_bootstrap
from evaluation.systems import (
    _box_normalized_window, _build_eval_model, _commit_box_s, _generation_kwargs, METHOD_CONFIGS, model_annotation
)
from utils import FRAME_SLACK, lambda_min_s, load_yaml, pick_device, target_language


def _clean_floor(args: argparse.Namespace, data_cfg: dict, inference_cfg: dict, device, role: str):
    # (model, tokenizer, its train labels) of the scorer of a re-split row: THIS language's trained clean AR floor.
    model, tokenizer = _build_eval_model(
        "baseline", args.checkpoint, args.language, data_cfg,
        load_yaml(args.method_config or METHOD_CONFIGS["baseline"], language=args.language), inference_cfg, device
    )
    model.eval()
    if load_checkpoint_meta(model.checkpoint_path).get("architecture") != "clean_translation": raise SystemExit(
        f"{model.checkpoint_path}: the {role} scorer must be the {args.language!r} trained clean AR floor "
        f"(checkpoints/baseline_train/{args.language}/model.pt); pass it with --checkpoint")
    return model, tokenizer, model_annotation(model, data_cfg, args.language)


def _own_attention_reader(args: argparse.Namespace, data_cfg: dict, inference_cfg: dict, device):
    """(model, tokenizer, its train labels, attend) for --resplit own-attention: the gated AR arm that wrote the events (--checkpoint)
    reads its own cross-attention on each split span, as its `--segments` row reads a span (evaluation.rq2.run_cascade): the crop
    [start, end + 1 frame] cut to the frames held at the commit, normalized with ONE body box over the buffer that commits the span
    (_commit_box_s), and the gate anchored on the span from frame 0 to its terminator frame (-1 when the crop ends before it)."""
    method_cfg = load_yaml(args.method_config or METHOD_CONFIGS["ar"], language=args.language)
    model, tokenizer = _build_eval_model("ar", args.checkpoint, args.language, data_cfg, method_cfg, inference_cfg, device)
    model.eval()
    lead_s, tail_s = _commit_box_s(inference_cfg)

    def attend(rec, start: float, end: float, group):
        off, fps = float(rec.pose.offset_s), float(rec.pose.fps)
        held = float(rec.pose.duration_s)
        if group[0].commit_time_s is not None: held = min(held, float(group[0].commit_time_s) - off - 1e-6)
        span_end = float(group[-1].end_s) - off
        poses, ts = _box_normalized_window(rec.pose, start, end, start - lead_s, min(held, span_end + 1.0 / fps + tail_s))
        term = int(np.searchsorted(ts, span_end - FRAME_SLACK / fps))
        read = {k: v for k, v in _generation_kwargs("ar", inference_cfg, method_cfg, 0, [fps]).items() if k.startswith("gate_")}
        read.update(
            gate_stream_start=True, gate_anchor=torch.tensor([[0, term if term < len(ts) else -1]], dtype=torch.long, device=device)
        )
        rel = np.asarray(ts, dtype=np.float32) - float(start)   # window-relative, as _translate_windows prepares a row
        return own_span_attention(model, tokenizer, poses, rel, [e.text or "" for e in group], device, read)

    return model, tokenizer, model_annotation(model, data_cfg, args.language), attend


def run_resplit(args: argparse.Namespace) -> dict:
    """--resplit likelihood|attention|own-attention: every split span of a RQ2 events file (--events) cut again by a model-evidence 
    rule (infer.unit_split). likelihood and attention read ONE common scorer, the language's trained clean AR floor (--checkpoint), 
    for every arm; own-attention reads the gated AR arm that wrote the events (--checkpoint). Prints F1 avg and DVC BLEU-4 avg for 
    joined spans, word-share split (the method rule) and the re-cut, and the paired video bootstrap of re-cut minus the word share 
    (eval.yaml split_rule). The attention cut also gives the cut error on spans that hold as many gold units as parts. Writes re-cut 
    events (--output, else <events>_<rule>-cut.json beside events file), which `report.py results` reads as a system, and the summary 
    beside them (_scores.json). --fit-layer (attention, train only) instead writes the mean cut error of every candidate layer set 
    (_layers.json): the rule that sets eval.yaml attention_cut.layers."""
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to re-split on test without --allow-test")
    if args.fit_layer and (args.resplit != "attention" or args.split != "train"): raise SystemExit(
        "--fit-layer chooses the attention-cut layers on train only: --resplit attention --split train"
    )
    if not args.events: raise SystemExit("--resplit needs --events, a RQ2 events file")

    data_cfg, inference_cfg, eval_cfg = load_yaml(args.data_config), load_yaml(args.inference_config), load_yaml(args.eval_config)
    boot = (int(eval_cfg["split_rule"]["bootstrap_resamples"]), int(eval_cfg["split_rule"]["bootstrap_seed"]))
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    warn_annotation_mismatch(args.events, records, "events file")
    gold, raw = _gold_events(records), load_event_predictions(args.events)
    device = pick_device(args.device)
    attend = None

    if args.resplit == "own-attention":
        model, tokenizer, scorer_labels, attend = _own_attention_reader(args, data_cfg, inference_cfg, device)
        wrote = json.loads(Path(args.events).read_text(encoding="utf-8")).get("provenance", {}).get("checkpoint")
        if wrote is None: print(
            f"[resplit] WARNING: {args.events} names no checkpoint; own-attention assumes "
            f"{model.checkpoint_path} wrote its texts.", flush=True
        )
        elif Path(wrote).resolve() != Path(model.checkpoint_path).resolve(): raise SystemExit(
            f"--resplit own-attention reads the arm that wrote the events: {args.events} was written by {wrote}, "
            f"not by --checkpoint {model.checkpoint_path}"
        )
    else: model, tokenizer, scorer_labels = _clean_floor(args, data_cfg, inference_cfg, device, args.resplit)
    out = Path(args.output) if args.output else Path(args.events).with_name(f"{Path(args.events).stem}_{args.resplit}-cut.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    seconds = lambda c: {"seconds_mean": round(float(np.mean(c["seconds"])), 2) if c["seconds"] else 0.0,
                         "seconds_max": round(float(np.max(c["seconds"])), 2) if c["seconds"] else 0.0}
    summary = {"rule": args.resplit, "scorer": str(model.checkpoint_path), 
               "scorer_annotation": scorer_labels, "events": str(args.events)}

    def paired(a, b):   # mean cut error a - b (the same cuts in two versions) with its 95 % CI, videos resampled
        r = paired_bootstrap(np.array([e for _, e in a]), np.array([e for _, e in b]), [v for v, _ in b], *boot)
        return [r["mean_delta"], *r["ci95"]]

    if args.resplit == "likelihood":
        prior = fit_duration_prior(load_language_records(data_cfg, args.language, split="train")[0])
        recut, cost = likelihood_recut(
            raw, records, model, tokenizer, prior, device, float(inference_cfg["boundary_stability"]["delta_s"]),
            lambda_min_s(inference_cfg), eval_cfg["likelihood_cut"]
        )
        summary["prior"] = dict(zip(("a_s", "b_s_per_word", "sigma_s"), (round(x, 3) for x in prior)))
        summary["cost"] = {
            "split_spans": cost["spans"], "unmatched_spans": cost["unmatched"],
            "crop_encodes_mean": round(float(np.mean(cost["encodes"])), 1) if cost["encodes"] else 0.0, **seconds(cost)
        }
        split_note = {"unit_split": "likelihood + length prior (eval.py --resplit likelihood)", "length_prior": summary["prior"]}
    else:
        cfg = eval_cfg["attention_cut"]
        if args.fit_layer:
            # The decoder layer sets (1-based) that --fit-layer compares, fixed before the train run: each layer alone, 
            # the last half (Whisper's default alignment heads) and all layers.
            n = int(model.front_end.lm_model.config.num_decoder_layers)
            sets = [[l] for l in range(1, n + 1)] + [list(range(n // 2 + 1, n + 1)), list(range(1, n + 1))]
        else: sets = [[int(l) for l in cfg["layers"]]]
        recuts, cost = attention_recut(
            raw, records, model, tokenizer, device, lambda_min_s(inference_cfg), sets, float(cfg["median_half_width_s"]), attend
        )
        word_err, errs = cut_errors(raw, gold), [cut_errors(ev, gold) for ev in recuts]
        mean_err = lambda err: round(float(np.mean([e for _, e in err])), 3) if err else None
        summary["cost"] = {
            "split_spans": cost["spans"], "unmatched_spans": cost["unmatched"], "too_short_spans": cost["too_short"], **seconds(cost)
        }
        if args.fit_layer:
            best, runner_up = sorted(range(len(sets)), key=lambda s: np.mean([e for _, e in errs[s]]) if errs[s] else np.inf)[:2]
            summary.update({
                "boundaries": len(word_err), "word_share_cut_error_s": mean_err(word_err),
                "cut_error_s": {",".join(map(str, s)): mean_err(err) for s, err in zip(sets, errs)}, 
                "best_layers": sets[best], "best_minus_word_share_s": paired(errs[best], word_err), 
                "best_minus_runner_up_s": paired(errs[best], errs[runner_up]), "runner_up_layers": sets[runner_up],
                "floored_spans": dict(zip((",".join(map(str, s)) for s in sets), cost["floored"]))
            })
            out.with_name(f"{out.stem}_layers.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
            print(json.dumps(summary, indent=2), flush=True)
            return summary
        
        recut = recuts[0]
        summary.update({
            "layers": sets[0], "cut_error_s": {
                args.resplit: mean_err(errs[0]), "word share": mean_err(word_err), 
                "boundaries": len(word_err), f"{args.resplit}_minus_word_share": paired(errs[0], word_err)
            },
            "layers_chosen_by": "eval.py --resplit attention --fit-layer on train (configs/eval.yaml attention_cut)",
            "median_half_width_s": float(cfg["median_half_width_s"]), "floored_spans": cost["floored"][0]
        })
        split_note = {
            "unit_split": f"cross-attention DTW (eval.py --resplit {args.resplit})", 
            "attention_layers": sets[0], "median_half_width_s": summary["median_half_width_s"]
        }
    thresholds = [float(t) for t in eval_cfg["rq2"]["tiou_thresholds"]]
    char_level = char_level_for_target(target_language(data_cfg, args.language))
    versions = {
        name: scoreable_predictions(ev, records, tag=name)
        for name, ev in (("joined", joined_events(raw)), ("word share", raw), (args.resplit, recut))
    }
    summary["rows"] = {}
    for name, ev in versions.items():
        avg = evaluate_predicted_events(ev, gold, thresholds, char_level=char_level, densevid=True)["threshold_average"]
        summary["rows"][name] = {
            "f1_avg": round(float(avg["segmentation_f1"]), 4), "dvc_bleu4_avg": round(float(avg["densevid_bleu4"]), 2)
        }
    f1 = headline_f1_bootstrap(versions[args.resplit], versions["word share"], gold, thresholds, *boot)
    vids, dvc_word = dvc_by_video(versions["word share"], gold, thresholds, char_level)
    dvc = paired_bootstrap(dvc_by_video(versions[args.resplit], gold, thresholds, char_level)[1], dvc_word, vids, *boot)
    summary[f"{args.resplit}_minus_word_share"] = {
        "f1_avg": [f1["mean_delta"], *f1["ci95"]], "dvc_bleu4_avg": [dvc["mean_delta"], *dvc["ci95"]]
    }
    provenance = json.loads(Path(args.events).read_text(encoding="utf-8")).get("provenance", {})
    _write_events_json(recut, out, {**provenance, **split_note, "unit_split_scorer": summary["scorer"]})
    out.with_name(f"{out.stem}_scores.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def score_separator_commits(
    replay: dict, gold: dict, deployed: dict, tiou: float = 0.5, precision_min: float = 0.90, false_rate_max: float = 0.05
) -> dict:
    """Scores replayed commits (`infer.unit_split.separator_commits`) on reliable gold units: precision (unit matches a gold unit at 
    tIoU >= `tiou`). False-separator rate on incomplete 1st units (gold unit at the track start still ran past the buffer end); median 
    end error of matched units; and the latency gain (deployed commit of 1st event that holds >= half of matched unit, minus replayed 
    commit). Pass rule, stated before any data: precision >= precision_min, false rate <= false_rate_max, median end error <= delta_s."""
    tiou_of = lambda a, b: max(0.0, min(a[1], b[1]) - max(a[0], b[0])) / max(max(a[1], b[1]) - min(a[0], b[0]), 1e-9)
    delta_s = replay["rule"]["delta_s"]
    rows = []
    for c in replay["commits"]:
        vid, span = c["video_id"], (c["start_s"], c["end_s"])
        units = [(g.start_s, g.end_s) for g in gold.get(vid, [])]
        best = max(units, key=lambda g: tiou_of(span, g), default=None)
        first = min((g for g in units if g[0] - delta_s <= span[0] < g[1]), key=lambda g: abs(g[0] - span[0]), default=None)
        row = {
            "video_id": vid, "start_s": span[0], "end_s": span[1], "commit_time_s": c["commit_time_s"], 
            "text": c["text"], "matched": bool(best is not None and tiou_of(span, best) >= tiou),
            "first_unit_incomplete": bool(first is not None and first[1] > c["buffer_end_s"])
        }
        if row["matched"]:
            row["end_error_s"] = abs(span[1] - best[1])
            need = 0.5 * (best[1] - best[0])
            held = [
                e.commit_time_s for e in deployed.get(vid, []) 
                if e.commit_time_s is not None and min(e.end_s, best[1]) - max(e.start_s, best[0]) >= need
            ]
            if held: row["latency_gain_s"] = min(held) - c["commit_time_s"]
        rows.append(row)

    n = len(rows)
    med = lambda xs: round(float(np.median(xs)), 3) if xs else float("nan")
    errors = [r["end_error_s"] for r in rows if "end_error_s" in r]
    gains = [r["latency_gain_s"] for r in rows if "latency_gain_s" in r]
    summary = {
        "tracks": replay["tracks"], "tracks_with_separator": replay["tracks_with_separator"], "commits": n,
        "precision": round(sum(r["matched"] for r in rows) / n, 4) if n else float("nan"),
        "false_separator_incomplete_rate": round(sum(r["first_unit_incomplete"] for r in rows) / n, 4) if n else float("nan"),
        "median_end_error_s": med(errors), "median_latency_gain_s": med(gains), "latency_gain_n": len(gains),
        "rule": {**replay["rule"], "tiou": float(tiou), "precision_min": precision_min, "false_rate_max": false_rate_max}
    }
    summary["replay_pass"] = bool(n and summary["precision"] >= precision_min 
                                    and summary["false_separator_incomplete_rate"] <= false_rate_max
                                    and summary["median_end_error_s"] <= delta_s)
    return {"summary": summary, "rows": rows}


def run_separator_replay(args: argparse.Namespace) -> dict:
    """--separator-replay: replay the commits on --shadow-file (the open-span decodes of `--rq 2 --online --shadow-open-spans`) and
    score them against gold and the same run's events (--events). Writes {summary, rows} to --output, else beside the shadow file."""
    if args.split == "test" and not args.allow_test: 
        raise SystemExit("Refusing to replay separator commits on test without --allow-test")
    if not args.shadow_file or not args.events: 
        raise SystemExit("--separator-replay needs --shadow-file and --events (the same run's RQ2 events file)")
    
    records, _ = load_language_records(load_yaml(args.data_config), args.language, split=args.split)
    warn_annotation_mismatch(args.shadow_file, records, "shadow file"); warn_annotation_mismatch(args.events, records, "events file")
    shadow = json.loads(Path(args.shadow_file).read_text(encoding="utf-8"))
    replay = separator_commits(
        shadow["hypotheses"], k=int(shadow["provenance"]["hysteresis_strides"]), delta_s=float(shadow["provenance"]["delta_s"]),
        stride_s=float(shadow["provenance"]["stride_s"]), fps={r.pose.video_id: float(r.pose.fps) for r in records}
    )
    res = score_separator_commits(replay, _gold_events(records), load_event_predictions(args.events))
    out = Path(args.output) if args.output else Path(args.shadow_file).with_name(f"{Path(args.shadow_file).stem}_replay.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(res["summary"], indent=2), flush=True)
    return res["summary"]
