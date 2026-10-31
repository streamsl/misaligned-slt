"""One report per language and split: tables, figures and a Markdown skeleton, from the events files of the rows.

    python report.py results --language asf --split test \\
        --events "true cuts + pretrimmed"=outputs/offline/offline_gold_none_pretrimmed_asf_test.json \\
        --events "Cascade: external segmenter"=outputs/offline/offline_moryossef_argmax_pretrimmed_asf_test.json \\
        --events "joint DLM, online"=outputs/online/online_joint-dlm_duration_dlm_asf_test.json \\
        --reference "joint DLM, online" \\
        --history AR=checkpoints/ar/asf/history.csv --segmentation-scores s1=outputs/offline/offline_s1_duration_none_asf_test_scores.json

Writes outputs/report/results_<language>_<split>/results.md (the generated skeleton; a hand-written report.md beside it is never touched), fig/*.png and tables/*.csv. Every system is scored under one
protocol on one gold index, so all comparisons are paired. Any events file of a row is a system: a cascade row,
the offline row, the online row, an ablation. Row 1 (the true cuts read by the pretrimmed translator, `eval.py --offline --spans gold
--text pretrimmed`) among the systems adds the metric-protocol section. The optional inputs add sections: `--boundary-sweep-file`
(a sweep of the pretrimmed translator, analyze.py --stage boundary-sweep) adds the confidence panel; `--history` adds training
curves; `--segmentation-scores` (the _scores.json of a segmenter's no-text row, eval.py --offline|--online --text none) adds the
segmentation-only comparison; `--stability-file` adds the reveal-policy panel.

    python report.py progress --language asf --language bfi            # one report over every finished experiment (outputs/report/progress)

    python report.py data    --language asf                            # the corpus audit (outputs/report/data)
    python report.py poses   --language asf --video-id <id>            # the pose stream one video feeds the network (outputs/report/poses)
    python report.py predict --language asf --text dlm --video-id <id>     # one model's whole-video outputs (outputs/report/predict)
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

from reporting import figures as F
from reporting.outcomes import (LOCATED, OUTCOME_COLUMNS, boundary_stats, case_studies, edge_recall, flip_table, loose_cell, unit_outcomes,
                                unit_span_classes)
from reporting.protocol import (case_view, decompose, located_pairs, oracle_pairs, oracle_row, showcase_videos, two_video_example,
                                worked_example_video)
from data.loader import corpus_config, load_language_records
from evaluation.predictions import (BOUNDARY_CLASSES, CHUNK_CLASSES, corpus_token, gold_edges, gold_events, chunk_classes, cue_breaks, gold_chunks,
                                    load_event_predictions, scoreable_predictions, train_text_videos, unk_stretches, warn_annotation_mismatch)
from evaluation.systems import warn_model_labels
from models.checkpointing import load_checkpoint_meta
from metrics import (EDGE_CAPTION_TYPES, EDGE_UNIT_TYPES, SODA_KEYS, TEXT_KEYS, TEXT_NAME, cap_cut_captions, caption_shape, text_level,
                     cue_edge_share, edge_boundary_hits, edge_chunk_classes, edge_chunk_scores, edge_error_types, evaluate_predicted_events, held_unit_latency, long_form_scores, match_segments, sentence_bleu_scores,
                     text_scores, uncaptioned_units)
from utils import cfg_get, lambda_min_s, load_yaml, output_dir, seconds_to_frames, target_language


def _vocab_token(word: str) -> str:
    """Lowercase and strip EDGE punctuation before an in-vocabulary test.

    `str.split()` leaves punctuation glued to the word, so "lions." and "Chris." read as unseen types even when the
    bare word is in the training vocabulary. That inflates the reported OOV rate by roughly a factor of two.
    """
    return word.lower().strip(".,!?;:\"'()[]")


def _read_history(path: str) -> list[dict]:
    # A trainer's history.csv, numbers parsed. A non-eval epoch writes "-", which is not a number and is dropped.
    with open(path, newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row: dict = {}
            for key, value in raw.items():
                try: row[key] = float(value)
                except (TypeError, ValueError): pass
            if row: rows.append(row)
    return rows


def _segmentation_only(path, records, scoring) -> dict:
    """The _scores.json of a segmenter's no-text row (eval.py --offline|--online --spans moryossef|s1 --text none) for the segmentation-only
    table, as it is (the layout of every row): `main.edge_chunk` (the main score: the F1 of the tIoU x edge-limit grid, avg; the grid's F1
    per tIoU threshold and per edge limit tau; the reference row of the tIoU test alone; the caption and unit counts), per SOURCE video,
    so the table agrees with the rows, and `frame` (the frame and run scores of the Moryossef protocol, a mean over streams; its
    signing/non-signing F1 `bio_f1` has no counterpart in the rows). `scoring` is the report's eval.yaml scoring block: a row on another
    grid (eval.py --tiou-thresholds, or another eval.yaml) has another main score, so it is refused too."""
    warn_annotation_mismatch(path, records, "--segmentation-scores file")   # a file made under other cuts or units is flagged, then scored
    raw, row = json.load(open(path, encoding="utf-8")), Path(path).name.replace("_scores.json", "")
    # A row made before the current layout: an online frame block with fsm_bio_*, or a main block with no grid of metrics.edge_chunk_scores.
    edge = (raw.get("main") or {}).get("edge_chunk") or {}
    if "bio_f1" not in (raw.get("frame") or {}) or not {"taus_s", "tiou_thresholds", "cells", "by_tau", "by_tiou", "avg", "no_edge_limit"} <= set(edge):
        raise SystemExit(
            f"{path} is not in the current layout (`frame.bio_f1`, and `main.edge_chunk` with the tIoU x edge-limit grid): run its row "
            f"`{row}` again with this code (eval.py --offline|--online --spans moryossef|s1 --text none; the flags are the fields of that "
            f"name), then pass its new _scores.json."
        )
    grid = lambda taus, ths: ([float(t) for t in taus], [float(t) for t in ths])
    if grid(edge["taus_s"], edge["tiou_thresholds"]) != grid(scoring["edge_taus_s"], scoring["tiou_thresholds"]): raise SystemExit(
        f"{path} scores the grid edge tau {edge['taus_s']} s x tIoU {edge['tiou_thresholds']}, not the grid of this report (eval.yaml "
        f"scoring: edge tau {scoring['edge_taus_s']} s x tIoU {scoring['tiou_thresholds']}), so its main score is another average: run "
        f"its row `{row}` again on the eval.yaml grid (no --tiou-thresholds), then pass its new _scores.json."
    )
    return raw


def _segmentation_md(ev: dict, ths) -> str:
    """The segmentation-only table of `results` and `progress` (ev[label] = _segmentation_only): the main score first (the F1 of the
    tIoU x edge-limit grid, avg), the grid's F1 per tIoU threshold `ths` (each over the edge limits) and per edge limit tau (each over
    the thresholds), the reference F1 of the tIoU test alone (no edge limit, avg; in no average), then the frame F1 and the captions
    per reliable unit."""
    taus = list(dict.fromkeys(r["tau_s"] for m in ev.values() for r in m["main"]["edge_chunk"]["by_tau"]))
    def f1_at(m, block, key, x):   # the F1 of 1 row of the grid, "-" when the file has no such row
        f1 = next((r["f1"] for r in m["main"]["edge_chunk"][block] if float(r[key]) == float(x)), None)
        return "-" if f1 is None else f"{f1:.3f}"
    return _table(["segmenter", "F1 avg (main)", *[f"F1, tIoU {t:g}" for t in ths], *[f"F1, edge τ {t:g} s" for t in taus],
                   "F1, tIoU test alone (no edge limit; reference)", "signing/non-signing F1", "sentences produced / true count"],
                  [[l, f"{m['main']['edge_chunk']['avg']['f1']:.3f}", *[f1_at(m, "by_tiou", "tiou", t) for t in ths],
                    *[f1_at(m, "by_tau", "tau_s", t) for t in taus], f"{m['main']['edge_chunk']['no_edge_limit']['avg']['f1']:.3f}",
                    f"{m['frame']['bio_f1']:.3f}", f"{m['main']['edge_chunk']['captions'] / max(1, m['main']['edge_chunk']['units']):.2f}"]
                   for l, m in ev.items()])


def _main_line(edge: dict, uncaptioned: dict) -> str:
    """The console summary of a system (`results` and `progress`): its main score only, the F1 of the tIoU x edge-limit grid at edge tau
    avg and tIoU avg (`edge["avg"]`) with its precision and recall, the avg BLEU-4 of the DVC and of the SODA-style text F1 on the
    counted pairs when the captions have text, and the units with no caption."""
    avg = edge["avg"]
    text = f", DVC BLEU-4 {avg['dvc']['bleu4']:.2f}, SODA-style BLEU-4 F1 {avg['soda']['bleu4']:.2f}" if avg["dvc"] else ""
    return (f"{edge['captions']} captions | MAIN score: F1 {avg['f1']:.3f} at edge tau avg and tIoU avg (precision {avg['precision']:.3f}, "
            f"recall {avg['recall']:.3f}){text} | units with no caption {uncaptioned['uncaptioned']} of {uncaptioned['units']} "
            f"({uncaptioned['share']:.1%})")


def _name(x) -> str:   # a grid label: a tau or a threshold as written in the configs, or "avg" / "no limit" as it is
    return x if isinstance(x, str) else f"{x:g}"


def _grid_rows(edge: dict) -> list[tuple]:
    """The rows of the main grid of `edge` (metrics.edge_chunk_scores) as (edge tau, tIoU, score row), with "avg" for an average and
    "no limit" for the reference rows of the tIoU test alone (in no average): every cell, each tau over the thresholds, each threshold
    over the taus, all cells, then the reference rows. tables/edge_chunk.csv and `_grid_md` read the grid through this one function."""
    free = edge["no_edge_limit"]
    return [*((c["tau_s"], c["tiou"], c) for c in edge["cells"]), *((r["tau_s"], "avg", r) for r in edge["by_tau"]),
            *(("avg", r["tiou"], r) for r in edge["by_tiou"]), ("avg", "avg", edge["avg"]),
            *(("no limit", r["tiou"], r) for r in free["by_tiou"]), ("no limit", "avg", free["avg"])]


def _labelled(specs):
    out = {}
    for spec in specs or []:
        label, sep, path = spec.rpartition("=")
        if not sep or not label: raise SystemExit(f"expected LABEL=PATH, got {spec!r}")
        out[label] = path
    return out


def _attainable_md(c: dict, lang_name: str) -> list[str]:
    # The short-sentence limitation is a GENERATION property: our FSM cannot emit a span below Λ_min, so some gold
    # sentences are unwinnable for us and winnable for a cascade that declares no floor. Scoring applies no floor, so the report states it.
    att = c.get("attainable") or {}
    if not att: return []
    worst = max(att.values(), key=lambda a: a["unreachable"])
    if not worst["unreachable"]: return []
    rows = "; ".join(f"tIoU {t:g}: {a['max_recall']:.3f}" for t, a in sorted(att.items()))
    lo, hi = c["floor_s"]
    span = f"{lo:.3f} s" if hi - lo < 1e-6 else f"{lo:.3f}-{hi:.3f} s"
    out = [f"\nOur online rows may not emit a sentence shorter than Λ_min = {c['lambda_min_s']:g} s ({span} on the streams' "
           f"frame grids). A gold unit of duration *d* can then reach at most *d* / that floor "
           f"tIoU, so it is unwinnable for us above that. Highest recall our policy allows — {rows}. "
           f"The offline cascade rows declare no floor, so their ceiling is 1.000 at every threshold.\n"]
    if worst["unreachable"] / max(1, worst["n_gold"]) >= 0.05:
        out.append(f"\nThis is material for {lang_name}: {worst['unreachable']} of {worst['n_gold']} gold units "
                   f"({100*worst['unreachable']/worst['n_gold']:.0f} %) fall under the strictest threshold's bar. The floor "
                   f"is the fixed label-domain Λ_min (inference.yaml span_selection.min_span_s), not a measurement. **Question for you:** "
                   f"print this ceiling beside the table?\n")
    return out


def attainable_recall(gold, floor_s: dict[str, float], thresholds) -> dict[float, dict]:
    """Highest recall a system with a Λ_min emission floor can reach, per tIoU threshold.

    A span shorter than the floor cannot be produced. Against a gold unit of duration d < floor the best legal
    prediction covers it exactly, so the intersection is d and the union is the floor: tIoU = d / floor. The sentence
    is therefore unreachable at threshold t when d < t * floor. This is a property of the declared policy, not of the
    model, and it belongs beside recall — the offline cascades declare no floor, so their ceiling is 1.0.
    """
    durations = [(g.end_s - g.start_s, float(floor_s[vid])) for vid, v in gold.items() for g in v]   # floor_s per source video
    n = max(1, len(durations))
    out = {}
    for t in thresholds:
        lost = sum(1 for d, floor in durations if d < float(t) * floor)
        out[float(t)] = {"unreachable": lost, "n_gold": len(durations), "max_recall": 1.0 - lost / n}
    return out


def results(args) -> dict:
    systems_paths = _labelled(args.events)
    reference = args.reference or next(iter(systems_paths))
    if reference not in systems_paths: raise SystemExit(f"--reference {reference!r} is not one of {list(systems_paths)}")
    data_cfg = load_yaml(args.data_config)
    inference_cfg = load_yaml(args.inference_config)
    scoring_cfg = load_yaml(args.eval_config)["scoring"]
    thresholds = [float(t) for t in scoring_cfg["tiou_thresholds"]]   # the tIoU columns of the main grid and the densevid thresholds
    edge_taus = [float(t) for t in scoring_cfg["edge_taus_s"]]   # the edge limits of the main grid
    tau = float(args.tiou_threshold)
    rth = [float(t) for t in str(args.report_thresholds).split(",")] if args.report_thresholds else list(thresholds)
    if any(t not in thresholds for t in rth): raise SystemExit(f"--report-thresholds must be among the scored thresholds {thresholds}")
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    gold = gold_events(records)
    chunks = gold_chunks(records, float(scoring_cfg["chunk_gap_s"]))
    n_gold = sum(len(v) for v in gold.values())
    # The corpus's scoring level, as every row and monitor reads it (lowercase both sides when data.yaml sets lowercase_references).
    char_level = text_level(target_language(data_cfg, args.language), cfg_get(data_cfg, "languages", args.language, "lowercase_references"))
    cap_s = float(inference_cfg["buffer_cap_s"])
    # The unseen-word rates and the train-text split read the train set of the MODELS: the corpus they trained on (provenance
    # source_language), which a zero-shot row takes from another corpus than the one it scores (README §7.6).
    sources = {label: (json.loads(Path(path).read_text(encoding="utf-8")).get("provenance") or {}).get("source_language") or args.language
               for label, path in systems_paths.items()}
    if len(set(sources.values())) > 1: raise SystemExit(
        f"[report] the systems trained on different corpora ({', '.join(f'{l}: {s}' for l, s in sources.items())}): the unseen-word "
        f"rates and the train-text split read the train set of the models, so make 1 report per source corpus."
    )
    source = next(iter(sources.values()), args.language)
    train_records, _ = load_language_records(data_cfg, source, split="train")
    train_vocab = {t for r in train_records for sp in r.sentences if getattr(sp, "reliable", True)
                       for t in (_vocab_token(w) for w in sp.text.split()) if t}

    # Named S-to-T for zero-shot systems, so their report never rebuilds (and deletes) the report of the corpus's own models.
    out = Path(args.out) if args.out else output_dir("report") / f"results_{corpus_token(args.language, source)}_{args.split}"
    fig_dir, tab_dir = out / "fig", out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)
    for old in list(fig_dir.glob("*.png")) + list(tab_dir.iterdir()): old.unlink()   # a report is rebuilt whole; no stale files

    # Score each system's emitted events under the shared annotation-ignore policy.
    raw_events, edge, uncaptioned = {}, {}, {}
    systems, scores, tables, errors, count_ratio, bstats, exemplars, provenance = {}, {}, {}, {}, {}, {}, {}, {}
    for label, path in systems_paths.items():
        warn_annotation_mismatch(path, records, label)   # an events file from other labels is flagged, then scored
        # The model that made the events (provenance model_annotation): trained on other train labels than those at eval time?
        provenance[label] = json.loads(Path(path).read_text(encoding="utf-8")).get("provenance") or {}
        warn_model_labels(provenance[label].get("model_annotation"), f"{label} ({path})")
        raw_events[label] = load_event_predictions(path)
        pred = scoreable_predictions(raw_events[label], records, tag=label)   # never cut (the method rule): each committed span is one caption
        systems[label] = pred
        # The densevid scores: 1 row per threshold (`by_tiou`) under its float, and `avg` (the F1 of the mean precision and recall).
        dense = evaluate_predicted_events(pred, gold, thresholds, char_level=char_level, densevid=True)
        sc = {**{r["tiou"]: r for r in dense["by_tiou"]}, "avg": dense["avg"]}
        errors[label] = edge_error_types(pred, chunks, edge_taus, thresholds)   # the one error table: per cell, the types of the counted pairs
        for t in thresholds:   # a tIoU score: the corpus BLEU-4 of the one-to-one tIoU pairs
            tiou_pairs = located_pairs(pred, gold, t)
            sc[t]["located_corpus_bleu4"] = text_scores([h for _, _, h, _ in tiou_pairs], [r for _, _, _, r in tiou_pairs], char_level,
                                                        keys=("bleu4",))["bleu4"] if tiou_pairs else 0.0
        # the per-unit outcomes: each unit's error type in each cell of the grid, with the caption whose pair counts and its score
        unit_rows = unit_outcomes(chunks, pred, errors[label], char_level)
        for r in unit_rows:
            toks = [t for t in (_vocab_token(w) for w in r["ref"].split()) if t]
            r["oov_rate"] = round(sum(1 for w in toks if w not in train_vocab) / max(1, len(toks)), 3)
        tables[label] = {r["gold_id"]: r for r in unit_rows}
        bstats[label] = boundary_stats(errors[label])
        exemplars[label] = case_studies(unit_rows, pred, errors[label])
        count_ratio[label] = sum(len(v) for v in pred.values()) / max(1, n_gold)
        with (tab_dir / f"outcomes_{_slug(label)}.csv").open("w", encoding="utf-8", newline="") as f:   # 1 line per unit and cell
            w = csv.DictWriter(f, fieldnames=OUTCOME_COLUMNS, extrasaction="ignore"); w.writeheader()
            w.writerows({**r, "tau_s": cell[0], "tiou": cell[1], **at} for r in unit_rows for cell, at in r["cells"].items())
        scores[label], uncaptioned[label] = sc, uncaptioned_units(pred, gold)
        edge[label] = edge_chunk_scores(pred, chunks, edge_taus, thresholds, char_level)
        print(f"[report] {label}: {_main_line(edge[label], uncaptioned[label])}", flush=True)
    labels = list(systems)
    gold_ids = list(tables[reference])
    for label in labels:
        if list(tables[label]) != gold_ids: raise SystemExit(f"{label}: gold index differs from {reference!r}; all systems need the same split and GT settings")

    # ---- tables
    # A score row of the main grid or of the densevid block, flat: its scores, its text blocks as dvc_<key> and soda_<key>, its counts.
    flat = lambda r: {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items() if k not in ("tau_s", "tiou", "dvc", "soda")} | {
        f"{block}_{k}": round(v, 4) for block in ("dvc", "soda") for k, v in (r.get(block) or {}).items()}
    # The densevid scores (one caption per unit): 1 row per threshold with its counts, then the avg (scores only), as `densevid` of a
    # row's _scores.json.
    _write_csv(tab_dir / "densevid.csv", [{"system": l, "tiou": _name(t), **flat({k: v for k, v in scores[l][t].items() if k in ("precision", "recall", "f1", "dvc")}),
                                           **({} if t == "avg" else {"located_corpus_bleu4": round(scores[l][t]["located_corpus_bleu4"], 2), "located": scores[l][t]["scored_pairs"],
                                                                     "events": scores[l][t]["total_predictions"], "gold": scores[l][t]["total_gold"]})}
                                          for l in labels for t in [*thresholds, "avg"]])
    # The main scores on the grid (metrics.edge_chunk_scores): every cell, each tau over the thresholds (tiou "avg"), each threshold over
    # the taus (tau_s "avg"), all cells ("avg", "avg"), then the reference rows of the tIoU test alone (tau_s "no limit", in no average);
    # the counts of counted captions and units only on the rows that have them.
    _write_csv(tab_dir / "edge_chunk.csv", [{"system": l, "tau_s": _name(tau_s), "tiou": _name(tiou), **flat(r)}
                                           for l in labels for tau_s, tiou, r in _grid_rows(edge[l])])
    _write_csv(tab_dir / "uncaptioned.csv", [{"system": l, **{k: round(v, 4) if isinstance(v, float) else v for k, v in uncaptioned[l].items()}} for l in labels])
    # 1 row per system, side and cell of the grid (and the mean over the cells): the share of each type (per video, then the mean over
    # videos, as the main score) and its pooled count; a type of the other side is blank
    types = list(dict.fromkeys(EDGE_CAPTION_TYPES + EDGE_UNIT_TYPES))
    _write_csv(tab_dir / "error_types.csv", [{"system": l, "side": side, "tau_s": tau_s, "tiou": tiou, **{c: round(shares[c], 4) if c in shares else "" for c in types},
                                              **{f"n_{c}": counts.get(c, "") for c in types}, "captions_per_unit": round(count_ratio[l], 3)}
                                             for l in labels for side in ("units", "captions") for tau_s, tiou, shares, counts in
                                             [(c["tau_s"], c["tiou"], c[side], c[f"{side[:-1]}_counts"]) for c in errors[l]["cells"]] + [("avg", "avg", errors[l][f"{side}_avg"], {})]])
    _write_csv(tab_dir / "boundary.csv", [{"system": l, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in bstats[l].items()}} for l in labels])
    # ---- report-only readouts (no `_scores.json` key): metrics.py computes each one, on inputs that evaluation.predictions builds
    rnd = lambda row, n=4: {k: round(v, n) if isinstance(v, float) else v for k, v in row.items()}
    # The avg of a main-score dict (all cells of the grid: the main score and its text scores), flat: avg_precision, avg_recall,
    # avg_f1, avg_dvc_<key>, avg_soda_<key>.
    flat_avg = lambda e: {f"avg_{k}": e["avg"][k] for k in ("precision", "recall", "f1")} | {
        f"avg_{block}_{k}": v for block in ("dvc", "soda") for k, v in (e["avg"][block] or {}).items()}
    delta_s, gap_s = float(inference_cfg["boundary_stability"]["delta_s"]), float(scoring_cfg["chunk_gap_s"])
    # (1) the main localization at other chunk gaps, with BLEU-4 text only (no learned scorer); the main row stays at chunk_gap_s
    gap_chunks = {g: gold_chunks(records, g) for g in map(float, scoring_cfg["chunk_gap_rows_s"])}
    gaps = {l: {g: edge_chunk_scores(systems[l], c, edge_taus, thresholds, char_level, bleurt_checkpoint=None, keys=("bleu4",))
                for g, c in gap_chunks.items()} for l in labels}
    _write_csv(tab_dir / "edge_chunk_gaps.csv", [
        {"system": l, "chunk_gap_s": g, "main": g == gap_s, "chunks": e["chunks"], **rnd({f"avg_{k}": e["avg"][k] for k in ("precision", "recall", "f1")}),
         **{f"avg_{block}_bleu4": round(e["avg"][block]["bleu4"], 2) if e["avg"][block] else None for block in ("dvc", "soda")}}
        for l in labels for g, e in sorted({gap_s: edge[l], **gaps[l]}.items())])
    # (2) the caption shape: durations and the reliable units each caption holds; the gold units give the reference durations
    shape = {"gold units": caption_shape(gold, gold), **{l: caption_shape(systems[l], gold) for l in labels}}
    _write_csv(tab_dir / "caption_shape.csv", [{"system": l, "captions": s["captions"], **{f"duration_{k}_s": round(v, 3) for k, v in s["duration_s"].items()},
                                                **{f"units_held_{k}": round(v, 4) for k, v in s["units_held"].items()}} for l, s in shape.items()])
    # (3) the display-cue share: caption edges inside a unit near a junction of 2 of its display cues, against chance
    breaks = cue_breaks(records, data_cfg, args.language)
    cues = {l: cue_edge_share(systems[l], gold, breaks, edge_taus) for l in labels}
    _write_csv(tab_dir / "cue_breaks.csv", [{"system": l, **rnd(r)} for l in labels for r in cues[l]["taus"]])
    # (4) the units per chunk class; over_cap reads the longest span that commits whole (the cap minus δ)
    classes = chunk_classes(records, gap_s, cap_s - delta_s)
    chunk_cls = {l: edge_chunk_classes(systems[l], chunks, classes, edge_taus, thresholds) for l in labels}
    _write_csv(tab_dir / "chunk_classes.csv", [{"system": l, "class": c, **rnd(chunk_cls[l][c])} for l in labels for c in CHUNK_CLASSES if c in chunk_cls[l]])
    # (5) every main score on the videos with no train text and on those with it (the set is computed from the records at each run)
    dedup = corpus_config(data_cfg, args.language)["subtitles"]["dedup"]   # the copy check's floor for 1 cue and its boilerplate cap
    tt = train_text_videos(records, train_records, int(dedup["min_words"]), int(dedup["df_cap"]))
    _write_csv(tab_dir / "train_text_videos.csv", [{"video_id": v, "reliable_units": len(gold[v]), "units_with_train_text": tt.get(v, 0)} for v in sorted(gold)])
    video_sets = {"no train text": [v for v in gold if not tt.get(v)], "train text": [v for v in gold if tt.get(v)]}
    keep = lambda d, vids: {v: d[v] for v in vids if v in d}
    by_tt = {l: {name: (edge_chunk_scores(keep(systems[l], vids), keep(chunks, vids), edge_taus, thresholds, char_level),
                        uncaptioned_units(keep(systems[l], vids), keep(gold, vids))) for name, vids in video_sets.items() if vids} for l in labels}
    _write_csv(tab_dir / "edge_chunk_by_train_text.csv", [{"system": l, "videos": name, "n_videos": len(video_sets[name]), **rnd(flat_avg(e)),
                                                           "uncaptioned": u["uncaptioned"], "units": u["units"]} for l in labels for name, (e, u) in by_tt[l].items()])
    # (6) the cap-cut captions apart: count, their counted pairs per cell, their units, and the main scores without them (all captions'
    # when there are none)
    hours = sum(float(r.pose.duration_s) for r in records) / 3600.   # the pose time of the split's streams (its clean segments)
    cap = {l: cap_cut_captions(systems[l], chunks, gold, edge_taus, thresholds, char_level, hours) for l in labels}
    for l, c in cap.items():
        if c["without"] is None: c["without"], c["uncaptioned_without"] = edge[l], uncaptioned[l]
    _write_csv(tab_dir / "cap_cut.csv", [{"system": l, "captions": c["captions"], "per_hour": round(c["per_hour"], 3), "units_held": c["units_held"],
                                          **{f"matched@{r['tau_s']:g}/{r['tiou']:g}": r["matched"] for r in c["cells"]},
                                          **{f"matched_units@{r['tau_s']:g}/{r['tiou']:g}": r["matched_units"] for r in c["cells"]},
                                          "avg_f1": round(edge[l]["avg"]["f1"], 4), **{f"without_{k}": v for k, v in rnd(flat_avg(c["without"])).items()},
                                          "uncaptioned": uncaptioned[l]["uncaptioned"], "uncaptioned_without": c["uncaptioned_without"]["uncaptioned"]} for l, c in cap.items()])
    # (7) the gold unit edges by side and boundary class, and those that a caption edge of the same side pairs within τ, 1 to 1
    unit_edges = gold_edges(records, gap_s)
    bounds = {l: edge_boundary_hits(systems[l], unit_edges, edge_taus) for l in labels}
    _write_csv(tab_dir / "edge_boundaries.csv", [{"system": l, **rnd(r)} for l in labels for r in bounds[l]])
    # differences against the reference: each headline number of a system minus the reference's (an avg is that of all cells of the main
    # grid, or of every densevid threshold)
    lat_of = {label: held_unit_latency(gold, systems[label]) for label in labels}
    diff = []
    cells = [(c["tau_s"], c["tiou"]) for c in errors[reference]["cells"]]   # the cells of the grid, tau-major
    loc = lambda l, cell: float(np.mean([tables[l][g]["cells"][cell]["localized_bleu4"] for g in gold_ids])) if gold_ids else 0.0
    for label in labels:
        if label == reference: continue
        clean = lambda l: by_tt[l]["no train text"][0]["avg"] if "no train text" in by_tt[l] else None   # the main scores on the videos with no train text
        cs, cr, ea, eb = clean(label), clean(reference), edge[label]["avg"], edge[reference]["avg"]
        for metric, a, b, extra in (
            ("MAIN score: F1 of the tIoU x edge-limit grid, avg", ea["f1"], eb["f1"], {}),
            *[(f"DVC {TEXT_NAME[k]} on the counted pairs, grid avg (main)", ea["dvc"][k], eb["dvc"][k], {}) for k in TEXT_KEYS if ea["dvc"] and eb["dvc"]],
            *[(f"SODA-style {TEXT_NAME[k]} F1 on the counted pairs, grid avg (main)", ea["soda"][k], eb["soda"][k], {})
              for k in SODA_KEYS if ea["soda"] and eb["soda"]],
            ("densevid sentence tIoU F1, avg", scores[label]["avg"]["f1"], scores[reference]["avg"]["f1"], {}),
            ("densevid DVC BLEU-4, avg", scores[label]["avg"]["dvc"]["bleu4"], scores[reference]["avg"]["dvc"]["bleu4"], {}),
            *[(f"localized BLEU-4 at τ {c[0]:g} s, tIoU {c[1]:g} (mean per unit)", loc(label, c), loc(reference, c),
               flip_table(tables[label], tables[reference], c)) for c in cells],
            ("localized BLEU-4, mean over the cells (mean per unit)", float(np.mean([loc(label, c) for c in cells])),
             float(np.mean([loc(reference, c) for c in cells])), {}),
            *[(f"F1 of the grid avg, chunk gap {g:g} s", gaps[label][g]["avg"]["f1"], gaps[reference][g]["avg"]["f1"], {}) for g in gap_chunks],
            *([("MAIN score (F1 of the grid avg), videos with no train text", cs["f1"], cr["f1"], {})] if cs else []),
            *[(f"DVC {TEXT_NAME[k]}, grid avg (main), videos with no train text", cs["dvc"][k], cr["dvc"][k], {})
              for k in TEXT_KEYS if cs and cs["dvc"] and cr["dvc"]],
            *[(f"SODA-style {TEXT_NAME[k]} F1, grid avg (main), videos with no train text", cs["soda"][k], cr["soda"][k], {})
              for k in SODA_KEYS if cs and cs["soda"] and cr["soda"]],
        ):
            diff.append({"system": label, "reference": reference, "metric": metric, "system_value": round(a, 4), "reference_value": round(b, 4),
                         "difference": round(a - b, 4), **extra})
        if lat_of[label] and lat_of[reference]:   # both online: commit times exist
            a, b = lat_of[label]["median_s"], lat_of[reference]["median_s"]
            diff.append({"system": label, "reference": reference, "metric": "held-unit median latency (s)", "system_value": a, "reference_value": b,
                         "difference": round(a - b, 3)})
    if diff: _write_csv(tab_dir / "differences.csv", diff)
    # what the extra sentences cost: remove every event that locates nothing at tau, score again, change nothing else
    cost = []
    for label in labels:
        kept = {}
        for vid, evs in systems[label].items():
            ok = {pi for pi, _, _ in match_segments([e.segment for e in evs], [g.segment for g in gold.get(vid, [])], threshold=min(thresholds))}
            kept[vid] = [e for pi, e in enumerate(evs) if pi in ok]
        dense2 = evaluate_predicted_events(kept, gold, thresholds, char_level=char_level, densevid=True)
        sc2 = {**{r["tiou"]: r for r in dense2["by_tiou"]}, "avg": dense2["avg"]}
        for name, sc_ in (("as reported", scores[label]), ("unlocated events removed", sc2)):   # every threshold, then the avg
            cost.append({"system": label, "version": name, "events": sum(len(v) for v in (systems[label] if name == "as reported" else kept).values()),
                         **{f"tiou_f1@{_name(t)}": round(sc_[t]["f1"], 3) for t in [*thresholds, "avg"]},
                         **{f"dvc_bleu4@{_name(t)}": round(sc_[t]["dvc"]["bleu4"], 2) for t in [*thresholds, "avg"]}})
    _write_csv(tab_dir / "extra_sentences_cost.csv", cost)
    held = []
    for label in labels:
        lat = held_unit_latency(gold, systems[label])
        if lat: held.append({"system": label, **lat})
    if held: _write_csv(tab_dir / "held_unit_latency.csv", held)
    # Decision metrics, fixed before any trained arm is read (docs/implementation_notes.md): gold-edge recall (ends by boundary type: 
    # the sampler's target_pause_after rule), translation by the reference's span class, and ReSCUE's commit-only axes (long-form BLEU-4, 
    # LAAL/AL per unit).
    pause_after, unk = {}, unk_stretches(records)
    for rec in records:   # the order of gold_events: records in time order, reliable units in order
        gap = seconds_to_frames(delta_s, rec.pose.fps) / rec.pose.fps
        for span in rec.sentences:
            if not getattr(span, "reliable", True): continue
            after = min((u.start_s for u in rec.sentences if u is not span and u.start_s >= span.end_s), default=None)
            pause_after.setdefault(rec.pose.video_id, []).append(after is None or after - span.end_s > gap)
    edge_rows = [{"system": l, **edge_recall(gold, systems[l], delta_s, pause_after)} for l in labels]   # `edges` is the by-length bins below
    long_form = [{"system": l, **long_form_scores(gold, systems[l], unk, char_level)} for l in labels]
    cls_of = unit_span_classes({v: [u for chunk in c for u in chunk] for v, c in chunks.items()}, raw_events[reference])   # the rows' units, in their order
    by_ref_class = []
    for label in labels:
        if label == reference: continue
        for c in ("1-unit span", "multi-unit span", "none"):
            ids = [g for g in gold_ids if cls_of[tables[reference][g]["video_id"]][int(g.rsplit(":", 1)[1])] == c]
            if not ids: continue
            for cell in cells:
                by_ref_class.append({"system": label, "reference": reference, "tau_s": cell[0], "tiou": cell[1], "span_class": c, "n_units": len(ids),
                                     "mean_delta": round(float(np.mean([tables[label][g]["cells"][cell]["localized_bleu4"]
                                                                        - tables[reference][g]["cells"][cell]["localized_bleu4"] for g in ids])), 3)})
    for name, rows_ in (("edge_recall", edge_rows), ("long_form", long_form), ("bleu_by_reference_span_class", by_ref_class)):
        if rows_: _write_csv(tab_dir / f"{name}.csv", rows_)
    shared = []
    for label in labels:
        if label == reference: continue
        for cell in cells:   # the units that both systems match alone in the cell, so both corpora have the same references
            both = [g for g in gold_ids if tables[label][g]["cells"][cell]["type"] == "matched" and tables[reference][g]["cells"][cell]["type"] == "matched"]
            corpus = lambda l: round(text_scores([tables[l][g]["cells"][cell]["hyp"] for g in both], [tables[l][g]["ref"] for g in both], char_level, keys=("bleu4",))["bleu4"], 2) if both else 0.0
            shared.append({"system": label, "tau_s": cell[0], "tiou": cell[1], "n_shared": len(both), "corpus_bleu4": corpus(label),
                           "reference_corpus_bleu4": corpus(reference)})
    if shared: _write_csv(tab_dir / "shared_sentences.csv", shared)
    with (tab_dir / "cases.json").open("w", encoding="utf-8") as f: json.dump(exemplars, f, indent=1, ensure_ascii=False)

    # ---- figures
    F.experiment_map(fig_dir / "experiment_map.png"); F.metric_explained(fig_dir / "metric_explained.png")
    F.scores_at_thresholds(scores, labels, rth, fig_dir / "densevid.png", n_gold)
    F.precision_recall(scores, labels, rth, fig_dir / "precision_recall.png")
    F.located_vs_quality(scores, labels, rth, fig_dir / "located_vs_quality.png", n_gold)
    F.error_types(errors, labels, count_ratio, fig_dir / "error_types.png", n_gold)
    F.boundary_error(bstats, labels, fig_dir / "boundary_error.png", edge_taus)
    edges = [0.0, 2.0, 4.0, 6.0, 9.0, 14.0, cap_s, float("inf")]; lab = ["0-2 s", "2-4 s", "4-6 s", "6-9 s", "9-14 s", "14 s-cap", "> cap"]
    F.by_length(tables, labels, edges, lab, fig_dir / "by_length.png", cap_s)
    F.by_oov(tables, labels, fig_dir / "by_oov.png")
    F.bleu_by_edge_error(tables, labels, fig_dir / "bleu_by_edge_error.png")

    # ---- the BLEU protocol (needs the pretrimmed translator on perfect cuts: row 1 among the systems, the first one that reads the
    # gold spans with the pretrimmed translator)
    proto, orc, md_opt = None, None, []
    true_cuts = next((l for l in labels if provenance[l].get("segmenter_arch") == "gold" and provenance[l].get("method") == "baseline"), None)
    if true_cuts:
        orc = oracle_pairs(gold, raw_events[true_cuts])
        proto = decompose(gold, systems, thresholds, char_level, oracle=orc)
        with (tab_dir / "bleu_protocol.json").open("w", encoding="utf-8") as f:
            json.dump({"oracle": {k: v for k, v in proto["oracle"].items() if k != "hyp_of"} if proto["oracle"] else None,
                       "systems": {s: {str(t): {k: v for k, v in r.items() if k != "per_video"} for t, r in rows.items()} for s, rows in proto["systems"].items()}}, f, indent=1)
        if proto["oracle"]:
            orow = oracle_row(gold, orc, records, thresholds, char_level)
            with (tab_dir / "oracle_row.json").open("w", encoding="utf-8") as f: json.dump(orow, f, indent=1)
            F.protocol_two_ways(proto, fig_dir / "protocol_two_ways.png", args.language)
            F.protocol_per_threshold(proto, labels, thresholds, fig_dir / "protocol_per_threshold.png", args.language, orow)
            two = two_video_example(gold, systems, orc, proto["oracle"], tau, char_level)
            big_win = (gold[two["big"]][0].start_s - 1, gold[two["big"]][0].start_s + 59)
            _sys1 = {two["system"]: systems[two["system"]]}; _raw1 = {two["system"]: raw_events[two["system"]]}
            F.protocol_two_videos(two, {two["small"]: case_view(gold, _sys1, two["small"], tau, char_level, oracle=orc, raw=_raw1),
                                        two["big"]: case_view(gold, _sys1, two["big"], tau, char_level, window=big_win, oracle=orc, raw=_raw1)},
                                  proto, fig_dir / "protocol_two_videos.png", args.language, orow)
            ex_vid = worked_example_video(gold, systems, tau)
            if ex_vid:
                F.protocol_worked_example(case_view(gold, systems, ex_vid, tau, char_level, oracle=orc, raw=raw_events), proto, tau, fig_dir / "protocol_example.png", args.language)
                F.case_thresholds({t: case_view(gold, systems, ex_vid, t, char_level, oracle=orc, raw=raw_events) for t in thresholds},
                                  thresholds, fig_dir / "protocol_thresholds.png", args.language, "the same video, every threshold")
            md_opt.append(_protocol_md(proto["oracle"], proto, labels, thresholds, tau, ex_vid, two, orow))

    # ---- real videos: rule-picked showcases, pinned videos, then the reference system's longest failures
    shows = showcase_videos(gold, systems, thresholds, tau, proto["oracle"] if proto and proto["oracle"] else None)
    for spec in args.case_videos or []:
        vid, _, rest = spec.partition(":"); window, note = None, None
        if rest:
            rng, _, note = rest.partition(":")
            if "-" in rng: a, b = rng.split("-"); window = (float(a), float(b))
        if vid in gold and vid not in {c["video_id"] for c in shows}: shows.append({"video_id": vid, "window": window, "note": note or "pinned"})
    lc = loose_cell(cells)   # a unit still lost in the loosest cell of the grid is a clear error
    loose = next(c for c in exemplars[reference]["cells"] if (c["tau_s"], c["tiou"]) == lc)
    for kind in ("bridged", "cut", "missed"):
        c = next(iter(loose[kind]), None)
        if c and c["video_id"] not in {x["video_id"] for x in shows}:
            shows.append({"video_id": c["video_id"], "window": (max(0.0, c["start_s"] - 15), c["end_s"] + 15),
                          "note": f"longest {kind} caption unit of {reference} in the loosest cell (τ {lc[0]:g} s, tIoU {lc[1]:g})"})
    case_md = []
    for i, c in enumerate(shows):
        c_tau = float(c.get("tau", tau))
        view = case_view(gold, systems, c["video_id"], c_tau, char_level, window=c["window"], oracle=orc, raw=raw_events)
        name = f"case_{i+1}_{c['video_id']}.png"
        F.case_figure(view, c_tau, fig_dir / name, note=c["note"], language=args.language)
        case_md.append(f"**{c['video_id']}**: {c['note']}\n\n![](fig/{name})\n" + _sentence_tables_md(c["video_id"], tables, labels))

    # ---- other optional sections
    if args.boundary_sweep_file:
        warn_annotation_mismatch(args.boundary_sweep_file, records, "boundary sweep file")
        sweep = json.load(open(args.boundary_sweep_file, encoding="utf-8"))
        rows = [r for r in sweep.get("rows", []) if r.get("prediction") is not None and r.get("reference")][::max(1, len(sweep.get("rows", [])) // 4000)]
        if rows:
            b = sentence_bleu_scores([r["prediction"] for r in rows], [r["reference"] for r in rows], char_level)
            F.confidence_calibration({Path(args.boundary_sweep_file).stem.replace("boundary_sweep_", ""): list(zip([float(r.get("mean_confidence", 0)) for r in rows], b))}, fig_dir / "confidence_calibration.png")
            md_opt.append("## Confidence against quality\n\n![](fig/confidence_calibration.png)\n\nEach point is a group of boundary-sweep windows with similar model confidence; the y-axis is their mean sentence BLEU-4 (a per-window quantity, so a per-window score is the right one here). A rising curve means the model's confidence says something about quality.\n")
    if args.history:
        hist = {l: _read_history(p) for l, p in _labelled(args.history).items()}
        F.head_decay(hist, fig_dir / "training_segmentation.png")
        hd_rows, shipped = [], []
        for l, rows in hist.items():
            run_dir = Path(_labelled(args.history)[l]).parent
            best = run_dir / "best.json"
            kept = json.load(open(best)).get("epoch") if best.exists() else None
            # the epoch model.pt holds, as train stamps it; 0 means no trained epoch beat the initialization
            stamped = load_checkpoint_meta(run_dir / "model.pt").get("best_epoch") if (run_dir / "model.pt").exists() else None
            shipped.append([l, "-" if stamped is None else f"{stamped}" + (" (the initialization)" if stamped == 0 else "")])
            print(f"[report] {l}: model.pt best_epoch {shipped[-1][1]}", flush=True)
            pick = lambda ep: next((r for r in rows if r.get("epoch") is not None and float(r["epoch"]) == float(ep)), None)
            ep = lambda r: f"{r['epoch']:g}" if r.get("epoch") is not None else "-"
            # the run's monitor protocol: the offline row (val_offline_*) or the online row, the streaming FSM (val_online_*)
            mon = "val_offline" if any(r.get("val_offline_f1") is not None for r in rows) else "val_online"
            if not any(r.get(f"{mon}_f1") is not None for r in rows): print(
                f"[report] WARNING: {l}: the history logs neither val_offline_f1 nor val_online_f1 (an older run logged val_stream_* "
                "or val_joint_score), so its monitor cells and curves stay empty.", flush=True
            )
            evaluated = [r for r in rows if r.get(f"{mon}_f1") is not None] or rows   # an arm evaluates every eval_every_epochs
            for tag, r in ((f"first (epoch {ep(evaluated[0])})", evaluated[0]), (f"kept (epoch {kept})", pick(kept) if kept is not None else None), (f"last (epoch {ep(evaluated[-1])})", evaluated[-1])):
                if r: hd_rows.append([l, tag, r.get(f"{mon}_f1", "-"), r.get(f"{mon}_events_per_gold", "-"), r.get("train_bio_loss", "-"), r.get("val_bio_loss", "-")])
        for l, rows in hist.items():
            keys = dict.fromkeys(k for r in rows for k in r)   # every row's keys, in order: epochs without a dev evaluation have no val_ cells
            cols = [c for c in keys if (c.endswith("_loss") or c.startswith("val_")) and c not in ("epoch", "took", "eta", "ckpt") and any(r.get(c) not in (None, "", "-") for r in rows)]
            F.history_facets(rows, cols, f"Training curves: {l}", fig_dir / f"training_{_slug(l)}.png")
        md_opt.append("## Training curves: does joint training change the segmentation head?\n\n![](fig/training_segmentation.png)\n\n" +
                      _table(["run", "epoch", "dev monitor F1 avg", "events per gold unit", "train BIO loss", "dev BIO loss"], hd_rows) +
                      "\nShipped checkpoint (`model.pt` meta `best_epoch`):\n\n" + _table(["run", "best epoch"], shipped) +
                      "\n" + "".join(f"![](fig/training_{_slug(l)}.png)\n" for l in hist)
                      + "\nHow to read: the monitor F1 and the events per gold unit are the checkpoint monitor's own whole-video numbers on dev (logged at each dev evaluation). Events per gold unit that drift below 1 mean the head merges more units; a dev BIO loss that rises while the train loss falls is consistent with overfitting, but does not identify its cause.\n")
    if args.segmentation_scores:
        ev = {l: _segmentation_only(p, records, scoring_cfg) for l, p in _labelled(args.segmentation_scores).items()}
        F.frame_vs_sentence(ev, rth, fig_dir / "segmentation_only.png")
        md_opt.append("## Segmentation only, before any translation\n\n![](fig/segmentation_only.png)\n\n" + _segmentation_md(ev, rth) +
                      "\nHow to read: binary signing coverage can be high even when B boundaries are missing. Sentence localization checks both boundaries; binary signing F1 does not.\n")
    if args.stability_file:
        pol = {Path(p).stem.replace("stability_", ""): json.load(open(p, encoding="utf-8")) for p in args.stability_file}
        F.stability(pol, fig_dir / "stability.png")
        md_opt.append("## Reveal policies\n\n![](fig/stability.png)\n\nHow to read: further left is earlier text on screen; higher is more text that later had to change.\n")

    # ---- markdown, in the reading order of the weekly report
    corpus_name = cfg_get(data_cfg, "languages", args.language, "name", default=args.language)   # its data.yaml name
    intro = [f"**{args.language}: {corpus_name}. {args.split} set: {len(gold)} videos, {n_gold} caption units.**",
             f"The main score is the **F1 of the tIoU x edge-limit grid, avg** (section 2; every cell and every avg in `tables/edge_chunk.csv`): a caption "
             f"and a run of caption units pair one to one by tIoU, and the pair counts in the cell (τ, t) when its tIoU is at least t and both of its edges "
             f"are within τ of the run's edges, for τ = {', '.join(f'{t:g}' for t in edge_taus)} s and t = {', '.join(f'{t:g}' for t in thresholds)}. "
             f"Each avg is the F1 of the mean precision and the mean recall of its cells. The error types and the per-unit tables use the same counted pairs, "
             f"in each cell. The densevid scores (one caption per unit) are shown at {len(rth)} levels of strictness, **tIoU {', '.join(f'{t:g}' for t in rth)}**, "
             f"and the avg of every scored threshold. Every system is scored with the same code on the same caption units; the per-unit tables (each unit's "
             f"error type in each cell, with the caption whose pair counts with its run) are in `tables/outcomes_<system>.csv`, "
             f"and every scored threshold ({', '.join(f'{t:g}' for t in thresholds)}) of the densevid scores is in `tables/densevid.csv`."
             + (" Every text is scored in lowercase, for every system alike (data.yaml lowercase_references)." if char_level.lowercase else "") + "\n"]
    sec = []
    sec.append("## 1. The experiments\n\n![experiment map](fig/experiment_map.png)\n\n" + _table(["name in this report", "events file"], [[l, f"`{Path(p).name}`"] for l, p in systems_paths.items()]))
    sec.append("## 2. How the scores work\n\n![metric explained](fig/metric_explained.png)\n\n"
               f"- **The main score: the F1 of the tIoU x edge-limit grid.** A *chunk* is a group of caption units of one stream with gaps of at most {float(scoring_cfg['chunk_gap_s']):g} s between them (no pause). "
               "A prediction may hold any run of whole, consecutive units inside one chunk, so joining units that are signed without a pause is not an error, and no run crosses a pause. "
               "Per video, the predictions and the runs pair one to one by tIoU, the highest tIoU first; each prediction and each unit is used once. "
               "A pair counts at tIoU threshold t when its tIoU is at least t, and at edge limit τ when its start and its end are each within τ of the run's start and end: "
               "τ is an extra limit on a pair that passed the tIoU test, never a looser test. "
               f"Each cell (τ, t) of the grid (τ = {', '.join(f'{t:g}' for t in edge_taus)} s, t = {', '.join(f'{t:g}' for t in thresholds)}) gives a precision (counted predictions / predictions, per video, then the mean over videos) "
               "and a recall (units inside counted runs / units, the same way). An avg (each τ over the thresholds, each threshold over the τ values, or all cells) takes the mean "
               "precision and the mean recall of its cells, and its F1 is the F1 of these 2 means, not the mean of the F1 values, as in our NeurIPS protocol. "
               "The main score is the F1 at τ avg and tIoU avg. The line `no limit` applies the tIoU test alone (no edge limit), for reference: it is in no average and in no monitor.\n"
               "- **tIoU** = shared time / total time covered.\n"
               f"- **DVC on the counted pairs (main text scores).** The DVC of densevid_eval on the pairs that count in a cell: per video, each prediction whose pair counts is scored against the joined text of its run, and any other prediction against a random string; then the plain mean over videos; an avg is the mean of its cells: {', '.join(TEXT_NAME[k] for k in TEXT_KEYS)} (BLEU-4 is the corpus BLEU-4 of the video's pairs, with no smoothing). A missed unit adds no DVC pair.\n"
               f"- **SODA-style text F1 on the counted pairs (main).** Per video, precision = summed pair scores / predictions and recall = summed pair scores, each weighted by the units of its run, / units; the F1 of the two, then the mean over videos; an avg is the mean of its cells: {', '.join(TEXT_NAME[k] for k in SODA_KEYS)} (per pair: sentence BLEU-4; CIDEr-D has no per-pair score). It charges false predictions and missed units.\n"
               "- **Sentence tIoU F1 (densevid score)** counts located sentences, one prediction per sentence by tIoU (precision: of what we produced; recall: of what is true). Its avg is the F1 of the mean precision and the mean recall over the thresholds.\n"
               "- **tIoU DVC BLEU-4** is the DVC of densevid_eval on its own tIoU pairs: every prediction against every caption unit it overlaps by tIoU, per video, then the plain mean over videos. A prediction with no qualifying overlap is scored against a random string. A missed caption unit adds no DVC pair.\n"
               "- **BLEU-4 of the located pairs** is corpus BLEU-4 over the one-to-one located pairs only: text quality without the localization penalty.\n"
               "- Corpus BLEU-4 pools n-gram counts. DVC averages per-video corpus BLEU-4; localized diagnostics instead use sentence BLEU. Their labels and denominators must remain distinct. The text backend is SacreBLEU 13a, a documented difference from the original DVC COCO/PTB scorer.\n")
    seg_md = next((m for m in md_opt if m.startswith("## Segmentation only")), None)
    if seg_md: sec.append(seg_md.replace("## Segmentation only", "## 3. Segmentation only", 1))
    n0 = 4 if seg_md else 3
    n_tt = len(video_sets["train text"])
    text_labels = [l for l in labels if edge[l]["avg"]["dvc"]]
    sec.append(f"## {n0}. Main results\n\n**The main score: the F1 of the tIoU x edge-limit grid (avg), with its precision and recall, its F1 per edge limit τ "
               "(over the thresholds) and per tIoU threshold (over τ), and the F1 of the tIoU test alone (no edge limit, avg; for reference).**\n\n" + _edge_md(edge, uncaptioned, labels) +
               "\n**F1 in every cell of the grid** (one line per edge limit τ, then `avg` over τ and the reference line `no limit`; one column per tIoU threshold, then "
               "`avg` over the thresholds; the main score is the F1 at avg, avg).\n\n" + _grid_md(edge, labels, lambda r: f"{r['f1']:.3f}") +
               "\n**Precision (P, of the captions) / recall (R, of the units) in every cell.**\n\n" + _grid_md(edge, labels, lambda r: f"{r['precision']:.3f} / {r['recall']:.3f}") +
               "\n**DVC on the counted pairs (main text scores; avg of the grid).**\n\n" + _edge_text_md(edge, labels, "dvc", TEXT_KEYS) +
               ("\n**DVC BLEU-4 in every cell.**\n\n" + _grid_md(edge, text_labels, lambda r: f"{r['dvc']['bleu4']:.2f}") if text_labels else "") +
               "\n**SODA-style text F1 on the counted pairs (main; avg of the grid).**\n\n" + _edge_text_md(edge, labels, "soda", SODA_KEYS) +
               "\n![densevid scores](fig/densevid.png)\n\n**densevid scores (secondary; one caption per unit): sentence localization (sentence tIoU F1).**\n\n" + _metric_md(scores, labels, rth, lambda r: r["f1"], "{:.3f}") +
               "\n**densevid scores: dense captioning (tIoU DVC BLEU-4; mean over videos).**\n\n" + _metric_md(scores, labels, rth, lambda r: r["dvc"]["bleu4"], "{:.2f}") +
               "\n**Reading.**\n\n" + _reading_main(scores, edge, labels, rth) +
               "\n**Units by chunk class (report only).** Each reliable unit takes the class of its chunk, by the first rule that fits: `over_cap` (the chunk is longer than "
               f"the cap minus δ, {cap_s - delta_s:g} s, the longest span that the engine commits whole; online, a longer span can commit only as a cap-cut closed part, which "
               "ends at the lowest P(I) frame, not at a unit edge; the offline rows keep every span whole), `video_start` (it starts at frame 0 of the video), `cut_start` (it starts at a stream start made by a cut, or after a "
               "quarantined unit that crosses the stream start), `after_quarantine`, `end_blocked` (it ends at the stream end or before a quarantined unit) and `free` "
               f"(a pause at both edges); \"at\" means within the chunk gap, {gap_s:g} s. `matched` is the share of the units inside counted runs (mean over the cells of the grid); the "
               "units with no caption of the classes sum to the column above. `tables/chunk_classes.csv`.\n\n" + _chunk_class_md(chunk_cls, labels) +
               "\n**The main score at other chunk gaps (report only; avg of the grid; DVC with BLEU-4 only).** `tables/edge_chunk_gaps.csv`.\n\n" + _gap_md(edge, gaps, labels, gap_s) +
               f"\n**The main scores on the videos with no train text (report only; avg of the grid).** A video has train text when one of its reliable units of {dedup['min_words']} "
               f"or more words has the words of a reliable train unit; {n_tt} of {len(gold)} videos do (`tables/train_text_videos.csv`; scores in `tables/edge_chunk_by_train_text.csv`). "
               "A comparison on these sets is valid only between models trained on the same train labels.\n\n" + _train_text_md(by_tt, video_sets, labels) +
               "\n**The cap-cut captions apart (report only).** A cap-cut caption is the closed part of an open span that the cap cut at the frame with the lowest P(I) "
               f"(the declared exception to P1 at inference). `per hour`: per hour of the split's pose time ({hours:.1f} h of clean segments); `units they hold`: at least half of each; `counted`: per cell, "
               "the cap-cut captions whose pair counts in the main score, with the units of their runs in brackets. `tables/cap_cut.csv`.\n\n" + _cap_cut_md(cap, edge, uncaptioned, labels))
    sec.append(f"## {n0+1}. Precision against recall\n\n![precision recall](fig/precision_recall.png)\n\n" + _pr_md(scores, labels, rth))
    sec.append(f"## {n0+2}. What changes when the match gets stricter\n\n![threshold quality](fig/located_vs_quality.png)\n\n" + _located_md(scores, labels, rth, n_gold) +
               "\nThe clips and captions are fixed across thresholds. Raising tIoU changes which predictions receive real references and which become unmatched. Located-only BLEU may rise because it scores a smaller subset; this is not an improvement to any caption. An unmatched prediction receives a random reference, while a missed GT sentence adds no DVC pair. Report localization recall beside DVC.\n")
    sec.append(f"## {n0+3}. Error types\n\n![error types](fig/error_types.png)\n\n" + _error_types_md(errors, labels, n_gold))
    sec.append(f"## {n0+4}. Real videos\n\nChosen by rule, never by hand: the tiny video with the top per-video score, the longest merged span that fits one figure, the prediction that pairs with the most caption units, the longest prediction with a looping caption, a one-sentence video with an over-long prediction, then the longest bridged, cut and missed caption units of the reference system in the loosest cell of the grid, τ {lc[0]:g} s and tIoU {lc[1]:g} (the error types of `tables/outcomes_<system>.csv`). The figures pair by tIoU at {tau:g}. `--case-videos id[:t0-t1[:note]]` pins more; every exemplar is in `tables/cases.json`.\n\n" + "\n".join(case_md))
    n = n0 + 5
    hist_md = next((m for m in md_opt if m.startswith("## Training curves")), None)
    if hist_md: sec.append(hist_md.replace("## Training curves", f"## {n}. Training curves", 1)); n += 1
    sec.append(f"## {n}. Where the boundaries land\n\n![boundary error](fig/boundary_error.png)\n\n" + _boundary_md(bstats, labels, edge_taus) +
               "\n**Caption shape (report only).** Durations of the scored captions, and the reliable units each caption holds (at least half of each); "
               "the gold units give the reference durations. `tables/caption_shape.csv`.\n\n" + _shape_md(shape) +
               "\n**Display-cue share (report only).** Per τ: of the caption edges that lie inside a unit and more than τ from both of its edges (a pair with "
               "such an edge counts in no cell of τ), the share within τ of a junction between 2 display cues of that unit, and the share that chance gives (the time near a junction over "
               "the inner time of the unit). A share above chance means the system cuts inside units where the captioner broke the display. `tables/cue_breaks.csv`.\n\n" +
               _cue_md(cues, labels) +
               "\n**Edge hits by boundary type (report only).** Each caption unit gives a start edge and an end edge. The boundary at an edge takes its type "
               f"from the nearest unit on that side in the same stream: `abutting` (a caption unit at most the chunk gap, {gap_s:g} s, away, so the edge is inside "
               "a chunk), `quarantine` (a quarantined unit at most the chunk gap away), `pause` (a longer gap, or no unit and the stream edge farther away) or "
               "`stream` (no unit, and the stream edge within the chunk gap). Per side, the gold edges and the caption edges of the same side pair one to one, "
               "the smallest time difference first, within τ. Each cell gives the share of the gold edges paired at each τ and their count; the last column gives "
               "the share of the caption edges paired and their count. Each edge counts alone, so a correct merge of 2 abutting units misses their 2 inner "
               "edges. `tables/edge_boundaries.csv`.\n\n" + _edge_boundary_md(bounds, labels)); n += 1
    sec.append(f"## {n}. Breakdown by sentence length and by unseen words\n\n![by length](fig/by_length.png)\n\n![by oov](fig/by_oov.png)\n\n![bleu by edge error](fig/bleu_by_edge_error.png)\n\n"
               "These figures use a per-unit score, `localized BLEU-4`: in each cell of the grid, the sentence BLEU-4 of the caption whose pair with the unit's run counts, against the joined text of the run "
               "(a joined unit takes the score of its run), and 0 for a unit that is not located (not matched or joined). It gives every caption unit its own number, which the breakdowns need. "
               "The length and unseen-word figures average over the cells, and the middle length panel reads the strictest cell (the smallest τ and the largest t); the last figure reads the "
               "located units in the loosest cell (the largest τ and the smallest t), by the larger edge error of their caption. It is not the DVC column.\n"); n += 1
    sec.append(f"## {n}. What the extra sentences cost\n\n" + _cost_md(cost, rth) + "\n" + _shared_md(shared, reference) +
               "\nHow to read: the second row of each system removes every event that locates nothing at the loosest threshold and scores again. The change is what those events cost.\n"); n += 1
    sec.append(f"## {n}. Decision metrics (fixed before any trained arm)\n\n" + _decision_md(edge_rows, long_form, by_ref_class, reference) +
               "\nHow to read: (1) Edge recall: the share of gold starts and ends with a predicted edge within δ; ends split by the sampler's rule into `pause after` (the next unit starts more than δ later) and `abutting`. (2) Translation by the reference's span class: localized BLEU-4 against the reference in each cell of the grid, per class of the reference's span that holds the unit (mean difference per unit). (3) ReSCUE's axes: long-form BLEU-4 (per video, events in commit order against the gold in time order, one corpus; events at least half inside an uncaptioned or quarantined stretch dropped first), and LAAL/AL per unit from the commit time of the event that holds it (committed text never changes, so NE = 0).\n"); n += 1
    if held:
        sec.append(f"## {n}. Latency of every caption unit\n\n" + _held_md(held) +
                   "\nHow to read: per caption unit, the commit time of the first event that holds at least half of the unit, minus the unit's end. The headline latency uses only one-to-one matched pairs, so a unit inside a merged event drops out of it; here it counts, with the merged event's commit time. `covered` is the share of units that some event holds.\n"); n += 1
    sec.append(f"## {n}. Differences against `{reference}`\n\n" + _diff_md(diff) +
               "\nHow to read: each row is a headline number of the system, the same number of the reference, and their difference. `localized BLEU-4` is the mean per caption unit in each cell of the grid (a unit that is not located scores 0), so it charges missed units, which DVC does not. `fixed` and `broken` count the units that one system locates in the cell (matched or joined) and the other does not.\n"); n += 1
    for m in md_opt:
        if m.startswith(("## Segmentation only", "## Training curves")): continue
        sec.append(m.replace("## ", f"## {n}. ", 1)); n += 1
    md = intro + [x + "\n---\n" for x in sec]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")   # results.md: the generated skeleton; report.md stays yours
    print(f"[report] wrote {out}/results.md, {len(list(fig_dir.glob('*.png')))} figures, {len(list(tab_dir.iterdir()))} tables", flush=True)
    return {"out": str(out), "systems": labels, "thresholds": thresholds}


# ---------------------------------------------------------------- progress report for a supervisor
SEG_NAME = {"gold": "true cuts", "moryossef": "EMNLP", "s1": "S1", "joint-ar": "joint AR", "joint-dlm": "joint DLM"}
TRANS_NAME = {"pretrimmed": "pretrimmed", "pretrimmed-dlm": "pretrimmed DLM", "ar": "joint AR", "dlm": "joint DLM"}
FAM_ORACLE, FAM_OFFLINE, FAM_ONLINE, FAM_JOINT, FAM_CTRL = (
    "Oracle spans", "Offline cascade", "Online cascade", "Our joint model", "Same-span control")
FAM_ORDER = [FAM_ORACLE, FAM_OFFLINE, FAM_ONLINE, FAM_JOINT, FAM_CTRL]


def _family(when: str, spans: str, trans: str) -> str:
    if spans == "gold": return FAM_ORACLE                              # a reference, never a cascade
    if spans.startswith("joint"): return FAM_JOINT if spans == f"joint-{trans}" else FAM_CTRL
    return FAM_OFFLINE if when == "offline" else FAM_ONLINE


def _discover(lang: str) -> list[dict]:
    """Every finished row for one language, by file name. Missing files are simply skipped.

    The name shape is evaluation.predictions.output_stem: {when}_{spans}_{decode}_{translator}_{lang}_{split}, in outputs/{when}/. The four
    vocabularies are CLOSED here on purpose: an ablation row carries a suffixed translator token (`dlm-<stem>`) and a
    segmentation-only run carries `none`, so neither reaches the results table, and a file under any other name
    shape cannot be read as a row.
    """
    out = []
    for when in ("offline", "online"):
        for spans in ("gold", "moryossef", "s1", "joint-ar", "joint-dlm"):
            for dec in ("none", "argmax", "viterbi", "duration"):
                for trans in ("pretrimmed", "pretrimmed-dlm", "ar", "dlm"):
                    p = output_dir(when) / f"{when}_{spans}_{dec}_{trans}_{lang}_test.json"
                    if not p.exists(): continue
                    who = SEG_NAME[spans] + ("" if dec == "none" else f" {dec}")
                    label = f"{who}, {when}" if spans == f"joint-{trans}" else f"{who} + {TRANS_NAME[trans]}, {when}"
                    out.append({"family": _family(when, spans, trans), "seg": spans, "dec": dec, "when": when,
                                "trans": trans, "label": label, "path": str(p)})
    # `_html` spans one cell over each family's rows, so they must arrive contiguous and in reading order.
    out.sort(key=lambda r: (FAM_ORDER.index(r["family"]), r["when"] == "online", r["seg"], r["trans"]))
    return out


def _segmenter_rows(lang: str) -> dict:
    """The _scores.json of every no-text row of both segmenters for one language, by file name ({label: path}, offline first): README
    §7.1 offline, §7.4 online, each with the 3 decodes (argmax, viterbi, duration). The label names the segmenter, the decode and the
    access. Missing files are skipped."""
    return {f"{SEG_NAME[arch]} {dec}, {when}": p for when in ("offline", "online") for arch in ("moryossef", "s1")
            for dec in ("argmax", "viterbi", "duration")
            for p in [output_dir(when) / f"{when}_{arch}_{dec}_none_{lang}_test_scores.json"] if p.exists()}


def _html(rows, thresholds):
    """One HTML table, experiments merged by family: the main score first (the F1 and the DVC BLEU-4 of the tIoU x edge-limit grid, avg),
    then the secondary densevid scores per threshold and their avg (the F1 of the mean precision and recall; r[key]["avg"]). Bold is
    the best value in the column WITHIN its own family:
    offline and online are different settings, so their best values are not compared with each other. The true cuts are
    the rows of the family FAM_ORACLE (row 1: the pretrimmed translator; row 2: the arms)."""
    th = "".join(f"<th>{t:g}</th>" for t in thresholds) + "<th>avg</th>"
    h = ["<table>", "<thead>",
         f'<tr><th rowspan="2">Experiment</th><th rowspan="2">Boundaries from</th><th rowspan="2">Decoding</th>'
         f'<th rowspan="2">When it must decide</th><th rowspan="2">Translator</th>'
         f'<th colspan="2">Main score: tIoU x edge-limit grid &nbsp;(avg over τ and tIoU)</th>'
         f'<th colspan="{len(thresholds)+1}">Secondary: sentence tIoU F1 (densevid)</th>'
         f'<th colspan="{len(thresholds)+1}">Secondary: tIoU DVC BLEU-4 (densevid)</th></tr>',
         f"<tr><th>F1</th><th>DVC BLEU-4</th>{th}{th}</tr>", "</thead>", "<tbody>"]
    body = rows
    # (key, tIoU threshold or avg, format); a main score is 1 number (no threshold)
    cols = [("edge_f1", None, "{:.3f}"), ("edge_bleu4", None, "{:.2f}"),
            *[(key, t, fmt) for key, fmt in (("f1", "{:.3f}"), ("bleu", "{:.2f}")) for t in [*thresholds, "avg"]]]
    val = lambda r, key, t: r[key] if t is None else r[key][t]
    best = {}
    for fam in {r["family"] for r in rows}:
        fam_rows = [r for r in rows if r["family"] == fam]
        for key, t, _ in cols: best[(fam, key, t)] = max(val(r, key, t) for r in fam_rows)
    fam_n = {}
    for r in body: fam_n[r["family"]] = fam_n.get(r["family"], 0) + 1
    seen = set()
    for r in body:
        cells = []
        if r["family"] not in seen:
            seen.add(r["family"]); cells.append(f'<td rowspan="{fam_n[r["family"]]}"><b>{r["family"]}</b></td>')
        cells += [f'<td>{SEG_NAME.get(r["seg"], r["seg"])}</td>', f'<td>{r["dec"]}</td>', f'<td>{r["when"]}</td>',
                  f'<td>{TRANS_NAME.get(r["trans"], r["trans"])}</td>']
        for key, t, fmt in cols:
            v = val(r, key, t); s = fmt.format(v)
            if abs(v - best.get((r["family"], key, t), v + 1)) < 1e-9: s = f"<b>{s}</b>"
            cells.append(f"<td>{s}</td>")
        h.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(h + ["</tbody>", "</table>"])


def progress(args) -> dict:
    data_cfg = load_yaml(args.data_config)
    thresholds = [float(t) for t in load_yaml(args.eval_config)["scoring"]["tiou_thresholds"]]
    tau = float(args.tiou_threshold)
    out = Path(args.out) if args.out else output_dir("report") / "progress"; fig_dir, tab_dir = out / "fig", out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)
    for old in list(fig_dir.glob("*.png")) + list(tab_dir.glob("*.csv")): old.unlink()
    F.metric_explained(fig_dir / "metric_explained.png")

    per_lang = {}
    for lang in args.language:
        specs = _discover(lang)
        if not specs: print(f"[progress] {lang}: nothing found, skipped", flush=True); continue
        inference_cfg = load_yaml(args.inference_config)
        records, _ = load_language_records(data_cfg, lang, split="test")
        gold = gold_events(records); n_gold = sum(len(v) for v in gold.values())
        char_level = text_level(target_language(data_cfg, lang), cfg_get(data_cfg, "languages", lang, "lowercase_references"))
        scoring_cfg = load_yaml(args.eval_config)["scoring"]   # the main scores: the tIoU x edge-limit grid
        chunks, edge_taus = gold_chunks(records, float(scoring_cfg["chunk_gap_s"])), [float(t) for t in scoring_cfg["edge_taus_s"]]
        # Λ_min in frames of each stream's own rate, as the FSM applies it, and back to seconds, per source video (the gold key).
        lam_s = lambda_min_s(inference_cfg)
        floor_s = {r.pose.video_id: seconds_to_frames(lam_s, r.pose.fps) / float(r.pose.fps) for r in records}
        cap_s = float(inference_cfg["buffer_cap_s"])
        train_records, _ = load_language_records(data_cfg, lang, split="train")
        train_vocab = {t for r in train_records for sp in r.sentences if getattr(sp, "reliable", True)
                       for t in (_vocab_token(w) for w in sp.text.split()) if t}

        raw_events, systems, scores, tables, errors, bstats, ratio, rows = {}, {}, {}, {}, {}, {}, {}, []
        for spec in specs:
            spec["fresh"] = warn_annotation_mismatch(spec["path"], records, f"{lang}/{spec['label']}")   # made under the current labels
            raw_events[spec["label"]] = load_event_predictions(spec["path"])
            pred = scoreable_predictions(raw_events[spec["label"]], records, tag=f"{lang}/{spec['label']}")   # never cut (the method rule)
            systems[spec["label"]] = pred
            dense = evaluate_predicted_events(pred, gold, thresholds, char_level=char_level, densevid=True)
            sc = {**{r["tiou"]: r for r in dense["by_tiou"]}, "avg": dense["avg"]}   # the densevid scores per threshold and their avg
            errors[spec["label"]] = edge_error_types(pred, chunks, edge_taus, thresholds)   # the one error table, per cell of the grid
            for t in thresholds:   # a tIoU score: the corpus BLEU-4 of the one-to-one tIoU pairs
                tiou_pairs = located_pairs(pred, gold, t)
                sc[t]["located_corpus_bleu4"] = text_scores([h for _, _, h, _ in tiou_pairs], [r for _, _, _, r in tiou_pairs], char_level,
                                                            keys=("bleu4",))["bleu4"] if tiou_pairs else 0.0
            unit_rows = unit_outcomes(chunks, pred, errors[spec["label"]], char_level)   # each unit's error type per cell of the grid
            for r in unit_rows:
                toks = [t for t in (_vocab_token(w) for w in r["ref"].split()) if t]
                r["oov_rate"] = round(sum(1 for w in toks if w not in train_vocab) / max(1, len(toks)), 3)
            tables[spec["label"]] = {r["gold_id"]: r for r in unit_rows}
            bstats[spec["label"]] = boundary_stats(errors[spec["label"]])
            ratio[spec["label"]] = sum(len(v) for v in pred.values()) / max(1, n_gold)
            scores[spec["label"]] = sc
            edge = edge_chunk_scores(pred, chunks, edge_taus, thresholds, char_level, bleurt_checkpoint=None, keys=("bleu4",))   # only BLEU-4 is printed
            # The densevid scores per threshold and their avg (the F1 of the mean precision and recall), and the main score (avg of the grid).
            rows.append({**spec, "f1": {t: sc[t]["f1"] for t in [*thresholds, "avg"]}, "bleu": {t: sc[t]["dvc"]["bleu4"] for t in [*thresholds, "avg"]},
                         "edge_f1": edge["avg"]["f1"], "edge_bleu4": edge["avg"]["dvc"]["bleu4"] if edge["avg"]["dvc"] else 0.0})
            print(f"[progress] {lang} {spec['label']}: {_main_line(edge, uncaptioned_units(pred, gold))}", flush=True)
        labels = [r["label"] for r in rows]
        # The deployed system: the joint arm online, translating its own spans. Selected by FIELDS, because a label
        # spelling falls through silently to another row (the AR twin, or a control) as soon as it changes.
        reference = next((r["label"] for r in rows if r["family"] == FAM_JOINT and r["when"] == "online"
                          and r["trans"] == "dlm"), None) or labels[-1]

        # the reference on the true cuts and the protocol section: row 1 (the pretrimmed translator on the gold spans), when it exists
        ceiling, proto, orc, sweep_rows = None, None, None, []
        row1 = next((r for r in rows if r["seg"] == "gold" and r["trans"] == "pretrimmed" and r["when"] == "offline"), None)
        # A stale row 1 (other annotations, warned above) cannot pair with the current gold: the protocol section is left out.
        if row1 and row1["fresh"]:
            orc = oracle_pairs(gold, raw_events[row1["label"]])
            proto = decompose(gold, systems, thresholds, char_level, oracle=orc)
            if proto["oracle"]:
                orow = oracle_row(gold, orc, records, thresholds, char_level)
                ceiling = {"f1_by_t": {t: orow["f1"][t] for t in thresholds}, "bleu_by_t": {t: orow["dvc"][t] for t in thresholds},
                           "f1": orow["f1"][thresholds[0]], "bleu": orow["dvc"][thresholds[0]], "kept": orow["n_kept"], "n_gold": orow["n_gold"],
                           "edge_bleu4": row1["edge_bleu4"],   # the main text score on the true cuts (row 1)
                           "corpus": proto["oracle"]["corpus_bleu4"], "mean": proto["oracle"]["per_video_mean"],
                           "median": float(np.median(list(proto["oracle"]["per_video"].values()))), "weighted": proto["oracle"]["size_weighted_mean"],
                           "lambda_min_s": lam_s, "floor_s": [min(floor_s.values()), max(floor_s.values())],
                           "attainable": attainable_recall(gold, floor_s, thresholds)}
                two = two_video_example(gold, systems, orc, proto["oracle"], tau, char_level)
                big_win = (gold[two["big"]][0].start_s - 1, gold[two["big"]][0].start_s + 59)
                _s1 = {two["system"]: systems[two["system"]]}; _r1 = {two["system"]: raw_events[two["system"]]}
                F.protocol_two_videos(two, {two["small"]: case_view(gold, _s1, two["small"], tau, char_level, oracle=orc, raw=_r1),
                                            two["big"]: case_view(gold, _s1, two["big"], tau, char_level, window=big_win, oracle=orc, raw=_r1)},
                                      proto, fig_dir / f"{lang}_protocol.png", LANG_NAME.get(lang, lang), orow)
                ceiling["two"] = two

        # figures
        ex_vid = max(gold, key=lambda v: -abs(len(gold[v]) - 6))
        F.task_and_experiments(fig_dir / f"{lang}_task.png", ([(g.start_s, g.end_s, g.text) for g in gold[ex_vid]], ex_vid), lang,
                               f"{len(gold)} videos", f"{n_gold} sentences", LANG_NAME.get(lang, lang))
        F.results_overview(rows, thresholds, fig_dir / f"{lang}_results.png", lang, ceiling)
        F.located_vs_quality(scores, labels, thresholds, fig_dir / f"{lang}_stricter.png", n_gold)
        F.failure_overview(errors, labels, fig_dir / f"{lang}_failures.png", lang, n_gold, ratio)
        F.boundary_error(bstats, labels, fig_dir / f"{lang}_boundary.png", edge_taus)
        edges = [0.0, 2.0, 4.0, 6.0, 9.0, 14.0, cap_s, float("inf")]; blab = ["0-2 s", "2-4 s", "4-6 s", "6-9 s", "9-14 s", "14 s-cap", "> cap"]
        F.by_length(tables, labels, edges, blab, fig_dir / f"{lang}_by_length.png", cap_s)
        F.by_oov(tables, labels, fig_dir / f"{lang}_by_oov.png")
        seval = {label: _segmentation_only(p, records, scoring_cfg) for label, p in _segmenter_rows(lang).items()}
        if seval: F.frame_vs_sentence(seval, thresholds, fig_dir / f"{lang}_segmentation_only.png")
        sweep = output_dir("analysis") / f"boundary_sweep_pretrimmed_{lang}.json"   # the pretrimmed AR translator's sweep, when it was run
        if sweep.exists() and warn_annotation_mismatch(sweep, records, f"{lang} boundary sweep (its confidence panel is left out)"):
            sweep_rows = json.load(open(sweep, encoding="utf-8")).get("rows", [])
        if sweep_rows:
            sub = [r for r in sweep_rows if r.get("prediction") is not None and r.get("reference")][::max(1, len(sweep_rows) // 4000)]
            if sub:
                b = sentence_bleu_scores([r["prediction"] for r in sub], [r["reference"] for r in sub], char_level)
                F.confidence_calibration({LANG_NAME.get(lang, lang): list(zip([float(r.get("mean_confidence", 0)) for r in sub], b))},
                                         fig_dir / f"{lang}_confidence.png")

        # Real videos, drawn for two experiments only: more lanes make the picture unreadable.
        # the best offline cascade and our own online system: the two settings the professor compares, and enough
        # lanes to read. More systems make the picture unreadable.
        off = [r for r in rows if r["family"] == FAM_OFFLINE]
        shown = [max(off or rows, key=lambda r: r["edge_f1"])["label"]]   # the best by the main score
        if reference not in shown: shown.append(reference)
        case_sys = {l: systems[l] for l in shown}
        cases = []
        picks = showcase_videos(gold, case_sys, thresholds, tau, proto["oracle"] if proto and proto["oracle"] else None)
        for spec in (args.case_videos or []):
            cl, _, rest = spec.partition("="); vid, _, note = rest.partition(":")
            if cl == lang and vid in gold and vid not in {c["video_id"] for c in picks}:
                picks.append({"video_id": vid, "window": None, "note": note or "chosen by hand", "tau": tau})
        for c in picks[:6]:
            ctau = float(c.get("tau", tau))
            view = case_view(gold, case_sys, c["video_id"], ctau, char_level, window=c["window"], raw={l: raw_events[l] for l in shown})
            name = f"{lang}_case_{c['video_id']}.png"
            F.case_figure(view, ctau, fig_dir / name, note=c["note"], language=LANG_NAME.get(lang, lang))
            cases.append((c["video_id"], c["note"], name))

        # The densevid scores of the table below, with the names of `results` tables/densevid.csv (`f1`, `dvc_bleu4`): 1 line per
        # threshold, then their avg.
        _write_csv(tab_dir / f"{lang}_densevid.csv", [{"system": r["label"], "family": r["family"], "tiou": _name(t), "f1": round(r["f1"][t], 4),
                                                   "dvc_bleu4": round(r["bleu"][t], 3)} for r in rows for t in [*thresholds, "avg"]])
        per_lang[lang] = {"rows": rows, "ceiling": ceiling, "labels": labels, "reference": reference,
                          "n_gold": n_gold, "n_videos": len(gold), "errors": errors, "ratio": ratio, "cases": cases,
                          "seval": seval, "has_conf": bool(sweep_rows), "thresholds": thresholds}

    langs = list(per_lang)
    if not langs: raise SystemExit("no finished experiments found under outputs/")
    (out / "report.md").write_text(_progress_md(per_lang, thresholds, tau), encoding="utf-8")
    (out / "talk_script.md").write_text(_progress_talk(per_lang, thresholds, tau), encoding="utf-8")
    print(f"[progress] wrote {out}/report.md and {out}/talk_script.md, {len(list(fig_dir.glob('*.png')))} figures", flush=True)
    return {"out": str(out), "languages": langs}


def _progress_sections(per_lang, ths, tau):
    """The report, as an ordered list of sections. Each one carries its Markdown and its speaking lines, so the
    report and the script always have the same sections and the same numbers."""
    langs = list(per_lang); L = lambda l: LANG_NAME.get(l, l)
    S = []

    S.append({"title": "The task, and what changes between the experiments",
              "md": [f"![](fig/{l}_task.png)\n" for l in langs],
              "talk": ["Today I want to show you all the results I have so far, and where the problems are.", "",
                       "First the task. The model gets one long pose video with no marks in it. It must cut the video into "
                       "sentences, and translate every sentence. So a mistake in the cut is also a mistake in the translation.", "",
                       "In the second panel, the left column is the same in every experiment: the same test videos and the "
                       "same score. Only 4 "
                       "things change: who cuts the video, how the cuts are chosen, when the model must decide, and who "
                       "translates.", "",
                       "That gives 4 groups. The cascade cuts first and then translates each part. The online cascade does the "
                       "same, but it must decide while it is watching. Our joint model cuts and translates by itself. And "
                       "perfect boundaries is a reference condition: I give the true cuts to the pretrimmed translator."]})

    S.append({"title": "How the scores work",
              "md": ["![](fig/metric_explained.png)\n",
                     "Localization F1 counts the caption units we find. BLEU-4 is the text score inside them. A prediction "
                     "that finds nothing is scored against a random string, so extra sentences pull the score down.\n",
                     "The main score is the F1 of the tIoU x edge-limit grid. A prediction may cover a run of consecutive caption "
                     "units with no pause between them, so a correct merge of 2 touching sentences can count. Per video, the predictions "
                     "and the runs pair one to one by tIoU. A pair counts at tIoU threshold t when its tIoU is at least t, and at edge "
                     "limit τ when both of its edges are within τ of the edges of the run: τ is an extra limit on a pair that passed the "
                     "tIoU test. Each cell (τ, t) gives a precision and a recall; an avg takes the mean precision and the mean recall of "
                     "its cells and their F1, as in our NeurIPS paper. The main score is the F1 at τ avg and tIoU avg. The figure shows the "
                     "tIoU pairing of the secondary densevid scores, which pairs each prediction with at most one caption unit.\n"],
              "talk": ["Next the metric, because everything later depends on it.", "",
                       "At the top are the caption units from the subtitles. Very often there is no gap between one sentence "
                       "and the next.", "",
                       "In the middle is what a model produces: time intervals with text inside. This is the same output as "
                       "dense video captioning, so I use the dense video captioning score, like in our previous paper.", "",
                       "At the bottom is the secondary scoring, which I keep for comparison with other papers. It pairs each "
                       "prediction with at most one caption unit, by time overlap, or tIoU. Because one prediction can take only one "
                       "caption unit, a prediction that covers 2 true sentences will lose one of them.", "",
                       "The main score uses the same tIoU test, but a prediction may cover a run of consecutive caption units with no "
                       "pause between them, so a correct merge of 2 touching sentences can still count. A pair counts at a tIoU threshold "
                       "when its tIoU passes it, and at an edge limit τ when both of its edges are also within τ seconds of the edges of "
                       "the run. So τ makes the test stricter, never looser. I use 3 edge limits and 3 tIoU thresholds, which gives 9 cells, "
                       "and the main score is the F1 of their average precision and average recall. The translation score is BLEU-4 inside "
                       "the counted pairs. A prediction that finds nothing is scored against a random string, so extra sentences make the "
                       "score lower."]})

    if any(per_lang[l]["seval"] for l in langs):
        md, talk = [], []
        for l in langs:
            ev = per_lang[l]["seval"]
            if not ev: continue
            md += [f"### {L(l)}\n", f"![](fig/{l}_segmentation_only.png)\n", _segmentation_md(ev, ths), ""]
            main = {k: v["main"]["edge_chunk"]["avg"]["f1"] for k, v in ev.items()}   # the main score of each setting
            first = {k: next((r["f1"] for r in v["main"]["edge_chunk"]["by_tiou"] if float(r["tiou"]) == ths[0]), 0) for k, v in ev.items()}
            best = max(main, key=main.get)
            talk += [f"For {L(l)}: the best setting by the main score, the F1 of the tIoU x edge-limit grid, is {best} ({main[best]:.3f}). The "
                     f"segmenters reach a signing/non-signing F1 of about {100*np.mean([v['frame']['bio_f1'] for v in ev.values()]):.0f} %. At "
                     f"tIoU {ths[0]:g}, the loosest threshold (the mean over the edge limits), their best F1 is only about {100*max(first.values()):.0f} %.", ""]
        md.append("A frame score can stay high while sentences are missed: at a wrong sentence end the frames are still "
                  "signing. Compare the two columns, not one alone.\n")
        S.append({"title": "Segmentation only, before any translation", "md": md, "talk": talk})

    md = ["Bold is the best value in the column **inside its own group**. Offline and online are different settings, so "
          "their best values are not compared with each other.\n"]
    talk = ["Now the full systems, with translation. I put every experiment in one table so we can choose the best option "
            "for the paper.", "", "Read the oracle-spans rows first. They are the same translators evaluated with GT cuts.", ""]
    for l in langs:
        p = per_lang[l]; rows = p["rows"]
        md += [f"### {L(l)}\n", _html(rows, ths), "\n"]
        if p["ceiling"]:
            c = p["ceiling"]
            md.append("\n*Oracle spans* gives the true cuts to each translator under the same scoring protocol (row 1: the pretrimmed "
                      "translator). It is a reference condition, not a mathematical upper bound on caption quality. Its localization is "
                      "1.000 by construction: scoring charges every system for every event it emits and deletes none, so the true cuts "
                      "score perfectly.\n")
            md += _attainable_md(c, L(l))
        md.append(f"\n![](fig/{l}_results.png)\n")
        # The spoken numbers are the main score (the F1 and the DVC BLEU-4 of the tIoU x edge-limit grid, avg), as _findings.
        bf = max(_systems(rows), key=lambda r: r["edge_f1"]); bb = max(_systems(rows), key=lambda r: r["edge_bleu4"])
        line = (f"For {L(l)}: the best localization is **{bf['label']}** with main F1 {bf['edge_f1']:.3f} (the F1 of the tIoU x edge-limit "
                f"grid, avg). The best text is **{bb['label']}** with main DVC BLEU-4 {bb['edge_bleu4']:.2f}.")
        if bf["label"] != bb["label"]:
            line += " These are not the same experiment, and I would like your opinion on which one the paper should lead with."
        talk += [line, ""]
        if p["ceiling"]:
            cb = p["ceiling"]["edge_bleu4"]
            talk += [f"With perfect cuts the same translator reaches main DVC BLEU-4 {cb:.2f}. So the wrong boundaries cost us about "
                     f"{100*(1 - bb['edge_bleu4']/cb):.0f} % of the text score. That gap is the whole problem of this project.", ""]
        off = [r for r in rows if r["family"] == FAM_OFFLINE]; on = [r for r in rows if r["family"] == FAM_ONLINE]
        if off and on:
            bo = max(off, key=lambda r: r["edge_f1"]); bn = max(on, key=lambda r: r["edge_f1"])
            talk += [f"Online against offline: main F1 {bn['edge_f1']:.3f} online against {bo['edge_f1']:.3f} offline.", ""]
        pl = [r for r in rows if r["dec"] in ("argmax", "viterbi")]; du = [r for r in rows if r["dec"] == "duration"]
        if pl and du:
            dp = np.mean([r["edge_f1"] for r in du]) - np.mean([r["edge_f1"] for r in pl])
            talk += [f"Duration decoding minus argmax and viterbi decoding: {dp:+.3f} main F1 on average.", ""]
        jt = [r for r in rows if r["family"] == FAM_JOINT]
        if jt:
            bj = max(jt, key=lambda r: r["edge_bleu4"])
            cas = [r for r in rows if r["family"] in (FAM_OFFLINE, FAM_ONLINE)]
            if cas:
                bc = max(cas, key=lambda r: r["edge_bleu4"])
                talk += [f"Best joint model against best cascade: main DVC BLEU-4 {bj['edge_bleu4']:.2f} against {bc['edge_bleu4']:.2f}.", ""]
    S.append({"title": "All results in one place", "md": md, "talk": talk})

    S.append({"title": "What the numbers say", "md": [_findings(per_lang, ths, tau), ""],
              "talk": ["These are the same points written down, so you can read them later without me."]})

    S.append({"title": "What changes when the match gets stricter",
              "md": ["Two effects work against each other. A stricter match keeps fewer sentences, and the ones it keeps read "
                     "better. The first effect is larger, so the headline falls.\n"] + [f"![](fig/{l}_stricter.png)\n" for l in langs],
              "talk": ["I want to explain this one carefully, because at first sight it contradicts our previous paper.", "",
                       "Our previous paper reported that a higher tIoU gives a better BLEU, and that is still true here. The "
                       "located-only text score can rise as the threshold selects a different subset. The captions and clips "
                       "themselves have not changed.", "",
                       "But the number of sentences we find falls much faster. Every sentence that falls under the threshold "
                       "becomes a wrong pair, and it is scored against a random string. The second effect is larger, so the "
                       "headline number falls."]})

    talk = []
    for l in langs:
        p = per_lang[l]; bf = max(_systems(p["rows"]), key=lambda r: r["edge_f1"]); tx = p["errors"][bf["label"]]["units_avg"]
        talk += [f"For {L(l)}: in the best system, {100*(tx['matched'] + tx['joined']):.0f} % of the caption units are found "
                 f"({100*tx['joined']:.0f} % of them inside a correct joined caption), {100*tx['cut']:.0f} % are cut inside, "
                 f"{100*tx['bridged']:.0f} % sit in a caption that crosses a pause, {100*tx['offset']:.0f} % sit in a caption with a "
                 f"wrong edge, and {100*tx['missed']:.0f} % have no caption.", ""]
    talk += ["These types come from the counted pairs of the main score, in each cell of the grid; here they are the mean over the cells. "
             "They do not establish whether a boundary cue is visible in the motion."]
    S.append({"title": "Where the caption units are lost",
              "md": [_UNIT_TYPE_WORDS + "\n"] + [f"![](fig/{l}_failures.png)\n" for l in langs],
              "talk": talk})

    S.append({"title": "Where the boundaries land",
              "md": ["Only the captions that hold exactly one caption unit, which no other caption holds, are counted. Positive means "
                     "late. An end more than τ late fails the edge limit τ by itself.\n"] + [f"![](fig/{l}_boundary.png)\n" for l in langs],
              "talk": ["The left panel gives the median start and end errors of the captions that hold one caption unit. The right "
                       "panel gives how often the end alone is more than τ late, so that the caption fails the match at that τ."]})

    S.append({"title": "Sentence length, and words we never saw in training",
              "md": [f"![](fig/{l}_by_length.png)\n![](fig/{l}_by_oov.png)\n" for l in langs] +
                    ["Short sentences are the hard ones. Unseen words hurt every system the same way, so a gap that stays in "
                     "the high-OOV bins is a method difference, and one that closes there is a data limit.\n"],
              "talk": ["Short sentences are the hard ones: under 2 seconds, every system finds less than a third of them. "
                       "Unseen words hurt every system in the same way, so the differences between systems are method "
                       "differences, not data."]})

    md = ["These videos are chosen by rule, not by hand. Two experiments are drawn: our best offline cascade and our own "
          "online system.\n"]
    for l in langs:
        for vid, note, name in per_lang[l]["cases"]: md += [f"**{L(l)} {vid}** - {note}\n", f"![](fig/{name})\n"]
    talk = ["Now some real test videos. I draw only 2 experiments so it stays readable. Green means the caption was scored "
            "against a caption unit, orange means it was scored against a random string.", ""]
    for l in langs:
        for vid, note, _ in per_lang[l]["cases"]: talk.append(f"- {L(l)} {vid}: {note}")
    talk += ["Single-letter reference tokens may identify useful cases to inspect, but do not prove that the video lacks boundary cues."]
    S.append({"title": "Real videos", "md": md, "talk": talk})

    if any(per_lang[l]["ceiling"] for l in langs):
        md = ["The BLEU-4 column is a mean over videos: each video is scored alone, then the video scores are averaged with "
              "equal weight. A 2-sentence video counts as much as a 600-sentence video. So it must never be compared with a "
              "pooled corpus BLEU-4, which adds every sentence into one count.\n"]
        talk = ["I want to flag one thing before we compare numbers across tables. The BLEU-4 column here is a mean over "
                "videos: every video is scored alone, and then the video scores are averaged with equal weight. So a "
                "2-sentence video counts as much as a 600-sentence video.", ""]
        for l in langs:
            c = per_lang[l]["ceiling"]
            if not c: continue
            md += [f"### {L(l)}\n",
                   f"The same perfect-cut captions give **{c['corpus']:.2f}** as one corpus and **{c['mean']:.2f}** as a mean "
                   f"over videos (median video {c['median']:.2f}). The GT-boundary reference in the table above, **{c['bleu']:.2f}**, is the "
                   f"mean version, so it is the one our systems may be compared with.\n", f"![](fig/{l}_protocol.png)\n"]
            talk += [f"For {L(l)}, the same perfect-cut captions give {c['corpus']:.2f} as one corpus and {c['mean']:.2f} as a "
                     f"mean over videos. Same captions, 2 different answers. So when we compare our systems to the pretrimmed "
                     f"translator on the true cuts, we must use the mean version, {c['bleu']:.2f}, not the corpus one.", ""]
        S.append({"title": "One warning about the BLEU-4 column", "md": md, "talk": talk})

    if any(per_lang[l]["has_conf"] for l in langs):
        S.append({"title": "Does the model know when it is wrong?",
                  "md": ["Each point is a group of clips with similar model confidence. A rising curve means the confidence "
                         "carries information. The FSM commit does not read it; only the offline confidence-based reveal policies (infer/stability.py) do.\n"] +
                        [f"![](fig/{l}_confidence.png)\n" for l in langs if per_lang[l]["has_conf"]],
                  "talk": ["The last figure asks if the model's own confidence tells us anything about quality. The commit does "
                           "not use it: the FSM commits on boundary stability alone. It matters only for the reveal policies we test offline."]})
    return S


def _progress_md(per_lang, ths, tau):
    langs = list(per_lang)
    sizes = ", ".join(f"{LANG_NAME.get(l, l)} {per_lang[l]['n_videos']} videos and {per_lang[l]['n_gold']} sentences" for l in langs)
    m = ["# Sign language translation: progress report\n",
         f"Test sets: {sizes}. Every experiment below uses the same videos, the same caption units and the same score.\n"]
    for i, sec in enumerate(_progress_sections(per_lang, ths, tau), 1):
        m.append(f"## {i}. {sec['title']}\n"); m += sec["md"]
    return "\n".join(m)


def _progress_talk(per_lang, ths, tau):
    t = ["# Presentation script for `report.md`", "",
         "One part per section of the report. Read it in order.", ""]
    for i, sec in enumerate(_progress_sections(per_lang, ths, tau), 1):
        t += [f"## §{i} - {sec['title']}", ""] + sec["talk"] + [""]
    t += ["## Questions for you", "",
          "1. Which operating point should the paper lead with: find more sentences, or be more exact?",
          "2. Is the GT-boundary reference the right control to put next to the main table?",
          "3. Should I spend the next weeks on the boundary model, or on the translator?", ""]
    return "\n".join(t)


LANG_NAME = {"asf": "Auslan (asf)", "bfi": "BSL (bfi)", "ase": "ASL (ase)"}


def _systems(rows): return [r for r in rows if r["family"] != FAM_ORACLE] or rows   # the best-system picks skip the true cuts (rows 1, 2)


def _findings(per_lang, ths, tau):
    """Short statements, each one read off the table, with the experiment named: the main score (the main F1 and DVC BLEU-4: the F1
    and the DVC BLEU-4 of the tIoU x edge-limit grid, avg)."""
    out = []
    for l, p in per_lang.items():
        rows = p["rows"]
        bf = max(_systems(rows), key=lambda r: r["edge_f1"]); bb = max(_systems(rows), key=lambda r: r["edge_bleu4"])
        out.append(f"**{LANG_NAME.get(l, l)}.**")
        if bf["label"] == bb["label"]:
            out.append(f"- Best overall: **{bf['label']}**, main F1 {bf['edge_f1']:.3f} and main DVC BLEU-4 {bb['edge_bleu4']:.2f} (tIoU x edge-limit grid, avg).")
        else:
            out.append(f"- Best boundaries: **{bf['label']}** (main F1 {bf['edge_f1']:.3f}). Best text: **{bb['label']}** (main DVC "
                       f"BLEU-4 {bb['edge_bleu4']:.2f}). They are not the same experiment, so we must choose which one the paper leads with.")
        if p["ceiling"]:
            cb = p["ceiling"]["edge_bleu4"]
            out.append(f"- With the true cuts the same translator reaches main DVC BLEU-4 {cb:.2f}. Our best reaches {bb['edge_bleu4']:.2f}. "
                       f"Wrong boundaries cost about {100*(1 - bb['edge_bleu4']/cb):.0f} % of the text score.")
        pl = [r for r in rows if r["dec"] in ("argmax", "viterbi")]; du = [r for r in rows if r["dec"] == "duration"]
        if pl and du:
            dp = np.mean([r["edge_f1"] for r in du]) - np.mean([r["edge_f1"] for r in pl])
            worst = min(pl, key=lambda r: r["edge_f1"])
            wt = p["errors"][worst["label"]]["units_avg"]
            out.append(f"- Duration decoding minus argmax and viterbi decoding: {dp:+.3f} main F1 on average. "
                       f"The weakest argmax or viterbi setting is **{worst['label']}** (main F1 {worst['edge_f1']:.3f}); "
                       f"{100*wt['bridged']:.0f} % of its caption units sit in a caption that crosses a pause.")
        off = [r for r in rows if r["family"] == FAM_OFFLINE]; on = [r for r in rows if r["family"] == FAM_ONLINE]
        if off and on:
            bo = max(off, key=lambda r: r["edge_f1"]); bn = max(on, key=lambda r: r["edge_f1"])
            out.append(f"- Online against offline: main F1 {bn['edge_f1']:.3f} online against {bo['edge_f1']:.3f} offline.")
        em = [r for r in rows if r["seg"] == "moryossef"]; ou = [r for r in rows if r["seg"] == "s1"]
        if em and ou:
            de = np.mean([r["edge_f1"] for r in ou]) - np.mean([r["edge_f1"] for r in em])
            bou = max(ou, key=lambda r: r["edge_f1"]); bem = max(em, key=lambda r: r["edge_f1"])
            out.append(f"- Our segmenter against the EMNLP one, averaged over all settings: {de:+.3f} main F1. "
                       f"At its best setting S1 reaches {bou['edge_f1']:.3f} against {bem['edge_f1']:.3f}.")
        tx = p["errors"][bf["label"]]["units_avg"]; lost = ("cut", "bridged", "offset", "missed")
        main = max(lost, key=lambda k: tx[k])   # read off the table
        words = {"cut": "cut inside", "bridged": "in a caption that crosses a pause", "offset": "in a caption with a wrong edge",
                 "missed": "without a caption"}
        out.append(f"- The largest loss in **{bf['label']}** (the counted pairs of the main score, mean over the cells): {100*tx[main]:.0f} % of caption units are {words[main]} "
                   f"({', '.join(f'{k} {100*tx[k]:.0f} %' for k in lost)}).")
        jt = [r for r in rows if r["family"] == FAM_JOINT]
        if jt:
            bj = max(jt, key=lambda r: r["edge_bleu4"])
            cas = [r for r in rows if r["family"] in (FAM_OFFLINE, FAM_ONLINE)]
            if cas:
                bc = max(cas, key=lambda r: r["edge_bleu4"])
                out.append(f"- Best joint model against best cascade: main DVC BLEU-4 {bj['edge_bleu4']:.2f} against {bc['edge_bleu4']:.2f}.")
        out.append("")
    return "\n".join(out)


def _slug(s): return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower()
def _write_csv(path, rows):
    if not rows: return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)


def _bold_best(rows, cols, higher=True):
    """Bold the best value of each numeric column index in `cols` (ties all bold)."""
    for c in cols:
        vals = []
        for r in rows:
            try: vals.append(float(str(r[c]).replace("%", "").split()[0]))
            except (ValueError, IndexError): vals.append(None)
        good = [v for v in vals if v is not None]
        if len(good) < 2: continue
        best = max(good) if higher else min(good)
        for r, v in zip(rows, vals):
            if v is not None and v == best: r[c] = f"**{r[c]}**"
    return rows

def _metric_md(scores, labels, ths, pick, fmt):
    # 1 densevid score (`pick` of a score row) per threshold and its avg (metrics.evaluate_predicted_events `avg`: over every scored
    # threshold, its F1 the F1 of the mean precision and recall)
    rows = [[l, *[fmt.format(pick(scores[l][t])) for t in ths], fmt.format(pick(scores[l]["avg"]))] for l in labels]
    return _table(["system", *[f"tIoU {t:g}" for t in ths], "avg"], _bold_best(rows, range(1, len(ths) + 2)))

def _edge_md(edge, uncaptioned, labels):
    # The main score of each system (the F1 of the tIoU x edge-limit grid, avg) with its precision and recall, its F1 per edge limit τ
    # (over the thresholds) and per tIoU threshold (over τ), and the F1 of the tIoU test alone (no edge limit, avg; in no average).
    # The last column (lower is better, so not bolded) counts the units that no caption overlaps; they include the units that P1 at
    # inference gives no event (the rest of a span after the cap cut, signing with no decoded start, an open span at the stream end),
    # and the main score alone cannot tell such a unit from a unit under a wrong caption.
    first = edge[labels[0]]
    rows = [[l, *[f"{edge[l]['avg'][k]:.3f}" for k in ("f1", "precision", "recall")], *[f"{r['f1']:.3f}" for r in edge[l]["by_tau"]],
             *[f"{r['f1']:.3f}" for r in edge[l]["by_tiou"]], f"{edge[l]['no_edge_limit']['avg']['f1']:.3f}"] for l in labels]
    n = 4 + len(first["by_tau"]) + len(first["by_tiou"])
    rows = [r + [f"{uncaptioned[l]['share']:.1%} ({uncaptioned[l]['uncaptioned']})"] for r, l in zip(_bold_best(rows, range(1, n)), labels)]
    return _table(["system", "F1 avg (main)", "precision avg", "recall avg", *[f"F1, edge τ {r['tau_s']:g} s" for r in first["by_tau"]],
                   *[f"F1, tIoU {r['tiou']:g}" for r in first["by_tiou"]], "F1, tIoU test alone (no edge limit; reference)", "units with no caption"], rows)

def _grid_md(edge, labels, pick):
    """1 score of the main grid per system (`pick` formats 1 score row), in the layout of the screen table of a row: one line per edge
    limit τ, `avg` (each threshold over τ) and `no limit` (the tIoU test alone, for reference, in no average); one column per tIoU
    threshold and `avg` (each line over the thresholds). The main score is at avg, avg."""
    first = edge[labels[0]]
    rows = []
    for l in labels:
        grid = {(tau, t): r for tau, t, r in _grid_rows(edge[l])}
        rows += [[l, _name(tau), *[pick(grid[(tau, t)]) for t in [*first["tiou_thresholds"], "avg"]]] for tau in [*first["taus_s"], "avg", "no limit"]]
    return _table(["system", "edge τ (s)", *[f"tIoU {t:g}" for t in first["tiou_thresholds"]], "tIoU avg"], rows)

def _gap_md(edge, gaps, labels, gap_s):
    # The F1 and the DVC BLEU-4 of the grid (avg) at each chunk gap; the main gap is marked.
    gs = sorted({gap_s, *gaps[labels[0]]})
    pick = lambda l, g: (edge[l] if g == gap_s else gaps[l][g])["avg"]
    rows = [[l, *[f"{pick(l, g)['f1']:.3f}" for g in gs], *[f"{pick(l, g)['dvc']['bleu4']:.2f}" if pick(l, g)["dvc"] else "-" for g in gs]] for l in labels]
    name = lambda g: f"{g:g} s" + (" (main)" if g == gap_s else "")
    return _table(["system", *[f"F1 avg, gap {name(g)}" for g in gs], *[f"DVC BLEU-4 avg, gap {name(g)}" for g in gs]], rows)

def _chunk_class_md(chunk_cls, labels):
    # Per chunk class: the share of its units inside counted runs (mean over the cells of the grid) and its units with no caption.
    cls = [c for c in CHUNK_CLASSES if c in chunk_cls[labels[0]]]
    head = ["system", *[f"{c} (units: {chunk_cls[labels[0]][c]['units']})" for c in cls]]
    return _table(head, [[l, *[f"{chunk_cls[l][c]['matched_share']:.3f} matched, {chunk_cls[l][c]['uncaptioned']} with no caption" for c in cls]] for l in labels])

def _train_text_md(by_tt, video_sets, labels):
    # The main scores (avg of the grid) on the videos with no train text and on those with it.
    rows = [[l, f"{name} ({len(video_sets[name])} videos)", f"{e['avg']['f1']:.3f}", f"{e['avg']['dvc']['bleu4']:.2f}" if e["avg"]["dvc"] else "-",
             f"{e['avg']['soda']['bleu4']:.2f}" if e["avg"]["soda"] else "-", f"{u['share']:.1%} ({u['uncaptioned']})"]
            for l in labels for name, (e, u) in by_tt[l].items()]
    return _table(["system", "videos", "F1 avg (main)", "DVC BLEU-4 avg", "SODA-style BLEU-4 F1 avg", "units with no caption"], rows)

def _cap_cut_md(cap, edge, uncaptioned, labels):
    # Per cell of the grid (one column per edge limit τ, its tIoU thresholds in the cell): the cap-cut captions whose pair counts and,
    # in brackets, the units of their runs; then the main scores (avg of the grid) with and without the cap-cut captions.
    cells = [(r["tau_s"], r["tiou"]) for r in cap[labels[0]]["cells"]]
    taus, ths = list(dict.fromkeys(c[0] for c in cells)), list(dict.fromkeys(c[1] for c in cells))
    bleu = lambda e: f"{e['avg']['dvc']['bleu4']:.2f}" if e["avg"]["dvc"] else "-"
    rows = []
    for l in labels:
        at = {(r["tau_s"], r["tiou"]): r for r in cap[l]["cells"]}
        rows.append([l, cap[l]["captions"], f"{cap[l]['per_hour']:.2f}", cap[l]["units_held"],
                     *[" / ".join(f"{at[(tau, t)]['matched']} ({at[(tau, t)]['matched_units']})" for t in ths) for tau in taus],
                     f"{edge[l]['avg']['f1']:.3f} / {cap[l]['without']['avg']['f1']:.3f}", f"{bleu(edge[l])} / {bleu(cap[l]['without'])}",
                     f"{uncaptioned[l]['uncaptioned']} / {cap[l]['uncaptioned_without']['uncaptioned']}"])
    return _table(["system", "cap-cut captions", "per hour", "units they hold",
                   *[f"counted at τ {tau:g} s, tIoU {' / '.join(f'{t:g}' for t in ths)} (units)" for tau in taus],
                   "F1 avg (main): all / without them", "DVC BLEU-4 avg: all / without", "units with no caption: all / without"], rows)

def _shape_md(shape):
    return _table(["captions of", "count", "p10 (s)", "p50 (s)", "p90 (s)", "max (s)", "mean (s)", "holds 0 units", "1", "2", "3", "4+", "mean units held"],
                  [[l, s["captions"], *[f"{s['duration_s'][k]:.2f}" for k in ("p10", "p50", "p90", "max", "mean")],
                    *[f"{s['units_held'][k]:.2f}" for k in ("0", "1", "2", "3", "4+", "mean")]] for l, s in shape.items()])

def _cue_md(cues, labels):
    taus = [r["tau_s"] for r in cues[labels[0]]["taus"]]
    g = cues[labels[0]]
    return (_table(["system", *[f"τ {t:g} s: at a cue break (chance), inside edges" for t in taus]],
                   [[l, *[f"{r['at_cue_break']:.3f} ({r['chance']:.3f}), {r['inside_edges']}" for r in cues[l]["taus"]]] for l in labels]) +
            f"\nReliable units that hold a junction between 2 of their display cues: {g['units_with_break']} of {g['units']} (junctions: {g['breaks']}).\n")

def _edge_boundary_md(bounds, labels):
    # Per system and side: per boundary class, the share of the gold edges paired at each τ (of the gold edges counted), and the share
    # of the caption edges paired (of the caption edges counted).
    taus = list(dict.fromkeys(r["tau_s"] for r in bounds[labels[0]]))
    cls = [c for c in (*BOUNDARY_CLASSES, "all") if any(r["class"] == c for r in bounds[labels[0]])]
    def cell(l, side, c, key, n):
        rs = [r for r in bounds[l] if r["side"] == side and r["class"] == c]
        return " / ".join(f"{100 * r[key]:.1f}" for r in rs) + f" % of {rs[0][n]}" if rs else "-"
    rows = [[l, side, *[cell(l, side, c, "hit_rate", "gold_edges") for c in cls], cell(l, side, "all", "edge_precision", "caption_edges")]
            for l in labels for side in ("start", "end")]
    return _table(["system", "side", *[f"{c} at τ {' / '.join(f'{t:g}' for t in taus)} s" for c in cls], "caption edges paired"], rows)

def _edge_text_md(edge, labels, block, keys):
    # The DVC (`dvc`, TEXT_KEYS) or the SODA-style text F1 (`soda`, SODA_KEYS) on the counted pairs of the main grid, avg of the cells.
    labels = [l for l in labels if edge[l]["avg"][block]]
    if not labels: return "(no caption text)\n"
    rows = [[l, *[f"{edge[l]['avg'][block][k]:.{2 if k == 'bleu4' else 4}f}" for k in keys]] for l in labels]
    return _table(["system", *[TEXT_NAME[k] for k in keys]], _bold_best(rows, range(1, len(keys) + 1)))

def _reading_main(scores, edge, labels, ths):
    f1 = {l: [scores[l][t]["f1"] for t in ths] for l in labels}; bl = {l: [scores[l][t]["dvc"]["bleu4"] for t in ths] for l in labels}
    out = []
    def best(vals, fmt):
        top = max(vals.values()); names = [l for l, v in vals.items() if fmt.format(v) == fmt.format(top)]   # a tie at the printed precision is a tie
        return ("a tie between " if len(names) > 1 else "") + " and ".join(f"**{n}**" for n in names) + f" ({fmt.format(top)})"
    out.append(f"- Main scores (the tIoU x edge-limit grid, avg): best F1 {best({l: edge[l]['avg']['f1'] for l in labels}, 'F1 {:.3f}')}" +
               (f"; best DVC BLEU-4 {best({l: edge[l]['avg']['dvc']['bleu4'] for l in labels if edge[l]['avg']['dvc']}, 'BLEU-4 {:.2f}')}"
                if any(edge[l]["avg"]["dvc"] for l in labels) else "") + ".")
    for i, t in enumerate(ths):
        out.append(f"- densevid scores, at tIoU {t:g}: best sentence F1 {best({l: f1[l][i] for l in labels}, 'F1 {:.3f}')}; best DVC score {best({l: bl[l][i] for l in labels}, 'DVC BLEU-4 {:.2f}')}.")
    if len(ths) > 1:
        drops = [1 - f1[l][-1] / f1[l][0] for l in labels if f1[l][0] > 0]
        if drops: out.append(f"- From tIoU {ths[0]:g} to {ths[-1]:g} the systems lose {100*min(drops):.0f}-{100*max(drops):.0f} % of their F1. The stricter criterion accepts fewer of the same predicted intervals.")
    return "\n".join(out) + "\n"

# What each error type of the counted pairs of the main score means (metrics.edge_error_types: in each cell (τ, t) of the grid), for the reports
_UNIT_TYPE_WORDS = ("**matched** = the unit alone is a run whose pair counts in the cell. **joined** = the unit is inside a run of 2 or more "
                    "units of one chunk whose pair counts (a correct merge). **cut** = a caption edge lies inside the unit, more than τ from "
                    "its edges. **bridged** = a caption holds the unit (at least half of it) and a unit of another chunk (it crosses a pause). "
                    "**offset** = a caption holds the unit, but its pair does not count in the cell: an edge more than τ from the run edges, "
                    "a tIoU below t, or another caption took the match. **missed** = no caption holds the unit or cuts it.")
_CAPTION_TYPE_WORDS = ("**matched** and **joined** = the pair of the caption counts in the cell, with a run of 1 unit, of 2 or more units "
                       "of one chunk. **cut** = an edge lies inside a unit, more than τ from its edges. **bridged** = it holds units of 2 or "
                       "more chunks. **offset** = it holds units of one chunk, but its pair does not count in the cell. **phantom** = it holds "
                       "no unit and cuts none.")


def _error_types_md(errors, labels, n_units):
    # Per side, 1 line per system and edge limit τ (in each column the shares at the tIoU thresholds of the grid), then the mean over
    # the cells.
    cells = [(c["tau_s"], c["tiou"]) for c in errors[labels[0]]["cells"]]
    taus, ths = list(dict.fromkeys(c[0] for c in cells)), list(dict.fromkeys(c[1] for c in cells))
    out = []
    for side, types, words, what in (("units", EDGE_UNIT_TYPES, _UNIT_TYPE_WORDS, f"of the {n_units} caption units"),
                                     ("captions", EDGE_CAPTION_TYPES, _CAPTION_TYPE_WORDS, "of the captions of each system")):
        rows = []
        for l in labels:
            share = {(c["tau_s"], c["tiou"]): c[side] for c in errors[l]["cells"]}
            rows += [[l, f"{tau:g}", *[" / ".join(f"{100*share[(tau, t)][c]:.0f}" for t in ths) + " %" for c in types]] for tau in taus]
            rows.append([l, "mean of the cells", *[f"{100*errors[l][f'{side}_avg'][c]:.0f} %" for c in types]])
        out.append(f"**Share {what}**, in each cell of the grid (one line per edge limit τ; in each column, tIoU {' / '.join(f'{t:g}' for t in ths)}) "
                   "and the mean over the cells:\n\n" + _table(["system", "edge τ (s)", *types], rows) + "\n" + words + "\n")
    return "\n".join(out) + ("\nEach caption and each unit takes the first type whose rule fits, in the column order. All types come from "
                              "the counted pairs of the main score: in each cell, matched + joined is its precision (captions) and its recall (units). "
                              "`tables/error_types.csv`.\n")


def _table(header, rows):
    return "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)

def _pr_md(scores, labels, ths):
    # The precision and the recall of the densevid sentence tIoU F1 per threshold, and their avg (over every scored threshold)
    h = ["system", *[f"precision {t:g}" for t in ths], "avg", *[f"recall {t:g}" for t in ths], "avg"]
    rows = [[l, *[f"{scores[l][t][k]:.3f}" for k in ("precision", "recall") for t in [*ths, "avg"]]] for l in labels]
    return _table(h, _bold_best(rows, range(1, 2 * len(ths) + 3)))

def _located_md(scores, labels, ths, n_gold):
    h = ["system", *[f"located @{t:g}" for t in ths], *[f"BLEU-4 of those @{t:g}" for t in ths]]
    rows = [[l, *[str(scores[l][t]["scored_pairs"]) for t in ths], *[f"{scores[l][t]['located_corpus_bleu4']:.2f}" for t in ths]] for l in labels]
    return _table(h, _bold_best(rows, range(1, 2 * len(ths) + 1))) + f"\nof {n_gold} caption units.\n"

def _boundary_md(bstats, labels, taus):
    rows = [[l, f"{bstats[l]['median_head_s']:+.2f} s" if bstats[l]["median_head_s"] is not None else "-", f"{bstats[l]['median_tail_s']:+.2f} s" if bstats[l]["median_tail_s"] is not None else "-",
             *[f"{100 * (bstats[l][f'late_end@{t:g}'] or 0):.0f} %" for t in taus], bstats[l]["n"]] for l in labels]
    return (_table(["system", "median start error", "median end error", *[f"end more than {t:g} s late" for t in taus], "captions"], rows) +
            "\nThe captions that hold exactly one caption unit (at least half of it) that no other caption holds; caption minus unit, positive = late. "
            "An end more than τ late fails the edge limit τ by itself.\n")

def _cost_md(cost, ths):
    # The densevid scores per shown threshold and their avg (over every scored threshold), as reported and without the unlocated events
    h = ["system", "version", "sentences", "sentence tIoU F1 " + " / ".join(f"{t:g}" for t in ths), "avg",
         "tIoU DVC BLEU-4 " + " / ".join(f"{t:g}" for t in ths), "avg"]
    return _table(h, [[r["system"], r["version"], r["events"], " / ".join(f"{r[f'tiou_f1@{t:g}']:.3f}" for t in ths), f"{r['tiou_f1@avg']:.3f}",
                       " / ".join(f"{r[f'dvc_bleu4@{t:g}']:.2f}" for t in ths), f"{r['dvc_bleu4@avg']:.2f}"] for r in cost])

def _decision_md(edges, long_form, by_ref_class, reference):
    out = []
    out.append("**(1) Gold-edge recall within δ**\n\n" + _table(["system", "starts", "ends", "ends, pause after", "ends, abutting"],
               [[r["system"], f"{r['start_recall']:.3f} ({r['n_start']})", f"{r['end_recall']:.3f} ({r['n_end']})", f"{r['pause_recall']:.3f} ({r['n_pause']})",
                 f"{r['abutting_recall']:.3f} ({r['n_abutting']})"] for r in edges]))
    if by_ref_class:   # 1 line per system, span class and edge limit τ; 1 column per tIoU threshold
        ths, lines = list(dict.fromkeys(r["tiou"] for r in by_ref_class)), {}
        for r in by_ref_class: lines.setdefault((r["system"], r["span_class"], r["n_units"], r["tau_s"]), {})[r["tiou"]] = r["mean_delta"]
        out.append(f"**(2) Localized BLEU-4 against `{reference}`, by the class of `{reference}`'s span (mean delta per unit, in each cell of the grid)**\n\n" +
                   _table(["system", "span class", "units", "edge τ (s)", *[f"tIoU {t:g}" for t in ths]],
                          [[system, c, n, f"{tau:g}", *[at[t] for t in ths]] for (system, c, n, tau), at in lines.items()]))
    out.append("**(3) ReSCUE's commit-only axes**\n\n" + _table(["system", "long-form BLEU-4", "dropped events", "LAAL (s)", "AL (s)", "units covered"],
               [[r["system"], f"{r['long_form_bleu4']:.2f}", f"{r['dropped_share']:.1%}", r["laal_s"], r["al_s"], f"{r['covered_share']:.1%}"] for r in long_form]))
    return "\n".join(out)

def _held_md(rows):
    return _table(["system", "units", "covered", "median (s)", "p90 (s)", "mean (s)"],
                  [[r["system"], r["units"], f"{r['covered_share']:.1%}", r["median_s"], r["p90_s"], r["mean_s"]] for r in rows])

def _shared_md(shared, reference):
    if not shared: return ""
    return ("On the caption units that BOTH systems match alone in a cell of the grid (type matched, so both sides have the same references), corpus BLEU-4 of each system's captions:\n\n" +
            _table(["system", "edge τ (s)", "tIoU", "shared units", "this system's BLEU-4", f"{reference}'s BLEU-4 on the same units"],
                   [[r["system"], f"{r['tau_s']:g}", f"{r['tiou']:g}", r["n_shared"], f"{r['corpus_bleu4']:.2f}", f"{r['reference_corpus_bleu4']:.2f}"] for r in shared]))

def _sentence_tables_md(vid, tables, labels):
    """Two or three caption units of one video with every system's output: the one most systems locate (matched or joined in
    some cell of the grid), the one fewest locate, and the longest one nobody locates."""
    rows = {l: {r["start_s"]: r for r in tables[l].values() if r["video_id"] == vid} for l in labels}
    starts = sorted(rows[labels[0]])
    if not starts: return ""
    cells = list(rows[labels[0]][starts[0]]["cells"])   # the cells of the grid, tau-major
    taus, ths = list(dict.fromkeys(c[0] for c in cells)), list(dict.fromkeys(c[1] for c in cells))
    hits = lambda l, st: [c for c in cells if rows[l][st]["cells"][c]["type"] in LOCATED]   # the cells in which system l locates the unit
    n_loc = {st: sum(1 for l in labels if hits(l, st)) for st in starts}
    n_hit = {st: sum(len(hits(l, st)) for l in labels) for st in starts}
    picks = []
    best = max(starts, key=lambda st: (n_loc[st], n_hit[st]))
    if n_loc[best]: picks.append(("located by the most systems", best))
    few = [st for st in starts if 0 < n_loc[st] < len(labels)]
    if few:
        st = min(few, key=lambda st: (n_loc[st], -n_hit[st]))
        if st != best: picks.append((f"located by {n_loc[st]} of {len(labels)} systems", st))
    none = [st for st in starts if n_loc[st] == 0]
    if none: picks.append(("located by no system", max(none, key=lambda st: rows[labels[0]][st]["duration_s"])))
    out = []
    for why, st in picks:
        r0 = rows[labels[0]][st]
        out.append(f"\n*{why}* ({r0['start_s']:.1f}-{r0['end_s']:.1f} s): \"{r0['ref']}\"\n")
        out.append(_table(["system", *[f"type at τ {tau:g} s, tIoU {' / '.join(f'{t:g}' for t in ths)}" for tau in taus],
                           "output (in the first cell that locates it)"],
                          [[l, *[" / ".join(rows[l][st]["cells"][(tau, t)]["type"] for t in ths) for tau in taus],
                            f"\"{rows[l][st]['cells'][hits(l, st)[0]]['hyp']}\"" if hits(l, st) else "not located"] for l in labels]))
    return "\n".join(out)

def _diff_md(diff):
    if not diff: return "(one system: nothing to compare)\n"
    return _table(["system", "metric", "system value", "reference value", "difference", "fixed", "broken", "better", "worse"],
                  [[r["system"], r["metric"], r["system_value"], r["reference_value"], f"{r['difference']:+g}",
                    *(r.get(k, "-") for k in ("fixed", "broken", "better", "worse"))] for r in diff])

def _two_video_md(two):
    small, big = two["small"], two["big"]
    a, b = two["oracle"][small], two["oracle"][big]
    return (f"Video `{small}` has {a['n']} sentences and scores {a['video_bleu']:.2f}. "
            f"Video `{big}` has {b['n']} sentences and scores {b['video_bleu']:.2f}. "
            f"Pooling their actual n-gram counts gives **{two['oracle']['pooled_two']:.2f}**; "
            f"averaging the two video scores gives **{two['oracle']['mean_of_two']:.2f}**. "
            "The captions are identical in both calculations. Corpus BLEU is not a sentence-count-weighted average of video BLEU.\n")


def _protocol_md(o, proto, labels, ths, tau, ex, two, orow=None):
    lines = ["## The two ways to add up BLEU-4 (why a cascade can read above the pooled number)\n",
             "The pooled number and the DVC number are both called corpus BLEU-4, but they add up the same pairs in two different ways.\n",
             f"- **Pooled (one corpus):** all {o['n']} sentence pairs go into ONE count. One long video with many sentences weighs as much as its sentences.\n"
             f"- **DVC:** each video gets its own BLEU-4 from its own pairs. Then the {len(o['per_video'])} video scores are averaged with EQUAL weight. A 2-sentence video counts as much as a 600-sentence video.\n",
             f"With perfect cuts and the pretrimmed translator, the same captions give **{o['corpus_bleu4']:.2f}** one way and **{o['per_video_mean']:.2f}** the other way (median video {np.median(list(o['per_video'].values())):.2f}; mean weighted by video size {o['size_weighted_mean']:.2f}).\n",
             _two_video_md(two),
             (f"**Which number a DVC column should be compared with.** A system's DVC value is a mean over videos, so its "
              f"reference must be a mean over videos too. Perfect cuts scored by the code of the rows, with the same span filter every "
              f"row gets ({orow['n_kept']} of {orow['n_gold']} spans reach the scorer), give **DVC BLEU-4 {orow['dvc'][ths[0]]:.2f}** at every "
              f"threshold, sentence F1 {orow['f1'][ths[0]]:.3f}. That is the GT-boundary reference for the DVC column, and it is flat "
              f"because the predictions are the true spans, so no threshold can break a pair. The pooled {o['corpus_bleu4']:.2f} is not a DVC "
              f"number.\n\n"
              f"**A DVC value is a mean over {len(o['per_video'])} videos** (the reference: median video {np.median(list(o['per_video'].values())):.2f}, "
              f"size-weighted mean {o['size_weighted_mean']:.2f}). Short videos supply few 4-grams, so their BLEU-4 is near 0 most of the time and occasionally very "
              f"high, and the mean inherits that. Whether the mean lands above or below the corpus number is a property of the corpus, not of the "
              f"metric: it is above on this split and below on others. Do not describe the corpus number as a penalty on long videos.\n" if orow else ""),
             "![](fig/protocol_two_videos.png)\n", "![](fig/protocol_two_ways.png)\n", "![](fig/protocol_per_threshold.png)\n",
             _table(["system", "tIoU", "located pairs only, one corpus", "all DVC pairs, one corpus", "reported DVC (mean over videos)", "perfect cuts on the same sentences, one corpus"],
                    [[l, f"{t:g}", f"{proto['systems'][l][t]['located_corpus_bleu4']:.2f}", f"{proto['systems'][l][t]['all_pairs_corpus_bleu4']:.2f}", f"{proto['systems'][l][t]['dvc_mean']:.2f}", f"{proto['systems'][l][t].get('oracle_same_gold_corpus_bleu4', float('nan')):.2f}"] for l in labels for t in ths]),
             "\nHow to read: compare all-DVC-pairs corpus BLEU with reported DVC to isolate aggregation on fixed pairs. Compare located-only BLEU with the captions of the pretrimmed translator on exactly the same GT spans to control subset selection. Neither comparison makes DVC recall-sensitive. The all-caption video mean of the pretrimmed translator above is an aggregation check; apply the same proposal filter before presenting it as a row on true cuts.\n"]
    if ex: lines += ["![](fig/protocol_example.png)\n", f"The example video `{ex}` shows every pair under both scorers at tIoU {tau:g}, with the n-gram counts each pair adds.\n",
                     "![](fig/protocol_thresholds.png)\n",
                     "The same video at every threshold. An overlap becomes a pair only when its tIoU passes the threshold; below it the caption is scored against a random string, however much time the two share. This is why the DVC value falls as the threshold rises.\n"]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("results", help="tables, figures and the Markdown skeleton from the events files of the rows")
    r.add_argument("--events", action="append", required=True, metavar="LABEL=PATH")
    r.add_argument("--reference", default=None, help="system the differences compare against (default: the first --events)")
    r.add_argument("--language", required=True); r.add_argument("--split", default="test", choices=["train", "dev", "test"])
    r.add_argument("--boundary-sweep-file", default=None, help="a boundary_sweep_*.json of the pretrimmed translator (analyze.py --stage boundary-sweep): adds the confidence section")
    r.add_argument("--history", action="append", default=None, metavar="LABEL=PATH", help="a stage-2 history.csv: adds training curves")
    r.add_argument("--segmentation-scores", action="append", default=None, metavar="LABEL=PATH",
                   help="the _scores.json of a segmenter's no-text row (eval.py --offline|--online --text none): adds the segmentation-only section")
    r.add_argument("--stability-file", action="append", default=None)
    r.add_argument("--tiou-threshold", type=float, default=0.5, help="tIoU threshold of the case figures and the protocol section")
    r.add_argument("--case-videos", action="append", default=None, metavar="ID[:T0-T1[:NOTE]]", help="pin a video (and window, note) to the real-video section")
    r.add_argument("--report-thresholds", default=None, help="densevid thresholds shown in the tables and figures (default: every scored threshold); every scored "
                                                              "threshold stays in tables/*.csv, and each avg is that of every scored threshold (the F1 of the mean precision and recall)")
    r.add_argument("--data-config", default="configs/data.yaml"); r.add_argument("--inference-config", default="configs/inference.yaml"); r.add_argument("--eval-config", default="configs/eval.yaml")
    r.add_argument("--out", default=None)
    g = sub.add_parser("progress", help="one supervisor-facing report over every finished experiment in outputs/")
    g.add_argument("--language", action="append", required=True, help="repeatable; asf bfi ...")
    g.add_argument("--tiou-threshold", type=float, default=0.5, help="tIoU threshold of the example figures")
    g.add_argument("--case-videos", action="append", default=None, metavar="LANG=ID[:NOTE]", help="add a video to the real-video section")
    g.add_argument("--data-config", default="configs/data.yaml"); g.add_argument("--inference-config", default="configs/inference.yaml")
    g.add_argument("--eval-config", default="configs/eval.yaml"); g.add_argument("--out", default=None)

    d = sub.add_parser("data", help="corpus audit: what every preprocessing rule removed or repaired, per split")
    d.add_argument("--language", action="append", required=True, help="repeatable; asf bfi ase")
    d.add_argument("--data-config", default="configs/data.yaml"); d.add_argument("--out", default=None, help="output dir (default outputs/report/data)")

    for name in ("poses", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--data-config", default="configs/data.yaml"); p.add_argument("--inference-config", default="configs/inference.yaml")
        p.add_argument("--language", default=None); p.add_argument("--split", default="dev", choices=["train", "dev", "test"])
        p.add_argument("--video-id", default=None); p.add_argument("--out", default=None, help=f"output dir (default outputs/report/{name})")
        p.add_argument("--device", default=None)
        if name == "poses":
            p.add_argument("--fps", type=float, default=12.5); p.add_argument("--frames", type=int, default=40)
            p.add_argument("--max-render-frames", type=int, default=900); p.add_argument("--synthetic", action="store_true")
        else:
            p.add_argument("--text", default="dlm", choices=["pretrimmed", "ar", "dlm"], help="the translator, as eval.py --text")
            p.add_argument("--pretrimmed-checkpoint", default=None); p.add_argument("--arm-checkpoint", default=None)
            p.add_argument("--max-sentences", type=int, default=12)
    return ap


def main():
    args = build_parser().parse_args()
    if args.cmd == "results": results(args); return
    if args.cmd == "progress": progress(args); return
    if args.cmd == "data":
        from reporting import data_audit
        data_audit.run(args); return
    from reporting import debug
    if args.language is None: args.language = str(load_yaml(args.data_config)["active_languages"][0])
    print(json.dumps((debug.cmd_poses if args.cmd == "poses" else debug.cmd_predict)(args), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
