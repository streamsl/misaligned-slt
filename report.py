"""One report per language and split: tables, figures and a Markdown skeleton, from the events files of the rows.

    python report.py results --language asf --split test \\
        --events "true cuts + clean"=outputs/offline/offline_gold_none_clean_asf_test.json \\
        --events "Cascade: external segmenter"=outputs/offline/offline_moryossef_duration_clean_asf_test.json \\
        --events "joint DLM, online"=outputs/online/online_joint-dlm_duration_dlm_asf_test.json \\
        --reference "joint DLM, online" \\
        --history AR=checkpoints/ar/asf/history.csv --segmenter-eval s1=outputs/segmenter/segmenter_eval_s1_duration_asf_test.json

Writes outputs/report/results_<language>_<split>/results.md (the generated skeleton; a hand-written report.md beside it is never touched), fig/*.png and tables/*.csv. Every system is scored under one
protocol on one gold index, so all comparisons are paired. Any events file of a row is a system: a cascade row,
the offline row, the online row, an ablation. Row 1 (the true cuts read by the clean translator, `eval.py --segments` on the
gold span file) among the systems adds the metric-protocol section. The optional inputs add sections: `--boundary-sweep-file`
(a sweep of the clean translator, analyze.py --stage boundary-sweep) adds the confidence panel; `--history` adds training
curves; `--segmenter-eval` adds the segmentation-only comparison; `--stability-file` adds the reveal-policy panel.

    python report.py progress --language asf --language bfi            # one report over every finished experiment (outputs/report/progress)

    python report.py data    --language asf                            # the corpus audit (outputs/report/data)
    python report.py poses   --language asf --video-id <id>            # the pose stream one video feeds the network (outputs/report/poses)
    python report.py predict --language asf --method dlm --video-id <id>   # one model's whole-video outputs (outputs/report/predict)
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

from reporting import figures as F
from reporting.outcomes import (LOCATED, OUTCOME_COLUMNS, boundary_stats, case_studies, edge_recall, flip_table, unit_outcomes,
                                unit_span_classes)
from reporting.protocol import (case_view, decompose, located_pairs, oracle_pairs, oracle_row, showcase_videos, two_video_example,
                                worked_example_video)
from data.loader import corpus_config, load_language_records
from evaluation.predictions import (CHUNK_CLASSES, corpus_token, gold_events, chunk_classes, cue_breaks, gold_chunks, load_event_predictions,
                                    scoreable_predictions, train_text_videos, unk_stretches, warn_annotation_mismatch)
from evaluation.systems import warn_model_labels
from models.checkpointing import load_checkpoint_meta
from metrics import (EDGE_CAPTION_TYPES, EDGE_UNIT_TYPES, SODA_KEYS, TEXT_KEYS, cap_cut_captions, caption_shape, text_level,
                     cue_edge_share, edge_chunk_classes, edge_chunk_scores, edge_error_types, evaluate_predicted_events, held_unit_latency, long_form_scores, match_segments, sentence_bleu_scores,
                     text_scores, uncaptioned_units)
from utils import cfg_get, lambda_min_s, load_yaml, output_dir, seconds_to_frames, target_language

TEXT_NAME = {"bleu4": "BLEU-4", "bleurt": "BLEURT", "rougeL": "ROUGE-L", "meteor": "METEOR", "cider": "CIDEr-D"}   # metrics.TEXT_KEYS in tables


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


def _segmentation_only(path, records) -> dict:
    """A segmenter-eval file for the segmentation-only table: the edge chunk F1 avg (`edge_chunk`, the main score), the sentence tIoU F1
    and the span-count ratio per SOURCE video (`tiou`), so the table agrees with the segmentation numbers of the rows. Signing/non-signing F1 is a
    frame metric with no counterpart in the rows; it comes from the Moryossef-protocol `metrics` block (mean over streams)."""
    warn_annotation_mismatch(path, records, "segmenter-eval file")   # a file made under other cuts or units is flagged, then scored
    raw = json.load(open(path, encoding="utf-8"))
    rows = {t: v for t, v in raw["tiou"].items() if t != "avg"}
    return {"bio_f1": raw["metrics"]["bio_f1"], "edge_chunk_f1_avg": raw["edge_chunk"]["f1_avg"], "phrase_segment_count_ratio": next(iter(rows.values()))["count_ratio"],
            **{f"phrase_tiou_f1@{t}": v["f1"] for t, v in rows.items()}}


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
    thresholds = [float(t) for t in scoring_cfg["tiou_thresholds"]]
    edge_taus = [float(t) for t in scoring_cfg["edge_taus_s"]]   # the main localization score, the edge chunk score
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
        sc = {float(r["tiou_threshold"]): r for r in evaluate_predicted_events(pred, gold, thresholds, char_level=char_level, densevid=True)["thresholds"]}
        errors[label] = edge_error_types(pred, chunks, edge_taus)   # the one error table: the error types of the edge chunk match
        for t in thresholds:   # a tIoU score: the corpus BLEU-4 of the one-to-one tIoU pairs
            tiou_pairs = located_pairs(pred, gold, t)
            sc[t]["located_corpus_bleu4"] = text_scores([h for _, _, h, _ in tiou_pairs], [r for _, _, _, r in tiou_pairs], char_level,
                                                        keys=("bleu4",))["bleu4"] if tiou_pairs else 0.0
        # the per-unit outcomes: each unit's error type at each edge τ, with its located caption and that caption's score
        unit_rows = unit_outcomes(chunks, pred, errors[label], char_level)
        for r in unit_rows:
            toks = [t for t in (_vocab_token(w) for w in r["ref"].split()) if t]
            r["oov_rate"] = round(sum(1 for w in toks if w not in train_vocab) / max(1, len(toks)), 3)
        tables[label] = {r["gold_id"]: r for r in unit_rows}
        bstats[label] = boundary_stats(errors[label])
        exemplars[label] = case_studies(unit_rows, pred, errors[label])
        count_ratio[label] = sum(len(v) for v in pred.values()) / max(1, n_gold)
        with (tab_dir / f"outcomes_{_slug(label)}.csv").open("w", encoding="utf-8", newline="") as f:   # 1 line per unit and τ
            w = csv.DictWriter(f, fieldnames=OUTCOME_COLUMNS, extrasaction="ignore"); w.writeheader()
            w.writerows({**r, "tau_s": t, **at} for r in unit_rows for t, at in r["taus"].items())
        scores[label], edge[label], uncaptioned[label] = sc, edge_chunk_scores(pred, chunks, edge_taus, char_level), uncaptioned_units(pred, gold)
        print(f"[report] {label}: {sum(len(v) for v in pred.values())} events, edge chunk F1 avg {edge[label]['f1_avg']:.3f}, "
              f"sentence F1@{tau:g} {sc[tau]['segmentation']['f1']:.3f}, units with no caption {uncaptioned[label]['share']:.1%}", flush=True)
    labels = list(systems)
    gold_ids = list(tables[reference])
    for label in labels:
        if list(tables[label]) != gold_ids: raise SystemExit(f"{label}: gold index differs from {reference!r}; all systems need the same split and GT settings")

    # ---- tables
    def _row(label, t): s = scores[label][t]; return {"system": label, "tiou": t, **{f"{k}": round(v, 4) for k, v in s["segmentation"].items()},
                                                      **{k: round(v, 4) for k, v in s["text_metrics"].items()}, "located_corpus_bleu4": round(s["located_corpus_bleu4"], 2),
                                                      "located": s["scored_pairs"], "events": s["total_predictions"], "gold": s["total_gold"]}
    rows = [_row(l, t) for l in labels for t in thresholds]
    _write_csv(tab_dir / "scores.csv", rows)
    # The main scores, per tau and averaged: the edge chunk F1 and, on its pairs, the DVC and the SODA-style text F1.
    flat = lambda r: {k: v for k, v in r.items() if k not in ("dvc", "soda")} | {
        f"{block}_{k}": v for block in ("dvc", "soda") for k, v in (r.get(block) or {}).items()}
    _write_csv(tab_dir / "edge_chunk.csv", [{"system": l, **{k: round(v, 4) if isinstance(v, float) else v for k, v in flat(r).items()}}
                                           for l in labels for r in edge[l]["taus"] + [{"tau_s": "avg", "precision": edge[l]["precision_avg"],
                                           "recall": edge[l]["recall_avg"], "f1": edge[l]["f1_avg"], "dvc": edge[l]["dvc_avg"], "soda": edge[l]["soda_avg"]}]])
    _write_csv(tab_dir / "uncaptioned.csv", [{"system": l, **{k: round(v, 4) if isinstance(v, float) else v for k, v in uncaptioned[l].items()}} for l in labels])
    # 1 row per system, side and tau (and the mean): the share of each type (per video, then the mean over videos, as the main score) and
    # its pooled count; a type of the other side is blank
    types = list(dict.fromkeys(EDGE_CAPTION_TYPES + EDGE_UNIT_TYPES))
    _write_csv(tab_dir / "error_types.csv", [{"system": l, "side": side, "tau_s": tau_s, **{c: round(shares[c], 4) if c in shares else "" for c in types},
                                              **{f"n_{c}": counts.get(c, "") for c in types}, "captions_per_unit": round(count_ratio[l], 3)}
                                             for l in labels for side in ("units", "captions") for tau_s, shares, counts in
                                             [(row["tau_s"], row[side], row[f"{side[:-1]}_counts"]) for row in errors[l]["taus"]] + [("avg", errors[l][f"{side}_avg"], {})]])
    _write_csv(tab_dir / "boundary.csv", [{"system": l, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in bstats[l].items()}} for l in labels])
    # ---- report-only readouts (no `_scores.json` key): metrics.py computes each one, on inputs that evaluation.predictions builds
    rnd = lambda row, n=4: {k: round(v, n) if isinstance(v, float) else v for k, v in row.items()}
    delta_s, gap_s = float(inference_cfg["boundary_stability"]["delta_s"]), float(scoring_cfg["chunk_gap_s"])
    # (1) the main localization at other chunk gaps, with BLEU-4 text only (no learned scorer); the main row stays at chunk_gap_s
    gap_chunks = {g: gold_chunks(records, g) for g in map(float, scoring_cfg["chunk_gap_rows_s"])}
    gaps = {l: {g: edge_chunk_scores(systems[l], c, edge_taus, char_level, bleurt_checkpoint=None, keys=("bleu4",)) for g, c in gap_chunks.items()}
            for l in labels}
    _write_csv(tab_dir / "edge_chunk_gaps.csv", [
        {"system": l, "chunk_gap_s": g, "main": g == gap_s, "chunks": e["chunks"], **rnd({k: e[k] for k in ("f1_avg", "precision_avg", "recall_avg")}),
         "dvc_bleu4_avg": round(e["dvc_avg"]["bleu4"], 2) if e["dvc_avg"] else None}
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
    chunk_cls = {l: edge_chunk_classes(systems[l], chunks, classes, edge_taus) for l in labels}
    _write_csv(tab_dir / "chunk_classes.csv", [{"system": l, "class": c, **rnd(chunk_cls[l][c])} for l in labels for c in CHUNK_CLASSES if c in chunk_cls[l]])
    # (5) every main score on the videos with no train text and on those with it (the set is computed from the records at each run)
    dedup = corpus_config(data_cfg, args.language)["subtitles"]["dedup"]   # the copy check's floor for 1 cue and its boilerplate cap
    tt = train_text_videos(records, train_records, int(dedup["min_words"]), int(dedup["df_cap"]))
    _write_csv(tab_dir / "train_text_videos.csv", [{"video_id": v, "reliable_units": len(gold[v]), "units_with_train_text": tt.get(v, 0)} for v in sorted(gold)])
    video_sets = {"no train text": [v for v in gold if not tt.get(v)], "train text": [v for v in gold if tt.get(v)]}
    keep = lambda d, vids: {v: d[v] for v in vids if v in d}
    by_tt = {l: {name: (edge_chunk_scores(keep(systems[l], vids), keep(chunks, vids), edge_taus, char_level), uncaptioned_units(keep(systems[l], vids), keep(gold, vids)))
                 for name, vids in video_sets.items() if vids} for l in labels}
    flat_main = lambda e: {k: e[k] for k in ("f1_avg", "precision_avg", "recall_avg")} | {
        f"{block}_{k}": v for block in ("dvc_avg", "soda_avg") for k, v in (e[block] or {}).items()}
    _write_csv(tab_dir / "edge_chunk_by_train_text.csv", [{"system": l, "videos": name, "n_videos": len(video_sets[name]), **rnd(flat_main(e)),
                                                           "uncaptioned": u["uncaptioned"], "units": u["units"]} for l in labels for name, (e, u) in by_tt[l].items()])
    # (6) the cap-cut captions apart: count, matches per τ, their units, and the main scores without them (all captions' when there are none)
    hours = sum(float(r.pose.duration_s) for r in records) / 3600.   # the pose time of the split's streams (its clean segments)
    cap = {l: cap_cut_captions(systems[l], chunks, gold, edge_taus, char_level, hours) for l in labels}
    for l, c in cap.items():
        if c["without"] is None: c["without"], c["uncaptioned_without"] = edge[l], uncaptioned[l]
    _write_csv(tab_dir / "cap_cut.csv", [{"system": l, "captions": c["captions"], "per_hour": round(c["per_hour"], 3), "units_held": c["units_held"],
                                          **{f"matched@{r['tau_s']:g}": r["matched"] for r in c["taus"]}, **{f"matched_units@{r['tau_s']:g}": r["matched_units"] for r in c["taus"]},
                                          "f1_avg": round(edge[l]["f1_avg"], 4), **{f"without_{k}": v for k, v in rnd(flat_main(c["without"])).items()},
                                          "uncaptioned": uncaptioned[l]["uncaptioned"], "uncaptioned_without": c["uncaptioned_without"]["uncaptioned"]} for l, c in cap.items()])
    # differences against the reference: each headline number of a system minus the reference's
    avg_of = lambda label, key, part: float(np.mean([scores[label][t][part][key] for t in rth]))
    lat_of = {label: held_unit_latency(gold, systems[label]) for label in labels}
    diff = []
    loc = lambda l, t: float(np.mean([tables[l][g]["taus"][t]["localized_bleu4"] for g in gold_ids])) if gold_ids else 0.0
    for label in labels:
        if label == reference: continue
        clean = lambda l: by_tt[l]["no train text"][0] if "no train text" in by_tt[l] else None   # the main scores on the videos with no train text
        cs, cr = clean(label), clean(reference)
        for metric, a, b, extra in (
            ("edge chunk F1 avg (main)", edge[label]["f1_avg"], edge[reference]["f1_avg"], {}),
            *[(f"edge chunk DVC {TEXT_NAME[k]} avg (main)", edge[label]["dvc_avg"][k], edge[reference]["dvc_avg"][k], {})
              for k in TEXT_KEYS if edge[label]["dvc_avg"] and edge[reference]["dvc_avg"]],
            *[(f"SODA-style {TEXT_NAME[k]} F1 avg (main)", edge[label]["soda_avg"][k], edge[reference]["soda_avg"][k], {})
              for k in SODA_KEYS if edge[label]["soda_avg"] and edge[reference]["soda_avg"]],
            ("sentence tIoU F1 avg", avg_of(label, "f1", "segmentation"), avg_of(reference, "f1", "segmentation"), {}),
            ("tIoU DVC BLEU-4 avg", avg_of(label, "densevid_bleu4", "text_metrics"), avg_of(reference, "densevid_bleu4", "text_metrics"), {}),
            *[(f"localized BLEU-4 at τ {t:g} s (mean per unit)", loc(label, t), loc(reference, t), flip_table(tables[label], tables[reference], t))
              for t in edge_taus],
            *[(f"edge chunk F1 avg, chunk gap {g:g} s", gaps[label][g]["f1_avg"], gaps[reference][g]["f1_avg"], {}) for g in gap_chunks],
            *([("edge chunk F1 avg (main), videos with no train text", cs["f1_avg"], cr["f1_avg"], {})] if cs else []),
            *[(f"edge chunk DVC {TEXT_NAME[k]} avg (main), videos with no train text", cs["dvc_avg"][k], cr["dvc_avg"][k], {})
              for k in TEXT_KEYS if cs and cs["dvc_avg"] and cr["dvc_avg"]],
            *[(f"SODA-style {TEXT_NAME[k]} F1 avg (main), videos with no train text", cs["soda_avg"][k], cr["soda_avg"][k], {})
              for k in SODA_KEYS if cs and cs["soda_avg"] and cr["soda_avg"]],
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
        sc2 = {float(r["tiou_threshold"]): r for r in evaluate_predicted_events(kept, gold, thresholds, char_level=char_level, densevid=True)["thresholds"]}
        for name, sc_ in (("as reported", scores[label]), ("unlocated events removed", sc2)):
            cost.append({"system": label, "version": name, "events": sum(len(v) for v in (systems[label] if name == "as reported" else kept).values()),
                         **{f"f1@{t:g}": round(sc_[t]["segmentation"]["f1"], 3) for t in thresholds}, "f1_avg": round(float(np.mean([sc_[t]["segmentation"]["f1"] for t in thresholds])), 3),
                         **{f"dvc_bleu4@{t:g}": round(sc_[t]["text_metrics"]["densevid_bleu4"], 2) for t in thresholds}, "dvc_avg": round(float(np.mean([sc_[t]["text_metrics"]["densevid_bleu4"] for t in thresholds])), 2)})
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
            for t in edge_taus:
                by_ref_class.append({"system": label, "reference": reference, "tau_s": t, "span_class": c, "n_units": len(ids), "mean_delta": round(float(np.mean(
                    [tables[label][g]["taus"][t]["localized_bleu4"] - tables[reference][g]["taus"][t]["localized_bleu4"] for g in ids])), 3)})
    for name, rows_ in (("edge_recall", edge_rows), ("long_form", long_form), ("bleu_by_reference_span_class", by_ref_class)):
        if rows_: _write_csv(tab_dir / f"{name}.csv", rows_)
    shared = []
    for label in labels:
        if label == reference: continue
        for t in edge_taus:   # the units that both systems match alone at τ, so both corpora have the same references
            both = [g for g in gold_ids if tables[label][g]["taus"][t]["type"] == "matched" and tables[reference][g]["taus"][t]["type"] == "matched"]
            corpus = lambda l: round(text_scores([tables[l][g]["taus"][t]["hyp"] for g in both], [tables[l][g]["ref"] for g in both], char_level, keys=("bleu4",))["bleu4"], 2) if both else 0.0
            shared.append({"system": label, "tau_s": t, "n_shared": len(both), "corpus_bleu4": corpus(label), "reference_corpus_bleu4": corpus(reference)})
    if shared: _write_csv(tab_dir / "shared_sentences.csv", shared)
    with (tab_dir / "cases.json").open("w", encoding="utf-8") as f: json.dump(exemplars, f, indent=1, ensure_ascii=False)

    # ---- figures
    F.experiment_map(fig_dir / "experiment_map.png"); F.metric_explained(fig_dir / "metric_explained.png")
    F.scores_at_thresholds(scores, labels, rth, fig_dir / "scores.png", n_gold)
    F.precision_recall(scores, labels, rth, fig_dir / "precision_recall.png")
    F.located_vs_quality(scores, labels, rth, fig_dir / "located_vs_quality.png", n_gold)
    F.error_types(errors, labels, count_ratio, fig_dir / "error_types.png", n_gold)
    F.boundary_error(bstats, labels, fig_dir / "boundary_error.png", edge_taus)
    edges = [0.0, 2.0, 4.0, 6.0, 9.0, 14.0, cap_s, float("inf")]; lab = ["0-2 s", "2-4 s", "4-6 s", "6-9 s", "9-14 s", "14 s-cap", "> cap"]
    F.by_length(tables, labels, edges, lab, fig_dir / "by_length.png", cap_s, edge_taus)
    F.by_oov(tables, labels, fig_dir / "by_oov.png")
    F.bleu_by_edge_error(tables, labels, fig_dir / "bleu_by_edge_error.png", edge_taus)

    # ---- the BLEU protocol (needs the clean translator on perfect cuts: row 1 among the systems, the first one that reads the
    # gold spans with the clean translator)
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
    loose = f"{max(edge_taus):g}"   # a unit still lost at the largest τ is a clear error
    for kind in ("bridged", "cut", "missed"):
        c = next(iter(exemplars[reference]["taus"][loose][kind]), None)
        if c and c["video_id"] not in {x["video_id"] for x in shows}:
            shows.append({"video_id": c["video_id"], "window": (max(0.0, c["start_s"] - 15), c["end_s"] + 15),
                          "note": f"longest {kind} caption unit of {reference} at τ {loose} s"})
    case_md = []
    for i, c in enumerate(shows):
        c_tau = float(c.get("tau", tau))
        view = case_view(gold, systems, c["video_id"], c_tau, char_level, window=c["window"], oracle=orc, raw=raw_events)
        name = f"case_{i+1}_{c['video_id']}.png"
        F.case_figure(view, c_tau, fig_dir / name, note=c["note"], language=args.language)
        case_md.append(f"**{c['video_id']}**: {c['note']}\n\n![](fig/{name})\n" + _sentence_tables_md(c["video_id"], tables, labels, edge_taus))

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
    if args.segmenter_eval:
        ev = {l: _segmentation_only(p, records) for l, p in _labelled(args.segmenter_eval).items()}
        F.frame_vs_sentence(ev, rth, fig_dir / "segmentation_only.png")
        ths_ = rth
        md_opt.append("## Segmentation only, before any translation\n\n![](fig/segmentation_only.png)\n\n" +
                      _table(["segmenter", "edge chunk F1 avg (main)", "signing/non-signing F1", *[f"sentence F1, tIoU {t:g}" for t in ths_], "sentence F1 average", "sentences produced / true count"],
                             [[l, f"{m['edge_chunk_f1_avg']:.3f}", f"{m['bio_f1']:.3f}", *[f"{m.get(f'phrase_tiou_f1@{t:g}', 0):.3f}" for t in ths_], f"{np.mean([m.get(f'phrase_tiou_f1@{t:g}', 0) for t in ths_]):.3f}", f"{m['phrase_segment_count_ratio']:.2f}"] for l, m in ev.items()]) +
                      "\nHow to read: binary signing coverage can be high even when B boundaries are missing. Sentence localization checks both boundaries; binary signing F1 does not.\n")
    if args.stability_file:
        pol = {Path(p).stem.replace("stability_", ""): json.load(open(p, encoding="utf-8")) for p in args.stability_file}
        F.stability(pol, fig_dir / "stability.png")
        md_opt.append("## Reveal policies\n\n![](fig/stability.png)\n\nHow to read: further left is earlier text on screen; higher is more text that later had to change.\n")

    # ---- markdown, in the reading order of the weekly report
    corpus_name = cfg_get(data_cfg, "languages", args.language, "name", default=args.language)   # its data.yaml name
    intro = [f"**{args.language}: {corpus_name}. {args.split} set: {len(gold)} videos, {n_gold} caption units.**",
             f"The main localization score is the **edge chunk F1**, averaged over edge tolerances τ = {', '.join(f'{t:g}' for t in edge_taus)} s (section 2; `tables/edge_chunk.csv`). "
             f"The error types use the same edge match, at each τ. "
             f"Every other score is reported at {len(rth)} levels of strictness, **tIoU {', '.join(f'{t:g}' for t in rth)}**, and their average. "
             f"Every system is scored with the same code on the same caption units; the per-unit tables (each unit's error type at each τ, with the caption "
             f"that the edge match pairs with it) are in `tables/outcomes_<system>.csv`, "
             f"and every scored threshold ({', '.join(f'{t:g}' for t in thresholds)}) is in `tables/scores.csv`."
             + (" Every text is scored in lowercase, for every system alike (data.yaml lowercase_references)." if char_level.lowercase else "") + "\n"]
    sec = []
    sec.append("## 1. The experiments\n\n![experiment map](fig/experiment_map.png)\n\n" + _table(["name in this report", "events file"], [[l, f"`{Path(p).name}`"] for l, p in systems_paths.items()]))
    sec.append("## 2. How the scores work\n\n![metric explained](fig/metric_explained.png)\n\n"
               f"- **Edge chunk F1 (the main localization score).** A *chunk* is a group of caption units of one stream with gaps of at most {float(scoring_cfg['chunk_gap_s']):g} s between them (no pause). "
               "A prediction may hold any run of whole, consecutive units inside one chunk, so joining units that are signed without a pause is not an error. "
               "A prediction matches such a run only when its start and its end are each within τ of the run's start and end; the F1 is averaged over the τ values. "
               "Each prediction and each unit is used once. Precision counts predictions; recall counts units. A prediction that crosses a pause, or that cuts a unit, matches nothing.\n"
               "- **tIoU** = shared time / total time covered. A prediction locates a caption unit when tIoU passes the threshold, and one prediction can take only one sentence.\n"
               f"- **Edge chunk DVC (main text scores).** The DVC of densevid_eval on the edge chunk pairs: per video, each prediction that the edge chunk F1 matches is scored against the joined text of its units, and a prediction with no match against a random string; then the plain mean over videos, averaged over τ: {', '.join(TEXT_NAME[k] for k in TEXT_KEYS)} (BLEU-4 is the corpus BLEU-4 of the video's pairs, with no smoothing). A missed unit adds no DVC pair.\n"
               f"- **SODA-style text F1 on the same pairs (main).** Per video, precision = summed pair scores / predictions and recall = summed pair scores, each weighted by the units of its pair, / units; the F1 of the two, then the mean over videos, averaged over τ: {', '.join(TEXT_NAME[k] for k in SODA_KEYS)} (per pair: sentence BLEU-4; CIDEr-D has no per-pair score). It charges false predictions and missed units.\n"
               "- **Sentence tIoU F1** counts located sentences, one prediction per sentence by tIoU (precision: of what we produced; recall: of what is true).\n"
               "- **tIoU DVC BLEU-4** is the DVC of densevid_eval on its own tIoU pairs: every prediction against every caption unit it overlaps by tIoU, per video, then the plain mean over videos. A prediction with no qualifying overlap is scored against a random string. A missed caption unit adds no DVC pair.\n"
               "- **BLEU-4 of the located pairs** is corpus BLEU-4 over the one-to-one located pairs only: text quality without the localization penalty.\n"
               "- Corpus BLEU-4 pools n-gram counts. DVC averages per-video corpus BLEU-4; localized diagnostics instead use sentence BLEU. Their labels and denominators must remain distinct. The text backend is SacreBLEU 13a, a documented difference from the original DVC COCO/PTB scorer.\n")
    seg_md = next((m for m in md_opt if m.startswith("## Segmentation only")), None)
    if seg_md: sec.append(seg_md.replace("## Segmentation only", "## 3. Segmentation only", 1))
    n0 = 4 if seg_md else 3
    n_tt = len(video_sets["train text"])
    sec.append(f"## {n0}. Main results\n\n![main results](fig/scores.png)\n\n**Localization, main score (edge chunk F1).**\n\n" + _edge_md(edge, uncaptioned, labels) +
               "\n**Precision (P, of the captions) and recall (R, of the units) of the edge chunk F1 at each τ.**\n\n" + _edge_pr_md(edge, labels) +
               "\n**Edge chunk DVC (main text scores; mean over τ).**\n\n" + _edge_text_md(edge, labels, "dvc_avg", TEXT_KEYS) +
               "\n**SODA-style text F1 on the edge chunk pairs (main; mean over τ).**\n\n" + _edge_text_md(edge, labels, "soda_avg", SODA_KEYS) +
               "\n**tIoU scores: sentence localization (sentence tIoU F1).**\n\n" + _metric_md(scores, labels, rth, lambda s_, t: s_[t]["segmentation"]["f1"], "{:.3f}") +
               "\n**tIoU scores: dense captioning (tIoU DVC BLEU-4; mean over videos).**\n\n" + _metric_md(scores, labels, rth, lambda s_, t: s_[t]["text_metrics"]["densevid_bleu4"], "{:.2f}") +
               "\n**Reading.**\n\n" + _reading_main(scores, edge, labels, rth) +
               "\n**Units by chunk class (report only).** Each reliable unit takes the class of its chunk, by the first rule that fits: `over_cap` (the chunk is longer than "
               f"the cap minus δ, {cap_s - delta_s:g} s, the longest span that the engine commits whole; online, a longer span can commit only as a cap-cut closed part, which "
               "ends at the lowest P(I) frame, not at a unit edge; the offline rows keep every span whole), `video_start` (it starts at frame 0 of the video), `cut_start` (it starts at a stream start made by a cut, or after a "
               "quarantined unit that crosses the stream start), `after_quarantine`, `end_blocked` (it ends at the stream end or before a quarantined unit) and `free` "
               f"(a pause at both edges); \"at\" means within the chunk gap, {gap_s:g} s. `matched` is the share of the units inside matched runs (mean over τ); the "
               "units with no caption of the classes sum to the column above. `tables/chunk_classes.csv`.\n\n" + _chunk_class_md(chunk_cls, labels) +
               "\n**The main localization at other chunk gaps (report only; DVC with BLEU-4 only).** `tables/edge_chunk_gaps.csv`.\n\n" + _gap_md(edge, gaps, labels, gap_s) +
               f"\n**The main scores on the videos with no train text (report only).** A video has train text when one of its reliable units of {min_words} "
               f"or more words has the words of a reliable train unit; {n_tt} of {len(gold)} videos do (`tables/train_text_videos.csv`; scores in `tables/edge_chunk_by_train_text.csv`). "
               "A comparison on these sets is valid only between models trained on the same train labels.\n\n" + _train_text_md(by_tt, video_sets, labels) +
               "\n**The cap-cut captions apart (report only).** A cap-cut caption is the closed part of an open span that the cap cut at the frame with the lowest P(I) "
               f"(the declared exception to P1 at inference). `per hour`: per hour of the split's pose time ({hours:.1f} h of clean segments); `units they hold`: at least half of each; `matched`: cap-cut captions that the edge chunk match pairs, "
               "with the units of their runs. `tables/cap_cut.csv`.\n\n" + _cap_cut_md(cap, edge, uncaptioned, labels))
    sec.append(f"## {n0+1}. Precision against recall\n\n![precision recall](fig/precision_recall.png)\n\n" + _pr_md(scores, labels, rth))
    sec.append(f"## {n0+2}. What changes when the match gets stricter\n\n![threshold quality](fig/located_vs_quality.png)\n\n" + _located_md(scores, labels, rth, n_gold) +
               "\nThe clips and captions are fixed across thresholds. Raising tIoU changes which predictions receive real references and which become unmatched. Located-only BLEU may rise because it scores a smaller subset; this is not an improvement to any caption. An unmatched prediction receives a random reference, while a missed GT sentence adds no DVC pair. Report localization recall beside DVC.\n")
    sec.append(f"## {n0+3}. Error types\n\n![error types](fig/error_types.png)\n\n" + _error_types_md(errors, labels, n_gold, edge_taus))
    sec.append(f"## {n0+4}. Real videos\n\nChosen by rule, never by hand: the tiny video with the top per-video score, the longest merged span that fits one figure, the prediction that pairs with the most caption units, the longest prediction with a looping caption, a one-sentence video with an over-long prediction, then the longest bridged, cut and missed caption units of the reference system at τ = {loose} s (the error types of `tables/outcomes_<system>.csv`). The figures pair by tIoU at {tau:g}. `--case-videos id[:t0-t1[:note]]` pins more; every exemplar is in `tables/cases.json`.\n\n" + "\n".join(case_md))
    n = n0 + 5
    hist_md = next((m for m in md_opt if m.startswith("## Training curves")), None)
    if hist_md: sec.append(hist_md.replace("## Training curves", f"## {n}. Training curves", 1)); n += 1
    sec.append(f"## {n}. Where the boundaries land\n\n![boundary error](fig/boundary_error.png)\n\n" + _boundary_md(bstats, labels, edge_taus) +
               "\n**Caption shape (report only).** Durations of the scored captions, and the reliable units each caption holds (at least half of each); "
               "the gold units give the reference durations. `tables/caption_shape.csv`.\n\n" + _shape_md(shape) +
               "\n**Display-cue share (report only).** Per τ: of the caption edges that lie inside a unit and more than τ from both of its edges (they can match "
               "nothing at τ), the share within τ of a junction between 2 display cues of that unit, and the share that chance gives (the time near a junction over "
               "the inner time of the unit). A share above chance means the system cuts inside units where the captioner broke the display. `tables/cue_breaks.csv`.\n\n" +
               _cue_md(cues, labels)); n += 1
    sec.append(f"## {n}. Breakdown by sentence length and by unseen words\n\n![by length](fig/by_length.png)\n\n![by oov](fig/by_oov.png)\n\n![bleu by edge error](fig/bleu_by_edge_error.png)\n\n"
               "These figures use a per-unit score, `localized BLEU-4`: at each τ, the sentence BLEU-4 of the caption that the edge chunk match pairs with the unit's run, against the joined text of the run "
               "(a joined unit takes the score of its run), and 0 for a unit that is not located (not matched or joined). It gives every caption unit its own number, which the breakdowns need. "
               "The length and unseen-word figures average over τ; the last figure reads the located units at the largest τ, by the larger edge error of their caption. It is not the DVC column.\n"); n += 1
    sec.append(f"## {n}. What the extra sentences cost\n\n" + _cost_md(cost, rth) + "\n" + _shared_md(shared, reference) +
               "\nHow to read: the second row of each system removes every event that locates nothing at the loosest threshold and scores again. The change is what those events cost.\n"); n += 1
    sec.append(f"## {n}. Decision metrics (fixed before any trained arm)\n\n" + _decision_md(edge_rows, long_form, by_ref_class, reference) +
               "\nHow to read: (1) Edge recall: the share of gold starts and ends with a predicted edge within δ; ends split by the sampler's rule into `pause after` (the next unit starts more than δ later) and `abutting`. (2) Translation by the reference's span class: localized BLEU-4 against the reference at each τ, per class of the reference's span that holds the unit (mean difference per unit). (3) ReSCUE's axes: long-form BLEU-4 (per video, events in commit order against the gold in time order, one corpus; events at least half inside an uncaptioned or quarantined stretch dropped first), and LAAL/AL per unit from the commit time of the event that holds it (committed text never changes, so NE = 0).\n"); n += 1
    if held:
        sec.append(f"## {n}. Latency of every caption unit\n\n" + _held_md(held) +
                   "\nHow to read: per caption unit, the commit time of the first event that holds at least half of the unit, minus the unit's end. The headline latency uses only one-to-one matched pairs, so a unit inside a merged event drops out of it; here it counts, with the merged event's commit time. `covered` is the share of units that some event holds.\n"); n += 1
    sec.append(f"## {n}. Differences against `{reference}`\n\n" + _diff_md(diff) +
               "\nHow to read: each row is a headline number of the system, the same number of the reference, and their difference. `localized BLEU-4` is the mean per caption unit at each τ (a unit that is not located scores 0), so it charges missed units, which DVC does not. `fixed` and `broken` count the units that one system locates at τ (matched or joined) and the other does not.\n"); n += 1
    for m in md_opt:
        if m.startswith(("## Segmentation only", "## Training curves")): continue
        sec.append(m.replace("## ", f"## {n}. ", 1)); n += 1
    md = intro + [x + "\n---\n" for x in sec]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")   # results.md: the generated skeleton; report.md stays yours
    print(f"[report] wrote {out}/results.md, {len(list(fig_dir.glob('*.png')))} figures, {len(list(tab_dir.iterdir()))} tables", flush=True)
    return {"out": str(out), "systems": labels, "thresholds": thresholds}


# ---------------------------------------------------------------- progress report for a supervisor
SEG_NAME = {"gold": "true cuts", "moryossef": "EMNLP", "s1": "S1", "joint-ar": "joint AR", "joint-dlm": "joint DLM"}
TRANS_NAME = {"clean": "clean", "clean-dlm": "clean DLM", "ar": "joint AR", "dlm": "joint DLM"}
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
            for dec in ("none", "plain", "duration"):
                for trans in ("clean", "clean-dlm", "ar", "dlm"):
                    p = output_dir(when) / f"{when}_{spans}_{dec}_{trans}_{lang}_test.json"
                    if not p.exists(): continue
                    who = SEG_NAME[spans] + ("" if dec == "none" else f" {dec}")
                    label = f"{who}, {when}" if spans == f"joint-{trans}" else f"{who} + {TRANS_NAME[trans]}, {when}"
                    out.append({"family": _family(when, spans, trans), "seg": spans, "dec": dec, "when": when,
                                "trans": trans, "label": label, "path": str(p)})
    # `_html` spans one cell over each family's rows, so they must arrive contiguous and in reading order.
    out.sort(key=lambda r: (FAM_ORDER.index(r["family"]), r["when"] == "online", r["seg"], r["trans"]))
    return out


def _html(rows, thresholds):
    """One HTML table, experiments merged by family. Bold is the best value in the column WITHIN its own family:
    offline and online are different settings, so their best values are not compared with each other. The true cuts are
    the rows of the family FAM_ORACLE (row 1: the clean translator; row 2: the arms)."""
    th = "".join(f"<th>{t:g}</th>" for t in thresholds) + "<th>avg</th>"
    h = ["<table>", "<thead>",
         f'<tr><th rowspan="2">Experiment</th><th rowspan="2">Boundaries from</th><th rowspan="2">Decoding</th>'
         f'<th rowspan="2">When it must decide</th><th rowspan="2">Translator</th>'
         f'<th colspan="{len(thresholds)+1}">Localization F1 &nbsp;(tIoU)</th>'
         f'<th colspan="{len(thresholds)+1}">Translation BLEU-4 &nbsp;(tIoU)</th></tr>',
         f"<tr>{th}{th}</tr>", "</thead>", "<tbody>"]
    body = rows
    val = lambda r, key, t: r[key][t] if t != "avg" else float(np.mean([r[key][x] for x in thresholds]))
    best = {}
    for fam in {r["family"] for r in rows}:
        fam_rows = [r for r in rows if r["family"] == fam]
        for key in ("f1", "bleu"):
            for t in list(thresholds) + ["avg"]: best[(fam, key, t)] = max(val(r, key, t) for r in fam_rows)
    fam_n = {}
    for r in body: fam_n[r["family"]] = fam_n.get(r["family"], 0) + 1
    seen = set()
    for r in body:
        cells = []
        if r["family"] not in seen:
            seen.add(r["family"]); cells.append(f'<td rowspan="{fam_n[r["family"]]}"><b>{r["family"]}</b></td>')
        cells += [f'<td>{SEG_NAME.get(r["seg"], r["seg"])}</td>', f'<td>{r["dec"]}</td>', f'<td>{r["when"]}</td>',
                  f'<td>{TRANS_NAME.get(r["trans"], r["trans"])}</td>']
        for key, fmt in (("f1", "{:.3f}"), ("bleu", "{:.2f}")):
            for t in list(thresholds) + ["avg"]:
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
        scoring_cfg = load_yaml(args.eval_config)["scoring"]   # the main scores: the edge chunk match
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
            sc = {float(r["tiou_threshold"]): r for r in evaluate_predicted_events(pred, gold, thresholds, char_level=char_level, densevid=True)["thresholds"]}
            errors[spec["label"]] = edge_error_types(pred, chunks, edge_taus)   # the one error table (the edge chunk match)
            for t in thresholds:   # a tIoU score: the corpus BLEU-4 of the one-to-one tIoU pairs
                tiou_pairs = located_pairs(pred, gold, t)
                sc[t]["located_corpus_bleu4"] = text_scores([h for _, _, h, _ in tiou_pairs], [r for _, _, _, r in tiou_pairs], char_level,
                                                            keys=("bleu4",))["bleu4"] if tiou_pairs else 0.0
            unit_rows = unit_outcomes(chunks, pred, errors[spec["label"]], char_level)   # each unit's error type per edge τ
            for r in unit_rows:
                toks = [t for t in (_vocab_token(w) for w in r["ref"].split()) if t]
                r["oov_rate"] = round(sum(1 for w in toks if w not in train_vocab) / max(1, len(toks)), 3)
            tables[spec["label"]] = {r["gold_id"]: r for r in unit_rows}
            bstats[spec["label"]] = boundary_stats(errors[spec["label"]])
            ratio[spec["label"]] = sum(len(v) for v in pred.values()) / max(1, n_gold)
            scores[spec["label"]] = sc
            edge = edge_chunk_scores(pred, chunks, edge_taus, char_level, bleurt_checkpoint=None, keys=("bleu4",))   # only BLEU-4 is printed
            rows.append({**spec, "f1": {t: sc[t]["segmentation"]["f1"] for t in thresholds},
                         "bleu": {t: sc[t]["text_metrics"]["densevid_bleu4"] for t in thresholds},
                         "edge_f1": edge["f1_avg"], "edge_bleu4": edge["dvc_avg"]["bleu4"] if edge["dvc_avg"] else 0.0})
            print(f"[progress] {lang} {spec['label']}: MAIN edge chunk F1 avg {edge['f1_avg']:.3f}, edge chunk DVC BLEU-4 "
                  f"{rows[-1]['edge_bleu4']:.2f} | tIoU: sentence F1@{tau:g} {sc[tau]['segmentation']['f1']:.3f}, "
                  f"DVC BLEU-4 {sc[tau]['text_metrics']['densevid_bleu4']:.2f}", flush=True)
        labels = [r["label"] for r in rows]
        # The deployed system: the joint arm online, translating its own spans. Selected by FIELDS, because a label
        # spelling falls through silently to another row (the AR twin, or a control) as soon as it changes.
        reference = next((r["label"] for r in rows if r["family"] == FAM_JOINT and r["when"] == "online"
                          and r["trans"] == "dlm"), None) or labels[-1]

        # the reference on the true cuts and the protocol section: row 1 (the clean translator on the gold spans), when it exists
        ceiling, proto, orc, sweep_rows = None, None, None, []
        row1 = next((r for r in rows if r["seg"] == "gold" and r["trans"] == "clean" and r["when"] == "offline"), None)
        # A stale row 1 (other annotations, warned above) cannot pair with the current gold: the protocol section is left out.
        if row1 and row1["fresh"]:
            orc = oracle_pairs(gold, raw_events[row1["label"]])
            proto = decompose(gold, systems, thresholds, char_level, oracle=orc)
            if proto["oracle"]:
                orow = oracle_row(gold, orc, records, thresholds, char_level)
                ceiling = {"f1_by_t": {t: orow["f1"][t] for t in thresholds}, "bleu_by_t": {t: orow["dvc"][t] for t in thresholds},
                           "f1": orow["f1"][thresholds[0]], "bleu": orow["dvc"][thresholds[0]], "kept": orow["n_kept"], "n_gold": orow["n_gold"],
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
        F.by_length(tables, labels, edges, blab, fig_dir / f"{lang}_by_length.png", cap_s, edge_taus)
        F.by_oov(tables, labels, fig_dir / f"{lang}_by_oov.png")
        seval = {}
        for arch in ("moryossef", "s1"):
            for dec in ("plain", "duration"):
                p = output_dir("segmenter") / f"segmenter_eval_{arch}_{dec}_{lang}_test.json"   # --output in the runbook carries the decode token
                if p.exists(): seval[f"{SEG_NAME[arch]} {dec}"] = _segmentation_only(p, records)
        if seval: F.frame_vs_sentence(seval, thresholds, fig_dir / f"{lang}_segmentation_only.png")
        sweep = output_dir("analysis") / f"boundary_sweep_baseline_{lang}.json"   # the clean floor's boundary sweep, when it was run
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
        shown = [max(off or rows, key=lambda r: _avg(r, "f1", thresholds))["label"]]
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

        _write_csv(tab_dir / f"{lang}_scores.csv", [{"system": r["label"], "family": r["family"], "tiou": t,
                                                     "f1": round(r["f1"][t], 4), "bleu4": round(r["bleu"][t], 3)} for r in rows for t in thresholds])
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
                       "perfect boundaries is a reference condition: I give the true cuts to the clean translator."]})

    S.append({"title": "How the scores work",
              "md": ["![](fig/metric_explained.png)\n",
                     "Localization F1 counts the caption units we find. BLEU-4 is the text score inside them. A prediction "
                     "that finds nothing is scored against a random string, so extra sentences pull the score down.\n"],
              "talk": ["Next the metric, because everything later depends on it.", "",
                       "At the top are the caption units from the subtitles. Very often there is no gap between one sentence "
                       "and the next.", "",
                       "In the middle is what a model produces: time intervals with text inside. This is the same output as "
                       "dense video captioning, so I use the dense video captioning score, like in our previous paper.", "",
                       "At the bottom is the scoring. I pair each prediction with at most one caption unit, by time overlap, "
                       "or tIoU. Because one prediction can take only one caption unit, a prediction that covers 2 true "
                       "sentences will lose one of them. The translation score is BLEU-4 inside the sentences we find. A "
                       "prediction that finds nothing is scored against a random string, so extra sentences make the score lower."]})

    if any(per_lang[l]["seval"] for l in langs):
        md, talk = [], []
        for l in langs:
            ev = per_lang[l]["seval"]
            if not ev: continue
            md += [f"### {L(l)}\n", f"![](fig/{l}_segmentation_only.png)\n",
                   _table(["segmenter", "signing/non-signing F1", *[f"tIoU {t:g}" for t in ths], "average", "sentences produced / true count"],
                          [[k, f"{v['bio_f1']:.3f}", *[f"{v.get(f'phrase_tiou_f1@{t:g}', 0):.3f}" for t in ths],
                            f"{np.mean([v.get(f'phrase_tiou_f1@{t:g}', 0) for t in ths]):.3f}", f"{v['phrase_segment_count_ratio']:.2f}"] for k, v in ev.items()]), ""]
            best = max(ev, key=lambda k: np.mean([ev[k].get(f"phrase_tiou_f1@{x:g}", 0) for x in ths]))
            talk += [f"For {L(l)}: the segmenters reach a signing/non-signing F1 of about {100*np.mean([v['bio_f1'] for v in ev.values()]):.0f} %. "
                     f"At tIoU "
                     f"{ths[0]:g} they find only about {100*max(ev[k].get(f'phrase_tiou_f1@{ths[0]:g}', 0) for k in ev):.0f} % "
                     f"of the sentences. The best setting is {best}.", ""]
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
            md.append("\n*Oracle spans* gives the true cuts to each translator under the same scoring protocol (row 1: the clean "
                      "translator). It is a reference condition, not a mathematical upper bound on caption quality. Its localization is "
                      "1.000 by construction: scoring charges every system for every event it emits and deletes none, so the true cuts "
                      "score perfectly.\n")
            md += _attainable_md(c, L(l))
        md.append(f"\n![](fig/{l}_results.png)\n")
        bf = max(_systems(rows), key=lambda r: _avg(r, "f1", ths)); bb = max(_systems(rows), key=lambda r: _avg(r, "bleu", ths))
        line = (f"For {L(l)}: the best localization is **{bf['label']}** with F1 {_avg(bf,'f1',ths):.3f}. The best text is "
                f"**{bb['label']}** with BLEU-4 {_avg(bb,'bleu',ths):.2f}.")
        if bf["label"] != bb["label"]:
            line += " These are not the same experiment, and I would like your opinion on which one the paper should lead with."
        talk += [line, ""]
        if p["ceiling"]:
            cb = float(np.mean([p["ceiling"]["bleu_by_t"][x] for x in ths]))
            talk += [f"With perfect cuts the same translator reaches BLEU-4 {cb:.2f}. So the wrong boundaries cost us about "
                     f"{100*(1 - _avg(bb,'bleu',ths)/cb):.0f} % of the text score. That gap is the whole problem of this project.", ""]
        off = [r for r in rows if r["family"] == FAM_OFFLINE]; on = [r for r in rows if r["family"] == FAM_ONLINE]
        if off and on:
            bo = max(off, key=lambda r: _avg(r, "f1", ths)); bn = max(on, key=lambda r: _avg(r, "f1", ths))
            talk += [f"Online against offline: F1 {_avg(bn,'f1',ths):.3f} online against {_avg(bo,'f1',ths):.3f} offline.", ""]
        pl = [r for r in rows if r["dec"] == "plain"]; du = [r for r in rows if r["dec"] == "duration"]
        if pl and du:
            dp = np.mean([_avg(r, "f1", ths) for r in du]) - np.mean([_avg(r, "f1", ths) for r in pl])
            talk += [f"Duration minus plain decoding: {dp:+.3f} F1 on average.", ""]
        jt = [r for r in rows if r["family"] == FAM_JOINT]
        if jt:
            bj = max(jt, key=lambda r: _avg(r, "bleu", ths))
            cas = [r for r in rows if r["family"] in (FAM_OFFLINE, FAM_ONLINE)]
            if cas:
                bc = max(cas, key=lambda r: _avg(r, "bleu", ths))
                talk += [f"Best joint model against best cascade: BLEU-4 {_avg(bj,'bleu',ths):.2f} against {_avg(bc,'bleu',ths):.2f}.", ""]
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
        p = per_lang[l]; bf = max(_systems(p["rows"]), key=lambda r: _avg(r, "f1", ths)); tx = p["errors"][bf["label"]]["units_avg"]
        talk += [f"For {L(l)}: in the best system, {100*(tx['matched'] + tx['joined']):.0f} % of the caption units are found "
                 f"({100*tx['joined']:.0f} % of them inside a correct joined caption), {100*tx['cut']:.0f} % are cut inside, "
                 f"{100*tx['bridged']:.0f} % sit in a caption that crosses a pause, {100*tx['offset']:.0f} % sit in a caption with a "
                 f"wrong edge, and {100*tx['missed']:.0f} % have no caption.", ""]
    talk += ["These types come from the same edge match as the main score. They do not establish whether a boundary cue is visible in the motion."]
    S.append({"title": "Where the caption units are lost",
              "md": [_UNIT_TYPE_WORDS + "\n"] + [f"![](fig/{l}_failures.png)\n" for l in langs],
              "talk": talk})

    S.append({"title": "Where the boundaries land",
              "md": ["Only the captions that hold exactly one caption unit, which no other caption holds, are counted. Positive means "
                     "late. An end more than τ late fails the edge match at τ by itself.\n"] + [f"![](fig/{l}_boundary.png)\n" for l in langs],
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
                     f"mean over videos. Same captions, 2 different answers. So when we compare our systems to the clean "
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


def _avg(r, key, ths): return float(np.mean([r[key][t] for t in ths]))
def _systems(rows): return [r for r in rows if r["family"] != FAM_ORACLE] or rows   # the best-system picks skip the true cuts (rows 1, 2)


def _findings(per_lang, ths, tau):
    """Short statements, each one read off the table, with the experiment named."""
    out = []
    for l, p in per_lang.items():
        rows = p["rows"]
        bf = max(_systems(rows), key=lambda r: _avg(r, "f1", ths)); bb = max(_systems(rows), key=lambda r: _avg(r, "bleu", ths))
        out.append(f"**{LANG_NAME.get(l, l)}.**")
        if bf["label"] == bb["label"]:
            out.append(f"- Best overall: **{bf['label']}**, F1 {_avg(bf,'f1',ths):.3f} and BLEU-4 {_avg(bb,'bleu',ths):.2f}.")
        else:
            out.append(f"- Best boundaries: **{bf['label']}** (F1 {_avg(bf,'f1',ths):.3f}). Best text: **{bb['label']}** (BLEU-4 {_avg(bb,'bleu',ths):.2f}). "
                       f"They are not the same experiment, so we must choose which one the paper leads with.")
        if p["ceiling"]:
            cb = float(np.mean([p["ceiling"]["bleu_by_t"][t] for t in ths]))
            out.append(f"- With the true cuts the same translator reaches BLEU-4 {cb:.2f}. Our best reaches {_avg(bb,'bleu',ths):.2f}. "
                       f"Wrong boundaries cost about {100*(1 - _avg(bb,'bleu',ths)/cb):.0f} % of the text score.")
        pl = [r for r in rows if r["dec"] == "plain"]; du = [r for r in rows if r["dec"] == "duration"]
        if pl and du:
            dp = np.mean([_avg(r, "f1", ths) for r in du]) - np.mean([_avg(r, "f1", ths) for r in pl])
            worst = min(pl, key=lambda r: _avg(r, "f1", ths))
            wt = p["errors"][worst["label"]]["units_avg"]
            out.append(f"- Duration minus plain decoding: {dp:+.3f} F1 on average. "
                       f"The weakest plain-decoding setting is **{worst['label']}** (F1 {_avg(worst,'f1',ths):.3f}); "
                       f"{100*wt['bridged']:.0f} % of its caption units sit in a caption that crosses a pause.")
        off = [r for r in rows if r["family"] == FAM_OFFLINE]; on = [r for r in rows if r["family"] == FAM_ONLINE]
        if off and on:
            bo = max(off, key=lambda r: _avg(r, "f1", ths)); bn = max(on, key=lambda r: _avg(r, "f1", ths))
            out.append(f"- Online against offline: F1 {_avg(bn,'f1',ths):.3f} online against {_avg(bo,'f1',ths):.3f} offline.")
        em = [r for r in rows if r["seg"] == "moryossef"]; ou = [r for r in rows if r["seg"] == "s1"]
        if em and ou:
            de = np.mean([_avg(r, "f1", ths) for r in ou]) - np.mean([_avg(r, "f1", ths) for r in em])
            bou = max(ou, key=lambda r: _avg(r, "f1", ths)); bem = max(em, key=lambda r: _avg(r, "f1", ths))
            out.append(f"- Our segmenter against the EMNLP one, averaged over all settings: {de:+.3f} F1. "
                       f"At its best setting S1 reaches {_avg(bou,'f1',ths):.3f} against {_avg(bem,'f1',ths):.3f}.")
        tx = p["errors"][bf["label"]]["units_avg"]; lost = ("cut", "bridged", "offset", "missed")
        main = max(lost, key=lambda k: tx[k])   # read off the table
        words = {"cut": "cut inside", "bridged": "in a caption that crosses a pause", "offset": "in a caption with a wrong edge",
                 "missed": "without a caption"}
        out.append(f"- The largest loss in **{bf['label']}** (edge chunk match, mean over τ): {100*tx[main]:.0f} % of caption units are {words[main]} "
                   f"({', '.join(f'{k} {100*tx[k]:.0f} %' for k in lost)}).")
        jt = [r for r in rows if r["family"] == FAM_JOINT]
        if jt:
            bj = max(jt, key=lambda r: _avg(r, "bleu", ths))
            cas = [r for r in rows if r["family"] in (FAM_OFFLINE, FAM_ONLINE)]
            if cas:
                bc = max(cas, key=lambda r: _avg(r, "bleu", ths))
                out.append(f"- Best joint model against best cascade: BLEU-4 {_avg(bj,'bleu',ths):.2f} against {_avg(bc,'bleu',ths):.2f}.")
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
    rows = [[l, *[fmt.format(pick(scores[l], t)) for t in ths], fmt.format(np.mean([pick(scores[l], t) for t in ths]))] for l in labels]
    return _table(["system", *[f"tIoU {t:g}" for t in ths], "average"], _bold_best(rows, range(1, len(ths) + 2)))

def _edge_md(edge, uncaptioned, labels):
    # The last column (lower is better, so not bolded) counts the units that no caption overlaps; they include the units that P1 at
    # inference gives no event (the rest of a span after the cap cut, signing with no decoded start, an open span at the stream end),
    # and the edge score alone cannot tell such a unit from a unit under a wrong caption.
    taus = [r["tau_s"] for r in edge[labels[0]]["taus"]]
    rows = [[l, *[f"{r['f1']:.3f}" for r in edge[l]["taus"]], f"{edge[l]['f1_avg']:.3f}", f"{edge[l]['precision_avg']:.3f}", f"{edge[l]['recall_avg']:.3f}"]
            for l in labels]
    rows = [r + [f"{uncaptioned[l]['share']:.1%} ({uncaptioned[l]['uncaptioned']})"] for r, l in zip(_bold_best(rows, range(1, len(taus) + 4)), labels)]
    return _table(["system", *[f"τ {t:g} s" for t in taus], "average", "precision avg", "recall avg", "units with no caption"], rows)

def _edge_pr_md(edge, labels):
    # Precision and recall of the edge chunk F1 at each τ (both are means over videos, as the F1 reads them).
    taus = [r["tau_s"] for r in edge[labels[0]]["taus"]]
    rows = [[l, *[f"{r[k]:.3f}" for r in edge[l]["taus"] for k in ("precision", "recall")]] for l in labels]
    return _table(["system", *[f"τ {t:g} s {name}" for t in taus for name in ("P", "R")]], _bold_best(rows, range(1, 2 * len(taus) + 1)))

def _gap_md(edge, gaps, labels, gap_s):
    # The edge chunk F1 avg and the edge chunk DVC BLEU-4 avg at each chunk gap; the main gap is marked.
    gs = sorted({gap_s, *gaps[labels[0]]})
    pick = lambda l, g: edge[l] if g == gap_s else gaps[l][g]
    rows = [[l, *[f"{pick(l, g)['f1_avg']:.3f}" for g in gs], *[f"{pick(l, g)['dvc_avg']['bleu4']:.2f}" if pick(l, g)["dvc_avg"] else "-" for g in gs]] for l in labels]
    name = lambda g: f"{g:g} s" + (" (main)" if g == gap_s else "")
    return _table(["system", *[f"F1 avg, gap {name(g)}" for g in gs], *[f"DVC BLEU-4, gap {name(g)}" for g in gs]], rows)

def _chunk_class_md(chunk_cls, labels):
    # Per chunk class: the share of its units inside matched runs (mean over τ) and its units with no caption.
    cls = [c for c in CHUNK_CLASSES if c in chunk_cls[labels[0]]]
    head = ["system", *[f"{c} (units: {chunk_cls[labels[0]][c]['units']})" for c in cls]]
    return _table(head, [[l, *[f"{chunk_cls[l][c]['matched_share']:.3f} matched, {chunk_cls[l][c]['uncaptioned']} with no caption" for c in cls]] for l in labels])

def _train_text_md(by_tt, video_sets, labels):
    rows = [[l, f"{name} ({len(video_sets[name])} videos)", f"{e['f1_avg']:.3f}", f"{e['dvc_avg']['bleu4']:.2f}" if e["dvc_avg"] else "-",
             f"{e['soda_avg']['bleu4']:.2f}" if e["soda_avg"] else "-", f"{u['share']:.1%} ({u['uncaptioned']})"] for l in labels for name, (e, u) in by_tt[l].items()]
    return _table(["system", "videos", "edge chunk F1 avg", "edge chunk DVC BLEU-4", "SODA-style BLEU-4 F1", "units with no caption"], rows)

def _cap_cut_md(cap, edge, uncaptioned, labels):
    taus = [r["tau_s"] for r in cap[labels[0]]["taus"]]
    bleu = lambda e: f"{e['dvc_avg']['bleu4']:.2f}" if e["dvc_avg"] else "-"
    rows = [[l, cap[l]["captions"], f"{cap[l]['per_hour']:.2f}", cap[l]["units_held"], *[f"{r['matched']} ({r['matched_units']} units)" for r in cap[l]["taus"]],
             f"{edge[l]['f1_avg']:.3f} / {cap[l]['without']['f1_avg']:.3f}", f"{bleu(edge[l])} / {bleu(cap[l]['without'])}",
             f"{uncaptioned[l]['uncaptioned']} / {cap[l]['uncaptioned_without']['uncaptioned']}"] for l in labels]
    return _table(["system", "cap-cut captions", "per hour", "units they hold", *[f"matched at τ {t:g} s" for t in taus], "edge chunk F1 avg: all / without them",
                   "edge chunk DVC BLEU-4: all / without", "units with no caption: all / without"], rows)

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

def _edge_text_md(edge, labels, block, keys):
    # The edge chunk DVC (`dvc_avg`, TEXT_KEYS) or the SODA-style text F1 on the same pairs (`soda_avg`, SODA_KEYS), averaged over the taus.
    labels = [l for l in labels if edge[l][block]]
    if not labels: return "(no caption text)\n"
    rows = [[l, *[f"{edge[l][block][k]:.{2 if k == 'bleu4' else 4}f}" for k in keys]] for l in labels]
    return _table(["system", *[TEXT_NAME[k] for k in keys]], _bold_best(rows, range(1, len(keys) + 1)))

def _reading_main(scores, edge, labels, ths):
    f1 = {l: [scores[l][t]["segmentation"]["f1"] for t in ths] for l in labels}; bl = {l: [scores[l][t]["text_metrics"]["densevid_bleu4"] for t in ths] for l in labels}
    out = []
    def best(vals, fmt):
        top = max(vals.values()); names = [l for l, v in vals.items() if fmt.format(v) == fmt.format(top)]   # a tie at the printed precision is a tie
        return ("a tie between " if len(names) > 1 else "") + " and ".join(f"**{n}**" for n in names) + f" ({fmt.format(top)})"
    out.append(f"- Main scores: best edge chunk F1 {best({l: edge[l]['f1_avg'] for l in labels}, 'F1 avg {:.3f}')}" +
               (f"; best edge chunk DVC BLEU-4 {best({l: edge[l]['dvc_avg']['bleu4'] for l in labels if edge[l]['dvc_avg']}, 'BLEU-4 {:.2f}')}"
                if any(edge[l]["dvc_avg"] for l in labels) else "") + ".")
    for i, t in enumerate(ths):
        out.append(f"- tIoU scores, at tIoU {t:g}: best sentence F1 {best({l: f1[l][i] for l in labels}, 'F1 {:.3f}')}; best DVC score {best({l: bl[l][i] for l in labels}, 'DVC BLEU-4 {:.2f}')}.")
    if len(ths) > 1:
        drops = [1 - f1[l][-1] / f1[l][0] for l in labels if f1[l][0] > 0]
        if drops: out.append(f"- From tIoU {ths[0]:g} to {ths[-1]:g} the systems lose {100*min(drops):.0f}-{100*max(drops):.0f} % of their F1. The stricter criterion accepts fewer of the same predicted intervals.")
    return "\n".join(out) + "\n"

# What each error type of the edge chunk match means (metrics.edge_error_types), for the reports
_UNIT_TYPE_WORDS = ("**matched** = the unit alone is matched by a caption. **joined** = the unit is inside a matched caption of 2 or more "
                    "units of one chunk (a correct merge). **cut** = a caption edge lies inside the unit, more than τ from its edges. "
                    "**bridged** = a caption holds the unit (at least half of it) and a unit of another chunk (it crosses a pause). "
                    "**offset** = a caption holds the unit, but an edge is more than τ from the run edges, or another caption took the "
                    "match. **missed** = no caption holds the unit or cuts it.")
_CAPTION_TYPE_WORDS = ("**matched** and **joined** = the caption matches a run of 1 unit, of 2 or more units of one chunk. **cut** = an "
                       "edge lies inside a unit, more than τ from its edges. **bridged** = it holds units of 2 or more chunks. **offset** = "
                       "it holds units of one chunk, but it is not matched. **phantom** = it holds no unit and cuts none.")


def _error_types_md(errors, labels, n_units, taus):
    out = []
    for side, types, words, what in (("units", EDGE_UNIT_TYPES, _UNIT_TYPE_WORDS, f"of the {n_units} caption units"),
                                     ("captions", EDGE_CAPTION_TYPES, _CAPTION_TYPE_WORDS, "of the captions of each system")):
        h = ["system", *[f"{c} {' / '.join(f'{t:g}' for t in taus)} / avg" for c in types]]
        rows = [[l, *[" / ".join(f"{100*r[side][c]:.0f}" for r in errors[l]["taus"]) + f" / {100*errors[l][f'{side}_avg'][c]:.0f} %"
                      for c in types]] for l in labels]
        out.append(f"**Share {what}**, at each τ = {', '.join(f'{t:g}' for t in taus)} s and the mean:\n\n" + _table(h, rows) + "\n" + words + "\n")
    return "\n".join(out) + ("\nEach caption and each unit takes the first type whose rule fits, in the column order. All types come from "
                              "the edge chunk match of the main score: matched + joined is its precision (captions) and its recall (units) "
                              "at each τ. `tables/error_types.csv`.\n")


def _table(header, rows):
    return "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)

def _pr_md(scores, labels, ths):
    h = ["system", *[f"precision {t:g}" for t in ths], "avg", *[f"recall {t:g}" for t in ths], "avg"]
    rows = []
    for l in labels:
        p = [scores[l][t]["segmentation"]["precision"] for t in ths]; r = [scores[l][t]["segmentation"]["recall"] for t in ths]
        rows.append([l, *[f"{x:.3f}" for x in p], f"{np.mean(p):.3f}", *[f"{x:.3f}" for x in r], f"{np.mean(r):.3f}"])
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
            "An end more than τ late fails the edge match at τ by itself.\n")

def _cost_md(cost, ths):
    h = ["system", "version", "sentences", "F1 " + " / ".join(f"{t:g}" for t in ths), "avg", "DVC BLEU-4 " + " / ".join(f"{t:g}" for t in ths), "avg"]
    return _table(h, [[r["system"], r["version"], r["events"], " / ".join(f"{r[f'f1@{t:g}']:.3f}" for t in ths), f"{np.mean([r[f'f1@{t:g}'] for t in ths]):.3f}",
                       " / ".join(f"{r[f'dvc_bleu4@{t:g}']:.2f}" for t in ths), f"{np.mean([r[f'dvc_bleu4@{t:g}'] for t in ths]):.2f}"] for r in cost])

def _decision_md(edges, long_form, by_ref_class, reference):
    out = []
    out.append("**(1) Gold-edge recall within δ**\n\n" + _table(["system", "starts", "ends", "ends, pause after", "ends, abutting"],
               [[r["system"], f"{r['start_recall']:.3f} ({r['n_start']})", f"{r['end_recall']:.3f} ({r['n_end']})", f"{r['pause_recall']:.3f} ({r['n_pause']})",
                 f"{r['abutting_recall']:.3f} ({r['n_abutting']})"] for r in edges]))
    if by_ref_class:
        out.append(f"**(2) Localized BLEU-4 against `{reference}`, by the class of `{reference}`'s span**\n\n" + _table(["system", "span class", "τ (s)", "units", "mean delta"],
                   [[r["system"], r["span_class"], f"{r['tau_s']:g}", r["n_units"], r["mean_delta"]] for r in by_ref_class]))
    out.append("**(3) ReSCUE's commit-only axes**\n\n" + _table(["system", "long-form BLEU-4", "dropped events", "LAAL (s)", "AL (s)", "units covered"],
               [[r["system"], f"{r['long_form_bleu4']:.2f}", f"{r['dropped_share']:.1%}", r["laal_s"], r["al_s"], f"{r['covered_share']:.1%}"] for r in long_form]))
    return "\n".join(out)

def _held_md(rows):
    return _table(["system", "units", "covered", "median (s)", "p90 (s)", "mean (s)"],
                  [[r["system"], r["units"], f"{r['covered_share']:.1%}", r["median_s"], r["p90_s"], r["mean_s"]] for r in rows])

def _shared_md(shared, reference):
    if not shared: return ""
    return ("On the caption units that BOTH systems match alone at τ (type matched, so both sides have the same references), corpus BLEU-4 of each system's captions:\n\n" +
            _table(["system", "τ (s)", "shared units", "this system's BLEU-4", f"{reference}'s BLEU-4 on the same units"],
                   [[r["system"], f"{r['tau_s']:g}", r["n_shared"], f"{r['corpus_bleu4']:.2f}", f"{r['reference_corpus_bleu4']:.2f}"] for r in shared]))

def _sentence_tables_md(vid, tables, labels, taus):
    """Two or three caption units of one video with every system's output: the one most systems locate (matched or joined at
    some τ), the one fewest locate, and the longest one nobody locates."""
    rows = {l: {r["start_s"]: r for r in tables[l].values() if r["video_id"] == vid} for l in labels}
    starts = sorted(rows[labels[0]])
    if not starts: return ""
    hits = lambda l, st: [t for t in taus if rows[l][st]["taus"][t]["type"] in LOCATED]   # the τ at which system l locates the unit
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
        out.append(_table(["system", f"type at τ {' / '.join(f'{t:g}' for t in taus)} s", "output (at the smallest τ that locates it)"],
                          [[l, " / ".join(rows[l][st]["taus"][t]["type"] for t in taus),
                            f"\"{rows[l][st]['taus'][hits(l, st)[0]]['hyp']}\"" if hits(l, st) else "not located"] for l in labels]))
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
             f"With perfect cuts and the clean translator, the same captions give **{o['corpus_bleu4']:.2f}** one way and **{o['per_video_mean']:.2f}** the other way (median video {np.median(list(o['per_video'].values())):.2f}; mean weighted by video size {o['size_weighted_mean']:.2f}).\n",
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
             "\nHow to read: compare all-DVC-pairs corpus BLEU with reported DVC to isolate aggregation on fixed pairs. Compare located-only BLEU with clean captions on exactly the same GT spans to control subset selection. Neither comparison makes DVC recall-sensitive. The all-caption clean video mean above is an aggregation check; apply the same proposal filter before presenting it as a row on true cuts.\n"]
    if ex: lines += ["![](fig/protocol_example.png)\n", f"The example video `{ex}` shows every pair under both scorers at tIoU {tau:g}, with the n-gram counts each pair adds.\n",
                     "![](fig/protocol_thresholds.png)\n",
                     "The same video at every threshold. An overlap becomes a pair only when its tIoU passes the threshold; below it the caption is scored against a random string, however much time the two share. This is why the DVC value falls as the threshold rises.\n"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("results", help="tables, figures and the Markdown skeleton from the events files of the rows")
    r.add_argument("--events", action="append", required=True, metavar="LABEL=PATH")
    r.add_argument("--reference", default=None, help="system the differences compare against (default: the first --events)")
    r.add_argument("--language", required=True); r.add_argument("--split", default="test", choices=["train", "dev", "test"])
    r.add_argument("--boundary-sweep-file", default=None, help="a boundary_sweep_*.json of the clean translator (analyze.py --stage boundary-sweep): adds the confidence section")
    r.add_argument("--history", action="append", default=None, metavar="LABEL=PATH", help="a stage-2 history.csv: adds training curves")
    r.add_argument("--segmenter-eval", action="append", default=None, metavar="LABEL=PATH", help="segmenter_eval_*.json: adds the segmentation-only section")
    r.add_argument("--stability-file", action="append", default=None)
    r.add_argument("--tiou-threshold", type=float, default=0.5, help="tIoU threshold of the case figures and the protocol section")
    r.add_argument("--case-videos", action="append", default=None, metavar="ID[:T0-T1[:NOTE]]", help="pin a video (and window, note) to the real-video section")
    r.add_argument("--report-thresholds", default=None, help="thresholds shown in the tables and figures (default: every scored threshold); every scored threshold stays in tables/*.csv")
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
            p.add_argument("--method", default="dlm", choices=["dlm", "ar", "baseline"]); p.add_argument("--checkpoint", default=None)
            p.add_argument("--max-sentences", type=int, default=12)
    args = ap.parse_args()
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
