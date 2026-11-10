"""Per-unit outcome tables and the breakdowns built on them.

Every reliable gold unit gets one row with a stable `gold_id`, so two systems evaluated on the same split join
row for row. In each cell (tau, t) of the edge-limit grid of the main match the row holds the unit's error type of the edge chunk match
(`metrics.edge_error_types`), on the pairs that count in that cell, so these tables use its names. Flip tables and every
breakdown are column operations on these tables. The reported scores are in metrics.py.
"""
from __future__ import annotations
import numpy as np

from metrics import holds, sentence_bleu_scores, unit_holders

LOCATED = ("matched", "joined")   # the unit types inside a run whose pair counts in a cell of the edge-limit grid
OUTCOME_COLUMNS = ["gold_id", "video_id", "tau_s", "tiou", "type", "start_s", "end_s", "duration_s", "ref_words", "oov_rate", "run_units",
                   "start_error_s", "end_error_s", "bleu4", "localized_bleu4", "ref", "run_ref", "hyp"]


def strict_cell(cells) -> tuple:   # the strictest cell (tau, t) of the grid: the smallest edge limit tau, then the largest t
    return min(cells, key=lambda c: (c[0], -c[1]))


def loose_cell(cells) -> tuple:    # the loosest cell (tau, t) of the grid: the largest edge limit tau, then the smallest t
    return max(cells, key=lambda c: (c[0], -c[1]))


def unit_outcomes(chunks: dict, pred: dict, errors: dict, char_level: bool) -> list[dict]:
    """One row per reliable gold unit of `chunks` (evaluation.predictions.gold_chunks: per source video, in time order), so
    `gold_id` is `<source video>:<unit index>`, the unit ids of `errors` = metrics.edge_error_types(pred, chunks, taus, thresholds).
    Per cell (tau, t) of the grid, `cells[(tau, t)]` holds the unit's type and, for a located unit (matched or joined), the caption
    whose pair with the unit's run counts in that cell: the run's units and joined text (`run_ref`), the caption's start and end
    minus the run's, and `bleu4`, the sentence BLEU-4 of the caption against `run_ref`, so a joined unit takes the score of its
    run. `localized_bleu4` is that score, and 0 for a unit that is not located."""
    cells, rows, pairs = [(c["tau_s"], c["tiou"]) for c in errors["cells"]], [], {}
    for vid in sorted(chunks):
        units, evs = [u for chunk in chunks[vid] for u in chunk], pred.get(vid, [])
        if not units: continue
        per = errors["per_video"][vid]
        runs = [{k: (pi, ids) for pi, ids in c["runs"] for k in ids} for c in per["cells"]]
        for k, u in enumerate(units):
            row = {"gold_id": f"{vid}:{k}", "video_id": vid, "start_s": float(u.start_s), "end_s": float(u.end_s),
                   "duration_s": float(u.end_s - u.start_s), "ref": u.text, "ref_words": len(u.text.split()), "cells": {}}
            for j, cell in enumerate(cells):
                at = row["cells"][cell] = {"type": per["cells"][j]["units"][k], "run_units": 0, "start_error_s": None, "end_error_s": None,
                                           "bleu4": None, "localized_bleu4": 0.0, "run_ref": None, "hyp": None}
                if k not in runs[j]: continue
                pi, ids = runs[j][k]
                at.update(run_units=len(ids), start_error_s=float(evs[pi].start_s - units[ids[0]].start_s),
                          end_error_s=float(evs[pi].end_s - units[ids[-1]].end_s), run_ref=" ".join(units[i].text for i in ids),
                          hyp=evs[pi].text or "")
                pairs.setdefault((at["hyp"], at["run_ref"]), []).append(at)   # each distinct pair is scored once
            rows.append(row)
    keys = list(pairs)
    for key, b in zip(keys, sentence_bleu_scores([h for h, _ in keys], [r for _, r in keys], char_level=char_level)):
        for at in pairs[key]: at["bleu4"] = at["localized_bleu4"] = round(float(b), 2)
    return rows


def unit_span_classes(gold: dict, events: dict, cover: float = 0.5) -> dict:
    """{video: [per reliable gold unit: '1-unit span', 'multi-unit span' or 'none']}: the class of the decoded span that holds it, by
    how many gold units that span holds. It splits a translation comparison on the same spans into the part that conditioning and
    training explain (1-unit spans) and the part multi-unit targets explain."""
    holders = unit_holders(gold, events, cover)
    return {vid: ["none" if h is None else ("1-unit span" if sum(holds(h, u, cover) for u in units) == 1 else "multi-unit span")
                  for h in holders[vid]] for vid, units in gold.items()}


def edge_recall(gold: dict, pred: dict, delta_s: float, pause_after: dict) -> dict:
    """Gold-edge recall within delta_s (A2b, A2c): the share of reliable gold starts and ends that some predicted event edge lies
    within delta_s of. Ends are split by boundary type with the sampler's rule (`target_pause_after`): 'pause' when the record's
    next unit starts more than delta_s later, else 'abutting'. `pause_after` = {video: [bool per gold unit]}."""
    n = {"start": 0, "end": 0, "pause": 0, "abutting": 0}; hit = dict.fromkeys(n, 0)
    for vid, units in gold.items():
        starts = np.array([e.start_s for e in pred.get(vid, [])]); ends = np.array([e.end_s for e in pred.get(vid, [])])
        near = lambda xs, t: bool(len(xs) and np.min(np.abs(xs - t)) <= delta_s)
        for g, pause in zip(units, pause_after[vid]):
            kind = "pause" if pause else "abutting"
            for key, ok in (("start", near(starts, g.start_s)), ("end", near(ends, g.end_s)), (kind, near(ends, g.end_s))):
                n[key] += 1; hit[key] += int(ok)
    return {**{f"{k}_recall": round(hit[k] / n[k], 4) if n[k] else float("nan") for k in n}, **{f"n_{k}": n[k] for k in n}}


def flip_table(rows_a: dict, rows_b: dict, cell: tuple, margin: float = 1.0) -> dict:
    """Per unit in the cell (tau, t), system A against B: `fixed` (A locates it, B does not), `broken` (B locates it, A does not), and
    of the units both locate, `better` and `worse` (A's sentence BLEU-4 more than `margin` above or below B's). Located = matched or
    joined."""
    fixed = broken = better = worse = 0
    for gid, ra in rows_a.items():
        a, b = ra["cells"][cell], rows_b[gid]["cells"][cell]
        am, bm = a["type"] in LOCATED, b["type"] in LOCATED
        if am and not bm: fixed += 1
        elif bm and not am: broken += 1
        elif am and bm:
            d = float(a["bleu4"]) - float(b["bleu4"])
            if d > margin: better += 1
            elif d < -margin: worse += 1
    return {"fixed": fixed, "broken": broken, "better": better, "worse": worse}


def bin_stats(rows, edges, key, labels=None):
    """Per bin of `key` (a field with no cell, such as `duration_s`): the units, the located share (matched or joined) as the mean
    over the cells of the grid, the located share in the strictest cell (`strict_cell`), and the mean localized BLEU-4 over the
    units and the cells."""
    out = []
    for k, (lo, hi) in enumerate(zip(edges, edges[1:])):
        sel = [r for r in rows if lo <= r[key] < hi]
        cells = list(sel[0]["cells"]) if sel else []
        located = lambda cell: float(np.mean([r["cells"][cell]["type"] in LOCATED for r in sel]))
        out.append({"bin": labels[k] if labels else f"[{lo:g},{hi:g})", "n": len(sel),
                    "located_rate": float(np.mean([located(c) for c in cells])) if sel and cells else None,
                    "strict_rate": located(strict_cell(cells)) if sel and cells else None,
                    "mean_localized_bleu4": float(np.mean([r["cells"][c]["localized_bleu4"] for r in sel for c in cells])) if sel and cells else None})
    return out


def boundary_stats(errors: dict) -> dict:
    """Where the edges land, from the captions that hold exactly 1 unit that no other caption holds (`pairs` of
    metrics.edge_error_types, the edge offsets of analyze.py --stage segmenter-errors): the median start and end errors (caption
    minus unit, positive = late), and per edge limit tau the share of these captions whose end is more than tau late, which alone
    fails the edge limit at tau. Edge offsets read tau only, never a tIoU threshold."""
    pairs = [(dh, dt) for v in errors["per_video"].values() for _, _, dh, dt, _ in v["pairs"]]
    heads, tails = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    taus = dict.fromkeys(c["tau_s"] for c in errors["cells"])   # the taus of the grid, in its order
    return {"median_head_s": float(np.median(heads)) if pairs else None, "median_tail_s": float(np.median(tails)) if pairs else None,
            **{f"late_end@{tau:g}": float(np.mean(tails > tau)) if pairs else None for tau in taus}, "n": len(pairs)}


def case_studies(rows: list[dict], pred: dict, errors: dict, k: int = 5) -> dict:
    """Rule-selected exemplars; never hand-picked. Per cell (tau, t) of the grid, in the order of errors["cells"]: the k longest units
    (by reference words) of each lost type, the first k phantom captions, and the k located units with the lowest sentence BLEU-4;
    then the first k forced-partial and cap-cut commits."""
    cells = []
    for j, c in enumerate(errors["cells"]):
        cell = (c["tau_s"], c["tiou"])
        flat = [{**{f: x[f] for f in x if f != "cells"}, "tau_s": cell[0], "tiou": cell[1], **x["cells"][cell]} for x in rows]
        cases = {"tau_s": cell[0], "tiou": cell[1], **{kind: sorted((x for x in flat if x["type"] == kind), key=lambda x: -x["ref_words"])[:k]
                                                       for kind in ("cut", "bridged", "offset", "missed")}}
        cases["phantom"] = [{"video_id": v, "start_s": float(e.start_s), "end_s": float(e.end_s), "text": (e.text or "")[:120]}
                            for v, per in sorted(errors["per_video"].items())
                            for e, kind in zip(pred.get(v, []), per["cells"][j]["captions"]) if kind == "phantom"][:k]
        cases["worst_located"] = sorted((x for x in flat if x["type"] in LOCATED), key=lambda x: (x["bleu4"], -x["duration_s"]))[:k]
        cells.append(cases)
    flagged = lambda name: [{"video_id": v, "start_s": float(e.start_s), "end_s": float(e.end_s)}
                            for v in sorted(pred) for e in pred[v] if getattr(e, name, False)][:k]
    return {"cells": cells, "forced_partial": flagged("flagged_partial"), "cap_cut": flagged("cap_cut")}
