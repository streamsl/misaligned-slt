"""Every report figure: data in, one PNG out. No file parsing here; `report.py` supplies the numbers."""
from __future__ import annotations
import textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from metrics import EDGE_CAPTION_TYPES, EDGE_UNIT_TYPES
from reporting.outcomes import LOCATED, bin_stats, loose_cell, strict_cell

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "figure.facecolor": "white",
                     "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
PALETTE = ["#8c8c8c", "#5b5b5b", "#7fb3d5", "#a9cce3", "#c0392b", "#e59866", "#2e7d32", "#8e44ad"]
# The error types of the edge chunk match (metrics.EDGE_CAPTION_TYPES, EDGE_UNIT_TYPES): green = matched, the rest = errors.
TYPE_COLOR = {"matched": "#2e7d32", "joined": "#82c785", "cut": "#e59866", "bridged": "#c0392b", "offset": "#8e44ad",
              "phantom": "#34495e", "missed": "#7f8c8d"}
GOLD, PRED_OK, PRED_BAD = "#2e7d32", "#c6e0b4", "#f8cbad"
SMALL_C, BIG_C, REST_C = "#3b6ea5", "#c0392b", "#d9d9d9"   # the two example videos, and every other video


def colors(labels): return {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(labels)}
def _save(fig, path, tight=True):
    # tight=False for figures that place their own axes: a tight box would grow the page around long text.
    Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, bbox_inches="tight" if tight else None); plt.close(fig)
def _wrap(s, n=28): return "\n".join(textwrap.wrap(s, n)) or s


# ---------------------------------------------------------------- static schematics
def experiment_map(path):
    fig, ax = plt.subplots(figsize=(13, 7.4)); ax.set_xlim(0, 100); ax.set_ylim(-8, 100); ax.axis("off")
    def box(x, y, w, h, text, fc, fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", fc=fc, ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    def arrow(x1, y1, x2, y2, color="#555", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color=color, lw=1.4, linestyle=ls, shrinkA=2, shrinkB=2))
    ax.text(50, 96, "What is shared, and what changes, between the experiments", ha="center", fontsize=14, weight="bold")
    box(2, 74, 22, 12, "Pose video", "#eef2f7"); box(2, 57, 22, 12, "Pose encoders\n(one per model)", "#eef2f7"); arrow(13, 74, 13, 69)
    box(29, 78, 33, 9, "Segmenter A: external baseline", "#f2f2f2", 9.5)
    box(29, 65, 33, 10, "Segmenter B: S1\n(pre-trained on 3 sign languages)", "#dfe9f5", 9.5)
    arrow(24, 63, 29, 70); arrow(24, 63, 29, 82.5)
    box(66, 71, 31, 10, "Sentence-length model\n(same for both segmenters)", "#dfe9f5", 9.5); arrow(62, 82.5, 66, 78); arrow(62, 70, 66, 74)
    box(29, 43, 33, 10, "Translator 1: pretrimmed translator\ntrained on perfect segments only", "#f2f2f2", 9.5)
    box(29, 27, 33, 13, "Translator 2: our model\ntrained on imperfect segments\n(AR and DLM versions)", "#dfe9f5", 9.5)
    arrow(24, 59, 29, 48); arrow(24, 59, 29, 33.5)
    box(66, 43, 31, 10, "CASCADE\nsegment first, then translate each part", "#fdf3e6", 9.5)
    box(66, 29, 31, 10, "OFFLINE\nour model segments the whole video", "#fdf3e6", 9.5)
    box(66, 15, 31, 10, "ONLINE\nour model segments and translates\nas it watches", "#fce8e6", 9.5)
    box(66, 1, 31, 10, "Oracle\ntrue segment times are given", "#f2f2f2", 9.5)
    arrow(62, 48, 66, 48); arrow(62, 34, 66, 34); arrow(62, 30, 66, 20); arrow(50, 43, 64, 8, color="#999", ls=":"); arrow(82, 71, 82, 53)
    ax.text(81.5, -5.0, "Every experiment is scored with the SAME metric on the SAME gold units.", ha="center", fontsize=9.5, style="italic")
    _save(fig, path)


def metric_explained(path):
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 6.6), gridspec_kw={"height_ratios": [1, 1.25, 1.6]})
    T = 60; gold = [(2, 20), (20, 38), (42, 57)]; pred = [(2, 19), (19, 38), (43, 58)]
    for ax in axes: ax.set_xlim(0, T); ax.set_yticks([]); ax.set_xticks(range(0, T + 1, 10)); ax.spines["left"].set_visible(False)
    axes[0].set_ylim(0, 1); axes[0].set_title("1. True sentences from the subtitles. Sentence 1 and 2 have touching annotation intervals", loc="left")
    axes[0].add_patch(Rectangle((0, 0.25), T, 0.5, fc="#f0f0f0", ec="none"))
    for i, (a, b) in enumerate(gold):
        axes[0].add_patch(Rectangle((a, 0.25), b - a, 0.5, fc=GOLD, ec="white", lw=1.5)); axes[0].text((a + b) / 2, 0.5, f"S{i+1}", ha="center", va="center", color="white", weight="bold")
    ax = axes[1]; ax.set_ylim(-0.05, 1.3); ax.set_title("2. The system produces time intervals with text", loc="left")
    for a, b in [(0, 2), (38, 43), (58, T)]: ax.add_patch(Rectangle((a, 0.25), b - a, 0.5, fc="#dcdcdc", ec="white"))
    for i, (a, b) in enumerate(pred):
        ax.add_patch(Rectangle((a, 0.25), b - a, 0.5, fc="#1565c0", ec="white", lw=1.5)); ax.text((a + b) / 2, 0.5, f"P{i+1}", ha="center", va="center", color="white", weight="bold")
    ax.text(0.5, 1.12, "grey = no signing", fontsize=9.5, color="#555")
    ax = axes[2]; ax.set_ylim(-0.45, 1.3); ax.set_title("3. Scoring: pair each prediction with at most one caption unit, by time overlap (tIoU)", loc="left")
    for i, (a, b) in enumerate(gold): ax.add_patch(Rectangle((a, 0.75), b - a, 0.35, fc=GOLD, ec="white")); ax.text((a + b) / 2, 0.925, f"true S{i+1}", ha="center", va="center", color="white", fontsize=9)
    for i, (a, b) in enumerate(pred): ax.add_patch(Rectangle((a, 0.1), b - a, 0.35, fc="#1565c0", ec="white")); ax.text((a + b) / 2, 0.275, f"pred P{i+1}", ha="center", va="center", color="white", fontsize=9)
    for gi, txt in enumerate(["tIoU 0.85", "tIoU 0.94", "tIoU 0.86"]):
        gx = sum(gold[gi]) / 2; px = sum(pred[gi]) / 2
        ax.annotate("", xy=(gx, 0.75), xytext=(px, 0.45), arrowprops=dict(arrowstyle="<->", color="#555")); ax.text((gx + px) / 2 + 1, 0.60, txt, fontsize=8.5, ha="center")
    ax.text(30, -0.30, "tIoU = shared time / total time covered. One prediction can take only ONE caption unit.", ha="center", fontsize=9.5, style="italic")
    ax.set_xlabel("time (frames)"); fig.suptitle("How a per-frame model is scored with a localization metric", fontsize=13, weight="bold", y=1.0)
    fig.tight_layout(); _save(fig, path)


# ---------------------------------------------------------------- scores
def _grouped(ax, systems, values, thresholds, fmt, ylabel):
    x = np.arange(len(systems)); w = 0.8 / len(thresholds); shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(thresholds)))
    for j, t in enumerate(thresholds):
        v = [values[s][t] for s in systems]
        ax.bar(x + (j - (len(thresholds) - 1) / 2) * w, v, w, color=shades[j], label=f"tIoU {t:g}")
        for i, y in enumerate(v): ax.text(i + (j - (len(thresholds) - 1) / 2) * w, y, fmt.format(y), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([_wrap(s, 18) for s in systems], fontsize=8); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)


def scores_at_thresholds(scores, systems, thresholds, path, n_gold):
    """scores[system][t] = the row of threshold t of the densevid block (metrics.evaluate_predicted_events `by_tiou`): the tIoU scores."""
    fig, axes = plt.subplots(2, 1, figsize=(max(9, 2.2 * len(systems)), 8))
    _grouped(axes[0], systems, {s: {t: scores[s][t]["f1"] for t in thresholds} for s in systems}, thresholds, "{:.3f}", "F1")
    _grouped(axes[1], systems, {s: {t: scores[s][t]["dvc"]["bleu4"] for t in thresholds} for s in systems}, thresholds, "{:.2f}", "DVC BLEU-4")
    axes[0].set_title("tIoU scores: sentence tIoU F1"); axes[1].set_title("tIoU scores: DVC BLEU-4 (video mean)")
    axes[0].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.005, 0.5))
    fig.suptitle(f"{n_gold} caption units", fontsize=13, weight="bold"); fig.tight_layout(); _save(fig, path)


def precision_recall(scores, systems, thresholds, path):
    fig, axes = plt.subplots(2, 1, figsize=(max(9, 2.2 * len(systems)), 8))
    _grouped(axes[0], systems, {s: {t: scores[s][t]["precision"] for t in thresholds} for s in systems}, thresholds, "{:.3f}", "precision")
    _grouped(axes[1], systems, {s: {t: scores[s][t]["recall"] for t in thresholds} for s in systems}, thresholds, "{:.3f}", "recall")
    axes[0].set_title("Precision: of what we produce, how much is right"); axes[1].set_title("Recall: of the caption units, how many we find")
    axes[0].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.005, 0.5)); fig.tight_layout(); _save(fig, path)


def located_vs_quality(scores, systems, thresholds, path, n_gold):
    """How many sentences are located, and the matched-pair sentence BLEU of those, per threshold."""
    fig, axes = plt.subplots(2, 1, figsize=(max(9, 2.2 * len(systems)), 8))
    _grouped(axes[0], systems, {s: {t: scores[s][t]["scored_pairs"] for t in thresholds} for s in systems}, thresholds, "{:.0f}", f"sentences located (of {n_gold})")
    _grouped(axes[1], systems, {s: {t: scores[s][t]["located_corpus_bleu4"] for t in thresholds} for s in systems}, thresholds, "{:.1f}", "corpus BLEU-4 of the located pairs")
    axes[0].set_title("Sentences located per tIoU threshold"); axes[1].set_title("Corpus BLEU-4 of the located pairs per tIoU threshold")
    axes[0].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.005, 0.5)); fig.tight_layout(); _save(fig, path)


def _type_bars(ax, shares, systems, types, min_label):
    # 1 stacked bar per system: the share (%) of each error type, in the order of its rule
    bottom = np.zeros(len(systems))
    for c in types:
        v = np.array([shares[s][c] for s in systems]) * 100
        ax.bar(np.arange(len(systems)), v, bottom=bottom, color=TYPE_COLOR[c], label=c)
        for i, (a, b) in enumerate(zip(v, bottom)):
            if a > min_label: ax.text(i, b + a / 2, f"{a:.0f}", ha="center", va="center", color="white", fontsize=8.5, weight="bold")
        bottom += v


def error_types(errors, systems, count_ratio, path, n_units):
    """errors[system] = metrics.edge_error_types output; count_ratio[system] = captions / units. Top row: the share of the
    reliable units of each type, per edge limit tau (the mean over the tIoU thresholds of its cells) and over all cells (`avg`,
    `units_avg`); bottom row: the same for the captions. Every cell is in tables/error_types.csv."""
    taus = list(dict.fromkeys(c["tau_s"] for c in errors[systems[0]]["cells"]))
    at = lambda s, side, tau: {k: float(np.mean([c[side][k] for c in errors[s]["cells"] if c["tau_s"] == tau])) for k in errors[s][f"{side}_avg"]}
    panels = [(f"τ = {tau:g} s", {side: {s: at(s, side, tau) for s in systems} for side in ("units", "captions")}) for tau in taus]
    panels.append(("all cells (avg)", {side: {s: errors[s][f"{side}_avg"] for s in systems} for side in ("units", "captions")}))
    fig, axes = plt.subplots(2, len(panels) + 1, figsize=(4.2 * len(panels) + 3.4, 9.0), gridspec_kw={"width_ratios": [1] * len(panels) + [0.8]})
    for j, (name, shares) in enumerate(panels):
        _type_bars(axes[0, j], shares["units"], systems, EDGE_UNIT_TYPES, 7)
        _type_bars(axes[1, j], shares["captions"], systems, EDGE_CAPTION_TYPES, 7)
        axes[0, j].set_title(f"units, {name}"); axes[1, j].set_title(f"captions, {name}")
    axes[0, 0].set_ylabel(f"share of the {n_units} caption units (%)"); axes[1, 0].set_ylabel("share of the captions (%)")
    ratio = [count_ratio[s] for s in systems]; col = colors(systems)
    axes[0, -1].bar(np.arange(len(systems)), ratio, color=[col[s] for s in systems]); axes[0, -1].axhline(1.0, color=GOLD, ls="--", lw=1.5)
    axes[0, -1].set_title("captions produced\n/ caption units")
    for i, v in enumerate(ratio): axes[0, -1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    axes[1, -1].axis("off")
    for ax in axes.flat:
        if not ax.axison: continue
        ax.set_xticks(np.arange(len(systems))); ax.set_xticklabels([_wrap(s, 16) for s in systems], fontsize=7.5, rotation=30, ha="right"); ax.grid(axis="y", alpha=0.3)
    handles = {lab: h for ax in (axes[0, 0], axes[1, 0]) for h, lab in zip(*ax.get_legend_handles_labels())}
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=7, fontsize=10, frameon=False)
    fig.suptitle("Error types of the counted pairs of the edge-limit grid (each τ: the mean over the tIoU thresholds)", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1]); _save(fig, path)


def boundary_error(stats, systems, path, taus):
    """stats[system] = reporting.outcomes.boundary_stats: the median edge errors of the captions that hold exactly 1 unit that no
    other caption holds, and per tau the share of them whose end is more than tau late."""
    x = np.arange(len(systems)); w = 0.38
    fig, axes = plt.subplots(1, 2 if taus else 1, figsize=(13.5 if taus else 7.0, 4.6), squeeze=False)   # no edge limit τ: no right panel
    axes = axes[0]
    mh = [stats[s]["median_head_s"] or 0 for s in systems]; mt = [stats[s]["median_tail_s"] or 0 for s in systems]
    axes[0].bar(x - w / 2, mh, w, color="#7fb3d5", label="start error"); axes[0].bar(x + w / 2, mt, w, color="#c0392b", label="end error")
    for i, (a, b) in enumerate(zip(mh, mt)): axes[0].text(i - w / 2, a, f"{a:+.2f}", ha="center", va="bottom", fontsize=8); axes[0].text(i + w / 2, b, f"{b:+.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].axhline(0, color="#333", lw=1); axes[0].set_ylabel("median error (seconds)"); axes[0].set_title("Median edge error of the captions that hold one unit (positive = late)"); axes[0].legend(fontsize=9)
    wt = 0.8 / max(len(taus), 1); shades = plt.cm.Reds(np.linspace(0.45, 0.95, len(taus)))
    for j, t in enumerate(taus if len(axes) > 1 else []):
        v = [100 * (stats[s][f"late_end@{t:g}"] or 0) for s in systems]; dx = (j - (len(taus) - 1) / 2) * wt
        axes[1].bar(x + dx, v, wt, color=shades[j], label=f"end more than {t:g} s late")
        for i, y in enumerate(v): axes[1].text(i + dx, y, f"{y:.0f}%", ha="center", va="bottom", fontsize=7)
    if len(axes) > 1: axes[1].set_ylabel("share of these captions (%)"); axes[1].set_title("How often the end alone fails the edge limit τ"); axes[1].legend(fontsize=8)
    for ax in axes: ax.set_xticks(x); ax.set_xticklabels([_wrap(s, 16) for s in systems], fontsize=8, rotation=20, ha="right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); _save(fig, path)


def by_length(tables, systems, edges, labels, path, cap_s):
    col = colors(systems); cells = next(iter(tables[systems[0]].values()))["cells"]; tau, t = strict_cell(cells) if cells else (None, None)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    counts = None
    for s in systems:
        st = bin_stats(list(tables[s].values()), edges, "duration_s", labels)
        if counts is None: counts = [b["n"] for b in st]
        axes[0].plot(labels, [b["located_rate"] for b in st], marker="o", color=col[s], label=s)
        axes[1].plot(labels, [b["strict_rate"] for b in st], marker="o", color=col[s])
        axes[2].plot(labels, [b["mean_localized_bleu4"] for b in st], marker="o", color=col[s])
    axes[0].set_title("Located (matched or joined), mean over the cells"); axes[1].set_title(f"Located at τ = {tau:g} s and tIoU {t:g} (the strictest cell)" if cells else "no edge limit τ: no strictest cell")
    axes[2].set_title("Localized BLEU-4, mean over the cells")
    axes[0].set_ylabel("share of caption units"); axes[0].legend(fontsize=8)
    for i, c in enumerate(counts or []): axes[0].annotate(f"n={c}", (i, 0.02), fontsize=8, ha="center", color="#666")
    for ax in axes: ax.set_xlabel(f"caption unit length (cap = {cap_s:.0f} s)"); ax.grid(alpha=0.3); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); _save(fig, path)


def by_oov(tables, systems, path):
    edges = [0.0, 0.001, 0.1, 0.2, 0.35, 1.01]; labels = ["0", "(0,.1)", "[.1,.2)", "[.2,.35)", ">=.35"]; col = colors(systems)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for s in systems:
        st = bin_stats(list(tables[s].values()), edges, "oov_rate", labels)
        ax.plot(labels, [b["mean_localized_bleu4"] for b in st], marker="o", color=col[s], label=s)
    ax.set_xlabel("share of reference words not seen in training"); ax.set_ylabel("mean localized BLEU-4 (mean over the cells)"); ax.set_title("Localized BLEU-4 by share of unseen reference words")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout(); _save(fig, path)


def bleu_by_edge_error(tables, systems, path):
    """The mean sentence BLEU-4 of the located units (matched or joined) in the loosest cell of the grid (the largest tau, the
    smallest t), by the larger edge error of their caption against the run (|start error| or |end error|), binned up to that tau."""
    cells = list(next(iter(tables[systems[0]].values()))["cells"])
    cell, taus = loose_cell(cells), sorted({tau for tau, _ in cells})
    col = colors(systems); top = cell[0]; edges = [0.0, taus[0] / 2, *taus[:-1], top + 1e-9]
    err = lambda a: max(abs(a["start_error_s"]), abs(a["end_error_s"]))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for s in systems:
        m = [r["cells"][cell] for r in tables[s].values() if r["cells"][cell]["type"] in LOCATED]
        pts = [(0.5 * (lo + min(hi, top)), np.mean([a["bleu4"] for a in m if lo <= err(a) < hi])) for lo, hi in zip(edges, edges[1:]) if any(lo <= err(a) < hi for a in m)]
        if pts: ax.plot(*zip(*pts), marker="o", color=col[s], label=s)
    ax.set_xlabel(f"larger edge error of the located caption (s, bin centre; τ = {top:g} s, tIoU {cell[1]:g})"); ax.set_ylabel("sentence BLEU-4")
    ax.set_title("Translation quality against boundary quality"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, path)


def frame_vs_sentence(evals, thresholds, path):
    """evals[label] = report._segmentation_only(the _scores.json of a no-text row), in the layout of every row: the main score (the F1
    of the tIoU match with no edge limit, avg) and its F1 per tIoU threshold, the stricter grid's F1 per edge limit tau (over the
    thresholds), and the captions per unit, per source video; the frame F1 per stream."""
    names = list(evals); x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw={"width_ratios": [1.8, 1]})
    edge = {n: evals[n]["main"]["edge_chunk"] for n in names}
    f1_by = lambda rows, key: {float(r[key]): r["f1"] for r in rows}
    taus = list(dict.fromkeys(r["tau_s"] for n in names for r in edge[n]["by_tau"]))
    reds, purples = plt.cm.Reds(np.linspace(0.45, 0.95, len(thresholds))), plt.cm.Purples(np.linspace(0.45, 0.95, len(taus)))
    bars = [("F1 avg (main: tIoU match)", GOLD, [edge[n]["no_edge_limit"]["avg"]["f1"] for n in names]),
            ("signing/non-signing F1", "#90caf9", [evals[n]["frame"]["bio_f1"] for n in names]),
            *[(f"F1, tIoU {t:g}", reds[j], [f1_by(edge[n]["no_edge_limit"]["by_tiou"], "tiou").get(float(t), 0) for n in names])
              for j, t in enumerate(thresholds)],
            *[(f"F1, edge τ {t:g} s (stricter)", purples[j], [f1_by(edge[n]["by_tau"], "tau_s").get(float(t), 0) for n in names])
              for j, t in enumerate(taus)]]
    w = 0.8 / len(bars)
    for i, (label, color, v) in enumerate(bars): axes[0].bar(x + (i - (len(bars) - 1) / 2) * w, v, w, color=color, label=label)
    axes[0].set_ylim(0, 1.25); axes[0].set_ylabel("score"); axes[0].legend(fontsize=7.5, ncol=3, loc="upper center"); axes[0].set_title("Signing coverage can be high while boundaries are wrong")
    ratio = [edge[n]["captions"] / max(1, edge[n]["units"]) for n in names]
    axes[1].bar(x, ratio, color=PALETTE[: len(names)]); axes[1].axhline(1.0, color=GOLD, ls="--"); axes[1].set_title("Sentences produced / true count")
    for i, v in enumerate(ratio): axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center")
    for ax in axes: ax.set_xticks(x); ax.set_xticklabels([_wrap(n, 16) for n in names]); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Segmentation only, no translation", fontsize=13, weight="bold"); fig.tight_layout(); _save(fig, path)


# ---------------------------------------------------------------- training curves
def _series(rows, col):
    xs, ys = [], []
    for i, r in enumerate(rows):
        v = r.get(col)
        if v in (None, "", "-"): continue
        xs.append(float(r["epoch"]) if r.get("epoch") not in (None, "", "-") else i + 1); ys.append(float(v))
    return xs, ys


def head_decay(histories, path):
    """histories[label] = rows of a stage-2 history.csv. The monitor's whole-video segmentation on dev and the BIO losses."""
    col = colors(list(histories)); fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, rows in histories.items():
        mon = "val_offline" if any(r.get("val_offline_f1") not in (None, "", "-") for r in rows) else "val_online"   # the run's monitor protocol
        for ax, key in zip(axes[:2], (f"{mon}_f1", f"{mon}_events_per_gold")):
            xs, ys = _series(rows, key)
            if ys: ax.plot(xs, ys, color=col[name], label=name)
        for key, ls in (("val_bio_loss", "-"), ("train_bio_loss", ":")):
            xs, ys = _series(rows, key)
            if ys: axes[2].plot(xs, ys, color=col[name], ls=ls, label=f"{name} ({key.split('_')[0]})")
    axes[1].axhline(1.0, color=GOLD, ls="-.", lw=1.4)
    axes[0].set_title("Monitor on dev: F1 of the tIoU match, avg\n(val_offline_f1 or val_online_f1)"); axes[1].set_title("Events per gold unit\n(dashed green = the gold rate)"); axes[2].set_title("BIO loss: train and dev")
    for ax in axes: ax.set_xlabel("epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, path)


def history_facets(rows, cols, title, path):
    ncol = min(3, len(cols)); nrow = (len(cols) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.7 * nrow), squeeze=False)
    for k, c in enumerate(cols):
        ax = axes[k // ncol][k % ncol]; xs, ys = _series(rows, c)
        if not ys: ax.set_axis_off(); continue
        ax.plot(xs, ys, marker=".", ms=3, color="#4c78a8")
        if all(y > 0 for y in ys) and max(ys) / min(ys) > 100: ax.set_yscale("log")
        best = (min(ys), "min") if "loss" in c else (max(ys), "max")
        ax.axhline(best[0], color="#e45756", ls=":", lw=0.8); ax.set_title(f"{c}\n({best[1]} {best[0]:.4g})", fontsize=8); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    for k in range(len(cols), nrow * ncol): axes[k // ncol][k % ncol].set_axis_off()
    fig.suptitle(title, fontsize=11); fig.supxlabel("epoch", fontsize=9); fig.tight_layout(rect=(0, 0.02, 1, 0.97)); _save(fig, path)


# ---------------------------------------------------------------- timelines
NGRAM_HEADER = f"{'':52s}{'tIoU':>6s}   {'1-gram':>9s} {'2-gram':>9s} {'3-gram':>9s} {'4-gram':>9s}   (matched/total)"


def case_legend(tau, width=150):
    """What the numbers count, and what the colours and links mean. Wrapped, so a figure never grows around it."""
    parts = ["S<k> = the k-th caption unit of the whole video. P<k> = the k-th SCORED event of that system in the video. Both keep their whole-video number when a window is shown. "
             "Grey hatched boxes are events excluded by annotation ignore regions; they carry no number.",
             f"A link joins an event to every sentence it materially overlaps, labelled with the tIoU. DVC pairs an event with a sentence ONLY when the tIoU reaches {tau:g}: "
             f"solid = the one-to-one match, dotted grey = an extra DVC pair, dotted red = overlap below {tau:g}. An event with no pair is scored against a random string (orange bar); "
             f"a paired event is scored against its caption unit (green bar)."]
    return "\n".join("\n".join(textwrap.wrap(t, width)) for t in parts)


def _pair_row(label, tiou, c):
    """One aligned line of a text block: a label, the tIoU when there is one, and the matched/total n-grams."""
    cells = " ".join(f"{m}/{t}".rjust(9) for m, t in zip(c["counts"], c["totals"]))
    return f"{label:52s}{(f'{tiou:.2f}' if tiou is not None else ''):>6s}   {cells}"


MONO_COLS, PAGE_W, LANE_L = 200, 13.5, 0.185   # monospace columns that fit PAGE_W; page width; left margin for lane labels
LANE_H, TITLE_H, XAXIS_H, MONO_H = 0.46, 0.20, 0.44, 0.105   # inches: one lane, one title line, the x axis, one text line


def _block_h(n_lanes, title_lines=1):
    """Inches a timeline needs: its title (drawn above the box), the lanes (inside it), the ticks and x label (below)."""
    return title_lines * TITLE_H + LANE_H * n_lanes + 0.24 + XAXIS_H


def _timeline_axes(fig, top_frac, n_lanes, H, title_lines=1):
    """Place a timeline whose block starts at `top_frac`; returns (axes, the fraction where the block ends)."""
    block = _block_h(n_lanes, title_lines)
    box_h = LANE_H * n_lanes + 0.24
    ax = fig.add_axes([LANE_L, top_frac - (title_lines * TITLE_H + box_h) / H, 0.985 - LANE_L, box_h / H])
    return ax, top_frac - block / H


def _wrap_mono(lines, width=MONO_COLS):
    """Wrap the monospace block to a fixed column count, keeping the indent, so the page never widens."""
    out = []
    for l in lines:
        if len(l) <= width: out.append(l); continue
        pad = " " * (len(l) - len(l.lstrip()) + 4)
        out.extend(textwrap.wrap(l, width, subsequent_indent=pad, break_long_words=False, break_on_hyphens=False) or [l])
    return out


def _text_figure(draw, n_lanes, lines, path, title):
    """A timeline drawn by `draw(ax)` and a monospace block under it. Both are sized from their content and placed
    directly, so there is no dead space between them and a long line wraps instead of stretching the page."""
    lines = _wrap_mono(lines)
    tl = title.count("\n") + 1
    h_text = MONO_H * len(lines)
    H = _block_h(n_lanes, tl) + 0.10 + h_text + 0.14
    fig = plt.figure(figsize=(PAGE_W, H))
    ax, top = _timeline_axes(fig, 1.0 - 0.06 / H, n_lanes, H, tl); draw(ax)
    ax.set_title(title, loc="left", fontsize=8, fontweight="normal")
    fig.text(LANE_L, top - 0.04 / H, "\n".join(lines), va="top", ha="left", fontsize=6.8, family="monospace")
    _save(fig, path, tight=False)


def _draw_case(ax, view, tau, show_oracle=False):
    """One video. True sentences in the middle lane; systems alternate above and below it, so every link runs straight
    to the caption units without crossing another system. Colour = the DVC pair type (green: scored against a true
    sentence; orange: against a random string). A link joins a prediction to every caption unit it covers, with the
    tIoU at its midpoint: solid = the one-to-one match, dotted = an extra DVC pair, red = below the threshold."""
    gold = view["gold"]; t0, t1 = view["window"]; systems = list(view["systems"])
    above, below = systems[0::2], systems[1::2]
    y_g = len(below) + 1
    y = {**{s: y_g + 1 + i for i, s in enumerate(above)}, **{s: y_g - 1 - i for i, s in enumerate(below)}}
    span = t1 - t0
    g_in = [gi for gi, (ga, gb, _) in enumerate(gold) if gb > t0 and ga < t1]
    for gi in g_in:
        ga, gb, _ = gold[gi]
        ax.add_patch(Rectangle((max(ga, t0), y_g - 0.3), min(gb, t1) - max(ga, t0), 0.6, fc="#bdd7ee", ec="#1f4e79", lw=0.8))
        lab = f"S{gi+1}" + (f"  {gb - ga:.1f} s" if (min(gb, t1) - max(ga, t0)) > 0.06 * span else "")
        ax.text((max(ga, t0) + min(gb, t1)) / 2, y_g, lab, ha="center", va="center", fontsize=7)
    g_lab = f"caption units S{g_in[0]+1}-S{g_in[-1]+1} of {len(gold)}" if g_in else "caption units"
    ax.text(t0 - 0.008 * span, y_g, g_lab + (f"\n(perfect cuts: video BLEU-4 {view['oracle']['video_bleu']:.1f})" if show_oracle and view.get("oracle") else ""), ha="right", va="center", fontsize=7.5)
    for s in systems:
        d = view["systems"][s]; up = y[s] > y_g; sign = -1 if up else 1     # links leave the bar on the side facing the caption units
        by_p = {}
        for r in d["pairs"]: by_p.setdefault(r["p"], []).append(r)
        narrow_row = 0
        for pi, (a, b, _) in sorted(d["events"].items()):
            rows = by_p.get(pi, []); real = [r for r in rows if r["g"] is not None]
            xa, xb = max(a, t0), min(b, t1)
            ax.add_patch(Rectangle((xa, y[s] - 0.3), xb - xa, 0.6, fc=PRED_OK if real else PRED_BAD, ec="#333", lw=0.7))
            best = max((iou for _, iou, *_ in d.get("covers", {}).get(pi, [])), default=0.0)
            words = " + ".join(f"{r['counts']['counts'][0]}/{r['counts']['totals'][0]}" for r in rows)
            if real:   # every bar says what it was scored against and why, so the reader need not follow a link
                lab = f"P{pi+1}  pairs " + " + ".join(f"S{r['g']+1} at {r['tiou']:.2f}" for r in real) + f"  |  {words} words match"
                short = f"P{pi+1} {words}"
            else:
                why = f"best overlap {best:.2f}, under {tau:g}" if best > 0 else "no overlap"
                lab = f"P{pi+1}  {why}  |  no pair, so a random string  |  {words} words"
                short = f"P{pi+1} {best:.2f}<{tau:g}, no pair" if best > 0 else f"P{pi+1} no overlap"
            if (xb - xa) > 0.22 * span: ax.text((xa + xb) / 2, y[s], lab, ha="center", va="center", fontsize=6.6)
            else:  # short bar: a short label on the side AWAY from the links, on alternating rows
                ax.text((xa + xb) / 2, y[s] - sign * (0.33 + 0.14 * narrow_row), short, ha="center", va="bottom" if up else "top", fontsize=5.8)
                narrow_row = (narrow_row + 1) % 2
            paired = {r["g"] for r in real}
            links = [(gi, iou) for gi, iou, *_ in d.get("covers", {}).get(pi, []) if gold[gi][1] > t0 and gold[gi][0] < t1]
            for k, (gi, iou) in enumerate(links):
                ga, gb = gold[gi][0], gold[gi][1]
                one = d["located"].get(pi, (None, None))[0] == gi
                xm, xg = (xa + xb) / 2, (max(ga, t0) + min(gb, t1)) / 2
                ya, yb = y[s] + sign * 0.3, y_g - sign * 0.3
                col = "#333" if one else ("#777" if gi in paired else "#c0392b")
                ax.plot([xm, xg], [ya, yb], color=col, lw=1.1 if one else 0.8, ls="-" if one else ":")
                f = (0.5, 0.34, 0.66, 0.24, 0.76)[k % 5]      # stagger the labels of one event's links
                ax.text(xm + (xg - xm) * f, ya + (yb - ya) * f, f"{iou:.2f}", fontsize=7, ha="center", va="center", color=col,
                        fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=0.6))
        for da, db in d.get("dropped", []):   # events excluded by annotation ignore regions: they explain the gaps, and carry no number
            ax.add_patch(Rectangle((max(da, t0), y[s] - 0.22), min(db, t1) - max(da, t0), 0.44, fc="none", ec="#999", ls=":", lw=0.9, hatch="///"))
        shown = sorted(d["events"]); rng = f"P{shown[0]+1}-P{shown[-1]+1} of {d['n_events']} scored" if shown else "no scored events"
        drop = f", {len(d['dropped'])} dropped" if d.get("dropped") else ""
        ax.text(t0 - 0.008 * span, y[s], f"{s}\n({rng}{drop}; video BLEU-4 {d['video_bleu']:.1f})", ha="right", va="center", fontsize=7.5)
    ys = [y_g, *y.values()]
    ax.set_xlim(t0, t1); ax.set_ylim(min(ys) - 0.68, max(ys) + 0.68); ax.set_yticks([])
    ax.spines["left"].set_visible(False); ax.set_xlabel("time in the video (seconds)")


def _case_text(view):
    """The captions of a case: every caption unit in the window, then every event of every system. Pairing, tIoU and
    the n-gram counts are in the figure above and in the protocol section, so this panel carries the text only."""
    t0, t1 = view["window"]
    lines = [f"S{i+1} [{a:.1f}-{b:.1f} s]  {t}" for i, (a, b, t) in enumerate(view["gold"]) if b > t0 and a < t1]
    for s, d in view["systems"].items():
        lines += ["", f"{s}  ·  events P{min(d['events'])+1}-P{max(d['events'])+1} of {d['n_events']} in this window  ·  video BLEU-4 {d['video_bleu']:.1f} from all of its pairs in the whole video" if d["events"] else f"{s}  ·  no events in this window"]
        shown = sorted(d["events"].items())
        lines += [f"  P{pi+1} [{a:.1f}-{b:.1f} s]  {t}" for pi, (a, b, t) in shown[:14]]
        if len(shown) > 14: lines.append(f"  ... {len(shown) - 14} more events in this window")
    return lines


def case_figure(view, tau, path, note=None, language=""):
    lines = _case_text(view)
    title = textwrap.fill(f"{language} {view['video_id']}" + (f":  {note}" if note else ""), 150) + "\n" + case_legend(tau)
    _text_figure(lambda ax: _draw_case(ax, view, tau), 1 + len(view["systems"]), lines, path, title)


def case_thresholds(views, thresholds, path, language="", note=""):
    """The same video's segments at every tIoU threshold: which overlaps become pairs and which fall to a random string."""
    lanes = 1 + len(next(iter(views.values()))["systems"]); vid = next(iter(views.values()))["video_id"]
    fig, axes = plt.subplots(len(thresholds), 1, figsize=(13.5, (1.0 + 0.72 * lanes) * len(thresholds)), squeeze=False)
    for ax, t in zip(axes[:, 0], thresholds):
        v = views[t]; _draw_case(ax, v, t)
        real = sum(1 for d in v["systems"].values() for r in d["pairs"] if r["g"] is not None)
        garb = sum(1 for d in v["systems"].values() for r in d["pairs"] if r["g"] is None)
        ax.set_title(f"tIoU {t:g}:  {real} pairs against a caption unit,  {garb} against a random string", loc="left", fontsize=9.5)
    fig.suptitle(f"{language} {vid}: the same segments at every tIoU threshold" + (f"  ({note})" if note else ""), fontsize=12, weight="bold")
    fig.text(0.01, 0.002, case_legend(thresholds[0]).split("\n")[0], fontsize=6.6, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.015, 1, 0.985)); _save(fig, path)


# ---------------------------------------------------------------- the BLEU protocol
def protocol_two_ways(proto, path, language):
    """Perfect cuts: the same captions as one corpus and as a mean over videos. Bar width = number of sentences."""
    o = proto["oracle"]
    vids = sorted(o["per_video"], key=lambda v: o["sizes"][v]); sizes = np.array([o["sizes"][v] for v in vids], float); vals = np.array([o["per_video"][v] for v in vids])
    fig, ax = plt.subplots(figsize=(11, 4.4))
    left = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    ax.bar(left, vals, width=sizes, align="edge", color="#3b6ea5", ec="white", lw=0.5)
    ax.axhline(o["corpus_bleu4"], color="#222", lw=1.6, label=f"pretrimmed: all {int(sizes.sum())} sentences as ONE corpus = {o['corpus_bleu4']:.2f}")
    ax.axhline(o["per_video_mean"], color="#c0392b", lw=1.6, ls="--", label=f"DVC: plain mean of the {len(vids)} video scores = {o['per_video_mean']:.2f}")
    ax.axhline(o["size_weighted_mean"], color="#e59866", lw=1.2, ls=":", label=f"mean of the video scores weighted by their sentences = {o['size_weighted_mean']:.2f}")
    ax.set_xlabel(f"videos sorted by size; the width of each bar is its number of sentences (total {int(sizes.sum())})")
    ax.set_ylabel("per-video corpus BLEU-4"); ax.set_ylim(0, 112); ax.set_xlim(-5, sizes.sum() + 5)
    ax.set_title(f"{language}: perfect cuts, pretrimmed translator. The same captions, two ways to add up BLEU-4")
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    order = np.argsort(-vals)
    for rank, j in enumerate(order[:2]):
        ax.annotate(f"{int(sizes[j])} sentences, BLEU {vals[j]:.0f}", (left[j] + sizes[j] / 2, vals[j]), textcoords="offset points", xytext=(40, -6 - 14 * rank), fontsize=8, arrowprops=dict(arrowstyle="-", color="#555"))
    k = int(np.argmax(sizes)); ax.annotate(f"{int(sizes[k])} sentences, BLEU {vals[k]:.1f}", (left[k] + sizes[k] / 2, vals[k]), textcoords="offset points", xytext=(-40, 30), fontsize=8, arrowprops=dict(arrowstyle="-", color="#555"))
    fig.tight_layout(); _save(fig, path)


def protocol_per_threshold(proto, systems, thresholds, path, language, true_cuts=None):
    """For each system: the reported DVC mean, the same pairs as one corpus, and the two perfect-cut references."""
    o = proto["oracle"]; n = len(systems)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False, sharey=True)
    for ax, s in zip(axes[0], systems):
        rows = proto["systems"][s]; x = np.arange(len(thresholds)); w = 0.27
        ax.bar(x - w, [rows[t]["located_corpus_bleu4"] for t in thresholds], w, color="#7fb3d5", label="located pairs only, one corpus")
        ax.bar(x, [rows[t]["all_pairs_corpus_bleu4"] for t in thresholds], w, color="#5b5b5b", label="all DVC pairs (incl. garbage), one corpus")
        ax.bar(x + w, [rows[t]["dvc_mean"] for t in thresholds], w, color="#c0392b", label="reported DVC: mean over videos")
        for j, t in enumerate(thresholds):
            for dx, v in ((-w, rows[t]["located_corpus_bleu4"]), (0, rows[t]["all_pairs_corpus_bleu4"]), (w, rows[t]["dvc_mean"])): ax.text(j + dx, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        if true_cuts: ax.axhline(true_cuts["dvc"][thresholds[0]], color="#c0392b", lw=1.6, ls="--",
                                 label=f"perfect cuts, scored as a DVC row = {true_cuts['dvc'][thresholds[0]]:.2f}  (the same-protocol reference)")
        if o: ax.axhline(o["corpus_bleu4"], color="#222", lw=1.4, label=f"perfect cuts, one corpus = {o['corpus_bleu4']:.2f}  (a pooled corpus, not comparable)")
        ax.set_xticks(x); ax.set_xticklabels([f"tIoU {t:g}" for t in thresholds]); ax.set_title(s, fontsize=10); ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("BLEU-4")
    for ax in axes[0]: ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    h, lab = axes[0][0].get_legend_handles_labels(); fig.legend(h, lab, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.suptitle(f"{language}: the same predicted segments under the two ways of adding up BLEU-4", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 1)); _save(fig, path)


def protocol_worked_example(view, proto, tau, path, language=""):
    """One small real video: the segments, then the n-gram counts every pair contributes, added up the pretrimmed way
    (into the corpus sum) and the DVC way (into this video's own BLEU-4, which then counts once in the mean)."""
    o = proto["oracle"]; systems = list(view["systems"])
    def sums(pairs): return {"counts": [sum(r["counts"]["counts"][n] for r in pairs) for n in range(4)], "totals": [sum(r["counts"]["totals"][n] for r in pairs) for n in range(4)]}
    lines = ["PRETRIMMED (perfect cuts): one pair per caption unit. The matched n-grams of every pair go into ONE sum over all videos.", NGRAM_HEADER]
    if view.get("oracle"):
        for r in view["oracle"]["pairs"]:
            lines.append(f"  S{r['g']+1}  ref: {r['ref']}"); lines.append(f"      hyp: {r['hyp']}")
            lines.append(_pair_row("        this pair adds", None, r["counts"]))
        lines.append(_pair_row("  this video adds to the corpus sum", None, sums(view["oracle"]["pairs"])) + f"   (alone: BLEU-4 {view['oracle']['video_bleu']:.1f}, a number that is never used)")
        lines.append(f"  all {o['n']} sentences of all videos added up  ->  corpus BLEU-4 {o['corpus_bleu4']:.2f}")
    for s_ in systems:
        d = view["systems"][s_]
        lines += ["", f"DVC at tIoU {tau:g}, {s_}: one pair per (prediction, caption unit) with enough overlap; a prediction that overlaps nothing gets a random string.", NGRAM_HEADER]
        for r in d["pairs"]:
            lines.append(f"  P{r['p']+1}  hyp: {r['hyp']}")
            lines.append(_pair_row(f"        paired with S{r['g']+1}" if r["g"] is not None else "        paired with a random string", r["tiou"], r["counts"]))
        lines.append(_pair_row("  this video's own sum", None, sums(d["pairs"])) + f"   ->  video BLEU-4 {d['video_bleu']:.1f}, ONE of {len(o['per_video'])} videos in the mean")
        lines.append(f"  mean over videos  ->  reported DVC BLEU-4 {proto['systems'][s_][tau]['dvc_mean']:.2f}      the same pairs added up as one corpus  ->  {proto['systems'][s_][tau]['all_pairs_corpus_bleu4']:.2f}")
    title = f"{language} {view['video_id']}: {len(view['gold'])} caption units. The same segments added up the pretrimmed way and the DVC way\n" + case_legend(tau)
    _text_figure(lambda ax: _draw_case(ax, view, tau, show_oracle=True), 1 + len(systems), lines, path, title)


def _weight_bars(ax, w, small, big):
    """Two bars over the same videos: the share each video has in the DVC mean (always 1/N) and in the corpus
    (its share of the words). The two example videos are highlighted so the swap is visible."""
    vids = sorted(w["sizes"], key=lambda v: w["sizes"][v])
    n, total_words = w["n_videos"], max(1, w["total_words"])
    rows = [("DVC: mean of the %d video scores" % n, [1.0 / n] * len(vids), f"every video weighs 1/{n} = {100/n:.1f} %, whatever its size"),
            ("pretrimmed: one corpus of %s words" % f"{total_words:,}", [w["words"].get(v, 0) / total_words for v in vids], "a video weighs its share of the words")]
    for ri, (name, widths, note) in enumerate(rows):
        y = 1 - ri; left = 0.0
        for v, wd in zip(vids, widths):
            fc = SMALL_C if v == small else BIG_C if v == big else REST_C
            ax.add_patch(Rectangle((left, y + 0.12), wd, 0.42, fc=fc, ec="white", lw=0.35))
            if v in (small, big):
                x = left + wd / 2
                ax.annotate(f"{v}\n{100*wd:.1f} %", (x, y + 0.54), xytext=(x if ri else x, y + 0.78), fontsize=6.8, ha="center", va="bottom",
                            arrowprops=dict(arrowstyle="-", color=fc, lw=0.8), color=fc)
            left += wd
        ax.text(-0.006, y + 0.33, name, ha="right", va="center", fontsize=8)
        ax.text(1.004, y + 0.33, note, ha="left", va="center", fontsize=6.8, color="#555")
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 2.05); ax.axis("off")


def protocol_two_videos(ex, views, proto, path, language="", true_cuts=None):
    """Two real videos, then the arithmetic of both numbers: the mean over videos and the one corpus."""
    o = proto["oracle"]; small, big, sysname, tau = ex["small"], ex["big"], ex["system"], ex["tau"]
    w = ex["weights"]; n = w["n_videos"]
    def counts(rows): return " ".join(f"{sum(r['counts'][i] for r in rows)}/{sum(r['totals'][i] for r in rows)}" for i in range(4))
    c_small, c_big = w["contribution"][small], w["contribution"][big]
    share = lambda c: f"{100 * c / w['mean']:4.1f} %" if w["mean"] else " n/a"   # a mean of 0: every video scored 0
    L = ["PERFECT CUTS (pretrimmed).  One pair per caption unit.    matched/total n-grams for n = 1..4",
         f"  {small:14s} {ex['oracle'][small]['n']:4d} sentences   {counts(ex['oracle'][small]['counts']):34s}  its own BLEU-4 {ex['oracle'][small]['video_bleu']:6.1f}",
         f"  {big:14s} {ex['oracle'][big]['n']:4d} sentences   {counts(ex['oracle'][big]['counts']):34s}  its own BLEU-4 {ex['oracle'][big]['video_bleu']:6.1f}",
         f"     DVC way, these two videos:  mean of the two video scores = ({ex['oracle'][small]['video_bleu']:.1f} + {ex['oracle'][big]['video_bleu']:.1f}) / 2 = {ex['oracle']['mean_of_two']:.1f}",
         f"     pretrimmed way, same pairs: add both videos' counts = {counts(ex['oracle'][small]['counts'] + ex['oracle'][big]['counts'])}  ->  BLEU-4 {ex['oracle']['pooled_two']:.1f}",
         f"     all {n} videos:            DVC way {w['mean']:.2f}       pretrimmed way {w['corpus']:.2f}", "",
         f"FROM TWO VIDEOS TO ALL {n}.  The same two steps, with what each video is worth in the result.",
         f"  DVC way: score each video on its own, then divide the sum of the {n} video scores by {n}.",
         f"     {small:14s} {w['sizes'][small]:4d} sentences   BLEU-4 {w['per_video'][small]:6.1f}   ->  {w['per_video'][small]:6.1f} / {n} = {c_small:5.2f}   ({share(c_small)} of the final number)",
         f"     {big:14s} {w['sizes'][big]:4d} sentences   BLEU-4 {w['per_video'][big]:6.1f}   ->  {w['per_video'][big]:6.1f} / {n} = {c_big:5.2f}   ({share(c_big)} of the final number)",
         f"     the other {n-2} videos                                ->  {w['mean'] - c_small - c_big:5.2f}",
         f"     sum of the {n} video scores {w['score_sum']:.1f}  /  {n}  =  {w['mean']:.2f}",
         f"  Pretrimmed way: a video's weight is its share of the words, not 1/{n}.",
         f"     {small:14s} {w['words'][small]:6,d} of {w['total_words']:,} words ({100*w['word_share'][small]:4.1f} %)",
         f"     {big:14s} {w['words'][big]:6,d} of {w['total_words']:,} words ({100*w['word_share'][big]:4.1f} %)",
         f"     all {n} videos   {w['total_words']:,} words  ->  BLEU-4 {w['corpus']:.2f}",
         f"  Same captions, two answers: {w['mean']:.2f} and {w['corpus']:.2f}. In the mean a {w['sizes'][small]}-sentence video weighs as much as a {w['sizes'][big]}-sentence one.", ""]
    if true_cuts:
        L += ["WHICH NUMBER A DVC COLUMN SHOULD BE COMPARED WITH",
              "  A system's DVC column is a mean over videos, so its reference must be a mean over videos too.",
              f"  Perfect cuts scored by the code of the rows, with the same span filter every row gets ({true_cuts['n_kept']} of {true_cuts['n_gold']} spans reach the scorer):",
              f"     DVC BLEU-4 {true_cuts['dvc'][tau]:.2f} at every threshold (the spans are the true ones, so the threshold never changes a pair)   F1 {true_cuts['f1'][tau]:.3f}",
              f"  That is the GT-boundary reference for a DVC column. The pooled corpus {w['corpus']:.2f} does not compare with a DVC column.", ""]
    L += [f"{sysname}, DVC pairs at tIoU {tau:g}.  An event pairs only when the tIoU reaches {tau:g}; otherwise its caption is scored against a random string.",
          f"  {small:14s} {ex['dvc'][small]['n']:4d} pairs ({ex['dvc'][small]['n_garbage']:3d} random)   {counts(ex['dvc'][small]['counts']):34s}  its own BLEU-4 {ex['dvc'][small]['video_bleu']:6.1f}",
          f"  {big:14s} {ex['dvc'][big]['n']:4d} pairs ({ex['dvc'][big]['n_garbage']:3d} random)   {counts(ex['dvc'][big]['counts']):34s}  its own BLEU-4 {ex['dvc'][big]['video_bleu']:6.1f}",
          f"     DVC way, these two videos:  ({ex['dvc'][small]['video_bleu']:.1f} + {ex['dvc'][big]['video_bleu']:.1f}) / 2 = {ex['dvc']['mean_of_two']:.1f}",
          f"     same pairs as one corpus:   {counts(ex['dvc'][small]['counts'] + ex['dvc'][big]['counts'])}  ->  BLEU-4 {ex['dvc']['pooled_two']:.1f}",
          f"     all {n} videos:            reported DVC {proto['systems'][sysname][tau]['dvc_mean']:.2f}       same pairs as one corpus {proto['systems'][sysname][tau]['all_pairs_corpus_bleu4']:.2f}       perfect cuts as one corpus {w['corpus']:.2f}"]
    L = _wrap_mono(L)
    legend = case_legend(tau)
    h_head, h_w, h_text, h_leg = 0.36, 1.30, 0.122 * len(L), 0.125 * (legend.count("\n") + 1)   # 0.122 in per line at 7.2 pt
    lanes = {vid: 1 + len(views[vid]["systems"]) for vid in (small, big)}
    H = h_head + sum(_block_h(lanes[v]) + 0.10 for v in (small, big)) + 0.30 + h_w + 0.16 + h_text + 0.34 + h_leg + 0.14
    fig = plt.figure(figsize=(PAGE_W, H))
    fig.suptitle(f"{language}: two real videos, and the arithmetic behind the two BLEU-4 numbers", fontsize=12, weight="bold", y=1.0 - 0.12 / H)
    top = 1.0 - h_head / H
    for vid in (small, big):
        ax, top = _timeline_axes(fig, top, lanes[vid], H); _draw_case(ax, views[vid], tau, show_oracle=True)
        w0, w1 = views[vid]["window"]
        ax.set_title(f"video {vid}: {o['sizes'][vid]} caption units" + ("" if vid == small else f"   (only {w0:.0f}-{w1:.0f} s shown)"), loc="left", fontsize=9.5)
        top -= 0.10 / H
    axw = fig.add_axes([LANE_L, top - (0.30 + h_w) / H, 0.80 - LANE_L, h_w / H])
    axw.set_title(f"what one video is worth in each number ({n} videos, sorted by size)", loc="left", fontsize=9)
    _weight_bars(axw, w, small, big)
    top -= (0.30 + h_w + 0.16) / H
    fig.text(LANE_L, top, "\n".join(L), va="top", ha="left", fontsize=7.2, family="monospace")
    fig.text(LANE_L, top - (h_text + 0.34) / H, legend, va="top", ha="left", fontsize=6.6)
    _save(fig, path, tight=False)


# ---------------------------------------------------------------- optional panels
def confidence_calibration(curves, path):
    """curves[label] = list of (mean confidence, sentence BLEU) points."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5)); edges = np.linspace(0, 1, 11)
    for label, pts in curves.items():
        xs, ys, ns = [], [], []
        for lo, hi in zip(edges, edges[1:]):
            sel = [b for c, b in pts if lo <= c < (hi if hi < 1 else 1.01)]
            if sel: xs.append((lo + hi) / 2); ys.append(float(np.mean(sel))); ns.append(len(sel))
        ax.plot(xs, ys, marker="o", label=f"{label} (n={sum(ns)})")
    ax.set_xlabel("mean token confidence"); ax.set_ylabel("mean sentence BLEU-4"); ax.set_title("Does the model's confidence track quality?"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, path)


def stability(policies, path):
    """policies[label] = {policy name: row} from a stability JSON."""
    fig, ax = plt.subplots(figsize=(7.5, 5)); markers = ["o", "s", "D", "^", "v", "P"]
    for si, (label, pol) in enumerate(policies.items()):
        for pi, (name, r) in enumerate(sorted(pol.items())):
            if not (np.isfinite(r["first_token_latency_s"]) and np.isfinite(r["frozen_prefix_error"])): continue   # NaN: no committed sentence
            ax.scatter(r["first_token_latency_s"], r["frozen_prefix_error"], s=40 + 260 * np.nan_to_num(float(r.get("revealed_fraction", 0.0))), marker=markers[si % len(markers)], alpha=0.75, color=PALETTE[pi % len(PALETTE)])
            ax.annotate(f"{name}\n({label})", (r["first_token_latency_s"], r["frozen_prefix_error"]), fontsize=6, textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("first-token latency vs commit (s; negative = earlier)"); ax.set_ylabel("frozen-prefix error"); ax.set_title("Reveal policies: latency saved vs premature-freeze cost (marker area = revealed fraction)")
    fig.tight_layout(); _save(fig, path)


# ---------------------------------------------------------------- progress report (one page per idea)
FAMILY_C = {"Perfect boundaries": "#7f8c8d", "Offline cascade": "#7fb3d5", "Online cascade": "#e59866", "Our joint model": "#c0392b"}


def task_and_experiments(path, gold_example, language, n_videos, n_sent, lang_name=None):
    """Two panels: what the task is, on a real video; and what changes between the experiments."""
    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.5], hspace=0.42)

    ax = fig.add_subplot(gs[0])
    spans, vid = gold_example
    t0, t1 = min(a for a, _, _ in spans) - 1, max(b for _, b, _ in spans) + 1
    ax.add_patch(Rectangle((t0, 0.62), t1 - t0, 0.26, fc="#f0f0f0", ec="#bbb"))
    ax.text(t0 + 0.4, 0.75, "what the model sees: one pose stream, no marks", fontsize=8.5, va="center", style="italic", color="#555")
    for i, (a, b, _) in enumerate(spans):
        ax.add_patch(Rectangle((a, 0.18), b - a, 0.26, fc="#bdd7ee", ec="#1f4e79"))
        ax.text((a + b) / 2, 0.31, f"S{i+1}", ha="center", va="center", fontsize=8)
    ax.text(t0 - 0.01 * (t1 - t0), 0.31, "caption units\n(what we must find)", ha="right", va="center", fontsize=8.5)
    ax.text(t0 - 0.01 * (t1 - t0), 0.75, "the video", ha="right", va="center", fontsize=8.5)
    ax.set_xlim(t0, t1); ax.set_ylim(0.05, 1.02); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("time in the video (seconds)")
    ax.set_title(f"1. The task: cut the stream into sentences, and translate each one.    {lang_name or language} test set: {n_videos}, {n_sent}",
                 loc="left", fontsize=11)

    ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    def box(x, y, w, h, text, fc, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", fc=fc, ec="#444", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    def arrow(x1, y1, x2, y2, c="#666", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=c, lw=1.2, linestyle=ls, shrinkA=2, shrinkB=2))
    ax.text(0, 96, "2. What is the same in every experiment, and what changes", fontsize=11, weight="bold")
    ax.add_patch(Rectangle((0, 10), 30, 78, fc="#f7f9fc", ec="#c9d6e4", lw=1.0, zorder=0))
    ax.text(15, 84, "THE SAME EVERYWHERE", ha="center", fontsize=9, weight="bold", color="#37536b")
    box(3, 64, 24, 12, "pose keypoints\n(one converter)", "#eef2f7")
    box(3, 48, 24, 12, "pretrimmed translator\ntrained on perfect cuts", "#eef2f7")
    box(3, 33, 24, 11, "test videos, caption units\nand the score", "#eef2f7")
    ax.text(15, 19, "so any difference below comes\nfrom the boundaries, not from\nthe data or the score", ha="center", fontsize=8, style="italic", color="#555")

    ax.text(65, 84, "WHAT CHANGES", ha="center", fontsize=9, weight="bold", color="#37536b")
    box(36, 66, 27, 12, "who cuts the video\nEMNLP segmenter  or  S1", "#dfe9f5")
    box(36, 50, 27, 12, "how the cuts are chosen\nargmax, viterbi  or  duration", "#dfe9f5")
    box(36, 34, 27, 12, "when it must decide\noffline  or  online", "#dfe9f5")
    box(36, 18, 27, 12, "who translates\npretrimmed  or  our joint model", "#fce8e6")
    for y in (72, 56, 40, 24): arrow(63, y, 70, y)
    box(70, 66, 28, 12, "OFFLINE CASCADE\ncut first, then translate", "#d6e9f8", 8.5)
    box(70, 50, 28, 12, "ONLINE CASCADE\ncut and translate while watching", "#fdf0e2", 8.5)
    box(70, 34, 28, 12, "OUR JOINT MODEL\nit cuts and translates itself", "#fbdedb", 8.5)
    box(70, 18, 28, 12, "PERFECT BOUNDARIES\nthe true cuts are given", "#eeeeee", 8.5)
    ax.text(84, 12, "reference: the same translator with GT cuts", ha="center", fontsize=8, style="italic", color="#555")
    _save(fig, path)


def results_overview(rows, thresholds, path, language, ceiling=None):
    """One bar group per experiment: the densevid scores (sentence tIoU F1 and DVC BLEU-4) at each threshold, GT-boundary reference
    drawn as a line. `report.py progress` prints the main scores (the F1 and the DVC BLEU-4 of the tIoU match, no edge limit, avg) per
    experiment."""
    labels = [r["label"] for r in rows]; fams = [r["family"] for r in rows]
    x = np.arange(len(rows)); w = 0.8 / len(thresholds)
    fig, axes = plt.subplots(2, 1, figsize=(max(10, 1.5 * len(rows)), 8.4), sharex=True)
    for ax, key, name, fmt in ((axes[0], "f1", "sentence tIoU F1", "{:.2f}"), (axes[1], "bleu", "tIoU DVC BLEU-4", "{:.1f}")):
        shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(thresholds)))
        for j, t in enumerate(thresholds):
            v = [r[key][t] for r in rows]
            ax.bar(x + (j - (len(thresholds) - 1) / 2) * w, v, w, color=shades[j], label=f"tIoU {t:g}")
            for i, y in enumerate(v): ax.text(i + (j - (len(thresholds) - 1) / 2) * w, y, fmt.format(y), ha="center", va="bottom", fontsize=6.5)
        if ceiling and key in ceiling:
            note = " (by definition)" if key == "f1" and ceiling[key] > 0.99 else ""
            ax.axhline(ceiling[key], color="#c0392b", lw=1.5, ls="--", label=f"perfect boundaries = {fmt.format(ceiling[key])}{note}")
        ax.set_ylabel(name); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
    axes[0].legend(fontsize=8, ncol=4, loc="upper right")
    # family bands under the x axis
    axes[1].set_xticks(x); axes[1].set_xticklabels([_wrap(l, 18) for l in labels], fontsize=7.5)
    prev, start = None, 0
    for i, f in enumerate(fams + [None]):
        if f != prev and prev is not None:
            axes[1].annotate(prev, ((start + i - 1) / 2, -0.30), xycoords=("data", "axes fraction"), ha="center", fontsize=8.5,
                             weight="bold", color=FAMILY_C.get(prev, "#333"), annotation_clip=False)
            start = i
        prev = f if f is not None else prev
    fig.suptitle(f"{language}: every experiment, same test set and same score (tIoU scores; the main scores are printed)",
                 fontsize=12.5, weight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.98)); _save(fig, path)


def failure_overview(errors, labels, path, language, n_units, counts=None):
    """What happens to each caption unit in each experiment: the unit shares of the error types of the counted pairs of the main
    score (metrics.edge_error_types `units_avg`), the mean over the cells of the grid."""
    fig, ax = plt.subplots(figsize=(max(9, 1.35 * len(labels)), 5.0))
    _type_bars(ax, {l: errors[l]["units_avg"] for l in labels}, labels, EDGE_UNIT_TYPES, 6)
    if counts:
        for i, l in enumerate(labels): ax.text(i, 101.5, f"{counts[l]:.2f}x", ha="center", fontsize=7.5, color="#555")
        ax.text(len(labels) - 0.5, 106, "captions produced / caption units", ha="right", fontsize=7.5, color="#555")
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels([_wrap(l, 18) for l in labels], fontsize=7.5)
    ax.set_ylabel(f"share of the {n_units} caption units (%)"); ax.set_ylim(0, 112)
    ax.legend(fontsize=8.5, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False)
    ax.set_title(f"{language}: what happens to each caption unit (the counted pairs of the edge-limit grid, mean over the cells)", fontsize=11)
    fig.tight_layout(); _save(fig, path)
