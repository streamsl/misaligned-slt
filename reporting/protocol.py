"""Why a DVC BLEU-4 column and a corpus BLEU-4 differ on the same captions, shown on real videos.

Both are corpus BLEU-4. The RQ1 number adds the n-gram counts of every pair into one sum (metrics.compute_text_metrics);
DVC scores each video from its own pairs (metrics.densevid_text_metrics, per_video=True) and then takes the plain mean over videos.
This module recomputes both with the shared scorers and returns per-pair views (tIoU, matched n-grams) for the figures.
"""
from __future__ import annotations
import random
import numpy as np

from eval import densevid_pairs
from reporting.outcomes import video_bootstrap_ci
from metrics import (_DENSEVID_GARBAGE_SEED, bleu_from_counts, bleu_pair_counts, compute_text_metrics, densevid_text_metrics,
                     garbage_reference, match_segments, temporal_iou)


def corpus_bleu4(hyps, refs, char_level: bool) -> float:
    return compute_text_metrics(list(hyps), list(refs), char_level=char_level)["translation_bleu4"] if hyps else 0.0


def oracle_pairs(gold: dict, rq1_rows: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """(hypothesis, reference) per video for the clean translator on gold-span crops, from an RQ1 (0,0) file."""
    key = lambda v, s, e: (v, round(float(s), 3), round(float(e), 3))
    by_gold = {key(r["video_id"], r["gt_start_s"], r["gt_end_s"]): r["prediction"]
               for r in rq1_rows if r["grid_head"] == 0.0 and r["grid_tail"] == 0.0}
    out = {}
    for vid, evs in gold.items():
        missing = [g for g in evs if key(vid, g.start_s, g.end_s) not in by_gold]
        if missing: raise ValueError(f"Clean RQ1 is missing {len(missing)} gold-caption predictions for {vid}; use a full zero-offset run")
        pairs = [(by_gold[key(vid, g.start_s, g.end_s)], g.text) for g in evs]
        if pairs: out[vid] = pairs
    return out


def oracle_row(gold: dict, oracle: dict, records, thresholds, char_level: bool) -> dict:
    """Perfect cuts as an RQ2 row: the gold spans are the predictions, so every event locates its own sentence, and the
    same annotation-ignore policy is applied. Correct short predictions remain scoreable. This is a DVC-comparable GT-boundary reference, not a bound on caption quality."""
    from eval import PredictionEvent, evaluate_predicted_events, scoreable_predictions
    events = {v: [PredictionEvent(video_id=v, start_s=g.start_s, end_s=g.end_s, text=h)
                  for g, (h, _) in zip(gold[v], oracle[v])] for v in oracle}
    kept = scoreable_predictions(events, records, tag="oracle row")
    res = evaluate_predicted_events(kept, gold, list(thresholds), char_level=char_level, densevid=True)
    rows = {float(r["tiou_threshold"]): r for r in res["thresholds"]}
    t0 = float(list(thresholds)[0])
    pv = densevid_text_metrics({v: densevid_pairs(kept.get(v, []), ge, t0) for v, ge in gold.items() if ge},
                               char_level=char_level, bleurt_checkpoint=None, per_video=True)
    return {"dvc": {t: r["text_metrics"]["densevid_bleu4"] for t, r in rows.items()},
            "ci": video_bootstrap_ci([r["bleu4"] for r in pv.values()]),
            "soda": {t: r["text_metrics"]["soda_bleu4"] for t, r in rows.items()},
            "f1": {t: r["segmentation"]["f1"] for t, r in rows.items()},
            "scored": {t: r["scored_pairs"] for t, r in rows.items()},
            "n_gold": sum(len(v) for v in gold.values()), "n_kept": sum(len(v) for v in kept.values())}


def located_pairs(pred, gold, tau):
    """One-to-one located pairs at `tau`: (video, gold index, hypothesis, reference)."""
    out = []
    for vid, ge in gold.items():
        pe = pred.get(vid, [])
        for pi, gi, _ in match_segments([e.segment for e in pe], [g.segment for g in ge], threshold=tau):
            if pe[pi].text is not None: out.append((vid, gi, pe[pi].text, ge[gi].text))
    return out


def decompose(gold: dict, systems: dict, thresholds, char_level: bool, oracle: dict | None = None) -> dict:
    """Per system and threshold: the reported DVC mean, the same pairs as one corpus, the located pairs as one corpus,
    and the oracle captions of the located sentences as one corpus. `oracle` maps video -> [(hyp, ref)]."""
    out = {"oracle": None, "systems": {}}
    if oracle:
        hyps = [h for p in oracle.values() for h, _ in p]; refs = [r for p in oracle.values() for _, r in p]
        per_video = {v: r["bleu4"] for v, r in densevid_text_metrics(oracle, char_level=char_level, bleurt_checkpoint=None, per_video=True).items()}
        sizes = {v: len(p) for v, p in oracle.items()}
        out["oracle"] = {"corpus_bleu4": corpus_bleu4(hyps, refs, char_level), "n": len(hyps), "per_video": per_video, "sizes": sizes,
                         "per_video_mean": float(np.mean(list(per_video.values()))),
                         "size_weighted_mean": float(np.average([per_video[v] for v in sizes], weights=[sizes[v] for v in sizes])),
                         "hyp_of": {(v, i): h for v, p in oracle.items() for i, (h, _) in enumerate(p)}}
    for label, pred in systems.items():
        rows = {}
        for tau in thresholds:
            pairs = {v: densevid_pairs(pred.get(v, []), ge, tau) for v, ge in gold.items() if ge}
            per_video = {v: r["bleu4"] for v, r in densevid_text_metrics(pairs, char_level=char_level, bleurt_checkpoint=None, per_video=True).items()}
            rng = random.Random(_DENSEVID_GARBAGE_SEED)
            all_h, all_r = [], []
            for v in sorted(pairs):
                for h, r in pairs[v]: all_h.append(h); all_r.append(r if r is not None else garbage_reference(rng))
            loc = located_pairs(pred, gold, tau)
            row = {"dvc_mean": float(np.mean(list(per_video.values()))) if per_video else 0.0, "per_video": per_video,
                   "dvc_ci": video_bootstrap_ci(per_video.values()),
                   "n_pairs": len(all_h), "n_garbage": sum(1 for p in pairs.values() for _, r in p if r is None), "n_located": len(loc),
                   "all_pairs_corpus_bleu4": corpus_bleu4(all_h, all_r, char_level),
                   "located_corpus_bleu4": corpus_bleu4([h for _, _, h, _ in loc], [r for _, _, _, r in loc], char_level)}
            if out["oracle"]:
                o = [(out["oracle"]["hyp_of"][(v, i)], r) for v, i, _, r in loc]
                row["oracle_same_gold_corpus_bleu4"] = corpus_bleu4([h for h, _ in o], [r for _, r in o], char_level)
            rows[float(tau)] = row
        out["systems"][label] = rows
    return out


def material_overlaps(e, ge, tau: float):
    """Every caption unit this event shares real time with, as (index, tIoU, share of the sentence, share of the event).
    Material means at least half of ONE side, or enough tIoU to pair: a merge covers the sentences, a fragment sits
    inside one. Judging only by the sentence's share hides every fragment, which is what a one-sided rule does."""
    out = []
    for gi, g in enumerate(ge):
        inter = min(e.end_s, g.end_s) - max(e.start_s, g.start_s)
        if inter <= 0: continue
        f_gold, f_pred = inter / max(1e-9, g.end_s - g.start_s), inter / max(1e-9, e.end_s - e.start_s)
        iou = temporal_iou(e.segment, g.segment)
        if f_gold >= 0.5 or f_pred >= 0.5 or iou >= tau: out.append((gi, iou, f_gold, f_pred))
    return out


def case_view(gold: dict, systems: dict, vid: str, tau: float, char_level: bool, window=None, oracle: dict | None = None,
              raw: dict | None = None) -> dict:
    """Everything a timeline needs for one video: gold spans; each system's scored events with their DVC pairs at `tau`,
    the tIoU and n-gram counts of each pair, the overlaps that were NOT paired, the events excluded by annotation ignore regions,
    and the video's DVC BLEU-4.

    NUMBERING. S<k> is the k-th caption unit of the whole video, in time order. P<k> is the k-th SCORED event of the
    same system in that video, in time order; events excluded by annotation ignore regions carry no number and are drawn grey.
    Both keep their whole-video number when only a window is shown."""
    ge = gold[vid]
    t0, t1 = window or (min(g.start_s for g in ge) - 2, max(g.end_s for g in ge) + 2)
    view = {"video_id": vid, "window": (t0, t1), "gold": [(g.start_s, g.end_s, g.text) for g in ge], "systems": {}}
    if oracle and vid in oracle:
        pairs = oracle[vid]; cnt = bleu_pair_counts([h for h, _ in pairs], [r for _, r in pairs], char_level)
        view["oracle"] = {"pairs": [{"g": i, "hyp": h, "ref": r, "counts": c} for i, ((h, r), c) in enumerate(zip(pairs, cnt))],
                          "video_bleu": bleu_from_counts(cnt)}
    for label, pred in systems.items():
        all_ev = pred.get(vid, [])
        idx = [i for i, e in enumerate(all_ev) if e.end_s > t0 and e.start_s < t1]   # whole-video event numbers inside the window
        rng = random.Random(_DENSEVID_GARBAGE_SEED)
        rows, covers, unpaired = [], {}, {}
        for pi in idx:
            e = all_ev[pi]
            if e.text is None: continue
            covers[pi] = material_overlaps(e, ge, tau)
            paired = {gi for gi, iou, *_ in covers[pi] if iou >= tau}
            over = [(gi, temporal_iou(e.segment, g.segment)) for gi, g in enumerate(ge) if temporal_iou(e.segment, g.segment) >= tau]
            for gi, iou in over: rows.append({"p": pi, "g": gi, "tiou": iou, "hyp": e.text, "ref": ge[gi].text})
            if not over: rows.append({"p": pi, "g": None, "tiou": None, "hyp": e.text, "ref": garbage_reference(rng)})
            unpaired[pi] = [(gi, iou, fg, fp) for gi, iou, fg, fp in covers[pi] if gi not in paired]
        cnt = bleu_pair_counts([r["hyp"] for r in rows], [r["ref"] for r in rows], char_level) if rows else []
        for r, c in zip(rows, cnt): r["counts"] = c
        one = {pi: (gi, iou) for pi, gi, iou in match_segments([e.segment for e in all_ev], [g.segment for g in ge], threshold=tau)}
        all_pairs = densevid_pairs(all_ev, ge, tau)
        kept = {(round(e.start_s, 4), round(e.end_s, 4)) for e in all_ev}
        dropped = [(e.start_s, e.end_s) for e in ((raw or {}).get(label, {}) or {}).get(vid, [])
                   if (round(e.start_s, 4), round(e.end_s, 4)) not in kept and e.end_s > t0 and e.start_s < t1]
        view["systems"][label] = {"events": {pi: (all_ev[pi].start_s, all_ev[pi].end_s, all_ev[pi].text) for pi in idx}, "n_events": len(all_ev),
                                  "pairs": rows, "located": one, "covers": covers, "unpaired": unpaired, "dropped": dropped,
                                  "video_bleu": densevid_text_metrics({vid: all_pairs}, char_level=char_level, bleurt_checkpoint=None, per_video=True)[vid]["bleu4"] if all_pairs else 0.0}
    return view


def two_video_example(gold: dict, systems: dict, oracle: dict, oracle_scores: dict, tau: float, char_level: bool) -> dict:
    """The smallest video with the top per-video score and the largest video, added up both ways with real counts:
    perfect cuts (pretrimmed) and the first system's DVC pairs at `tau`."""
    small = max([v for v in oracle_scores["per_video"] if oracle_scores["sizes"][v] <= 3] or list(oracle_scores["per_video"]), key=lambda v: oracle_scores["per_video"][v])
    big = max(oracle_scores["sizes"], key=oracle_scores["sizes"].get)
    label, pred = next(iter(systems.items()))
    out = {"small": small, "big": big, "system": label, "tau": tau, "oracle": {}, "dvc": {}}
    for vid in (small, big):
        pairs = oracle[vid]; cnt = bleu_pair_counts([h for h, _ in pairs], [r for _, r in pairs], char_level)
        out["oracle"][vid] = {"n": len(pairs), "counts": cnt, "video_bleu": bleu_from_counts(cnt)}
        dp = densevid_pairs(pred.get(vid, []), gold[vid], tau); rng = random.Random(_DENSEVID_GARBAGE_SEED)
        hyps = [h for h, _ in dp]; refs = [r if r is not None else garbage_reference(rng) for _, r in dp]
        cnt = bleu_pair_counts(hyps, refs, char_level) if dp else []
        out["dvc"][vid] = {"n": len(dp), "n_garbage": sum(1 for _, r in dp if r is None), "counts": cnt, "video_bleu": bleu_from_counts(cnt)}
    for key in ("oracle", "dvc"):
        a, b = out[key][small], out[key][big]
        out[key]["mean_of_two"] = (a["video_bleu"] + b["video_bleu"]) / 2
        out[key]["pooled_two"] = bleu_from_counts(a["counts"] + b["counts"])
    # How much of each final number one video supplies. In the mean every video weighs 1/N whatever its size; in the
    # corpus it weighs its share of the words, so the two example videos swap places.
    words = {v: sum(c["totals"][0] for c in bleu_pair_counts([h for h, _ in p], [r for _, r in p], char_level)) for v, p in oracle.items()}
    n = len(oracle_scores["per_video"])
    out["weights"] = {
        "n_videos": n, "total_words": sum(words.values()), "words": words, "per_video": oracle_scores["per_video"],
        "sizes": oracle_scores["sizes"], "mean": oracle_scores["per_video_mean"], "corpus": oracle_scores["corpus_bleu4"],
        "score_sum": sum(oracle_scores["per_video"].values()),
        "contribution": {v: oracle_scores["per_video"][v] / n for v in (small, big)},
        "word_share": {v: words.get(v, 0) / max(1, sum(words.values())) for v in (small, big)},
    }
    return out


def spelled_out(texts) -> float:
    """Share of reference tokens that are single letters; this does not establish visible motion or pauses."""
    words = [w.strip(".,!?-'\"") for t in texts for w in str(t or "").split()]
    words = [w for w in words if w]
    return sum(1 for w in words if len(w) == 1 and w.isalpha()) / max(1, len(words))


def showcase_videos(gold: dict, systems: dict, thresholds, tau: float, oracle_scores: dict | None = None) -> list[dict]:
    """Videos picked by rule to show the known effects: the tiny video with the top per-video score; the longest
    prediction under 200 s that covers at least three caption units (a merge that still fits one figure); the prediction
    with the most DVC pairs at the loosest threshold; the longest prediction with a looping caption; the longest prediction
    over a passage with many single-letter reference tokens; and a one-sentence video with an over-long prediction."""
    picks = []
    def add(vid, window, note, case_tau=tau):
        if vid and vid not in {p["video_id"] for p in picks}: picks.append({"video_id": vid, "window": window, "note": note, "tau": case_tau})
    def covered_spans(e, ge): return [g for g in ge if min(e.end_s, g.end_s) - max(e.start_s, g.start_s) >= 0.5 * (g.end_s - g.start_s)]
    def covered(e, ge): return len(covered_spans(e, ge))
    def why(e, ge):
        share = spelled_out([g.text for g in covered_spans(e, ge)])
        return (f" {100*share:.0f} % of the reference words here are single letters. "
                "The subtitle text alone does not establish whether boundary cues are visible.") if share >= 0.4 else ""
    if oracle_scores:
        small = [v for v in oracle_scores["per_video"] if oracle_scores["sizes"][v] <= 3]
        if small:
            v = max(small, key=lambda x: oracle_scores["per_video"][x]); big = max(oracle_scores["sizes"], key=oracle_scores["sizes"].get)
            add(v, None, f"{oracle_scores['sizes'][v]} caption units, per-video BLEU-4 {oracle_scores['per_video'][v]:.0f}. In the DVC mean this video counts as much as the {oracle_scores['sizes'][big]}-sentence video.")
    taken = lambda: {p["video_id"] for p in picks}
    lo = min(thresholds); longest, most = None, None
    for label, pred in systems.items():
        for vid, ge in gold.items():
            if vid in taken(): continue
            for e in pred.get(vid, []):
                k = covered(e, ge); dur = e.end_s - e.start_s
                if k >= 3 and dur <= 200 and (longest is None or dur > longest[0]): longest = (dur, vid, e, k, label)   # readable window
                m = sum(1 for g in ge if temporal_iou(e.segment, g.segment) >= lo)
                if m >= 2 and (most is None or m > most[0]): most = (m, vid, e, label)
    if longest:
        dur, vid, e, k, label = longest
        add(vid, (max(0, e.start_s - 5), e.end_s + 5), f"{label}: one prediction of {dur:.0f} s covers {k} caption units (a merge). At tIoU {tau:g} it pairs with none of them, so its caption is scored against a random string." + why(e, gold[vid]))
    if most:
        m, vid, e, label = most
        add(vid, (max(0, e.start_s - 5), e.end_s + 5), f"{label}: at tIoU {lo:g} one prediction pairs with {m} caption units; DVC scores its caption once per pair.", case_tau=lo)
    # a degenerate caption: one word 3-gram repeated at least four times (a repeated-output pattern; its cause is not inferred here)
    def loops(text):
        w = (text or "").lower().split(); grams = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
        return max((grams.count(g) for g in set(grams)), default=0) >= 4
    worst = None
    for label, pred in systems.items():
        for vid, ge in gold.items():
            if vid in taken(): continue
            for e in pred.get(vid, []):
                if loops(e.text) and (e.end_s - e.start_s) <= 200 and (worst is None or (e.end_s - e.start_s) > worst[0]): worst = (e.end_s - e.start_s, vid, e, label)
    if worst:
        dur, vid, e, label = worst
        add(vid, (max(0, e.start_s - 5), e.end_s + 5), f"{label}: a {dur:.0f} s prediction whose caption repeats one phrase. It covers {covered(e, gold[vid])} caption units." + (why(e, gold[vid]) or " The output shows repetition; it does not establish its cause."))
    spell = None
    for label, pred in systems.items():
        for vid, ge in gold.items():
            if vid in taken(): continue
            for e in pred.get(vid, []):
                cov = covered_spans(e, ge)
                if len(cov) >= 2 and spelled_out([g.text for g in cov]) >= 0.4 and (spell is None or (e.end_s - e.start_s) > spell[0]):
                    spell = (e.end_s - e.start_s, vid, e, label)
    if spell:
        dur, vid, e, label = spell
        add(vid, (max(0, e.start_s - 5), e.end_s + 5), f"{label}: a passage with many single-letter reference tokens. One prediction of {dur:.0f} s covers "
            f"{covered(e, gold[vid])} caption units." + why(e, gold[vid]))
    for vid, ge in gold.items():
        if len(ge) != 1 or vid in taken(): continue
        g = ge[0]
        for label, pred in systems.items():
            e = next((e for e in pred.get(vid, []) if temporal_iou(e.segment, g.segment) > 0 and (e.end_s - e.start_s) > 2 * (g.end_s - g.start_s)), None)
            if e: add(vid, (max(0, min(e.start_s, g.start_s) - 3), max(e.end_s, g.end_s) + 3), f"one caption unit of {g.end_s - g.start_s:.0f} s; {label} emits a {e.end_s - e.start_s:.0f} s prediction."); break
        if len(picks) >= 5: break
    return picks[:6]


def worked_example_video(gold: dict, systems: dict, tau: float, min_sentences: int = 3, max_sentences: int = 6) -> str | None:
    """A small video where the first system has both real and garbage pairs at `tau`, so both pair types appear."""
    ref = next(iter(systems.values()))
    def kinds(v):
        pairs = densevid_pairs(ref.get(v, []), gold[v], tau)
        return sum(1 for _, r in pairs if r is not None), sum(1 for _, r in pairs if r is None)
    cands = [v for v, ge in gold.items() if min_sentences <= len(ge) <= max_sentences and len(ref.get(v, [])) <= 8 and all(kinds(v))]
    return sorted(cands, key=lambda v: (len(gold[v]), v))[len(cands) // 2] if cands else None
