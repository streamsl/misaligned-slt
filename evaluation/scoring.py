"""Every number that evaluation reports, computed from saved events and gold.

The RQ2 scorer (one-to-one localization F1, SODA-style text, the DVC columns) and the rows built on it; metrics per reliable gold unit
read through one holding rule (an event holds at least `cover` of the unit): held-unit latency, long-form BLEU-4, the cut error of split
spans; the display replay scores of the reveal policies (`--stability`: first-text latency, erasure, rewrites, LAAL, boundary flips);
and the resampling statistics of the reported metrics (paired and video bootstraps, the checkpoint monitors' spread)."""
from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
import warnings

import numpy as np
from data.loader import VideoRecord
from data.windowing import TRUSTED_GAP_S, untrusted_o_intervals
from infer.stability import Track, _lcp, displays
from infer.unit_split import span_groups
from metrics import compute_text_metrics, densevid_text_metrics, match_segments, segmentation_prf, temporal_iou
from evaluation.predictions import _gold_events, joined_events, PredictionEvent, scoreable_predictions, to_source


def densevid_pairs(pred_events, gold_events, threshold: float) -> list[tuple[str, str | None]]:
    """DVC pairs of 1 video at 1 threshold: every (prediction, gold) pair with tIoU >= threshold, and (prediction, None)
    for a prediction that overlaps no gold (None = garbage reference, drawn by the scorer)."""
    out = []
    for ev in pred_events:
        if ev.text is None: continue
        over = [gt.text for gt in gold_events if temporal_iou(ev.segment, gt.segment) >= float(threshold) and gt.text is not None]
        out.extend((ev.text, r) for r in over) if over else out.append((ev.text, None))
    return out


def evaluate_predicted_events(
    predicted: dict[str, list[PredictionEvent]], gold: dict[str, list[PredictionEvent]], 
    thresholds: list[float], char_level: bool = False, densevid: bool = False,
) -> dict[str, Any]:
    """RQ2 scoring, SODA-style (Fujita et al. 2020) with translations in place of captions. Per tIoU threshold: predictions and gold are 
    matched 1-to-1 by tIoU (metrics.match_segments); segmentation block reports that localization P/R/F1; TEXT block is a localization-aware 
    F1 — per-pair sentence scores summed, then precision = Σ/n_pred and recall = Σ/n_gold (`compute_text_metrics(localization_aware=True)`), 
    macro-averaged over videos.

    The SODA-style fusion charges spurious predictions and missed gold. With densevid=True, separate DVC-style columns use many-to-many
    matching, garbage references for unmatched predictions, and per-video corpus scores averaged over gold videos. These columns do not
    charge missed gold individually. Neither calculation is RQ1's whole-corpus BLEU. `threshold_average` averages the threshold grid;
    emission latency and matched tIoU are reported separately."""
    results = []
    all_video_ids = sorted(set(predicted) | set(gold))
    pair_memo: dict[tuple[str, str], dict[str, float]] = {}  # (hyp, ref) -> sentence scores; reused across thresholds
    for threshold in thresholds:
        dv_pairs: dict[str, list[tuple[str, str | None]]] = {}
        per_video: list[dict[str, float]] = []
        vid_prec, vid_rec, latencies, best_tious = [], [], [], []
        n_pairs = n_unmatched_pred = n_missed_gold = total_pred = total_gold = 0
        for video_id in all_video_ids:
            pred_events, gold_events = predicted.get(video_id, []), gold.get(video_id, [])
            if not pred_events and not gold_events: continue
            total_pred += len(pred_events); total_gold += len(gold_events)

            # SEGMENTATION: the codebase's 1 canonical segment metric — greedy 1-to-1 tIoU matching, shared verbatim with segmenter-eval 
            # and the BIO monitor. densevid's own proposal P/R is coverage-based (many-to-many), which lets duplicate spans all "match" 
            # 1 gold and keeps precision at 1.0; 1-to-1 charges the duplicate, so it is the defensible number for a segmentation claim.
            pred_segs, gold_segs = [ev.segment for ev in pred_events], [gt.segment for gt in gold_events]
            if densevid and gold_events: # DVC pairs (metrics.densevid_text_metrics documents the protocol and its recall-blindness)
                dv_pairs[video_id] = densevid_pairs(pred_events, gold_events, threshold)

            prf = segmentation_prf(pred_segs, gold_segs, tiou_threshold=float(threshold))
            if pred_events: vid_prec.append(prf["precision"])
            if gold_events: vid_rec.append(prf["recall"])
            one_to_one = match_segments(pred_segs, gold_segs, threshold=float(threshold))

            # Counts come from SAME 1-to-1 matching as P/R, so they never contradict it: a merged span covering 2 gold sentences matches 
            # 1 and leaves the other missed — under many-to-many bookkeeping it would read "0 missed" against that same recall.
            n_missed_gold += len(gold_events) - len(one_to_one)
            for pi, gi, iou in one_to_one:
                best_tious.append(float(iou))
                if pred_events[pi].commit_time_s is not None: # Emission latency.
                    latencies.append(float(pred_events[pi].commit_time_s) - float(gold_events[gi].end_s))

            # TEXT: per-pair sentence scores over the 1-to-1 matched pairs, memoised across thresholds (BLEURT is a model forward). 
            # The video's F1 normalises by ITS n_pred/n_gold, so an empty-match video scores 0 and is charged — not skipped.
            hyps, refs = [], []
            for pi, gi, _ in one_to_one:
                if pred_events[pi].text is None or gold_events[gi].text is None: continue
                hyps.append(pred_events[pi].text); refs.append(gold_events[gi].text)
            n_pairs += len(hyps)
            n_unmatched_pred += len(pred_events) - len(one_to_one)
            per_video.append(compute_text_metrics(
                hyps, refs, localization_aware=True, n_pred=len(pred_events), n_gold=len(gold_events), memo=pair_memo, char_level=char_level
            ))

        # Macro over every video with predictions OR gold (empty-match videos contribute 0, charging the miss).
        text_metrics = {
            k: float(np.mean([r[k] for r in per_video])) for k in per_video[0]
        } if per_video else compute_text_metrics([], [], localization_aware=True, n_pred=0, n_gold=0, char_level=char_level)
        if densevid: text_metrics.update(densevid_text_metrics(dv_pairs, char_level=char_level))
        precision = float(np.mean(vid_prec)) if vid_prec else 0.0
        recall = float(np.mean(vid_rec)) if vid_rec else 0.0
        
        latency_block, tiou_block = {}, {}
        if latencies:
            arr = np.sort(np.asarray(latencies, dtype=np.float64))
            latency_block = {
                "median_latency_s": float(np.median(arr)),
                "p90_latency_s": float(arr[min(len(arr) - 1, int(round(0.9 * (len(arr) - 1))))]),
                "n": len(arr)
            }
        if best_tious:
            arr = np.asarray(best_tious, dtype=np.float64)
            tiou_block = {
                "min": float(np.min(arr)), "p25": float(np.percentile(arr, 25)),
                "median": float(np.median(arr)), "p75": float(np.percentile(arr, 75)),
                "max": float(np.max(arr))
            }
        results.append({
            "tiou_threshold": float(threshold),
            "segmentation": {
                "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            },
            "text_metrics": text_metrics,
            # scored_pairs feed the text score; unmatched_predictions support `precision`, missed_gold `recall`.
            "scored_pairs": n_pairs, "unmatched_predictions": n_unmatched_pred, "missed_gold": n_missed_gold,
            "total_predictions": total_pred, "total_gold": total_gold, 
            "mean_matched_tiou": float(np.mean(best_tious)) if best_tious else 0.0,
            "matched_tiou_distribution": tiou_block, "emission_latency": latency_block,
        })
    keys = results[0]["text_metrics"] if results else {}
    summary = {k: float(np.mean([r["text_metrics"][k] for r in results])) for k in keys}
    summary["segmentation_f1"] = float(np.mean([r["segmentation"]["f1"] for r in results]))
    return {"thresholds": results, "threshold_average": summary}


def _joined_threshold_average(predicted: dict, records, gold: dict, thresholds, char_level: bool, densevid: bool = True) -> dict | None:
    """RQ2 threshold averages of `predicted` (source-keyed, before the quarantine drop) with every split span joined back into the
    span the model decoded (evaluation.predictions.joined_events): what the unit split changed, with the model, the spans, the commit
    times and the words fixed. None when no span was split."""
    if not any(e.parts > 1 for evs in predicted.values() for e in evs): return None
    joined = scoreable_predictions(joined_events(predicted), records, tag="joined")
    return evaluate_predicted_events(joined, gold, list(thresholds), char_level=char_level, densevid=densevid)["threshold_average"]


def _segment_rq2_rows(segments_by_video: dict, records: list[VideoRecord], thresholds, tag: str) -> list[dict]:
    """Segmenter spans on each stream's clock (keyed by record id) -> RQ2 segmentation rows, one per tIoU threshold: source
    time, quarantine-dropped events, reliable gold per source video. `--segmenter-eval` and the segmenter dev monitor share it."""
    events = to_source({vid: [
        PredictionEvent(video_id=vid, start_s=float(s.start_s), end_s=float(s.end_s)) for s in segs
    ] for vid, segs in segments_by_video.items()}, records)
    events = scoreable_predictions(events, records, tag=tag)
    return evaluate_predicted_events(events, _gold_events(records), list(thresholds))["thresholds"]


def _holds(e, g, cover: float) -> bool:   # the event holds at least `cover` of the unit's duration
    return min(e.end_s, g.end_s) - max(e.start_s, g.start_s) >= cover * max(g.end_s - g.start_s, 1e-6)


def cut_errors(events: dict, gold: dict) -> list[tuple[str, float]]:
    """(video, |cut - end of gold unit i|) for every inner cut of each split span that holds exactly as many reliable gold units as it
    has parts (at least half of each), in event order, so two versions of one events file pair up cut by cut."""
    out = []
    for vid, evs in events.items():
        for group in span_groups(evs):
            if len(group) < 2: continue
            held = [g for g in gold.get(vid, []) if _holds(replace(group[0], end_s=group[-1].end_s), g, 0.5)]
            if len(held) == len(group): out.extend((vid, abs(e.end_s - g.end_s)) for e, g in zip(group[:-1], held))
    return out


def unit_holders(gold: dict, pred: dict, cover: float = 0.5) -> dict:
    """{video: [per reliable gold unit, the event that holds at least `cover` of it and was emitted first (commit time, else
    order), or None]}: the one event-to-unit rule of the held-unit latency, the span classes and the commit-only latency."""
    out = {}
    for vid, units in gold.items():
        evs = pred.get(vid, [])
        order = sorted(range(len(evs)), key=lambda i: (evs[i].commit_time_s if evs[i].commit_time_s is not None else float("inf"), i))
        out[vid] = [next((evs[i] for i in order if _holds(evs[i], g, cover)), None) for g in units]
    return out


def held_unit_latency(gold: dict, pred: dict, cover: float = 0.5) -> dict | None:
    """Latency per reliable gold unit: the commit time of the first event that holds at least `cover` of the unit, minus the
    unit's end. The headline latency uses only one-to-one matched pairs, so a unit inside a merged event (it waits for the whole
    span) drops out of it; here that unit counts, with the merged event's commit time. Units that no event holds are counted
    as not covered, never as zero latency. None when the events carry no commit times (offline rows)."""
    if not any(e.commit_time_s is not None for evs in pred.values() for e in evs): return None
    lat = np.concatenate([v for v in _unit_latencies(gold, pred, cover).values()] or [np.zeros(0)])
    n, arr = len(lat), np.sort(lat[np.isfinite(lat)])
    pick = lambda q: round(float(arr[min(len(arr) - 1, int(round(q * (len(arr) - 1))))]), 3) if len(arr) else float("nan")
    return {"units": n, "covered": len(arr), "covered_share": round(len(arr) / max(n, 1), 4),
            "median_s": round(float(np.median(arr)), 3) if len(arr) else float("nan"), "p90_s": pick(0.9), "mean_s": round(float(arr.mean()), 3) if len(arr) else float("nan")}


def _unit_latencies(gold: dict, pred: dict, cover: float) -> dict:
    # {video: per-unit latency (commit of its holder minus its end), +inf when no event with a commit time holds it}
    timed = {vid: [e for e in evs if e.commit_time_s is not None] for vid, evs in pred.items()}
    return {vid: np.array([np.inf if h is None else h.commit_time_s - g.end_s for g, h in zip(units, holders)], dtype=float)
            for vid, units in gold.items() for holders in [unit_holders({vid: units}, timed, cover)[vid]]}


def long_form_scores(gold: dict, pred: dict, unk: dict, char_level: bool, cover: float = 0.5) -> dict:
    """ReSCUE's axes for a commit-only row (A2f). Long-form BLEU-4: per source video, the event texts in commit order (ties by
    start) against the reliable gold texts in time order, scored as one corpus of documents; an event at least half inside an
    UNK stretch (`unk` = {video: [(start, end)]}: uncaptioned gaps and quarantined units) has no reference there and is dropped
    first, and the dropped share is reported. LAAL and AL per gold unit (`laal`): every word of the unit's holder
    (the first event holding at least `cover` of it) appears at that event's commit time, because committed text never changes
    (NE = 0); a mean per video, then over videos. Units that no event holds are reported as the uncovered share."""
    hyps, refs, dropped, total, laal_v, al_v, covered, n_units = [], [], 0, 0, [], [], 0, 0
    for vid, units in gold.items():
        evs = pred.get(vid, []); total += len(evs)
        keep = [e for e in evs if sum(max(0.0, min(e.end_s, b) - max(e.start_s, a)) for a, b in unk.get(vid, [])) < 0.5 * max(e.end_s - e.start_s, 1e-6)]
        dropped += len(evs) - len(keep)
        keep.sort(key=lambda e: (e.commit_time_s if e.commit_time_s is not None else e.end_s, e.start_s))
        hyps.append(" ".join(e.text for e in keep if e.text)); refs.append(" ".join(g.text for g in units if g.text))
        lv, av = [], []
        for g, h in zip(units, unit_holders({vid: units}, {vid: keep}, cover)[vid]):
            n_units += 1
            if h is None or h.commit_time_s is None or not h.text: continue
            covered += 1
            words, ref_words = [h.commit_time_s] * len(h.text.split()), len((g.text or "").split())
            lv.append(laal(words, g.start_s, g.end_s, ref_words)); av.append(laal(words, g.start_s, g.end_s, 0))
        if lv: laal_v.append(float(np.mean(lv))); al_v.append(float(np.mean(av)))
    bleu = compute_text_metrics(hyps, refs, char_level=char_level, bleurt_checkpoint=None)["translation_bleu4"] if hyps else 0.0
    return {"long_form_bleu4": round(float(bleu), 2), "dropped_share": round(dropped / max(total, 1), 4),
            "laal_s": round(float(np.mean(laal_v)), 3) if laal_v else float("nan"), "al_s": round(float(np.mean(al_v)), 3) if al_v else float("nan"),
            "covered_share": round(covered / max(n_units, 1), 4), "videos": len(hyps)}


def unk_stretches(records) -> dict:
    """{source video: [(start, end)]} in source time where long-form BLEU-4 has no reference: the uncaptioned gaps (UNK) and the
    quarantined units of every stream."""
    unk: dict = {}
    for rec in records:
        off = rec.pose.offset_s
        stretches = rec.untrusted_s if rec.untrusted_s is not None else untrusted_o_intervals(rec.sentences, rec.pose.duration_s, TRUSTED_GAP_S)
        unk.setdefault(rec.pose.video_id, []).extend([(a + off, b + off) for a, b in stretches] +
                                                     [(u.start_s + off, u.end_s + off) for u in rec.sentences if not getattr(u, "reliable", True)])
    return unk


# ---------------------------------------------------------------- display replay scores (eval.py --rq 2 --online --stability)

# Rates rebuilt from summed counts when videos merge (numerator, denominator): exact, and a video with no committed track counts.
COUNT_RATES = {
    "phantom_reveal_rate": ("n_phantom_reveals", "n_tracks"), "normalized_erasure": ("n_erased_tokens", "n_final_tokens"), 
    "boundary_flip_rate": ("n_boundary_flips", "n_tracks"), "laal_s": ("laal_total_s", "n_laal_tracks")
}
SUMS = ("laal_total_s",)   # summed like the n_* counts, never averaged over n_tracks


def _units(tokens: list[int], separator_id: int) -> list[tuple[int, ...]]:
    units, unit = [], []
    for t in tokens:
        if t == separator_id: units.append(tuple(unit)); unit = []
        else: unit.append(t)
    return units + [tuple(unit)]


def _boundary_flip(previous: list[tuple[int, ...]], units: list[tuple[int, ...]]) -> bool:
    # A separator added or removed, or one moved: the units on BOTH sides of a separator changed. A rewrite inside 1 unit
    # changes only that unit, so it is an erasure, not a flip.
    if len(previous) != len(units): return True
    return any(previous[j] != units[j] and previous[j + 1] != units[j + 1] for j in range(len(units) - 1))


def strip_separator(track: Track, separator_id: int | None) -> tuple[Track, int]:
    """(the track with the separator removed from every hypothesis, its boundary flips): the number of strides whose separators
    changed from the previous stride's (`_boundary_flip`: 1 added or removed, or 1 moved). The separator is scored as a space, 
    as in dev BLEU, RQ1 and RQ2, so a flip changes no displayed word; the flips are counted apart."""
    if separator_id is None: return track, 0
    hyps, flips, previous = [], 0, None
    for h in track.hyps:
        keep = h.token_ids != int(separator_id)
        units = _units([int(t) for t in h.token_ids.tolist()], int(separator_id))
        flips += int(previous is not None and _boundary_flip(previous, units))
        previous = units
        hyps.append(SimpleNamespace(
            commit_time_s=h.commit_time_s, span_start_s=h.span_start_s, committed=h.committed,
            token_ids=h.token_ids[keep], token_confidence=h.token_confidence[keep]
        ))
    return Track(span_start_s=track.span_start_s, hyps=hyps), flips



def erased_tokens(screens: list[tuple[float, list[int]]], committed: bool) -> int:
    # Tokens removed from the screen from one stride to the next (Arivazhagan et al. 2020); 
    # a sentence that never commits leaves the screen in full.
    erased, previous = 0, []
    for _, shown in screens:
        erased += len(previous) - _lcp(previous, shown)
        previous = shown
    return erased + (0 if committed else len(previous))


def stabilization_times(screens: list[tuple[float, list[int]]], final: list[int]) -> list[float]:
    # Per final token: the time from which the screen shows it at its position on every later stride.
    times = [float("nan")] * len(final)
    for j, token in enumerate(final):
        for t, shown in screens:
            ok = j < len(shown) and shown[j] == token
            if ok and np.isnan(times[j]): times[j] = t
            elif not ok: times[j] = float("nan")
    return times


def laal(word_times_s: list[float], unit_start_s: float, unit_end_s: float, reference_words: int) -> float:
    """Length-adaptive average lagging (Papi et al. 2022) of one caption unit, in seconds. Word i appears at word_times_s[i];
    delays count from the unit's start; the source is the unit's own duration; the oracle rate uses the longer of the output
    and the reference, so an over-long output is not rewarded. tau is the first word shown after the unit has ended."""
    delays = [float(t) - float(unit_start_s) for t in word_times_s]
    if not delays: return float("nan")
    duration = float(unit_end_s) - float(unit_start_s)
    rate = duration / max(len(delays), int(reference_words), 1)
    tau = next((i for i, d in enumerate(delays, 1) if d >= duration), len(delays))
    return float(sum(delays[i] - i * rate for i in range(tau)) / tau)


def score_policy(
    tracks: list[Track], policy, separator_id: int | None = None, units: dict | None = None, word_start=None, **kw
) -> dict:
    """Latency and prefix-correctness for one policy over all tracks.

    first_token_latency_s    when the FIRST token becomes visible, relative to the commit the FSM would have made:
                             negative = the policy showed text EARLIER than commit_only (the point of the exercise).
    frozen_prefix_error      the largest number of shown tokens that disagree with the sentence finally committed, on any
                             screen before the commit, per most tokens shown before it — the price of revealing early.
                             0 for commit_only by construction. This is the FIRST-ORDER cost only: the traces come from 
                             unfrozen re-translation, so it cannot capture a frozen prefix degrading the continuation. 
                             That needs prefix-forced decoding.
    revealed_fraction        the longest prefix of the COMMITTED sentence that a screen showed before the commit, per its
                             length, in [0, 1]. Wrong or surplus shown tokens add nothing here; frozen_prefix_error and
                             vanished_track_rate charge them.
    contradicted_track_rate  fraction of sentences with at least 1 screen before the commit that disagrees with the
                             committed sentence — the rewrite rate, and the decision number for prefix forcing. Text that
                             disappears and returns unchanged is no contradiction; normalized_erasure counts it.
    phantom_reveal_rate      tracks that never commit (a target the FSM decoded on deferred strides, or an open span that
                             vanished before its terminator appeared, and never emitted) on which the policy showed at 
                             least 1 token, per committed track: text on screen that the system never emits. Such tracks 
                             add nothing to the other fields, which describe committed sentences.
    normalized_erasure       tokens removed from the screen (a changed shown token, a shorter hypothesis, a never-committed
                             sentence leaving the screen) per committed final token, over all tracks (NE, Arivazhagan et al.
                             2020; the metric ReSCUE reports). 0 for commit_only: the deployed display is append-only.
    boundary_flip_rate       strides whose separators changed from the previous stride's of the same sentence (one added,
                             removed or moved), per committed track (the separator itself is dropped from every other field).
    laal_s                   length-adaptive average lagging in seconds (the latency ReSCUE reports), over committed tracks
                             that `units` maps to the gold units they hold ({track index: (start_s, end_s, reference words)},
                             same clock as the trace): word i appears when its tokens become stable on screen. `word_start`
                             (token ids -> word-start flags) makes it word-level; without it each token is a word.
    """
    lat, err, revealed, contradicted, vanished = [], [], [], [], []
    phantoms = erased = final_tokens = flips = n_laal = 0
    laal_total = 0.0

    for index, tr in enumerate(tracks):
        if not tr.hyps: continue
        tr, track_flips = strip_separator(tr, separator_id)
        reveals = policy(tr, **kw)
        screens = displays(tr, reveals)
        erased += erased_tokens(screens, tr.committed)
        if not tr.committed:
            phantoms += int(any(shown for _, shown in screens))   # text that was on screen and is never emitted
            continue

        flips += track_flips   # a committed track counts under every policy, an empty sentence too, so all share 1 track set
        final_tokens += len(tr.final_tokens)
        unit = (units or {}).get(index)

        if unit is not None and tr.final_tokens:
            times = stabilization_times(screens, tr.final_tokens)
            starts = word_start(tr.final_tokens) if word_start is not None else [True] * len(times)
            words: list[float] = []
            for t, first in zip(times, starts):
                if first or not words: words.append(t)
                else: words[-1] = max(words[-1], t)   # a word is shown when its last token is stable
            laal_total += laal(words, *unit); n_laal += 1

        final = tr.final_tokens
        commit_t = tr.commit_time_s
        # Every screen BEFORE the commit, not only those at a reveal: an unforced re-translation can 
        # rewrite a shown token on a stride that reveals nothing new.
        before = [shown for t, shown in screens if t < commit_t]
        lat.append(next((t for t, shown in screens if shown), commit_t) - commit_t)  # 1st stride with text on screen
        k = max((len(shown) for shown in before), default=0)

        # The committed sentence's own prefix on screen (at most all of it): wrong or surplus shown tokens 
        # are text that must change or vanish, and frozen_prefix_error / vanished_track_rate charge them.
        revealed.append(max((_lcp(shown, final) for shown in before), default=0) / max(1, len(final)))
        # Positions shown BEYOND the committed sentence are contradictions too: they were displayed and the final sentence does
        # not contain them (zip() alone would drop them for a hypothesis that shrinks between strides, as the DLM canvas does).
        wrong = max((
            sum(1 for a, b in zip(shown, final) if a != b) + max(0, len(shown) - len(final)) for shown in before
        ), default=0)
        err.append(wrong / max(1, k) if k else 0.0)
        contradicted.append(1.0 if wrong else 0.0)
        # Showing MORE tokens than the sentence finally has means displayed text later disappears — a distinct failure from a
        # token changing, and the signature of a reveal policy reading slots the decode never really scored (post-EOS padding
        # carries a fabricated confidence of 1.0). Should be ~0 on a trimmed trace.
        vanished.append(1.0 if k > len(final) else 0.0)
    return {
        "n_tracks": len(lat),
        "first_token_latency_s": float(np.mean(lat)) if lat else 0.0,
        "revealed_fraction": float(np.mean(revealed)) if revealed else 0.0,
        "frozen_prefix_error": float(np.mean(err)) if err else 0.0,
        # Fraction of SENTENCES whose display would visibly rewrite at least once. This is the number that decides
        # whether prefix-forced decoding is worth building: near 0 means unforced re-translation is already stable.
        "contradicted_track_rate": float(np.mean(contradicted)) if contradicted else 0.0,
        "vanished_track_rate": float(np.mean(vanished)) if vanished else 0.0,
        "n_phantom_reveals": phantoms, "phantom_reveal_rate": phantoms / len(lat) if lat else 0.0,
        "n_erased_tokens": erased, "n_final_tokens": final_tokens, 
        "normalized_erasure": erased / final_tokens if final_tokens else 0.0,
        "n_boundary_flips": flips, "boundary_flip_rate": flips / len(lat) if lat else 0.0,
        "n_laal_tracks": n_laal, "laal_total_s": laal_total,
        "laal_s": laal_total / n_laal if n_laal else 0.0,   # read with n_laal_tracks, as every rate with its count
    }


def accumulate_policy(totals: dict, name: str, row: dict) -> None:
    """Fold 1 video's `score_policy` row into a running (sum, count) so the caller never holds every Track.

    The `n_*` fields are counts and add up. Every other field is an unweighted mean over that call's committed tracks, 
    so weighting by `n_tracks` and dividing at the end reproduces scoring the whole split at once exactly; the COUNT_RATES 
    are rebuilt from their counts, so a video with phantoms but no committed track still counts.
    """
    acc = totals.setdefault(name, {})
    n = int(row["n_tracks"])
    for key, value in row.items():
        if key.startswith("n_"): acc[key] = acc.get(key, 0) + int(value)
        elif key in SUMS: acc[key] = acc.get(key, 0.0) + float(value)
        elif key not in COUNT_RATES: acc[key] = acc.get(key, 0.0) + float(value) * n


def merged_policy(totals: dict) -> dict:
    n = int(totals.get("n_tracks", 0))
    means = {k: (v / n if n else 0.0) for k, v in totals.items() if not k.startswith("n_") and k not in SUMS}
    counts = {k: (v if k in SUMS else int(v)) for k, v in totals.items() if k.startswith("n_") or k in SUMS}
    rates = {
        rate: (counts.get(num, 0) / counts[den] if counts.get(den) else 0.0) 
        for rate, (num, den) in COUNT_RATES.items()
    }
    return {**counts, **means, **rates}


def monitor_score_spread(pred: dict, gold: dict, thresholds: list, char_level: bool, n: int = 200, seed: int = 0) -> tuple[float, float]:
    """(the monitor score, its video-bootstrap SD). The statistic is the headline one: per threshold, precision averaged over the videos with events 
    and recall over the videos with gold give F1, DVC BLEU-4 is the mean over gold videos, and the score is threshold-average of F1 x DVC BLEU-4. 
    Each replicate redraws whole source videos, so the SD measures how much the monitor's own sample of videos moves the score, on any corpus."""
    vids = sorted(v for v in set(pred) | set(gold) if pred.get(v) or gold.get(v))
    shape = (len(vids), len(thresholds))
    precision, recall, dvc = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)

    for j, th in enumerate(thresholds):
        rows = densevid_text_metrics(
            {v: densevid_pairs(pred.get(v, []), gold[v], th) for v in vids if gold.get(v)},
            char_level=char_level, per_video=True, bleurt_checkpoint=None
        )   
        for i, v in enumerate(vids):
            p, g = [e.segment for e in pred.get(v, [])], [e.segment for e in gold.get(v, [])]
            prf = segmentation_prf(p, g, tiou_threshold=float(th))
            if p: precision[i, j] = prf["precision"]
            if g: recall[i, j], dvc[i, j] = prf["recall"], rows[v]["bleu4"]

    def score(idx) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # a draw with no event (or no gold) in a column scores 0 there
            p, r, d = (np.nan_to_num(np.nanmean(a[idx], 0)) for a in (precision, recall, dvc))
        f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
        return float(f1.mean() * d.mean())
    
    if not vids: return 0.0, 0.0
    rng = np.random.default_rng(seed)
    return score(np.arange(len(vids))), float(np.std([score(rng.integers(0, len(vids), len(vids))) for _ in range(n)]))


def dev_monitor_scores(raw: dict, records: list[VideoRecord], thresholds: list, char_level: bool, name: str) -> dict[str, float]:
    """A stage-2 checkpoint monitor's numbers (rq2_online.online_dev_scores, rq2_offline.offline_dev_scores) from the source-time events of
    whole dev videos, scored as RQ2 (threshold averages): `{name}_score` = F1 avg x DVC BLEU-4 avg, both factors, the video-bootstrap
    spread of the score (monitor_score_spread) and events per gold unit."""
    gold = _gold_events(records)
    pred = scoreable_predictions(raw, records, tag=f"val-{name}")
    avg = evaluate_predicted_events(pred, gold, list(thresholds), char_level=char_level, densevid=True)["threshold_average"]
    f1, bleu = float(avg["segmentation_f1"]), float(avg["densevid_bleu4"])
    _, spread = monitor_score_spread(pred, gold, list(thresholds), char_level)
    return {
        f"{name}_score": f1 * bleu, f"{name}_f1": f1, f"{name}_dvc_bleu4": bleu, f"{name}_score_sd": spread,
        f"{name}_events_per_gold": sum(map(len, pred.values())) / max(1, sum(map(len, gold.values()))),
    }


def dvc_by_video(pred, gold, thresholds, char_level: bool) -> tuple[list[str], np.ndarray]:
    """DVC BLEU-4 per gold video, averaged over `thresholds`: the rows the headline mean is taken over, so a paired
    bootstrap with one unit per video tests the headline difference itself, extra and phantom events included."""
    vids = sorted(v for v, ge in gold.items() if ge)
    per_t = [densevid_text_metrics({v: densevid_pairs(pred.get(v, []), gold[v], t) for v in vids}, char_level=char_level,
                                   bleurt_checkpoint=None, per_video=True) for t in thresholds]
    return vids, np.array([np.mean([pv[v]["bleu4"] for pv in per_t]) for v in vids])


def paired_bootstrap(scores_a: np.ndarray, scores_b: np.ndarray, video_ids, n: int = 2000, seed: int = 42) -> dict:
    """Paired cluster bootstrap: mean delta (a - b) per unit, 95 % CI and a two-sided p-value. A unit is a gold unit or a whole video.

    Gold units in one video share signer, topic and segmenter errors, so they are not independent. Each replicate
    draws VIDEOS with replacement (`video_ids[i]` is the video of unit i) and takes the sum of the drawn videos'
    summed deltas over the sum of their unit counts; a video drawn twice counts twice. An i.i.d. resample of units
    gives a CI that is too narrow when the delta clusters by video."""
    assert scores_a.shape == scores_b.shape and scores_a.ndim == 1 and len(video_ids) == len(scores_a)
    rng = np.random.default_rng(seed)
    deltas = scores_a - scores_b
    _, video = np.unique(np.asarray(video_ids), return_inverse=True)
    sums, counts = np.bincount(video, weights=deltas), np.bincount(video)
    if len(sums) < 2:  # one video: resampling it gives a zero-width interval, which would claim significance
        return {"mean_delta": round(float(deltas.mean()), 3) if len(deltas) else 0.0, "ci95": [float("nan")] * 2,
                "p_value": 1.0, "n_units": int(len(deltas))}
    idx = rng.integers(0, len(sums), size=(int(n), len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    p = 2.0 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return {"mean_delta": round(float(deltas.mean()), 3),
            "ci95": [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)],
            "p_value": round(min(1.0, p), 4), "n_units": int(len(deltas))}


def video_bootstrap_ci(values, n: int = 4000, seed: int = 0) -> list[float]:
    """95 % CI of a mean over videos, by resampling the videos. A DVC value is such a mean, and with few videos and
    short ones it is wide, so the interval belongs next to the number."""
    v = np.asarray(list(values), dtype=float)
    if v.size < 2: return [float(v.mean()) if v.size else 0.0] * 2
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(int(n), v.size))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)]


def headline_f1_bootstrap(pred_a: dict, pred_b: dict, gold: dict, thresholds, n: int = 2000, seed: int = 42) -> dict:
    """Paired video bootstrap of the headline segmentation F1 avg, system a minus system b. The statistic is the one
    `evaluation.scoring.evaluate_predicted_events` reports: precision averaged over the videos with predictions, recall over the videos with
    gold, their F1 at each threshold, then the mean over the thresholds. Each replicate draws videos with replacement and
    recomputes that statistic for both systems, so the CI is for the headline difference itself (F1 is not a mean over videos)."""
    vids = sorted(v for v in set(gold) | set(pred_a) | set(pred_b) if gold.get(v) or pred_a.get(v) or pred_b.get(v))
    def table(pred):  # (videos, thresholds, 4): precision, has predictions, recall, has gold
        t = np.zeros((len(vids), len(thresholds), 4))
        for i, v in enumerate(vids):
            p, g = [e.segment for e in pred.get(v, [])], [e.segment for e in gold.get(v, [])]
            for j, th in enumerate(thresholds):
                prf = segmentation_prf(p, g, tiou_threshold=float(th))
                t[i, j] = (prf["precision"], float(bool(p)), prf["recall"], float(bool(g)))
        return t
    def f1_avg(t, idx):
        has_p, has_g = t[idx, :, 1], t[idx, :, 3]
        p = (t[idx, :, 0] * has_p).sum(0) / np.maximum(has_p.sum(0), 1)
        r = (t[idx, :, 2] * has_g).sum(0) / np.maximum(has_g.sum(0), 1)
        return float(np.mean(np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)))
    ta, tb = table(pred_a), table(pred_b)
    every = np.arange(len(vids))
    out = {"f1_avg_a": round(f1_avg(ta, every), 4), "f1_avg_b": round(f1_avg(tb, every), 4)}
    out["mean_delta"] = round(out["f1_avg_a"] - out["f1_avg_b"], 4)
    if len(vids) < 2: return {**out, "ci95": [float("nan")] * 2, "p_value": 1.0, "n_videos": len(vids)}
    rng = np.random.default_rng(seed)
    deltas = np.array([f1_avg(ta, idx) - f1_avg(tb, idx) for idx in rng.integers(0, len(vids), size=(int(n), len(vids)))])
    p = 2.0 * min(float((deltas <= 0).mean()), float((deltas >= 0).mean()))
    return {**out, "ci95": [round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)],
            "p_value": round(min(1.0, p), 4), "n_videos": len(vids)}


def held_latency_bootstrap(gold: dict, pred_a: dict, pred_b: dict, cover: float = 0.5, n: int = 2000, seed: int = 42) -> dict:
    """Paired video bootstrap of the held-unit median latency, system a minus system b. A unit that no event holds counts as
    +inf (it never appeared), so a system cannot buy a lower median by dropping slow units."""
    la, lb = _unit_latencies(gold, pred_a, cover), _unit_latencies(gold, pred_b, cover)
    vids = [v for v in sorted(gold) if len(la[v])]
    med = lambda lat, idx: float(np.median(np.concatenate([lat[vids[i]] for i in idx]))) if len(idx) else float("nan")
    every = np.arange(len(vids))
    a, b = med(la, every), med(lb, every)
    out = {"median_a_s": round(a, 3), "median_b_s": round(b, 3), "mean_delta": round(a - b, 3) if np.isfinite(a - b) else float("nan"),
           "covered_a": round(float(np.isfinite(np.concatenate([la[v] for v in vids])).mean()), 4) if vids else float("nan"),
           "covered_b": round(float(np.isfinite(np.concatenate([lb[v] for v in vids])).mean()), 4) if vids else float("nan"), "n_videos": len(vids)}
    if len(vids) < 2: return {**out, "ci95": [float("nan")] * 2, "p_value": 1.0}
    rng = np.random.default_rng(seed)
    d = np.array([med(la, idx) - med(lb, idx) for idx in rng.integers(0, len(vids), size=(int(n), len(vids)))])
    d = d[np.isfinite(d)]   # a replicate whose median is +inf in both systems has no finite difference
    if not len(d): return {**out, "ci95": [float("nan")] * 2, "p_value": 1.0}
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return {**out, "ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)], "p_value": round(min(1.0, p), 4)}
