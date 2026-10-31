"""The online rows (eval.py --online) and the stage-2 online monitor.

Streaming FSM (infer/stream.py) runs over each stream, with the joint arm or an online segmenter cascade, and writes events and the
display replay of --stability (outputs/stability/). The online
monitor (train/slt.py, checkpoint.monitor val_online_score) runs the same FSM over whole dev videos and scores it as a row.
"""
from __future__ import annotations
from contextlib import nullcontext
from tqdm import tqdm
import argparse, json, time
import numpy as np

import torch
from poses import load_pose_window
from data.loader import annotation_fingerprint, ANNOTATION_PROTOCOL, load_language_records
from data.windowing import BIO, make_bio_labels
from infer.duration_decode import DurationModel
from infer.stability import build_policies, group_tracks, TAU_GRID
from infer.stream import MoryossefRunnerAdapter, S1RunnerAdapter
from models.bio_head import require_band_reach
from train import distributed as dist

from metrics import frame_scores
from evaluation.predictions import corpus_token, prediction_events, PredictionEvent, to_source
from evaluation.systems import (
    SEGMENTERS, build_eval_model, build_streaming_runner, load_segmenter, method_config_path, model_annotation, parse_grid,
    source_language, translator_token, translator_decoder
)
from evaluation.offline import dev_monitor_scores
from evaluation.segmenter_eval import segmenter_decode
from metrics import accumulate_policy, merged_policy, score_policy
from utils import load_yaml, output_dir, pick_device, UNIT_SEPARATOR


def _track_units(tracks, events, sentences, cover: float = 0.5) -> dict[int, tuple[float, float, int]]:
    """{track index: (start_s, end_s, reference words)} for the stability LAAL, on stream's clock. A committed track's span 
    runs from its start to the end of the event it emitted. It holds a reliable gold unit when it covers at least `cover` of it
    (metrics.held_unit_latency's rule), and each unit goes to the first committing track that holds it. A track that
    holds several units (a merge) is scored as one long unit over them."""
    ends = {(round(e.start_s, 4), round(e.commit_time_s, 4)): e.end_s for e in events}
    spans = []
    for index, tr in enumerate(tracks):
        h = tr.hyps[-1]
        end = ends.get((round(float(h.span_start_s), 4), round(float(h.commit_time_s), 4))) if tr.committed else None
        if end is not None: spans.append((float(h.commit_time_s), index, float(h.span_start_s), float(end)))

    held: dict[int, list] = {}
    for g in (u for u in sentences if u.reliable):
        need = cover * max(g.end_s - g.start_s, 1e-6)
        first = min((s for s in spans if min(s[3], g.end_s) - max(s[2], g.start_s) >= need), default=None)
        if first is not None: held.setdefault(first[1], []).append(g)
    return {i: (min(g.start_s for g in gs), max(g.end_s for g in gs), sum(len(g.text.split()) for g in gs)) for i, gs in held.items()}


@torch.no_grad()
def run_online(args: argparse.Namespace) -> dict[str, list[PredictionEvent]]:
    """Drive the sawtooth FSM end-to-end over each stream (a clean segment, from t = 0) → committed events on the source timeline.

    This is the *usable inference engine* of the online rows: a BIO head + commit gate, recompute-each-stride, no cross-stride decoder
    state. --spans ar|dlm: the arm's own head, and with --text the same arm its own translator. --spans moryossef|s1: the standalone
    segmenter under the same FSM (the online cascade), and with --text pretrimmed the pretrimmed translator on each proposed raw crop at commit.
    --text none: the spans alone, and the frame and run scores of the Moryossef protocol on the FSM's stitched tags (`run_online.last_frame`,
    the key set of the offline rows); evaluation.rows translates the spans for any other text (evaluation.offline.run_cascade).
    """
    segmenter_arch = args.spans if args.spans in SEGMENTERS else None
    translate = args.text != "none"
    data_cfg = load_yaml(args.data_config)
    inference_cfg = load_yaml(args.inference_config)
    source = source_language(args)   # the models' corpus (stamps, warm start, duration prior); the streams are --language's
    cascade_model = None
    run_online.last_model_annotation = None   # Translator's labels (model_annotation); none when nothing translates

    if segmenter_arch is not None:
        # The runner reads the pretrimmed translator's method config also with no text. A segmenter cascade has no Ω: the pretrimmed translator
        # and the segmenter adapters are ungated.
        method_cfg = load_yaml(method_config_path(args, "baseline"), language=source)
        method_cfg = {**method_cfg, "membership_gate": {**method_cfg["membership_gate"], "enabled": False}}
        model, device, velocity, _, segmenter_checkpoint = load_segmenter(args)
        model.eval().to(device)
        run_online.last_checkpoint = segmenter_checkpoint
        tokenizer = None

        if translate:
            if int(getattr(args, "num_beams", None) or method_cfg["validation"]["num_beams"]) != 1:
                raise ValueError("The online rows (the cascade and the arms) use greedy decoding (num_beams=1).")
            cascade_model, tokenizer = build_eval_model(
                "baseline", args.pretrimmed_checkpoint, source, data_cfg, method_cfg, inference_cfg, device
            )
            run_online.last_translation_checkpoint = getattr(cascade_model, "checkpoint_path", None)
            run_online.last_model_annotation = model_annotation(cascade_model, data_cfg, source)

        model = (S1RunnerAdapter(model) if segmenter_arch == "s1" else MoryossefRunnerAdapter(model, velocity)).eval().to(device)
        # The segmenter's decode (evaluation.segmenter_eval.segmenter_decode: --decode, else the decode that selected its checkpoint;
        # argmax online only by --decode argmax). The duration decode reads the train-fitted prior of the models' corpus, no
        # per-segmenter tuning; viterbi and argmax read no prior.
        decode = segmenter_decode(args, segmenter_checkpoint)
        model.duration_model = DurationModel.for_language(data_cfg, source) if decode == "duration" else None
        if decode == "viterbi": print("[online] viterbi decode: legal BIO paths only, no duration scores.", flush=True)
        if decode == "argmax": print("[online] argmax decode: each frame its most likely tag, no legal BIO path and no duration "
                                     "scores; the FSM's span, vote, cap and commit rules are the same.", flush=True)
    else:
        decode = "duration"   # the arm's own head, under the duration model it trained with
        method_cfg = load_yaml(method_config_path(args, args.spans), language=source)
        if getattr(args, "num_beams", None): method_cfg.setdefault("validation", {})["num_beams"] = int(args.num_beams)
        device = pick_device(args.device)
        model, tokenizer = build_eval_model(args.spans, args.arm_checkpoint, source, data_cfg, method_cfg, inference_cfg, device)
        run_online.last_arm_checkpoint = getattr(model, "checkpoint_path", None)
        run_online.last_model_annotation = model_annotation(model, data_cfg, source)

    run_online.last_decode = decode   # the {decode} field of the row's name (evaluation.rows)
    run_online.last_duration_model = model.duration_model.to_dict() if model.duration_model is not None else None
    run_online.last_segmentation_decode = "bio_argmax" if decode == "argmax" else \
        "semi_markov_viterbi" if model.duration_model is not None else "legal_viterbi"
    run_online.last_velocity = bool(velocity) if segmenter_arch == "moryossef" else False
    if not translate: print("[online] Segmentation only: no decode, events carry no text.", flush=True)
    runner = build_streaming_runner(
        model, inference_cfg, method_cfg, translate=translate, cascade_model=cascade_model,
        record_trace=bool(getattr(args, "stability", False)), argmax=decode == "argmax",
    )
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    # The runner refuses each stream at a rate where the head reads past the commit horizon (spec §1.4 rule 16) when it starts it:
    # every stream is checked here first, so such a stream stops the row at its start, not hours into it.
    if runner.band_reach is not None:
        for record in records: require_band_reach(runner.band_reach, float(record.pose.fps), (runner.hysteresis_strides - 1) * runner.stride_s)
    # --stability: every recorded decode as text, one entry per sentence track, per SOURCE video in source time: the provisional
    # text a live display would have had on each stride (the policies above choose what of it to show). Strings, not tensors.
    provisional_rows: dict[str, list[dict]] = {}
    # Scored per video and accumulated as (sum, count), not held as Track objects: a Track carries one cloned token
    # tensor per stride, so keeping the whole split pinned every hypothesis of every video (ase is 11,939 of them).
    # Every field score_policy returns is an unweighted mean over same track set, so a count-weighted merge is exact.
    stability_totals: dict[str, dict[str, float]] = {}
    stability_policies = build_policies()
    # The separator is no displayed word (scored as a space, as in the boundary sweep and the rows); its id comes from tokenizing it 
    # (convert_tokens_to_ids gives UNK). Words start at a "▁" piece, so LAAL counts words as the references do.
    separator_id = tokenizer(UNIT_SEPARATOR, add_special_tokens=False).input_ids[-1] if callable(tokenizer) else None
    word_start = (lambda toks: [str(p).startswith("▁") for p in tokenizer.convert_ids_to_tokens(toks)]) \
                                                       if hasattr(tokenizer, "convert_ids_to_tokens") else None

    predicted: dict[str, list[PredictionEvent]] = {}
    # The frame block of a no-text row (metrics.frame_scores, the key set of the offline rows) at the row's tIoU thresholds; a row with
    # text has none.
    thresholds = None if translate else parse_grid(
        getattr(args, "tiou_thresholds", None), load_yaml(args.eval_config)["scoring"]["tiou_thresholds"])
    frame_rows: list[dict[str, float]] = []
    stride_seconds: list[tuple[float, float]] = []
    for record in tqdm(records, desc="Processing records"):
        poses, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=False)
        if poses.shape[0] == 0:
            predicted[record.video_id] = []
            continue

        if runner.trace is not None: runner.trace.clear()
        # The released contract is isotropic and the poses are per axis, so external adapter needs the width/height of the frame 
        # the reader returns for THIS video (moryossef26.dataset.pose_aspect: 16:9 when the video has a size, else None).
        adapter = getattr(runner, "model", None)
        if hasattr(adapter, "aspect"):
            from moryossef26.dataset import pose_aspect
            adapter.aspect = pose_aspect(record.pose)

        events = runner.run(torch.as_tensor(poses, dtype=torch.float32), fps=float(record.pose.fps))
        stride_seconds += runner.stride_seconds
        if runner.trace:
            tracks = group_tracks(runner.trace)   # the commit gate's own span identity, recorded per hypothesis
            units = _track_units(tracks, events, record.sentences)
            for name, policy in stability_policies.items(): accumulate_policy(stability_totals, name, score_policy(
                tracks, policy, separator_id=separator_id, units=units, word_start=word_start))
            off = float(record.pose.offset_s)
            provisional_rows.setdefault(record.pose.video_id, []).extend({"committed": tr.committed, "hypotheses": [{
                "time_s": float(h.commit_time_s) + off, "span_start_s": float(h.span_start_s) + off,
                "kind": "committed" if h.committed else "open" if h.open_span else "closed",
                "text": tokenizer.decode(h.token_ids.tolist(), skip_special_tokens=True).strip() if tokenizer is not None else "",
            } for h in tr.hyps]} for tr in tracks)

        predicted[record.video_id] = prediction_events(record.video_id, events, tokenizer if translate else None)

        # The frame and run scores of the Moryossef protocol on the runner's stitched tags (the decode's tags, the argmax decode's too,
        # including the committed frontier, as the FSM read them; both key families read them) vs GT captions: latest estimate per frame.
        if thresholds is not None and runner._bio_timeline is not None:
            gold = make_bio_labels(
                timestamps, record.sentences, 0.0, float(record.pose.duration_s), 
                video_duration_s=record.pose.duration_s, untrusted=record.untrusted_s,
            )
            tags = runner._bio_timeline.clone()
            # Ignore-region: where gold is UNK (quarantine / untrusted gap) the pred is neither right nor wrong —
            # mask it too, or every phantom "segment" there deflates precision for a region that has no GT.
            gold_t = torch.as_tensor(np.asarray(gold)).long()
            # A stream with no gold signing (all UNK, or O/UNK only) gets no row (moryossef26.infer.evaluate_moryossef_whole_video):
            # its F1 is undefined, and a 0 or 1 there would move the mean. Its events still go to the scorer of the row.
            if not ((gold_t == BIO["B"]) | (gold_t == BIO["I"])).any(): continue
            tags[gold_t == BIO["UNK"]] = BIO["UNK"]
            logits_1hot = torch.nn.functional.one_hot(tags.clamp(min=0), num_classes=4).float().unsqueeze(0) * 10.0
            frame_rows.append(frame_scores(logits_1hot, logits_1hot, gold_t.unsqueeze(0), thresholds))

    # Real time: wall time per stride against the buffer length, on this device (ReSCUE App. G reports the mean, the p95 and 
    # the longest-buffer step). Live use needs each stride done within stride_s, the long buffers near the cap included.
    buffer_s, wall_s  = np.array([b for b, _ in stride_seconds]), np.array([w for _, w in stride_seconds])
    long_s = wall_s[buffer_s >= 30.0]
    stride_s = float(inference_cfg["stride_s"])
    run_online.last_compute = {
        "device": str(device), "strides": int(wall_s.size), "p50_s": float(np.median(wall_s)) if wall_s.size else None,
        "p95_s": float(np.percentile(wall_s, 95)) if wall_s.size else None, "max_s": float(wall_s.max()) if wall_s.size else None,
        "p95_s_buffer_ge_30s": float(np.percentile(long_s, 95)) if long_s.size else None,
        "share_within_stride": float((wall_s <= stride_s).mean()) if wall_s.size else None,
        # The stride times above exclude the decodes that only --stability needs (StreamingSLTRunner.trace_only_s), 
        # so a traced row still stamps the deployed cost; the trace's own cost is reported apart.
        "record_trace": runner.trace is not None, "trace_only_decodes": int(runner.gate_stats.get("trace_only_decoded", 0)),
        "trace_only_decode_s": float(getattr(runner, "trace_only_s", 0.0)),
        # The FSM's counters over every stream, saved so a reader can report them: committed, forced_commit (complete spans committed
        # before K votes), open_span_kept (open spans that the cap kept for the next buffer), open_span_cut (open spans that the cap cut:
        # 1 event each, flagged cap_cut and not counted in committed, so the events are committed + open_span_cut, spec §7.1;
        # open_span_cut_kept_before the cuts that only the keep-once rule causes: a span kept before that now starts more than δ after
        # the buffer start; a kept span whose start stays on its frame, exactly δ after it, is not counted) and open_span_dropped (open
        # spans that the cap or the stream end dropped: no event; open_span_dropped_at_cap the cap's share, a cut under Λ_min, and 
        # open_span_dropped_kept_before the drops that only the keep-once rule causes, counted the same way). They count passes, not units: a unit that the cap 
        # keeps twice, or keeps and then cuts, counts in each.
        "gate_stats": dict(runner.gate_stats),
    }
    print(f"[online] compute: {run_online.last_compute}", flush=True)
    # Why-did-it-(not)-commit: low streaming recall with near-perfect frame BIO = the boundary-stability gate suppressed emission.
    s = runner.gate_stats
    seen = s.get("spans_seen", 0)
    if seen: print(
        f"[online] gate: spans_seen={seen} boundary_ok={s.get('boundary_ok',0)} gated_decoded={s.get('gated_decoded',0)} "
        f"committed={s.get('committed',0)} forced={s.get('forced_commit',0)} open_span_kept={s.get('open_span_kept',0)} "
        f"open_span_cut={s.get('open_span_cut',0)} (kept_before={s.get('open_span_cut_kept_before',0)}) "
        f"open_span_dropped={s.get('open_span_dropped',0)} (at_cap={s.get('open_span_dropped_at_cap',0)}, "
        f"kept_before={s.get('open_span_dropped_kept_before',0)}) "
        f"trace_only_decoded={s.get('trace_only_decoded',0)}", flush=True
    )
    if stability_totals:
        # Stable-prefix comparison on the SAME decodes the FSM already produced: how much earlier could text appear,
        # and what does freezing it early cost? commit_only is the deployed behaviour (latency ceiling, 0 error).
        rows = {name: merged_policy(totals) for name, totals in stability_totals.items()}
        print(f"[stability] {max(r['n_tracks'] for r in rows.values())} tracks | tau grid {TAU_GRID}", flush=True)
        # ANCHORS DIFFER, deliberately. The row's `emission_latency` = commit_time - GOLD SENTENCE END: when the SCORED translation is complete, 
        # under commit_only, the only policy the FSM actually runs. `latency_s` here is first_reveal - COMMIT: how much earlier a policy 
        # would put text on screen. Latency to 1st text = emission_latency + latency_s (negative). Keep them separate — quoting a prefix's 
        # latency against the full sentence's BLEU would describe 2 different objects.
        print("latency_s is relative to the COMMIT (negative = earlier); add the row's emission_latency for gold-anchored number.", flush=True)
        print(f"{'policy':<30}{'latency_s':>11}{'revealed':>10}{'frozen_err':>12}{'rewrite_rate':>14}{'NE':>8}{'LAAL_s':>9}", flush=True)
        for name, r in rows.items(): print(
            f"{name:<30}{r['first_token_latency_s']:>11.3f}{r['revealed_fraction']:>10.3f}{r['frozen_prefix_error']:>12.3f}"
            f"{r['contradicted_track_rate']:>14.3f}{r['normalized_erasure']:>8.3f}{r['laal_s']:>9.3f}", flush=True
        )
        r = rows["commit_only"]
        print(f"[stability] LAAL on {r['n_laal_tracks']} of {r['n_tracks']} committed sentences (those holding >= half of a gold unit); "
              f"boundary flips per sentence {r['boundary_flip_rate']:.3f}", flush=True)
        # An online cascade is named after its segmenter, with -dlm when it translates with a pretrimmed DLM translator (its AR names stay).
        name = translator_token(args, args.spans, run_online.last_arm_checkpoint) if segmenter_arch is None else segmenter_arch + (
            "-dlm" if cascade_model is not None and translator_decoder("baseline", run_online.last_translation_checkpoint) == "dlm" else "")
        out = output_dir("stability") / f"stability_{name}_{corpus_token(args.language, source)}_{args.split}.json"
        out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # The texts themselves, beside the scores: per sentence track, each recorded decode with its stride time and kind (open: the
        # unit before its end appeared; closed: the span waiting for its votes; committed: the emitted text). Never read by scoring.
        texts = out.with_name(out.name.replace("stability_", "provisional_", 1))
        texts.write_text(json.dumps({"provenance": {
            "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(records), 
            "language": args.language, "source_language": source, "split": args.split, "hysteresis_strides": runner.hysteresis_strides,
            "unit_separator": UNIT_SEPARATOR, "delta_s": runner.delta_s, "stride_s": runner.stride_s,
        }, "tracks": provisional_rows}, indent=1) + "\n", encoding="utf-8")
        print(f"[stability] {sum(len(t['hypotheses']) for v in provisional_rows.values() for t in v)} recorded decodes -> {texts}", flush=True)

    # The frame block of a no-text row (evaluation.rows saves it as `frame` and does not print it); None: no stream had gold signing.
    run_online.last_frame = {
        **{k: float(sum(r[k] for r in frame_rows) / len(frame_rows)) for k in frame_rows[0]}, "metrics_mean_over": "streams",
        "frame_metrics_decode": {"bio": "fsm_stitched_tags", "phrase": "fsm_stitched_tags"},
    } if frame_rows else None
    return to_source(predicted, records)


@torch.no_grad()
def online_dev_scores(model, records, inference_cfg: dict, method_cfg: dict, edge_cfg: dict, char_level: bool) -> dict[str, float]:
    """Checkpoint selection on the deployed path (train/slt.py evaluate_slt, checkpoint.monitor val_online_score): streaming FSM 
    over `records` (whole source videos), scored as a row's main score (`dev_monitor_scores`; `edge_cfg` = eval.yaml scoring chunk_gap_s,
    edge_taus_s and tiou_thresholds). Each rank streams every world_size-th record and the events are merged, so every rank returns the
    same scores. `online_score` = the F1 of the tIoU x edge-limit grid (tau avg, tIoU avg) x the DVC BLEU-4 of the pairs that count (the
    same avg)."""
    started = time.perf_counter()
    runner = build_streaming_runner(model, inference_cfg, method_cfg)

    # Full precision, as `eval.py --online` runs: the training loop calls its monitor under CUDA autocast 
    # (see segmenter_dev_scores), and bf16/fp16 logits can move duration-Viterbi boundaries and greedy tokens.
    local = {}
    with torch.autocast(device_type="cuda", enabled=False) if torch.cuda.is_available() else nullcontext():
        for record in records[dist.rank()::dist.world_size()]:
            poses, _ = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=False)
            events = runner.run(torch.as_tensor(poses, dtype=torch.float32), fps=float(record.pose.fps)) if poses.shape[0] else []
            local[record.video_id] = prediction_events(record.video_id, events, model.tokenizer)

    scores = dev_monitor_scores(to_source(dist.merge_dicts(local), records), records, edge_cfg, char_level, "online")
    if dist.is_main(): print(f"slt | online monitor: {len(records)} streams in {time.perf_counter() - started:.0f}s", flush=True)
    return scores
