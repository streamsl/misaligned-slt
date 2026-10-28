"""The offline rows (eval.py --offline, --segments) and the stage-2 offline monitor.

Cascade rows (--segments) translate given spans: gold spans, a segmenter's span file, or spans an online run committed. The offline 
joint row (--offline) segments each whole stream once with the arm's own head and translates each span once. The offline monitor 
(train/slt.py, checkpoint.monitor val_offline_score) runs the offline joint row over whole dev videos and scores it as a row.
"""
from __future__ import annotations
from contextlib import nullcontext
from tqdm import tqdm
import argparse, time
from itertools import groupby
import numpy as np

import torch
from poses import load_pose_window
from data.loader import load_language_records, VideoRecord
from models.bio_head import chunk_normalized_logits
from infer.commit_gate import bio_complete_spans, candidate_spans
from infer.duration_decode import DurationDecoder
from train import distributed as dist

from metrics import Segment, edge_chunk_scores
from evaluation.predictions import (
    gold_events, translated_events, gold_chunks, load_prediction_file, load_prediction_rows, PredictionEvent,
    scoreable_predictions, to_source, warn_annotation_mismatch)
from evaluation.systems import (
    box_normalized_window, build_eval_model, commit_box_s, method_config_path, translate_windows, model_annotation, source_language)
from utils import FRAME_SLACK, lambda_min_s, load_yaml, pick_device, seconds_to_frames


@torch.no_grad()
def run_cascade(args: argparse.Namespace) -> dict[str, list[PredictionEvent]]:
    """Segment-then-translate cascade: predicted spans (analyze.py --stage segmenter-infer JSON, via --segments)
    are cut from the pose stream and translated offline. Scored by the same tIoU/translation harness as the streaming FSM.

    The prior-art FLOOR is an independent (Moryossef) segmenter's spans translated by the CLEAN baseline, so pass `--method baseline`.
    Translation uses `args.method` (NOT pinned), so `--method dlm` here is a DIFFERENT row (Moryossef spans + our DLM, offline) — 
    legitimate, but not the floor. Pass the --method for the row you want.

    Normalization follows each translator's training. Clean translator trains on caption crops, so it reads each crop normalized alone. 
    A gated arm trains on poses normalized over the whole stage-2 window, and the FSM normalizes its whole buffer, so an arm row reads  
    same crop normalized with the body box of the buffer that would commit the span (commit_box_s; for a cap-cut span, the cap buffer 
    from δ before it), clipped to the stream and, for an online span, to the frames its buffer held at commit.
    """
    data_cfg = load_yaml(args.data_config)
    source = source_language(args)   # the translator's corpus; the spans' streams and gold are --language's
    method_cfg = load_yaml(method_config_path(args), language=source)
    if getattr(args, "num_beams", None): method_cfg.setdefault("validation", {})["num_beams"] = int(args.num_beams)
    inference_cfg = load_yaml(args.inference_config)
    device = pick_device(args.device)

    model, tokenizer = build_eval_model(args.method, args.checkpoint, source, data_cfg, method_cfg, inference_cfg, device)
    run_cascade.last_checkpoint = getattr(model, "checkpoint_path", None)
    run_cascade.last_model_annotation = model_annotation(model, data_cfg, source)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)

    # Span files are on the source timeline; each span is cut from the stream (clean segment) that holds its start.
    streams: dict[str, list[VideoRecord]] = {}
    for record in records: streams.setdefault(record.pose.video_id, []).append(record)
    warn_annotation_mismatch(args.segments, records, "--segments")
    segments = load_prediction_file(args.segments)
    
    # Split comes from --split, NOT the JSON filename; mismatched video_ids translate nothing and score all-zero,
    # so fail loud ("gold_*_test.json but forgot --split test").
    matched = set(segments) & set(streams)
    if not matched: raise SystemExit(
        f"--segments has {len(segments)} video_ids, none in the {len(streams)} '{args.split}' videos — "
        f"nothing to translate (output would be all-zero). Pass --split test (+ --allow-test) for test gold spans."
    )
    if len(matched) < len(segments): print(
        f"[cascade] WARNING: {len(segments) - len(matched)}/{len(segments)} segment video_ids "
        f"are absent from --split {args.split}; scoring only the {len(matched)} that match.", flush=True
    )
    longest = max(((span.duration_s, vid, span.start_s, span.end_s) for vid in matched for span in segments[vid]), default=None)
    if longest is not None: print(
        f"[cascade] batch_size={int(args.batch_size)}; longest proposal: video={longest[1]} "
        f"start={longest[2]:.3f}s end={longest[3]:.3f}s duration={longest[0]:.3f}s", flush=True
    )
    # The supplied proposal fixes the crop, the reported interval and, for a joint translator, Ω: the gate reads the supplied span
    # (a point-mass anchor), and each span stays ONE event. So the row moves the translator only, at fixed spans.
    src_rows = load_prediction_rows(args.segments)   # an events file (online spans) carries commit time, forced and cap-cut flags
    predicted: dict[str, list[PredictionEvent]] = {}
    gated = args.method != "baseline"
    lead_s, tail_s = commit_box_s(inference_cfg) if gated else (0., 0.)   # an arm row's body box (see the docstring)
    unplaced = 0

    for video_id, spans in tqdm(segments.items(), desc="Processing segments"):
        if video_id not in streams: continue
        items, kept, kept_rows, anchors = [], [], [], []
        rows = src_rows.get(video_id) or []
        for i, span in enumerate(spans):
            record = next((
                r for r in streams[video_id] if r.pose.offset_s - 1e-6 <= span.start_s < r.pose.offset_s + r.pose.duration_s
            ), None)
            if record is None:  # starts inside a cut run or past the video: no stream holds it
                unplaced += 1
                continue
            start_s = float(span.start_s) - record.pose.offset_s
            # [start, end + 1 frame]: the crop holds the span's terminator frame unless the span ends at the stream end (as run_offline's
            # window does), so a joint translator's readout inside it doesn't flip between closed and open with the float rounding of `end`.
            end_s = float(span.end_s) - record.pose.offset_s + 1.0 / float(record.pose.fps)
            # An online span reads only the frames its buffer held at commit (timestamps < commit time). A cap-forced open span (old event
            # files only: the FSM no longer emits one, spec §7.1) ends at that time and has no terminator frame to add; the epsilon absorbs the float noise of the source-time round trip.
            held_s, row = float(record.pose.duration_s), (rows[i] if i < len(rows) else {})
            commit = row.get("commit_time_s")
            if commit is not None: held_s = min(held_s, float(commit) - record.pose.offset_s - 1e-6)
            w_end = min(held_s, end_s)
            # The box ends where the committing buffer ends: K - 1 strides after the terminator, or for a cap-cut span the cap itself.
            box_end = max(0., start_s - lead_s) + float(inference_cfg["buffer_cap_s"]) if row.get("cap_cut") else end_s + tail_s

            if gated: poses, timestamps = box_normalized_window(record.pose, start_s, w_end, start_s - lead_s, min(held_s, box_end))
            else: poses, timestamps = load_pose_window(record.pose, start_s, w_end, normalize=True)
            if poses.shape[0] == 0: continue
            items.append((poses, timestamps, start_s, float(record.pose.fps))); kept.append(span)
            kept_rows.append(row)
            # Window-relative (start, terminator) frames: the crop starts at the span, and its terminator frame is the 
            # first frame at or after the span end (-1 when the crop ends before it: the span then runs to the crop end).
            term = int(np.searchsorted(timestamps, float(span.end_s) - record.pose.offset_s - FRAME_SLACK / float(record.pose.fps)))
            anchors.append((0, term if term < int(poses.shape[0]) else -1))

        # A span longer than the buffer cap translates in a batch of its own, as in offline_predictions (memory only: each row's result
        # matches translating it alone): a segmenter merge can last minutes, and a batch pads every row to its longest.
        cap_s, results = float(inference_cfg["buffer_cap_s"]), []
        for alone, group in groupby(range(len(items)), key=lambda i: kept[i].duration_s > cap_s):
            for part in ([[i] for i in group] if alone else [list(group)]):
                results += translate_windows( # Cascade windows are cut exactly at the predicted span start, so frame 0 is that span's onset.
                    model, tokenizer, args.method, [items[i] for i in part], device, inference_cfg, method_cfg,
                    batch_size=int(args.batch_size), stream_start=True, anchor_frames=[anchors[i] for i in part] if gated else None,
                )
        # The event keeps the span file's source-time interval and row (commit time, forced and cap-cut flags): no second shift.
        predicted[video_id] = translated_events(video_id, kept, results, kept_rows)
    if unplaced: print(f"[cascade] WARNING: {unplaced} span(s) start in no stream (inside a "
                       f"cut run or past the video end); they are not translated.", flush=True)
    return predicted


@torch.no_grad()
def run_offline(args: argparse.Namespace) -> dict[str, list[PredictionEvent]]:
    """Offline self-segmentation followed by captioning of each unchanged proposal.

    Each proposal is captioned in window [start - δ, end + 1 frame] (δ before the span as committed context χ, the span, its terminator frame) 
    under a point-mass Ω on the span. The window is normalized with ONE Uni-Sign body box over [start - δ, end + 1 frame + (K - 1) strides] 
    (commit_box_s), the frames that a buffer committing the span would hold: the arm trains on poses normalized over its whole stage-2 window, 
    and FSM normalizes its whole buffer (which can start earlier, at previous commit's terminator - δ). Every complete proposal stays whole,
    at any length, as in a standard offline dense captioning row: the offline rows have no buffer cap, so they need no cap cut (the one
    declared exception to P1 is the engine's, spec §7.1); there is no commit/hysteresis and no cross-stride refinement.

    Held against `--online` at the SAME model to measure what streaming buys; it folds in the cost of causal (vs offline-bidirectional) 
    segmentation, so it is a conservative baseline, not a pure-refinement control.
    """
    if args.method == "baseline": raise SystemExit(
        "The --offline row uses the trained model's own BIO head (ar/dlm); the baseline has no "
        "trained head. For the external-segmenter cascade rows, pass --segments."
    )
    data_cfg = load_yaml(args.data_config)
    source = source_language(args)   # the arm's corpus (its stamps and its stamped duration prior); the streams are --language's
    method_cfg = load_yaml(method_config_path(args), language=source)
    if getattr(args, "num_beams", None): method_cfg.setdefault("validation", {})["num_beams"] = int(args.num_beams)
    inference_cfg = load_yaml(args.inference_config)
    device = pick_device(args.device)

    model, tokenizer = build_eval_model(args.method, args.checkpoint, source, data_cfg, method_cfg, inference_cfg, device)
    run_offline.last_checkpoint = getattr(model, "checkpoint_path", None)
    run_offline.last_model_annotation = model_annotation(model, data_cfg, source)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    predicted = offline_predictions(model, tokenizer, args.method, records, inference_cfg, method_cfg, device, int(args.batch_size))
    return to_source(predicted, records)


def offline_predictions(
    model, tokenizer, method: str, records: list[VideoRecord], inference_cfg: dict, method_cfg: dict, 
    device: torch.device, batch_size: int, progress: bool = True,
) -> dict[str, list[PredictionEvent]]:
    """The offline row's events per stream, in stream time (`run_offline`; the stage-2 monitor `offline_dev_scores`): the model's own BIO
    head over each whole stream in chunks at the cap, the duration decode, the span rule and Λ_min floor of the FSM with no buffer cap
    (infer.commit_gate.candidate_spans: every complete span whole), and one translation of each proposal."""
    _, tail_s = commit_box_s(inference_cfg)   # the window itself already starts δ before the span
    
    # SEGMENTATION chunks at the cap the arm trained under, which build_eval_model has checked equals the live cap. With the banded 
    # head a frame's logits depend only on its receptive field, so overlap-stitched chunks are exact once chunk/4 >= that field, for 
    # fixed per-frame features: each chunk has its own Uni-Sign body box, so the features themselves differ slightly between chunks.
    # The chunk is the head's input context only: no span is cut at it.
    seg_chunk_cap_s = float(inference_cfg["buffer_cap_s"])
    min_span_s = lambda_min_s(inference_cfg)
    predicted: dict[str, list[PredictionEvent]] = {}
    n_sub_lambda = 0
    for record in tqdm(records, desc="Offline (self-segment + translate)", disable=not progress):
        raw, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=False)
        if raw.shape[0] == 0:
            predicted[record.video_id] = []
            continue
        # Model's OWN BIO head over the video in chunks at the trained context, each chunk normalized on its own frames — 
        # the frame the head trained on (per-window scale). Chunk-length inputs make the head's RoPE pass a single pass.
        chunk = max(1, int(round(seg_chunk_cap_s * float(record.pose.fps))))
        _head_forward = lambda p, m, t: model.segment(p, m, t).logits
        
        bio_logits = chunk_normalized_logits(_head_forward, raw, timestamps, chunk, device)
        tags = DurationDecoder(model.duration_model).decode(
            bio_logits, torch.tensor([bio_logits.shape[1]], device=bio_logits.device), 
            timestamps_s=torch.as_tensor(timestamps, device=device)[None]
        )[0].cpu()
        fps = float(record.pose.fps)

        # Deployed span floor, offline too: Λ_min is POLICY (span_selection — a sub-floor span is unresolvable from boundary evidence and FSM 
        # never emits one), not streaming machinery. Without it this row emits flicker spans the deployed system cannot produce, each charged 
        # by SODA as a spurious prediction plus a junk translation — so (10 − 6) would partly measure the missing floor, not streaming.
        # It is a minimum proposal duration for event generation, in frames as the FSM tests it; it does not filter saved predictions.
        # No buffer cap: the buffer is streaming machinery, so a span of any length stays whole here (the engine's cap cut is online only).
        floor = seconds_to_frames(min_span_s, fps)
        n_sub_lambda += sum(term - start < floor for start, term in bio_complete_spans(tags))
        times = timestamps.tolist()
        segments = [Segment(times[start], times[end]) for start, end in candidate_spans(tags, floor)]

        # Load and translate in batches (1 window/span), the same inversion run_boundary_sweep uses.
        bounds, frontiers, anchors, results = [], [], [], []
        pending: list = []
        delta_lead_s = seconds_to_frames(inference_cfg["boundary_stability"]["delta_s"], record.pose.fps) / float(record.pose.fps)

        def _drain():
            if not pending: return
            batch, done = list(pending), len(results)  # hand over a copy: the batch is released when it returns
            pending.clear()
            results.extend(translate_windows(  # windows are extended by delta_lead_s, so frame 0 precedes the sentence
                model, tokenizer, method, batch, device, inference_cfg, method_cfg, batch_size=batch_size, 
                stream_start=False, commit_frontier_s=frontiers[done:], anchor_frames=anchors[done:],
            ))

        for span in segments: # The whole proposal + its terminator frame, at any length.
            w_start = max(0.0, float(span.start_s) - delta_lead_s)
            w_end = min(float(record.pose.duration_s), float(span.end_s) + 1.0 / float(record.pose.fps))
            span_poses, span_ts = box_normalized_window(record.pose, w_start, w_end, w_start, min(float(record.pose.duration_s), w_end + tail_s))
            if span_poses.shape[0] == 0: continue
            # ponytail: a window longer than the head's chunk translates in a batch of its own, so no other row pads to its length
            # (memory only: each row's result matches translating it alone); one multi-minute window still needs its own memory.
            alone = w_end - w_start > seg_chunk_cap_s
            if alone: _drain()
            pending.append((span_poses, span_ts, w_start, float(record.pose.fps)))
            bounds.append((float(span.start_s), float(span.end_s)))
            # χ for this window: the δ lead is the PREDECESSOR's tail — already handled, exactly the FSM's post-commit leftover. 
            # Without it, a back-to-back lead is mid-signing, the merged I-run cannot open a span, and Ω anchors on NEXT sentence.
            frontiers.append(max(0.0, float(span.start_s) - w_start))

            # Ω anchors on the emitted span itself: this row's gate and its scored event are one span by construction.
            n_frames = int(span_poses.shape[0])
            slack = FRAME_SLACK / float(record.pose.fps)   # a saved event time maps back to its own frame
            s_idx = min(int(np.searchsorted(span_ts, float(span.start_s) - slack)), n_frames - 1)
            e_idx = int(np.searchsorted(span_ts, float(span.end_s) - slack))
            anchors.append((s_idx, e_idx if e_idx < n_frames else -1))
            if alone or len(pending) >= batch_size: _drain()
        _drain()
        predicted[record.video_id] = translated_events(record.video_id, [Segment(s0, s1) for s0, s1 in bounds], results)
    if n_sub_lambda and progress: print(f"[offline] dropped {n_sub_lambda} sub-Λ_min span(s) (< {min_span_s:g} s in frames of the "
                                        f"stream) — parity with FSM's span selection, which can never commit them.", flush=True)
    return predicted


def dev_monitor_scores(raw: dict, records: list[VideoRecord], edge_cfg: dict, char_level: bool, name: str) -> dict[str, float]:
    """A stage-2 checkpoint monitor's numbers (evaluation.online.online_dev_scores, offline_dev_scores) from the source-time 
    events of whole dev videos, each committed span one caption (never cut), scored with the main scores (`metrics.edge_chunk_scores`; `edge_cfg` = eval.yaml scoring chunk_gap_s
    and edge_taus_s): `{name}_score` = edge chunk F1 avg (`{name}_f1`) x the DVC BLEU-4 of its pairs, averaged over the taus
    (`{name}_bleu4`), and events per gold unit. Only BLEU-4 is computed here."""
    gold = gold_events(records)
    pred = scoreable_predictions(raw, records, tag=f"val-{name}")
    taus = [float(t) for t in edge_cfg["edge_taus_s"]]
    edge = edge_chunk_scores(pred, gold_chunks(records, float(edge_cfg["chunk_gap_s"])), taus, char_level, bleurt_checkpoint=None, keys=("bleu4",))
    f1, bleu = float(edge["f1_avg"]), float(edge["dvc_avg"]["bleu4"]) if edge["dvc_avg"] else 0.0
    return {
        f"{name}_score": f1 * bleu, f"{name}_f1": f1, f"{name}_bleu4": bleu,
        f"{name}_events_per_gold": sum(map(len, pred.values())) / max(1, sum(map(len, gold.values()))),
    }


@torch.no_grad()
def offline_dev_scores(model, records, inference_cfg: dict, method_cfg: dict, edge_cfg: dict, char_level: bool) -> dict[str, float]:
    """Checkpoint selection on the offline row (train/slt.py evaluate_slt, checkpoint.monitor val_offline_score): `offline_predictions` 
    over `records` (whole source videos), scored with the main scores (`dev_monitor_scores`; `edge_cfg` = eval.yaml scoring
    chunk_gap_s and edge_taus_s); `offline_score` = edge chunk F1 avg x the BLEU-4 of its matched pairs. Each video is 
    segmented once and each proposal translated once, where the online monitor (evaluation.online.online_dev_scores) segments the buffer 
    on every stride. Each rank reads every world_size-th record and the events are merged, so every rank returns the same scores."""
    started = time.perf_counter()
    device = next(model.parameters()).device
    batch_size = dist.per_rank_batch_size(int(method_cfg["checkpoint"]["dev_batch_size"]))   # the global batch of every dev evaluation
    # Full precision, as `eval.py --offline` runs: the training loop calls its monitor under CUDA autocast.
    with torch.autocast(device_type="cuda", enabled=False) if torch.cuda.is_available() else nullcontext():
        local = offline_predictions(
            model, model.tokenizer, model.decoder_type, records[dist.rank()::dist.world_size()], inference_cfg, method_cfg, 
            device, batch_size, progress=False,
        )
    scores = dev_monitor_scores(to_source(dist.merge_dicts(local), records), records, edge_cfg, char_level, "offline")
    if dist.is_main(): print(f"slt | offline monitor: {len(records)} streams in {time.perf_counter() - started:.0f}s", flush=True)
    return scores
