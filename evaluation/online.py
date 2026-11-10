"""The online rows (eval.py --online) and the stage-2 online monitor.

Streaming FSM (infer/stream.py) runs over each stream, with the joint arm or an online segmenter cascade, and writes events and the
display replay of --stability (outputs/stability/). The online
monitor (train/slt.py, checkpoint.monitor val_online_score) runs the same FSM over whole dev videos and scores it as a row.
"""
from __future__ import annotations
from contextlib import nullcontext
from tqdm import tqdm
from types import SimpleNamespace
import argparse, json, os, queue as queue_lib, sys, time, traceback, zlib
import numpy as np

import torch
from poses import load_pose_window
from data.loader import annotation_fingerprint, ANNOTATION_PROTOCOL, load_language_records
from data.windowing import BIO, make_bio_labels
from infer.duration_decode import DurationModel
from infer.stability import build_policies, group_tracks, TAU_GRID
from infer import stream
from infer.stream import MoryossefRunnerAdapter, S1RunnerAdapter
from models.bio_head import require_band_reach
from train import distributed as dist

from metrics import frame_scores
from evaluation.predictions import corpus_token, prediction_events, PredictionEvent, to_source
from evaluation.systems import (
    SEGMENTERS, FLOP_RULE, FlopLedger, build_eval_model, build_streaming_runner, count_flops, device_name, load_segmenter, method_config_path, model_annotation,
    parse_grid,
    source_language, translator_token, translator_decoder
)
from evaluation.offline import dev_monitor_scores
from evaluation.segmenter_eval import arm_decode, segmenter_decode, stamped_decode
from models.checkpointing import load_checkpoint_meta
from train.helpers import segmenter_config_decode
from metrics import accumulate_policy, merged_policy, score_policy
from utils import load_yaml, output_dir, pick_device, UNIT_SEPARATOR


def _track_units(tracks, events, sentences, cover: float = 0.5) -> dict[int, list[tuple[float, float, int]]]:
    """{track index: [(start_s, end_s, reference words) of each gold unit it holds, in time order]} for the stability LAAL, on the
    stream's clock. A committed track's span runs from its start to the end of the event it emitted. It holds a reliable gold unit
    when it covers at least `cover` of it (metrics.held_unit_latency's rule), and each unit goes to the first committing track that
    holds it. A track that holds several units (a merge) gives each of them its LAAL, as metrics.long_form_scores scores a caption."""
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
    return {i: [(g.start_s, g.end_s, len(g.text.split())) for g in gs] for i, gs in held.items()}


def _online_system(args: argparse.Namespace) -> SimpleNamespace:
    """The models and the FSM runner of an online row, built once per process (each process of --workers builds its own), and the
    row's provenance on `run_online` (its last_* attributes)."""
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
        # The decode that chose the checkpoint's epoch (an untrained control has no file; it decodes as its config names).
        default_decode = stamped_decode(load_checkpoint_meta(segmenter_checkpoint)) \
                         if getattr(args, "segmenter_init", "finetuned") == "finetuned" else decode
        model.duration_model = DurationModel.for_language(data_cfg, source) if decode == "duration" else None
        if decode == "viterbi": print("[online] viterbi decode: legal BIO paths only, no duration scores.", flush=True)
        if decode == "argmax": print("[online] argmax decode: each frame its most likely tag, no legal BIO path and no duration "
                                     "scores; the FSM's span, vote, cap and commit rules are the same.", flush=True)
    else:
        method_cfg = load_yaml(method_config_path(args, args.spans), language=source)
        if getattr(args, "num_beams", None): method_cfg.setdefault("validation", {})["num_beams"] = int(args.num_beams)
        if translate and int(method_cfg["validation"]["num_beams"]) != 1:
            raise ValueError("The online rows (the cascade and the arms) use greedy decoding (num_beams=1).")
        device = pick_device(args.device)
        model, tokenizer = build_eval_model(args.spans, args.arm_checkpoint, source, data_cfg, method_cfg, inference_cfg, device)
        run_online.last_arm_checkpoint = getattr(model, "checkpoint_path", None)
        run_online.last_model_annotation = model_annotation(model, data_cfg, source)
        # The arm's own head under the decode of its rows (evaluation.segmenter_eval.arm_decode); the duration decode reads the duration
        # model it trained with, and the FSM gives Ω the span it selects, so argmax and viterbi read no prior.
        decode, default_decode = arm_decode(args, model, method_cfg), stamped_decode(getattr(model, "checkpoint_meta", None))
        if decode != "duration": model.duration_model = None

    run_online.last_decode = decode   # the {decode} field of the row's name (evaluation.rows)
    run_online.last_duration_model = model.duration_model.to_dict() if model.duration_model is not None else None
    run_online.last_segmentation_decode = "bio_argmax" if decode == "argmax" else \
        "semi_markov_viterbi" if model.duration_model is not None else "legal_viterbi"
    run_online.last_velocity = bool(velocity) if segmenter_arch == "moryossef" else False
    if not translate: print("[online] Segmentation only: no decode, events carry no text.", flush=True)
    runner = build_streaming_runner(
        model, inference_cfg, method_cfg, translate=translate, cascade_model=cascade_model,
        record_trace=bool(getattr(args, "stability", False)), argmax=decode == "argmax", trace_batch=int(getattr(args, "batch_size", None) or 1),
    )
    # --count-flops: the shapes of the plain pass's model calls (evaluation.systems.FlopLedger), built before the first pass. A row whose FSM
    # translates with a DLM counts in a pass of its own (_online_record), since its block decode logs no call: the same FLOPs, for more time.
    counts = bool(getattr(args, "count_flops", False))
    dlm = translate and getattr(cascade_model if cascade_model is not None else model, "decoder_type", None) == "dlm"
    ledger = FlopLedger(runner) if counts and not dlm else None
    # The separator is no displayed word (scored as a space, as in the boundary sweep and the rows); its id comes from tokenizing it 
    # (convert_tokens_to_ids gives UNK). Words start at a "▁" piece, so LAAL counts words as the references do.
    separator_id = tokenizer(UNIT_SEPARATOR, add_special_tokens=False).input_ids[-1] if callable(tokenizer) else None
    word_start = (lambda toks: [str(p).startswith("▁") for p in tokenizer.convert_ids_to_tokens(toks)]) \
                                                       if hasattr(tokenizer, "convert_ids_to_tokens") else None
    # The frame block of a no-text row (metrics.frame_scores, the key set of the offline rows) at the row's tIoU thresholds; a row with
    # text has none.
    thresholds = None if translate else parse_grid(
        getattr(args, "tiou_thresholds", None), load_yaml(args.eval_config)["scoring"]["tiou_thresholds"])
    # The parameters of the models that the FSM runs (Module.parameters() counts a tied weight once): the segmenter and the online
    # cascade's translator, or the whole arm (its 2 branches and its translator, with or without text).
    nets = {"arm": model} if segmenter_arch is None else {"segmenter": model, "translator": cascade_model}
    params = {name: sum(p.numel() for p in net.parameters()) for name, net in nets.items() if isinstance(net, torch.nn.Module)}
    return SimpleNamespace(
        runner=runner, tokenizer=tokenizer, translate=translate, separator_id=separator_id, word_start=word_start, thresholds=thresholds,
        policies=build_policies(), decode=decode, default_decode=default_decode, segmenter_arch=segmenter_arch, cascade_model=cascade_model,
        device=device, inference_cfg=inference_cfg, data_cfg=data_cfg, source=source, params=params,
        count_flops=counts, ledger=ledger, stability=bool(getattr(args, "stability", False)), traced=None,
    )


def traced_streams(records: list, hours: float | None) -> set[str] | None:
    """The streams of eval.py --stability-hours (None: every stream): a fixed order that only the stream ids set, until the chosen
    streams hold at least `hours` of pose. So every system of a corpus reads the same streams, in any process, and a split of at most
    `hours` is read whole."""
    if hours is None: return None
    streams, left = set(), float(hours) * 3600.0
    for record in sorted(records, key=lambda r: zlib.crc32(r.video_id.encode("utf-8"))):
        if left <= 0.0: break
        streams.add(record.video_id); left -= float(record.pose.duration_s)
    return streams


def _online_record(system: SimpleNamespace, record) -> dict:
    """1 stream of the row through the FSM: its events (stream clock), the stride wall times, with --stability its policy scores
    (`score_policy` per policy) and provisional texts, and with --count-flops the FLOPs of the deployed passes (from the shapes of the
    plain pass, or for a DLM in a pass of their own, so the events, the times and the trace come from the plain pass); a no-text row also
    gives its frame scores (None: no gold signing). Nothing here reads another stream, so a stream gives the same result in any process
    and in any order (--workers)."""
    runner, tokenizer = system.runner, system.tokenizer
    out = {"events": [], "stride_seconds": [], "flops": 0, "policy_rows": None, "provisional": None, "frame_row": None}
    poses, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=False)
    if poses.shape[0] == 0: return out

    # A stream outside --stability-hours runs as in a row with no --stability: the trace never changes an event.
    if system.stability: runner.trace = [] if system.traced is None or record.video_id in system.traced else None
    # The released contract is isotropic and the poses are per axis, so external adapter needs the width/height of the frame 
    # the reader returns for THIS video (moryossef26.dataset.pose_aspect: 16:9 when the video has a size, else None).
    adapter = getattr(runner, "model", None)
    if hasattr(adapter, "aspect"):
        from moryossef26.dataset import pose_aspect
        adapter.aspect = pose_aspect(record.pose)

    poses_t, fps = torch.as_tensor(poses, dtype=torch.float32), float(record.pose.fps)
    events = runner.run(poses_t, fps=fps)
    out["stride_seconds"] = list(runner.stride_seconds)
    if runner.trace:
        tracks = group_tracks(runner.trace)   # the commit gate's own span identity, recorded per hypothesis
        units = _track_units(tracks, events, record.sentences)
        out["policy_rows"] = {name: score_policy(tracks, policy, separator_id=system.separator_id, units=units, word_start=system.word_start)
                              for name, policy in system.policies.items()}
        off = float(record.pose.offset_s)
        out["provisional"] = [{"committed": tr.committed, "hypotheses": [{
            "time_s": float(h.commit_time_s) + off, "span_start_s": float(h.span_start_s) + off,
            "kind": "committed" if h.committed else "open" if h.open_span else "closed",
            "text": tokenizer.decode(h.token_ids.tolist(), skip_special_tokens=True).strip() if tokenizer is not None else "",
        } for h in tr.hyps]} for tr in tracks]

    out["events"] = prediction_events(record.video_id, events, tokenizer if system.translate else None)

    # The frame and run scores of the Moryossef protocol on the runner's stitched tags (the decode's tags, the argmax decode's too,
    # including the committed frontier, as the FSM read them; both key families read them) vs GT captions: latest estimate per frame.
    if system.thresholds is not None and runner._bio_timeline is not None:
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
        if ((gold_t == BIO["B"]) | (gold_t == BIO["I"])).any():
            tags[gold_t == BIO["UNK"]] = BIO["UNK"]
            logits_1hot = torch.nn.functional.one_hot(tags.clamp(min=0), num_classes=4).float().unsqueeze(0) * 10.0
            out["frame_row"] = frame_scores(logits_1hot, logits_1hot, gold_t.unsqueeze(0), system.thresholds)

    # --count-flops: the deployed calls of the plain pass (the trace's decodes are display only). The FLOP counter does not count the
    # decode's numpy path search. A DLM's row runs the FSM's passes once more, untimed and with no trace, under the counter
    # (evaluation.systems.count_flops); that pass's events and counters are not kept. Its encoder calls run unpadded, so it counts the
    # real frames, as the ledger does (the padding serves MPS kernel reuse only); a DLM has no confidence pass. So on MPS that pass
    # compiles kernels for each new length (wall time only), and on CUDA its encoder values equal the plain pass's only up to float
    # rounding (infer.stream.BufferEncoderCache), so a decision at a threshold can differ and its count describe a pass that differs
    # from the scored one by that decision.
    if system.ledger is not None: out["flops"] = system.ledger.total()
    elif system.count_flops:
        trace, stats, trace_s, step = runner.trace, dict(runner.gate_stats), runner.trace_only_s, stream.ENCODER_LENGTH_STEP
        runner.trace, stream.ENCODER_LENGTH_STEP = None, 1
        try: out["flops"] = count_flops(lambda: runner.run(poses_t, fps=fps))
        finally: runner.trace, runner.gate_stats, runner.trace_only_s, stream.ENCODER_LENGTH_STEP = trace, stats, trace_s, step
    return out


def _shards(records: list, n: int) -> list[list[int]]:
    # The record indices of each process of --workers: the longest streams first, each to the shard with the least pose time so far.
    shards, load = [[] for _ in range(n)], [0.0] * n
    for i in sorted(range(len(records)), key=lambda i: -float(records[i].pose.duration_s)):
        k = load.index(min(load))
        shards[k].append(i); load[k] += float(records[i].pose.duration_s)
    return [sorted(shard) for shard in shards]


@torch.no_grad()
def _online_worker(args: argparse.Namespace, indices: list[int], slot: int, queue) -> None:
    # 1 spawned process of --workers: its own models and runner, the streams `indices` of the row's records; the results go back to
    # the row's process, which prints the row's setup and report once (this process prints nothing). On a host with several CUDA
    # devices and no --device, process k takes device k (mod the device count).
    try:
        sys.stdout = open(os.devnull, "w")
        if args.device is None and torch.cuda.device_count() > 1: args.device = f"cuda:{slot % torch.cuda.device_count()}"
        system = _online_system(args)
        records, _ = load_language_records(system.data_cfg, args.language, split=args.split)
        system.traced = traced_streams(records, getattr(args, "stability_hours", None))
        queue.put(("ok", slot, {i: _online_record(system, records[i]) for i in indices}, dict(system.runner.gate_stats),
                   float(getattr(system.runner, "trace_only_s", 0.0))))
    except BaseException: queue.put(("error", slot, traceback.format_exc(), None, None))


def _online_results(args: argparse.Namespace, system: SimpleNamespace, records: list) -> tuple[list[dict], dict, float]:
    """Every stream's `_online_record` result in the order of `records`, the FSM counters summed over the streams, and the seconds of
    trace-only decodes. --workers N runs the streams in N processes: this one and N - 1 spawned ones, each with its own models. The FSM
    keeps no state from one stream to the next, so each stream's result is the one of a 1-process row, and the row merges them in
    the order of `records`: its events, scores and files equal those of 1 process (the stride wall times differ, as on any run)."""
    runner, n = system.runner, max(1, int(getattr(args, "workers", 1) or 1))
    trace_only = lambda: float(getattr(runner, "trace_only_s", 0.0))   # 0 for a runner that times no trace (a test stand-in)
    if n == 1: return [_online_record(system, r) for r in tqdm(records, desc="Processing records")], dict(runner.gate_stats), trace_only()
    shards = _shards(records, n)
    context = torch.multiprocessing.get_context("spawn")
    queue = context.Queue()
    workers = {slot: context.Process(target=_online_worker, args=(args, shard, slot, queue), daemon=True)
               for slot, shard in enumerate(shards[1:], 1) if shard}
    for w in workers.values(): w.start()
    try:
        results = {i: _online_record(system, records[i]) for i in tqdm(shards[0], desc=f"Processing records (process 1 of {n})")}
        stats, trace_s, pending = [dict(runner.gate_stats)], trace_only(), set(workers)
        while pending:
            try: status, slot, part, gate_stats, seconds = queue.get(timeout=30)
            except queue_lib.Empty:
                dead = [slot for slot in pending if not workers[slot].is_alive()]
                if dead: raise RuntimeError(f"--workers: process {dead[0] + 1} of {n} stopped with exit code {workers[dead[0]].exitcode} "
                                            f"and sent no result (out of memory?): run the row again with fewer workers.")
                continue
            if status != "ok": raise RuntimeError(f"--workers: process {slot + 1} of {n} failed:\n{part}")
            results.update(part); stats.append(gate_stats); trace_s += seconds; pending.discard(slot)
    except BaseException:
        for w in workers.values(): w.terminate()
        raise
    for w in workers.values(): w.join()
    gate: dict[str, int] = {}
    for counts in stats:
        for key, value in counts.items(): gate[key] = gate.get(key, 0) + int(value)
    return [results[i] for i in range(len(records))], gate, trace_s


@torch.no_grad()
def run_online(args: argparse.Namespace) -> dict[str, list[PredictionEvent]]:
    """Drive the sawtooth FSM end-to-end over each stream (a clean segment, from t = 0) → committed events on the source timeline.

    This is the *usable inference engine* of the online rows: a BIO head + commit gate, recompute-each-stride, no cross-stride decoder
    state. --spans ar|dlm: the arm's own head, and with --text the same arm its own translator. --spans moryossef|s1: the standalone
    segmenter under the same FSM (the online cascade), and with --text pretrimmed the pretrimmed translator on each proposed raw crop at commit.
    --text none: the spans alone, and the frame and run scores of the Moryossef protocol on the FSM's stitched tags (`run_online.last_frame`,
    the key set of the offline rows); evaluation.rows translates the spans for any other text (evaluation.offline.run_cascade).
    --workers N runs the streams in N processes (`_online_results`), with the same result.
    """
    system = _online_system(args)
    runner, decode, default_decode = system.runner, system.decode, system.default_decode
    segmenter_arch, cascade_model, source = system.segmenter_arch, system.cascade_model, system.source
    records, _ = load_language_records(system.data_cfg, args.language, split=args.split)
    system.traced = traced_streams(records, getattr(args, "stability_hours", None))
    # The runner refuses each stream at a rate where the head reads past the commit horizon (spec §1.4 rule 16) when it starts it:
    # every stream is checked here first, so such a stream stops the row at its start, not hours into it.
    if runner.band_reach is not None:
        for record in records: require_band_reach(runner.band_reach, float(record.pose.fps), (runner.hysteresis_strides - 1) * runner.stride_s)
    results, gate_stats, trace_only_s = _online_results(args, system, records)
    # --stability: every recorded decode as text, one entry per sentence track, per SOURCE video in source time: the provisional
    # text a live display would have had on each stride (the policies above choose what of it to show). Strings, not tensors.
    provisional_rows: dict[str, list[dict]] = {}
    # Scored per video and accumulated as (sum, count), not held as Track objects: a Track carries one cloned token
    # tensor per stride, so keeping the whole split pinned every hypothesis of every video (ase is 11,939 of them).
    # Every field score_policy returns is an unweighted mean over same track set, so a count-weighted merge is exact.
    stability_totals: dict[str, dict[str, float]] = {}
    predicted: dict[str, list[PredictionEvent]] = {}
    frame_rows: list[dict[str, float]] = []
    stride_seconds: list[tuple[float, float]] = []
    flops = 0
    for record, result in zip(records, results):   # in record order, as 1 process accumulates them
        predicted[record.video_id] = result["events"]
        flops += result["flops"]
        stride_seconds += result["stride_seconds"]
        for name, row in (result["policy_rows"] or {}).items(): accumulate_policy(stability_totals, name, row)
        if result["provisional"] is not None: provisional_rows.setdefault(record.pose.video_id, []).extend(result["provisional"])
        if result["frame_row"] is not None: frame_rows.append(result["frame_row"])

    # Real time: wall time per stride against the buffer length, on this device (ReSCUE App. G reports the mean, the p95 and 
    # the longest-buffer step). Live use needs each stride done within stride_s, the long buffers near the cap included.
    buffer_s, wall_s  = np.array([b for b, _ in stride_seconds]), np.array([w for _, w in stride_seconds])
    stride_s, cap_s = float(system.inference_cfg["stride_s"]), float(system.inference_cfg["buffer_cap_s"])
    long_s = wall_s[buffer_s >= 0.75 * cap_s]   # the long buffers: the last quarter of the cap
    pose_s = float(sum(float(r.pose.duration_s) for r in records))
    traced = records if system.traced is None else [r for r in records if r.video_id in system.traced]
    traced_s = float(sum(float(r.pose.duration_s) for r in traced))
    run_online.last_compute = {
        # What the numbers cover: the FSM's passes, with the translator's decodes when the FSM translates (an arm's own text, or the
        # online cascade's pretrimmed translator); a row whose text comes from run_cascade times that translation apart
        # (provenance translation_compute).
        "covers": "segmentation and translation" if system.translate else "segmentation",
        "device": str(system.device), "device_name": device_name(system.device),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "workers": max(1, int(getattr(args, "workers", 1) or 1)), "strides": int(wall_s.size),
        # Totals for a real-time factor: wall seconds of every stride of every stream, over the pose seconds of the streams; under 1,
        # a stream is processed faster than it plays. With --workers N on 1 device, each process's stride times include its waits for
        # the others, so they are slower than 1 process alone: time a row for these numbers with --workers 1.
        "pose_s": pose_s, "total_s": float(wall_s.sum()), "rtf": float(wall_s.sum()) / pose_s if pose_s else None,
        "mean_s": float(wall_s.mean()) if wall_s.size else None,
        "p50_s": float(np.median(wall_s)) if wall_s.size else None,
        "p95_s": float(np.percentile(wall_s, 95)) if wall_s.size else None, "max_s": float(wall_s.max()) if wall_s.size else None,
        "long_buffer_s": 0.75 * cap_s, "p95_s_long_buffer": float(np.percentile(long_s, 95)) if long_s.size else None,
        "share_within_stride": float((wall_s <= stride_s).mean()) if wall_s.size else None,
        # The stride times above exclude the decodes that only --stability needs (StreamingSLTRunner.trace_only_s), so a traced row
        # still stamps the cost of the plain pass (with the cache padding and the AR confidence pass, which the GFLOPs leave out);
        # the trace's own cost is reported apart.
        # They run trace_batch per batch, so trace_only_decode_s is the cost of that batching, not of a decode alone on each stride.
        "record_trace": system.stability, "trace_only_decodes": int(gate_stats.get("trace_only_decoded", 0)),
        "trace_only_decode_s": float(trace_only_s), "trace_batch": int(getattr(runner, "trace_batch", 1)),
        "trace_streams": len(traced) if system.stability else 0, "trace_pose_s": traced_s if system.stability else 0.0,
        # The FSM's counters over every stream, saved so a reader can report them: committed, forced_commit (complete spans committed
        # before K votes), open_span_kept (open spans that the cap kept for the next buffer), open_span_cut (open spans that the cap cut:
        # 1 event each, flagged cap_cut and not counted in committed, so the events are committed + open_span_cut, spec §7.1;
        # open_span_cut_kept_before the cuts that only the keep-once rule causes: a span kept before that now starts more than δ after
        # the buffer start; a kept span whose start stays on its frame, exactly δ after it, is not counted) and open_span_dropped (open
        # spans that the cap or the stream end dropped: no event; open_span_dropped_at_cap the cap's share, a cut under Λ_min, and 
        # open_span_dropped_kept_before the drops that only the keep-once rule causes, counted the same way). They count passes, not units: a unit that the cap 
        # keeps twice, or keeps and then cuts, counts in each.
        "gate_stats": dict(gate_stats),
        # Hardware-free: the parameters of the models it runs, the FLOPs of its deployed passes (--count-flops: evaluation.systems.FlopLedger,
        # or a pass of their own for a DLM; the same count on every device), and the design's latency floor: a span
        # commits only after its terminator holds for K strides, so no unit commits earlier than (K - 1) strides after its terminator
        # frame, whatever the compute (spec §7.3).
        "params": system.params,
        "gflop": flops / 1e9 if system.count_flops else None,
        "gflop_per_video_s": flops / 1e9 / pose_s if system.count_flops and pose_s else None,
        "flop_rule": FLOP_RULE if system.count_flops else None,
        "commit_floor_s": (int(system.inference_cfg["boundary_stability"]["hysteresis_strides"]) - 1) * stride_s,
    }
    print(f"[online] compute: {run_online.last_compute}", flush=True)
    # Why-did-it-(not)-commit: low streaming recall with near-perfect frame BIO = the boundary-stability gate suppressed emission.
    s = gate_stats
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
        hours = getattr(args, "stability_hours", None)
        if hours is not None: print(f"[stability] the trace reads {len(traced)} of {len(records)} streams ({traced_s / 3600:.2f} of "
                                    f"{pose_s / 3600:.2f} h of pose); the row's captions and scores read every stream", flush=True)
        # ANCHORS DIFFER, deliberately. The held-unit latency (metrics.held_unit_latency, report.py results tables/held_unit_latency.csv)
        # = commit time - GOLD UNIT END, when the SCORED translation is complete, under commit_only, the only policy the FSM runs.
        # `latency_s` here is first reveal - COMMIT: how much earlier a policy would put text on screen. Keep them separate: quoting a
        # prefix's latency against the full sentence's BLEU would describe 2 different objects. LAAL_s is gold-anchored (unit start).
        print("latency_s is relative to the COMMIT (negative = earlier); the held-unit latency from the gold unit end is in report.py "
              "results (tables/held_unit_latency.csv). NaN: no data.", flush=True)
        print(f"{'policy':<30}{'latency_s':>11}{'revealed':>10}{'frozen_err':>12}{'rewrite_rate':>14}{'NE':>8}{'LAAL_s':>9}", flush=True)
        for name, r in rows.items(): print(
            f"{name:<30}{r['first_token_latency_s']:>11.3f}{r['revealed_fraction']:>10.3f}{r['frozen_prefix_error']:>12.3f}"
            f"{r['contradicted_track_rate']:>14.3f}{r['normalized_erasure']:>8.3f}{r['laal_s']:>9.3f}", flush=True
        )
        r = rows["commit_only"]
        print(f"[stability] LAAL: the mean over {r['n_laal_units']} gold units held by the {r['n_tracks']} committed sentences (a sentence "
              f"holds a unit when it covers at least half of it; a merge gives each unit its LAAL); boundary flips per sentence "
              f"{r['boundary_flip_rate']:.3f}", flush=True)
        # An online cascade is named after its segmenter, with -dlm when it translates with a pretrimmed DLM translator (its AR names stay),
        # and a decode other than the one that chose the checkpoint's epoch (--decode) adds its name, so it never replaces the default's file.
        name = (translator_token(args, args.spans, run_online.last_arm_checkpoint) if segmenter_arch is None else segmenter_arch + (
            "-dlm" if cascade_model is not None and translator_decoder("baseline", run_online.last_translation_checkpoint) == "dlm" else "")
                ) + ("" if decode == default_decode else f"-{decode}") + ("" if hours is None else f"-{hours:g}h")
        out = output_dir("stability") / f"stability_{name}_{corpus_token(args.language, source)}_{args.split}.json"
        out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # The texts themselves, beside the scores: per sentence track, each recorded decode with its stride time and kind (open: the
        # unit before its end appeared; closed: the span waiting for its votes; committed: the emitted text). Never read by scoring.
        texts = out.with_name(out.name.replace("stability_", "provisional_", 1))
        texts.write_text(json.dumps({"provenance": {
            "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(records), 
            "language": args.language, "source_language": source, "split": args.split, "hysteresis_strides": runner.hysteresis_strides,
            "unit_separator": UNIT_SEPARATOR, "delta_s": runner.delta_s, "stride_s": runner.stride_s, "stability_hours": hours,
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
    over `records` (whole source videos), scored as a row's main score (`dev_monitor_scores`; `edge_cfg` = eval.yaml scoring chunk_gap_s
    and tiou_thresholds). Each rank streams every world_size-th record and the events are merged, so every rank returns the same
    scores. `online_score` = the main score (the F1 of the tIoU match at tIoU avg) x the DVC BLEU-4 of the pairs that count (the same
    avg). The FSM decodes as `method_cfg` names (segmenter_decode, as the offline monitor; argmax, or the duration decode under the
    model's duration model, which training keeps for its gate)."""
    started = time.perf_counter()
    runner = build_streaming_runner(model, inference_cfg, method_cfg, argmax=segmenter_config_decode(method_cfg) == "argmax")

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
