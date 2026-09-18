from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from statistics import median
from itertools import islice
from pathlib import Path
import json, argparse, math, os

import torch
import numpy as np
from tqdm import tqdm
from data.loader import ANNOTATION_PROTOCOL, annotation_fingerprint, load_language_records
from data.windowing import make_bio_labels, TRUSTED_GAP_S
from poses import normalize_keypoints_unisign, load_pose_window

from train import distributed as dist
from train.slt import build_slt_components, training_loss
from train.helpers import AmpHelper, move_to_device
from models.checkpointing import load_checkpoint_meta, load_model_checkpoint

from moryossef26.infer import predict_phrase_segments, whole_video_logits
from infer.commit_gate import bio_complete_spans, select_target_span
from infer.duration_decode import DurationModel, DurationDecoder
from infer.stream import S1RunnerAdapter, StreamingSLTRunner

from metrics import Segment, match_segments, moryossef_segment_metrics
from eval import (
    PredictionEvent, _gold_events, _load_segmenter, evaluate_predicted_events,
    load_prediction_file, require_annotation_match, save_prediction_file, scoreable_predictions
)
from utils import checkpoint_dir, lambda_min_frames, load_yaml, pick_device, pool_key, resolve_inference, update_yaml_scalar

# Low on purpose: near-misses feed the (Δ_head, Δ_tail) jitter CDF as matched pairs, not phantom/skip events;
# a high bar biases the CDF to zero. Override: --tiou-threshold.
SEGMENTER_ERROR_MATCH_TIOU = 0.1


def task_gradient_stats(bio_loss, translation_loss, parameters):
    # Unweighted task geometry on coordinates used by both losses; no optimizer update.
    params = list(dict.fromkeys(p for p in parameters if p.requires_grad))
    if not params or not bio_loss.requires_grad or not translation_loss.requires_grad: return None
    a = torch.autograd.grad(bio_loss, params, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(translation_loss, params, allow_unused=True)
    pairs = [(x.float(), y.float()) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs: return None
    aa = sum(float(x.square().sum()) for x, _ in pairs)
    bb = sum(float(y.square().sum()) for _, y in pairs)
    ab = sum(float((x * y).sum()) for x, y in pairs)
    if not all(math.isfinite(v) for v in (aa, bb, ab)):
        raise ValueError("Non-finite task gradients; no loss-weight recommendation can be made")
    if aa <= 0 or bb <= 0: return None  # Inactive tasks have no defined cosine or balance ratio.
    return {"bio_norm": math.sqrt(aa), "translation_norm": math.sqrt(bb), "cosine": max(-1., min(1., ab / math.sqrt(aa * bb)))}


def loss_balance(args): # Estimate an initial gradient-scale ratio from 16 train batches, then leave training unchanged.
    if args.split != 'train': raise ValueError("loss-balance uses the training split only")
    if args.checkpoint: raise ValueError("loss-balance uses the configured training initialization, not a selected final checkpoint")
    if args.write_config: raise ValueError("loss-balance reports a proposed ratio; copy it into the run config after review")
    if dist.is_distributed() or int(os.environ.get('WORLD_SIZE', '1')) > 1: raise ValueError("Run initial loss calibration on 1 process")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    c = build_slt_components(args.data_config, args.slt_config, args.inference_config, language=args.language, bio_config=args.bio_config)
    if min(float(c.slt_cfg.get(k, 1.)) for k in ('lambda_bio', 'lambda_trans')) <= 0:
        raise ValueError("Loss balance requires both tasks to be enabled")
    
    device = pick_device(args.device)
    model = c.model.to(device).train()
    params = list(model.front_end.pose_encoder.parameters()) + list(model.bio_head.parameters())
    amp = AmpHelper.from_config(c.slt_cfg, device)
    # Measure the intended joint phase, including gate/CB after any configured warmup.
    epoch = 1 + max(int(c.slt_cfg.get('membership_gate', {}).get('warmup_epochs', 0)),
                    int(c.slt_cfg.get('confidence_bound', {}).get('warmup_epochs', 1)))
    
    rows, inactive = [], 0
    for batch in tqdm(islice(c.train_loader, 16), total=min(16, len(c.train_loader)), desc='[loss-balance]'):
        with amp.autocast(): output = training_loss(model, move_to_device(batch, device), c.slt_cfg, epoch)
        scale = float(amp.scaler.get_scale()) if amp.scaler.is_enabled() else 1.
        row = task_gradient_stats(output.bio_loss * scale, output.translation_loss * scale, params)
        if row is None: inactive += 1; continue
        row['bio_norm'] /= scale; row['translation_norm'] /= scale
        rows.append(row)

    if len(rows) < 8: raise ValueError(f"Only {len(rows)} jointly active batches; at least eight are required for this calibration")
    ratio = math.exp(median([math.log(r['translation_norm'] / r['bio_norm']) for r in rows]))
    payload = {
        "language": args.language, "decoder": c.slt_cfg['decoder'], "split": 'train',
        "annotation_fingerprint": c.checkpoint_meta['annotation_fingerprint'],
        "active_batches": len(rows), "inactive_batches": inactive, "raw_ratio": ratio,
        "suggested": {"lambda_bio": ratio, "lambda_trans": 1.0} if .1 <= ratio <= 10. else None,
        "ratio_in_review_range": .1 <= ratio <= 10., "batches": rows, "interpretation": "Initial shared-gradient scale calibration"
    }
    path = Path(args.output or f"outputs/loss_balance_{c.slt_cfg['decoder']}_{args.language}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return payload


@dataclass(frozen=True)
class JitterSample:
    video_id: str
    pred_index: int
    gold_index: int
    tiou: float
    delta_head_s: float
    delta_tail_s: float

@dataclass(frozen=True)
class SegmenterErrorAnalysis:
    jitter_samples: list[JitterSample]
    mode_ratios: dict[str, float]
    event_counts: dict[str, int]
    matched_pairs: int
    regular_matches: int
    videos: int
    # Position in (0,1) of each spurious cut in an over-segmented GT sentence = Mode-2 truncation depth, 
    # not (Δ_head, Δ_tail) jitter. The Mode-2 window sampler draws cut depth from this; uniform if empty.
    overseg_cut_positions: list[float]

def _laplace_fit(values: list[float]) -> dict[str, float]:
    if not values: return {"loc": 0.0, "scale": 0.0}
    loc = float(median(values))
    scale = sum(abs(v - loc) for v in values) / max(1, len(values))
    return {"loc": loc, "scale": float(scale)}

def normalize_counts(counts: dict[str, int | float]) -> dict[str, float]:
    total = float(sum(max(0, v) for v in counts.values()))
    if total <= 0: return {"mode1": 1.0, "mode2": 0.0, "mode3": 0.0, "mode4": 0.0}
    return {key: float(max(0, value) / total) for key, value in counts.items()}

def mode_weights_from_events(counts: dict[str, int]) -> dict[str, float]:
    skipped_half = 0.5 * float(counts["skipped"]) # Skip mass splits between truncated-window and multi-complete-window cases.
    return {
        "mode1": float(counts["matched"]),
        "mode2": float(counts["oversegmentation"]) + skipped_half,
        "mode3": float(counts["undersegmentation"]) + skipped_half,
        "mode4": float(counts["phantom"]),
    }

def analyze_segmenter_errors(
    predicted: dict[str, list[Segment]], gold: dict[str, list[Segment]],
    durations: dict[str, float], material_overlap_s: float = 0.0, tiou_threshold: float = 0.1
) -> SegmenterErrorAnalysis: # Segmenter error counts and regular-match jitter samples.

    # `material_overlap_s`: a pred counts as covering a gold (for over/under-segmentation tests) only when their overlap exceeds this.
    # 0 = any-overlap, which DOUBLE-COUNTS small boundary offsets on back-to-back corpora: a span grazing its neighbour by a fraction of
    # a second is boundary JITTER (the Laplace's job), yet any-overlap reclassifies the event as under-segmentation, so stage 2 trained
    # on a far harsher mix than deployment produces. Principled floor = Λ_min in seconds: a fragment shorter than the minimum selectable
    # span cannot form a second window at deployment, so it cannot make the event multi-sentence.

    # Jitter excludes over-segmented GT and under-segmenting pred spans — separate window modes, not 1-to-1 boundary noise.
    jitter_samples: list[JitterSample] = []
    overseg_cut_positions: list[float] = []
    counts = {"matched": 0, "oversegmentation": 0, "undersegmentation": 0, "skipped": 0, "phantom": 0}
    matched_pairs, regular_matches = 0, 0

    for video_id in sorted(set(gold) | set(predicted)):
        pred_segments = list(predicted.get(video_id, []))
        gold_segments = list(gold.get(video_id, []))
        matches = match_segments(pred_segments, gold_segments, threshold=tiou_threshold)
        matched_pairs += len(matches)
        matched_pred = {pred_idx for pred_idx, _, _ in matches}
        matched_gold = {gold_idx for _, gold_idx, _ in matches}

        overlapping_pred_by_gold: dict[int, list[int]] = {}
        overlapping_gold_by_pred: dict[int, list[int]] = {}
        for pred_idx, pred in enumerate(pred_segments):
            for gold_idx, gt in enumerate(gold_segments):
                overlap = max(0.0, min(pred.end_s, gt.end_s) - max(pred.start_s, gt.start_s))
                if overlap > float(material_overlap_s):
                    overlapping_pred_by_gold.setdefault(gold_idx, []).append(pred_idx)
                    overlapping_gold_by_pred.setdefault(pred_idx, []).append(gold_idx)

        overseg_gold = {gold_idx for gold_idx, pred_indices in overlapping_pred_by_gold.items() if len(pred_indices) >= 2}
        underseg_pred = {pred_idx for pred_idx, gold_indices in overlapping_gold_by_pred.items() if len(gold_indices) >= 2}
        underseg_gold = {gold_idx for pred_idx in underseg_pred for gold_idx in overlapping_gold_by_pred.get(pred_idx, [])}
        # Pred-level categories are made MUTUALLY EXCLUSIVE, or the mode weights double-charge single events: matching uses tIoU with no 
        # absolute-overlap floor while the overlap maps use material_overlap_s, so a 1-to-1 matched pred whose raw overlap is under the 
        # floor would otherwise ALSO count as phantom, and a pred bridging 2 gold sentences (1 under-segmentation event) would otherwise 
        # ALSO be charged as an over-segmentation fragment of each over-segmented gold it touches.
        phantom_pred = {
            pred_idx for pred_idx, pred in enumerate(pred_segments) if pred_idx not in overlapping_gold_by_pred 
            and pred_idx not in matched_pred and pred.start_s >= 0.0 and pred.end_s <= float(durations.get(video_id, pred.end_s))
        }
        counts["oversegmentation"] += sum(sum(1 for pi in overlapping_pred_by_gold[g] if pi not in underseg_pred) for g in overseg_gold)
        counts["undersegmentation"] += len(underseg_pred)
        counts["phantom"] += len(phantom_pred)
        counts["skipped"] += len(set(range(len(gold_segments))) - matched_gold - overseg_gold - underseg_gold)

        # Mode-2 cut depths.
        for gold_idx in overseg_gold:
            gt = gold_segments[gold_idx]
            dur = gt.end_s - gt.start_s
            if dur <= 0: continue
            
            overlapping = sorted((pred_segments[pi] for pi in overlapping_pred_by_gold[gold_idx]), key=lambda s: s.start_s)
            for left, right in zip(overlapping, overlapping[1:]):
                cut = min(max(0.5 * (left.end_s + right.start_s), gt.start_s), gt.end_s)
                rel = (cut - gt.start_s) / dur
                if 0.0 < rel < 1.0: overseg_cut_positions.append(float(rel))

        for pred_idx, gold_idx, score in matches:
            if gold_idx in overseg_gold or pred_idx in underseg_pred: continue
            pred, gt = pred_segments[pred_idx], gold_segments[gold_idx]
            counts["matched"] += 1
            regular_matches += 1
            jitter_samples.append(JitterSample(
                video_id=video_id, pred_index=pred_idx, gold_index=gold_idx, tiou=float(score),
                delta_head_s=float(pred.start_s - gt.start_s), delta_tail_s=float(pred.end_s - gt.end_s),
            ))
    mode_counts = mode_weights_from_events(counts)
    return SegmenterErrorAnalysis(
        jitter_samples=jitter_samples, mode_ratios=normalize_counts(mode_counts), event_counts=counts,
        matched_pairs=matched_pairs, regular_matches=regular_matches, videos=len(set(gold) | set(predicted)),
        overseg_cut_positions=overseg_cut_positions,
    )


def write_segmenter_error_outputs(
    analysis: SegmenterErrorAnalysis, output_dir: str | Path, language: str, arch: str, split: str = "dev"
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jitter_rows = [asdict(sample) for sample in analysis.jitter_samples]
    head = [row["delta_head_s"] for row in jitter_rows]
    tail = [row["delta_tail_s"] for row in jitter_rows]
    # `split` in the NAME and the PAYLOAD: the dev artifacts feed the measured-jitter ABLATION (training input,
    # dev-only by the no-test-contamination rule); a test run is the REPORTED taxonomy only. Distinct names make
    # the overwrite impossible, and data/jitter.py + the sampler REFUSE a test-measured artifact as training input.
    jitter_payload = {
        "language": language, "split": str(split), "segmenter_arch": arch, "samples": jitter_rows,
        "laplace": {"head": _laplace_fit(head), "tail": _laplace_fit(tail)},
        "overseg_cut_positions": analysis.overseg_cut_positions,
    }
    mode_payload = {
        "language": language, "split": str(split), "segmenter_arch": arch,
        "mode_ratios": analysis.mode_ratios,
        "source_event_counts": analysis.event_counts,
        "source_weights": mode_weights_from_events(analysis.event_counts),
    }
    taxonomy_payload = {
        "language": language, "split": str(split), "segmenter_arch": arch,
        "event_counts": analysis.event_counts,
        "matched_pairs": analysis.matched_pairs,
        "regular_matches": analysis.regular_matches,
        "videos": analysis.videos,
        "moryossef_2020_mapping": {
            "matched": "Started Pre/Post-Signing and Signing Underflow/Overflow",
            "oversegmentation": "Signing Undetected Incorrectly", "undersegmentation": "Bridged",
            "skipped": "Skipped", "phantom": "Signing Detected Incorrectly",
        },
    }
    paths = {
        "jitter": str(output_dir / f"a_jitter_{arch}_{language}_{split}.json"),
        "mode_ratios": str(output_dir / f"a_mode_ratios_{arch}_{language}_{split}.json"),
        "taxonomy": str(output_dir / f"a_error_taxonomy_{arch}_{language}_{split}.json"),
    }
    Path(paths["jitter"]).write_text(json.dumps(jitter_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["mode_ratios"]).write_text(json.dumps(mode_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["taxonomy"]).write_text(json.dumps(taxonomy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def dataset_summary(args: argparse.Namespace) -> dict:
    cfg = load_yaml(args.data_config)
    records, splits = load_language_records(cfg, args.language, split=args.split)
    durations = [span.duration_s for rec in records for span in rec.sentences if getattr(span, "reliable", True)]
    # Λ_min (inference.yaml span_selection) = p1/2 of dev durations × median fps, an order of magnitude above the ≤2δ phantom scale. 
    # Per-corpus/per-fps: on a corpus switch rerun `--stage delta-enc --write-config`, the one writer of
    # span_selection.min_span_frames; dlm.yaml carries no mirror (train/slt._inject_gate_geometry derives it at load).
    p1_s = float(np.percentile(durations, 1)) if durations else 0.0
    median_fps = float(np.median([float(rec.pose.fps) for rec in records])) if records else 0.0
    return {
        "language": args.language, "split": args.split or "all", "records": len(records),
        "sentences": len(durations), "split_sizes": {k: len(v) for k, v in splits.items()},
        "mean_sentence_s": sum(durations) / len(durations) if durations else 0.0,
        "max_sentence_s": max(durations) if durations else 0.0,
        "p1_sentence_s": p1_s, "median_fps": median_fps,
        "suggested_min_span_frames": int(math.ceil(p1_s / 2.0 * median_fps)) if durations else 0,
    }


def segmenter_infer(args: argparse.Namespace) -> dict: # Upstream segmenter for error calibration and the RQ2 cascade.
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run segmenter inference on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    model, device, velocity, rope_chunk_s, checkpoint = _load_segmenter(args)
    decode = args.segmenter_decode or ("duration" if args.segmenter_arch == "s1" else "plain")
    duration = DurationModel.from_config(load_yaml(args.inference_config), args.language, args.segmenter_arch) if decode == "duration" else None
    print(f"[segmenter-infer] {args.segmenter_arch} from {checkpoint} (decode={decode})", flush=True)

    predictions = predict_phrase_segments(model, records, device=device, velocity=velocity, rope_chunk_s=rope_chunk_s, duration=duration)
    output = Path(args.output or f"outputs/segmenter_predictions_{args.segmenter_arch}_{args.language}_{args.split}.json")
    save_prediction_file(predictions, output, provenance={
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(records),
        "segmenter_arch": args.segmenter_arch, "decode": decode, "duration_model": duration.to_dict() if duration else None,
        "segmentation_decode": "semi_markov_viterbi" if duration else "bio_argmax",
        "pose_normalization": "chunk" if args.segmenter_arch == "s1" else "video",
        "checkpoint": checkpoint, "language": args.language, "split": args.split,
    })
    return {
        "language": args.language, "split": args.split, "videos": len(records), "segmenter_arch": args.segmenter_arch, 
        "checkpoint": checkpoint, "predicted_segments": sum(len(v) for v in predictions.values()), "output": str(output),
    }


def _assert_predictions_match_decode(args: argparse.Namespace) -> None:
    # Calibration must describe the same decoder as the reported segmentation results.
    stamped = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    if not isinstance(stamped, dict) or "provenance" not in stamped: return
    prov = stamped["provenance"]
    arch = str(prov.get("segmenter_arch") or args.segmenter_arch)
    expected = args.segmenter_decode or prov.get("decode") or ("duration" if arch == "s1" else "plain")
    if expected not in {"plain", "duration"}: raise ValueError("Regenerate predictions with a current plain or duration decoder")
    if expected == "duration":
        pinned = DurationModel.from_config(load_yaml(args.inference_config), args.language, arch).to_dict()
        if prov.get("duration_model") != pinned: raise ValueError("Regenerate predictions using the pinned duration model before calibration")
    if arch != args.segmenter_arch or prov.get("decode") != expected:
        raise ValueError(f"Regenerate {args.segmenter_arch} predictions with its {expected} BIO decoder before calibration")


def segmenter_errors(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test: raise SystemExit(
        "Segmenter-error analysis on test needs --allow-test. Legitimate: the REPORTED taxonomy (describes the "
        "same split as the results tables). NOT legitimate: feeding a test-measured artifact to the measured-jitter "
        "ablation — that is training-input contamination, and the training side refuses split=test artifacts."
    )
    cfg = load_yaml(args.data_config)
    records, _ = load_language_records(cfg, args.language, split=args.split)
    require_annotation_match(args.predictions, records, "--predictions")

    predictions = load_prediction_file(args.predictions)  # the segmenter-infer output file
    _assert_predictions_match_decode(args)
    gold_segments = {record.video_id: [
        Segment(span.start_s, span.end_s) for span in record.sentences if getattr(span, "reliable", True)
    ] for record in records}

    # Ignore-region, prediction side (mirrors eval._drop_quarantined_predictions): a segmenter span majority-inside
    # a quarantined region would otherwise count as PHANTOM and inflate mode4 in the measured mix.
    zones = {r.video_id: [(sp.start_s, sp.end_s) for sp in r.sentences if not getattr(sp, "reliable", True)] for r in records}
    predictions = {vid: [seg for seg in segs if sum(
        max(0.0, min(seg.end_s, b) - max(seg.start_s, a)) for a, b in zones.get(vid, [])
    ) / max(1e-9, seg.end_s - seg.start_s) <= 0.5] for vid, segs in predictions.items()}
    durations = {record.video_id: float(record.pose.duration_s) for record in records}

    # Convert the frame floor to this corpus's time base.
    median_fps = float(np.median([float(record.pose.fps) for record in records])) if records else 24.0
    lam_s = int(resolve_inference(load_yaml(args.inference_config), args.language)["span_selection"]["min_span_frames"]) / median_fps
    analysis = analyze_segmenter_errors(
        predicted=predictions, gold=gold_segments, durations=durations, material_overlap_s=lam_s,
        tiou_threshold=float(args.tiou_threshold if args.tiou_threshold is not None else SEGMENTER_ERROR_MATCH_TIOU)
    )
    print(f"[segmenter-errors] event taxonomy with material_overlap_s={lam_s:.3f}s (= span_selection.min_span_frames/{median_fps:g}fps): "
          f"a graze shorter than the minimum selectable span is boundary jitter, not a second sentence.", flush=True)
    paths = write_segmenter_error_outputs(analysis, args.output_dir, args.language, args.segmenter_arch, split=args.split)
    return {
        "language": args.language, "split": args.split, "segmenter_arch": args.segmenter_arch,
        "event_counts": analysis.event_counts, "mode_ratios": analysis.mode_ratios,
        "matched_pairs": analysis.matched_pairs, "regular_matches": analysis.regular_matches, "outputs": paths,
    }


def tune_decode(args: argparse.Namespace) -> dict:
    # Fit durations on train; select 2 score weights on dev from 1 forward per video, under SAME statistic the paper reports: F1 averaged over 
    # `eval.yaml rq2.tiou_thresholds`. Selecting on tIoU 0.5 alone rewarded a prior that completes spans at the cost of their edges, which is 
    # the part 0.7 and 0.9 measure.
    if args.split != "dev": raise ValueError("tune-decode selects on dev only")
    data_cfg = load_yaml(args.data_config)
    records, _ = load_language_records(data_cfg, args.language, split="dev")
    records = sorted(records, key=lambda r: r.video_id)
    if args.num_videos and args.num_videos < len(records):
        if args.write_config: raise ValueError("A partial-video smoke run cannot write final decoder settings")
        records = records[:args.num_videos]

    if len(records) < 2: raise ValueError("Decoder selection needs 2 nonempty dev folds")
    train_records, _ = load_language_records(data_cfg, args.language, split="train")
    prior = DurationModel.fit(train_records)
    model, device, velocity, context, checkpoint = _load_segmenter(args)
    
    if load_checkpoint_meta(checkpoint).get("annotation_protocol") != ANNOTATION_PROTOCOL:
        raise ValueError("tune-decode requires segmenter trained on current caption-unit annotations; retrain S1 or external segmenter first.")
    if load_checkpoint_meta(checkpoint).get("decoder") in ("ar", "dlm"):
        raise ValueError("Select duration conditioning on S1 before joint training; only commit lag is re-selected after joint training")
    
    model.eval().to(device)
    thresholds = tuple(float(t) for t in (load_yaml(args.eval_config).get("rq2", {}) or {}).get("tiou_thresholds", [0.5]))
    grid = [replace(prior, completion_bias=float(b), boundary_logit_weight=w) for b in range(7) for w in (.5, 1., 1.5, 2.)]
    scores = [[], []]
    plain, legal = [], []
    with torch.no_grad():
        for index, record in enumerate(tqdm(records, desc="[tune-decode] forward + batched paths")):
            logits, times = whole_video_logits(model, record, device, velocity, context)
            if logits is None: continue
            logits = logits.detach().cpu()
            gold = torch.tensor(make_bio_labels(
                times, record.sentences, 0., record.pose.duration_s, trusted_gap_s=TRUSTED_GAP_S, video_duration_s=record.pose.duration_s
            ))[None]
            ts = torch.tensor(times)[None]
            n = torch.tensor([logits.shape[1]])
            # Reuse the exact same max-product routine for all candidates; no separate grid decoder.
            tags = DurationDecoder(grid).decode(logits.expand(len(grid), -1, -1), n.expand(len(grid)), timestamps_s=ts.expand(len(grid), -1))
            score = lambda t: float(np.mean([
                moryossef_segment_metrics(logits, gold, pred_tags=t, tiou_threshold=th)["phrase_tiou_f1"] for th in thresholds
            ]))
            scores[index % 2].append([score(t[None]) for t in tags])
            plain.append(score(logits.argmax(-1)))
            legal.append(score(DurationDecoder().decode(logits, n)))

    if not all(scores): raise ValueError("Each dev fold needs at least 1 usable video")
    folds = [np.asarray(x).mean(0) for x in scores]
    tie = lambda i: (-abs(grid[i].completion_bias), -abs(grid[i].boundary_logit_weight-1))
    chosen = max(range(len(grid)), key=lambda i: (min(folds[0][i], folds[1][i]), (folds[0][i] + folds[1][i]) / 2, *tie(i)))
    heldout = [float(folds[1-j][max(range(len(grid)), key=lambda i: (folds[j][i], *tie(i)))]) for j in (0,1)]
    selected = grid[chosen]
    payload = {
        "language": args.language, "split": "dev", "segmenter_arch": args.segmenter_arch, "checkpoint": checkpoint, 
        "videos": sum(map(len, scores)), "segmentation_decode": "semi_markov_viterbi", "fit_split": "train", "selected": selected.to_dict(),
        "tiou_thresholds": list(thresholds), "heldout_f1_avg": float(np.mean(heldout)), 
        "plain_f1_avg": float(np.mean(plain)), "legal_f1_avg": float(np.mean(legal)),
        "grid": [{**m.to_dict(), "foldA_f1_avg": float(folds[0][i]), "foldB_f1_avg": float(folds[1][i])} for i,m in enumerate(grid)]
    }
    output = Path(args.output or f"outputs/tune_decode_{args.segmenter_arch}_{args.language}_dev.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    if args.write_config:
        path = ("duration_model", args.segmenter_arch, args.language)
        if not update_yaml_scalar(args.inference_config, path, json.dumps(selected.to_dict())):
            raise ValueError(f"Cannot write {'.'.join(path)}: the parent mapping must exist")
    summary = {k:v for k,v in payload.items() if k != "grid"}; summary["output"] = str(output)
    return summary


def tune_stream(args: argparse.Namespace) -> dict: # Never applied without --write-config.
    # Select commit lag on dev under duration-aware BIO decoding, under SAME statistic the paper reports (F1 averaged over `eval.yaml 
    # rq2.tiou_thresholds`, as tune_decode does); ties prefer lower lag.
    if args.split != "dev": raise SystemExit("tune-stream selects on dev only")
    if args.segmenter_arch != "s1": raise SystemExit("tune-stream selects lag for the in-system head (--segmenter-arch s1)")
    data_cfg = load_yaml(args.data_config)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    if args.num_videos:
        if args.write_config and args.num_videos < len(records): 
            raise ValueError("A partial-video smoke run cannot write final commit settings")
        records = sorted(records, key=lambda r: r.video_id)[: int(args.num_videos)]
    inference_cfg = resolve_inference(load_yaml(args.inference_config), args.language)
    if len(records) < 2: raise ValueError("tune-stream requires at least 2 dev videos for its 2 folds")

    model, device, _, _, checkpoint = _load_segmenter(args)
    model.eval().to(device)
    adapter = S1RunnerAdapter(model).to(device)
    meta = load_checkpoint_meta(checkpoint)
    adapter.duration_model = DurationModel(**meta["duration_model"]) if meta.get("duration_model") \
                                                                     else DurationModel.from_config(inference_cfg, args.language)
    if not meta.get("duration_model"): adapter.duration_model.require_calibration(inference_cfg, args.language)
    boundary = inference_cfg.get("boundary_stability", {}) or {}
    runner = StreamingSLTRunner(
        adapter, stride_s=float(inference_cfg.get("stride_s", 1.0)), buffer_cap_s=float(inference_cfg["buffer_cap_s"]),
        delta_enc_frames=int(boundary.get("delta_enc_frames", 3)), hysteresis_strides=int(boundary.get("hysteresis_strides", 3)),
        min_span_frames=(inference_cfg.get("span_selection", {}) or {}).get("min_span_frames"),
        forced_tail_policy=str(inference_cfg.get("forced_tail_policy", "skip")), gate_enabled=False, translate=False,
    )
    thresholds = [float(t) for t in (load_yaml(args.eval_config).get("rq2", {}) or {}).get("tiou_thresholds", [0.5])]
    grid_lag = [float(l) for l in args.grid_lag] if args.grid_lag else [0., 1., 2., 3., 4.]
    if not grid_lag or any(not math.isfinite(l) or l < 0 for l in grid_lag):
        raise ValueError("Commit lag candidates must be finite and nonnegative")
    print(f"[tune-stream] duration BIO head from {checkpoint}; commit lag grid {grid_lag}", flush=True)

    gold = _gold_events(records)
    ids = sorted(gold)
    folds = (set(ids[::2]), set(ids[1::2]))
    rows, best = [], None
    # VIDEO outer, lag inner: each video's raw frames are read once, run at every lag, then released. Caching them across the lag 
    # grid instead holds whole split's frames at once — ase dev is about 7 GiB of raw poses, and that is ANON memory, the kind an 
    # OOM killer acts on. Events are spans; keeping every lag's is free.
    by_lag: dict[float, dict[str, list[PredictionEvent]]] = {float(lag): {} for lag in grid_lag}
    for rec in tqdm(records, desc="[tune-stream] videos x lag grid"):
        poses, _ = load_pose_window(rec.pose, 0.0, rec.pose.duration_s, normalize=False)
        frames = torch.as_tensor(poses, dtype=torch.float32) if poses.shape[0] else None

        for lag in grid_lag:
            if frames is None: by_lag[float(lag)][rec.video_id] = []; continue
            runner.commit_lag_s = float(lag)  # `run` resets every per-stream state, so 1 runner serves the grid
            by_lag[float(lag)][rec.video_id] = [PredictionEvent(
                video_id=rec.video_id, start_s=float(e.start_s), end_s=float(e.end_s), text="",
                flagged_partial=bool(e.flagged_partial), commit_time_s=float(e.commit_time_s)
            ) for e in runner.run(frames, fps=float(rec.pose.fps))]
        del poses, frames

    for lag in grid_lag:
        events = scoreable_predictions(by_lag[float(lag)], records, tag=f"tune-stream lag={lag:g}")
        f1, lat = [], []
        for fold in folds:
            sub_p = {v: events.get(v, []) for v in fold}; sub_g = {v: gold[v] for v in fold}
            per_threshold = evaluate_predicted_events(sub_p, sub_g, thresholds)["thresholds"]
            f1.append(float(np.mean([float(row["segmentation"]["f1"]) for row in per_threshold])))
            lat.append(float((per_threshold[0].get("emission_latency") or {}).get("median_latency_s", float("nan"))))

        n_events = sum(len(v) for v in events.values())
        rows.append({
            "commit_lag_s": lag, "foldA_f1_avg": round(f1[0], 4), "foldB_f1_avg": round(f1[1], 4), 
            "median_latency_s": round(float(np.nanmean(lat)), 3), "events": n_events, "gold": sum(len(v) for v in gold.values())
        })
        key = (min(f1), sum(f1) / 2, -float(lag))
        if best is None or key > best[0]: best = (key, rows[-1])
        print(f"[tune-stream] lag={lag:g}: F1 avg {f1[0]:.3f}/{f1[1]:.3f} events {n_events} vs "
              f"gold {rows[-1]['gold']} latency {rows[-1]['median_latency_s']:.2f} s", flush=True)

    selected = dict(best[1])
    heldout = []
    for sel_i, eval_i in ((0, 1), (1, 0)):
        by_sel = max(rows, key=lambda r: (r[f"fold{'AB'[sel_i]}_f1_avg"], -r["commit_lag_s"]))
        heldout.append(by_sel[f"fold{'AB'[eval_i]}_f1_avg"])

    heldout_f1 = round(sum(heldout) / 2, 4)
    payload = {
        "language": args.language, "split": args.split, "segmenter_arch": "s1", "checkpoint": checkpoint, "videos": len(records), 
        "segmentation_decode": "semi_markov_viterbi", "duration_model": adapter.duration_model.to_dict(), "pose_normalization": "buffer", 
        "delta_enc_frames": int(boundary.get("delta_enc_frames", 0)), "min_span_frames": lambda_min_frames(inference_cfg), 
        "selected": selected, "tiou_thresholds": thresholds, "heldout_f1_avg": heldout_f1, "grid": rows, 
        "pin_as": {"boundary_stability": {"commit_lag_s": {args.language: selected["commit_lag_s"]}}}
    }
    output = Path(args.output or f"outputs/tune_stream_s1_{args.language}_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_config:
        if update_yaml_scalar(args.inference_config, ("boundary_stability", "commit_lag_s", args.language), selected["commit_lag_s"]):
            payload["config_updated"] = args.inference_config
        else: raise ValueError("Could not write the selected commit lag")
    print(f"[tune-stream] selected lag={selected['commit_lag_s']}s; held-out F1 avg={heldout_f1}", flush=True)
    out = dict(payload); out.pop("grid"); out["output"] = str(output)
    return out


def _s1_trained_context_s(bio_config: str | None, language: str | None = None) -> tuple[Path | None, float | None]:
    # (checkpoint, RoPE context stamped by train-bio). The PATH is returned so a refusal can name the file that carries
    # the number — the S1 log prints its own value, and the two disagree whenever the checkpoint predates a delta change.
    # `language` resolves a monolingual S1's ${corpus} dir exactly as train-bio --language does; a pooled config ignores it.
    if not bio_config: return None, None
    ckpt = Path(checkpoint_dir(load_yaml(bio_config, language=language), default="checkpoints/bio_s1") or "") / "model.pt"
    if not ckpt.exists(): return ckpt, None
    ctx = load_checkpoint_meta(ckpt).get("rope_eval_chunk_s")
    return ckpt, (float(ctx) if ctx else None)


def buffer_cap(args: argparse.Namespace) -> dict:
    """Write buffer_cap_s = p99 TRAIN-split sentence duration + stride_s + delta_enc/fps — a CAPACITY bound, from labels + config alone.

    The cap is a forced-commit TIMEOUT: the FSM buffer must hold a whole sentence (p99), plus the stride that detects its end, plus the 
    delta-frame overlap a commit leaves behind. No model is built and nothing is decoded — by design, not convenience: the cap parameterizes 
    TRAINING (the sampler clamps every window to it, and in the mode1-only baseline an over-cap sentence is silently unsupervised), so it 
    must not depend on any trained translator.

    A label-only statistic, so it is measured on the TRAIN split (constants that depend only on labels are measured on train; constants 
    that depend on a model are selected on dev). This stage is the ONE writer of inference.yaml buffer_cap_s. Run AFTER delta-enc (the 
    formula reads delta_enc_frames). Re-run per language and after any GT-preprocessing change (p99 is a label property).
    """
    if args.split != "train": raise SystemExit(
        "buffer-cap runs on the train split: constants that depend only on labels are measured on train, "
        "constants that depend on a model are selected on dev."
    )
    data_cfg = load_yaml(args.data_config)
    cfg = load_yaml(args.inference_config)
    cfg.pop("buffer_cap_s", None)  # this stage WRITES that row; resolving it would refuse the first run on a new language
    inference_cfg = resolve_inference(cfg, args.language)  # still strict on delta_enc_frames, which the formula reads
    records, _ = load_language_records(data_cfg, args.language, split=args.split)

    durations = [span.duration_s for rec in records for span in rec.sentences if getattr(span, "reliable", True)]
    p99_duration = float(np.percentile(durations, 99)) if durations else 0.0
    fps_hint = float(np.median([r.pose.fps for r in records])) if records else 24.0
    stride_s = float(inference_cfg.get("stride_s", 1.0))
    delta_s = float((inference_cfg.get("boundary_stability", {}) or {}).get("delta_enc_frames", 0)) / max(fps_hint, 1.0)
    cap = round(p99_duration + stride_s + delta_s, 2)
    # Coverage rule: the FSM must never run the head beyond its trained RoPE context. The writer refuses, not a later warning.
    s1_ckpt, s1_ctx = _s1_trained_context_s(getattr(args, "bio_config", None), args.language)
    if s1_ctx is not None and cap > s1_ctx + 1e-6: raise SystemExit(
        f"[buffer-cap] {args.language} cap {cap:.2f}s exceeds trained context {s1_ctx:.2f}s stamped in {s1_ckpt}. "
        f"Raise pretrain_geometry.buffer_cap_s above {cap:.2f}s and retrain S1 — that value is fixed, so retraining "
        f"without raising it reproduces the same stamp and the same refusal. The cap grew because delta-enc rewrote "
        f"delta_enc_frames after S1, or because a preprocessing change moved the train p99."
    )
    payload = {
        "language": args.language, "split": args.split, "sentences": len(durations),
        "p99_sentence_duration_s": p99_duration, "buffer_cap_s": cap,
        "cap_terms": {"p99_s": p99_duration, "stride_s": stride_s, "delta_enc_s": round(delta_s, 3)},
    }
    output = Path(args.output or f"outputs/buffer_cap_{args.language}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_config and update_yaml_scalar(args.inference_config, ("buffer_cap_s", str(args.language)), cap):
        payload["config_updated"] = args.inference_config
    print(f"[buffer-cap] buffer_cap_s={cap} (p99 {p99_duration:.2f} + stride {stride_s:g} + delta {delta_s:.3f})"
          + (f"; S1 context {s1_ctx:.2f}s covers it" if s1_ctx is not None else ""), flush=True)
    payload["output"] = str(output)
    return payload


def delta_enc(args: argparse.Namespace) -> dict:
    """BIO temporal noise floor sets the commit gate's delta_enc.

    Use the S1 in-system BIO head (checkpoints/bio_s1, from train-bio) — what the FSM runs — not the Moryossef external segmenter, not 
    the DLM checkpoint: δ_enc calibrates the gate's cut overlap against that head's noise, dlm.yaml asserts δ == this value, and reading 
    it off the DLM checkpoint is an ordering circularity. Measure as the head ENTERS stage 2.

    2 forwards per dev sentence window; shift of the selected span's terminator index under (a) dropped leading frame = stride-phase 
    misalignment from a growing buffer; (b) Gaussian keypoint noise sigma on x/y = pose jitter. delta_enc = ceil(p90 over both): 
    movement below the head's own noise floor must not block a commit.
    """
    if args.split != "dev": raise SystemExit("delta-enc calibration runs on dev only")
    from train.bio_pretrain import build_bio_s1_model
    data_cfg = load_yaml(args.data_config)
    cfg = load_yaml(args.bio_config, language=args.language)  # ${corpus} in checkpoint.dir -> pool dir, or this language's dir
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    device = pick_device(args.device)

    model = build_bio_s1_model(cfg)
    # checkpoint_dir substitutes the pool key on a pooled config, so a multilingual S1 resolves here exactly
    # as it does in train.py and eval.py. The language default is only for an untemplated monolingual config.
    checkpoint = args.checkpoint or str(Path(checkpoint_dir(cfg, default=f"checkpoints/bio_s1/{args.language}")) / "model.pt")
    # Same pool-provenance refusal as eval.py's _load_segmenter. It matters MOST here: delta-enc is the WRITER of the deployed gate geometry 
    # (--write-config persists delta/Lambda_min into inference.yaml AND dlm.yaml), a pooled and a monolingual BioS1Model are shape-identical 
    # so a wrong checkpoint strict-loads cleanly, and the result is a plausible-looking delta measured on a head the FSM never deploys.
    _meta = load_checkpoint_meta(checkpoint)
    if "pretrain_pool" in _meta and _meta.get("pretrain_pool") != pool_key(cfg): raise SystemExit(
        f"{checkpoint} was trained on pool {_meta.get('pretrain_pool')!r}, but this config expects {pool_key(cfg)!r}. "
        f"Point --checkpoint at the matching model, or align `pretrain_languages` — delta-enc calibrates the DEPLOYED "
        f"head's gate geometry, and a pooled checkpoint is a DIFFERENT model from a monolingual one."
    )
    load_model_checkpoint(model, checkpoint, strict=True)
    model.eval().to(device)
    print(f"[delta-enc] S1 BIO head from {checkpoint}", flush=True)
    sigma = float(args.noise_sigma)

    # Measure the selected duration-aware BIO terminator under the same minimum span as deployment.
    inference_cfg = resolve_inference(load_yaml(args.inference_config), args.language, strict=False)
    # Track the SELECTED span's terminator (select_target_span at the deployed Λ_min), not first_terminator_index over raw tags: 
    # raw tags carry phantom 1-frame micro-spans Λ_min filters at deployment, calibrating δ on spans the gate can never commit.
    min_span = lambda_min_frames(inference_cfg)
    duration = DurationModel.from_config(inference_cfg, args.language)
    print(f"[delta-enc] terminator decode: duration BIO; Lambda_min={min_span} frames", flush=True)

    sentences = [(rec, span) for rec in records for span in rec.sentences if getattr(span, "reliable", True)]
    if args.num_sentences and int(args.num_sentences) < len(sentences):
        # delta is a p90 over sentences, so a seeded random subset estimates it; first N would be first videos only.
        keep = sorted(np.random.default_rng(int(args.seed)).choice(len(sentences), size=int(args.num_sentences), replace=False))
        sentences = [sentences[i] for i in keep]
    rng = np.random.default_rng(int(args.seed))

    @torch.no_grad()
    def decode_variants(variants: list[tuple[np.ndarray, np.ndarray]], start_s: float) -> list[tuple[int, int] | None]:
        """(terminator_index, span_count) per variant, or None where the deployed FSM would select nothing.

        Equal lengths batch, other lengths get their own forward — correctness, not speed: the Uni-Sign pose encoder IGNORES frame_mask 
        (parity only) and its ST-GCN temporal convs have a ±6-frame receptive field, so a pad frame leaks into last real frames and global 
        attention perturbs EVERY frame's logits (|Δlogit| up to 0.077, flipping argmax tags). Padding would bias the drop-first-frame 
        family alone — manufacturing the jitter δ measures.
        """
        out: list[tuple[int, int] | None] = [None] * len(variants)
        groups: dict[int, list[int]] = {}
        for i, (p, _) in enumerate(variants): groups.setdefault(int(p.shape[0]), []).append(i)
        for n, idxs in groups.items():
            poses = torch.stack([torch.as_tensor(variants[i][0], dtype=torch.float32) for i in idxs]).to(device)
            ts = torch.stack([torch.as_tensor(variants[i][1] - start_s, dtype=torch.float32) for i in idxs]).to(device)
            mask = torch.ones(poses.shape[:2], dtype=torch.bool, device=device)  # exact length: no padding at all
            logits = model(poses, mask, timestamps_s=ts).logits
            decoded = DurationDecoder(duration).decode(logits, mask.long().sum(1), timestamps_s=ts)

            for j, i in enumerate(idxs):
                tags = decoded[j]
                span = select_target_span(tags, min_span)
                # No fallback to first_terminator_index: `span is None` = "no target this stride" (FSM waits), 
                # and a fallback would calibrate δ on a terminator the gate never tracks.
                out[i] = (int(span[1]), len(bio_complete_spans(tags))) if span is not None else None
        return out

    shifts: dict[str, list[int]] = {"drop_first_frame": [], "keypoint_noise": []}
    flips: dict[str, int] = {"drop_first_frame": 0, "keypoint_noise": 0}
    for rec, span in tqdm(sentences, desc="delta-enc"):
        start_s = max(0.0, span.start_s - 1.0)
        end_s = min(rec.pose.duration_s, span.end_s + 1.0)
        raw, timestamps = load_pose_window(rec.pose, start_s, end_s, normalize=False)
        if raw.shape[0] < 3: continue
        noisy = raw.copy()
        noisy[..., :2] += rng.normal(0., sigma, size=noisy[..., :2].shape).astype(noisy.dtype)
        variants = [(normalize_keypoints_unisign(raw), timestamps),
                    (normalize_keypoints_unisign(raw[1:]), timestamps[1:]),
                    (normalize_keypoints_unisign(noisy), timestamps)]
        base, dropped, perturbed = decode_variants(variants, start_s)
        if base is None: continue
        base_term, base_k = base
        # δ covers CONTINUOUS jitter only: a changed SEGMENT COUNT = re-decided segmentation (terminator moves ~a whole sentence), the χ 
        # commit log's business. Detect by count, not magnitude — magnitude would clip genuine jitter on short-sentence corpora. 
        # Dropped-buffer indices sit 1 frame earlier.
        for fam, val, off in (("drop_first_frame", dropped, 1), ("keypoint_noise", perturbed, 0)):
            if val is None: continue
            term, k = val
            if k != base_k: flips[fam] += 1; continue
            shifts[fam].append(abs((term + off) - base_term))

    # Retained (same-count) pairs only; discarded re-decisions sit beside them, not folded in.
    stats = {family: {
        "n": len(values), "median": float(np.median(values)) if values else 0.0,
        "p90": float(np.percentile(values, 90)) if values else 0.0, "max": int(max(values)) if values else 0,
        "count_flips_excluded": flips[family],
    } for family, values in shifts.items()}
    delta = int(math.ceil(max((s["p90"] for s in stats.values()), default=0.0))) or 1
    for family, s in stats.items():
        total = s["n"] + s["count_flips_excluded"]
        if total and s["count_flips_excluded"] / total > 0.1: print(
            f"[delta-enc] WARNING: {family} discarded {s['count_flips_excluded']}/{total} pairs as segment-count "
            f"re-decisions — δ={delta} rests on {s['n']} samples; inspect before --write-config.", flush=True)
        if s["n"] == 0: print(f"[delta-enc] WARNING: {family} has NO usable pairs; its p90 contributes 0.", flush=True)
    payload = {
        "language": args.language, "split": args.split, "noise_sigma": sigma, "noise_space": "raw_normalized_xy", 
        "pose_normalization": "per_variant_window", "segmentation_decode": "semi_markov_viterbi", "duration_model": duration.to_dict(),
        "sentences": len(sentences), "families": stats, "delta_enc_frames": delta,
    }
    if args.write_config and not any(s["n"] for s in stats.values()):
        raise ValueError("delta-enc has no usable boundary pairs; no geometry was written")
    if args.write_config:
        lang = str(args.language)
        written = [
            update_yaml_scalar(args.inference_config, ("boundary_stability", "delta_enc_frames", lang), delta),
            update_yaml_scalar(args.inference_config, ("boundary_stability", "duration_model_signature", lang), json.dumps(duration.signature))
        ]
        if not all(written): raise ValueError("Incomplete geometry update; check inference.yaml parent mappings")
        payload["config_updated"] = [c for c, ok in zip([args.inference_config] * 2, written) if ok]
        # Report what update_yaml_scalar actually changed — an unconditional "wrote" here would mask a failed write.
        if payload["config_updated"]: print(f"[delta-enc] wrote delta_enc_frames.{lang}={delta} to {args.inference_config}", flush=True)
        else: print(f"[delta-enc] WARNING: write-config changed nothing (key missing / file unwritable): {args.inference_config}", flush=True)

    output = Path(args.output or f"outputs/delta_enc_{args.language}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["output"] = str(output)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Misaligned-SLT analysis utilities")
    parser.add_argument("--stage", default="dataset-summary", choices=[
        "dataset-summary", "segmenter-infer", "tune-decode", "tune-stream", "segmenter-errors", "buffer-cap", "delta-enc", "loss-balance"
    ])
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--segmenter-arch", default="moryossef", choices=["moryossef", "s1"],
                        help="segmenter-infer backend: moryossef = external Moryossef segmenter, s1 = in-system BIO head")
    parser.add_argument("--moryossef-config", default="configs/moryossef26.yaml", help="Moryossef analysis-segmenter config")
    parser.add_argument("--segmenter-decode", choices=["plain", "duration"], default=None,
                        help="Segmenter-infer decoder; defaults to S1 duration or Moryossef plain")
    parser.add_argument("--bio-config", default="configs/bio_pretrain.yaml", help="S1 (in-system head) config for --segmenter-arch s1")
    parser.add_argument("--slt-config", default="configs/dlm.yaml")
    parser.add_argument("--baseline-config", default="configs/baseline_eval.yaml")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--language", default=None)  # None -> data.yaml active_languages[0] (never a stale hardcode)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--tiou-threshold", type=float, default=None)
    parser.add_argument("--num-sentences", type=int, default=None)
    parser.add_argument("--num-videos", type=int, default=None, help="tune-stream smoke: first N dev videos")
    parser.add_argument("--grid-lag", type=float, nargs="+", default=None, help="tune-stream: commit_lag_s values to sweep")
    parser.add_argument("--noise-sigma", type=float, default=0.005, help="delta-enc keypoint-noise std (normalized coords)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-config", action="store_true",
                        help="Persist measured constants: tune-decode -> duration_model; buffer-cap -> buffer_cap_s; "
                        "delta-enc -> delta_enc_frames; tune-stream -> commit_lag_s")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-test", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.language is None: args.language = str(load_yaml(args.data_config).get("active_languages", ["asf"])[0])
    if args.stage == "dataset-summary": result = dataset_summary(args)
    elif args.stage == "loss-balance": result = loss_balance(args)
    elif args.stage == "segmenter-infer": result = segmenter_infer(args)
    elif args.stage == "tune-decode": result = tune_decode(args)
    elif args.stage == "tune-stream": result = tune_stream(args)
    elif args.stage == "segmenter-errors":
        if not args.predictions: raise SystemExit("--predictions is required for --stage segmenter-errors")
        result = segmenter_errors(args)
    elif args.stage == "buffer-cap": result = buffer_cap(args)
    elif args.stage == "delta-enc": result = delta_enc(args)
    else: raise ValueError(f"Unsupported stage: {args.stage}")
    print(json.dumps(result, indent=2, sort_keys=True))
