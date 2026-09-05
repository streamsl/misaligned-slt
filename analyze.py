from __future__ import annotations
from dataclasses import asdict, dataclass
from statistics import median
from pathlib import Path
import json, argparse, math, csv

import torch
import numpy as np
from tqdm import tqdm
from data.loader import load_language_records
from data.windowing import BIO, TRUSTED_GAP_S, make_bio_labels
from poses import load_pose_window
from models.checkpointing import load_checkpoint_meta, load_model_checkpoint

from moryossef26.infer import predict_phrase_segments, whole_video_logits
from infer.commit_gate import bio_complete_spans, select_target_span
from infer.stream import S1RunnerAdapter, StreamingSLTRunner
from infer.duration_decode import (
    DEPLOYED_SEGMENTER_ARCH, STREAM_DECODE_ARCH, decode_config_key, duration_split_grid, duration_decode_params, 
    fit_duration_prior, streaming_decode_params, streaming_split_tags
)
from metrics import Segment, char_level_for_target, match_segments, moryossef_segment_metrics, sentence_bleu_scores
from eval import (
    PredictionEvent, _gold_events, _load_segmenter, evaluate_predicted_events, 
    load_event_predictions, load_prediction_file, save_prediction_file, scoreable_predictions
)
from utils import (
    checkpoint_dir, lambda_min_frames, load_yaml, pick_device, pool_key, 
    resolve_inference, resolve_pretrained, target_language, update_yaml_scalar
)
# Low on purpose: near-misses feed the (Δ_head, Δ_tail) jitter CDF as matched pairs, not phantom/skip events;
# a high bar biases the CDF to zero. Override: --tiou-threshold.
SEGMENTER_ERROR_MATCH_TIOU = 0.1

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
    # Decode defaults per arch, as in eval --segmenter-eval: s1 -> our semi-Markov re-split, moryossef -> faithful Moryossef argmax. 
    # `--segmenter-decode duration` on external measures the deployed decode's errors, not the baseline's.
    # arch-aware: each segmenter reads its own duration_decode_<arch> block, never the other's.
    dd = duration_decode_params(load_yaml(args.inference_config), args.language, arch=args.segmenter_arch)
    decode = args.segmenter_decode or ("duration" if (args.segmenter_arch == "s1" and dd is not None) else "plain")
    duration_prior = None
    if decode == "duration":
        train_records, _ = load_language_records(data_cfg, args.language, split="train")
        duration_prior = fit_duration_prior(train_records, **(dd or {}))
    print(f"[segmenter-infer] {args.segmenter_arch} from {checkpoint} (decode={'duration' if duration_prior else 'plain'})", flush=True)

    predictions = predict_phrase_segments(
        model, records, device=device, velocity=velocity, rope_chunk_s=rope_chunk_s, duration_prior=duration_prior,
    )
    output = Path(args.output or f"outputs/segmenter_predictions_{args.segmenter_arch}_{args.language}_{args.split}.json")
    save_prediction_file(predictions, output, provenance={
        "segmenter_arch": args.segmenter_arch, "decode": "duration" if duration_prior else "plain",
        "pose_normalization": "chunk" if args.segmenter_arch == "s1" else "video", "decode_hparams": dd if duration_prior else None, 
        "checkpoint": checkpoint, "language": args.language, "split": args.split,
    })
    return {
        "language": args.language, "split": args.split, "videos": len(records), "segmenter_arch": args.segmenter_arch, 
        "checkpoint": checkpoint, "predicted_segments": sum(len(v) for v in predictions.values()), "output": str(output),
    }


def _assert_predictions_match_pinned_decode(args: argparse.Namespace) -> None: # Refuse spans decoded with triple that is no longer pinned.
    # Segmenter-error analysis feeds the reported taxonomy and measured-jitter ABLATION (main recipe trains on the designed distributions). 
    # Re-tuning the upstream decode still invalidates the artifacts.
    stamped = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    if not isinstance(stamped, dict) or "provenance" not in stamped: return  # unstamped file: nothing to check
    prov = stamped["provenance"]
    arch = str(prov.get("segmenter_arch") or DEPLOYED_SEGMENTER_ARCH)
    used = prov.get("decode_hparams") if prov.get("decode") == "duration" else None
    pinned = duration_decode_params(load_yaml(args.inference_config), args.language, arch=arch)
    if used == pinned: return
    raise SystemExit(
        f"{args.predictions} was decoded with {used}, but {decode_config_key(arch)}.{args.language} "
        f"now pins {pinned}. Segmenter-error analysis must reflect the decode you report — re-run "
        f"`analyze.py --stage segmenter-infer --segmenter-arch {arch} --segmenter-decode duration` (and retrain "
        f"any measured-jitter ablation run that consumed them).")


def segmenter_errors(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test: raise SystemExit(
        "Segmenter-error analysis on test needs --allow-test. Legitimate: the REPORTED taxonomy (describes the "
        "same split as the results tables). NOT legitimate: feeding a test-measured artifact to the measured-jitter "
        "ablation — that is training-input contamination, and the training side refuses split=test artifacts."
    )
    cfg = load_yaml(args.data_config)
    records, _ = load_language_records(cfg, args.language, split=args.split)
    predictions = load_prediction_file(args.predictions)  # the segmenter-infer output file
    _assert_predictions_match_pinned_decode(args)
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
    """Two-fold dev tuning of the semi-Markov decode pair (split_bias, snap_radius_s) — infer/duration_decode.py.

    F1(split_bias) is a knife edge (segment-count Lagrange multiplier against a near-constant per-segment cost, so the split count steps) 
    and the knobs interact, so marginal sweeps mislead: joint FINE grid, one forward/video, decode-only re-sweep, per-fold macro F1@0.5, 
    max(min(foldA, foldB)). Run ONCE PER SEGMENTER: each arch's posteriors are calibrated differently, so `--segmenter-arch <arch>` pins 
    inference.yaml `duration_decode_<arch>.<lang>` — s1's block is the DEPLOYED one (FSM, membership gate, RQ1/RQ2), moryossef's is read 
    only by the baseline's own analysis. Never auto-applied without --write-config, keeping eval tuning-free.
    """
    if args.split == "test": raise SystemExit("tune-decode is dev-only: tuning on test is test contamination")
    data_cfg = load_yaml(args.data_config)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    train_records, _ = load_language_records(data_cfg, args.language, split="train")
    prior = fit_duration_prior(train_records)
    if prior is None: raise SystemExit("too few train captions to fit a duration prior")
    model, device, velocity, rope_chunk_s, checkpoint = _load_segmenter(args)
    model.eval().to(device)
    print(f"[tune-decode] {args.segmenter_arch} segmenter from {checkpoint}; one forward per video, then decode-only sweep", flush=True)

    cached = []  # (video_id, tags, pB, gold, fps)
    with torch.no_grad():
        for rec in records:
            logits, timestamps = whole_video_logits(model, rec, device, velocity, rope_chunk_s)
            if logits is None: continue
            logits = logits[0].float().cpu()
            gold = torch.as_tensor(np.asarray(make_bio_labels(
                timestamps, rec.sentences, 0.0, float(rec.pose.duration_s),
                trusted_gap_s=TRUSTED_GAP_S, video_duration_s=rec.pose.duration_s
            ))).long()
            pB = torch.softmax(logits, dim=-1)[:, BIO["B"]].numpy()  # snap evidence; matches duration_decode_tags
            cached.append((rec.video_id, logits.argmax(-1).numpy(), pB, gold, float(rec.pose.fps)))
    cached.sort(key=lambda c: c[0])
    folds = (cached[::2], cached[1::2])

    # Emission weight shifts count-optimal bias (its logits are negative off-boundary), so joint grid must cover higher bias than w=0 
    # sweep needed. Upper ends extend past every value selected so far — Triple selected AT grid edge means the optimum may lie outside.
    grid_bias = [round(float(b), 2) for b in np.arange(2.5, 16.01, 0.25)]
    grid_radius = [0.0, 0.5, 1.0, 1.5]
    grid_weight = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

    # 1 batched DP per video covers every (w, bias) cell (infer.duration_decode.duration_split_grid, equal to the scalar decode
    # cell by cell); the sweep cost is then the per-cell F1, not the DP. Per-fold macro F1@0.5 per cell.
    fold_scores: list[dict[tuple[float, float, float], list[float]]] = [{}, {}]
    for fold_i, fold in enumerate(folds):
        for _, tags, pB, gold, fps in tqdm(fold, desc=f"[tune-decode] fold {'AB'[fold_i]} DP"):
            grid_tags = duration_split_grid(tags, pB, fps, prior, grid_bias, grid_weight, grid_radius, device=None)
            for cell, t in grid_tags.items():
                oh = torch.nn.functional.one_hot(torch.as_tensor(t).long(), num_classes=4).float().unsqueeze(0)
                fold_scores[fold_i].setdefault(cell, []).append(
                    moryossef_segment_metrics(oh, gold.unsqueeze(0), prefix="p", tiou_threshold=0.5)["p_tiou_f1"]
                )

    def fold_f1(fold_i, bias, radius, w_b):
        f1s = fold_scores[fold_i].get((float(w_b), float(bias), float(radius)), [])
        return float(np.mean(f1s)) if f1s else 0.0

    rows, best = [], None
    for w_b in grid_weight:
        for bias in grid_bias:
            for radius in grid_radius:
                a, b = fold_f1(0, bias, radius, w_b), fold_f1(1, bias, radius, w_b)
                rows.append({
                    "split_bias": bias, "snap_radius_s": radius, "boundary_logit_weight": w_b, 
                    "foldA_f1@0.5": round(a, 4), "foldB_f1@0.5": round(b, 4)
                })
                key = (min(a, b), (a + b) / 2)  # fold-consistent first, mean as tie-break
                if best is None or key > best[0]: best = (key, rows[-1])
            print(f"[tune-decode] w={w_b:.2f} bias={bias:4.2f}: " + " ".join(
                f"r={r['snap_radius_s']:.1f}:{r['foldA_f1@0.5']:.3f}/{r['foldB_f1@0.5']:.3f}" for r in rows[-len(grid_radius):]
            ), flush=True)

    selected = dict(best[1])
    # Held-out estimate, the number to QUOTE: re-select per fold, evaluate on the other, average. 
    # The selected pair's own cells are in-selection maxima, overstating the gain.
    heldout = []
    for sel_i, eval_i in ((0, 1), (1, 0)):
        key_col = f"fold{'AB'[sel_i]}_f1@0.5"
        by_sel = max(rows, key=lambda r: r[key_col])
        heldout.append(by_sel[f"fold{'AB'[eval_i]}_f1@0.5"])
        
    heldout_f1 = round(sum(heldout) / 2, 4)
    output = Path(args.output or f"outputs/tune_decode_{args.segmenter_arch}_{args.language}_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    block = decode_config_key(args.segmenter_arch)
    payload = {
        "language": args.language, "split": args.split, "segmenter_arch": args.segmenter_arch, "checkpoint": checkpoint,
        "videos": len(cached), "prior": {"mu_log_s": prior.mu_log_s, "sd_log_s": prior.sd_log_s, "cap_s": prior.cap_s},
        "selected": selected, "heldout_f1@0.5": heldout_f1, "heldout_per_fold": [round(h, 4) for h in heldout], "grid": rows,
        "pin_as": {block: {str(args.language): {
            "split_bias": selected["split_bias"], "snap_radius_s": selected["snap_radius_s"],
            "boundary_logit_weight": selected["boundary_logit_weight"]
        }}},
        "rope_eval_chunk_s": rope_chunk_s,  # s1 only; the trained context this tuning is valid at
        "pose_normalization": "chunk" if args.segmenter_arch == "s1" else "video",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.write_config: # Written as a YAML flow mapping under <block>.<language>.
        # The baseline's triple lives in its own block: one shared entry would decode the baseline with the
        # triple tuned for our head (see duration_decode_params).
        flow = (f"{{split_bias: {selected['split_bias']}, snap_radius_s: {selected['snap_radius_s']}, "
                f"boundary_logit_weight: {selected['boundary_logit_weight']}}}")
        if update_yaml_scalar(args.inference_config, (block, str(args.language)), flow):
            payload["config_updated"] = args.inference_config
            print(f"[tune-decode] wrote {block}.{args.language}: {flow} to {args.inference_config}", flush=True)
        else:
            print(f"[tune-decode] could not find {block}.{args.language} in {args.inference_config}; "
                  f"add the row manually under {block}: {args.language}: {flow}", flush=True)
    print(f"[tune-decode] selected split_bias={selected['split_bias']} snap_radius_s={selected['snap_radius_s']} boundary_logit_weight="
          f"{selected['boundary_logit_weight']} (in-selection F1@0.5 {selected['foldA_f1@0.5']}/{selected['foldB_f1@0.5']}; HELD-OUT estimate "
          f"{heldout_f1} — quote the held-out number); pin the triple in inference.yaml {block}.{args.language}", flush=True)
    payload_out = dict(payload); payload_out["output"] = str(output)
    payload_out.pop("grid")  # grid lives in the JSON; keep stdout short
    return payload_out


def tune_stream(args: argparse.Namespace) -> dict: # Never applied without --write-config.
    """Select the FSM's decode triple UNDER THE STREAMING DECODE on dev (inference.yaml duration_decode_s1_stream.<lang>).

    tune-decode selects whole-video triple. FSM decodes growing buffer whose last sentence is right-censored, with the survival re-split, 
    terminator rule, hysteresis and Lambda_min: a different estimator, in which same split bias is far more permissive. So FSM's triple is 
    selected by running FSM itself (segmentation-only, no decoder) on S1 head over dev, 2 folds, max-min-fold segmentation F1@0.5 under RQ2 
    protocol. snap and w are held at the whole-video values; only the count knob (split_bias) is swept unless --grid-bias says otherwise. 
    """
    if args.split == "test": raise SystemExit("tune-stream is dev-only: tuning on test is test contamination")
    if args.segmenter_arch != "s1": raise SystemExit("tune-stream selects FSM's triple; FSM runs the in-system head (--segmenter-arch s1)")
    data_cfg = load_yaml(args.data_config)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    if args.num_videos: records = sorted(records, key=lambda r: r.video_id)[: int(args.num_videos)]
    train_records, _ = load_language_records(data_cfg, args.language, split="train")
    inference_cfg = resolve_inference(load_yaml(args.inference_config), args.language)
    model, device, _, _, checkpoint = _load_segmenter(args)
    model.eval().to(device)

    adapter = S1RunnerAdapter(model).to(device)
    boundary = inference_cfg.get("boundary_stability", {}) or {}
    runner = StreamingSLTRunner(
        adapter, stride_s=float(inference_cfg.get("stride_s", 1.0)), buffer_cap_s=float(inference_cfg["buffer_cap_s"]),
        delta_enc_frames=int(boundary.get("delta_enc_frames", 3)), hysteresis_strides=int(boundary.get("hysteresis_strides", 3)),
        min_span_frames=(inference_cfg.get("span_selection", {}) or {}).get("min_span_frames"),
        forced_tail_policy=str(inference_cfg.get("forced_tail_policy", "skip")), 
        gate_enabled=False, duration_prior=None, translate=False,
    )
    pinned = duration_decode_params(inference_cfg, args.language) or {}
    base_bias = float(pinned.get("split_bias", 4.0))
    snap = float(pinned.get("snap_radius_s", 1.0))
    w_b = float(pinned.get("boundary_logit_weight", 0.0))

    grid_bias = [float(b) for b in args.grid_bias] if args.grid_bias \
                                                   else [round(base_bias + d, 2) for d in range(-4, 3) if 2.5 <= base_bias + d <= 16.0]
    grid_lag = [float(l) for l in args.grid_lag] if args.grid_lag else [0.0, 1.0, 2.0, 3.0, 4.0]
    print(f"[tune-stream] S1 head from {checkpoint}; snap={snap} w={w_b} (whole-video values); "
          f"split_bias grid {grid_bias}; commit_lag_s grid {grid_lag}", flush=True)

    gold = _gold_events(records)
    ids = sorted(gold)
    folds = (set(ids[::2]), set(ids[1::2]))
    poses_cache: dict[str, tuple[np.ndarray, float]] = {}
    rows, best = [], None

    for bias, lag in [(b, l) for b in grid_bias for l in grid_lag]:
        prior = fit_duration_prior(train_records, split_bias=bias, snap_radius_s=snap, boundary_logit_weight=w_b)
        runner.duration_prior = prior; runner.commit_lag_s = float(lag)
        events: dict[str, list[PredictionEvent]] = {}

        for rec in tqdm(records, desc=f"[tune-stream] bias={bias:g} lag={lag:g}"):
            if rec.video_id not in poses_cache:
                poses, _ = load_pose_window(rec.pose, 0.0, rec.pose.duration_s, normalize=False)
                poses_cache[rec.video_id] = (poses, float(rec.pose.fps))

            poses, fps = poses_cache[rec.video_id]
            if poses.shape[0] == 0: events[rec.video_id] = []; continue
            evs = runner.run(torch.as_tensor(poses, dtype=torch.float32), fps=fps)
            events[rec.video_id] = [PredictionEvent(
                video_id=rec.video_id, start_s=float(e.start_s), end_s=float(e.end_s), text="", 
                flagged_partial=bool(e.flagged_partial), commit_time_s=float(e.commit_time_s)
            ) for e in evs]

        events = scoreable_predictions(events, records, inference_cfg, tag=f"tune-stream bias={bias:g} lag={lag:g}")
        f1, lat = [], []
        for fold in folds:
            sub_p = {v: events.get(v, []) for v in fold}; sub_g = {v: gold[v] for v in fold}
            row = evaluate_predicted_events(sub_p, sub_g, [0.5])["thresholds"][0]
            f1.append(float(row["segmentation"]["f1"]))
            lat.append(float((row.get("emission_latency") or {}).get("median_latency_s", float("nan"))))

        n_events = sum(len(v) for v in events.values())
        rows.append({
            "split_bias": bias, "snap_radius_s": snap, "boundary_logit_weight": w_b, "commit_lag_s": lag,
            "foldA_f1@0.5": round(f1[0], 4), "foldB_f1@0.5": round(f1[1], 4), "median_latency_s": round(float(np.nanmean(lat)), 3),
            "events": n_events, "gold": sum(len(v) for v in gold.values())
        })
        key = (min(f1), sum(f1) / 2)
        if best is None or key > best[0]: best = (key, rows[-1])
        print(f"[tune-stream] bias={bias:g} lag={lag:g}: F1@0.5 {f1[0]:.3f}/{f1[1]:.3f} events {n_events} vs gold {rows[-1]['gold']} "
              f"latency {rows[-1]['median_latency_s']:.2f} s", flush=True)

    selected = dict(best[1])
    heldout = []
    for sel_i, eval_i in ((0, 1), (1, 0)):
        by_sel = max(rows, key=lambda r: r[f"fold{'AB'[sel_i]}_f1@0.5"])
        heldout.append(by_sel[f"fold{'AB'[eval_i]}_f1@0.5"])

    heldout_f1 = round(sum(heldout) / 2, 4)
    block = decode_config_key(STREAM_DECODE_ARCH)
    payload = {
        "language": args.language, "split": args.split, "segmenter_arch": "s1", "checkpoint": checkpoint, "videos": len(records), 
        "pose_normalization": "buffer", "delta_enc_frames": int(boundary.get("delta_enc_frames", 0)),
        "min_span_frames": lambda_min_frames(inference_cfg), "selected": selected, "heldout_f1@0.5": heldout_f1, 
        "grid": rows, "pin_as": {block: {
            str(args.language): {k: selected[k] for k in ("split_bias", "snap_radius_s", "boundary_logit_weight")}
        }, "boundary_stability": {"commit_lag_s": {str(args.language): selected["commit_lag_s"]}}}
    }
    output = Path(args.output or f"outputs/tune_stream_s1_{args.language}_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_config:
        flow = (f"{{split_bias: {selected['split_bias']}, snap_radius_s: {selected['snap_radius_s']}, "
                f"boundary_logit_weight: {selected['boundary_logit_weight']}}}")
        ok1 = update_yaml_scalar(args.inference_config, (block, str(args.language)), flow)
        ok2 = update_yaml_scalar(args.inference_config, ("boundary_stability", "commit_lag_s", str(args.language)), selected["commit_lag_s"])
        if ok1 and ok2:
            payload["config_updated"] = args.inference_config
            print(f"[tune-stream] wrote {block}.{args.language}: {flow} and boundary_stability.commit_lag_s.{args.language}: "
                  f"{selected['commit_lag_s']} to {args.inference_config}", flush=True)
        else: print(f"[tune-stream] WARNING: write incomplete (triple {ok1}, lag {ok2}); add the missing row by hand", flush=True)

    print(f"[tune-stream] selected split_bias={selected['split_bias']} commit_lag_s={selected['commit_lag_s']} "
          f"(in-selection F1@0.5 {selected['foldA_f1@0.5']}/{selected['foldB_f1@0.5']}; HELD-OUT {heldout_f1}; "
          f"median latency {selected['median_latency_s']} s)", flush=True)
    out = dict(payload); out.pop("grid"); out["output"] = str(output)
    return out


def _s1_trained_context_s(bio_config: str | None, language: str | None = None) -> float | None:
    # RoPE context stamped by train-bio (rope_eval_chunk_s); None before S1 exists or when no bio config is given.
    # `language` resolves a monolingual S1's ${corpus} dir exactly as train-bio --language does; a pooled config ignores it.
    if not bio_config: return None
    ckpt = Path(checkpoint_dir(load_yaml(bio_config, language=language), default="checkpoints/bio_s1") or "") / "model.pt"
    if not ckpt.exists(): return None
    ctx = load_checkpoint_meta(ckpt).get("rope_eval_chunk_s")
    return float(ctx) if ctx else None


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
    s1_ctx = _s1_trained_context_s(getattr(args, "bio_config", None), args.language)
    if s1_ctx is not None and cap > s1_ctx + 1e-6: raise SystemExit(
        f"[buffer-cap] {args.language} cap {cap:.2f}s exceeds the S1 checkpoint's trained context {s1_ctx:.2f}s. Retrain S1 "
        f"(pretrain_geometry.buffer_cap_s: auto = train p99 + stride + 1 s per pool language; a number there is an explicit override "
        f"when delta exceeds one second) before pinning this cap."
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
    if args.split == "test" and not args.allow_test: raise SystemExit("Delta-enc runs on dev; --allow-test only for smoke debugging")
    from train.bio_pretrain import build_bio_s1_model
    data_cfg = load_yaml(args.data_config)
    cfg = load_yaml(args.bio_config, language=args.language)  # ${corpus} in checkpoint.dir -> pool dir, or this language's dir
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    device = pick_device(args.device)

    pretrained = resolve_pretrained(cfg, data_cfg, args.language, default="checkpoints/openasl_pose_only_slt.pth")
    model = build_bio_s1_model(cfg, pretrained_path=pretrained)
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

    # Measure δ_enc under the DEPLOYED decode: with inference.yaml duration_decode on the FSM re-splits every buffer BEFORE reading 
    # terminators (infer/stream.py step()). Raw argmax is the wrong instrument — on back-to-back corpora its first O-or-B jumps WHOLE 
    # SENTENCES per one-frame perturbation, whose p90 pushed min_span_frames past the buffer cap and broke FSM span selection.
    inference_cfg = resolve_inference(load_yaml(args.inference_config), args.language, strict=False)
    dd, dd_block = streaming_decode_params(inference_cfg, args.language)  # delta is measured under the FSM's decode
    duration_prior = None
    if dd is not None:
        train_records, _ = load_language_records(data_cfg, args.language, split="train")
        duration_prior = fit_duration_prior(train_records, **dd)
    # Track the SELECTED span's terminator (select_target_span at the deployed Λ_min), not first_terminator_index over raw tags: 
    # raw tags carry phantom 1-frame micro-spans Λ_min filters at deployment, calibrating δ on spans the gate can never commit.
    min_span = lambda_min_frames(inference_cfg)
    delta_tol = int((inference_cfg.get("boundary_stability", {}) or {}).get("delta_enc_frames", 3) or 3)
    print(f"[delta-enc] terminator decode: {('duration, ' + dd_block) if duration_prior else 'plain argmax'}; "
          f"Lambda_min={min_span} frames", flush=True)

    fps_hint = float(np.median([r.pose.fps for r in records])) if records else 24.0
    sentences = [(rec, span) for rec in records for span in rec.sentences if getattr(span, "reliable", True)]
    if args.num_sentences and int(args.num_sentences) < len(sentences):
        # delta is a p90 over sentences, so a seeded random subset estimates it; first N would be first videos only.
        keep = sorted(np.random.default_rng(int(args.seed)).choice(len(sentences), size=int(args.num_sentences), replace=False))
        sentences = [sentences[i] for i in keep]
    rng = np.random.default_rng(int(args.seed))

    @torch.no_grad()
    def decode_variants(variants: list[tuple[np.ndarray, np.ndarray]], start_s: float, fps: float) -> list[tuple[int, int] | None]:
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
            for j, i in enumerate(idxs):
                tags = logits[j].argmax(dim=-1)
                if duration_prior is not None and n > 2:
                    pB = torch.softmax(logits[j].float(), dim=-1)[:, BIO["B"]].cpu().numpy()
                    # The deployed streaming decode: interior splits by survival, the censored-tail split under the terminator
                    # rule with the CURRENT delta as tolerance (a fixed point like Lambda_min: re-run once if delta moves).
                    tags = torch.as_tensor(streaming_split_tags(tags.cpu().numpy(), pB, fps, duration_prior, delta_frames=delta_tol))
                span = select_target_span(tags, min_span)
                # No fallback to first_terminator_index: `span is None` = "no target this stride" (FSM waits), and
                # a fallback would calibrate δ on a terminator the gate never tracks.
                out[i] = (int(span[1]), len(bio_complete_spans(tags))) if span is not None else None
        return out

    shifts: dict[str, list[int]] = {"drop_first_frame": [], "keypoint_noise": []}
    flips: dict[str, int] = {"drop_first_frame": 0, "keypoint_noise": 0}
    for rec, span in tqdm(sentences, desc="delta-enc"):
        start_s = max(0.0, span.start_s - 1.0)
        end_s = min(rec.pose.duration_s, span.end_s + 1.0)
        poses, timestamps = load_pose_window(rec.pose, start_s, end_s, normalize=True)
        if poses.shape[0] < 3: continue
        fps = float(rec.pose.fps)
        noisy = poses.copy()
        noisy[..., :2] = noisy[..., :2] + rng.normal(0.0, sigma, size=noisy[..., :2].shape).astype(noisy.dtype)
        base, dropped, perturbed = decode_variants([(poses, timestamps), (poses[1:], timestamps[1:]), (noisy, timestamps)], start_s, fps)
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
        "language": args.language, "split": args.split, "noise_sigma": sigma,
        "sentences": len(sentences), "families": stats, "delta_enc_frames": delta,
    }
    # delta is not a standalone constant. On commit the buffer is cut at terminator-delta, so the next buffer opens
    # with a delta-frame leftover of the sentence just emitted; Lambda_min must exceed delta or that leftover is
    # selectable as a span (infer/stream.py's own rule and default). dlm.yaml's gate must equal both. Written
    # together so 4 values cannot drift — a stale pair is exactly what leaves the FSM in an invalid geometry.
    lam = delta + 1
    p10_frames = int(np.percentile([
        s.duration_s for r in records for s in r.sentences if getattr(s, "reliable", True)
    ], 10) * fps_hint) if records else 0
    if p10_frames and lam > p10_frames: print(
        f"[delta-enc] WARNING: Lambda_min={lam} exceeds the p10 sentence ({p10_frames} frames) — >10% of real "
        f"sentences become unselectable. delta is inflated by snap_radius_s; re-tune the decode before accepting.", flush=True
    )
    payload["min_span_frames"] = lam
    if args.write_config:
        lang = str(args.language)
        written = [update_yaml_scalar(args.inference_config, ("boundary_stability", "delta_enc_frames", lang), delta),
                   update_yaml_scalar(args.inference_config, ("span_selection", "min_span_frames", lang), lam)]
        payload["config_updated"] = [c for c, ok in zip([args.inference_config] * 2, written) if ok]
        # Report what update_yaml_scalar actually changed — an unconditional "wrote" here would mask a failed write.
        if payload["config_updated"]: 
            print(f"[delta-enc] wrote delta_enc_frames.{lang}={delta}, min_span_frames.{lang}={lam} to {args.inference_config}", flush=True)
        else: 
            print(f"[delta-enc] WARNING: write-config changed nothing (key missing / file unwritable): {args.inference_config}", flush=True)

    output = Path(args.output or f"outputs/delta_enc_{args.language}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["output"] = str(output)
    return payload


def system_outcomes(records, predicted, threshold: float, material_overlap_s: float) -> dict:
    """The canonical PER-GOLD outcome table — the comparison primitive every cross-system analysis derives from.

    Every reliable gold sentence gets exactly 1 row with STABLE id, so any 2 systems evaluated on same split join row-for-row: 
    paired significance tests, win/loss flips, and per-factor breakdowns all become column operations on identically-indexed tables. 
    Classes are mutually exclusive (matched > merged > split > missed for gold; matched > merger > fragment > phantom for events — 
    same precedence rule as the segmenter taxonomy); forced-PARTIAL commits are counted orthogonally.
    """
    gold_rows: list[dict] = []
    event_counts = {"matched": 0, "merger": 0, "fragment": 0, "phantom": 0}
    phantom_events: list[dict] = []
    partial_events: list[dict] = []

    for rec in sorted(records, key=lambda r: r.video_id):
        vid = rec.video_id
        golds = [sp for sp in rec.sentences if getattr(sp, "reliable", True)]
        events = list(predicted.get(vid, []))
        gold_segs = [Segment(sp.start_s, sp.end_s) for sp in golds]
        ev_segs = [ev.segment for ev in events]
        matches = match_segments(ev_segs, gold_segs, threshold=threshold)
        m_by_gold = {gi: (pi, iou) for pi, gi, iou in matches}
        m_by_ev = {pi for pi, _, _ in matches}

        cover_g: dict[int, list[int]] = {}
        cover_e: dict[int, list[int]] = {}
        for ei, ev in enumerate(ev_segs):
            for gi, gt in enumerate(gold_segs):
                if max(0.0, min(ev.end_s, gt.end_s) - max(ev.start_s, gt.start_s)) > float(material_overlap_s):
                    cover_g.setdefault(gi, []).append(ei)
                    cover_e.setdefault(ei, []).append(gi)
        mergers = {ei for ei, gis in cover_e.items() if len(gis) >= 2}
        frag_gold = {gi for gi, eis in cover_g.items() if len(eis) >= 2}

        for ei, ev in enumerate(events):
            if getattr(ev, "flagged_partial", False):
                partial_events.append({"video_id": vid, "start_s": float(ev.start_s), "end_s": float(ev.end_s)})
            if ei in m_by_ev: event_counts["matched"] += 1
            elif ei in mergers: event_counts["merger"] += 1
            elif any(ei in cover_g.get(gi, []) for gi in frag_gold): event_counts["fragment"] += 1
            elif ei not in cover_e:
                event_counts["phantom"] += 1
                phantom_events.append({
                    "video_id": vid, "start_s": float(ev.start_s), "end_s": float(ev.end_s), "text": (ev.text or "")[:120]
                })
            else: event_counts["fragment"] += 1  # materially covers 1 gold, unmatched: a fragment of it

        for gi, sp in enumerate(golds):
            if gi in m_by_gold: cls = "matched"
            elif any(ei in mergers for ei in cover_g.get(gi, [])): cls = "merged"
            elif gi in frag_gold: cls = "split"
            else: cls = "missed"
            row = {
                "gold_id": f"{vid}:{gi}", "video_id": vid, "class": cls, "start_s": float(sp.start_s), "end_s": float(sp.end_s),
                "duration_s": float(sp.end_s - sp.start_s), "ref": sp.text, "ref_words": len(sp.text.split()),
                "tiou": None, "delta_head_s": None, "delta_tail_s": None, "hyp": None,
            }
            if cls == "matched":
                ei, iou = m_by_gold[gi]
                ev = events[ei]
                row.update(
                    tiou=float(iou), delta_head_s=float(ev.start_s - sp.start_s), 
                    delta_tail_s=float(ev.end_s - sp.end_s), hyp=ev.text or ""
                )
            gold_rows.append(row)
    return {"gold_rows": gold_rows, "event_counts": event_counts, "phantom_events": phantom_events, "partial_events": partial_events}


def score_outcomes(gold_rows: list[dict], char_level: bool) -> None:
    """Attach sentence-BLEU to matched rows (in place) and a deployment localized_bleu4 to EVERY row (0 for unmatched).

    localized_bleu4 charges segmentation failures as zero translation — per-gold deployment outcome — so identically indexed localized_bleu4 
    vectors from 2 systems support PAIRED tests even when their matched sets differ. Slice by `class` to separate "missed" from "matched but 
    badly translated"; `bleu4` (matched rows only) is translation-intrinsic view. BLEURT is off: this is diagnostics, not the headline metric.
    """
    matched = [r for r in gold_rows if r["class"] == "matched"]
    bleus = sentence_bleu_scores([r["hyp"] for r in matched], [r["ref"] for r in matched], char_level=char_level)
    for row, b in zip(matched, bleus): row["bleu4"] = round(float(b), 2)
    for row in gold_rows:
        row.setdefault("bleu4", None)
        row["localized_bleu4"] = float(row["bleu4"]) if row["class"] == "matched" else 0.0


def system_errors(args: argparse.Namespace) -> dict:
    """Single-system failure report from an RQ2 events JSON; `report.py` compares several systems side by side.

    Emits per-gold OUTCOME TABLE as CSV + a JSON with: the gold/event taxonomy (frequencies x mean sentence-BLEU), sentence-BLEU by tIoU bin 
    (boundary-induced vs translation-intrinsic loss), duration bins incl. over-cap tail, and rule-selected case studies (top-k per failure 
    type by reference length). Matching: tIoU >= --tiou-threshold (default 0.5, system-level attribution); material floor = Lambda_min.
    """
    if not args.predictions: raise SystemExit("--predictions (an outputs/rq2_*_events_*.json) is required for --stage system-errors")
    data_cfg = load_yaml(args.data_config)
    inference_cfg = resolve_inference(load_yaml(args.inference_config), args.language)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    predicted = scoreable_predictions(load_event_predictions(args.predictions), records, inference_cfg, tag="system-errors")
    threshold = float(args.tiou_threshold if args.tiou_threshold is not None else 0.5)
    median_fps = float(np.median([float(r.pose.fps) for r in records])) if records else 24.0
    lam_s = lambda_min_frames(inference_cfg) / median_fps
    cap_s = float(inference_cfg.get("buffer_cap_s", 18.0))

    out = system_outcomes(records, predicted, threshold, lam_s)
    gold_rows, event_counts = out["gold_rows"], out["event_counts"]
    score_outcomes(gold_rows, char_level_for_target(target_language(data_cfg, args.language)))
    pairs = [r for r in gold_rows if r["class"] == "matched"]
    gold_counts = {c: sum(1 for r in gold_rows if r["class"] == c) for c in ("matched", "merged", "split", "missed")}

    def _mean_bleu(rows): return round(float(np.mean([r["bleu4"] for r in rows])), 2) if rows else None
    tiou_bins = [(threshold, 0.7), (0.7, 0.9), (0.9, 1.01)]
    decomposition = [{
        "tiou_bin": f"[{lo:g},{min(hi, 1.0):g})", "n": len(sel), "mean_bleu4": _mean_bleu(sel),
        "mean_abs_delta_head_s": round(float(np.mean([abs(r["delta_head_s"]) for r in sel])), 2) if sel else None,
        "mean_abs_delta_tail_s": round(float(np.mean([abs(r["delta_tail_s"]) for r in sel])), 2) if sel else None,
    } for lo, hi in tiou_bins for sel in [[r for r in pairs if lo <= r["tiou"] < hi]]]

    edges = [0.0, 2.0, 5.0, 10.0, 20.0, cap_s, float("inf")]
    dur_bins = []
    for lo, hi in zip(edges, edges[1:]):
        if hi <= lo: continue
        sel = [g for g in gold_rows if lo <= g["duration_s"] < hi]
        if not sel: continue
        mix = {c: sum(1 for g in sel if g["class"] == c) for c in gold_counts}
        dur_bins.append({
            "bin_s": f"[{lo:g},{'cap' if hi == cap_s else ('inf' if hi == float('inf') else f'{hi:g}')})",
            "n_gold": len(sel), "matched_rate": round(mix["matched"] / len(sel), 3),
            "class_mix": mix, "mean_bleu4": _mean_bleu([r for r in sel if r["class"] == "matched"])
        })
    k = 5
    case_studies = {cls: sorted(
        (r for r in gold_rows if r["class"] == cls), key=lambda r: -r["ref_words"]
    )[:k] for cls in ("missed", "merged", "split")}
    case_studies["phantom"] = out["phantom_events"][:k]
    case_studies["forced_partial"] = out["partial_events"][:k]
    case_studies["worst_matched"] = sorted(pairs, key=lambda r: (r["bleu4"], -r["duration_s"]))[:k]

    stem = Path(args.predictions).stem
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"system_outcomes_{stem}.csv"
    cols = ["gold_id", "video_id", "class", "start_s", "end_s", "duration_s", "ref_words",
            "tiou", "delta_head_s", "delta_tail_s", "bleu4", "localized_bleu4", "ref", "hyp"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in gold_rows: w.writerow(r)

    n_gold = max(1, len(gold_rows))
    payload = {
        "language": args.language, "split": args.split, "predictions": str(args.predictions),
        "tiou_threshold": threshold, "material_overlap_s": round(lam_s, 3),
        "gold_taxonomy": {"counts": gold_counts, "rates": {c: round(v / n_gold, 3) for c, v in gold_counts.items()}},
        "event_taxonomy": {"counts": event_counts, "forced_partial": len(out["partial_events"])},
        "matched_mean_bleu4": _mean_bleu(pairs), "mean_localized_bleu4": round(float(np.mean([r["localized_bleu4"] for r in gold_rows])), 2),
        "decomposition_by_tiou": decomposition, "duration_bins": dur_bins, "case_studies": case_studies, "outcomes_csv": str(csv_path),
    }
    output = Path(args.output or out_dir / f"system_errors_{stem}.json")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[system-errors] gold: {gold_counts} (n={len(gold_rows)}) | events: {event_counts} "
          f"partial={len(out['partial_events'])} | outcomes -> {csv_path}", flush=True)
    payload["output"] = str(output)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Misaligned-SLT analysis utilities")
    parser.add_argument("--stage", default="dataset-summary", choices=[
        "dataset-summary", "segmenter-infer", "tune-decode", "tune-stream", "segmenter-errors", "buffer-cap", "delta-enc", "system-errors"
    ])
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--segmenter-arch", default="moryossef", choices=["moryossef", "s1"],
        help="segmenter-infer backend: moryossef = external Moryossef segmenter, s1 = in-system BIO head"
    )
    parser.add_argument("--moryossef-config", default="configs/moryossef26.yaml", help="Moryossef analysis-segmenter config")
    parser.add_argument(
        "--segmenter-decode", default=None, choices=["duration", "plain"],
        help="whole-video decode; default per arch: s1 -> duration (our semi-Markov re-split), moryossef -> plain (Moryossef argmax)"
    )
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
    parser.add_argument("--grid-bias", type=float, nargs="+", default=None, help="tune-stream: split_bias values to sweep")
    parser.add_argument("--grid-lag", type=float, nargs="+", default=None, help="tune-stream: commit_lag_s values to sweep")
    parser.add_argument("--noise-sigma", type=float, default=0.005, help="delta-enc keypoint-noise std (normalized coords)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write-config", action="store_true",
        help="Persist measured constants: buffer-cap -> buffer_cap_s; delta-enc -> delta_enc_frames + "
        "min_span_frames (both configs); tune-decode -> duration_decode_<arch>.<language>"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-test", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.language is None: args.language = str(load_yaml(args.data_config).get("active_languages", ["asf"])[0])
    if args.stage == "dataset-summary": result = dataset_summary(args)
    elif args.stage == "segmenter-infer": result = segmenter_infer(args)
    elif args.stage == "tune-decode": result = tune_decode(args)
    elif args.stage == "tune-stream": result = tune_stream(args)
    elif args.stage == "segmenter-errors":
        if not args.predictions: raise SystemExit("--predictions is required for --stage segmenter-errors")
        result = segmenter_errors(args)
    elif args.stage == "buffer-cap": result = buffer_cap(args)
    elif args.stage == "delta-enc": result = delta_enc(args)
    elif args.stage == "system-errors": result = system_errors(args)
    else: raise ValueError(f"Unsupported stage: {args.stage}")
    print(json.dumps(result, indent=2, sort_keys=True))
