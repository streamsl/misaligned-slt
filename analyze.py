from __future__ import annotations
from dataclasses import asdict, dataclass
from statistics import median
from pathlib import Path
import json, argparse, math

import torch
import numpy as np
from data.loader import load_language_records
from poses import load_pose_window

from infer.commit_gate import first_terminator_index
from eval import _build_eval_model, _load_segmenter, _translate_window, load_prediction_file, save_prediction_file
from metrics import Segment, match_segments, temporal_iou, compute_text_metrics
from utils import load_yaml, update_yaml_scalar, pick_device, pretrained_checkpoint, checkpoint_dir

# Analysis-A pred↔GT matching bar: deliberately LOW so near-misses count as matched pairs feeding the 
# (Δ_head, Δ_tail) jitter CDF, not as phantom/skip events. A high bar biases the DF toward near-zero 
# offsets and miscalibrates the mode ratios. A corpus-level analysis knob, not a segmenter setting, 
# so it lives here; override per-run with --tiou-threshold.
ANALYSIS_A_MATCH_TIOU = 0.1


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
    # Relative position in (0,1) of each spurious internal cut the segmenter placed inside an
    # over-segmented GT sentence. This is what Mode 2 needs (where truncation lands), and it is NOT
    # captured by the matched-pair (Δ_head, Δ_tail) CDF — those are one-to-one boundary noise. The
    # Mode-2 window sampler draws its cut depth from this distribution (§5.0/§5.2); uniform is only a
    # last-resort fallback when no over-segmentation was observed.
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
    # Per streaming_slt_prompt.md §5.5: skip mass is split between
    # truncated-window and multi-complete-window training cases.
    skipped_half = 0.5 * float(counts["skipped"])
    return {
        "mode1": float(counts["matched"]),
        "mode2": float(counts["oversegmentation"]) + skipped_half,
        "mode3": float(counts["undersegmentation"]) + skipped_half,
        "mode4": float(counts["phantom"]),
    }

def analyze_segmenter_errors(
    predicted: dict[str, list[Segment]], gold: dict[str, list[Segment]],
    durations: dict[str, float], tiou_threshold: float = 0.1,
) -> SegmenterErrorAnalysis:
    """Compute Analysis-A event counts and regular-match jitter samples.

    Boundary-jitter samples intentionally exclude over-segmented GT spans and under-segmenting predicted spans. 
    Those are separate window modes, not ordinary boundary noise around a one-to-one segment.
    """
    jitter_samples: list[JitterSample] = []
    overseg_cut_positions: list[float] = []
    counts = {"matched": 0, "oversegmentation": 0, "undersegmentation": 0, "skipped": 0, "phantom": 0}
    matched_pairs, regular_matches = 0, 0

    for video_id in sorted(set(gold) | set(predicted)):
        pred_segments = list(predicted.get(video_id, []))
        gold_segments = list(gold.get(video_id, []))
        matches = match_segments(pred_segments, gold_segments, threshold=tiou_threshold)
        matched_pairs += len(matches)
        matched_gold = {gold_idx for _, gold_idx, _ in matches}

        overlapping_pred_by_gold: dict[int, list[int]] = {}
        overlapping_gold_by_pred: dict[int, list[int]] = {}
        for pred_idx, pred in enumerate(pred_segments):
            for gold_idx, gt in enumerate(gold_segments):
                overlap = max(0.0, min(pred.end_s, gt.end_s) - max(pred.start_s, gt.start_s))
                if overlap > 0:
                    overlapping_pred_by_gold.setdefault(gold_idx, []).append(pred_idx)
                    overlapping_gold_by_pred.setdefault(pred_idx, []).append(gold_idx)

        overseg_gold = {
            gold_idx for gold_idx, pred_indices in overlapping_pred_by_gold.items()
            if len(pred_indices) >= 2
        }
        underseg_pred = {
            pred_idx for pred_idx, gold_indices in overlapping_gold_by_pred.items()
            if len(gold_indices) >= 2
        }
        underseg_gold = {
            gold_idx for pred_idx in underseg_pred
            for gold_idx in overlapping_gold_by_pred.get(pred_idx, [])
        }
        phantom_pred = {
            pred_idx for pred_idx, pred in enumerate(pred_segments)
            if pred_idx not in overlapping_gold_by_pred
            and pred.start_s >= 0.0 and pred.end_s <= float(durations.get(video_id, pred.end_s))
        }
        counts["oversegmentation"] += len(overseg_gold)
        counts["undersegmentation"] += len(underseg_pred)
        counts["phantom"] += len(phantom_pred)
        counts["skipped"] += len(set(range(len(gold_segments))) - matched_gold - overseg_gold - underseg_gold)

        # Record where each spurious internal cut fell, relative to the over-segmented GT span.
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


def write_analysis_a_outputs(analysis: SegmenterErrorAnalysis, output_dir: str | Path, language: str) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jitter_rows = [asdict(sample) for sample in analysis.jitter_samples]
    head = [row["delta_head_s"] for row in jitter_rows]
    tail = [row["delta_tail_s"] for row in jitter_rows]
    jitter_payload = {
        "language": language, "samples": jitter_rows,
        "laplace": {"head": _laplace_fit(head), "tail": _laplace_fit(tail)},
        # Mode-2 truncation-depth distribution (relative cut positions of over-segmentation events).
        # The window sampler reads this; if empty it falls back to a uniform interior cut.
        "overseg_cut_positions": analysis.overseg_cut_positions,
    }
    mode_payload = {
        "language": language,
        "mode_ratios": analysis.mode_ratios,
        "source_event_counts": analysis.event_counts,
        "source_weights": mode_weights_from_events(analysis.event_counts),
    }
    taxonomy_payload = {
        "language": language,
        "event_counts": analysis.event_counts,
        "matched_pairs": analysis.matched_pairs,
        "regular_matches": analysis.regular_matches,
        "videos": analysis.videos,
        "moryossef_2020_mapping": {
            "matched": "Started Pre/Post-Signing and Signing Underflow/Overflow",
            "oversegmentation": "Signing Undetected Incorrectly",
            "undersegmentation": "Bridged",
            "skipped": "Skipped",
            "phantom": "Signing Detected Incorrectly",
        },
    }
    paths = {
        "jitter": str(output_dir / f"a_jitter_{language}.json"),
        "mode_ratios": str(output_dir / f"a_mode_ratios_{language}.json"),
        "taxonomy": str(output_dir / f"a_error_taxonomy_{language}.json"),
    }
    Path(paths["jitter"]).write_text(json.dumps(jitter_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["mode_ratios"]).write_text(json.dumps(mode_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["taxonomy"]).write_text(json.dumps(taxonomy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def dataset_summary(args: argparse.Namespace) -> dict:
    cfg = load_yaml(args.data_config)
    records, splits = load_language_records(cfg, args.language, split=args.split)
    durations = [span.duration_s for rec in records for span in rec.sentences]
    return {
        "language": args.language, "split": args.split or "all", "records": len(records),
        "sentences": len(durations), "split_sizes": {k: len(v) for k, v in splits.items()},
        "mean_sentence_s": sum(durations) / len(durations) if durations else 0.0,
        "max_sentence_s": max(durations) if durations else 0.0,
    }


def segmenter_infer(args: argparse.Namespace) -> dict:
    """Predicted phrase segments on a split — the upstream segmenter for Analysis A / Analysis B / the RQ2 cascade."""
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to run segmenter inference on test without --allow-test")
    from moryossef26.infer import predict_phrase_segments
    records, _ = load_language_records(load_yaml(args.data_config), args.language, split=args.split)
    model, device, velocity, rope_chunk, checkpoint = _load_segmenter(args)
    print(f"[segmenter-infer] {args.segmenter_arch} segmenter from {checkpoint}", flush=True)

    predictions = predict_phrase_segments(model, records, device=device, velocity=velocity, rope_chunk=rope_chunk)
    output = Path(args.output or f"outputs/segmenter_predictions_{args.segmenter_arch}_{args.language}_{args.split}.json")
    save_prediction_file(predictions, output)
    return {
        "language": args.language, "split": args.split, "videos": len(records),
        "segmenter_arch": args.segmenter_arch, "checkpoint": checkpoint,
        "predicted_segments": sum(len(v) for v in predictions.values()), "output": str(output),
    }


def analysis_a(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Analysis A must run on dev; use --allow-test only for smoke debugging")
    cfg = load_yaml(args.data_config)
    records, _ = load_language_records(cfg, args.language, split=args.split)
    predictions = load_prediction_file(args.predictions)  # the segmenter-infer output file
    gold_segments = {
        record.video_id: [Segment(span.start_s, span.end_s) for span in record.sentences]
        for record in records
    }
    durations = {record.video_id: float(record.pose.duration_s) for record in records}
    analysis = analyze_segmenter_errors(
        predicted=predictions, gold=gold_segments, durations=durations,
        tiou_threshold=float(args.tiou_threshold if args.tiou_threshold is not None else ANALYSIS_A_MATCH_TIOU),
    )
    paths = write_analysis_a_outputs(analysis, args.output_dir, args.language)
    return {
        "language": args.language, "split": args.split, "event_counts": analysis.event_counts, 
        "mode_ratios": analysis.mode_ratios, "matched_pairs": analysis.matched_pairs,
        "regular_matches": analysis.regular_matches, "outputs": paths,
    }


def _eval_model_for(method: str, args: argparse.Namespace, method_cfg: dict, device: torch.device):
    # Reuse eval.py's checkpoint-loading model builder (plain scalars — no fake namespace).
    data_cfg = load_yaml(args.data_config)
    return _build_eval_model(method, args.checkpoint, args.language, data_cfg, method_cfg, device)


def analysis_b(args: argparse.Namespace) -> dict:
    """Analysis B (§8.2) — the paper's MOTIVATING experiment: a CLEAN SLT model degrades under a REALISTIC
    upstream segmenter's boundary errors. Runs on dev, before SLT training, and is method-INDEPENDENT (the clean
    baseline is not our method). Produces the clean-vs-realistic headline table.

    Realistic point: translate the windows the EXTERNAL segmenter actually cut (`--predictions`, the segmenter-infer
    output), with each window's reference = the GT sentence it most overlaps. Clean point: translate the GT-trimmed
    windows. The controlled severity CURVE (§8.2 Step 2b / §9.1 no-robustness floor) is the SAME machinery as
    `eval.py --rq 1 --method baseline --split dev` — run that for the curve; this stage assembles the realistic gap.
    """
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Analysis B runs on dev; --allow-test only for smoke debugging")
    if not args.predictions:
        raise SystemExit("--predictions required: the external segmenter's spans (analyze.py --stage segmenter-infer)")
    data_cfg = load_yaml(args.data_config)
    base_cfg = load_yaml(args.baseline_config)
    inference_cfg = load_yaml(args.inference_config)
    device = pick_device(args.device)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    model, tokenizer = _eval_model_for("baseline", args, base_cfg, device)

    def _translate(start_s, end_s, rec):
        poses, ts = load_pose_window(rec.pose, start_s, end_s, normalize=True)
        if poses.shape[0] == 0: return None
        text, _ = _translate_window(model=model, tokenizer=tokenizer, method="baseline",
                                    poses_np=poses, timestamps_np=ts, start_s=start_s,
                                    device=device, inference_cfg=inference_cfg, method_cfg=base_cfg)
        return text

    # Clean point: GT-trimmed windows.
    clean_pred, clean_ref = [], []
    for rec in records:
        for s in rec.sentences:
            t = _translate(float(s.start_s), float(s.end_s), rec)
            if t is not None: clean_pred.append(t); clean_ref.append(s.text)

    # Realistic point: the external segmenter's predicted spans; reference = the max-overlap GT sentence.
    predicted = load_prediction_file(args.predictions)
    real_pred, real_ref = [], []
    by_id = {r.video_id: r for r in records}
    for vid, spans in predicted.items():
        rec = by_id.get(vid)
        if rec is None: continue
        for span in spans:
            best, best_ov = None, 0.0
            for gt in rec.sentences:
                ov = max(0.0, min(span.end_s, gt.end_s) - max(span.start_s, gt.start_s))
                if ov > best_ov: best_ov, best = ov, gt
            if best is None: continue  # phantom span in a gap → no GT reference; excluded from the corpus score
            t = _translate(float(span.start_s), float(span.end_s), rec)
            if t is not None: real_pred.append(t); real_ref.append(best.text)

    clean = compute_text_metrics(clean_pred, clean_ref, prefix="clean") if clean_pred else {}
    realistic = compute_text_metrics(real_pred, real_ref, prefix="realistic") if real_pred else {}
    payload = {
        "language": args.language, "split": args.split, "clean": clean, "realistic": realistic,
        "clean_windows": len(clean_pred), "realistic_windows": len(real_pred),
        "delta_bleu4": float(clean.get("clean_bleu4", 0.0) - realistic.get("realistic_bleu4", 0.0)),
        "note": "Controlled severity curve = eval.py --rq 1 --method baseline --split dev (the §9.1 no-robustness floor).",
    }
    output = Path(args.output or f"outputs/analysis_b_{args.language}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["output"] = str(output)
    return payload


def tail_benefit(args: argparse.Namespace) -> dict:
    """Tail-benefit curve sets BUFFER_CAP_S.

    Protocol: clean-trained translator (Analysis-B clean baseline), dev split, head fixed at true sentence start (Δ_head = 0), trailing 
    context Δ_tail swept upward; BLEU-4 per Δ_tail. The elbow is 1st grid point whose marginal BLEU per extra second drops below explicit 
    latency/quality coefficient (eval.yaml tail_benefit.latency_quality_coeff_bleu_per_s) — not hand-picked number or %-of-clean threshold.

    Buffer-cap semantics (spec ambiguity resolved, documented): the spec says "this elbow is the buffer cap", but the FSM buffer must hold 
    the WHOLE sentence plus the trailing context, so a 1–3 s tail elbow alone cannot be the cap. We persist buffer_cap_s = p99 sentence 
    duration + elbow (both raw values are in the JSON artifact for the paper).
    """
    if args.split == "test" and not args.allow_test: raise SystemExit("Tail-benefit runs on dev; --allow-test only for smoke debugging")
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    base_cfg = load_yaml(args.baseline_config)
    inference_cfg = load_yaml(args.inference_config)
    tb_cfg = eval_cfg.get("tail_benefit", {})
    grid = [float(x) for x in tb_cfg.get("tail_grid_s", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])]
    coeff = float(tb_cfg.get("latency_quality_coeff_bleu_per_s", 0.5))

    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    device = pick_device(args.device)
    model, tokenizer = _eval_model_for("baseline", args, base_cfg, device)

    durations = [span.duration_s for rec in records for span in rec.sentences]
    p99_duration = float(np.percentile(durations, 99)) if durations else 0.0
    sentences = [(rec, span) for rec in records for span in rec.sentences]
    if args.num_sentences: sentences = sentences[: int(args.num_sentences)]

    per_tail: dict[float, dict[str, list[str]]] = {dt: {"predictions": [], "references": []} for dt in grid}
    for rec, span in sentences:
        for dt in grid:
            end_s = min(rec.pose.duration_s, span.end_s + dt)
            poses, timestamps = load_pose_window(rec.pose, span.start_s, end_s, normalize=True)
            if poses.shape[0] == 0: continue
            text, _ = _translate_window(
                model=model, tokenizer=tokenizer, method="baseline",
                poses_np=poses, timestamps_np=timestamps, start_s=span.start_s,
                device=device, inference_cfg=inference_cfg, method_cfg=base_cfg,
            )
            per_tail[dt]["predictions"].append(text)
            per_tail[dt]["references"].append(span.text)

    curve = []
    for dt in grid:
        bleu = compute_text_metrics(per_tail[dt]["predictions"], per_tail[dt]["references"])["translation_bleu4"]
        curve.append({"delta_tail_s": dt, "bleu4": float(bleu), "n": len(per_tail[dt]["predictions"])})

    elbow = grid[-1]
    for prev, cur in zip(curve, curve[1:]):
        gap_s = cur["delta_tail_s"] - prev["delta_tail_s"]
        marginal = (cur["bleu4"] - prev["bleu4"]) / gap_s if gap_s > 0 else 0.0
        if marginal < coeff:
            elbow = prev["delta_tail_s"]
            break

    buffer_cap_s = round(p99_duration + elbow, 2)
    payload = {
        "language": args.language, "split": args.split, "sentences": len(sentences),
        "latency_quality_coeff_bleu_per_s": coeff, "curve": curve,
        "elbow_tail_s": elbow, "p99_sentence_duration_s": p99_duration, "buffer_cap_s": buffer_cap_s,
    }
    output = Path(args.output or f"outputs/tail_benefit_{args.language}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_config and update_yaml_scalar(args.inference_config, ("buffer_cap_s",), buffer_cap_s):
        payload["config_updated"] = args.inference_config
    payload["output"] = str(output)
    return payload


def delta_enc(args: argparse.Namespace) -> dict:
    """BIO temporal noise floor sets the commit gate's delta_enc.

    Uses the S1 in-system BIO head (the DEPLOYED terminator estimator the FSM runs; §4.4/§7.1), NOT the Moryossef
    analysis segmenter and NOT the DLM checkpoint. Two reasons it must be the S1 head: (1) δ_enc calibrates the
    commit gate's cut overlap against the noise of the head the FSM actually contains — the analysis segmenter never
    runs in the FSM; (2) δ_enc is a gate-GEOMETRY constant needed to TRAIN the DLM (dlm.yaml δ is asserted == this),
    so it must be measured from the head as it ENTERS stage 2 (checkpoints/bio_s1, produced by train-bio), before
    the DLM exists — reading it off the DLM checkpoint is an ordering circularity.

    Runs that head twice per dev sentence window under small input perturbations and measures how far the predicted
    terminator index (first_terminator_index — first O-or-B, the statistic the commit gate tracks) moves. Two
    perturbation families (the spec names none, so both are reported): (a) drop the leading frame — the stride-phase
    misalignment a growing buffer produces; (b) Gaussian keypoint noise sigma on x/y — pose-estimator jitter.
    delta_enc = ceil(p90 over both families): boundary movement below the head's own noise floor must not block a commit.
    """
    if args.split == "test" and not args.allow_test: raise SystemExit("Delta-enc runs on dev; --allow-test only for smoke debugging")
    from train.bio_pretrain import build_bio_s1_model
    data_cfg = load_yaml(args.data_config)
    cfg = load_yaml(args.bio_config)

    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    device = pick_device(args.device)
    model = build_bio_s1_model(cfg)
    checkpoint = args.checkpoint or str(Path(checkpoint_dir(cfg, default=f"checkpoints/bio_s1/{args.language}")) / "model.pt")
    blob = torch.load(checkpoint, map_location="cpu")

    model.load_state_dict(blob.get("model", blob) if isinstance(blob, dict) else blob, strict=True)
    model.eval().to(device)
    print(f"[delta-enc] S1 BIO head from {checkpoint}", flush=True)
    sigma = float(args.noise_sigma)

    sentences = [(rec, span) for rec in records for span in rec.sentences]
    if args.num_sentences: sentences = sentences[: int(args.num_sentences)]
    rng = np.random.default_rng(int(args.seed))

    @torch.no_grad()
    def closing_index(poses_np: np.ndarray, timestamps_np: np.ndarray, start_s: float) -> int | None:
        poses = torch.as_tensor(poses_np, dtype=torch.float32, device=device).unsqueeze(0)
        ts = torch.as_tensor(timestamps_np - start_s, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.ones(poses.shape[:2], dtype=torch.bool, device=device)
        tags = model(poses, mask, timestamps_s=ts).logits.argmax(dim=-1)[0]
        return first_terminator_index(tags)

    shifts: dict[str, list[int]] = {"drop_first_frame": [], "keypoint_noise": []}
    for rec, span in sentences:
        start_s = max(0.0, span.start_s - 1.0)
        end_s = min(rec.pose.duration_s, span.end_s + 1.0)
        poses, timestamps = load_pose_window(rec.pose, start_s, end_s, normalize=True)
        if poses.shape[0] < 3: continue
        base = closing_index(poses, timestamps, start_s)
        if base is None: continue

        dropped = closing_index(poses[1:], timestamps[1:], start_s)
        # The dropped buffer's indices sit one frame earlier on the original timeline.
        if dropped is not None: shifts["drop_first_frame"].append(abs((dropped + 1) - base))
        noisy = poses.copy()
        noisy[..., :2] = noisy[..., :2] + rng.normal(0.0, sigma, size=noisy[..., :2].shape).astype(noisy.dtype)
        perturbed = closing_index(noisy, timestamps, start_s)
        if perturbed is not None: shifts["keypoint_noise"].append(abs(perturbed - base))

    stats = {family: {
        "n": len(values), "median": float(np.median(values)) if values else 0.0,
        "p90": float(np.percentile(values, 90)) if values else 0.0, "max": int(max(values)) if values else 0,
    } for family, values in shifts.items()}
    delta = int(math.ceil(max((s["p90"] for s in stats.values()), default=0.0))) or 1
    payload = {
        "language": args.language, "split": args.split, "noise_sigma": sigma,
        "sentences": len(sentences), "families": stats, "delta_enc_frames": delta,
    }
    output = Path(args.output or f"outputs/delta_enc_{args.language}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_config and update_yaml_scalar(args.inference_config, ("boundary_stability", "delta_enc_frames"), delta):
        payload["config_updated"] = args.inference_config
    payload["output"] = str(output)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Misaligned-SLT analysis utilities")
    parser.add_argument(
        "--stage", default="dataset-summary",
        choices=["dataset-summary", "segmenter-infer", "analysis-a", "analysis-b", "tail-benefit", "delta-enc"],
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--segmenter-arch", default="external", choices=["external", "s1"],
        help="segmenter-infer backend: external = independent chunk-trained segmenter (default), s1 = in-system head (ablation)"
    )
    parser.add_argument("--segmenter-config", default="configs/moryossef26.yaml", help="external segmenter config")
    parser.add_argument("--bio-config", default="configs/bio_pretrain.yaml", help="S1 (in-system head) config for --segmenter-arch s1")
    parser.add_argument("--slt-config", default="configs/dlm.yaml")
    parser.add_argument("--baseline-config", default="configs/baseline.yaml")
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
    parser.add_argument("--noise-sigma", type=float, default=0.005, help="delta-enc keypoint-noise std (normalized coords)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-config", action="store_true", help="Persist the measured constant into configs/inference.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-test", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.language is None: args.language = str(load_yaml(args.data_config).get("active_languages", ["csl"])[0])
    if args.stage == "dataset-summary": result = dataset_summary(args)
    elif args.stage == "segmenter-infer": result = segmenter_infer(args)
    elif args.stage == "analysis-a":
        if not args.predictions: raise SystemExit("--predictions is required for --stage analysis-a")
        result = analysis_a(args)
    elif args.stage == "analysis-b": result = analysis_b(args)
    elif args.stage == "tail-benefit": result = tail_benefit(args)
    elif args.stage == "delta-enc": result = delta_enc(args)
    else: raise ValueError(f"Unsupported stage: {args.stage}")
    print(json.dumps(result, indent=2, sort_keys=True))
