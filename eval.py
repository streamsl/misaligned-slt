from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import json

import torch
import numpy as np
from data.loader import VideoRecord, load_language_records
from data.windowing import SentenceSpan
from poses import load_pose_window

from models.gfslt import GFSLTConfig, CleanARSLTModel, resolve_decoder_start_id
from models.streaming_slt import MisalignedSLTModel
from models.checkpointing import load_model_checkpoint
from metrics import Segment, match_segments, segmentation_prf, compute_text_metrics
from utils import checkpoint_dir, load_yaml, mbart_trimmed_dir


@dataclass(frozen=True)
class PredictionEvent:
    video_id: str
    start_s: float
    end_s: float
    text: str | None = None
    flagged_partial: bool = False
    commit_time_s: float | None = None  # stride time the commit fired (streaming only)

    @property
    def segment(self) -> Segment:
        return Segment(self.start_s, self.end_s)


@dataclass(frozen=True)
class ControlledWindow:
    video_id: str
    reference: str
    gt_start_s: float
    gt_end_s: float
    window_start_s: float
    window_end_s: float
    delta_head_s: float
    delta_tail_s: float


def _load_segments(path: str | Path) -> list[Segment]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Segment(float(row["start_s"]), float(row["end_s"])) for row in rows]


def _gold_events(records: list[VideoRecord]) -> dict[str, list[PredictionEvent]]:
    return {record.video_id: [PredictionEvent(
        video_id=record.video_id, start_s=float(span.start_s), end_s=float(span.end_s), text=span.text,
    ) for span in record.sentences] for record in records}


def _segment_from_any(value: Any, video_id: str | None = None) -> PredictionEvent:
    if isinstance(value, dict):
        vid = str(value.get("video_id", video_id or ""))
        text = value.get("text", value.get("prediction", value.get("translation")))
        commit = value.get("commit_time_s")
        return PredictionEvent(
            video_id=vid, start_s=float(value["start_s"]), end_s=float(value["end_s"]),
            text=str(text) if text is not None else None,
            flagged_partial=bool(value.get("flagged_partial", False)),
            commit_time_s=float(commit) if commit is not None else None,
        )
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        if video_id is None: raise ValueError("Tuple/list prediction rows require an enclosing video_id")
        return PredictionEvent(video_id=video_id, start_s=float(value[0]), end_s=float(value[1]))
    raise TypeError(f"Unsupported prediction segment row: {value!r}")


def load_event_predictions(path: str | Path) -> dict[str, list[PredictionEvent]]:
    # Load predicted segments, optionally with text, from common JSON layouts.
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, list[PredictionEvent]] = {}
    if isinstance(data, dict):
        for video_id, segments in data.items():
            out[str(video_id)] = [_segment_from_any(row, video_id=str(video_id)) for row in segments]
        return out

    if not isinstance(data, list): raise TypeError("Prediction file must be a dict or list JSON payload")
    for row in data:
        if isinstance(row, dict) and "segments" in row:
            video_id = str(row["video_id"])
            out[video_id] = [_segment_from_any(seg, video_id=video_id) for seg in row["segments"]]
        else:
            event = _segment_from_any(row)
            out.setdefault(event.video_id, []).append(event)
    return out


def controlled_windows(records: list[VideoRecord], grid_s: list[float], max_sentences: int | None = None) -> list[ControlledWindow]:
    """Build RQ1 signed-offset windows.

    Analysis A defines deltas as predicted boundary minus GT boundary, so the perturbed window uses start = gt_start + delta_head and 
    end = gt_end + delta_tail. Negative delta_tail is therefore the intended right-truncation stress test.
    """
    windows: list[ControlledWindow] = []
    count = 0
    for record in records:
        for span in record.sentences:
            if max_sentences is not None and count >= int(max_sentences): return windows
            count += 1
            for dh in grid_s:
                for dt in grid_s:
                    start_s = max(0.0, float(span.start_s) + float(dh))
                    end_s = min(float(record.pose.duration_s), float(span.end_s) + float(dt))
                    if end_s <= start_s: continue
                    windows.append(ControlledWindow(
                        video_id=record.video_id, reference=span.text,
                        gt_start_s=float(span.start_s), gt_end_s=float(span.end_s),
                        window_start_s=start_s, window_end_s=end_s,
                        delta_head_s=float(dh), delta_tail_s=float(dt),
                    ))
    return windows


def evaluate_predicted_events(
    predicted: dict[str, list[PredictionEvent]],
    gold: dict[str, list[PredictionEvent]],
    thresholds: list[float],
) -> dict[str, Any]:
    results = []
    all_video_ids = sorted(set(predicted) | set(gold))
    for threshold in thresholds:
        total_pred, total_gold = 0, 0
        pred_texts: list[str] = []
        ref_texts: list[str] = []
        latencies: list[float] = []
        matched_pairs = 0

        for video_id in all_video_ids:
            pred_events = predicted.get(video_id, [])
            gold_events = gold.get(video_id, [])
            pred_by_idx = [event.segment for event in pred_events]
            gold_by_idx = [event.segment for event in gold_events]
            total_pred += len(pred_by_idx)
            total_gold += len(gold_by_idx)

            matches = match_segments(pred_by_idx, gold_by_idx, threshold=float(threshold))
            matched_pairs += len(matches)
            for pred_idx, gold_idx, _ in matches:
                pred_event = pred_events[pred_idx]
                gold_text = gold_events[gold_idx].text
                if pred_event.text is not None and gold_text is not None:
                    pred_texts.append(pred_event.text)
                    ref_texts.append(gold_text)
                if pred_event.commit_time_s is not None: # Emission latency: commit time minus GT sentence end (spec §9.2).
                    latencies.append(float(pred_event.commit_time_s) - float(gold_events[gold_idx].end_s))

        precision = matched_pairs / total_pred if total_pred else 0.0
        recall = matched_pairs / total_gold if total_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        latency_block = {}
        if latencies:
            arr = np.sort(np.asarray(latencies, dtype=np.float64))
            latency_block = {
                "median_latency_s": float(np.median(arr)),
                "p90_latency_s": float(arr[min(len(arr) - 1, int(round(0.9 * (len(arr) - 1))))]),
                "n": len(arr),
            }
        results.append({
            "tiou_threshold": float(threshold),
            "segmentation": {
                "precision": precision, "recall": recall,
                "f1": f1, "matches": float(matched_pairs),
            },
            "matched_segments": matched_pairs,
            "matched_translation_pairs": len(pred_texts),
            "translation_metrics": compute_text_metrics(pred_texts, ref_texts),
            "emission_latency": latency_block,
        })
    return {"thresholds": results}


def _parse_grid(value: str | None, fallback: list[float]) -> list[float]:
    if not value: return [float(x) for x in fallback]
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _method_config_path(args: argparse.Namespace) -> str:
    if args.method_config: return str(args.method_config)
    defaults = {
        "stage2_baseline": "configs/stage2_baseline.yaml",
        "stage2_ar": "configs/stage2_ar.yaml",
        "stage2_dlm": "configs/stage2_dlm.yaml",
    }
    return defaults[args.method]


def _load_tokenizer(stage1_cfg: dict, data_cfg: dict, language: str):
    from transformers import AutoTokenizer
    target_lang = data_cfg["languages"][language].get("target_lang", "en_XX")
    return AutoTokenizer.from_pretrained(mbart_trimmed_dir(stage1_cfg), src_lang=target_lang, tgt_lang=target_lang)


def _gfslt_config(stage1_cfg: dict, method_cfg: dict) -> GFSLTConfig:
    return GFSLTConfig(
        embed_dim=int(stage1_cfg.get("embed_dim", 1024)),
        hidden_size=int(stage1_cfg.get("hidden_size", 1024)),
        temporal_kernel=int(stage1_cfg.get("temporal_kernel", 3)),
        mbart_name=mbart_trimmed_dir(stage1_cfg),  # same trimmed mBART training used
        use_temporal_conv=bool(method_cfg.get("use_temporal_conv", stage1_cfg.get("use_temporal_conv", False))),
    )


def _build_eval_model(args: argparse.Namespace, data_cfg: dict, stage1_cfg: dict, method_cfg: dict, device: torch.device):
    tokenizer = _load_tokenizer(stage1_cfg, data_cfg, args.language)
    if args.method == "stage2_baseline":
        model = CleanARSLTModel(_gfslt_config(stage1_cfg, method_cfg), decoder_start_token_id=resolve_decoder_start_id(tokenizer))
    else:
        decoder = "ar" if args.method == "stage2_ar" else "dlm"
        model = MisalignedSLTModel(
            gfslt_config=_gfslt_config(stage1_cfg, method_cfg),
            tokenizer=tokenizer,
            decoder=decoder,
            bio_hidden_dim=int(method_cfg.get("bio_hidden_dim", 384)),
            block_size=int(method_cfg.get("block_size", 8)),
        )
    checkpoint = Path(args.checkpoint or checkpoint_dir(method_cfg, default="") or "")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for {args.method}: {checkpoint}. Train the method first or pass --checkpoint.")
    load_model_checkpoint(model, checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def _translate_window(
    model, tokenizer, method: str,
    poses_np: np.ndarray, timestamps_np: np.ndarray, start_s: float,
    device: torch.device, inference_cfg: dict, method_cfg: dict,
) -> tuple[str, float]:
    poses = torch.as_tensor(poses_np, dtype=torch.float32, device=device).unsqueeze(0)
    timestamps = torch.as_tensor(timestamps_np - float(start_s), dtype=torch.float32, device=device).unsqueeze(0)
    frame_mask = torch.ones(poses.shape[:2], dtype=torch.bool, device=device)
    max_tokens = int(method_cfg.get("max_text_tokens", inference_cfg.get("translation", {}).get("max_text_tokens", 128)))
    if method == "stage2_baseline":
        tokens = model.generate(poses=poses, frame_mask=frame_mask, timestamps_s=timestamps, max_new_tokens=max_tokens)
        confidence = torch.ones(tokens.shape, dtype=torch.float32, device=device)
    else:
        trans_cfg = inference_cfg.get("translation", {})
        dcd_cfg = trans_cfg.get("dcd", method_cfg.get("dcd", {}))
        spd_cfg = method_cfg.get("spd", {})
        _, tokens, confidence = model.generate_from_poses(
            poses=poses, frame_mask=frame_mask, timestamps_s=timestamps, max_text_tokens=max_tokens,
            diffusion_steps=int(trans_cfg.get("diffusion_steps", method_cfg.get("diffusion_steps", 64))),
            tau_dec=float(dcd_cfg.get("tau_dec", trans_cfg.get("commit_confidence_tau", 0.75))),
            spd_top_k=int(spd_cfg.get("top_k", 1)),
            spd_renormalize=bool(spd_cfg.get("renormalize", True)),
            spd_revision=bool(spd_cfg.get("revision", True)),
            temperature=float(dcd_cfg.get("temperature", 0.0)),
            dcd_window_length=int(dcd_cfg.get("initial_window_length", method_cfg.get("block_size", 8))),
            dcd_max_window_length=int(dcd_cfg.get("max_window_length", 64)),
            dcd_window_type=str(dcd_cfg.get("window_type", "sliding")),
            dcd_decode_algo=str(dcd_cfg.get("decode_algo", "threshold")),
            dcd_decode_param=dcd_cfg.get("decode_param", trans_cfg.get("commit_confidence_tau", 0.75)),
            dcd_sample_top_k=None if dcd_cfg.get("top_k") is None else int(dcd_cfg.get("top_k")),
            dcd_top_p=None if dcd_cfg.get("top_p") is None else float(dcd_cfg.get("top_p")),
            dcd_cache_type=str(dcd_cfg.get("cache_type", "none")),
            dcd_refresh_count=int(dcd_cfg.get("refresh_count", 16)),
        )
    text = tokenizer.batch_decode(tokens.detach().cpu(), skip_special_tokens=True)[0].strip()
    mean_conf = float(confidence.detach().float().mean().cpu().item()) if confidence.numel() else 0.0
    return text, mean_conf


def run_rq1(args: argparse.Namespace) -> dict[str, Any]:
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run RQ1 on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    stage1_cfg = load_yaml(args.stage1_config)
    method_cfg = load_yaml(_method_config_path(args))
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    rq_cfg = eval_cfg.get("rq1", {})
    grid = _parse_grid(args.severity_grid_s, rq_cfg.get("smoke_grid_s" if args.smoke else "severity_grid_s", [0.0]))
    max_sentences = args.num_sentences
    if max_sentences is None and args.smoke: max_sentences = int(rq_cfg.get("smoke_num_sentences", 10))
    windows = controlled_windows(records, grid, max_sentences=max_sentences)
    records_by_id = {record.video_id: record for record in records}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    inference_cfg = load_yaml(args.inference_config)
    model, tokenizer = _build_eval_model(args, data_cfg, stage1_cfg, method_cfg, device)

    grouped: dict[tuple[float, float], dict[str, list]] = {}
    rows = []
    for window in windows:
        record = records_by_id[window.video_id]
        poses, timestamps = load_pose_window(record.pose, window.window_start_s, window.window_end_s, normalize=True)
        if poses.shape[0] == 0: continue
        prediction, confidence = _translate_window(
            model=model, tokenizer=tokenizer, method=args.method,
            poses_np=poses, timestamps_np=timestamps, start_s=window.window_start_s,
            device=device, inference_cfg=inference_cfg, method_cfg=method_cfg,
        )
        key = (window.delta_head_s, window.delta_tail_s)
        grouped.setdefault(key, {"predictions": [], "references": [], "confidences": []})
        grouped[key]["predictions"].append(prediction)
        grouped[key]["references"].append(window.reference)
        grouped[key]["confidences"].append(confidence)
        rows.append({**asdict(window), "prediction": prediction, "mean_confidence": confidence})

    severity = []
    for (dh, dt), values in sorted(grouped.items()):
        confs = values["confidences"]
        severity.append({
            "delta_head_s": dh, "delta_tail_s": dt, "windows": len(values["predictions"]),
            "mean_translation_confidence": float(sum(confs) / len(confs)) if confs else 0.0,
            "text_metrics": compute_text_metrics(values["predictions"], values["references"]),
        })
    summary = {
        "rq": "1", "language": args.language, "split": args.split, "method": args.method, 
        "grid_s": grid, "windows": len(rows), "severity": severity,
    }
    output = Path(args.output or f"outputs/rq1_{args.method}_{args.language}_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["output"] = str(output)
    return summary


def _build_streaming_runner(model, inference_cfg: dict, method_cfg: dict):
    from infer.stream import StreamingSLTRunner
    trans = inference_cfg.get("translation", {})
    dcd = trans.get("dcd", {})
    spd = method_cfg.get("spd", {})
    boundary = inference_cfg.get("boundary_stability", {})

    return StreamingSLTRunner(
        model,
        stride_s=float(inference_cfg.get("stride_s", 1.0)),
        buffer_cap_s=float(inference_cfg.get("buffer_cap_s", 18.0)),
        delta_enc_frames=int(boundary.get("delta_enc_frames", 3)),
        hysteresis_strides=int(boundary.get("hysteresis_strides", 2)),
        token_confidence_tau=float(trans.get("commit_confidence_tau", 0.75)),
        max_text_tokens=int(trans.get("max_text_tokens", method_cfg.get("max_text_tokens", 128))),
        diffusion_steps=int(trans.get("diffusion_steps", method_cfg.get("diffusion_steps", 64))),
        tau_dec=float(dcd.get("tau_dec", trans.get("commit_confidence_tau", 0.75))),
        spd_top_k=int(spd.get("top_k", 1)),
        spd_renormalize=bool(spd.get("renormalize", True)),
        spd_revision=bool(spd.get("revision", True)),
        temperature=float(dcd.get("temperature", 0.0)),
        dcd_window_length=int(dcd.get("initial_window_length", trans.get("block_size", 8))),
        dcd_max_window_length=int(dcd.get("max_window_length", 64)),
        dcd_window_type=str(dcd.get("window_type", "sliding")),
        dcd_decode_algo=str(dcd.get("decode_algo", "threshold")),
        dcd_decode_param=dcd.get("decode_param", trans.get("commit_confidence_tau", 0.75)),
        dcd_cache_type=str(dcd.get("cache_type", "none")),
        dcd_refresh_count=int(dcd.get("refresh_count", 16)),
        decode_conditioning=str(trans.get("decode_conditioning", "window")),
    )


@torch.no_grad()
def run_streaming(args: argparse.Namespace) -> dict[str, list[PredictionEvent]]:
    """Drive the sawtooth FSM end-to-end over each video → committed events.

    This is the *usable inference engine* for RQ2: a raw pose stream in, committed `(start, end, text, flagged_partial, commit_time)` 
    events out — our own BIO head and commit gate, recompute-each-stride, no cross-stride decoder state (§7). Only valid for the FSM 
    methods (stage2_dlm / stage2_ar); the clean baseline's RQ2 is the segment-then-translate pipeline floor (use --predictions).
    """
    if args.method == "stage2_baseline":
        raise SystemExit("Streaming RQ2 uses the FSM (stage2_dlm/stage2_ar). For the baseline pipeline-floor, pass --predictions.")
    data_cfg = load_yaml(args.data_config)
    stage1_cfg = load_yaml(args.stage1_config)
    method_cfg = load_yaml(_method_config_path(args))
    inference_cfg = load_yaml(args.inference_config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer = _build_eval_model(args, data_cfg, stage1_cfg, method_cfg, device)
    runner = _build_streaming_runner(model, inference_cfg, method_cfg)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)

    predicted: dict[str, list[PredictionEvent]] = {}
    for record in records:
        poses, _ = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=True)
        if poses.shape[0] == 0:
            predicted[record.video_id] = []
            continue

        events = runner.run(torch.as_tensor(poses, dtype=torch.float32), fps=float(record.pose.fps))
        predicted[record.video_id] = [PredictionEvent(
            video_id=record.video_id, start_s=float(ev.start_s), end_s=float(ev.end_s),
            text=tokenizer.decode(ev.token_ids.tolist(), skip_special_tokens=True).strip(),
            flagged_partial=bool(ev.flagged_partial), commit_time_s=float(ev.commit_time_s),
        ) for ev in events]
    return predicted


@torch.no_grad()
def run_pipeline_floor(args: argparse.Namespace) -> dict[str, list[PredictionEvent]]:
    """Segment-then-translate pipeline floor (§9.2 baseline; §8.2's realistic-perturbation point).

    The retrained Moryossef segmenter's predicted spans (analyze.py --stage segmenter-infer JSON,
    via --segments) are cut from the pose stream and translated by the clean-trained GFSLT
    baseline — the natural pipeline with no robustness training anywhere. Scored by the same
    tIoU/translation harness as the streaming FSM, so floor and method are directly comparable.
    """
    from moryossef26.infer import load_prediction_file
    data_cfg = load_yaml(args.data_config)
    stage1_cfg = load_yaml(args.stage1_config)
    method_cfg = load_yaml(_method_config_path(args))
    inference_cfg = load_yaml(args.inference_config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer = _build_eval_model(args, data_cfg, stage1_cfg, method_cfg, device)

    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    records_by_id = {record.video_id: record for record in records}
    segments = load_prediction_file(args.segments)

    predicted: dict[str, list[PredictionEvent]] = {}
    for video_id, spans in segments.items():
        record = records_by_id.get(video_id)
        if record is None: continue
        events: list[PredictionEvent] = []
        for span in spans:
            poses, timestamps = load_pose_window(record.pose, span.start_s, span.end_s, normalize=True)
            if poses.shape[0] == 0: continue
            text, _ = _translate_window(
                model=model, tokenizer=tokenizer, method=args.method,
                poses_np=poses, timestamps_np=timestamps, start_s=span.start_s,
                device=device, inference_cfg=inference_cfg, method_cfg=method_cfg,
            )
            events.append(PredictionEvent(video_id=video_id, start_s=float(span.start_s), end_s=float(span.end_s), text=text))
        predicted[video_id] = events
    return predicted


def run_rq2(args: argparse.Namespace) -> dict[str, Any]:
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run RQ2 on test without --allow-test")
    if not args.predictions and not args.stream and not args.segments: raise SystemExit(
        "--rq 2 needs --stream (FSM engine), --segments (segment-then-translate pipeline floor), or --predictions (score an events JSON)"
    )
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    thresholds = _parse_grid(args.tiou_thresholds, eval_cfg.get("rq2", {}).get("tiou_thresholds", [0.3, 0.5, 0.7, 0.9]))
    if args.stream:
        predicted = run_streaming(args)
        events_path = Path(args.events_out or f"outputs/rq2_stream_events_{args.method}_{args.language}_{args.split}.json")
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text(json.dumps({
            vid: [asdict(ev) for ev in evs] for vid, evs in predicted.items()
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        source = str(events_path)
    elif args.segments:
        predicted = run_pipeline_floor(args)
        events_path = Path(args.events_out or f"outputs/rq2_pipeline_floor_events_{args.method}_{args.language}_{args.split}.json")
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text(json.dumps({
            vid: [asdict(ev) for ev in evs] for vid, evs in predicted.items()
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        source = str(events_path)
    else:
        predicted = load_event_predictions(args.predictions)
        source = args.predictions
        
    gold = _gold_events(records)
    summary = {
        "rq": "2", "language": args.language, "split": args.split, "method": args.method,
        "predictions": source, "streamed": bool(args.stream),
        **evaluate_predicted_events(predicted, gold, thresholds),
    }
    output = Path(args.output or f"outputs/rq2_events_{args.language}_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["output"] = str(output)
    return summary


def run_segment_prf(args: argparse.Namespace) -> dict[str, Any]:
    if not args.pred or not args.gold: raise SystemExit("--pred and --gold JSON files are required when --rq is omitted")
    pred = _load_segments(args.pred)
    gold = _load_segments(args.gold)
    return segmentation_prf(pred, gold, tiou_threshold=args.tiou_threshold)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Misaligned-SLT evaluation entry points")
    parser.add_argument("--rq", choices=["1", "2"], default=None)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--stage1-config", default="configs/stage1_vlp.yaml")
    parser.add_argument("--method-config", default=None)
    parser.add_argument("--language", default="asf")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--method", default="stage2_dlm", choices=["stage2_baseline", "stage2_ar", "stage2_dlm"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--severity-grid-s", default=None, help="Comma-separated signed RQ1 offset grid in seconds")
    parser.add_argument("--num-sentences", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--predictions", default=None, help="RQ2 event JSON with start_s/end_s and optional text")
    parser.add_argument("--segments", default=None, help="RQ2 pipeline floor: segmenter spans JSON (analyze --stage segmenter-infer)")
    parser.add_argument("--stream", action="store_true", help="RQ2: run the streaming FSM engine to produce events")
    parser.add_argument("--events-out", default=None, help="Where to write streamed RQ2 events JSON")
    parser.add_argument("--tiou-thresholds", default=None, help="Comma-separated RQ2 tIoU thresholds")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--pred", default=None, help="Legacy segment-PRF predicted JSON list")
    parser.add_argument("--gold", default=None, help="Legacy segment-PRF gold JSON list")
    parser.add_argument("--tiou-threshold", type=float, default=0.1)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.rq == "1": result = run_rq1(args)
    elif args.rq == "2": result = run_rq2(args)
    else: result = run_segment_prf(args)
    print(json.dumps(result, indent=2, sort_keys=True))
