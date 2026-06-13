from __future__ import annotations
from pathlib import Path
import json

import torch
import numpy as np
from data.loader import VideoRecord
from data.windowing import BIO, make_bio_labels
from poses import load_pose_window

from models.checkpointing import load_model_checkpoint
from moryossef26.dataset import append_velocity
from moryossef26.model import MoryossefSegmenter
from metrics import Segment, bio_frame_metrics, moryossef_segment_metrics, signing_runs_with_b_splits


def bio_tags_to_segments(tags: torch.Tensor | list[int], timestamps_s: torch.Tensor | list[float]) -> list[Segment]:
    """Time-domain wrapper over `metrics.signing_runs_with_b_splits` (one splitting rule).

    Prediction decode = contiguous signing runs (Moryossef's `likeliest_probs_to_segments` — his
    decode never requires a predicted `B`), additionally split at interior `B`s so adjacent predicted
    sentences stay separate when the model does emit a boundary — Analysis A's over/under-segmentation
    counts read those splits. A B-required decode yields zero segments whenever the model detects
    signing but never wins argmax with `B` (one B frame per sentence; 68% of caption boundaries have
    no visual pause). End time = onset of the frame after the last in-span frame, extrapolating one
    frame step past the sequence end for still-open spans.
    """
    if not isinstance(tags, torch.Tensor): tags = torch.as_tensor(tags)
    times = [float(t) for t in (timestamps_s.tolist() if isinstance(timestamps_s, torch.Tensor) else timestamps_s)]
    if not times: return []

    step = (times[-1] - times[-2]) if len(times) > 1 else 0.04  # assume 25fps when a single frame
    segments: list[Segment] = []
    for seg in signing_runs_with_b_splits(tags):
        end_idx = int(seg["end"]) + 1
        end_t = times[end_idx] if end_idx < len(times) else times[-1] + step
        segments.append(Segment(times[int(seg["start"])], end_t))
    return segments

    
def save_prediction_file(predictions: dict[str, list[Segment]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"video_id": video_id, "segments": [{"start_s": float(segment.start_s), "end_s": float(segment.end_s)} for segment in segments]}
        for video_id, segments in sorted(predictions.items())
    ]
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_prediction_file(path: str | Path) -> dict[str, list[Segment]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.items()
        return {str(video_id): [Segment(float(row["start_s"]), float(row["end_s"])) for row in rows] for video_id, rows in items}

    predictions: dict[str, list[Segment]] = {}
    for row in raw:
        if "video_id" in row and "segments" in row:
            predictions[str(row["video_id"])] = [Segment(float(item["start_s"]), float(item["end_s"])) for item in row["segments"]]
        elif "video_id" in row and "start_s" in row and "end_s" in row:
            predictions.setdefault(str(row["video_id"]), []).append(Segment(float(row["start_s"]), float(row["end_s"])))
        else: raise ValueError(f"Unsupported prediction row format: {row}")
    return predictions


def load_segmenter_for_inference(checkpoint: str | Path, model: MoryossefSegmenter, device: torch.device,) -> MoryossefSegmenter:
    load_model_checkpoint(model, checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_phrase_segments(
    model: MoryossefSegmenter, records: list[VideoRecord], 
    device: torch.device, velocity: bool = True,
) -> dict[str, list[Segment]]:
    predictions: dict[str, list[Segment]] = {}
    for record in records:
        poses, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=True)
        if poses.shape[0] == 0:
            predictions[record.video_id] = []
            continue

        if velocity: poses = append_velocity(poses, timestamps)
        poses_t = torch.as_tensor(poses, dtype=torch.float32, device=device).unsqueeze(0)
        timestamps_t = torch.as_tensor(timestamps, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(poses_t, timestamps_s=timestamps_t)["phrase"]
        tags = logits.argmax(dim=-1)[0].detach().cpu()
        segments = bio_tags_to_segments(tags, timestamps)
        predictions[record.video_id] = segments
    return predictions


@torch.no_grad()
def evaluate_segmenter_whole_video(
    model: MoryossefSegmenter, records: list[VideoRecord],
    device: torch.device, velocity: bool = True,
) -> dict[str, float]:
    """Moryossef evaluate.py-style metrics on the held-out split: process each video whole (the encoder chunks internally), 
    build GT phrase BIO from caption spans, and average frame-F1 / segment-IoU / segment-F1 + frame P/R/F1 over videos.

    This is the faithful test-set protocol — phrase level only (no sign head) — as
    opposed to the random-chunk dev loss used inside the training loop.
    """
    was_training = model.training
    model.eval()
    rows: list[dict[str, float]] = []
    for record in records:
        poses, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=True)
        if poses.shape[0] == 0: continue
        gold = make_bio_labels(
            timestamps, record.sentences, 0.0, float(record.pose.duration_s),
            video_duration_s=record.pose.duration_s,  # long uncaptioned stretches -> UNK (excluded from eval)
        )
        if velocity: poses = append_velocity(poses, timestamps)
        
        poses_t = torch.as_tensor(poses, dtype=torch.float32, device=device).unsqueeze(0)
        timestamps_t = torch.as_tensor(timestamps, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(poses_t, timestamps_s=timestamps_t)["phrase"].detach().cpu()
        labels = torch.as_tensor(np.asarray(gold)).long().unsqueeze(0)
        rows.append({
            **bio_frame_metrics(logits, labels, prefix="phrase"),
            **moryossef_segment_metrics(logits, labels, prefix="phrase"),
        })
    if was_training: model.train()
    if not rows: return {}
    keys = rows[0].keys()
    return {k: float(sum(r[k] for r in rows) / len(rows)) for k in keys}
