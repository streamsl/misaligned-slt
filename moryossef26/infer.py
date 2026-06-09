from __future__ import annotations
from pathlib import Path
import json
import torch

from data.loader import VideoRecord
from poses import load_pose_window
from models.checkpointing import load_model_checkpoint
from moryossef26.dataset import append_velocity
from moryossef26.model import MoryossefSegmenter
from metrics import Segment


def bio_tags_to_segments(tags: torch.Tensor | list[int], timestamps_s: torch.Tensor | list[float]) -> list[Segment]:
    if not isinstance(tags, torch.Tensor): tags = torch.as_tensor(tags)
    if not isinstance(timestamps_s, torch.Tensor): timestamps_s = torch.as_tensor(timestamps_s, dtype=torch.float32)
    tags_list = tags.tolist()
    times = timestamps_s.tolist()
    segments: list[Segment] = []
    start: float | None = None
    for idx, tag in enumerate(tags_list):
        if tag == BIO["B"]:
            if start is not None: segments.append(Segment(start, times[idx]))
            start = times[idx]
        elif tag == BIO["O"] and start is not None:
            segments.append(Segment(start, times[idx]))
            start = None
    if start is not None and times: # Assume 25 FPS if only 1 timestamp is present, to give segments with some duration.
        step = (times[-1] - times[-2]) if len(times) > 1 else 0.04 
        segments.append(Segment(start, times[-1] + step))
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
