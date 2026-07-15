"""Whole-stream inference + standalone evaluation for the segmenters (shared by both models).

ONE inference wrapper for both the faithful Moryossef segmenter (raw keypoints + velocity) and the in-system BIO
head run as the `s1` ablation (Uni-Sign features). The wrapper just calls `model.forward` and decodes — chunking
lives INSIDE the model (`chunked_rope_encode` at the head's train-time chunk size), which is the correct,
train-consistent Moryossef inference. The only per-model differences are the `velocity` flag (append raw-keypoint
velocity for the UNet segmenter; the frozen-encoder S1 head reads raw poses) and the logits container.
"""
from __future__ import annotations
import numpy as np
import torch

from data.loader import VideoRecord
from data.windowing import TRUSTED_GAP_S, make_bio_labels
from poses import load_pose_window
from moryossef26.dataset import append_velocity
from metrics import Segment, bio_frame_metrics, moryossef_segment_metrics, signing_runs_with_b_splits


def bio_tags_to_segments(tags, timestamps_s) -> list[Segment]:
    """Decode BIO argmax tags → time-domain phrase Segments (the shared span-decode rule).

    Prediction decode = contiguous signing runs (Moryossef's `likeliest_probs_to_segments`; his decode never
    requires a predicted `B`), additionally split at interior `B`s so adjacent predicted sentences stay separate.
    End time = onset of the frame after the last in-span frame, extrapolating one frame step past the sequence end
    for still-open spans.
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


def _phrase_logits(
    model, poses_np: np.ndarray, timestamps_np: np.ndarray, 
    device: torch.device, velocity: bool
) -> torch.Tensor:
    # Run one whole stream through `model.forward` → per-frame phrase logits (chunking is internal to the model).
    if velocity: poses_np = append_velocity(poses_np, timestamps_np)
    poses = torch.as_tensor(poses_np, dtype=torch.float32, device=device).unsqueeze(0)
    timestamps = torch.as_tensor(timestamps_np, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.ones(poses.shape[:2], dtype=torch.bool, device=device)
    out = model(poses, frame_mask=mask, timestamps_s=timestamps)
    return out["phrase"] if isinstance(out, dict) else out.logits  # MoryossefSegmenter dict vs BIOHeadOutput


def _set_rope_chunk(model, record, rope_chunk_s: float | None) -> None:
    # S1 head only (Moryossef chunks internally by its train num_frames): chunk RoPE at the head's TRAINED context,
    # expressed in SECONDS (= buffer_cap_s, the clamp on training windows) and converted to frames at THIS stream's
    # fps — dataset-general (30fps CSL → 540, 25fps PHOENIX → 450, per-video YouTube fps → its own count).
    if rope_chunk_s is not None and hasattr(model, "bio_head"):
        model.bio_head.chunk_size = max(1, round(float(rope_chunk_s) * float(record.pose.fps)))


@torch.no_grad()
def predict_phrase_segments(
    model, records: list[VideoRecord], device: torch.device,
    velocity: bool = True, rope_chunk_s: float | None = None,
) -> dict[str, list[Segment]]:
    # Predicted phrase segments per video (Analysis A / Analysis B / RQ2 cascade upstream).
    model.eval().to(device)
    predictions: dict[str, list[Segment]] = {}

    for record in records:
        poses, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=True)
        if poses.shape[0] == 0:
            predictions[record.video_id] = []
            continue
        _set_rope_chunk(model, record, rope_chunk_s)
        tags = _phrase_logits(model, poses, timestamps, device, velocity).argmax(dim=-1)[0].detach().cpu()
        predictions[record.video_id] = bio_tags_to_segments(tags, timestamps.tolist())
    return predictions


@torch.no_grad()
def evaluate_segmenter_whole_video(
    model, records: list[VideoRecord], device: torch.device, velocity: bool = True, rope_chunk_s: float | None = None,
    trusted_gap_s: float | None = TRUSTED_GAP_S, tiou_thresholds: tuple[float, ...] = (0.5,),
) -> dict[str, float]:
    """Moryossef evaluate.py-style STANDALONE eval: process each video whole (encoder chunks internally), build GT
    phrase BIO from caption spans, and average frame-F1 / 1-to-1 tIoU segment P/R/F1 over videos.

    This is the faithful whole-video test protocol (phrase level only, no sign head), and — for `--segmenter-arch
    s1` — the way to score the pretrained in-system BIO head on its own, without waiting for joint fine-tuning.
    `rope_chunk_s` (S1 only): the head's trained context in SECONDS (= buffer_cap_s), converted to frames per stream.
    `trusted_gap_s` defaults to TRUSTED_GAP_S so GT labeling matches training: long uncaptioned stretches become
    UNK (excluded from the metric), not O — else the segmenter is penalized for firing on possibly-uncaptioned signing.
    `tiou_thresholds`: pass eval.yaml's rq2.tiou_thresholds for numbers directly comparable to RQ2's segmentation
    block; thresholds beyond the first get an `@t` key suffix (first threshold keeps the bare keys for the monitor).
    """
    model.eval().to(device)
    rows: list[dict[str, float]] = []

    for record in records:
        poses, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=True)
        if poses.shape[0] == 0: continue
        _set_rope_chunk(model, record, rope_chunk_s)
        gold = make_bio_labels(
            timestamps, record.sentences, 0.0, float(record.pose.duration_s),
            trusted_gap_s=trusted_gap_s, video_duration_s=record.pose.duration_s,
        )
        logits = _phrase_logits(model, poses, timestamps, device, velocity).detach().cpu()
        labels = torch.as_tensor(np.asarray(gold)).long().unsqueeze(0)
        # prefix "bio" (trainer convention): bio_f1 is the BINARY signing-vs-not frame F1 — under prefix "phrase"
        # it would sit next to phrase_frame_f1 (macro O/B/I, the §4.6 acceptance number) and read as the same thing.
        row = dict(bio_frame_metrics(logits, labels, prefix="bio"))
        for j, t in enumerate(tiou_thresholds):
            seg = moryossef_segment_metrics(logits, labels, prefix="phrase", tiou_threshold=float(t))
            if j == 0: row.update(seg)
            else: row.update({f"{k}@{t:g}": v for k, v in seg.items() if k != "phrase_frame_f1"})
        rows.append(row)
    if not rows: return {}
    return {k: float(sum(r[k] for r in rows) / len(rows)) for k in rows[0].keys()}
