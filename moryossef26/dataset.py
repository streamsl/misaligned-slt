"""Whole-video-chunk dataset for the faithful Moryossef segmenter (segmenter-error analysis + RQ2 cascade floor).

Random natural-timeline chunks — the Moryossef 2026 segmentation regime, not the SLT window sampler that trains
the in-system BIO head. This model reads RAW pose keypoints (+ velocity) via a UNet CNN, so the raw-keypoint 
augmentations Moryossef uses apply here: fps_aug (essential), frame_dropout, body_part_dropout, and appended 
per-keypoint velocity. The in-system head reads FROZEN Uni-Sign features and drops these.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

from data.loader import VideoRecord
from data.windowing import BIO, TRUSTED_GAP_S, make_bio_labels
from poses import load_pose_window, apply_fps_aug


class SegmenterChunkDataset(Dataset):
    def __init__(
        self, records: list[VideoRecord], num_frames: int = 1024, steps_per_epoch: int | None = None, 
        records_for_epoch=None, fps_aug_enabled: bool = True, fps_aug_min: float = 15.0, fps_aug_max: float = 30.0,
        velocity: bool = True, training: bool = True, frame_dropout: float = 0.0, body_part_dropout: float = 0.0,
        seed: int = 42, trusted_gap_s: float | None = TRUSTED_GAP_S,
    ):
        if not records: raise ValueError("SegmenterChunkDataset requires at least one record")
        self.records = records
        # Optional `epoch -> records` provider, identical to StreamingWindowDataset's: a multilingual pool
        # re-draws its balanced sub-sample each epoch so coverage rotates. Without it this arm trains on one
        # fixed epoch-0 slice while S1 rotates, and the RQ2 cascade would compare methods AND data exposure.
        self._records_for_epoch = records_for_epoch
        self.num_frames = int(num_frames)
        if steps_per_epoch is None and training:
            # Epoch = enough chunks to COVER the corpus once, not one per video — 
            # else epoch-based early stopping kills runs after a handful of steps.
            total_frames = sum(int(r.pose.total_frames) for r in records)
            steps_per_epoch = max(len(records), total_frames // max(1, self.num_frames))
        self.steps_per_epoch = int(steps_per_epoch or len(records))
        self.trusted_gap_s = trusted_gap_s
        self.fps_aug_enabled = bool(fps_aug_enabled)
        self.fps_aug_min = float(fps_aug_min)
        self.fps_aug_max = float(fps_aug_max)
        self.velocity = bool(velocity)
        self.training = bool(training)
        self.frame_dropout = max(0.0, float(frame_dropout))
        self.body_part_dropout = max(0.0, float(body_part_dropout))
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        if not self.training or self._records_for_epoch is None: return
        records = self._records_for_epoch(int(epoch))
        if records: self.records = records

    def __getitem__(self, index: int) -> dict:
        # Training: fresh random chunks every epoch (persistent rng). Eval: rng derived from (seed, index) so the 
        # SAME chunks are scored every epoch — a per-epoch-random dev set makes the early-stopping monitor noise.
        rng = self.rng if self.training else np.random.default_rng(self.seed * 100_003 + int(index))
        rec = self.records[int(index) % len(self.records)]
        chunk_s = self.num_frames / rec.pose.fps

        start_s = 0.0 if rec.pose.duration_s <= chunk_s else float(rng.uniform(0.0, rec.pose.duration_s - chunk_s))
        end_s = min(rec.pose.duration_s, start_s + chunk_s)
        poses, abs_timestamps = load_pose_window(rec.pose, start_s, end_s, normalize=True)
        if poses.shape[0] > self.num_frames:
            poses, abs_timestamps = poses[: self.num_frames], abs_timestamps[: self.num_frames]

        # All augmentations are train-only (Moryossef gates fps_aug/dropouts on split==TRAIN; eval runs native fps).
        if self.training and self.fps_aug_enabled and poses.shape[0] > 1:
            poses, abs_timestamps, _ = apply_fps_aug(
                poses, source_fps=rec.pose.fps,
                min_fps=self.fps_aug_min, max_fps=self.fps_aug_max, rng=rng,
                source_timestamps_s=abs_timestamps,
            )

        if self.training and self.body_part_dropout > 0.0:
            poses = apply_body_part_dropout(poses, self.body_part_dropout, rng)
        if self.training and self.frame_dropout > 0.0:
            poses, abs_timestamps = apply_frame_dropout(poses, abs_timestamps, self.frame_dropout, rng)
        if self.velocity:
            poses = append_velocity(poses, abs_timestamps)
        labels = make_bio_labels(
            abs_timestamps, rec.sentences, start_s, end_s,
            trusted_gap_s=self.trusted_gap_s, video_duration_s=rec.pose.duration_s,
        )
        return {
            "poses": poses, "timestamps_s": abs_timestamps - start_s,
            "phrase_bio": labels, "frame_mask": np.ones((poses.shape[0],), dtype=bool),
            "video_id": rec.video_id, "start_s": start_s, "end_s": end_s,
        }


def collate_segmenter_chunks(batch: list[dict]) -> dict:
    # Right-pad; labels pad with UNK so padded frames drop out of the loss.
    max_len = max(item["poses"].shape[0] for item in batch)
    pose_shape = batch[0]["poses"].shape[1:]
    poses, timestamps, labels, masks, meta = [], [], [], [], []
    for item in batch:
        n = item["poses"].shape[0]
        pad = max_len - n
        poses.append(torch.nn.functional.pad(torch.as_tensor(item["poses"]).float(), (0, 0, 0, 0, 0, pad)))
        timestamps.append(torch.nn.functional.pad(torch.as_tensor(item["timestamps_s"]).float(), (0, pad)))
        labels.append(torch.cat([
            torch.as_tensor(item["phrase_bio"]).long(),
            torch.full((pad,), BIO["UNK"], dtype=torch.long)
        ]))
        masks.append(torch.cat([torch.ones(n, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)]))
        meta.append({k: item[k] for k in ("video_id", "start_s", "end_s")})
    return {
        "poses": torch.stack(poses).reshape(len(batch), max_len, *pose_shape),
        "timestamps_s": torch.stack(timestamps), "phrase_bio": torch.stack(labels),
        "frame_mask": torch.stack(masks), "meta": meta,
    }


def append_velocity(poses: np.ndarray, timestamps_s: np.ndarray, clip: float = 50.0) -> np.ndarray:
    """Append units/second velocity (Moryossef 2026 utils/pose.compute_velocity: diff / dt).

    Two guards theirs does not need (our Uni-Sign-normalized keypoints are zeroed on dropped detections): 
    velocity spanning a zeroed endpoint is flicker, not motion (0↔value at 25 fps reads as |Δ|×25) → masked 
    to 0; and clipped to ±clip (real signing peaks ~20 group-units/s).
    """
    if poses.shape[0] <= 1: velocity = np.zeros_like(poses, dtype=np.float32)
    else:
        dt = np.diff(timestamps_s.astype(np.float32))
        dt = np.maximum(dt, 1e-6)
        inner = np.diff(poses.astype(np.float32), axis=0) / dt[:, None, None]
        # Per-keypoint validity = any nonzero channel (invalid points are all-zero).
        valid = np.any(poses != 0.0, axis=-1)             # (T, K)
        pair_valid = (valid[1:] & valid[:-1])[..., None]  # (T-1, K, 1)
        inner = np.where(pair_valid, inner, 0.0)
        np.clip(inner, -float(clip), float(clip), out=inner)
        velocity = np.concatenate([np.zeros_like(poses[:1], dtype=np.float32), inner], axis=0)
    return np.concatenate([poses.astype(np.float32, copy=False), velocity.astype(np.float32, copy=False)], axis=-1)


def apply_body_part_dropout(poses: np.ndarray, probability: float, rng: np.random.Generator) -> np.ndarray:
    # Zero left/right hand channels independently (Moryossef repo default train aug).
    out = poses.copy()
    if rng.random() < float(probability): out[:, 9:30, :] = 0.0
    if rng.random() < float(probability): out[:, 30:51, :] = 0.0
    return out


def apply_frame_dropout(
    poses: np.ndarray, timestamps_s: np.ndarray, max_rate: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]: # Drop 0..max_rate of middle frames; edges preserved.
    if poses.shape[0] <= 2: return poses, timestamps_s
    drop_rate = float(rng.uniform(0.0, max(0.0, float(max_rate))))

    n_drop = int((poses.shape[0] - 2) * drop_rate)
    if n_drop <= 0: return poses, timestamps_s
    middle = np.arange(1, poses.shape[0] - 1)
    drop = rng.choice(middle, size=n_drop, replace=False)
    keep = np.ones((poses.shape[0],), dtype=bool)
    keep[drop] = False
    return poses[keep], timestamps_s[keep]
