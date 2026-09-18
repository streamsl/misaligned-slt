"""Whole-video-chunk dataset for the faithful Moryossef segmenter (segmenter-error analysis + RQ2 cascade floor).

Random natural-timeline chunks — the Moryossef 2026 segmentation regime, not the SLT window sampler that trains
the in-system BIO head. It reads the RELEASED model's own input contract — their 50 landmarks (8 body, two hands,
no face), their shoulder-and-standardise normalisation, velocity after — so this arm reproduces Moryossef 2026
rather than adapting it, and shares no preprocessing with the in-system head.
"""
from __future__ import annotations
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset

from data.loader import VideoRecord
from data.windowing import BIO, TRUSTED_GAP_S, make_bio_labels
from poses import load_pose_window, apply_fps_aug
from poses.preprocessing import UNISIGN_LEFT_IDX, UNISIGN_RIGHT_IDX

# ── The RELEASED weights' own input contract (this arm trains and infers under it) ────────────────────────────
# Their 50 landmarks after reduce_holistic, as DWPose COCO-WholeBody-133 indices: 8 body points then the 2
# 21-point hands, which are the canonical topology in the same order: their own `preprocess_pose` emits
# LEFT_HAND_LANDMARKS then RIGHT_HAND_LANDMARKS at output indices 8:29 and 29:50.
# Body is LEFT/RIGHT shoulder, elbow, wrist, hip — MediaPipe pose landmarks 11,12,13,14,15,16,23,24 in index order.
RELEASE_BODY_IDX = (5, 6, 7, 8, 9, 10, 11, 12)
RELEASE_KP_IDX = RELEASE_BODY_IDX + tuple(UNISIGN_LEFT_IDX) + tuple(UNISIGN_RIGHT_IDX)
RELEASE_LHAND = (len(RELEASE_BODY_IDX), len(RELEASE_BODY_IDX) + len(UNISIGN_LEFT_IDX))     # their 8:29
RELEASE_RHAND = (RELEASE_LHAND[1], RELEASE_LHAND[1] + len(UNISIGN_RIGHT_IDX))              # their 29:50
RELEASE_POSE_DIMS = (len(RELEASE_KP_IDX), 6)


class MoryossefChunkDataset(Dataset):
    def __init__(
        self, records: list[VideoRecord], num_frames: int = 1024, steps_per_epoch: int | None = None, 
        records_for_epoch=None, fps_aug_enabled: bool = True, fps_aug_min: float = 15.0, fps_aug_max: float = 30.0,
        velocity: bool = True, training: bool = True, frame_dropout: float = 0.0, body_part_dropout: float = 0.0,
        seed: int = 42, trusted_gap_s: float | None = TRUSTED_GAP_S, release_stats: dict | None = None,
    ):
        if not records: raise ValueError("MoryossefChunkDataset requires at least one record")
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
        self.release_stats = release_stats   # fitted once on the corpus; None falls back to per-clip moments

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
        # The released model's own input contract (to_release_coords), so this arm reproduces Moryossef 2026
        # rather than adapting it: their 50 landmarks, their normalisation, no face, velocity added after the
        # augmentations. Needs no video crop box — their normaliser is shoulder-relative.
        raw, abs_timestamps = load_pose_window(rec.pose, start_s, end_s, normalize=False)
        poses = to_release_coords(
            raw, self.release_stats, aspect=pose_aspect(rec.pose)
        ) if raw.shape[1:] == (133, 3) else raw
        if poses.shape[0] > self.num_frames:
            poses, abs_timestamps = poses[: self.num_frames], abs_timestamps[: self.num_frames]

        # All augmentations are train-only (Moryossef gates fps_aug/dropouts on split==TRAIN; eval runs native fps).
        if self.training and self.fps_aug_enabled and poses.shape[0] > 1:
            poses, abs_timestamps, _ = apply_fps_aug(
                poses, source_fps=rec.pose.fps, min_fps=self.fps_aug_min, max_fps=self.fps_aug_max, 
                rng=rng, source_timestamps_s=abs_timestamps,
            )
        if self.training and self.body_part_dropout > 0.0:
            poses = apply_body_part_dropout(poses, self.body_part_dropout, rng)
        if self.training and self.frame_dropout > 0.0:
            poses, abs_timestamps = apply_frame_dropout(poses, abs_timestamps, self.frame_dropout, rng)
        if self.velocity: poses = release_velocity(poses, abs_timestamps)
        labels = make_bio_labels(
            abs_timestamps, rec.sentences, start_s, end_s,
            trusted_gap_s=self.trusted_gap_s, video_duration_s=rec.pose.duration_s,
        )
        return {
            "poses": poses, "timestamps_s": abs_timestamps - start_s,
            "phrase_bio": labels, "frame_mask": np.ones((poses.shape[0],), dtype=bool),
            "video_id": rec.video_id, "start_s": start_s, "end_s": end_s,
        }


def collate_moryossef_chunks(batch: list[dict]) -> dict:
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


def pose_aspect(pose_index) -> float | None:
    # width/height of the source video, or None when the sidecar omits either (poses/pose_io.py allows blanks).
    w, h = getattr(pose_index, "width", None), getattr(pose_index, "height", None)
    return float(w) / float(h) if w and h else None


def fit_release_stats(records, frames_per_video: int = 256, max_videos: int = 400, seed: int = 42) -> dict:
    """Per-keypoint-per-dimension mean/std of the shoulder-normalised coordinates, over the training corpus.

    Their `normalize_mean_std` standardises against a FIXED table, which is what makes a chunk and a whole-video
    pass produce identical coordinates. `pose_anonymization` does ship that table, but it is fitted on MediaPipe
    landmarks in their normalised units and can't be applied to DWPose coordinates, so the equivalent is fitted 
    here ONCE on our own records and stamped in the checkpoint. 4 slots the release holds at 0 (both hips, both 
    hand roots) come out at std 0 and stay 0, because the divisor guard below leaves a 0 std as 1. Estimating it 
    per clip instead would put a training chunk and the inference pass in different coordinates.

    Subsampled: a per-keypoint second moment converges long before the corpus does, and reading every frame of
    every video would cost a full corpus pass (138 GiB on ase) for 300 numbers.
    """
    rng = np.random.default_rng(int(seed))
    picked = sorted(records, key=lambda r: r.video_id)
    if len(picked) > int(max_videos):
        picked = [picked[i] for i in rng.choice(len(picked), int(max_videos), replace=False)]

    total = sq = count = 0.0
    for rec in picked:
        span = min(float(rec.pose.duration_s), float(frames_per_video) / max(1.0, float(rec.pose.fps)))
        raw, _ = load_pose_window(rec.pose, 0.0, span, normalize=False)
        if raw.shape[1:] != (133, 3) or raw.shape[0] == 0: continue
        xy = _shoulder_normalised(raw, aspect=pose_aspect(rec.pose))
        ok = np.isfinite(xy).all(axis=-1)                                  # (T,50)
        seen = np.where(ok[..., None], np.nan_to_num(xy), 0.0)
        total, sq, count = total + seen.sum(axis=0), sq + (seen ** 2).sum(axis=0), count + ok.sum(axis=0)[:, None]

    count = np.maximum(count, 1.0)
    mean = total / count
    std = np.sqrt(np.maximum(sq / count - mean ** 2, 0.0))
    return {"mean": np.asarray(mean, dtype=np.float32).tolist(), "std": np.asarray(std, dtype=np.float32).tolist(),
            "videos": len(picked), "frames_per_video": int(frames_per_video)}


def _shoulder_normalised(raw: np.ndarray, conf_thr: float = 0.3, aspect: float | None = None) -> np.ndarray:
    # Their pre_process_pose: select their 50 landmarks, centre and scale on shoulders, re-centre each hand on its 
    # own root, and hold 2 hip slots at 0. Undetected point is NaN — it carries no position. `aspect` = width/height. 
    # Their .pose files hold MediaPipe in PIXELS (pose_format holistic: x*width, y*height), which is ISOTROPIC; ours 
    # are stored x/W, y/H (poses/preprocessing.py), which is not. Shoulder vector is near-horizontal, so dividing 
    # both axes by its norm does NOT undo that — y would stay stretched by W/H. y/H divided by W/H is y/W, so both 
    # axes land in units of W and shoulder division then removes W. None leaves the stretch.
    kp = np.nan_to_num(np.asarray(raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    xy = kp[:, RELEASE_KP_IDX, :2].copy()
    xy[kp[:, RELEASE_KP_IDX, 2] <= float(conf_thr)] = np.nan
    if aspect: xy[..., 1] /= float(aspect)

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning) # a keypoint never detected is all-NaN
        shoulders = xy[:, 0:2, :]
        xy = xy - np.nanmean(shoulders, axis=1, keepdims=True)
        width = np.linalg.norm(np.nan_to_num(shoulders[:, 0] - shoulders[:, 1]), axis=-1)[:, None, None]
        xy = xy / np.where(width > 0, width, 1.0)
        # Their `shift_hands` subtracts each hand's OWN root, not the body wrist, so the root lands at exactly 0.
        for lo, hi in (RELEASE_LHAND, RELEASE_RHAND): xy[:, lo:hi, :] -= xy[:, lo:lo + 1, :]
    # Their `pose_hide_legs` zeroes LEFT/RIGHT_HIP before `reduce_holistic` keeps the slots, so the released weights
    # read these 2 columns as a constant. Feeding live hips there drives fc columns fitted on a constant.
    xy[:, 6:8, :] = 0.0
    return xy


def to_release_coords(
    raw: np.ndarray, stats: dict | None = None, conf_thr: float = 0.3, aspect: float | None = None
) -> np.ndarray:
    """Raw DWPose (T,133,3) -> their (T,50,3) normalised coordinates: [x, y, z=0].

    Velocity comes AFTER, on these coordinates, exactly as `datasets/common.py` orders it — so the train-only
    augmentations go in between. DWPose has no depth, so z (and therefore vz) is held at 0: a third of the spatial
    input their model reads is absent, and no score from this representation is comparable to their published DGS
    results. `stats` is the fitted table (fit_release_stats); without one the clip's own moments are used, which
    does NOT reproduce the training coordinates.
    """
    if raw.ndim != 3 or raw.shape[1:] != (133, 3):
        raise ValueError(f"Expected raw (T,133,3) DWPose keypoints, got {raw.shape}")
    xy = _shoulder_normalised(raw, conf_thr=conf_thr, aspect=aspect)
    if stats is None:
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean, std = np.nanmean(xy, axis=0, keepdims=True), np.nanstd(xy, axis=0, keepdims=True)
        mean, std = np.nan_to_num(mean), np.nan_to_num(std)
    else:
        mean = np.asarray(stats["mean"], dtype=np.float32)[None]
        std = np.asarray(stats["std"], dtype=np.float32)[None]
    out = np.zeros((xy.shape[0], len(RELEASE_KP_IDX), 3), dtype=np.float32)
    out[..., :2] = np.nan_to_num((xy - mean) / np.where(std > 1e-6, std, 1.0), nan=0.0)
    return out


def release_velocity(coords: np.ndarray, timestamps_s: np.ndarray) -> np.ndarray:
    """Their `compute_velocity` (diff / dt, zero on the first frame), appended to (T,50,6).

    One guard theirs does not need: DWPose drops detections far more often than MediaPipe, and a step spanning a
    zeroed endpoint is a detection flicker rather than motion, so it is masked to 0. No magnitude clip — these
    coordinates are standardised, so a clip in body-bbox units would have no meaning.
    """
    if coords.shape[0] <= 1: velocity = np.zeros_like(coords, dtype=np.float32)
    else:
        dt = np.maximum(np.diff(np.asarray(timestamps_s, dtype=np.float32)), 1e-6)
        inner = np.diff(coords.astype(np.float32), axis=0) / dt[:, None, None]
        valid = np.any(coords != 0.0, axis=-1)
        inner = np.where((valid[1:] & valid[:-1])[..., None], inner, 0.0)
        velocity = np.concatenate([np.zeros_like(coords[:1], dtype=np.float32), inner], axis=0)
    return np.concatenate([coords.astype(np.float32, copy=False), velocity], axis=-1)


def apply_body_part_dropout(poses: np.ndarray, probability: float, rng: np.random.Generator) -> np.ndarray:
    # Zero left/right hand channels independently (Moryossef repo default train aug).
    out = poses.copy()
    if rng.random() < float(probability): out[:, RELEASE_LHAND[0]:RELEASE_LHAND[1], :] = 0.0
    if rng.random() < float(probability): out[:, RELEASE_RHAND[0]:RELEASE_RHAND[1], :] = 0.0
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
