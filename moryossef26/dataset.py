"""Whole-video-chunk dataset for the faithful Moryossef segmenter (the RQ2 cascade floor).

The release input contract that S1's chunk pipeline (data/chunks.py ChunkDataset with `release`) applies for this arm: 
the release's 50-landmark layout, its standardisation and velocity features; the trainer uses fixed-length chunks. 
DWPose transfer, fitted coordinate statistics and caption-unit supervision adapt it to the target corpus.
"""
from __future__ import annotations
import warnings
import numpy as np

from data.batch import collate_windows
from poses import load_pose_window
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


def collate_moryossef_chunks(batch: list[dict]) -> dict:
    # collate_windows (UNK label padding, frame_mask), with the release's ZERO pose padding: its model attends over the pad.
    out = collate_windows(batch)
    out["poses"] = out["poses"] * out["frame_mask"][:, :, None, None]
    return out

def pose_aspect(pose_index) -> float | None:
    # width/height of the source video (video_meta.csv, else data.yaml pose.width/height; poses.build_pose_index), or None.
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


def to_release_coords(raw: np.ndarray, stats: dict, conf_thr: float = 0.3, aspect: float | None = None) -> np.ndarray:
    """Raw DWPose (T,133,3) -> their (T,50,3) normalised coordinates: [x, y, z=0].

    Velocity comes AFTER, on these coordinates, exactly as `datasets/common.py` orders it — so the train-only
    augmentations go in between. DWPose has no depth, so z (and therefore vz) is held at 0: a 3rd of the spatial
    input their model reads is absent, and no score from this representation is comparable to their published 
    DGS results. `stats` is the fitted table (fit_release_stats), the standardisation the arm trained under.
    """
    if raw.ndim != 3 or raw.shape[1:] != (133, 3):
        raise ValueError(f"Expected raw (T,133,3) DWPose keypoints, got {raw.shape}")
    xy = _shoulder_normalised(raw, conf_thr=conf_thr, aspect=aspect)
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
