"""Pose-sequence augmentation for the TRAIN chunks of both segmentation arms (data/chunks.py `ChunkDataset`, with `release` 
for the Moryossef arm). Stage 2 uses none. There is 1 recipe, the Moryossef 2026 release's: `apply_hand_dropout` on raw
COCO-WholeBody-133 keypoints, then the arm's own normalization, then `apply_fps_aug` and `apply_frame_dropout`. Last 2 
remove frames and return the kept timestamps, so the caller rebuilds BIO labels from them. No spatial transform.
"""
import numpy as np
from . import UNISIGN_LEFT_IDX, UNISIGN_RIGHT_IDX


def apply_fps_aug(
    poses: np.ndarray, source_timestamps_s: np.ndarray, source_fps: float, target_fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Moryossef-style fps augmentation (frame-density resampling to `target_fps`, no speed change). 
    The caller draws `target_fps` (data/chunks.py), because the release arm needs it before it loads the chunk.

    The kept frames keep their source timestamps: `load_pose_window` floors the start frame, so timestamps rebuilt from 
    the requested window start could shift BIO/RoPE timing by up to 1 frame. fps_aug changes frame density, not physical
    time attached to a frame.
    """
    num_frames = int(poses.shape[0])
    source_timestamps_s = np.asarray(source_timestamps_s, dtype=np.float32)
    if source_timestamps_s.shape[0] != num_frames: raise ValueError("source_timestamps_s must have 1 timestamp per pose frame")
    if source_fps <= target_fps * 1.05: return poses, source_timestamps_s

    target_len = max(1, round(num_frames * target_fps / float(source_fps)))
    indices = np.round(np.arange(target_len) * (num_frames - 1) / max(1, target_len - 1))
    indices = indices.astype(np.int64).clip(0, num_frames - 1)
    return poses[indices], source_timestamps_s[indices]


def apply_hand_dropout(raw: np.ndarray, probability: float, rng: np.random.Generator) -> np.ndarray:
    # The release's body_part_dropout: each hand independently, with `probability`, for the whole chunk. 
    # Applied on raw COCO-133 as confidence 0, so each arm's own normalization removes it (Uni-Sign zeroes 
    # conf <= 0.3; the release contract makes it NaN, then 0).
    out = np.asarray(raw, dtype=np.float32).copy()
    for hand in (UNISIGN_LEFT_IDX, UNISIGN_RIGHT_IDX):
        if rng.random() < float(probability): out[:, hand, 2] = 0.0
    return out


def apply_frame_dropout(
    poses: np.ndarray, timestamps_s: np.ndarray, max_rate: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]: # The release's frame_dropout: drop U(0, max_rate) of the interior frames; both edges kept.
    if poses.shape[0] <= 2: return poses, timestamps_s
    n_drop = int((poses.shape[0] - 2) * float(rng.uniform(0.0, max(0.0, float(max_rate)))))
    if n_drop <= 0: return poses, timestamps_s
    keep = np.ones((poses.shape[0],), dtype=bool)
    keep[rng.choice(np.arange(1, poses.shape[0] - 1), size=n_drop, replace=False)] = False
    return poses[keep], timestamps_s[keep]
