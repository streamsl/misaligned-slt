from __future__ import annotations
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from .preprocessing import normalize_keypoints, threshold_confidence

SEGMENT_RE = re.compile(r"_segment_(\d+)$")


@dataclass(frozen=True)
class PoseIndex:
    video_id: str
    paths: tuple[Path, ...]
    frame_counts: tuple[int, ...]
    fps: float
    width: int | None = None
    height: int | None = None

    @property
    def total_frames(self) -> int:
        return int(sum(self.frame_counts))

    @property
    def duration_s(self) -> float:
        return self.total_frames / float(self.fps)

    @property
    def cumulative_frames(self) -> np.ndarray:
        return np.cumsum([0, *self.frame_counts])


def base_video_id(path_or_stem: str | Path) -> str:
    stem = Path(path_or_stem).stem
    return SEGMENT_RE.sub("", stem)


def build_pose_index(
    pose_root: str | Path, fps: float,
    width: int | None = None, height: int | None = None,
) -> dict[str, PoseIndex]:
    pose_root = Path(pose_root)
    grouped: dict[str, list[Path]] = {}
    for path in sorted(pose_root.glob("*.npy")):
        grouped.setdefault(base_video_id(path), []).append(path)

    index: dict[str, PoseIndex] = {}
    for video_id, paths in grouped.items():
        def _seg_key(p: Path) -> tuple[int, str]:
            m = SEGMENT_RE.search(p.stem)
            return (int(m.group(1)) if m else -1, p.name)

        ordered = tuple(sorted(paths, key=_seg_key))
        counts = tuple(int(np.load(path, mmap_mode="r").shape[0]) for path in ordered)
        index[video_id] = PoseIndex(
            video_id=video_id, paths=ordered,
            frame_counts=counts, fps=float(fps),
            width=width, height=height,
        )
    return index


def load_pose_frames(pose_index: PoseIndex, start_frame: int, end_frame: int) -> np.ndarray:
    if start_frame < 0 or end_frame < start_frame: raise ValueError(f"Invalid frame range [{start_frame}, {end_frame})")
    end_frame = min(end_frame, pose_index.total_frames)
    cumulative = pose_index.cumulative_frames
    if start_frame >= end_frame: return np.zeros((0, 133, 3), dtype=np.float32)

    start_file = int(np.searchsorted(cumulative, start_frame, side="right") - 1)
    end_file = int(np.searchsorted(cumulative, end_frame - 1, side="right") - 1)
    chunks: list[np.ndarray] = []
    for file_idx in range(start_file, end_file + 1):
        local_start = max(0, start_frame - int(cumulative[file_idx]))
        local_end = min(pose_index.frame_counts[file_idx], end_frame - int(cumulative[file_idx]))
        arr = np.load(pose_index.paths[file_idx], mmap_mode="r")
        chunks.append(np.asarray(arr[local_start:local_end], dtype=np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 133, 3), dtype=np.float32)


def load_pose_window(pose_index: PoseIndex, start_s: float, end_s: float, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    # Load a real-timeline pose window and relative timestamps
    start_s = max(0.0, float(start_s))
    end_s = min(float(end_s), pose_index.duration_s)
    start_frame = int(np.floor(start_s * pose_index.fps))
    end_frame = int(np.ceil(end_s * pose_index.fps))
    poses = load_pose_frames(pose_index, start_frame, end_frame)
    if normalize and poses.shape[1:] == (133, 3):
        poses = normalize_keypoints(poses, width=pose_index.width, height=pose_index.height)
    poses = threshold_confidence(poses)
    timestamps = (np.arange(poses.shape[0], dtype=np.float32) + start_frame) / float(pose_index.fps)
    return poses.astype(np.float32, copy=False), timestamps
    