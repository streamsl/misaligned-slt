from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import os, re, csv, json
import numpy as np
from .preprocessing import normalize_keypoints_unisign

SEGMENT_RE = re.compile(r"_segment_(\d+)$")
META_FILENAME = "video_meta.csv"
# caption_source: caption provenance — human | mt (NLLB machine-translation) | shard (raw bundled YouTube track) | none.
# undetected_ratio: share of frames with nobody detected, from SignVerse payload's per-frame `num_persons` (the .npy keeps
# 1 primary person/frame, `prepare_yt25._primary_slots`, so it cannot be recovered from it later). An audit diagnostic only:
# no rule reads it. `--stage convert` writes it for a video it converts; `--stage person-counts` backfills a row left blank
# (UNKNOWN), from the shard, without re-converting.
# masked_runs / empty_runs: JSON lists of half-open .npy frame runs [a, b) where a second real body is present and moves /
# where no real body is present (prepare_yt25.masked_runs / empty_runs); [] = measured, no run; blank = unknown. Same writers 
# as the counts. The loader cuts both out of every stream (data.loader.split_clean_segments).
# handless_runs: the same JSON form, runs where the kept body shows no hand (prepare_yt25.handless_runs, read from the .npy).
# The loader quarantines a caption unit mostly inside them and cuts no frame.
RUN_FIELDS = ("masked_runs", "empty_runs", "handless_runs")
META_FIELDS = ("video_id", "duration_s", "width", "height", "caption_source", "undetected_ratio", *RUN_FIELDS)


@dataclass(frozen=True)
class PoseIndex:
    # `video_id` is the SOURCE video. A clean-segment record (data.loader.split_clean_segments) views frames
    # [start_frame, start_frame + num_frames) of it; the defaults (0, None) view the whole file. Every frame index and
    # time of a view is local (frame 0 = start_frame, time 0 = offset_s on the source timeline).
    video_id: str
    paths: tuple[Path, ...]
    frame_counts: tuple[int, ...]
    fps: float
    # Pixel frame size of THIS video (video_meta.csv width/height, else data.yaml pose.width/height), read only by 
    # the Moryossef aspect correction (moryossef26.dataset.pose_aspect); Uni-Sign normalization is bbox-relative 
    # and resolution-independent.
    width: int | None = None
    height: int | None = None
    start_frame: int = 0
    num_frames: int | None = None

    @property
    def total_frames(self) -> int:
        return int(sum(self.frame_counts)) if self.num_frames is None else int(self.num_frames)

    @property
    def offset_s(self) -> float:  # source time of the view's frame 0
        return self.start_frame / float(self.fps)

    @property
    def duration_s(self) -> float:
        return self.total_frames / float(self.fps)

    @property
    def cumulative_frames(self) -> np.ndarray:
        return np.cumsum([0, *self.frame_counts])


def base_video_id(path_or_stem: str | Path) -> str:
    stem = Path(path_or_stem).stem
    return SEGMENT_RE.sub("", stem)


def load_video_meta(path: str | Path) -> dict[str, dict]:
    """Read the video_meta.csv sidecar -> {video_id: {duration_s, width, height, caption_source}}.

    width/height may be blank (yt-dlp can omit them); caption_source (human|mt|shard|none) may be blank; 
    a blank/zero-duration row is SKIPPED (loader falls back to config pose_fps).
    """
    path = Path(path)
    if not path.exists(): return {}

    def _opt_int(value: str | None) -> int | None:
        value = (value or "").strip()
        return int(float(value)) if value else None

    def _opt_float(value: str | None) -> float | None:
        value = (value or "").strip()
        return float(value) if value else None

    meta: dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            video_id = (row.get("video_id") or "").strip()
            duration = (row.get("duration_s") or "").strip()
            if not video_id or not duration: continue
            meta[video_id] = {
                "duration_s": float(duration),
                "width": _opt_int(row.get("width")), "height": _opt_int(row.get("height")),
                "caption_source": (row.get("caption_source") or "").strip() or None,
                "undetected_ratio": _opt_float(row.get("undetected_ratio")),
                **{key: json.loads(runs) if (runs := (row.get(key) or "").strip()) else None for key in RUN_FIELDS},
            }
    return meta


def drop_from_page_cache(path: str | Path) -> None:
    """Tell the kernel this file is no longer needed, after a one-shot sequential read.

    A full-corpus measurement pass otherwise leaves the whole corpus in the page cache (138 GiB on ase). The cache is
    reclaimable, so this is not a correctness fix; it keeps a maintenance pass from evicting everything else the box
    was holding. Best-effort: platforms without `posix_fadvise` simply skip it.
    """
    advise = getattr(os, "posix_fadvise", None)
    if advise is None: return
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try: advise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally: os.close(fd)
    except OSError: pass


def save_video_meta(path: str | Path, meta: dict[str, dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(META_FIELDS)
        for video_id in sorted(meta):
            m = meta[video_id]
            writer.writerow([
                video_id, m.get("duration_s"),
                "" if m.get("width") is None else m["width"],
                "" if m.get("height") is None else m["height"],
                m.get("caption_source") or "",
                "" if m.get("undetected_ratio") is None else f"{float(m['undetected_ratio']):.4f}",
                *("" if m.get(key) is None else json.dumps(m[key]) for key in RUN_FIELDS),
            ])


def build_pose_index( # Index pose .npy files; fps and frame size are resolved PER VIDEO when `video_meta` covers them
    pose_root: str | Path, fps: float, width: int | None = None, height: int | None = None,
    video_meta: dict[str, dict] | None = None,
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
        meta = (video_meta or {}).get(video_id) or {}
        duration = meta.get("duration_s")
        video_fps = sum(counts) / float(duration) if duration and float(duration) > 0 else float(fps)
        sized = bool(meta.get("width") and meta.get("height"))   # both or neither: a mixed pair is no aspect
        index[video_id] = PoseIndex(
            video_id=video_id, paths=ordered, frame_counts=counts, fps=float(video_fps),
            width=meta["width"] if sized else width, height=meta["height"] if sized else height,
        )
    return index


@lru_cache(maxsize=64)
def _pose_memmap(path_str: str):
    # ONE read-only memmap per pose file, LRU-bounded: re-opening the .npy per window read dominates wall-time on
    # network filesystems. Pose files are immutable; each worker process holds its own cache.
    return np.load(path_str, mmap_mode="r")


def load_pose_frames(pose_index: PoseIndex, start_frame: int, end_frame: int) -> np.ndarray:
    # Frames of the VIEW (PoseIndex.start_frame is its frame 0). The only reader of pose files, so every reader honours it.
    if start_frame < 0 or end_frame < start_frame: raise ValueError(f"Invalid frame range [{start_frame}, {end_frame})")
    end_frame = min(end_frame, pose_index.total_frames)
    if start_frame >= end_frame: return np.zeros((0, 133, 3), dtype=np.float32)
    start_frame, end_frame = start_frame + pose_index.start_frame, end_frame + pose_index.start_frame
    cumulative = pose_index.cumulative_frames

    start_file = int(np.searchsorted(cumulative, start_frame, side="right") - 1)
    end_file = int(np.searchsorted(cumulative, end_frame - 1, side="right") - 1)
    chunks: list[np.ndarray] = []
    for file_idx in range(start_file, end_file + 1):
        local_start = max(0, start_frame - int(cumulative[file_idx]))
        local_end = min(pose_index.frame_counts[file_idx], end_frame - int(cumulative[file_idx]))
        arr = _pose_memmap(str(pose_index.paths[file_idx]))
        chunks.append(np.asarray(arr[local_start:local_end], dtype=np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 133, 3), dtype=np.float32)


def load_pose_window(
    pose_index: PoseIndex, start_s: float, end_s: float, normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    # Load a real-timeline pose window + relative timestamps. `normalize` converts raw (T,133,3) 
    # DWPose to Uni-Sign 69-kp representation (poses.normalize_keypoints_unisign).
    start_s = max(0.0, float(start_s))
    end_s = min(float(end_s), pose_index.duration_s)
    start_frame = int(np.floor(start_s * pose_index.fps))
    end_frame = int(np.ceil(end_s * pose_index.fps))
    poses = load_pose_frames(pose_index, start_frame, end_frame)
    if normalize and poses.shape[1:] == (133, 3): poses = normalize_keypoints_unisign(poses)
    timestamps = (np.arange(poses.shape[0], dtype=np.float32) + start_frame) / float(pose_index.fps)
    return poses.astype(np.float32, copy=False), timestamps
    