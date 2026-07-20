from __future__ import annotations
import re, csv, json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from .preprocessing import normalize_keypoints_unisign

SEGMENT_RE = re.compile(r"_segment_(\d+)$")
META_FILENAME = "video_meta.csv"
# caption_source: provenance of this video's English caption — human | mt (NLLB machine-translation) | shard
# (raw bundled YouTube track) | none. Blank for the own-extraction (ase) path, which does not resolve captions here.
META_FIELDS = ("video_id", "duration_s", "width", "height", "caption_source")
# Default --format for the metadata fetch. Must match the yt-dlp --format the videos were 
# downloaded with, so metadata width/height describe the downloaded stream; override per language via 
# `python -m poses <lang_root> --format SEL` when a language was downloaded at a different (e.g. higher) 
# resolution. Duration — the only fps-calibration input — is the same at every resolution.
YTDLP_FORMAT = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b"


@dataclass(frozen=True)
class PoseIndex:
    video_id: str
    paths: tuple[Path, ...]
    frame_counts: tuple[int, ...]
    fps: float
    # Pixel frame size (from data.yaml pose.width/height), used only by the train-time spatial augmentations
    # (affine / spatial_mask); Uni-Sign normalization is bbox-relative and resolution-independent.
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


def load_video_meta(path: str | Path) -> dict[str, dict]:
    """Read the video_meta.csv sidecar -> {video_id: {duration_s, width, height, caption_source}}.

    CSV columns: video_id, duration_s, width, height, caption_source. width/height may be blank 
    (yt-dlp can omit them); caption_source (human|mt|shard|none) may be blank; a blank/zero duration 
    row is skipped (the loader then falls back to config pose_fps).
    """
    path = Path(path)
    if not path.exists(): return {}

    def _opt_int(value: str | None) -> int | None:
        value = (value or "").strip()
        return int(float(value)) if value else None

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
            }
    return meta


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
            ])


def build_pose_index(
    pose_root: str | Path, fps: float, width: int | None = None, height: int | None = None,
    video_meta: dict[str, dict] | None = None,
) -> dict[str, PoseIndex]:
    """Index pose .npy files; fps is resolved PER VIDEO when `video_meta` covers it.

    Per-video fps = total_pose_frames / real_video_duration. The extraction pipeline kept every
    2nd frame, so effective pose fps is native/2 and the NATIVE rate varies with the source video
    (measured on Auslan: 12.0/12.5/~15.0, plus a few full-rate 25.0). A single config constant
    (kept only as fallback) misplaces frame timestamps: the original 25.0 drifted ~2x (BIO labels
    on the wrong frames — label-motion correlation 0.10 vs 0.26 corrected — and ~44% of captions
    dropped by the loader duration filter); even a hand-set 12.5 leaves 37.7% of videos with >5%
    error (median 26s end-of-video misalignment vs ~3s sentences).

    `width`/`height` come from the explicit args (data.yaml pose.width/height) and are used only by the
    train-time spatial augmentations; Uni-Sign normalization is resolution-independent.
    """
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
        index[video_id] = PoseIndex(
            video_id=video_id, paths=ordered, frame_counts=counts, 
            fps=float(video_fps), width=width, height=height
        )
    return index


def fetch_youtube_meta(
    video_ids: list[str], ytdlp_format: str = YTDLP_FORMAT, workers: int = 4, chunk: int = 25,
) -> dict[str, dict]:
    """yt-dlp METADATA-ONLY fetch -> {video_id: {duration_s, width, height}}. No video download.

    Raw videos are never needed (too heavy to keep for large languages, e.g. ase): duration — the only 
    input the per-video fps calibration needs — comes from YouTube metadata, in WHOLE SECONDS (<=0.5s 
    error ~ 0.3% fps drift, far below sentence length). width/height resolve via `ytdlp_format`; pass 
    the SAME --format selector the videos were downloaded with (`python -m poses <lang_root> --format ...`) 
    so they describe the downloaded stream — note yt-dlp may resolve fewer formats than a browser shows 
    (JS-runtime/PO-token limits), so treat them as advisory. Removed/private videos are skipped and fall 
    back to config pose_fps with a loud loader warning.
    """
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    def _fetch(ids: list[str]) -> dict[str, dict]:
        urls = [f"https://www.youtube.com/watch?v={vid}" for vid in ids]
        try:
            out = subprocess.run(
                ["python", "-m", "yt_dlp", "--skip-download", "--no-warnings", "--ignore-errors",
                 "--format", ytdlp_format, "--print", "%(id)s %(duration)s %(width)s %(height)s", *urls],
                capture_output=True, text=True, timeout=120 * len(ids),
            ).stdout
        except Exception: return {}

        result: dict[str, dict] = {}
        for line in out.splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[1].replace(".", "", 1).isdigit():
                result[parts[0]] = {
                    "duration_s": float(parts[1]),
                    "width": int(parts[2]) if parts[2].isdigit() else None,
                    "height": int(parts[3]) if parts[3].isdigit() else None,
                }
        return result

    chunks = [video_ids[i:i + chunk] for i in range(0, len(video_ids), chunk)]
    meta: dict[str, dict] = {}
    with ThreadPoolExecutor(workers) as ex:
        futures = [ex.submit(_fetch, c) for c in chunks]
        with tqdm(total=len(video_ids), desc="yt-dlp metadata", unit="video") as bar:
            for future in as_completed(futures):
                partial = future.result()
                meta.update(partial)
                bar.update(len(partial))
    return meta


def build_video_meta(lang_root: str | Path, ytdlp_format: str = YTDLP_FORMAT) -> dict[str, dict]:
    """Build/refresh <lang_root>/video_meta.csv. CLI: `python -m poses <lang_root> [--format SEL]`.

    Existing sidecar rows are kept; yt-dlp metadata is fetched only for pose ids still missing (no video 
    download — raw videos are never required). A single constant fps CANNOT replace this: poses are extracted 
    at native/2 and native fps varies per YouTube video — measured on asf, a constant 12.5 leaves 37.7% of 
    videos with >5% timestamp error (median 26s end-of-video misalignment vs ~3s sentences).
    """
    lang_root = Path(lang_root)
    out_path = lang_root / META_FILENAME
    meta = load_video_meta(out_path)
    pose_ids = {base_video_id(p) for p in (lang_root / "poses").glob("*.npy")}

    missing = sorted(pose_ids - set(meta))
    if missing:
        print(f"fetching {len(missing)} videos' metadata from YouTube (yt-dlp, no download)...")
        meta.update(fetch_youtube_meta(missing, ytdlp_format=ytdlp_format))
        
    save_video_meta(out_path, meta)
    still_missing = sorted(pose_ids - set(meta))
    print(f"{len(meta)} videos -> {out_path}; pose ids still missing: {len(still_missing)}"
          + (f" (will fall back to config pose_fps): {still_missing[:5]}..." if still_missing else ""))
    return meta


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


def load_pose_window(
    pose_index: PoseIndex, start_s: float, end_s: float,
    normalize: bool = True, augment=None,
) -> tuple[np.ndarray, np.ndarray]:
    # Load a real-timeline pose window and relative timestamps. When `normalize`, raw (T,133,3) DWPose is
    # converted to the Uni-Sign 69-kp representation (poses.normalize_keypoints_unisign). `augment`, when
    # given (train only), is a callable (raw_poses, width, height) -> raw_poses applied to the RAW pixel
    # keypoints BEFORE normalization — spatial, length-preserving, so timestamps/BIO stay aligned.
    start_s = max(0.0, float(start_s))
    end_s = min(float(end_s), pose_index.duration_s)
    start_frame = int(np.floor(start_s * pose_index.fps))
    end_frame = int(np.ceil(end_s * pose_index.fps))
    poses = load_pose_frames(pose_index, start_frame, end_frame)
    if augment is not None and poses.shape[1:] == (133, 3): poses = augment(poses, pose_index.width, pose_index.height)
    if normalize and poses.shape[1:] == (133, 3): poses = normalize_keypoints_unisign(poses)
    timestamps = (np.arange(poses.shape[0], dtype=np.float32) + start_frame) / float(pose_index.fps)
    return poses.astype(np.float32, copy=False), timestamps
    