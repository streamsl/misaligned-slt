from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

import os, re, csv, json, warnings
import numpy as np
from .preprocessing import normalize_keypoints_unisign
from utils import FRAME_SLACK

SEGMENT_RE = re.compile(r"_segment_(\d+)$")
META_FILENAME = "video_meta.csv"
# caption_source: caption provenance — human | mt (NLLB machine-translation) | shard (raw bundled YouTube track) | none.
# undetected_ratio: share of frames with nobody detected, from SignVerse payload's per-frame `num_persons` (the .npy keeps
# 1 primary person/frame, `prepare_yt25._kept_slots`, so it cannot be recovered from it later). An audit diagnostic only:
# no rule reads it. `--stage convert` writes it for a video it converts; `--stage person-counts` backfills a row left blank
# (UNKNOWN), from the shard, without re-converting.
# empty_runs: a JSON list of half-open .npy frame runs [a, b) where no real body is present (prepare_yt25.empty_runs); [] = measured,
# no run; blank = unknown. Same writers as the counts. The loader cuts them out of every stream (data.loader.split_clean_segments).
# A column that no field names (masked_runs of an earlier converter) is ignored on reading, and the next write drops it.
# handless_runs: the same JSON form, runs where the kept body shows no hand (prepare_yt25.handless_runs, read from the .npy).
# The loader quarantines a caption unit mostly inside them and cuts no frame.
# switch_runs / default_signing_runs: the same JSON form, runs where the kept body is the signer switch's body and not the converter's
# default body / where that default body signs (prepare_yt25._signer_slots, from the SignVerse payloads). The loader quarantines a unit
# by its shares of both (poses.switch_unit_share, poses.switch_doubt_share) and cuts no frame. Only `--stage convert` writes them, with
# the pose file: a blank switch_runs means that the current converter did not write that pose file.
RUN_FIELDS = ("empty_runs", "handless_runs", "switch_runs", "default_signing_runs")
# The frame shape (W, H) that load_pose_frames maps every video to. Stored x/W, y/H keep the frame's aspect, and the Uni-Sign
# body box scales both axes by one number, so a 4:3 or portrait video gives a wider or narrower skeleton than the same person
# in 16:9. 16:9 is the warm start's geometry (OpenASL is YouTube) and most of our hours, so a 16:9 video reads unchanged.
REFERENCE_ASPECT = (16, 9)
# The stamp of that geometry in the meta of every Uni-Sign pose model (S1, the clean floor, the arms; not the Moryossef arm, whose
# shoulder normalisation removes the map, except in a frame with one visible shoulder); eval warns on a checkpoint with another one
# (models.checkpointing.warn_pose_geometry).
POSE_GEOMETRY = f"x_to_{REFERENCE_ASPECT[0]}x{REFERENCE_ASPECT[1]}"
# copy_of: the video-level copy check (prepare_yt25.py --stage dedup, data.loader.train_copies). [held-out video id, score] = this train
# video copies that dev/test video (a re-upload) and the train loader drops it whole; [] = a checked train video that copies none;
# blank = unknown or not a train video (dev and test stay blank). The train loader refuses a blank train record, and a dev or test
# record with a value (a split change after the check). The check reads the captions of every split, so every stage
# that writes captions (convert, subs) blanks it for every video of that language.
# unreliable_spans: [start_s, end_s] spans in SECONDS of the source video where the corpus's own annotation says that its subtitle track
# does not hold the signing (slp-pose writes them: BOBSL's manual cues tagged "not signed" and the like);
# [] = checked, none: every corpus without such annotation (prepare_yt25.py writes [] in every SignVerse row). The loader quarantines
# every unit that overlaps one and puts a quarantined span on it (data.loader mark_unreliable). Blank = unknown, as in slp-pose, which
# exits with 1 while it leaves one (a span source it cannot read); the loader reads a blank cell, and a sidecar written before the
# column, as no span.
JSON_FIELDS = (*RUN_FIELDS, "copy_of", "unreliable_spans")
META_FIELDS = ("video_id", "duration_s", "width", "height", "caption_source", "undetected_ratio", *JSON_FIELDS)
# The body points (COCO-WholeBody 0-16) in the form SignVerse stores them (prepare_yt25._part_arrays): its DWPose output marks a body
# joint visible when the joint's confidence is above BODY_VISIBLE_SCORE and keeps no confidence, so a YouTube-SL-25 body score is 1
# (visible) or 0 (absent, x = y = 0), and SignVerse zeroes a point more than OFFSCREEN_TOLERANCE (in frame units) outside the frame.
# load_pose_frames gives the body of every corpus in that form, so raw DWPose body scores of another extractor read as every
# YouTube-SL-25-trained model learned them; on SignVerse files it changes nothing. Hands and face keep their raw scores everywhere.
BODY_SLICE = slice(0, 17)
BODY_VISIBLE_SCORE = 0.3
OFFSCREEN_TOLERANCE = 0.25
# Repeated frames (content_frames): the points whose motion tells a stored frame that repeats its predecessor (the same image, decoded
# again) from a new one: shoulders, elbows, wrists and both hands. The noise floor of their motion spectrum is the median line at
# these frequencies (cycles per stored frame), away from every pulldown line and its harmonics.
MOTION_POINTS = np.r_[5:11, 91:133]
_FLOOR_FREQUENCIES = (0.137, 0.233, 0.271, 0.311, 0.371, 0.429, 0.457)


@dataclass(frozen=True)
class PoseIndex:
    # `video_id` is the SOURCE video. A clean-segment record (data.loader.split_clean_segments) views frames
    # [start_frame, start_frame + num_frames) of it; the defaults (0, None) view the whole file. Every frame index and
    # time of a view is local (frame 0 = start_frame, time 0 = offset_s on the source timeline). Every frame index outside this
    # module is on the `fps` grid; only the reader (stored_frames) touches the stored frames.
    video_id: str
    paths: tuple[Path, ...]
    frame_counts: tuple[int, ...]   # STORED frames per file
    fps: float
    # Pixel frame size of THIS video (video_meta.csv width/height, else data.yaml pose.width/height). load_pose_frames
    # reads it to map the stored per-axis x/W, y/H to the geometry of a 16:9 frame (REFERENCE_ASPECT).
    width: int | None = None
    height: int | None = None
    start_frame: int = 0
    num_frames: int | None = None
    # The stored rate when the file is not on the `fps` grid (data.yaml poses.model_fps): the reader then gives, at each frame time of
    # the grid, the nearest stored frame. None: frame k is stored frame k (every SignVerse file, and any file at the model rate).
    source_fps: float | None = None
    # The stored frame that each grid frame of the whole video reads, when the file repeats frames to fill its stored rate (How2Sign's
    # 24 fps video in 30 and 60 fps files): the content frames only (content_frames), each near its time (_grid_read). None: the
    # nearest stored frame in time (stored_frames).
    grid_stored: tuple[int, ...] | None = None

    @property
    def grid_frames(self) -> int:  # frames of the whole video on the `fps` grid
        stored = int(sum(self.frame_counts))
        if self.source_fps is None: return stored
        if self.grid_stored is not None: return len(self.grid_stored)
        # The last grid frame k reads stored frame floor(k * source_fps / fps + 0.5), which must exist: k < (stored - 0.5) * fps / source_fps.
        return max(0, int(np.ceil((stored - 0.5) * self.fps / self.source_fps)))

    @property
    def total_frames(self) -> int:
        return self.grid_frames if self.num_frames is None else int(self.num_frames)

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
                **{key: json.loads(runs) if (runs := (row.get(key) or "").strip()) else None for key in JSON_FIELDS},
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
    # A write stopped half way (a kill, an error) must not leave a cut file: the loader would read it and drop the rows after the cut.
    # So write a temporary file in the same folder, then put it on the target in one step (os.replace is atomic on one file system).
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as f:
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
                    *("" if m.get(key) is None else json.dumps(m[key]) for key in JSON_FIELDS),
                ])
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally: tmp.unlink(missing_ok=True)


def _pulldowns(stored_fps: float, content_fps: Iterable[float]) -> list[tuple[float, int, int]]:
    # (content rate, N, M) for each content rate below `stored_fps` that a pulldown can fill it from: M of each N stored frames repeat
    # a frame, with M / N the share of repeats (1 - content / stored) to 0.1 frame, N at most 12 (a share of 1 repeat in 25 frames, a
    # 24 fps video in a 25 fps file, is no pulldown that this reader repairs).
    out = []
    for c in content_fps:
        share = 1.0 - float(c) / float(stored_fps)
        if share <= 0: continue
        n = next((n for n in range(1, 13) if round(n * share) >= 1 and abs(n * share - round(n * share)) < 0.1), None)
        if n is not None: out.append((float(c), n, int(round(n * share))))
    return out


def _motion_into(poses: np.ndarray) -> np.ndarray:
    # The motion into each stored frame: the median displacement, from the frame before, of the points visible in both (`poses` holds
    # the MOTION_POINTS); inf for frame 0 and for a frame with no such point, so neither is ever taken for a repeat.
    seen = (poses[1:, :, 2] > BODY_VISIBLE_SCORE) & (poses[:-1, :, 2] > BODY_VISIBLE_SCORE)
    step = np.where(seen, np.linalg.norm(poses[1:, :, :2] - poses[:-1, :, :2], axis=-1), np.nan)
    with warnings.catch_warnings():   # a frame with no visible point is an all-NaN row: inf below
        warnings.simplefilter("ignore", RuntimeWarning)
        step = np.nanmedian(step, axis=1)
    return np.concatenate([[np.inf], np.where(np.isfinite(step), step, np.inf)])


def content_frames(
    poses: np.ndarray, stored_fps: float, content_fps: Iterable[float], line_ratio: float, repeat_motion: float,
) -> tuple[int, ...] | None:
    """The stored frames that show new content, when a pose file repeats frames to fill its stored rate (a pulldown: video of a rate in
    `content_fps` stored at a higher rate, such as How2Sign's 24 fps video in 30 and 60 fps files or 25 fps broadcasts in 29.97 fps
    files); else None. `poses` holds the MOTION_POINTS of every stored frame. The repeats make the motion into each frame periodic:
    content of rate c shows a line at c / stored_fps (folded to [0, 0.5] cycles per stored frame) in the motion spectrum, which must
    exceed `line_ratio` times its noise floor (data.yaml poses.pulldown_line_ratio); the strongest such line names the content rate.
    Then each window of N stored frames holds M repeats (_pulldowns), and its M frames with the least motion are the repeats, as
    ffmpeg's decimate filter drops them from the images. A repeat shows about no motion: the median motion into these frames must be
    at most `repeat_motion` times the median motion into the others (data.yaml poses.pulldown_repeat_motion), else a periodic motion of
    real frames made the line, and None; and a frame that moves more than that is kept whatever its window, so a window with no repeat
    (the first one, or a gap of a drifting pulldown) loses no frame. The content frames are then the frames of the video before it was
    filled (a few noisy repeats can stay: _grid_read keeps the timing whatever the content frames are)."""
    candidates = _pulldowns(stored_fps, content_fps)
    into = _motion_into(np.asarray(poses, dtype=np.float32))
    finite = into[np.isfinite(into)]
    if not candidates or len(finite) < 2 or float(finite.sum()) <= 0: return None
    x = np.where(np.isfinite(into), into, np.median(finite))
    line = lambda f: float(np.abs(np.dot(x, np.exp(-2j * np.pi * f * np.arange(len(x))))) / x.sum())
    floor = float(np.median([line(f) for f in _FLOOR_FREQUENCIES]))
    folded = lambda c: min(c / float(stored_fps) % 1.0, 1.0 - c / float(stored_fps) % 1.0)
    strength, n, m = max(((line(folded(c)), n, m) for c, n, m in candidates), key=lambda t: t[0])
    if strength <= float(line_ratio) * floor: return None
    keep = np.ones(len(into), dtype=bool)
    for w in range(0, len(into), n):
        segment = into[w:w + n]
        keep[w + np.argsort(segment, kind="stable")[:int(round(len(segment) * m / n))]] = False
    kept, dropped = into[keep & np.isfinite(into)], into[~keep & np.isfinite(into)]
    if not len(kept) or not len(dropped): return None
    limit = float(repeat_motion) * float(np.median(kept))
    if float(np.median(dropped)) > limit: return None
    keep |= into > limit
    return tuple(int(i) for i in np.flatnonzero(keep))


def _grid_read(content: tuple[int, ...], stored_fps: float, fps: float, grid_frames: int) -> tuple[int, ...]:
    # The stored frame that each grid frame k reads from the content frames: the one nearest in time to k / fps (by its first stored
    # frame), except that a content frame read already gives way to the next one when that one comes at most 1 grid period plus 1
    # stored frame after k / fps (the jitter of a first stored frame). So a tie never reads 1 content frame twice, no frame is read
    # before an earlier one, and the time error stays within that bound anywhere in the file (a rate fitted to the count of content
    # frames would drift by seconds where noisy repeats stay in). Sequential, so the whole video at once, at index time.
    frames = np.asarray(content, dtype=np.int64)
    times, t = frames / float(stored_fps), np.arange(grid_frames) / float(fps)
    pos = np.searchsorted(times, t)
    lo, hi = np.clip(pos - 1, 0, len(times) - 1), np.clip(pos, 0, len(times) - 1)
    # A tie goes to the earlier frame whatever the last bits of the stored rate (24 fps content in a 30 fps file puts every 5th grid
    # time midway between 2 content frames, and a rate such as 4416 / 147.2 s = 30.000000000000004 tipped such ties to the later one,
    # from which the give-way below never came back): no read runs 1 content frame ahead through a tie.
    nearest = np.where(np.abs(times[lo] - t) <= np.abs(times[hi] - t) + 1e-9, lo, hi)
    out, prev = np.empty(grid_frames, dtype=np.int64), -1
    for k in range(grid_frames):
        j = int(nearest[k])
        if j <= prev and prev + 1 < len(times) and times[prev + 1] - t[k] <= 1.0 / float(fps) + 1.0 / float(stored_fps): j = prev + 1
        prev = out[k] = max(j, prev)
    return tuple(int(i) for i in frames[out])


def build_pose_index( # Index pose .npy files; fps (frames / duration_s) and frame size are resolved PER VIDEO from `video_meta`
    pose_root: str | Path, fps: float | None = None, width: int | None = None, height: int | None = None,
    video_meta: dict[str, dict] | None = None, video_ids: Iterable[str] | None = None, model_fps: float | None = None,
    pulldown: tuple[Iterable[float], float, float] | None = None,
) -> dict[str, PoseIndex]:
    # `video_ids`: index only these videos, so a loader reads the .npy header of its own split's videos only.
    # `model_fps` (data.yaml poses.model_fps): a video whose stored rate is not this rate to 0.01 fps (the precision the attention band
    # reads, train.bio_pretrain.slowest_train_fps) is indexed on the model grid, and the reader resamples it (stored_frames). A video at
    # the model rate keeps its own frames / duration_s rate, so a SignVerse file (24 fps, duration rounded to 1 ms) reads as before.
    # None: every video keeps its stored grid (the converter's own measurements in prepare_yt25 run on stored frames).
    # `pulldown` (data.yaml poses.pulldown_content_fps, pulldown_line_ratio, pulldown_repeat_motion): a resampled video at a rate that a pulldown can fill
    # from 1 of these content rates is read whole once, for its repeated frames (content_frames); every other video only by its header.
    pose_root = Path(pose_root)
    wanted = None if video_ids is None else set(video_ids)
    grouped: dict[str, list[Path]] = {}
    for path in sorted(pose_root.glob("*.npy")):
        if wanted is None or base_video_id(path) in wanted: grouped.setdefault(base_video_id(path), []).append(path)

    def _seg_key(p: Path) -> tuple[int, str]:
        m = SEGMENT_RE.search(p.stem)
        return (int(m.group(1)) if m else -1, p.name)

    def _frames(path: Path) -> int | str:
        # An OSError on the header is the storage, not the data (a network or cloud mount that does not deliver the file, a copy
        # that did not finish), so every such file is listed at once below instead of the first one with a bare errno.
        try: return int(np.load(path, mmap_mode="r").shape[0])
        except OSError as exc: return f"{path.name} ({exc.strerror or exc})"

    ordered_by_video = {video_id: tuple(sorted(paths, key=_seg_key)) for video_id, paths in grouped.items()}
    files = [path for paths in ordered_by_video.values() for path in paths]
    # The headers in parallel threads: on a network or cloud mount each read waits on the storage, not on this process.
    with ThreadPoolExecutor() as pool: frames = dict(zip(files, tqdm(
        pool.map(_frames, files), total=len(files), desc=f"[pose_io] indexing {len(grouped)} videos under {pose_root}", unit="file"
    )))
    index: dict[str, PoseIndex] = {}
    unreadable = [n for n in frames.values() if isinstance(n, str)]
    for video_id, ordered in ordered_by_video.items():
        if any(isinstance(frames[path], str) for path in ordered): continue
        counts = tuple(frames[path] for path in ordered)
        meta = (video_meta or {}).get(video_id) or {}
        duration = meta.get("duration_s")
        video_fps = sum(counts) / float(duration) if duration and float(duration) > 0 else fps  # `fps`: optional fallback rate
        if video_fps is None: raise SystemExit(
            f"[pose_io] {video_id}: no duration_s in video_meta.csv, so its frame rate is unknown. Run "
            f"`python prepare_yt25.py --stage meta --languages <lang>` (yt-dlp metadata, no video download)."
        )
        sized = bool(meta.get("width") and meta.get("height"))   # both or neither: a mixed pair is no aspect
        source_fps = grid_stored = None
        if model_fps is not None and round(float(video_fps), 2) != round(float(model_fps), 2):
            source_fps, video_fps = float(video_fps), float(model_fps)
            if pulldown is not None and _pulldowns(source_fps, pulldown[0]):
                points = np.concatenate([np.asarray(np.load(path, mmap_mode="r")[:, MOTION_POINTS]) for path in ordered])
                if (content := content_frames(points, source_fps, *pulldown)) is not None:
                    stored = int(sum(counts))
                    grid_stored = _grid_read(content, source_fps, video_fps, max(0, int(np.ceil((stored - 0.5) * video_fps / source_fps))))
        index[video_id] = PoseIndex(
            video_id=video_id, paths=ordered, frame_counts=counts, fps=float(video_fps),
            width=meta["width"] if sized else width, height=meta["height"] if sized else height, source_fps=source_fps,
            grid_stored=grid_stored,
        )
    if unreadable: raise SystemExit(
        f"[pose_io] the operating system cannot read {len(unreadable)} pose file(s) under {pose_root}: {', '.join(unreadable[:10])}"
        f"{', ...' if len(unreadable) > 10 else ''}. Copy these files again, or move the data to a local disk."
    )
    return index


@lru_cache(maxsize=64)
def _pose_memmap(path_str: str):
    # ONE read-only memmap per pose file, LRU-bounded: re-opening the .npy per window read dominates wall-time on
    # network filesystems. Pose files are immutable; each worker process holds its own cache.
    return np.load(path_str, mmap_mode="r")


def stored_frames(pose_index: PoseIndex, start_frame: int, end_frame: int) -> np.ndarray:
    # The stored frame that each grid frame [start_frame, end_frame) of the VIEW reads: itself on a file at the grid rate, else the
    # nearest stored frame in time (floor(k * source_fps / fps + 0.5), as ffmpeg's fps filter picks frames); on a file with repeated
    # frames, the content frame that the index chose for it (PoseIndex.grid_stored).
    k = np.arange(start_frame + pose_index.start_frame, end_frame + pose_index.start_frame, dtype=np.int64)
    if pose_index.source_fps is None: return k
    if pose_index.grid_stored is not None: return np.asarray(pose_index.grid_stored, dtype=np.int64)[k]
    nearest = np.floor(k * (float(pose_index.source_fps) / float(pose_index.fps)) + 0.5).astype(np.int64)
    return np.minimum(nearest, int(sum(pose_index.frame_counts)) - 1)


def grid_runs(pose_index: PoseIndex, runs: list | None) -> list | None:
    # Stored-frame runs [a, b) of a whole video (the video_meta.csv run columns) on the grid that the reader gives it: a grid frame is
    # inside a run when the stored frame it reads is. A file at the grid rate keeps its runs as they are; None (unknown) stays None.
    if runs is None or pose_index.source_fps is None or not runs: return runs
    inside = np.zeros(int(sum(pose_index.frame_counts)), dtype=bool)
    for a, b in runs: inside[int(a):int(b)] = True
    flags = inside[stored_frames(replace(pose_index, start_frame=0, num_frames=None), 0, pose_index.grid_frames)]
    edges = np.flatnonzero(np.diff(np.concatenate([[False], flags, [False]]).astype(np.int8)))
    return [[int(a), int(b)] for a, b in zip(edges[::2], edges[1::2])]


def _signverse_body(out: np.ndarray) -> None:
    # The body in SignVerse's form, in place (see BODY_VISIBLE_SCORE): a visible, on-screen point scores 1, any other point is all 0.
    body = out[:, BODY_SLICE]
    visible = (body[..., 2] > BODY_VISIBLE_SCORE) & (np.abs(body[..., :2] - 0.5) <= 0.5 + OFFSCREEN_TOLERANCE).all(axis=-1)
    body[..., 2] = visible
    body[~visible] = 0.0


def load_pose_frames(pose_index: PoseIndex, start_frame: int, end_frame: int) -> np.ndarray:
    # Grid frames of the VIEW (PoseIndex.start_frame is its frame 0). The only reader of pose files, so every reader honours it.
    # The body takes SignVerse's form (_signverse_body), then x is mapped to the REFERENCE_ASPECT geometry: x/W * (W/H) / (16/9),
    # y/H unchanged (a video with no size is read as stored). A SignVerse file at the grid rate reads byte-identical.
    if start_frame < 0 or end_frame < start_frame: raise ValueError(f"Invalid frame range [{start_frame}, {end_frame})")
    end_frame = min(end_frame, pose_index.total_frames)
    if start_frame >= end_frame: return np.zeros((0, 133, 3), dtype=np.float32)
    frames = stored_frames(pose_index, start_frame, end_frame)
    cumulative = pose_index.cumulative_frames
    # Each file serves one block of the (non-decreasing) stored frames: a slice on the grid rate, else the frames it holds.
    bounds = np.searchsorted(frames, cumulative)
    chunks: list[np.ndarray] = []
    for file_idx in range(len(pose_index.frame_counts)):
        lo, hi = int(bounds[file_idx]), int(bounds[file_idx + 1])
        if hi <= lo: continue
        arr, local = _pose_memmap(str(pose_index.paths[file_idx])), frames[lo:hi] - int(cumulative[file_idx])
        rows = arr[int(local[0]):int(local[-1]) + 1] if pose_index.source_fps is None else arr[local]
        chunks.append(np.asarray(rows, dtype=np.float32))
    if not chunks: return np.zeros((0, 133, 3), dtype=np.float32)
    out = np.concatenate(chunks, axis=0)   # a copy, so the changes below never write the memmap
    _signverse_body(out)
    (rw, rh), w, h = REFERENCE_ASPECT, pose_index.width, pose_index.height
    if w and h and w * rh != h * rw: out[..., 0] *= np.float32(w * rh / (h * rw))   # exact 16:9: no multiply, byte-identical
    return out


def window_frames(fps: float, start_s: float, end_s: float) -> tuple[int, int]:
    # [start_s, end_s) in frames [start_frame, end_frame); a bound within FRAME_SLACK of a frame time is that frame,
    # so a saved event time maps back to its frame. load_pose_window and the S1 tiling (data.chunks.draw_chunks) read it.
    return int(np.floor(start_s * fps + FRAME_SLACK)), int(np.ceil(end_s * fps - FRAME_SLACK))


def load_pose_window(
    pose_index: PoseIndex, start_s: float, end_s: float, normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    # Load a real-timeline pose window + relative timestamps. `normalize` converts raw (T,133,3)
    # DWPose to Uni-Sign 69-kp representation (poses.normalize_keypoints_unisign).
    start_s = max(0.0, float(start_s))
    end_s = min(float(end_s), pose_index.duration_s)
    start_frame, end_frame = window_frames(pose_index.fps, start_s, end_s)
    poses = load_pose_frames(pose_index, start_frame, end_frame)
    if normalize and poses.shape[1:] == (133, 3): poses = normalize_keypoints_unisign(poses)
    timestamps = (np.arange(poses.shape[0], dtype=np.float32) + start_frame) / float(pose_index.fps)
    return poses.astype(np.float32, copy=False), timestamps
    