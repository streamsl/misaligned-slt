"""Synthetic streaming-corpus loader (PHOENIX / CSL-Daily / How2Sign / ...).

Loads any corpus built by the author's e2e-slt-streaming `data_synth` pipeline — pre-trimmed SLT
benchmark cues concatenated into continuous streams with gaps/phantoms — onto the same
`VideoRecord`/`SentenceSpan`/`PoseIndex` abstraction the YouTube path uses, so every downstream stage
(VLP, segmenter, stage-2, analysis, eval) runs unchanged via `load_language_records` dispatch.

All `data_synth` corpora share ONE schema, so this single module serves every dataset; a corpus is
selected by its `data.yaml` language entry (key + `root` + `target_lang`), not by code here:
  <root>/
    manifest.json          # build params incl. target_fps; per-stream construction metadata
    subset2episode.json    # {"train":[stream_id,...], "val":[...], "test":[...]}
    poses/<stream_id>.npy  # (T, 133, 3) COCO-WholeBody, RAW pixel coords, exactly target_fps
    vtt/<stream_id>.vtt    # per-cue (start_s, end_s, text) on the FINAL stream timeline

Why this is its own loader rather than a config of the YouTube one:
  - fps is an EXACT constant (target_fps) by construction: T/target_fps == stream duration to the frame,
    so there is no per-video fps calibration (the whole video_meta.csv machinery is unnecessary here).
  - splits are explicit (subset2episode.json), not a SignVerse CSV / hash fallback. "val" -> "dev".
  - one canonical .vtt per stream, so no find_best_subtitle suffix scoring / flattened-transcript guard.

Per-dataset differences (German PHOENIX vs Chinese CSL-Daily vs English How2Sign) live entirely in the
config entry's `target_lang` (drives the mBART trim) — the schema and this code are language-agnostic.
"""
from __future__ import annotations
from pathlib import Path
import json

import numpy as np
from data.loader import VideoRecord, merge_rolling_captions, parse_vtt
from data.windowing import SentenceSpan
from poses.pose_io import PoseIndex


def _stream_frame_size(root: Path, lang_cfg: dict) -> tuple[int | None, int | None]:
    # Pixel frame the raw pose coordinates live in, needed by the MSKA (dsta) pose representation's
    # global normalization (x/w, (h-y)/h). Authoritative source is the builder's manifest src_meta
    # (PHOENIX: src_w=210, src_h=260); config keys width/height override. The CoSign
    # representation ignores these (its group normalization is resolution-independent).
    width = lang_cfg.get("width")
    height = lang_cfg.get("height")
    if width and height: return int(width), int(height)
    manifest = root / "manifest.json"
    if manifest.exists():
        try:
            src = json.loads(manifest.read_text(encoding="utf-8")).get("src_meta", {})
            if src.get("src_w") and src.get("src_h"): return int(src["src_w"]), int(src["src_h"])
        except (OSError, ValueError, json.JSONDecodeError): pass
    return (int(width) if width else None, int(height) if height else None)


def _stream_target_fps(root: Path, lang_cfg: dict) -> float:
    # Prefer the builder's recorded target_fps (authoritative: poses were resampled to it); fall back to config.
    manifest = root / "manifest.json"
    if manifest.exists():
        try:
            fps = json.loads(manifest.read_text(encoding="utf-8")).get("target_fps")
            if fps: return float(fps)
        except (OSError, ValueError, json.JSONDecodeError): pass
    return float(lang_cfg.get("pose_fps", 12.5))


def _stream_splits(root: Path) -> dict[str, list[str]]:
    # Read subset2episode.json -> {train, dev, test} stream-id lists ('val' renamed 'dev').
    index = root / "subset2episode.json"
    raw: dict[str, list[str]]
    if index.exists(): raw = json.loads(index.read_text(encoding="utf-8"))
    else: # # Falls back to the stream-id filename prefix (train_/val_/test_) if the index is absent.
        raw = {"train": [], "val": [], "test": []}
        for path in sorted((root / "poses").glob("*.npy")):
            prefix = path.stem.split("_", 1)[0]
            raw.setdefault(prefix, []).append(path.stem)

    out: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for key, ids in raw.items():
        dest = "dev" if key == "val" else key
        if dest in out: out[dest] = sorted(ids)
    return out


def load_stream_records(data_cfg: dict, language: str, split: str | None = None) -> tuple[list[VideoRecord], dict[str, list[str]]]:
    # Build `VideoRecord`s for a synthetic streaming corpus (drop-in for `load_language_records`).
    lang_cfg = data_cfg["languages"][language]
    root = Path(lang_cfg["root"])
    fps = _stream_target_fps(root, lang_cfg)
    width, height = _stream_frame_size(root, lang_cfg)
    splits = _stream_splits(root)
    selected_ids = splits.get(split, []) if split else sorted(sid for ids in splits.values() for sid in ids)

    subtitle_cfg = data_cfg.get("subtitles", {})
    min_dur = float(subtitle_cfg.get("min_duration_s", 0.2))
    max_dur = float(subtitle_cfg.get("max_duration_s", 60.0))
    # Synthetic captions are clean; noise/flatten/rolling guards are YouTube-specific. merge_rolling is a
    # no-op on non-overlapping cues, kept only to defend the time-ordered-span invariant downstream assumes.
    drop_noise = bool(subtitle_cfg.get("drop_noise_captions", False))

    records: list[VideoRecord] = []
    for stream_id in selected_ids:
        pose_path = root / "poses" / f"{stream_id}.npy"
        vtt_path = root / "vtt" / f"{stream_id}.vtt"
        if not pose_path.exists() or not vtt_path.exists(): continue

        n_frames = int(np.load(pose_path, mmap_mode="r").shape[0])
        pose = PoseIndex(
            video_id=stream_id, paths=(pose_path,), frame_counts=(n_frames,),
            fps=float(fps), width=width, height=height,
            conf_threshold=float(lang_cfg.get("confidence_threshold", 0.5)),
        )
        captions = merge_rolling_captions(parse_vtt(vtt_path, drop_noise=drop_noise))
        spans = tuple(
            SentenceSpan(video_id=stream_id, start_s=s, end_s=e, text=t) for s, e, t in captions
            if min_dur <= (e - s) <= max_dur and e <= pose.duration_s + 1.0
        )
        if spans: records.append(VideoRecord(language, stream_id, pose, vtt_path, spans))
    return records, splits
