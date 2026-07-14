"""Uni-Sign-format PRE-TRIMMED dataset (CSL-Daily / How2Sign as released by ZechengLi19/Uni-Sign) as ordinary `VideoRecord`s, 
so the EXISTING entry points evaluate it unchanged — no parallel eval mode.

`eval.py --rq 1 --language csl_pretrimmed --severity-grid-s 0.0 --method baseline` is the clean literature-comparison 
point (Uni-Sign CSL-Daily test BLEU-4 25.61; reproduced 25.57 via this path). Which non-zero grid points are valid here follows 
from what the clips contain: CSL-Daily is a studio corpus, ONE sentence per take, and Uni-Sign's own loader feeds exactly 
[start, end) — so the stored margin frames are the same take's REST POSE (median ~0.55 s per side on test), never a neighbouring
sentence. Truncation points (head ≥ 0, tail ≤ 0) are therefore fully valid (they need no context — the linchpin right-truncation 
sweep runs here, on the official sentence set); extension points mostly clamp and can never produce neighbour-sentence contamination 
— run those on a continuous corpus (synthetic streams / native video). Synthetic streams remain for training and RQ2.

Format (Uni-Sign datasets.py S2T_Dataset / load_pose):
  labels.{train,dev,test}  gzip pickle: {name: {name, gloss, text, video_path}}
  poses/<name>.pkl         plain pickle: keypoints [T×(1,133,2) normalized xy], scores [T×(1,133)],
                           start/end (valid sentence range inside the take), w_h.

Each pkl becomes one single-sentence VideoRecord on its own timeline: sentence span = [start/fps, end/fps), pose = the WHOLE take 
(context included). Poses are materialized once into (T,133,3) .npy next to the pkls (poses_npy/) so `PoseIndex`/`load_pose_window` 
— and therefore normalization, augmentation hooks, and every consumer — run the exact same code path as the other corpora. Coordinates 
stay in the stored normalized [0,1] frame: `normalize_keypoints_unisign` is bbox-relative and resolution-independent (w_h feeds only 
the RGB branch upstream, which we drop).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import gzip, pickle
import numpy as np
from data.windowing import SentenceSpan
from poses.pose_io import PoseIndex


@dataclass(frozen=True)
class PretrimmedSample:
    name: str
    text: str
    pose_path: Path


def load_pretrimmed_labels(root: str | Path, split: str) -> list[PretrimmedSample]:
    root = Path(root)
    with gzip.open(root / f"labels.{split}", "rb") as f:
        raw = pickle.load(f)
    samples = []
    for key in raw:  # upstream iterates dict order (S2T_Dataset: self.list = list(raw.keys()))
        row = raw[key]
        pose_name = str(row["video_path"]).replace(".mp4", ".pkl")
        samples.append(PretrimmedSample(name=str(row["name"]), text=str(row["text"]), pose_path=root / "poses" / pose_name))
    return samples


def _materialize_npy(sample: PretrimmedSample, npy_dir: Path) -> tuple[Path, int, int, int]:
    # pkl -> cached (T,133,3) float32 .npy of the WHOLE take. Returns (npy_path, total_frames, start, end).
    npy_path = npy_dir / f"{sample.name}.npy"
    meta_path = npy_dir / f"{sample.name}.meta.npy"
    if npy_path.exists() and meta_path.exists():
        total, start, end = (int(v) for v in np.load(meta_path))
        return npy_path, total, start, end
        
    with open(sample.pose_path, "rb") as f:
        pose = pickle.load(f)
    kp = np.stack([np.asarray(fr[0], dtype=np.float32) for fr in pose["keypoints"]])  # (T,133,2)
    sc = np.stack([np.asarray(fr[0], dtype=np.float32) for fr in pose["scores"]])     # (T,133)
    arr = np.concatenate([kp, sc[..., None]], axis=-1)                                # (T,133,3)
    start = int(pose.get("start", 0))
    end = int(pose.get("end", arr.shape[0]))
    npy_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, arr)
    np.save(meta_path, np.asarray([arr.shape[0], start, end], dtype=np.int64))
    return npy_path, int(arr.shape[0]), start, end


def load_pretrimmed_records(data_cfg: dict, language: str, split: str | None = None):
    # `load_language_records` backend for `source: pretrimmed` (drop-in return shape).
    from data.loader import VideoRecord  # local import: loader imports this module's sibling symbols

    lang_cfg = data_cfg["languages"][language]
    root = Path(lang_cfg["root"])
    pose_cfg = lang_cfg.get("pose", {}) or {}
    fps = float(pose_cfg.get("fps", 30.0))
    width = int(pose_cfg["width"]) if pose_cfg.get("width") is not None else None
    height = int(pose_cfg["height"]) if pose_cfg.get("height") is not None else None
    split = split or "test"

    npy_dir = root / "poses_npy"
    records: list[VideoRecord] = []
    for sample in load_pretrimmed_labels(root, split):
        npy_path, total, start, end = _materialize_npy(sample, npy_dir)
        pose = PoseIndex(
            video_id=sample.name, paths=(npy_path,), frame_counts=(total,),
            fps=fps, width=width, height=height,
        )
        span = SentenceSpan(sample.name, start / fps, end / fps, sample.text)
        records.append(VideoRecord(language, sample.name, pose, sample.pose_path, (span,)))
    return records, {split: [r.video_id for r in records]}
