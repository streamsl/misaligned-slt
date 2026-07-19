"""SignVerse-2M (HuggingFace SignerX/SignVerse-2M) → canonical COCO-WholeBody-133 pose converter.

SignVerse ships DWPose keypoints in the 128-point OpenPose layout (18 body + 68 face + 2×21 hands; NO feet),
coordinates NORMALIZED to the image (x/W, y/H), one npz per video in one of TWO packaging schemes (both verified
on real shard bytes):
  - consolidated: <vid>/npz/poses.npz with header arrays (video_id, fps, total_frames, frame_widths,
    frame_heights, frame_indices, frame_payloads[object] — one dict per frame)
  - per-frame:    <vid>/npz/00000001.npz ... (1-based file index = frame index; same keys per file; no fps header
    — the corpus is extracted at a unified 24 fps per the dataset card)

This module converts a video's npz payloads to the repo-canonical (T, 133, 3) float32 RAW-PIXEL COCO-WholeBody
array — the exact input contract of poses.preprocessing.normalize_keypoints_unisign — so every downstream stage
(build_pose_index, load_pose_window, augmentations, both segmenters, SLT) runs unchanged on SignVerse data.

Verified data quirks handled here (real bytes, shards 000006 + 000270):
  - body_scores sometimes contain the INDICES 0..17 instead of confidences (~4% of frames): detected per
    part+frame and replaced by 1.0 for in-frame coords / 0.0 otherwise (a naive conf>thr gate would
    permanently delete the nose, score 0.0).
  - frames with num_persons == 0 carry no keypoint arrays → zero frames (the pipeline's missing-detection
    convention; velocity masking + conf gating already treat all-zero joints as absent).
  - multiple persons can be present (person_001 seen in the wild) → person_000 (primary signer) only.
  - sentinel coordinates far outside the frame on off-screen joints → score forced to 0 beyond a tolerance.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# OpenPose BODY_18 index → COCO-17 body slot (COCO-WholeBody 0..16). OpenPose neck (1) has no COCO slot; COCO
# feet (17..22) have no OpenPose-18 source and stay zero. Verified against real-shard geometry (nose above neck,
# viewer-left = subject-right, ankles at the bottom).
#   COCO:      0 nose, 1 Leye, 2 Reye, 3 Lear, 4 Rear, 5 Lsh, 6 Rsh, 7 Lel, 8 Rel, 9 Lwr, 10 Rwr,
#              11 Lhip, 12 Rhip, 13 Lknee, 14 Rknee, 15 Lankle, 16 Rankle
#   OpenPose:  0 nose, 15 Leye, 14 Reye, 17 Lear, 16 Rear, 5 Lsh, 2 Rsh, 6 Lel, 3 Rel, 7 Lwr, 4 Rwr,
#              11 Lhip, 8 Rhip, 12 Lknee, 9 Rknee, 13 Lankle, 10 Rankle
OPENPOSE18_TO_COCO17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
COCO_FACE_SLICE = slice(23, 91)     # 68 face points
COCO_LEFT_SLICE = slice(91, 112)    # 21 left-hand points
COCO_RIGHT_SLICE = slice(112, 133)  # 21 right-hand points

SIGNVERSE_DEFAULT_FPS = 24.0        # dataset card: "unified DWPose keypoint sequences ... at 24 FPS"
_COORD_TOLERANCE = 0.25             # normalized coords beyond [-tol, 1+tol] are off-screen sentinels → score 0
_PARTS = (  # (payload key stem, expected joint count)
    ("body", 18), ("face", 68), ("left_hand", 21), ("right_hand", 21),
)

def _part_arrays(payload: dict, part: str, count: int) -> tuple[np.ndarray, np.ndarray] | None:
    kp = payload.get(f"person_000_{part}_keypoints")
    sc = payload.get(f"person_000_{part}_scores")
    if kp is None or sc is None: return None
    kp = np.asarray(kp, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32).reshape(-1)
    if kp.shape != (count, 2) or sc.shape != (count,): return None

    # Extraction bug (the DOMINANT case on real shards): scores hold the joint INDEX (0..N-1) instead of a confidence, 
    # MIXED with -1 failed-detection sentinels, e.g. [0,1,2,...,7,-1,-1,10,...]. A real confidence is bounded ~[0,1.1], 
    # so an entry equal to its own index >= 2 is an impossible confidence — the unambiguous tell. When every non-(-1) 
    # score equals its index (and at least 1 such index >= 2 confirms the pattern), the frame is a bug frame: replace 
    # the index-scores with 1.0 (present); the -1 slots stay and are zeroed below as absences. (`allclose(sc, arange)` 
    # check never matched these hybrid frames, so it deleted the nose — index 0, value 0.0 — in ~100% of body-detected 
    # frames and leaked raw indices in as confidences.)
    idx = np.arange(count, dtype=np.float32)
    is_idx = sc == idx
    if count > 2 and bool((is_idx | (sc == -1.0)).all()) and bool((is_idx & (idx >= 2)).any()):
        sc = np.where(is_idx, 1.0, sc)
    # Off-screen sentinel coords (far outside the normalized frame) are absences, not detections.
    off = (kp[:, 0] < -_COORD_TOLERANCE) | (kp[:, 0] > 1 + _COORD_TOLERANCE) | \
          (kp[:, 1] < -_COORD_TOLERANCE) | (kp[:, 1] > 1 + _COORD_TOLERANCE)
    sc = np.where(off, 0.0, sc)
    # Failed detections carry score sentinels (-1 observed in the wild) with garbage coords. Zero BOTH so the
    # canonical array uses the pipeline's absence convention (all-zero joint) instead of storing junk geometry.
    bad = sc <= 0.0
    sc = np.where(bad, 0.0, sc)
    kp = np.where(bad[:, None], 0.0, kp)
    return kp, sc


def payload_to_coco133(payload: dict, width: float, height: float) -> np.ndarray:
    # One SignVerse frame payload → (133, 3) RAW-PIXEL COCO-WholeBody (x, y, conf). Zeros where absent.
    out = np.zeros((133, 3), dtype=np.float32)
    if not isinstance(payload, dict) or int(payload.get("num_persons", 0) or 0) < 1: return out

    parts = {p: _part_arrays(payload, p, n) for p, n in _PARTS}
    body = parts["body"]
    if body is not None:
        kp, sc = body
        for coco_idx, op_idx in enumerate(OPENPOSE18_TO_COCO17):
            out[coco_idx, 0] = kp[op_idx, 0] * width
            out[coco_idx, 1] = kp[op_idx, 1] * height
            out[coco_idx, 2] = sc[op_idx]
    for part, dest in (("face", COCO_FACE_SLICE), ("left_hand", COCO_LEFT_SLICE), ("right_hand", COCO_RIGHT_SLICE)):
        got = parts[part]
        if got is None: continue
        kp, sc = got
        out[dest, 0] = kp[:, 0] * width
        out[dest, 1] = kp[:, 1] * height
        out[dest, 2] = sc
    return out


def _load_consolidated(npz_path: Path) -> tuple[np.ndarray, float, int, int]:
    z = np.load(npz_path, allow_pickle=True)
    # SignVerse is a UNIFORM 24 fps corpus (dataset card; verified on real shards). Pin to SIGNVERSE_DEFAULT_FPS so
    # convert_video and prepare_data._backfill_meta (which has only the .npy, no fps) agree by construction; warn if
    # a header ever disagrees rather than silently using a per-video rate the crash-recovery path can't reproduce.
    header_fps = float(np.asarray(z["fps"]).item()) if "fps" in z.files else SIGNVERSE_DEFAULT_FPS
    if abs(header_fps - SIGNVERSE_DEFAULT_FPS) > 0.5: print(
        f"[signverse] WARNING: {npz_path} header fps={header_fps} != {SIGNVERSE_DEFAULT_FPS}; "
        f"using {SIGNVERSE_DEFAULT_FPS}", flush=True
    )
    fps = SIGNVERSE_DEFAULT_FPS
    total = int(np.asarray(z["total_frames"]).item()) if "total_frames" in z.files else 0
    indices = np.asarray(z["frame_indices"]).astype(np.int64)   # 1-based video frame numbers
    payloads = z["frame_payloads"]
    widths = np.asarray(z["frame_widths"]).astype(np.float64) if "frame_widths" in z.files else None
    heights = np.asarray(z["frame_heights"]).astype(np.float64) if "frame_heights" in z.files else None

    n_frames = max(total, int(indices.max()) if indices.size else 0)
    poses = np.zeros((n_frames, 133, 3), dtype=np.float32)
    frame_w = frame_h = 0
    for row, frame_no in enumerate(indices):
        if int(frame_no) < 1: continue  # frame_indices are 1-based; guard against a 0/negative writing poses[-1]
        payload = payloads[row]
        if not isinstance(payload, dict): continue
        w = float(widths[row]) if widths is not None else float(payload.get("frame_width", 0) or 0)
        h = float(heights[row]) if heights is not None else float(payload.get("frame_height", 0) or 0)
        if w <= 0 or h <= 0: continue
        frame_w, frame_h = int(w), int(h)
        poses[int(frame_no) - 1] = payload_to_coco133(payload, w, h)
    return poses, fps, frame_w, frame_h


def _load_per_frame(npz_dir: Path) -> tuple[np.ndarray, float, int, int]:
    frames: dict[int, Path] = {}
    for path in npz_dir.glob("*.npz"):
        if path.name == "poses.npz": continue
        try: frames[int(path.stem)] = path              # 00000001.npz → frame 1 (1-based)
        except ValueError: continue
    if not frames: return np.zeros((0, 133, 3), dtype=np.float32), SIGNVERSE_DEFAULT_FPS, 0, 0

    n_frames = max(frames)
    poses = np.zeros((n_frames, 133, 3), dtype=np.float32)
    frame_w = frame_h = 0
    for frame_no, path in frames.items():
        if frame_no < 1: continue                       # 1-based filenames; guard poses[-1]
        z = np.load(path)  # per-frame npz hold plain numeric arrays only — no allow_pickle (untrusted HF bytes)
        payload = {k: z[k] for k in z.files}
        w = float(np.asarray(payload.get("frame_width", 0)).item() or 0)
        h = float(np.asarray(payload.get("frame_height", 0)).item() or 0)
        if w <= 0 or h <= 0: continue
        frame_w, frame_h = int(w), int(h)
        poses[frame_no - 1] = payload_to_coco133(payload, w, h)
    return poses, SIGNVERSE_DEFAULT_FPS, frame_w, frame_h


def load_signverse_video(npz_dir: str | Path) -> tuple[np.ndarray, float, int, int]:
    # One SignVerse video's npz dir (either packaging scheme) → ((T,133,3) raw-pixel poses, fps, width, height).
    npz_dir = Path(npz_dir)
    consolidated = npz_dir / "poses.npz"
    if consolidated.exists(): return _load_consolidated(consolidated)
    return _load_per_frame(npz_dir)


def convert_video(npz_dir: str | Path, out_npy: str | Path) -> dict:
    # Convert one video and write `<out_npy>` ((T,133,3) float32). Returns summary stats for reporting.
    poses, fps, width, height = load_signverse_video(npz_dir)
    out_npy = Path(out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, poses)
    detected = np.any(poses[..., 2] > 0, axis=1) if poses.size else np.zeros((0,), dtype=bool)
    return {
        "frames": int(poses.shape[0]), "fps": float(fps),
        "duration_s": float(poses.shape[0] / fps) if fps > 0 else 0.0,
        "empty_frames": int((~detected).sum()),
        "width": int(width), "height": int(height),
    }
