"""SignVerse-2M (HuggingFace SignerX/SignVerse-2M) → canonical COCO-WholeBody-133 pose converter.

DWPose keypoints in the 128-point OpenPose layout (18 body + 68 face + 2×21 hands; NO feet), coords NORMALIZED
to the image (x/W, y/H), one npz per video in one of TWO packaging schemes (both verified on real shard bytes):
  - consolidated: <vid>/npz/poses.npz + header arrays (video_id, fps, total_frames, frame_widths, frame_heights,
    frame_indices, frame_payloads[object] — one dict per frame)
  - per-frame:    <vid>/npz/00000001.npz ... (1-based file index = frame index; no fps header — the corpus is
    extracted at a unified 24 fps per the dataset card)

Output (T, 133, 3) float32 keeps coords NORMALIZED [0,1] — the contract of poses.preprocessing.normalize_keypoints_unisign 
used by the released Uni-Sign weights. Do NOT scale to pixels: released weights were trained on x/W,y/H, and crop_scale 
uses ONE shared scale=max(bbox_w,bbox_h) for both axes (aspect-ratio dependent). 

The deleted pre-release README confirms `person_000` = primary signer and OpenPose-18 body (viz: --style openpose), 
but its schema is stale (nested `person_0: {body: float[18,3]}`, `caption.json`, "pixel space") vs the release: 
flat `person_000_*_keypoints (18,2)` + `_scores`, per-video `.vtt`, normalized coords.

Verified quirks handled here (real bytes, shards 000006 + 000270):
  - body_scores sometimes hold the INDICES 0..17, not confidences (~4% of frames) → 1.0 in-frame / 0.0 otherwise.
  - num_persons == 0 → no keypoint arrays → zero frames (velocity masking + conf gating treat all-zero joints as absent).
  - multiple persons (person_001 in the wild) → person_000 only; the per-frame `num_persons` is COUNTED and reported 
    (convert_video `multi_person_frames`), because person_000 is "largest bbox" per frame, so on a two-signer video the 
    tracked identity can switch between frames while the caption stays with 1 signer.
  - off-frame sentinel coords → score 0 beyond a tolerance.
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
_PARTS = (("body", 18), ("face", 68), ("left_hand", 21), ("right_hand", 21))  # (payload key stem, expected joint count)
_EXTRA_PARTS = (("body", 18), ("left_hand", 21), ("right_hand", 21))
_OP_SHOULDERS, _OP_ARMS = (2, 5), (3, 6, 4, 7)   # OpenPose-18: R/L shoulder, then R/L elbow and R/L wrist
_OP_WRIST = {"left_hand": 7, "right_hand": 4}    # OpenPose-18 wrists, the joint a hand crop must attach to
_HAND_GATE = 0.08                                # normalized-image distance: a crop further than this is nobody's

def _part_arrays(payload: dict, part: str, count: int, who: str = "person_000") -> tuple[np.ndarray, np.ndarray] | None:
    kp = payload.get(f"{who}_{part}_keypoints")
    sc = payload.get(f"{who}_{part}_scores")
    if kp is None or sc is None: return None
    kp = np.asarray(kp, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32).reshape(-1)
    if kp.shape != (count, 2) or sc.shape != (count,): return None

    # Extraction bug (DOMINANT on real shards): scores hold the joint INDEX (0..N-1), not a confidence, MIXED with -1 
    # failed-detection sentinels, e.g. [0,1,2,...,7,-1,-1,10,...]. Real confidences are ~[0,1.1], so a score = its own 
    # index >= 2 is impossible — the tell. All non-(-1) scores equal to their index (>= 1 of them >= 2) ⇒ bug frame: 
    # index-scores → 1.0; -1 slots zeroed below. An `allclose(sc, arange)` check missed these hybrid frames: it deleted 
    # the nose (index 0, value 0.0) in ~100% of body-detected frames and leaked indices in as confidences.
    idx = np.arange(count, dtype=np.float32)
    is_idx = sc == idx
    if count > 2 and bool((is_idx | (sc == -1.0)).all()) and bool((is_idx & (idx >= 2)).any()):
        sc = np.where(is_idx, 1.0, sc)
    # Coords far outside the normalized frame are off-screen sentinels: absences, not detections.
    off = (kp[:, 0] < -_COORD_TOLERANCE) | (kp[:, 0] > 1 + _COORD_TOLERANCE) | \
          (kp[:, 1] < -_COORD_TOLERANCE) | (kp[:, 1] > 1 + _COORD_TOLERANCE)
    sc = np.where(off, 0.0, sc)
    # Failed detections carry score sentinels (-1 in the wild) with garbage coords. Zero BOTH: absences must use
    # the pipeline's all-zero-joint convention, not junk geometry.
    bad = sc <= 0.0
    sc = np.where(bad, 0.0, sc)
    kp = np.where(bad[:, None], 0.0, kp)
    return kp, sc


def _confident_coord_sample(payload: dict) -> np.ndarray:
    """RAW detected body coords from one frame, for the pixel-space check. Reads the payload DIRECTLY, not via
    `_part_arrays` (whose off-screen filter zeroes pixel coords first); drops only failed rows (score <= 0)."""
    if not isinstance(payload, dict): return np.empty(0, dtype=np.float32)
    kp = payload.get("person_000_body_keypoints")
    sc = payload.get("person_000_body_scores")
    if kp is None or sc is None: return np.empty(0, dtype=np.float32)
    kp = np.asarray(kp, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32).reshape(-1)
    if kp.shape != (18, 2) or sc.shape != (18,): return np.empty(0, dtype=np.float32)
    return kp[sc > 0.0].reshape(-1)

def assert_normalized_coords(payloads, video_id: str = "") -> None:
    """Fail LOUD if a shard ships PIXEL-space coords: forwarded unchanged into crop_scale they are OOD, and
    `_part_arrays` filters them as off-screen → a silently-empty video. Cheap: first populated frames only."""
    sample: list[float] = []
    for payload in payloads:
        vals = _confident_coord_sample(payload)
        if vals.size: sample.append(float(np.median(np.abs(vals))))
        if len(sample) >= 64: break
    if sample and float(np.median(sample)) > 2.0: raise ValueError( # normalized median ~0.5; pixels are in hundreds
        f"SignVerse video {video_id!r}: confident body coords look PIXEL-space (median |coord| {np.median(sample):.1f}), "
        f"but the converter forwards NORMALIZED [0,1] coords to crop_scale (as every verified shard is). Verify the shard"
        f"shard format (and re-normalize to x/W,y/H) before converting."
    )

def _person_keypoints(payload, who: str) -> np.ndarray:
    # Fixed landmark positions: a missing left hand must not move the right hand into its columns.
    parts = []
    for part, count in _EXTRA_PARTS:
        got = _part_arrays(payload, part, count, who=who)
        xy = np.full((count, 2), np.nan, dtype=np.float64)
        if got is not None:
            kp, score = got
            valid = (score > 0) & np.isfinite(score) & np.isfinite(kp).all(axis=-1)
            xy[valid] = kp[valid]
        parts.append(xy)
    return np.concatenate(parts, axis=0)

def num_persons(payload) -> int:
    return int(payload.get("num_persons", 0) or 0) if isinstance(payload, dict) else 0


def extra_person_motion(payloads) -> float | None:
    """Largest extra-slot ARM travel, in that detection's own shoulder widths.

    A joint needs 10 observations. None means that extra detections exist but cannot be measured, including slots
    whose arms are never detected. Zero covers no extra detections or measured arms with no variation.
    """
    # Slots are detector ranks, not tracked identities; cuts, rank changes and detector errors can raise this score. 
    # It is a conservative data filter, not proof of a living person. Use video review if that matters.
    frames, has_extra = {}, False
    for payload in payloads:
        for slot in range(1, num_persons(payload)):
            has_extra = True
            xy = _person_keypoints(payload, f"person_{slot:03d}")
            right, left = xy[_OP_SHOULDERS[0]], xy[_OP_SHOULDERS[1]]
            if not (np.isfinite(right).all() and np.isfinite(left).all()): continue
            width = float(np.linalg.norm(right - left))
            if width <= 1e-6: continue
            frames.setdefault(slot, []).append((width, xy[list(_OP_ARMS)] - (right + left) / 2.0))
    scores = []
    for rows in frames.values():
        widths = np.array([w for w, _ in rows])
        # Each frame is scaled by ITS OWN shoulder width, so a camera zoom cancels. The floor at half the median
        # stops 1 foreshortened or mis-detected shoulder pair from dividing by nearly 0 and reporting a still
        # picture as violent motion.
        scale = np.maximum(widths, 0.5 * float(np.median(widths)))[:, None, None]
        arms = np.stack([a for _, a in rows]) / scale
        seen = np.isfinite(arms).all(axis=-1)
        keep = seen.sum(axis=0) >= 10
        if not keep.any(): continue
        with np.errstate(invalid="ignore"):
            sd = np.nanstd(np.where(seen[..., None], arms, np.nan), axis=0)
        scores.append(float(np.nanmax(sd[keep])))
    return max(scores) if scores else (None if has_extra else 0.0)





def _primary_hands(payload: dict, body: tuple[np.ndarray, np.ndarray] | None) -> dict:
    """The hands of person_000, taken from whichever slot actually holds them.

    DWPose crops hands from WHOLE IMAGE and distributes the crops over person slots in an order unrelated to body slot, 
    so `person_000_left_hand` is regularly another body's hand. Reading the slot would put a painting's frozen hand on 
    the signer in half the frames of a quarter of the corpus. So each of person_000's 2 wrists claims the crop whose 
    root (point 0) is nearest to it, over every slot in the frame, and a crop further than `_HAND_GATE` is left out — 
    an unclaimed wrist keeps the pipeline's all-zero convention.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if body is None: return out
    body_kp, body_sc = body
    offered: list[tuple[np.ndarray, np.ndarray]] = []

    for slot in range(int(num_persons(payload))):
        for part in ("left_hand", "right_hand"):
            got = _part_arrays(payload, part, 21, who=f"person_{slot:03d}")
            if got is not None and float(np.abs(got[0]).max()) > 0.0: offered.append(got)

    taken: set[int] = set()
    for part, op_idx in _OP_WRIST.items():
        if body_sc[op_idx] <= 0.0: continue
        wrist = body_kp[op_idx]
        best, best_d = None, _HAND_GATE
        for i, (kp, _) in enumerate(offered):
            if i in taken: continue
            d = float(np.linalg.norm(kp[0] - wrist))
            if d < best_d: best, best_d = i, d
        if best is not None:
            taken.add(best)
            out[part] = offered[best]
    return out


def payload_to_coco133(payload: dict) -> np.ndarray:
    # One frame payload → (133, 3) COCO-WholeBody (x, y, conf), coords NORMALIZED [0,1] as stored (see module
    # docstring); zeros where absent. person_000 = primary (largest-bbox) signer; person_001+ ignored.
    out = np.zeros((133, 3), dtype=np.float32)
    if num_persons(payload) < 1: return out
    parts = {p: _part_arrays(payload, p, n) for p, n in _PARTS}
    parts.update(_primary_hands(payload, parts["body"]))   # a hand belongs to wrist it sits on, not to the slot
    body = parts["body"]
    if body is not None:
        kp, sc = body
        for coco_idx, op_idx in enumerate(OPENPOSE18_TO_COCO17):
            out[coco_idx, :2] = kp[op_idx]
            out[coco_idx, 2] = sc[op_idx]
    for part, dest in (("face", COCO_FACE_SLICE), ("left_hand", COCO_LEFT_SLICE), ("right_hand", COCO_RIGHT_SLICE)):
        got = parts[part]
        if got is None: continue
        kp, sc = got
        out[dest, :2] = kp
        out[dest, 2] = sc
    return out


def _load_consolidated(npz_path: Path) -> tuple[np.ndarray, float, int, int, int, int]:
    z = np.load(npz_path, allow_pickle=True)
    # UNIFORM 24 fps corpus (dataset card; verified). Pin to SIGNVERSE_DEFAULT_FPS so convert_video and
    # prepare_data._backfill_meta (which sees only the .npy) agree by construction; warn on a disagreeing header
    # rather than using a rate the crash-recovery path can't reproduce.
    header_fps = float(np.asarray(z["fps"]).item()) if "fps" in z.files else SIGNVERSE_DEFAULT_FPS
    if abs(header_fps - SIGNVERSE_DEFAULT_FPS) > 0.5: print(
        f"[signverse] WARNING: {npz_path} header fps={header_fps} != {SIGNVERSE_DEFAULT_FPS}; "
        f"using {SIGNVERSE_DEFAULT_FPS}", flush=True
    )
    fps = SIGNVERSE_DEFAULT_FPS
    total = int(np.asarray(z["total_frames"]).item()) if "total_frames" in z.files else 0
    payloads = z["frame_payloads"]
    # Two shard layouts exist. SPARSE carries `frame_indices` (1-based video frame numbers) alongside a payload
    # list that may skip undetected frames. DENSE omits it and stores one payload per frame, so the row position
    # IS the frame. Requiring `frame_indices` crashed with KeyError on dense shards.
    if "frame_indices" in z.files: indices = np.asarray(z["frame_indices"]).astype(np.int64)
    else:
        if total and total != len(payloads): raise ValueError(
            f"{npz_path}: dense layout expected one payload per frame, got total_frames={total} but "
            f"{len(payloads)} payloads and no `frame_indices` to align them (keys: {sorted(z.files)})"
        )
        indices = np.arange(1, len(payloads) + 1, dtype=np.int64)
    widths = np.asarray(z["frame_widths"]).astype(np.float64) if "frame_widths" in z.files else None
    heights = np.asarray(z["frame_heights"]).astype(np.float64) if "frame_heights" in z.files else None

    assert_normalized_coords((p for p in payloads if isinstance(p, dict)), video_id=npz_path.parent.parent.name)
    n_frames = max(total, int(indices.max()) if indices.size else 0)
    poses = np.zeros((n_frames, 133, 3), dtype=np.float32)
    frame_w = frame_h = multi = detected = 0
    for row, frame_no in enumerate(indices):
        if int(frame_no) < 1: continue  # 1-based; guard a 0/negative from writing poses[-1]
        payload = payloads[row]
        if not isinstance(payload, dict): continue
        multi += num_persons(payload) >= 2
        detected += num_persons(payload) >= 1
        poses[int(frame_no) - 1] = payload_to_coco133(payload)  # normalized coords — no W/H scaling
        w = float(widths[row]) if widths is not None else float(payload.get("frame_width", 0) or 0)
        h = float(heights[row]) if heights is not None else float(payload.get("frame_height", 0) or 0)
        if w > 0 and h > 0: frame_w, frame_h = int(w), int(h)  # recorded in video_meta (feeds spatial augs only)
    return poses, fps, frame_w, frame_h, multi, detected


def _load_per_frame(npz_dir: Path) -> tuple[np.ndarray, float, int, int, int, int]:
    frames: dict[int, Path] = {}
    for path in npz_dir.glob("*.npz"):
        if path.name == "poses.npz": continue
        try: frames[int(path.stem)] = path              # 00000001.npz → frame 1 (1-based)
        except ValueError: continue
    if not frames: return np.zeros((0, 133, 3), dtype=np.float32), SIGNVERSE_DEFAULT_FPS, 0, 0, 0, 0

    n_frames = max(frames)
    poses = np.zeros((n_frames, 133, 3), dtype=np.float32)
    frame_w = frame_h = multi = detected = 0
    checked = False
    for frame_no, path in sorted(frames.items()):
        if frame_no < 1: continue                       # guard poses[-1]
        z = np.load(path)  # plain numeric arrays only — no allow_pickle (untrusted HF bytes)
        payload = {k: z[k] for k in z.files}
        if not checked:  # pixel-space guard, first POPULATED frame
            assert_normalized_coords([payload], video_id=npz_dir.parent.name)
            checked = bool(_confident_coord_sample(payload).size)
        w = float(np.asarray(payload.get("frame_width", 0)).item() or 0)
        h = float(np.asarray(payload.get("frame_height", 0)).item() or 0)
        multi += num_persons(payload) >= 2
        detected += num_persons(payload) >= 1
        poses[frame_no - 1] = payload_to_coco133(payload)  # coords are normalized [0,1] — no W/H scaling
        if w > 0 and h > 0: frame_w, frame_h = int(w), int(h)  # recorded in video_meta (feeds spatial augs only)
    return poses, SIGNVERSE_DEFAULT_FPS, frame_w, frame_h, multi, detected


def load_signverse_video(npz_dir: str | Path) -> tuple[np.ndarray, float, int, int, int, int]:
    # 1 video's npz dir (either scheme) → ((T,133,3) NORMALIZED-[0,1] poses, fps, width, height,
    # frames with >= 2 detected persons, frames with >= 1 detected person).
    npz_dir = Path(npz_dir)
    consolidated = npz_dir / "poses.npz"
    if consolidated.exists(): return _load_consolidated(consolidated)
    return _load_per_frame(npz_dir)

def convert_video(npz_dir: str | Path, out_npy: str | Path) -> dict:
    # Convert one video → `<out_npy>` ((T,133,3) float32) + summary stats for reporting.
    poses, fps, width, height, multi, detected = load_signverse_video(npz_dir)
    motion = _extra_motion_from_dir(Path(npz_dir)) if multi else 0.0
    out_npy = Path(out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, poses)
    # Use the same detector count as sidecar_stats_from_npz. A detected body with
    # unusable joints is still counted; missing sparse frames count as undetected.
    return {
        "frames": int(poses.shape[0]), "fps": float(fps),
        "duration_s": float(poses.shape[0] / fps) if fps > 0 else 0.0, "empty_frames": int(poses.shape[0] - detected),
        # Share of frames with 2+ people ON SCREEN AT ONCE, from the payload's own per-frame `num_persons`.
        "multi_person_ratio": float(multi / poses.shape[0]) if poses.shape[0] else 0.0, "extra_person_motion": motion,
        "width": int(width), "height": int(height),
    }

def _extra_motion_from_dir(npz_dir: Path) -> float | None:
    consolidated = npz_dir / "poses.npz"
    if consolidated.exists():
        with np.load(consolidated, allow_pickle=True) as z:
            return extra_person_motion(z["frame_payloads"] if "frame_payloads" in z.files else [])
    def payloads():
        for path in sorted(npz_dir.glob("*.npz")):
            with np.load(path) as z:
                yield {k: z[k] for k in z.files}
    return extra_person_motion(payloads())


def sidecar_stats_from_npz(z) -> dict:
    """The video_meta columns only the SHARD carries: person counts and the pixel frame size.

    Reads person counts, frame size and extra-slot keypoints. The `.npy` keeps only the primary pose, so these
    measurements use the source archive. No pose file is written.
    """
    files = set(z.files)
    payloads = z["frame_payloads"] if "frame_payloads" in files else []
    total = int(np.asarray(z["total_frames"]).item()) if "total_frames" in files else 0
    counts = [num_persons(p) for p in payloads]
    # Use the same timeline as _load_consolidated: missing sparse rows still occupy video time.
    last_frame = int(np.max(z["frame_indices"], initial=0)) if "frame_indices" in files else 0
    frames = max(total, len(counts), last_frame)

    width = height = 0
    if "frame_widths" in files and "frame_heights" in files:
        widths, heights = np.asarray(z["frame_widths"]).astype(np.float64), np.asarray(z["frame_heights"]).astype(np.float64)
        good = np.nonzero((widths > 0) & (heights > 0))[0]
        if good.size: width, height = int(widths[good[-1]]), int(heights[good[-1]])

    if not (width and height):
        for payload in payloads:
            if not isinstance(payload, dict): continue
            w = float(np.asarray(payload.get("frame_width", 0)).item() or 0)
            h = float(np.asarray(payload.get("frame_height", 0)).item() or 0)
            if w > 0 and h > 0: width, height = int(w), int(h)
    return {"frames": frames, "multi": sum(1 for c in counts if c >= 2), "detected": sum(1 for c in counts if c >= 1),
            "width": width, "height": height, "extra_motion": extra_person_motion(payloads)}
