"""YouTube-SL-25 via SignVerse-2M → the repo's language layout: shard download, pose converter, captions, video_meta.csv.

Fetches only the shards holding the requested languages' videos from HuggingFace SignerX/SignVerse-2M (shards are mixed-language; 
asf+bfi span ~206 of 723 shards, ~80 GB vs ~1.3 TB), converts each wanted video's DWPose-128 npz to (T,133,3) .npy (POSE CONVERTER below), 
writes 1 caption per video and video_meta.csv (duration = npz frames / the corpus's unified 24 fps — no yt-dlp). The language then behaves 
like the repo's own extractions: train.py / analyze.py / eval.py run unchanged with --language asf|bfi.

    python prepare_yt25.py --stage plan [--size]            # shard/video counts (+ HEAD size estimate); no download
    python prepare_yt25.py --stage all --languages asf bfi  # download + convert + subs gap-fill (resumable)
    python prepare_yt25.py --stage subs --languages asf     # gap-fill captions from the subtitles tar only
    python prepare_yt25.py --stage convert --delete-tars    # convert already-downloaded shards, free disk
    python prepare_yt25.py --stage meta --languages asf [--format SEL]  # yt-dlp video_meta rows for pose ids without one

ON-DISK LAYOUT (root = data/youtube-sl-25):
    <root>/SignVerse-2M-metadata_split.csv        # splits (shipped)
    <root>/archive_upload_progress.json           # video→shard index (fetched once)
    <root>/signverse_subtitles_with_english.tar   # captions tar (fetched only if gaps)
    <root>/signverse_shards/*.tar                 # shard-tar CACHE ONLY (transient; --delete-tars frees it)
    <root>/{asf,bfi}/poses/<vid>.npy , subs/<vid>.<target>.vtt , video_meta.csv

CAPTIONS — 1 `<vid>.<target>.vtt` per video, 1 selection rule (data.loader.best_subtitle), 2 paths: `--stage convert` harvests shard-bundled  
track (caption_source=shard, no extra download); `--stage subs` gap-fills the rest from the curated subtitles tar in the target language 
(`configs/data.yaml` target_lang), HUMAN over NLLB machine-English (see `_pick_caption`). No gaps → 700MB tar is never fetched. Provenance → 
video_meta.csv `caption_source` (human|mt|shard|none) so the loader holds the TEST split to human references (`subtitles.human_only_splits`).

The authoritative video→shard index is runtime_state/archive_upload_progress.json::uploaded_folders. Some asf/bfi videos are not yet 
uploaded upstream (past the upload frontier, no failure markers), a few are indexed but absent from their shard (upstream packaging gap):
both are reported and reconciled, not errors.

META — `--stage meta` builds <root>/<lang>/video_meta.csv from YouTube metadata (yt-dlp, no video download) for pose ids it lacks 
(`build_video_meta`), the route for poses extracted outside SignVerse at video's native rate. It reads no split CSV and no shard index. 
Pass --format the SAME selector the videos were downloaded with so width/height columns describe the downloaded stream. `--stage all` 
does not run it: `--stage convert` writes every SignVerse row itself.

POSE CONVERTER — SignVerse-2M DWPose → canonical COCO-WholeBody-133.

DWPose keypoints in the 128-point OpenPose layout (18 body + 68 face + 2×21 hands; NO feet), coords NORMALIZED to the image (x/W, y/H), 
1 npz/video in 1 of 2 packaging schemes (both verified on real shard bytes):
  - consolidated: <vid>/npz/poses.npz + header arrays (video_id, fps, total_frames, frame_widths, frame_heights, frame_indices, 
    frame_payloads[object] — 1 dict/frame)
  - per-frame:    <vid>/npz/00000001.npz ... (1-based file index = frame index; no fps header — the corpus is extracted at a unified 
    24 fps per the dataset card)

Output (T, 133, 3) float32 keeps coords NORMALIZED [0,1] — the contract of poses.preprocessing.normalize_keypoints_unisign used by 
the released Uni-Sign weights. Do NOT scale to pixels: released weights were trained on x/W,y/H, and crop_scale uses ONE shared 
scale=max(bbox_w,bbox_h) for both axes (aspect-ratio dependent).

The converter reads the flat `person_<k>_*_keypoints (18,2)` + `_scores` schema. Slot 0 is the detector's top-scored person, chosen again 
in every frame, so slot order does not certify body size, signing activity, or a persistent identity. Detector boxes and signer identities 
are not present in this schema. `_primary_slots` keeps slot 0 except where it flickers to another real body for < 0.5s, or where slot 0
holds no real body and another slot does. Other slots supply candidate hand crops and the masked-run measurement.

Verified quirks handled here (real bytes, shards 000006 + 000270):
  - body_scores always hold upstream DWPose index code 18*i+j (joint visible, detector conf > 0.3) or -1, never a confidence → 1.0 in-frame / 
    0.0 otherwise. So no per-body confidence exists to rank persons.
  - num_persons == 0 → no keypoint arrays → zero frames (velocity masking + conf gating treat all-zero joints as absent).
  - multiple persons → body/face from the `_primary_slots` slot. The retained body changes identity only where the detector's own switch lasts 
    >= 0.5s (turn-taking, a cut, a sustained wrong pick) or where retained body is lost (< 8 visible joints, or it moves > 0.05 frame heights 
    in 1 frame), and it can differ from the captioned signer. `masked_runs` marks the frame runs where a second real body is present and moves, 
    and `empty_runs` the runs with no real body; the loader cuts both out of every stream (data.loader.split_clean_segments). `handless_runs` 
    marks the runs where the kept body shows no hand; the loader quarantines a unit mostly inside them and cuts no frame.
  - off-frame sentinel coords → score 0 beyond a tolerance.
"""
from __future__ import annotations
import argparse, bisect, io, csv, json, shutil, sys, tarfile, urllib.request, zipfile
import numpy as np

from pathlib import Path
from data.loader import best_subtitle
from poses.pose_io import (META_FILENAME, base_video_id, build_pose_index, drop_from_page_cache, 
                           load_pose_frames, load_video_meta, save_video_meta)
from utils import load_yaml

HF_BASE = "https://huggingface.co/datasets/SignerX/SignVerse-2M/resolve/main"
PROGRESS_JSON = "runtime_state/archive_upload_progress.json"
SUBTITLES_TAR = "signverse_subtitles_with_english.tar"
DEFAULT_SPLIT_CSV = "data/youtube-sl-25/SignVerse-2M-metadata_split.csv"
DEFAULT_CACHE = "data/youtube-sl-25/signverse_shards"
DEFAULT_ROOT = "data/youtube-sl-25"

# Per video the subtitles tar holds: english.en.native.vtt (HUMAN English — the source was already English),
# english.en.nllb.vtt (English MACHINE-translated by Meta's NLLB — noisy), original.<lang>.manual.vtt (raw HUMAN
# upload in the video's own language; `manual` = human).
_ENGLISH_TRACKS = ("english.en.native.vtt", "english.en.nllb.vtt")  # SignVerse only auto-normalizes to English

# Default --format for the `--stage meta` fetch. Must match the yt-dlp --format the videos were downloaded with, so metadata width/height 
# describe the downloaded stream; override per language via `--stage meta --format SEL` when a language was downloaded at a different 
# (e.g. higher) resolution. Duration — the only fps-calibration input — is the same at every resolution.
YTDLP_FORMAT = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b"

# OpenPose BODY_18 index → COCO-17 body slot (COCO-WholeBody 0..16). OpenPose neck (1) has no COCO slot; COCO feet (17..22) have no 
# OpenPose-18 source and stay 0. Verified against real-shard geometry (nose above neck, viewer-left = subject-right, ankles at bottom).
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
_OP_UPPER_TORSO = [0, 1, 2, 5]                   # OpenPose-18 nose, neck, R/L shoulder: the tracked body centre
_REAL_BODY_JOINTS = 8                            # visible body joints for a slot to count as a real body
_SLOT_JUMP, _SLOT_NEAR = 0.10, 0.05              # frame heights: slot-0 jump / other-slot match to the last chosen centre
_SLOT_HOLD_MAX = 12                              # frames (0.5 s at 24 fps): a longer hold is the detector's real choice
_MASK_RUN = 12                                   # frames (0.5 s at 24 fps): runs closer than this merge; shorter masked runs drop
_EMPTY_RUN = 24                                  # frames (1 s at 24 fps): shorter empty runs drop
_MOTION_BLOCK = 48                               # frames (2 s at 24 fps): a masked run is scored and cut per window this long
_HAND_CONF = 0.3                                 # a hand keypoint counts when its confidence is above Uni-Sign's validity threshold (poses.preprocessing.normalize_keypoints_unisign thr)
_HAND_MIN_POINTS = 10                            # of a hand's 21 keypoints: half a hand is a visible hand
_HANDLESS_RUN = _MASK_RUN                        # frames (0.5 s at 24 fps): a shorter no-hand stretch is a sign transition, an occlusion or motion blur, not an absent signer

def _part_arrays(payload: dict, part: str, count: int, who: str = "person_000") -> tuple[np.ndarray, np.ndarray] | None:
    kp = payload.get(f"{who}_{part}_keypoints")
    sc = payload.get(f"{who}_{part}_scores")
    if kp is None or sc is None: return None
    kp = np.asarray(kp, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32).reshape(-1)
    if kp.shape != (count, 2) or sc.shape != (count,): return None

    # Extraction bug (DOMINANT on real shards): scores hold joint's INDEX CODE count*slot + j, not a confidence, MIXED with -1 failed-detection 
    # sentinels, e.g. slot 0 [0,1,2,...,7,-1,-1,10,...], slot 1 [18,19,-1,...]. Real confidences are ~[0,1.1], so a score = its own code >= 2 is 
    # impossible — the tell. All non-(-1) scores equal to their code (>= 1 of them >= 2) ⇒ bug frame: code-scores → 1.0; -1 slots zeroed below. 
    # An `allclose(sc, arange)` check misses these hybrid frames: it would delete the nose (code 0, value 0.0) in ~100% of body-detected frames
    # and leak codes in as confidences.
    slot = int(who.rsplit("_", 1)[1])
    idx = np.arange(slot * count, (slot + 1) * count, dtype=np.float32)
    is_idx = sc == idx
    if count > 2 and bool((is_idx | (sc == -1.0)).all()) and bool((is_idx & (idx >= 2)).any()): sc = np.where(is_idx, 1.0, sc)
    # Coords far outside the normalized frame are off-screen sentinels: absences, not detections.
    off = (kp[:, 0] < -_COORD_TOLERANCE) | (kp[:, 0] > 1 + _COORD_TOLERANCE) | \
          (kp[:, 1] < -_COORD_TOLERANCE) | (kp[:, 1] > 1 + _COORD_TOLERANCE)
    sc = np.where(off, 0.0, sc)
    # Failed detections carry score sentinels (-1 in the wild) with garbage coords. 
    # Zero BOTH: absences must use the pipeline's all-zero-joint convention, not junk geometry.
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


def _shoulders(xy: np.ndarray) -> tuple[np.ndarray, float] | None:
    # (shoulder midpoint, shoulder width) of a `_person_keypoints` array, when both shoulders are visible.
    right, left = xy[_OP_SHOULDERS[0]], xy[_OP_SHOULDERS[1]]
    if not (np.isfinite(right).all() and np.isfinite(left).all()): return None
    width = float(np.linalg.norm(right - left))
    return ((right + left) / 2.0, width) if width > 1e-6 else None

def _extra_arms(payload, kept: int) -> tuple[bool, float | None, list[tuple[int, np.ndarray]]]:
    # What `_arm_motion` reads from 1 frame: whether a real body other than the kept slot shows an arm, the KEPT body's shoulder width
    # (the scale of every arm), and per such body with both shoulders visible, (slot, elbows and wrists minus its shoulder midpoint).
    # A body with no visible elbow or wrist (a head-and-shoulders photo, an inset cut at the chest) cannot sign in this frame, so it
    # is not an extra body here: otherwise its arms are unmeasurable and the whole run is cut.
    kept_shoulders = _shoulders(_person_keypoints(payload, f"person_{kept:03d}"))
    extra, arms = False, []
    for slot in range(num_persons(payload)):
        if slot == kept or _real_body(payload, slot) is None: continue
        xy = _person_keypoints(payload, f"person_{slot:03d}")
        if not np.isfinite(xy[list(_OP_ARMS)]).any(): continue
        extra = True
        if (shoulders := _shoulders(xy)) is not None: arms.append((slot, xy[list(_OP_ARMS)] - shoulders[0]))
    return extra, (kept_shoulders[1] if kept_shoulders else None), arms


def _arm_motion(per_frame) -> float | None:
    """Largest ARM travel of the other real bodies over the `_extra_arms` of a set of frames, in the KEPT body's shoulder widths.

    Each frame is scaled by the kept body's shoulder width in that frame, floored at half its median over the frames (a frame without 
    it takes the median), so a camera zoom cancels. The scale is the signer's, not the other body's own: a partial or false detection 
    (a profile, a body cut at the frame edge, an object) has shoulders near 0 wide, and its own scale turned jitter into tens of widths.
    A joint needs 10 observations. None means that other bodies show arms but can't be measured (too few arm observations, no shoulders 
    on other body, or no kept shoulders in any frame). Zero covers no other body with a visible arm, or measured arms with no variation. 
    Slots are detector ranks, not tracked identities.
    """
    rows = list(per_frame)
    has_extra = any(extra for extra, _, _ in rows)
    known = [width for _, width, _ in rows if width is not None]
    if not known: return None if has_extra else 0.0
    median = float(np.median(known))
    frames = {}
    for _, width, arms in rows:
        scale = max(median if width is None else width, 0.5 * median)
        for slot, xy in arms: frames.setdefault(slot, []).append(xy / scale)
    scores = []
    for joints in frames.values():
        arms = np.stack(joints)
        seen = np.isfinite(arms).all(axis=-1)
        keep = seen.sum(axis=0) >= 10
        if not keep.any(): continue
        # Select the joints BEFORE the standard deviation. A joint with under 2 observations has no deviation, 
        # and numpy warns once per such slice; dropping them first asks only for values that are used.
        sd = np.nanstd(np.where(seen[..., None], arms, np.nan)[:, keep], axis=0)
        scores.append(float(np.nanmax(sd)))
    return max(scores) if scores else (None if has_extra else 0.0)


def _real_body(payload, slot: int) -> tuple[np.ndarray, np.ndarray] | None:
    # The slot's body arrays when at least _REAL_BODY_JOINTS of its joints are visible, else None.
    got = _part_arrays(payload, "body", 18, who=f"person_{slot:03d}")
    return got if got is not None and int((got[1] > 0).sum()) >= _REAL_BODY_JOINTS else None

def _real_bodies(payload) -> int:
    return sum(_real_body(payload, slot) is not None for slot in range(num_persons(payload)))


def masked_runs(frames, slots: np.ndarray, min_motion: float) -> list[list[int]]:
    """Half-open .npy frame runs [a, b) where a real body other than the kept one is present AND moves.

    `frames` gives (0-based frame, payload) in frame order; an absent frame has no body. `slots[t]` is the slot the .npy keeps in frame t
    (`_primary_slots`), so signer's own motion never counts, whichever slot holds it. Frame qualifies with >=2 real bodies. Runs separated 
    by fewer than _MASK_RUN frames merge, and runs shorter than _MASK_RUN frames are dropped. Each run is scored in windows of _MOTION_BLOCK 
    frames at half-window steps, the last one ending at the run end (a run shorter than 1 window is 1 window), so a still picture beside the 
    signer neither hides a short mover nor gets cut with it, and a mover off the window grid is still covered. A window whose own frames are 
    too few to measure (`_arm_motion` None) takes the motion of the whole run, so a picture the detector finds only now and then is not cut.
    A window is cut when its motion is None (the whole run is unmeasurable) or above `min_motion` (data.yaml poses.min_extra_person_motion); 
    overlapping and adjacent cut windows merge. A still 2nd body (picture), or one that never shows an elbow or wrist, is therefore no run.
    """
    runs: list[list[int]] = []
    run: list = []   # [a, b, (frame, _extra_arms) rows since a]: the arm rows, not the payloads, bound memory on a long run

    def close() -> None:
        a, b, rows = run
        if b - a < _MASK_RUN: return
        rows = [row for row in rows if row[0] < b]
        times, arms = [t for t, _ in rows], [row[1] for row in rows]
        whole = _arm_motion(arms)
        for lo in [*range(a, b - _MOTION_BLOCK, _MOTION_BLOCK // 2), max(a, b - _MOTION_BLOCK)]:
            hi = min(lo + _MOTION_BLOCK, b)
            motion = _arm_motion(arms[bisect.bisect_left(times, lo):bisect.bisect_left(times, hi)])
            if motion is None: motion = whole
            if motion is None or motion > min_motion:
                if runs and runs[-1][1] >= lo: runs[-1][1] = max(runs[-1][1], hi)
                else: runs.append([lo, hi])

    for t, payload in frames:
        if run and t - run[1] >= _MASK_RUN: close(); run = []  # no later frame can merge into this run
        qualifies = _real_bodies(payload) >= 2
        if run or qualifies:
            if not run: run = [t, t + 1, []]
            run[2].append((t, _extra_arms(payload, int(slots[t]))))
            if qualifies: run[1] = t + 1
    if run: close()
    return runs


def empty_runs(frames, n_frames: int) -> list[list[int]]:
    """Half-open .npy frame runs [a, b) with no real body (no slot with _REAL_BODY_JOINTS visible joints).

    `frames` gives (0-based frame, payload) for frames < `n_frames`; a frame it omits has no body. 
    Runs separated by fewer than _MASK_RUN frames merge, and runs shorter than _EMPTY_RUN frames are dropped.
    """
    empty = np.ones(n_frames, dtype=np.int8)
    for t, payload in frames: empty[t] = _real_bodies(payload) == 0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], empty, [0]])))
    runs: list[list[int]] = []
    for a, b in zip(edges[::2].tolist(), edges[1::2].tolist()):
        if runs and a - runs[-1][1] < _MASK_RUN: runs[-1][1] = b
        else: runs.append([a, b])
    return [run for run in runs if run[1] - run[0] >= _EMPTY_RUN]


def handless_runs(poses: np.ndarray) -> list[list[int]]:
    """Half-open .npy frame runs [a, b) of at least _HANDLESS_RUN frames in which the kept body shows no hand.

    `poses` is the converted (T, 133, 3) array, so the hands are the ones `_primary_slots` kept. A hand is present in a frame when at least 
    _HAND_MIN_POINTS of its 21 keypoints have confidence above _HAND_CONF; a frame with no body has no hand. Runs never merge across a frame 
    with a hand. Loader quarantines a caption unit that lies mostly inside these runs (data.yaml poses.handless_unit_share) and cuts no frame.
    """
    hands = (poses[:, COCO_LEFT_SLICE, 2] > _HAND_CONF).sum(axis=1) >= _HAND_MIN_POINTS
    hands |= (poses[:, COCO_RIGHT_SLICE, 2] > _HAND_CONF).sum(axis=1) >= _HAND_MIN_POINTS
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (~hands).astype(np.int8), [0]])))
    return [[a, b] for a, b in zip(edges[::2].tolist(), edges[1::2].tolist()) if b - a >= _HANDLESS_RUN]


def fill_handless_runs(lang_root: Path, meta: dict[str, dict], remeasure: bool = False) -> int:
    """Fill every blank handless_runs value of `meta` (every value under `remeasure`) from pose files under <lang_root>/poses; return count.

    A video's `_segment_N` files are read in order as one stream, as the loader reads them. No archive is read, so
    `--stage person-counts` (SignVerse) and `--stage meta` (poses extracted outside SignVerse) both fill the column.
    """
    blank = [vid for vid, row in meta.items() if remeasure or row.get("handless_runs") is None]
    if not blank: return 0
    index = build_pose_index(lang_root / "poses", fps=1.0)   # frame counts only: fps is not read
    blank = [vid for vid in blank if vid in index]
    for vid in blank: meta[vid]["handless_runs"] = handless_runs(load_pose_frames(index[vid], 0, index[vid].total_frames))
    return len(blank)


def _body_centres(payload, width: float, height: float) -> list[np.ndarray | None]:
    # Per slot, the upper-torso centre in frame heights (x scaled by the aspect ratio) of a real body, else None.
    aspect = width / height if width > 0 and height > 0 else 16 / 9   # sizes are always present on the corpus
    out: list[np.ndarray | None] = []
    for slot in range(num_persons(payload)):
        got = _real_body(payload, slot)
        if got is None: out.append(None); continue
        kp, seen = got[0][_OP_UPPER_TORSO], got[1][_OP_UPPER_TORSO] > 0
        out.append(kp[seen].mean(axis=0) * (aspect, 1.0) if seen.any() else None)
    return out


def _primary_slots(centres: list[list[np.ndarray | None]]) -> np.ndarray:
    """Per frame, the slot whose body and face the .npy keeps (`centres[t]` from `_body_centres`).

    The detector re-ranks persons every frame, and slot 0 briefly flickers to another real body. Pass 1 tracks chosen body: where slot 
    0 jumps > _SLOT_JUMP from last chosen center and another real slot is within _SLOT_NEAR of it, that slot is kept. A hold stops at 
    _SLOT_HOLD_MAX frames and tracking moves to slot 0, so a later flicker back to old body is smoothed too. Pass 2 gives every hold 
    of >= _SLOT_HOLD_MAX frames back to slot 0, so detector's sustained choice stays. A frame whose slot 0 holds no real body keeps 
    the real slot nearest to the tracked body (the first real slot before any body is tracked), and pass 2 never gives such a frame 
    back to slot 0. A single-person video whose body is in slot 0 is unchanged. Body size is no guide: largest-body rule picks a 
    still, handless bystander in most frames of some videos.
    """
    slots = np.zeros(len(centres), dtype=np.int64)
    forced = np.zeros(len(centres), dtype=bool)   # slot 0 holds no real body here, so slot 0 is not a choice
    last, held = None, 0
    for t, frame in enumerate(centres):
        if frame and frame[0] is None:
            real = [(0.0 if last is None else float(np.linalg.norm(c - last)), k) for k, c in enumerate(frame) if c is not None]
            if real: slots[t], forced[t] = min(real)[1], True
        elif frame and last is not None and held < _SLOT_HOLD_MAX and np.linalg.norm(frame[0] - last) > _SLOT_JUMP:
            d, k = min((
                (np.linalg.norm(c - last), k) for k, c in enumerate(frame[1:], 1) if c is not None
            ), default=(np.inf, 0))
            if d <= _SLOT_NEAR: slots[t] = k
        held = held + 1 if slots[t] and not forced[t] else 0
        if frame and frame[slots[t]] is not None: last = frame[slots[t]]

    held_frames = (slots != 0) & ~forced
    t = 0
    while t < len(slots):
        u = t + 1
        if held_frames[t]:
            while u < len(slots) and held_frames[u]: u += 1
            if u - t >= _SLOT_HOLD_MAX: slots[t:u] = 0
        t = u
    return slots


def _primary_hands(payload: dict, body: tuple[np.ndarray, np.ndarray] | None) -> dict:
    """The hands of the kept body, taken from whichever slot actually holds them.

    DWPose crops hands from WHOLE IMAGE and distributes crops over person slots in an order unrelated to body slot, so `person_000_left_hand` 
    is regularly another body's hand. Reading slot would put a painting's frozen hand on the signer in half the frames of 1/4 the corpus. 
    So the kept body's 2 wrists claim 2 crops whose roots (point 0) are nearest to them, over every slot in the frame, and a crop further 
    than `_HAND_GATE` is left out. The pairing claims the most sides, then has smallest total wrist-to-root distance, so 2 touching hands 
    are not swapped (with 2 wrists, the better of 2 one-wrist-first greedy orders is that pairing). A side that no wrist claims (no crop 
    within the gate, or an invisible wrist) is absent from the result, and `payload_to_coco133` decides it: 0 in a frame with 2 or more 
    real bodies, the kept slot's own crop otherwise.
    """
    if body is None: return {}
    body_kp, body_sc = body
    offered: list[tuple[np.ndarray, np.ndarray]] = []

    for slot in range(int(num_persons(payload))):
        for part in ("left_hand", "right_hand"):
            got = _part_arrays(payload, part, 21, who=f"person_{slot:03d}")
            if got is not None and float(np.abs(got[0]).max()) > 0.0: offered.append(got)

    def claim(order):
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        taken, total = set(), 0.0
        for part, op_idx in order:
            if body_sc[op_idx] <= 0.0: continue
            wrist = body_kp[op_idx]
            best, best_d = None, _HAND_GATE
            for i, (kp, _) in enumerate(offered):
                if i in taken: continue
                d = float(np.linalg.norm(kp[0] - wrist))
                if d < best_d: best, best_d = i, d
            if best is not None:
                taken.add(best)
                out[part], total = offered[best], total + best_d
        return out, total

    order = list(_OP_WRIST.items())
    return min((claim(order), claim(order[::-1])), key=lambda r: (-len(r[0]), r[1]))[0]


def payload_to_coco133(payload: dict, who: str = "person_000") -> np.ndarray:
    # 1 frame payload → (133, 3) COCO-WholeBody (x, y, conf), coords NORMALIZED [0,1] as stored (see module docstring); zeros where absent. 
    # Body/face from slot `who`; bind hands by wrist location across all slots. A hand that no wrist claims is 0 in a frame with 2+ real 
    # bodies, because the slot's own crop can be another person's hand. With at most 1 real body the slot's own crop stays: no second real 
    # body is there to own it, though the crop can sit far from a visible wrist (crops are not tied to their slot, `_primary_hands`).
    out = np.zeros((133, 3), dtype=np.float32)
    if num_persons(payload) < 1: return out
    parts = {p: _part_arrays(payload, p, n, who=who) for p, n in _PARTS}
    if _real_bodies(payload) >= 2: parts["left_hand"] = parts["right_hand"] = None
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


def _load_consolidated(npz_path: Path) -> tuple[np.ndarray, float, int, int, int]:
    z = np.load(npz_path, allow_pickle=True)
    # UNIFORM 24 fps corpus (dataset card; verified). Pin to SIGNVERSE_DEFAULT_FPS so convert_video and _backfill_meta (which sees only the 
    # .npy) agree by construction; warn on a disagreeing header rather than using a rate the crash-recovery path can't reproduce.
    header_fps = float(np.asarray(z["fps"]).item()) if "fps" in z.files else SIGNVERSE_DEFAULT_FPS
    if abs(header_fps - SIGNVERSE_DEFAULT_FPS) > 0.5: print(
        f"[signverse] WARNING: {npz_path} header fps={header_fps} != {SIGNVERSE_DEFAULT_FPS}; using {SIGNVERSE_DEFAULT_FPS}", flush=True
    )
    fps = SIGNVERSE_DEFAULT_FPS
    total = int(np.asarray(z["total_frames"]).item()) if "total_frames" in z.files else 0
    payloads = z["frame_payloads"]
    # 2 shard layouts exist. SPARSE carries `frame_indices` (1-based video frame numbers) alongside a payload list that may skip undetected 
    # frames. DENSE omits it and stores one payload per frame, so the row position IS the frame, so `frame_indices` is optional.
    if "frame_indices" in z.files: indices = np.asarray(z["frame_indices"]).astype(np.int64)
    else:
        if total and total != len(payloads): raise ValueError(
            f"{npz_path}: dense layout expected 1 payload/frame, got total_frames={total} but "
            f"{len(payloads)} payloads and no `frame_indices` to align them (keys: {sorted(z.files)})"
        )
        indices = np.arange(1, len(payloads) + 1, dtype=np.int64)
    assert_normalized_coords((p for p in payloads if isinstance(p, dict)), video_id=npz_path.parent.parent.name)
    n_frames = max(total, int(indices.max()) if indices.size else 0)
    poses = np.zeros((n_frames, 133, 3), dtype=np.float32)
    frame_w = frame_h = detected = 0
    rows = _frame_rows(z, payloads)
    slots = _kept_slots(rows, n_frames)
    for t, payload, w, h in rows:
        detected += num_persons(payload) >= 1
        poses[t] = payload_to_coco133(payload, who=f"person_{slots[t]:03d}") # normalized coords — no W/H scaling
        if w > 0 and h > 0: frame_w, frame_h = int(w), int(h) # recorded in video_meta (feeds Moryossef aspect correction only)
    return poses, fps, frame_w, frame_h, detected, slots


def _load_per_frame(npz_dir: Path) -> tuple[np.ndarray, float, int, int, int]:
    frames: dict[int, Path] = {}
    for path in npz_dir.glob("*.npz"):
        if path.name == "poses.npz": continue
        try: frames[int(path.stem)] = path  # 00000001.npz → frame 1 (1-based)
        except ValueError: continue
    if not frames: return np.zeros((0, 133, 3), dtype=np.float32), SIGNVERSE_DEFAULT_FPS, 0, 0, 0, np.zeros(0, dtype=np.int64)

    n_frames = max(frames)
    poses = np.zeros((n_frames, 133, 3), dtype=np.float32)
    frame_w = frame_h = detected = 0
    frames = sorted((f, p) for f, p in frames.items() if f >= 1)   # guard poses[-1]
    centres: list[list] = [[] for _ in range(n_frames)]

    for frame_no, path in frames:  # pass 1 reads body arrays only: the slot choice needs no face or hands
        with np.load(path) as z:   # plain numeric arrays only — no allow_pickle (untrusted HF bytes)
            body = {k: z[k] for k in z.files if "_body_" in k or not k.startswith("person_")}
        w = float(np.asarray(body.get("frame_width", 0)).item() or 0)
        h = float(np.asarray(body.get("frame_height", 0)).item() or 0)
        centres[frame_no - 1] = _body_centres(body, w, h)
        if w > 0 and h > 0: frame_w, frame_h = int(w), int(h)  # recorded in video_meta (feeds Moryossef aspect correction only)

    slots = _primary_slots(centres)
    checked = False
    for frame_no, path in frames:
        with np.load(path) as z: payload = {k: z[k] for k in z.files}
        if not checked:  # pixel-space guard, first POPULATED frame
            assert_normalized_coords([payload], video_id=npz_dir.parent.name)
            checked = bool(_confident_coord_sample(payload).size)
        detected += num_persons(payload) >= 1
        poses[frame_no - 1] = payload_to_coco133(payload, who=f"person_{slots[frame_no - 1]:03d}")  # normalized coords
    return poses, SIGNVERSE_DEFAULT_FPS, frame_w, frame_h, detected, slots


def load_signverse_video(npz_dir: str | Path) -> tuple[np.ndarray, float, int, int, int, np.ndarray]:
    # 1 video's npz dir (either scheme) → ((T,133,3) NORMALIZED-[0,1] poses, fps, width, height, frames 
    # with >= 1 detected person, (T,) kept slot per frame).
    npz_dir = Path(npz_dir)
    consolidated = npz_dir / "poses.npz"
    if consolidated.exists(): return _load_consolidated(consolidated)
    return _load_per_frame(npz_dir)

def convert_video(npz_dir: str | Path, out_npy: str | Path, min_motion: float) -> dict:
    # Convert 1 video → `<out_npy>` ((T,133,3) float32) + summary stats for reporting. 
    # `min_motion` is the masked-run motion threshold (data.yaml poses.min_extra_person_motion).
    poses, fps, width, height, detected, slots = load_signverse_video(npz_dir)
    masked, empty = _person_measures(Path(npz_dir), min_motion, poses.shape[0], slots)
    out_npy = Path(out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, poses)
    # Use the same detector count as sidecar_stats_from_npz. A detected body with
    # unusable joints is still counted; missing sparse frames count as undetected.
    return {
        "frames": int(poses.shape[0]), "fps": float(fps), "duration_s": float(poses.shape[0] / fps) if fps > 0 else 0.0, 
        "empty_frames": int(poses.shape[0] - detected), "masked_runs": masked, "empty_runs": empty, "handless_runs": handless_runs(poses), 
        "width": int(width), "height": int(height),
    }

def _frame_rows(z, payloads) -> list[tuple[int, dict, float, float]]:
    # (0-based .npy frame, payload, frame width, frame height) of a consolidated archive `z` whose `frame_payloads` table is `payloads`
    # (loaded once by the caller: each `z[...]` read unpickles it again), in frame order. A SPARSE archive carries 1-based `frame_indices`;
    # a DENSE one stores 1 payload/frame, so its row position is the frame. The size comes from `frame_widths`/`frame_heights` when the
    # archive has them, else from the payload.
    indices = np.asarray(z["frame_indices"]).astype(np.int64) if "frame_indices" in z.files else np.arange(1, len(payloads) + 1)
    widths = np.asarray(z["frame_widths"]).astype(np.float64) if "frame_widths" in z.files else None
    heights = np.asarray(z["frame_heights"]).astype(np.float64) if "frame_heights" in z.files else None
    rows = []
    for row, (f, p) in enumerate(zip(indices, payloads)):
        if f < 1 or not isinstance(p, dict): continue
        w = float(widths[row]) if widths is not None else float(p.get("frame_width", 0) or 0)
        h = float(heights[row]) if heights is not None else float(p.get("frame_height", 0) or 0)
        rows.append((int(f) - 1, p, w, h))
    return sorted(rows, key=lambda r: r[0])

def _kept_slots(rows, n_frames: int) -> np.ndarray:
    # The slot `_primary_slots` keeps per frame, from `_frame_rows` rows: the same choice in --stage convert and --stage person-counts.
    centres: list[list] = [[] for _ in range(n_frames)]
    for t, payload, w, h in rows: centres[t] = _body_centres(payload, w, h)
    return _primary_slots(centres)

def _person_measures(npz_dir: Path, min_motion: float, n_frames: int, slots: np.ndarray) -> tuple[list[list[int]], list[list[int]]]:
    # The masked and the empty runs, from the source payloads: the .npy keeps one body per frame (`slots`, from the loader).
    consolidated = npz_dir / "poses.npz"
    if consolidated.exists():
        with np.load(consolidated, allow_pickle=True) as z: 
            rows = [(t, p) for t, p, _, _ in _frame_rows(z, z["frame_payloads"] if "frame_payloads" in z.files else [])]
        frames = lambda: rows
    else:
        def frames():   # read twice, 1 file at a time, in frame order (numeric, as `_load_per_frame`)
            for path in sorted((p for p in npz_dir.glob("*.npz") if p.stem.isdigit() and int(p.stem) >= 1), key=lambda p: int(p.stem)):
                with np.load(path) as z: yield int(path.stem) - 1, {k: z[k] for k in z.files}
    return masked_runs(frames(), slots, min_motion), empty_runs(frames(), n_frames)


def sidecar_stats_from_npz(z, min_motion: float) -> dict:
    """The video_meta columns only the SHARD carries: detection count, masked and empty runs, pixel frame size.

    Reads person counts, frame size and body keypoints. `.npy` keeps only the primary pose, so these measurements use the source archive. 
    No pose file is written. The runs use same code as `convert_video` (`_frame_rows`, `masked_runs` at same `min_motion`, `empty_runs`).
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
    rows = _frame_rows(z, payloads)
    pairs = [(t, p) for t, p, _, _ in rows]
    return {"frames": frames, "detected": sum(1 for c in counts if c >= 1), "width": width, "height": height, 
            "masked_runs": masked_runs(pairs, _kept_slots(rows, frames), min_motion), "empty_runs": empty_runs(pairs, frames)}


def _lang_targets(data_cfg: dict, languages) -> dict[str, str]:  # en_XX->en, de_DE->de, zh_CN->zh
    langs = data_cfg.get("languages", {})
    return {lang: str(langs.get(lang, {}).get("target_lang", "en_XX") or "en_XX").split("_")[0].lower() for lang in set(languages)}

def _is_caption_file(fname: str) -> bool:
    return fname in _ENGLISH_TRACKS or (fname.startswith("original.") and fname.endswith(".manual.vtt"))

def _pick_caption(files: dict[str, bytes], target_code: str) -> tuple[str, bytes] | None:
    """Best caption in the TARGET language → (provenance 'human'|'mt', vtt bytes), or None.

    Priority: (1) human — english.en.native for target=en, else original.<target>*.manual; (2) NLLB machine-English, ONLY for target=en 
    (SignVerse produces no MT for other targets), so a non-English target with no human original.<target> is genuinely caption-less — 
    the English tracks are the wrong language.
    """
    if target_code == "en" and "english.en.native.vtt" in files: return "human", files["english.en.native.vtt"]
    for name, content in sorted(files.items()):
        if name.startswith(f"original.{target_code}") and name.endswith(".manual.vtt"): return "human", content
    if target_code == "en" and "english.en.nllb.vtt" in files: return "mt", files["english.en.nllb.vtt"]
    return None


def _download(url: str, dest: Path, resume: bool = True) -> Path:
    """Download to `<dest>.part` and promote ONLY once the byte count matches what the server advertised.

    An interrupted transfer ends `copyfileobj` normally, so promoting unconditionally publishes a truncated file that every later stage treats 
    as complete — the download stage skips it ("already present") and the convert stage dies with `tarfile.ReadError: unexpected end of data`. 
    Leaving the `.part` in place instead keeps the bytes for the next resume.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {}
    start = tmp.stat().st_size if resume and tmp.exists() else 0
    if start: headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    expected = None
    try:
        with urllib.request.urlopen(req) as r:
            # Server ignored our Range (200, not 206) → it is sending the WHOLE file;
            # appending onto the partial would corrupt the tar. Restart from byte 0.
            append = start > 0 and r.status == 206
            # 206 reports "bytes A-B/TOTAL"; 200 reports the whole length in Content-Length.
            crange = r.headers.get("Content-Range")
            if crange and "/" in crange: expected = int(crange.rsplit("/", 1)[1])
            elif r.headers.get("Content-Length") is not None: expected = int(r.headers["Content-Length"]) + (start if append else 0)
            with open(tmp, "ab" if append else "wb") as f: shutil.copyfileobj(r, f, length=1 << 20)
    except urllib.error.HTTPError as e:
        if e.code == 416 and tmp.exists(): pass  # already fully downloaded (range beyond EOF)
        else: raise
    got = tmp.stat().st_size if tmp.exists() else 0
    if expected is not None and got != expected: raise IOError(
        f"{dest.name}: incomplete download ({got}/{expected} bytes). Kept {tmp.name} — re-run download stage to resume from byte {got}."
    )
    tmp.rename(dest)
    return dest


def load_plan(split_csv: Path, root: Path, languages: list[str]) -> dict:
    # video→shard plan from the split CSV + the upload index. The index is DATASET metadata (like the split CSV),
    # so it lives in root/ — NOT in the shard-tar cache.
    progress = root / "archive_upload_progress.json"
    if not progress.exists():
        print(f"prepare | fetching upload index → {progress}", flush=True)
        _download(f"{HF_BASE}/{PROGRESS_JSON}", progress)
    uploaded = json.loads(progress.read_text())["uploaded_folders"]

    with split_csv.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("sign_language") in set(languages)]
    plan = {"videos": {}, "missing": [], "shards": {}}
    for r in rows:
        vid, lang = r["video_id"], r["sign_language"]
        shard = uploaded.get(vid)
        if shard is None:
            plan["missing"].append({"video_id": vid, "language": lang})
            continue
        plan["videos"][vid] = {"language": lang, "shard": shard}
        plan["shards"].setdefault(shard, []).append(vid)
    return plan


def stage_plan(args, plan: dict) -> None:
    langs: dict[str, int] = {}
    for v in plan["videos"].values(): langs[v["language"]] = langs.get(v["language"], 0) + 1
    shards = sorted(plan["shards"])
    done = [s for s in shards if (Path(args.cache) / s).exists()]
    print(f"plan | videos: {len(plan['videos'])} ({langs}) | shards: {len(shards)} "
          f"| not yet uploaded upstream: {len(plan['missing'])}")
    print(f"plan | shards already downloaded: {len(done)}/{len(shards)} in {args.cache}")
    if args.size:  # opt-in HEAD probe per undownloaded shard — off by default (206 round-trips)
        total = sum((Path(args.cache) / s).stat().st_size for s in done)
        for s in (x for x in shards if x not in set(done)):
            try:
                req = urllib.request.Request(f"{HF_BASE}/dataset/{s}", method="HEAD")
                with urllib.request.urlopen(req) as r: total += int(r.headers.get("content-length", 0))
            except OSError: pass
        print(f"plan | estimated total download size: {total / 1e9:.1f} GB")
    print("plan | first shards: " + ", ".join(shards[:8]) + (" ..." if len(shards) > 8 else ""))


def stage_download(args, plan: dict) -> None:
    cache = Path(args.cache)
    shards = sorted(plan["shards"])
    if args.limit: shards = shards[: args.limit]
    for i, shard in enumerate(shards, 1):
        dest = cache / shard
        if dest.exists():
            print(f"download | [{i}/{len(shards)}] {shard} already present", flush=True)
            continue
        print(f"download | [{i}/{len(shards)}] {shard} ...", flush=True)
        _download(f"{HF_BASE}/dataset/{shard}", dest)


def stage_verify(args, plan: dict) -> None:
    """Check every cached shard against the size the server reports, and name the ones to re-download.

    Local size alone cannot decide this — shards legitimately range from ~1 MB to ~3 GB — and reading the archive conflates a truncated 
    download with a cloud-storage file that has not hydrated. The server's Content-Length is the only authority.
    """
    cache = Path(args.cache or Path(args.root) / "signverse_shards")
    shards = sorted(plan["shards"])
    if args.limit: shards = shards[: args.limit]
    bad, missing, ok = [], [], 0
    for i, shard in enumerate(shards, 1):
        dest = cache / shard
        if not dest.exists(): missing.append(shard); continue
        req = urllib.request.Request(f"{HF_BASE}/dataset/{shard}", method="HEAD")
        try:
            with urllib.request.urlopen(req) as r: expected = int(r.headers["Content-Length"])
        except Exception as e:
            print(f"verify | [{i}/{len(shards)}] {shard}: HEAD failed ({e}) — skipped", flush=True); continue
        got = dest.stat().st_size
        if got == expected: ok += 1
        else:
            bad.append((shard, got, expected))
            print(f"verify | [{i}/{len(shards)}] {shard}: {got} bytes, server says {expected} — TRUNCATED", flush=True)
    print(f"\nverify | {ok} complete, {len(bad)} truncated, {len(missing)} not downloaded", flush=True)
    if bad:
        print("verify | delete these and re-run --stage download:", flush=True)
        for shard, _, _ in bad: print(f"    rm {cache / shard}", flush=True)


def _convert_one(
    tar: tarfile.TarFile, vid: str, lang_root: Path, tmp_dir: Path, tcode: str, subtitle_cfg: dict, min_motion: float
) -> dict | None:
    members = [m for m in tar.getmembers() if m.name.startswith(f"{vid}/")]
    if not any(m.name.startswith(f"{vid}/npz/") for m in members): return None
    tar.extractall(tmp_dir, members=members, filter="data")
    stats = convert_video(tmp_dir / vid / "npz", lang_root / "poses" / f"{vid}.npy", min_motion)

    # Harvest the SINGLE best shard-bundled caption → the canonical `<vid>.<tcode>.vtt` that `--stage subs` also writes, so both paths agree 
    # on exactly 1 caption/video. lang_prefix=tcode restricts `best_subtitle` (the loader's rule) to TARGET-language tracks: a non-en target 
    # never harvests a mislabelled English track. None → caption_source "none", and `--stage subs` gap-fills it.
    best = best_subtitle(tmp_dir / vid / "captions", vid, subtitle_cfg, lang_prefix=tcode) if (tmp_dir / vid / "captions").exists() else None
    if best is not None:
        subs_dir = lang_root / "subs"
        subs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best, subs_dir / f"{vid}.{tcode}.vtt")
        stats["caption_source"] = "shard"
    else: stats["caption_source"] = "none"
    shutil.rmtree(tmp_dir / vid, ignore_errors=True)
    return stats


def _backfill_meta(vid: str, lang_root: Path, meta: dict[str, dict]) -> None:
    # A run that crashed before save leaves a .npy with no meta row. Recompute duration from the .npy frame count
    # at the corpus's fixed 24 fps (cheap: mmap header only) so video_meta stays complete across resumes.
    if vid in meta: return
    npy = lang_root / "poses" / f"{vid}.npy"
    if not npy.exists(): return
    frames = int(np.load(npy, mmap_mode="r").shape[0])
    # caption_source "none", not blank: a blank is not in human_only_exclude_sources, so a crash-recovered row
    # would silently pass as a human caption on the test split. "none" is what `--stage subs` gap-fills.
    meta[vid] = {"video_id": vid, "duration_s": f"{frames / SIGNVERSE_DEFAULT_FPS:.3f}", "width": "", "height": "", "caption_source": "none"}


def stage_convert(args, plan: dict) -> None:
    cache, root = Path(args.cache), Path(args.root)
    data_cfg = load_yaml(args.data_config)
    subtitle_cfg = data_cfg.get("subtitles", {}) or {}
    min_motion = float(data_cfg["poses"]["min_extra_person_motion"])  # the masked-run motion threshold
    targets = _lang_targets(data_cfg, {v["language"] for v in plan["videos"].values()})
    per_lang_meta: dict[str, dict[str, dict]] = {}
    report = {"converted": 0, "skipped_existing": 0, "empty_heavy": [],
              "npz_missing": [], "npz_corrupt": [], "not_downloaded": 0, "no_caption": 0}

    def _meta_for(lang: str) -> dict:
        if lang not in per_lang_meta: per_lang_meta[lang] = load_video_meta(root / lang / META_FILENAME)
        return per_lang_meta[lang]

    for shard in sorted(plan["shards"]):
        tar_path = cache / shard
        if not tar_path.exists():
            # Shard tar absent — a --limit smoke run, an interrupted download, or `--delete-tars` freed it. Videos
            # whose pose IS on disk were converted from that tar: count them as already present (backfilling meta a
            # crash lost), NOT "not downloaded", else reconciliation advises a needless re-download.
            touched = set()
            for vid in plan["shards"][shard]:
                lang = plan["videos"][vid]["language"]
                if (root / lang / "poses" / f"{vid}.npy").exists():
                    _backfill_meta(vid, root / lang, _meta_for(lang)); touched.add(lang)
                    report["skipped_existing"] += 1
                else: report["not_downloaded"] += 1
            for lang in touched: save_video_meta(root / lang / META_FILENAME, per_lang_meta[lang])
            continue

        in_shard = plan["shards"][shard]
        wanted = [v for v in in_shard if args.overwrite or not (root / plan["videos"][v]["language"] / "poses" / f"{v}.npy").exists()]
        # Even for fully-converted shards: backfill meta rows a prior crash lost, then honor --delete-tars ("re-run convert --delete-tars 
        # to free disk" flow must delete these too). A backfilled row carries the duration only: frame size and person counts live in the 
        # shard, not the `.npy`, so `--stage person-counts` fills those and it needs shard tars (handless_runs it fills from the `.npy`).
        touched_langs = set()
        for vid in in_shard:
            if vid not in wanted:
                lang = plan["videos"][vid]["language"]
                _backfill_meta(vid, root / lang, _meta_for(lang))
                touched_langs.add(lang)
        report["skipped_existing"] += len(in_shard) - len(wanted)

        if wanted:
            print(f"convert | {shard}: {len(wanted)} video(s)", flush=True)
            tmp_dir = cache / "_extract"
            # A truncated shard raises deep inside tarfile with no clue which file is bad. Name it and say what to do: the archive is 
            # unusable, so the only fix is to delete and re-download it. getmembers(), not open(): a truncated tar opens fine (its 1st 
            # header is intact) and only fails when the index is walked to the end.
            try:
                with tarfile.open(tar_path) as _probe: _probe.getmembers()
            except tarfile.ReadError as e: raise SystemExit(
                f"convert | {tar_path} is corrupt or truncated ({e}). This is a partial download, not a data problem. "
                f"Delete it and re-run the download stage:\n    rm {tar_path}"
            ) from e
            with tarfile.open(tar_path) as tar:
                shard_corrupt: list[str] = []
                for vid in wanted:
                    lang = plan["videos"][vid]["language"]
                    lang_root = root / lang
                    # A per-video npz can be corrupt inside a structurally valid tar (CRC failure on the inner zip member). That is 1 
                    # unusable video, not a bad run: record it and continue, so a multi-hour conversion is not lost to a single upstream 
                    # byte error. A shard where MANY videos fail is a bad download instead, and is escalated below.
                    try: stats = _convert_one(tar, vid, lang_root, tmp_dir, targets.get(lang, "en"), subtitle_cfg, min_motion)
                    except (zipfile.BadZipFile, EOFError, ValueError) as e:
                        shutil.rmtree(tmp_dir / vid, ignore_errors=True)
                        shard_corrupt.append(vid)
                        report["npz_corrupt"].append(vid)
                        print(f"convert |   {vid}: unreadable npz in {shard} ({type(e).__name__}: {e}) — skipped", flush=True)
                        continue
                    if stats is None: # Upstream inconsistency, nothing to retry — record it so the final tally reconciles.
                        report["npz_missing"].append(vid)
                        print(f"convert |   {vid}: npz absent from {shard} (upstream index/packaging gap) — skipped", flush=True)
                        continue
                    frames = max(1, int(stats["frames"]))
                    _meta_for(lang)[vid] = {
                        "video_id": vid, "duration_s": f"{stats['duration_s']:.3f}",
                        "width": str(stats["width"] or ""), "height": str(stats["height"] or ""), 
                        "caption_source": stats["caption_source"], "undetected_ratio": stats["empty_frames"] / frames,
                        "masked_runs": stats["masked_runs"], "empty_runs": stats["empty_runs"], "handless_runs": stats["handless_runs"],
                    }
                    touched_langs.add(lang)
                    report["converted"] += 1
                    if stats["caption_source"] == "none": report["no_caption"] += 1
                    if stats["frames"] and stats["empty_frames"] / stats["frames"] > 0.5:
                        report["empty_heavy"].append((vid, round(stats["empty_frames"] / stats["frames"], 2)))

                # Widespread failure in ONE shard is a corrupt download, not upstream data: same remedy as a
                # corrupt tar index, so fail with the same instruction rather than silently dropping the shard.
                if len(shard_corrupt) >= max(3, len(wanted) // 2): raise SystemExit(
                    f"convert | {len(shard_corrupt)}/{len(wanted)} videos in {tar_path} have unreadable npz data. That is a "
                    f"corrupt download, not an upstream gap. Delete it and re-run the download stage:\n    rm {tar_path}"
                )

        # Persist meta AFTER EACH SHARD: a crash then costs 1 shard, not the whole run.
        for lang in touched_langs: save_video_meta(root / lang / META_FILENAME, per_lang_meta[lang])
        if args.delete_tars: tar_path.unlink()

    shutil.rmtree(cache / "_extract", ignore_errors=True)
    for lang, meta in per_lang_meta.items(): print(f"convert | {root / lang / META_FILENAME}: {len(meta)} rows", flush=True)

    # Full reconciliation: every split video lands in exactly 1 bucket, so the counts add up on their own.
    n_split = len(plan["videos"]) + len(plan["missing"])
    accounted = (report["converted"] + report["skipped_existing"] + len(plan["missing"]) + len(report["npz_missing"])
                 + len(report["npz_corrupt"]) + report["not_downloaded"])
    print(f"convert | reconciliation ({n_split} split videos = {accounted} accounted): converted {report['converted']} + already present "
          f"{report['skipped_existing']} + not uploaded upstream {len(plan['missing'])} + npz absent from shard {len(report['npz_missing'])} + "
          f"unreadable npz {len(report['npz_corrupt'])} + shard not downloaded {report['not_downloaded']}")
    if report["not_downloaded"]: print(f"convert | {report['not_downloaded']} video(s) are in shards not present in {cache} — "
                                       f"run `--stage download` (or `--stage all` without --limit) to fetch them.")
    print(f"convert | {report['converted'] - report['no_caption']} video(s) captioned from their shard track (caption_source=shard); "
          f"{report['no_caption']} without a usable shard caption → `--stage subs` gap-fills from subtitles tar (else the loader drops them)")
    if report["empty_heavy"]: print(f"convert | {len(report['empty_heavy'])} converted video(s) have >50% frames with no detected body.")
    if report["npz_missing"]: print("convert | npz absent from shard (upstream gap, unrecoverable here): " + ", ".join(report["npz_missing"]))
    if report["npz_corrupt"]: print(
        f"convert | {len(report['npz_corrupt'])} videos had unreadable npz data and were skipped: " + ", ".join(report["npz_corrupt"]) +
        "\nconvert | these are excluded from video_meta.csv. Re-download affected shards and re-run --stage convert to recover them."
    )


def stage_person_counts(args, plan: dict) -> None:
    """Fill undetected_ratio, masked and empty runs and frame size in video_meta.csv from source archives, and handless_runs from pose files.

    Reads the counts and keypoints 1 video at a time. Pose files and caption_source are preserved. Complete rows are skipped, except under
    `--stage person-counts --overwrite`, which measures every video again (after a change of the run rules). The loader cuts nothing for a 
    run column left blank, and quarantines nothing for a blank handless_runs.
    """
    cache, root = Path(args.cache), Path(args.root)
    min_motion = float(load_yaml(args.data_config)["poses"]["min_extra_person_motion"])  # as in `--stage convert`
    remeasure = bool(args.overwrite) and args.stage == "person-counts"   # `--stage all` has just measured them in convert
    per_lang_meta: dict[str, dict[str, dict]] = {}
    filled = skipped = 0
    missing_shards: list[str] = []

    def _meta_for(lang: str) -> dict:
        if lang not in per_lang_meta: per_lang_meta[lang] = load_video_meta(root / lang / META_FILENAME)
        return per_lang_meta[lang]

    def _needs(vid: str) -> bool:
        lang = plan["videos"][vid]["language"]
        row = _meta_for(lang).get(vid)
        if row is None or not (root / lang / "poses" / f"{vid}.npy").exists(): return False
        return remeasure or any(row.get(key) is None for key in ("undetected_ratio", "masked_runs", "empty_runs", "width", "height"))

    # handless_runs reads the kept body, which the .npy holds (`handless_runs`, as in `--stage convert`): no archive needed.
    handless_filled = 0
    for lang in sorted({v["language"] for v in plan["videos"].values()}):
        if filled_here := fill_handless_runs(root / lang, _meta_for(lang), remeasure):
            save_video_meta(root / lang / META_FILENAME, per_lang_meta[lang])
            handless_filled += filled_here
    print(f"person-counts | handless_runs filled {handless_filled} from the pose files", flush=True)

    for shard in sorted(plan["shards"]):
        wanted = {v for v in plan["shards"][shard] if _needs(v)}
        skipped += sum(1 for v in plan["shards"][shard] if v in plan["videos"] and not _needs(v)
                         and (root / plan["videos"][v]["language"] / "poses" / f"{v}.npy").exists())
        if not wanted: continue
        tar_path = cache / shard
        if not tar_path.exists():
            missing_shards.append(shard)
            continue

        with tarfile.open(tar_path, "r|") as tar:   # stream: 1 sequential pass, nothing seeks back
            for member in tar:
                vid = member.name.split("/")[0]
                if vid not in wanted or not member.name.endswith("/npz/poses.npz"): continue
                z = np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=True)
                stats = sidecar_stats_from_npz(z, min_motion)
                frames = stats["frames"]
                lang = plan["videos"][vid]["language"]
                row = _meta_for(lang)[vid]
                row.update({
                    "undetected_ratio": 1.0 - stats["detected"] / frames if frames else 1.0,
                    "masked_runs": stats["masked_runs"], "empty_runs": stats["empty_runs"],
                })
                # Only when the shard knows it: a blank stays blank rather than becoming a wrong 0, 
                # because Moryossef segmenter's aspect correction reads these.
                if stats["width"] and stats["height"]: row.update({"width": stats["width"], "height": stats["height"]})
                filled += 1
                wanted.discard(vid)

        drop_from_page_cache(tar_path)
        for lang in per_lang_meta: save_video_meta(root / lang / META_FILENAME, per_lang_meta[lang])
        print(f"person-counts | {shard}: filled {len(plan['shards'][shard]) - len(wanted)}, "
              f"per-frame layout left unknown {len(wanted)}", flush=True)

    print(f"person-counts | filled {filled}, already known {skipped}, shards not in cache {len(missing_shards)}", flush=True)
    # Name what is still blank. The loader's visual cuts read the run columns and the segmenter's aspect correction reads
    # the frame size. Report missing values so the user can distinguish unmeasured videos from measured absences.
    for lang, meta in sorted(per_lang_meta.items()):
        blank = lambda *keys: sum(1 for row in meta.values() if any(row.get(k) is None for k in keys))
        print(
            f"person-counts | {lang}: rows still unknown: undetected_ratio {blank('undetected_ratio')}/{len(meta)}, "
            f"frame size {blank('width', 'height')}/{len(meta)}, masked_runs {blank('masked_runs')}/{len(meta)}, "
            f"empty_runs {blank('empty_runs')}/{len(meta)}, handless_runs {blank('handless_runs')}/{len(meta)}", flush=True
        )
    if missing_shards: print(
        f"person-counts | {len(missing_shards)} shard(s) absent from {cache}; their videos stay UNKNOWN and data/loader.py cuts nothing "
        f"for an unknown run column. Download them (`--stage download`) or copy a video_meta.csv measured elsewhere.", flush=True
    )


def stage_subs(args, plan: dict) -> None:
    """GAP-FILL captions `--stage convert` could not harvest from shard tracks (module docstring: CAPTIONS).

    A video already carrying `<vid>.<target>.vtt` in subs/ is left untouched — convert wrote that same canonical name from its shard track. 
    Only GAPS get one from the tar. Provenance (human | mt) → video_meta.csv `caption_source`."""
    root = Path(args.root)
    data_cfg = load_yaml(args.data_config)
    lang_target = _lang_targets(data_cfg, {info["language"] for info in plan["videos"].values()})

    # A GAP = a video with a POSE but no `<vid>.<target>.vtt`. Gating on the .npy avoids orphan captions for
    # unconverted videos and lets the "no gaps → skip the tar" fast path fire on partial runs. Filesystem-based, so
    # deleting subs/ & re-running re-fills from the tar even when video_meta still says shard.
    meta_by_lang: dict[str, dict] = {}
    def _meta_for(lang: str) -> dict:
        if lang not in meta_by_lang: meta_by_lang[lang] = load_video_meta(root / lang / META_FILENAME)
        return meta_by_lang[lang]

    gaps: dict[str, str] = {}      # vid -> language: a pose but no caption file, so the tar supplies one
    unknown: dict[str, str] = {}   # vid -> language: a caption file whose provenance the meta row does not carry
    for vid, info in plan["videos"].items():
        lang = info["language"]
        if not (root / lang / "poses" / f"{vid}.npy").exists(): continue
        if not (root / lang / "subs" / f"{vid}.{lang_target[lang]}.vtt").exists(): gaps[vid] = lang
        elif (_meta_for(lang).get(vid, {}).get("caption_source") or "none") == "none": unknown[vid] = lang
    if not gaps and not unknown:
        print("subs | every converted video already has a `<vid>.<target>.vtt` (convert harvested the shard tracks) "
              "— no gaps, subtitles tar not needed.", flush=True)
        return
    # `caption_source` decides which videos the human-only test rule excludes (subtitles.human_only_splits), and a row rebuilt from `.npy` 
    # alone carries "none". Recover it from tar by BYTES: caption file identical to NLLB track is machine translation whatever row says.
    if unknown: print(f"subs | {len(unknown)} caption file(s) with no provenance in {META_FILENAME}; "
                      f"re-deriving `caption_source` from the tar", flush=True)

    # Reuse an already-downloaded tar (--subs-tar, else root/) before fetching 700 MB.
    candidates = [Path(args.subs_tar)] if getattr(args, "subs_tar", None) else [root / SUBTITLES_TAR]
    tar_path = next((p for p in candidates if p.exists()), None)
    if tar_path is None:
        tar_path = root / SUBTITLES_TAR
        print(f"subs | {len(gaps)} caption gap(s); fetching {SUBTITLES_TAR} (~700 MB, once) → {tar_path}", flush=True)
        _download(f"{HF_BASE}/{SUBTITLES_TAR}", tar_path)
    else: print(f"subs | {len(gaps)} caption gap(s); using existing {tar_path}", flush=True)

    buf: dict[str, dict[str, bytes]] = {}  # vid -> {caption filename: bytes}, only for gap videos
    with tarfile.open(tar_path) as tar:
        for m in tar:  # stream: subtitles/<vid>/<file>
            if not m.isfile(): continue
            parts = m.name.split("/")
            if len(parts) != 3 or parts[0] != "subtitles" or (parts[1] not in gaps and parts[1] not in unknown): continue
            if not _is_caption_file(parts[2]): continue  # buffer all originals; _pick_caption chooses by target
            f = tar.extractfile(m)
            if f is not None: buf.setdefault(parts[1], {})[parts[2]] = f.read()

    source_by_lang: dict[str, dict[str, str]] = {}  # lang -> {vid: human|mt|none}
    for vid, lang in unknown.items(): # File on disk is the reference, so match on its bytes rather than re-picking a track.
        on_disk = (root / lang / "subs" / f"{vid}.{lang_target[lang]}.vtt").read_bytes()
        match = next((name for name, content in (buf.get(vid) or {}).items() if content == on_disk), None)
        # The only 2 writers are this stage (tar bytes, copied verbatim) and convert (a shard track), so a file
        # that matches no tar track came from the shard.
        source_by_lang.setdefault(lang, {})[vid] = ("shard" if match is None else "mt" if match.endswith(".nllb.vtt") else "human")

    for vid, lang in gaps.items():
        tcode = lang_target[lang]
        picked = _pick_caption(buf.get(vid, {}), tcode)
        if picked is not None:
            source, content = picked  # human | mt
            subs_dir = root / lang / "subs"
            subs_dir.mkdir(parents=True, exist_ok=True)
            (subs_dir / f"{vid}.{tcode}.vtt").write_bytes(content)
        else: source = "none"  # no target-language caption anywhere → the loader drops this video
        source_by_lang.setdefault(lang, {})[vid] = source

    total_mt = 0
    for lang, by_vid in source_by_lang.items():
        meta = load_video_meta(root / lang / META_FILENAME)
        for vid, source in by_vid.items():
            # Gaps are pose-gated so the .npy exists: give the row a duration_s, because load_video_meta DROPS rows
            # without one — erasing provenance and letting an `mt` caption slip past human_only_splits.
            _backfill_meta(vid, root / lang, meta)
            meta.setdefault(vid, {"video_id": vid})["caption_source"] = source
        save_video_meta(root / lang / META_FILENAME, meta)
        c = {s: sum(v == s for v in by_vid.values()) for s in ("human", "mt", "none", "shard")}
        total_mt += c["mt"]
        print(f"subs | {lang}: filled {c['human']} human + {c['mt']} NLLB-MT + {c['shard']} from the shard; {c['none']} still "
              f"caption-less (dropped by loader). Provenance → {root / lang / META_FILENAME} caption_source.", flush=True)
    if total_mt: print("subs | NLLB-machine-translated captions are noisy SLT targets — the loader scores the TEST split "
                       "against human references only (subtitles.human_only_splits + caption_source in video_meta.csv).")


def fetch_youtube_meta(video_ids: list[str], ytdlp_format: str = YTDLP_FORMAT, workers: int = 4, chunk: int = 25) -> dict[str, dict]:
    """yt-dlp METADATA-ONLY fetch -> {video_id: {duration_s, width, height}}. No video download.

    Raw videos are never needed (too heavy for large languages, e.g. ase): duration — the only fps-calibration input — comes from YouTube 
    metadata in WHOLE SECONDS. width/height resolve via `ytdlp_format`; pass the SAME --format the videos were downloaded with, and treat 
    them as advisory (yt-dlp may resolve fewer formats than a browser: JS-runtime/PO-token limits). Removed/private videos are skipped → 
    config pose_fps fallback with a loud loader warning.
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
    """Build/refresh <lang_root>/video_meta.csv. CLI: `--stage meta [--format SEL]`, lang_root = <root>/<lang>.

    Existing sidecar rows are kept; yt-dlp metadata is fetched only for pose ids still missing (no video download). A single constant fps 
    CANNOT replace this: own extractions vary in fps per video, and a wrong rate shifts every caption timestamp against the poses. A blank 
    handless_runs is filled from pose files (`fill_handless_runs`); masked_runs and empty_runs need SignVerse archives, so they stay blank.
    """
    lang_root = Path(lang_root)
    out_path = lang_root / META_FILENAME
    meta = load_video_meta(out_path)
    pose_ids = {base_video_id(p) for p in (lang_root / "poses").glob("*.npy")}

    missing = sorted(pose_ids - set(meta))
    if missing:
        print(f"fetching {len(missing)} videos' metadata from YouTube (yt-dlp, no download)...")
        meta.update(fetch_youtube_meta(missing, ytdlp_format=ytdlp_format))
    if handless_filled := fill_handless_runs(lang_root, meta): print(f"handless_runs filled {handless_filled} from the pose files")
    save_video_meta(out_path, meta)
    still_missing = sorted(pose_ids - set(meta))
    print(f"{len(meta)} videos -> {out_path}; pose ids still missing: {len(still_missing)}"
          + (f" (will fall back to config pose_fps): {still_missing[:5]}..." if still_missing else ""))
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignVerse-2M → repo language layout (poses/ + subs/ + video_meta.csv)")
    parser.add_argument("--stage", default="all", choices=["plan", "download", "verify", "convert", "person-counts", "subs", "meta", "all"])
    parser.add_argument("--languages", nargs="+", default=["ase", "asf", "bfi"])
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--data-config", default="configs/data.yaml", help="reads languages[lang].target_lang for the caption language")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="shard tar cache dir")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="language roots parent (root/<lang>/poses etc.)")
    parser.add_argument("--limit", type=int, default=0, help="stop after N shards (smoke runs)")
    parser.add_argument("--size", action="store_true", help="plan: HEAD-probe undownloaded shards for a size estimate")
    parser.add_argument("--subs-tar", default=None, help="path to an already-downloaded signverse_subtitles_with_english.tar")
    parser.add_argument("--delete-tars", action="store_true",
                        help="delete each shard tar after conversion. Shards are MIXED-language, so include every "
                             "language you will ever want in ONE run (e.g. --languages asf bfi) — a later run for a "
                             "language whose videos sat in an already-deleted shard would have to re-download it.")
    parser.add_argument("--overwrite", action="store_true",
                        help="convert: re-convert videos whose .npy already exists; person-counts: measure every video's run columns again")
    parser.add_argument("--format", default=YTDLP_FORMAT, help="meta: yt-dlp format selector; use the one the videos were downloaded with")
    args = parser.parse_args()

    if args.stage == "meta":   # reads <root>/<lang>/poses only: no split CSV, no shard index
        for lang in args.languages: build_video_meta(Path(args.root) / lang, ytdlp_format=args.format)
        sys.exit(0)
    split_csv = Path(args.split_csv)
    if not split_csv.exists(): sys.exit(f"split CSV not found: {split_csv}")
    plan = load_plan(split_csv, Path(args.root), list(args.languages))
    if plan["missing"]: print(f"prepare | {len(plan['missing'])} video(s) not yet uploaded upstream (no failure markers) — skipped")
    if args.stage in ("plan",): stage_plan(args, plan)
    if args.stage in ("download", "all"): stage_download(args, plan)
    if args.stage == "verify": stage_verify(args, plan)
    if args.stage in ("convert", "all"): stage_convert(args, plan)
    if args.stage in ("person-counts", "all"): stage_person_counts(args, plan)
    if args.stage in ("subs", "all"): stage_subs(args, plan)
