import numpy as np
from . import *


def normalize_keypoints_cosign(
    keypoints: np.ndarray, width: int | None = None, height: int | None = None,
    min_scale_px: float = 2.0, clip: float = 10.0, conf_threshold: float | None = None,
) -> np.ndarray:
    '''
    CoSign group normalization (subtract root, divide by group reference length), made
    numerically safe for in-the-wild DWPose output.

    Groups: Body (root: mid-shoulders / shoulder width), Left/Right hand (wrist / wrist→MCP9),
    Mouth (center / lip width), Face (nose / contour width).

    CoSign was designed for studio corpora (Phoenix/CSL) with reliable detections. On YouTube
    video, DWPose hand detection routinely degenerates: the wrist and MCP keypoints collapse to
    (near-)identical garbage coordinates, the reference length → ~0, and the division amplifies
    by 1e6+ (measured: hand features up to 2.6e7, velocities 6.6e8 — enough to stop any model
    from learning past the label prior). Three guards, applied in the right order:

    1. Confidence thresholding FIRST (raw points with conf < conf_threshold are invalid and must
       not define roots/scales — the old code normalized with raw coords and thresholded after).
    2. A frame's group is valid only if its root+reference points are valid AND the reference
       length >= `min_scale_px` (pixels). Invalid group-frames are zeroed entirely — same
       "unreliable → 0" semantics the confidence threshold already establishes.
    3. Normalized coords are clipped to ±`clip` (centered coords are O(1–6) by construction;
       anything bigger is detection garbage, not signal).

    Args:
        keypoints: (frames, 133, 3) x, y, confidence — raw pixel coordinates.
    Returns:
        (frames, 77, 3) normalized keypoints; invalid points/groups zeroed.
    '''
    if keypoints.shape[1] != 133: raise ValueError(f'Invalid pose shape: {keypoints.shape}, expected (frames, 133, 3)')
    # Ensure float and finite to avoid overflow/underflow in linalg ops
    keypoints = np.asarray(keypoints, dtype=np.float32, order='C').copy()
    np.nan_to_num(keypoints, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Guard 1: invalidate low-confidence points BEFORE anything reads their coordinates. 
    # The threshold is per-dataset (PoseIndex.conf_threshold from data.yaml).
    invalid_pt = keypoints[:, :, 2] < conf_threshold  # (T, 133)
    keypoints[invalid_pt] = 0.0                       # Zero out x,y,conf

    # Split into groups
    body_kpts = keypoints[:, BODY_IDS]
    left_hand_kpts = keypoints[:, LEFT_HAND_IDS]
    right_hand_kpts = keypoints[:, RIGHT_HAND_IDS]
    mouth_kpts = keypoints[:, MOUTH_IDS]
    face_kpts = keypoints[:, FACE_IDS]

    # Root keypoints
    root_body = (body_kpts[:, 3, :2] + body_kpts[:, 4, :2]) / 2  # Mid-shoulder
    root_left = left_hand_kpts[:, 0, :2]                         # Wrist index 0 in hand
    root_right = right_hand_kpts[:, 0, :2]                       # Wrist index 0 in hand
    root_mouth = mouth_kpts[:, 4, :2]                            # Mouth center
    root_face = face_kpts[:, -1, :2]                             # Nose as root

    # Group reference lengths from x,y using np.hypot (stable sqrt(x^2+y^2))
    _length = lambda a, b, eps=1e-6: (v := a - b, np.hypot(v[:, 0], v[:, 1])[:, None, None] + eps)[-1]
    shoulder_length = _length(body_kpts[:, 3, :2], body_kpts[:, 4, :2])
    left_hand_length = _length(left_hand_kpts[:, 0, :2], left_hand_kpts[:, 9, :2])
    right_hand_length = _length(right_hand_kpts[:, 0, :2], right_hand_kpts[:, 9, :2])
    mouth_length = _length(mouth_kpts[:, 0, :2], mouth_kpts[:, 4, :2])
    face_length = _length(face_kpts[:, 0, :2], face_kpts[:, 8, :2])

    # Guard 2: a group-frame is usable only when its root/reference points are confident AND the
    # reference length is a plausible pixel distance (degenerate length = collapsed detection).
    def _group_valid(length: np.ndarray, *ref_valid: np.ndarray) -> np.ndarray:
        ok = length[:, 0, 0] >= float(min_scale_px)           # (T,)
        for v in ref_valid: ok = ok & v
        return ok[:, None, None]                              # (T, 1, 1) broadcast over points/coords

    pt_ok = keypoints[:, :, 2] > 0                            # post-threshold validity per point
    body_ok = _group_valid(shoulder_length, pt_ok[:, BODY_IDS[3]], pt_ok[:, BODY_IDS[4]])
    left_ok = _group_valid(left_hand_length, pt_ok[:, LEFT_HAND_IDS[0]], pt_ok[:, LEFT_HAND_IDS[9]])
    right_ok = _group_valid(right_hand_length, pt_ok[:, RIGHT_HAND_IDS[0]], pt_ok[:, RIGHT_HAND_IDS[9]])
    mouth_ok = _group_valid(mouth_length, pt_ok[:, MOUTH_IDS[0]], pt_ok[:, MOUTH_IDS[4]])
    face_ok = _group_valid(face_length, pt_ok[:, FACE_IDS[0]], pt_ok[:, FACE_IDS[8]], pt_ok[:, FACE_IDS[-1]])

    norm_kpts = np.zeros((keypoints.shape[0], 77, 3), dtype=np.float32)
    norm_kpts[:, 0:9, :2] = (body_kpts[:, :, :2] - root_body[:, None]) / shoulder_length * 3 * body_ok    # 0..8 (9 points)
    norm_kpts[:, 9:30, :2] = (left_hand_kpts[:, :, :2] - root_left[:, None]) / left_hand_length * 2 * left_ok
    norm_kpts[:, 30:51, :2] = (right_hand_kpts[:, :, :2] - root_right[:, None]) / right_hand_length * 2 * right_ok
    norm_kpts[:, 51:59, :2] = (mouth_kpts[:, :, :2] - root_mouth[:, None]) / mouth_length * mouth_ok
    norm_kpts[:, 59:77, :2] = (face_kpts[:, :, :2] - root_face[:, None]) / face_length * 2 * face_ok

    # Confidence channel: zero for invalid points and for frames whose group was dropped.
    norm_kpts[:, 0:9, 2] = body_kpts[:, :, 2] * body_ok[:, :, 0]
    norm_kpts[:, 9:30, 2] = left_hand_kpts[:, :, 2] * left_ok[:, :, 0]
    norm_kpts[:, 30:51, 2] = right_hand_kpts[:, :, 2] * right_ok[:, :, 0]
    norm_kpts[:, 51:59, 2] = mouth_kpts[:, :, 2] * mouth_ok[:, :, 0]
    norm_kpts[:, 59:77, 2] = face_kpts[:, :, 2] * face_ok[:, :, 0]

    # Individually-invalid points inside a valid group: re-zero (centering moved them to -root/len).
    sel_invalid = invalid_pt[:, ALL_SELECTED_IDS]
    norm_kpts[sel_invalid] = 0.0

    # Guard 3: hard cap — centered/scaled coords are O(1–6); beyond ±clip is garbage, not signal.
    np.clip(norm_kpts[:, :, :2], -float(clip), float(clip), out=norm_kpts[:, :, :2])

    # Global scaling
    if width is not None and height is not None:
        norm_kpts[:, :, 0] /= float(width)   # x
        norm_kpts[:, :, 1] /= float(height)  # y
    return norm_kpts


def normalize_keypoints_mska(keypoints: np.ndarray, width: int | None, height: int | None, clip: float = 1.5) -> np.ndarray:
    '''
    MSKA global frame-normalization (github.com/sutwangyan/MSKA, datasets.py:
    S2T_Dataset.augment_preprocess_inputs), the input representation MSKA's DSTA backbone expects.

    Unlike CoSign's per-group root-relative normalization (which is offset-invariant but DESTROYS the
    inter-part spatial layout — where the hands sit relative to face/body), this keeps ALL 133
    COCO-WholeBody keypoints in a single shared frame so DSTA's 79-node body stream can learn that
    layout (a phonological location cue in sign). MSKA pipeline, applied per (x, y, conf):

        x' = x / width ;  y' = (height - y) / height ;  (x', y') -> (·-0.5)/0.5   # -> [-1, 1]
        conf channel: untouched (DSTA feeds it as a 3rd input channel, letting the net down-weight
                      unreliable points — so we do NOT confidence-threshold-zero like CoSign).

    One deviation from MSKA, for our noisier synthetic poses: a hard clip of the normalized x,y to
    ±`clip`. MSKA's studio HRNet keypoints are clean; ours carry occasional detection-failure outliers
    (measured x down to -719 px in a 210-wide frame). Confident points map well inside [-1, 1] (measured
    p1..p99 ~ [-0.7, 0.7]); `clip`=1.5 leaves a margin for legit frame-edge points and neutralises only
    garbage. (Mirrors the Guard-3 clip rationale in `normalize_keypoints_cosign`.)

    Args:
        keypoints: (frames, 133, 3) x, y, confidence — RAW pixel coordinates.
        width, height: the pixel frame the coordinates live in (PHOENIX synth: manifest src_meta
                       src_w=210, src_h=260, carried on PoseIndex.width/height).
    Returns:
        (frames, 133, 3) MSKA-normalized keypoints (x, y in [-clip, clip]; conf unchanged).
    '''
    if keypoints.shape[1:] != (133, 3): raise ValueError(f'Invalid pose shape: {keypoints.shape}, expected (frames, 133, 3)')
    if not width or not height: raise ValueError(
        'normalize_keypoints_mska needs the pixel frame width/height (PoseIndex.width/height). '
        'For synth corpora these come from manifest.json src_meta; set them on the language entry or rebuild the manifest.'
    )
    kp = np.asarray(keypoints, dtype=np.float32, order='C').copy()
    np.nan_to_num(kp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    x = kp[:, :, 0] / float(width)
    y = (float(height) - kp[:, :, 1]) / float(height)
    kp[:, :, 0] = (x - 0.5) / 0.5
    kp[:, :, 1] = (y - 0.5) / 0.5
    np.clip(kp[:, :, :2], -float(clip), float(clip), out=kp[:, :, :2])
    return kp


if __name__ == '__main__':
    dummy_keypoints = np.random.rand(100, 133, 3)  # frames, kpts, (x,y,conf)
    dummy_keypoints[:, :, 2] = np.random.uniform(0.4, 1.0, (100, 133))  # Random conf
    print('cosign 77:', normalize_keypoints_cosign(dummy_keypoints).shape)
    print('mska 133:', normalize_keypoints_mska(dummy_keypoints * 200, width=210, height=260).shape)
