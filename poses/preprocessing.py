import numpy as np
from . import *


def _unisign_crop_scale_body(body, thr):
    """Uni-Sign crop_scale on the BODY part: bbox-normalise (x,y) to [-1,1] and return the body scale that
    every other part is divided by. body: (T, 9, 3) (x,y,conf). Returns (result (T,9,3), scale float)."""
    result = body.copy()
    valid = body[body[..., 2] > thr][:, :2]
    if valid.shape[0] < 4: return np.zeros_like(body), 0.0

    xmin, xmax = float(valid[:, 0].min()), float(valid[:, 0].max())
    ymin, ymax = float(valid[:, 1].min()), float(valid[:, 1].max())
    scale = max(xmax - xmin, ymax - ymin)  # ratio = 1 (Uni-Sign disables the train-time scale jitter here)
    if scale == 0: return np.zeros_like(body), 0.0
    
    xs = (xmin + xmax - scale) / 2.0
    ys = (ymin + ymax - scale) / 2.0
    result[..., 0] = (body[..., 0] - xs) / scale
    result[..., 1] = (body[..., 1] - ys) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2.0
    # Uni-Sign crop_scale clips the WHOLE array — including the confidence channel (RTMPose scores exceed
    # 1.0 on easy joints; upstream the model only ever sees conf <= 1). Clipping xy-only leaves conf > 1.
    np.clip(result, -1.0, 1.0, out=result)
    result[result[..., 2] <= thr] = 0.0
    return result, float(scale)


def normalize_keypoints_unisign(keypoints: np.ndarray, thr: float = 0.3) -> np.ndarray:
    """Uni-Sign pose normalisation (ZechengLi19/Uni-Sign datasets.py load_part_kp + crop_scale).

    Body is bbox-normalised to [-1,1]; hands are re-centred on the wrist (joint 0) and the face on the nose
    (joint 53, last in the selection), then ALL parts are divided by the BODY bbox scale and clipped to
    [-1,1], so a part's size encodes its real scale relative to the body (a phonological cue). Low-confidence
    points (conf <= thr) are zeroed. `thr=0.3` is Uni-Sign's hardcoded value — keep it to match their weights.

    Args:
        keypoints: (frames, 133, 3) x, y, confidence — RAW pixel coordinates (COCO-WholeBody).
    Returns:
        (frames, 69, 3) in part order [body 9 | left 21 | right 21 | face_all 18].
    """
    if keypoints.shape[1:] != (133, 3):
        raise ValueError(f'Invalid pose shape: {keypoints.shape}, expected (frames, 133, 3)')
    kp = np.asarray(keypoints, dtype=np.float32, order='C').copy()
    np.nan_to_num(kp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    body_norm, scale = _unisign_crop_scale_body(kp[:, UNISIGN_BODY_IDX, :].copy(), thr)

    def _recentered_part(idx_list, anchor):
        part = kp[:, idx_list, :].copy()                       # (T, V, 3)
        a = anchor if anchor >= 0 else part.shape[1] + anchor  # support -1 (face nose = last joint)
        part[..., :2] = part[..., :2] - part[:, a:a + 1, :2]   # per-frame re-centre on anchor joint
        if scale == 0: return np.zeros_like(part)

        part[..., :2] = part[..., :2] / scale
        np.clip(part, -1.0, 1.0, out=part)  # full-array clip incl. conf, matching load_part_kp
        part[part[..., 2] <= thr] = 0.0
        return part

    left = _recentered_part(UNISIGN_LEFT_IDX, 0)               # wrist = joint 0
    right = _recentered_part(UNISIGN_RIGHT_IDX, 0)
    face = _recentered_part(UNISIGN_FACE_IDX, -1)              # nose (idx 53) = last selected joint
    return np.concatenate([body_norm, left, right, face], axis=1).astype(np.float32)


if __name__ == '__main__':
    dummy_keypoints = np.random.rand(100, 133, 3)  # frames, kpts, (x,y,conf)
    dummy_keypoints[:, :, 2] = np.random.uniform(0.4, 1.0, (100, 133))  # random conf
    print('unisign 69:', normalize_keypoints_unisign(dummy_keypoints * 200).shape)
