import numpy as np
from . import *


def normalize_keypoints(keypoints: np.ndarray, width: int | None = None, height: int | None = None) -> np.ndarray:
    '''
    CoSign keypoints normalize: Group-specific centralization (subtract root per group).
    
    Groups: Body (root: mid-shoulders), Left hand (wrist), Right hand (wrist), Face (nose).
    
    Args:
        keypoints (np.ndarray): (frames, 133, 3) x,y,confidence
        width (int): Image width for global scaling.
        height (int): Image height for global scaling.

    Returns:
        np.ndarray: Normalized keypoints.
    '''
    if keypoints.shape[1] != 133: 
        raise ValueError(f'Invalid pose shape: {keypoints.shape}, expected (frames, 133, 3)')
    
    # Ensure float and finite to avoid overflow/underflow in linalg ops
    keypoints = np.asarray(keypoints, dtype=np.float32, order='C')
    np.nan_to_num(keypoints, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

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

    # Group lengths only for x,y coordinates using np.hypot (stable √(x² + y²) to avoid overflow)
    # v := is a walrus operator, it assigns the value of a - b to v and returns it
    _length = lambda a, b, eps=1e-6: (v := a - b, np.hypot(v[:, 0], v[:, 1])[:, None, None] + eps)[-1] 
    shoulder_length = _length(body_kpts[:, 3, :2], body_kpts[:, 4, :2])
    left_hand_length = _length(left_hand_kpts[:, 0, :2], left_hand_kpts[:, 9, :2])
    right_hand_length = _length(right_hand_kpts[:, 0, :2], right_hand_kpts[:, 9, :2])
    mouth_length = _length(mouth_kpts[:, 0, :2], mouth_kpts[:, 4, :2])
    face_length = _length(face_kpts[:, 0, :2], face_kpts[:, 8, :2])
    
    # Normalize keypoints
    norm_kpts = np.zeros((keypoints.shape[0], 77, 3), dtype=np.float32)
    norm_kpts[:, 0:9, :2] = (body_kpts[:, :, :2] - root_body[:, None]) / shoulder_length * 3            # 0..8 (9 points)
    norm_kpts[:, 9:30, :2] = (left_hand_kpts[:, :, :2] - root_left[:, None]) / left_hand_length * 2     # 9..29 (21 points)
    norm_kpts[:, 30:51, :2] = (right_hand_kpts[:, :, :2] - root_right[:, None]) / right_hand_length * 2 # 30..50 (21 points)
    norm_kpts[:, 51:59, :2] = (mouth_kpts[:, :, :2] - root_mouth[:, None]) / mouth_length               # 51..58 (8 points)
    norm_kpts[:, 59:77, :2] = (face_kpts[:, :, :2] - root_face[:, None]) / face_length * 2              # 59..76 (18 points)
    
    # Copy confidence values without modification
    norm_kpts[:, 0:9, 2] = body_kpts[:, :, 2]
    norm_kpts[:, 9:30, 2] = left_hand_kpts[:, :, 2]
    norm_kpts[:, 30:51, 2] = right_hand_kpts[:, :, 2]
    norm_kpts[:, 51:59, 2] = mouth_kpts[:, :, 2]
    norm_kpts[:, 59:77, 2] = face_kpts[:, :, 2]

    # Global scaling
    if width is not None and height is not None:
        norm_kpts[:, :, 0] /= float(width)   # x
        norm_kpts[:, :, 1] /= float(height)  # y
    return norm_kpts


def threshold_confidence(keypoints: np.ndarray) -> np.ndarray:
    '''
    Set low-conf keypoints to 0 (x,y=0, conf=0 if < threshold)
    Input/Output: keypoints (frames, kpts, 3)
    '''
    low_conf_mask = keypoints[:, :, 2] < CONF_THRESHOLD
    keypoints[low_conf_mask] = 0  # Zero out x,y,conf
    # print(f'Applied confidence threshold: {np.sum(low_conf_mask)} keypoints zeroed')
    return keypoints


def subsample_frames(keypoints: np.ndarray, target_fps: float) -> np.ndarray:
    '''
    Subsample to lower FPS if needed (e.g., from 12.5 to 5 fps)
    Input: keypoints (frames, kpts, 3), target_fps (float)
    Output: subsampled keypoints
    '''
    if target_fps >= FPS: return keypoints
    step = int(FPS / target_fps)
    subsampled = keypoints[::step, :, :]
    print(f'Subsampled from {keypoints.shape[0]} to {subsampled.shape[0]} frames (target FPS: {target_fps})')
    return subsampled


if __name__ == '__main__':
    dummy_keypoints = np.random.rand(100, 133, 3)  # frames, kpts, (x,y,conf)
    dummy_keypoints[:, :, 2] = np.random.uniform(0.4, 1.0, (100, 133))  # Random conf
    normalized = normalize_keypoints(dummy_keypoints)
    thresholded = threshold_confidence(normalized)
    subsampled = subsample_frames(thresholded, target_fps=6.0)
