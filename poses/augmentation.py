"""Pose-sequence augmentation — dataset-driven (frame width/height passed in, never global).

Applied to RAW pixel keypoints (T, 133, 3) BEFORE normalization, TRAIN ONLY. Everything here is
SPATIAL and LENGTH-PRESERVING, so per-frame timestamps and BIO labels stay aligned (the streaming /
segmentation contract). TEMPORAL augmentation is handled separately by `moryossef26.apply_fps_aug`,
which rebuilds the timestamps it resamples.

Two augmentation families are exposed through `PoseAugmentor` / `build_pose_augmentor`:
  - **MSKA** rotation: `rotate` reproduces MSKA's `random_move` (github.com/sutwangyan/MSKA
    datasets.py:224 — rotate ±deg around the keypoint centroid; translation/scale were commented out
    there). This is the augmentation MSKA actually trains its DSTA backbone with.
  - **GFSLT-VLP** spatial: `flip_lr` + `affine` are the pose-space analogue of GFSLT-VLP's video
    augmentation (RandomHorizontalFlip + RandomResizedCrop; arXiv 2307.14768). ColorJitter has no pose
    analogue and is dropped.

Width/height come from `PoseIndex.width/height` (per dataset — PHOENIX 210x260, CSL-Daily 512x512,
etc.). When they are unknown, frame-referenced ops (flip / affine / spatial_mask) are skipped and only
the centroid-based rotation runs.
"""
import numpy as np
from scipy.ndimage import zoom
from . import LEFT_HAND_IDS, RIGHT_HAND_IDS, FACE_IDS, MOUTH_IDS


# ── Spatial primitives (parametrised by frame size; operate on x,y only, conf left intact) ─────────────
def rotate(keypoints: np.ndarray, max_angle_deg: float = 15.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """MSKA `random_move`: rotate (x, y) by U(-max_angle_deg, max_angle_deg) around the centroid of the
    non-zero keypoints (github.com/sutwangyan/MSKA/blob/main/datasets.py#L224). Needs no frame size."""
    keypoints = np.asarray(keypoints, dtype=np.float32).copy()
    if keypoints.ndim != 3 or keypoints.shape[-1] < 2: return keypoints
    rng = rng or np.random.default_rng()
    rad = float(rng.uniform(-max_angle_deg, max_angle_deg)) * np.pi / 180.0
    cos_v, sin_v = float(np.cos(rad)), float(np.sin(rad))
    rot = np.array([[cos_v, -sin_v], [sin_v, cos_v]], dtype=np.float32)
    xy = keypoints[..., :2]
    flat = xy.reshape(-1, 2)
    nz = np.any(flat != 0, axis=-1)
    center = flat[nz].mean(axis=0) if nz.any() else np.zeros(2, dtype=np.float32)
    keypoints[..., :2] = (xy - center) @ rot.T + center
    return keypoints


def flip_lr(keypoints: np.ndarray, width: float) -> np.ndarray:
    """Horizontal mirror (x -> width - x) with left/right keypoint-group swap, for COCO-WholeBody 133.
    Produces a valid mirrored signer (left-handed <-> right-handed); the text label is unchanged."""
    if keypoints.ndim != 3 or keypoints.shape[-1] != 3: raise ValueError(f'Expected (T, K, 3), got {keypoints.shape}')
    keypoints = np.asarray(keypoints, dtype=np.float32).copy()
    valid = keypoints[..., 2] > 0
    keypoints[..., 0] = np.where(valid, float(width) - keypoints[..., 0], keypoints[..., 0])

    left_hand = keypoints[:, LEFT_HAND_IDS, :].copy()
    keypoints[:, LEFT_HAND_IDS, :] = keypoints[:, RIGHT_HAND_IDS, :]
    keypoints[:, RIGHT_HAND_IDS, :] = left_hand
    for li, ri in [(1, 2), (5, 6), (7, 8), (9, 10)]:  # eyes / shoulders / elbows / wrists
        tmp = keypoints[:, li, :].copy()
        keypoints[:, li, :] = keypoints[:, ri, :]
        keypoints[:, ri, :] = tmp

    keypoints[:, FACE_IDS] = keypoints[:, FACE_IDS][:, ::-1].copy()   # face contour reverses under mirror
    keypoints[:, MOUTH_IDS] = keypoints[:, MOUTH_IDS][:, ::-1].copy()
    return keypoints


def affine(
    keypoints: np.ndarray, width: float, height: float,
    scale: tuple[float, float] | None = (0.9, 1.1),
    shift: tuple[float, float] | None = (-0.05, 0.05),
    degree: tuple[float, float] | None = None,
    shear: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random affine (scale/shift/rotate/shear) around the frame centre, on conf>0 points only.

    GFSLT-VLP RandomResizedCrop analogue: `scale` ~ zoom, `shift` ~ crop offset (fraction of frame).
    Leave `degree` None when a separate `rotate` is used, to avoid double-rotating.
    """
    if keypoints.ndim != 3 or keypoints.shape[-1] != 3: raise ValueError(f'Expected (T, K, 3), got {keypoints.shape}')
    rng = rng or np.random.default_rng()
    keypoints = np.asarray(keypoints, dtype=np.float32).copy()
    xy = keypoints[..., :2]
    valid = keypoints[..., 2] > 0
    center = np.array([float(width) / 2.0, float(height) / 2.0], dtype=np.float32)

    if scale is not None: # Scale around center
        s = float(rng.uniform(*scale))
        xy[valid] = (xy[valid] - center) * s + center

    if shear is not None: # Shear around center
        sh = float(rng.uniform(*shear))
        sx, sy = (sh, 0.0) if rng.random() < 0.5 else (0.0, sh)
        shear_mat = np.array([[1.0, sx], [sy, 1.0]], dtype=np.float32)
        xy[valid] = (xy[valid] - center) @ shear_mat + center

    if degree is not None: # Rotate around center
        rad = float(rng.uniform(*degree)) * np.pi / 180.0
        cos_v, sin_v = float(np.cos(rad)), float(np.sin(rad))
        xy[valid] = (xy[valid] - center) @ np.array([[cos_v, sin_v], [-sin_v, cos_v]], dtype=np.float32) + center

    if shift is not None: # Shift in pixels (fraction of width/height)
        xy[valid, 0] += float(rng.uniform(*shift)) * float(width)
        xy[valid, 1] += float(rng.uniform(*shift)) * float(height)

    keypoints[..., :2] = xy
    return keypoints


def spatial_mask(
    keypoints: np.ndarray, width: float, height: float,
    size: tuple[float, float] = (0.1, 0.2), rng: np.random.Generator | None = None,
) -> np.ndarray: # Zero a random spatial box (cutout). Masked points become invalid (normalization zeroes them).
    rng = rng or np.random.default_rng()
    keypoints = np.asarray(keypoints, dtype=np.float32).copy()
    box = float(rng.uniform(*size)) * min(float(width), float(height))
    ox, oy = float(rng.uniform(0.0, float(width))), float(rng.uniform(0.0, float(height)))
    mask = ((ox < keypoints[..., 0]) & (keypoints[..., 0] < ox + box)
            & (oy < keypoints[..., 1]) & (keypoints[..., 1] < oy + box))
    keypoints[mask] = 0.0
    return keypoints


# ── Temporal primitives (NOT used by the default augmentor — they change length / alignment) ───────────
def interp1d_(keypoints: np.ndarray, target_len: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Resize along time (axis 0) by spline interpolation. WARNING: changes T -> breaks timestamp/BIO
    alignment; use `moryossef26.apply_fps_aug` (rebuilds timestamps) for temporal augmentation instead."""
    keypoints = np.asarray(keypoints)
    target_len = int(max(1, target_len))
    src_len = int(keypoints.shape[0])
    if src_len <= 1: return np.repeat(keypoints[:1], target_len, axis=0) if src_len else keypoints
    if src_len == target_len: return keypoints
    rng = rng or np.random.default_rng()
    order = 1 if rng.random() < 0.33 else (3 if rng.random() < 0.5 else 0)
    out = zoom(keypoints, zoom=(target_len / src_len, 1.0, 1.0), order=order, mode='nearest', prefilter=(order > 1))
    if out.shape[0] > target_len: out = out[:target_len]
    elif out.shape[0] < target_len: out = np.pad(out, ((0, target_len - out.shape[0]), (0, 0), (0, 0)))
    return out


# ── Orchestration ─────────────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "rotation": {"deg": 15.0, "prob": 0.5},          # MSKA random_move
    "flip": {"prob": 0.0},                           # Horizontal flip
    "affine": {"prob": 0.0, "scale": [0.9, 1.1], "shift": [-0.05, 0.05], "degree": None, "shear": None},
    "spatial_mask": {"prob": 0.0, "size": [0.1, 0.2]},
}
def _pair(v):
    return None if v is None else (float(v[0]), float(v[1]))


class PoseAugmentor:
    """Composes the configured spatial augmentations. Callable: (poses (T,133,3) raw, width, height) ->
    (T,133,3). Length-preserving. width/height may be None (frame-referenced ops are then skipped)."""
    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.rng = rng
        c = {**_DEFAULTS, **(cfg or {})}
        self.rot = {**_DEFAULTS["rotation"], **(c.get("rotation") or {})}
        self.flip = {**_DEFAULTS["flip"], **(c.get("flip") or {})}
        self.aff = {**_DEFAULTS["affine"], **(c.get("affine") or {})}
        self.smask = {**_DEFAULTS["spatial_mask"], **(c.get("spatial_mask") or {})}

    def __call__(self, poses: np.ndarray, width: float | None, height: float | None) -> np.ndarray:
        poses = np.asarray(poses, dtype=np.float32)
        if poses.ndim != 3 or poses.shape[0] == 0: return poses
        has_frame = bool(width) and bool(height)
        if has_frame and float(self.flip.get("prob", 0.0)) > 0 and self.rng.random() < float(self.flip["prob"]):
            poses = flip_lr(poses, width)
        if has_frame and float(self.aff.get("prob", 0.0)) > 0 and self.rng.random() < float(self.aff["prob"]):
            poses = affine(poses, width, height, scale=_pair(self.aff.get("scale")), shift=_pair(self.aff.get("shift")),
                           degree=_pair(self.aff.get("degree")), shear=_pair(self.aff.get("shear")), rng=self.rng)
        if float(self.rot.get("prob", 0.0)) > 0 and self.rng.random() < float(self.rot["prob"]):
            poses = rotate(poses, max_angle_deg=float(self.rot.get("deg", 15.0)), rng=self.rng)
        if has_frame and float(self.smask.get("prob", 0.0)) > 0 and self.rng.random() < float(self.smask["prob"]):
            poses = spatial_mask(poses, width, height, size=_pair(self.smask.get("size")) or (0.1, 0.2), rng=self.rng)
        return poses


def build_pose_augmentor(cfg: dict | None, rng: np.random.Generator | None = None) -> PoseAugmentor | None:
    """Build a `PoseAugmentor` from an `augment:` config block, or None when disabled. TRAIN-ONLY callers
    pass their config + a seeded rng; eval/val pass None (no augmentation)."""
    if not cfg or not bool(cfg.get("enabled", False)): return None
    return PoseAugmentor(cfg, rng or np.random.default_rng())
