from __future__ import annotations
import math, torch


def gfslt_padded_length(length: int, multiple: int = 4, halo: int = 8) -> int:
    # GFSLT-VLP visual padding length: round to multiple of 4 plus 8-frame halos.
    if length <= 0: raise ValueError("GFSLT visual padding requires a non-empty sequence")
    rounded = int(math.ceil(int(length) / float(multiple)) * multiple)
    return rounded + 2 * int(halo)


def pad_visual_sequence_gfslt(
    poses: torch.Tensor, timestamps_s: torch.Tensor,
    bio_labels: torch.Tensor | None = None, pad_label: int | None = None,
    multiple: int = 4, halo: int = 8, halo_is_valid: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Pad a single visual sequence with repeated boundary frames, GFSLT-VLP style.

    GFSLT-VLP pads video by repeating the first frame on the left, repeating the last frame on the right, 
    and rounding the real sequence length to a multiple of 4 before adding the 8-frame halos. By default, 
    the halo is valid visual context, matching GFSLT-VLP; batch-level padding added later remains masked. 
    BIO labels for halo positions should still use `UNK`.
    """
    if poses.ndim < 2: raise ValueError(f"Expected poses with time dimension, got shape {tuple(poses.shape)}")
    n = int(poses.shape[0])
    total = gfslt_padded_length(n, multiple=multiple, halo=halo)
    right = total - n - int(halo)
    if right < 0: raise ValueError(f"Invalid GFSLT padding for length {n}: right pad {right}")

    left_pose = poses[:1].expand(int(halo), *poses.shape[1:])
    right_pose = poses[-1:].expand(right, *poses.shape[1:])
    padded_poses = torch.cat([left_pose, poses, right_pose], dim=0)

    if timestamps_s.numel() > 1: dt = torch.median(torch.diff(timestamps_s.float())).clamp_min(1e-6)
    else: dt = torch.tensor(1.0 / 25.0, dtype=torch.float32, device=timestamps_s.device)
    left_ts = timestamps_s[:1].float() - dt * torch.arange(int(halo), 0, -1, device=timestamps_s.device)
    right_ts = timestamps_s[-1:].float() + dt * torch.arange(1, right + 1, device=timestamps_s.device)
    padded_timestamps = torch.cat([left_ts, timestamps_s.float(), right_ts], dim=0)

    frame_mask = torch.cat([
        torch.full((int(halo),), bool(halo_is_valid), dtype=torch.bool, device=poses.device),
        torch.ones(n, dtype=torch.bool, device=poses.device),
        torch.full((right,), bool(halo_is_valid), dtype=torch.bool, device=poses.device),
    ], dim=0)

    padded_labels = None
    if bio_labels is not None:
        if pad_label is None: raise ValueError("pad_label is required when bio_labels are provided")
        padded_labels = torch.cat([
            torch.full((int(halo),), int(pad_label), dtype=bio_labels.dtype, device=bio_labels.device),
            bio_labels,
            torch.full((right,), int(pad_label), dtype=bio_labels.dtype, device=bio_labels.device),
        ], dim=0)
    return padded_poses, padded_timestamps, frame_mask, padded_labels
