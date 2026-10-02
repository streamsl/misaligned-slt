"""Segmentation training data for both arms: whole-input random chunks, BIO only (dev monitor: train.helpers.evaluate_bio_chunks).

Each draw tiles the selected videos with chunk lengths U(L_min, C), independent of caption boundaries. For S1, C is the
buffer cap and L_min = δ + K·stride sets a short training context (it is not a minimum legal streaming-buffer length);
chunk lengths vary so that normalization sees both short and long windows. The Moryossef arm uses the same `ChunkDataset`
with L_min = C = its fixed chunk and its release input contract (`release`). The arms differ by model and input contract,
and by the unit of the chunk: S1's is SECONDS (its head is banded in seconds and its Uni-Sign box depends on the window, 
like a stream buffer of at most C), the release's is FRAMES (1 1024-frame RoPE pass): under fps augmentation the release 
arm loads the native span that resamples to its fixed frame count, as the release's `load_and_augment` does. The fixed 
epoch budget can omit or repeat chunks from a draw (see ChunkDataset), so neither complete frame coverage nor exactly 1 
expected visit per frame is guaranteed.
"""
from __future__ import annotations
import numpy as np

from torch.utils.data import Dataset
from data.loader import VideoRecord
from data.windowing import make_bio_labels
from moryossef26.dataset import pose_aspect, release_velocity, to_release_coords
from poses import apply_fps_aug, apply_frame_dropout, apply_hand_dropout, load_pose_window, normalize_keypoints_unisign


def draw_chunks(
    records: list[VideoRecord], chunk_s: float, rng: np.random.Generator, min_chunk_s: float,
) -> list[tuple[VideoRecord, float, float]]:
    """Tile each record (1 clean segment) with chunks (record, start_s, end_s) of length U(min_chunk_s, C), in random order.

    The first chunk starts at a random phase; later cut positions follow the sampled lengths. No caption boundary is
    used to place a cut. The first and last chunks are clipped by the record ends. A record shorter than C can still have
    several chunks when short lengths are drawn.
    """
    if not 0 < min_chunk_s <= chunk_s: raise ValueError(f"need 0 < min_chunk_s <= chunk_s, got {min_chunk_s}, {chunk_s}")
    chunks = []
    for rec in records:
        duration = float(rec.pose.duration_s)
        length = float(rng.uniform(min_chunk_s, chunk_s))
        start = -float(rng.uniform(0.0, length))
        while start < duration:
            chunks.append((rec, max(start, 0.0), min(start + length, duration)))
            start += length
            length = float(rng.uniform(min_chunk_s, chunk_s))
    rng.shuffle(chunks)  # a capped epoch (see ChunkDataset) drops a random tail, not the last videos
    return chunks


class ChunkDataset(Dataset):
    """Random whole-input chunks for a segmenter. Train: shared augmentation (poses/augmentation.py), chunks re-drawn every
    epoch (and the pooled records rotated through `records_for_epoch`). `deterministic=True` (dev): chunks drawn once from
    `seed`, no augmentation, so every epoch scores the identical set. `release` picks the input contract: None = S1's Uni-Sign 
    69-point normalization; {"stats": fitted table, "velocity": bool} = Moryossef release contract (moryossef26/dataset.py), 
    with velocity computed after the frame-removing augmentations.

    The epoch length is fixed at 1st draw to keep the optimizer-step schedule fixed. A later draw with more chunks omits a 
    shuffled tail; one with fewer chunks repeats its head. Given M drawn chunks and an epoch budget of N, each chunk has N/M 
    expected visits under the shuffle. This need not be one. Pool rotation guarantees record selection over a full cycle, not 
    that all frames of those records reach the model. FPS augmentation and frame dropout also remove frames.
    """
    def __init__(
        self, records: list[VideoRecord], chunk_s: float, min_chunk_s: float, augmentation: dict | None = None, 
        seed: int = 42, records_for_epoch=None, deterministic: bool = False, release: dict | None = None,
    ):
        if not records: raise ValueError(f"{type(self).__name__} requires at least one record")
        self.records, self.chunk_s, self.min_chunk_s, self.seed = records, float(chunk_s), float(min_chunk_s), int(seed)
        self.deterministic = bool(deterministic)
        self._records_for_epoch = records_for_epoch
        self._epoch = 0
        self.augmentation = None if self.deterministic else augmentation
        self.release = release
        self.chunks = draw_chunks(records, self.chunk_s, np.random.default_rng((self.seed, 0)), self.min_chunk_s)
        self.epoch_size = len(self.chunks)

    @property # data.loader.streaming_loader / LengthBucketSampler read chunk lengths through dataset.sampler.spec_frames
    def sampler(self): return self

    def __len__(self) -> int: return self.epoch_size

    def set_epoch(self, epoch: int) -> None:
        if self.deterministic: return
        self._epoch = int(epoch)
        if self._records_for_epoch is not None: self.records = self._records_for_epoch(self._epoch) or self.records
        self.chunks = draw_chunks(self.records, self.chunk_s, np.random.default_rng((self.seed, self._epoch)), self.min_chunk_s)

    def spec_frames(self, index: int) -> int:
        # Upper bound on the chunk's frame count (load_pose_window framing; the augmentations only remove frames).
        rec, start_s, end_s = self.chunks[int(index) % len(self.chunks)]
        fps = float(rec.pose.fps)
        return max(1, int(np.ceil(end_s * fps)) - int(np.floor(start_s * fps)) + 1)

    def __getitem__(self, index: int) -> dict:
        rec, start_s, end_s = self.chunks[int(index) % len(self.chunks)]
        # 1 rng per (seed, epoch, index): augmentation is independent of the DataLoader worker layout and replays on resume.
        rng = np.random.default_rng((self.seed, self._epoch, int(index)))
        aug, rel, fps = self.augmentation, self.release, float(rec.pose.fps)
        target_fps = float(rng.uniform(float(aug["fps"]["min_fps"]), float(aug["fps"]["max_fps"]))) if aug else fps
        if rel is not None and fps > 1.05 * target_fps:
            # Release order: load the native span that resamples to this chunk's frame count, kept inside the record.
            span = (end_s - start_s) * fps / target_fps
            end_s = min(float(rec.pose.duration_s), start_s + span)
            start_s = max(0.0, end_s - span)
        raw, timestamps = load_pose_window(rec.pose, start_s, end_s, normalize=False)
        if aug: raw = apply_hand_dropout(raw, float(aug["body_part_dropout"]), rng)
        poses = normalize_keypoints_unisign(raw) if rel is None else to_release_coords(raw, rel["stats"], aspect=pose_aspect(rec.pose))
        if aug and poses.shape[0] > 1:
            poses, timestamps = apply_fps_aug(poses, timestamps, fps, target_fps)
            poses, timestamps = apply_frame_dropout(poses, timestamps, float(aug["frame_dropout"]), rng)
        # BIO from the kept frame times: UNK over untrusted gaps, a fragment cut by the left edge is I (no B).
        # The left edge is the first LOADED frame (the loader floors the start frame; frame dropout keeps both edges), so an 
        # onset after it keeps its B. The collator pads with UNK, never O.
        labels = make_bio_labels(timestamps, rec.sentences, float(timestamps[0]), end_s, video_duration_s=rec.pose.duration_s)
        if rel is not None and rel["velocity"]: poses = release_velocity(poses, timestamps)
        return {
            "poses": poses, "timestamps_s": timestamps - start_s, "bio_labels": labels,
            "spec": {"video_id": rec.video_id, "start_s": start_s, "end_s": end_s},
        }
