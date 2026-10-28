"""Segmentation training data for both arms: whole-input random chunks, BIO only (dev monitor: train.helpers.evaluate_segmenter).

Each draw tiles selected videos with chunk lengths U(L_min, C), independent of caption boundaries. For S1, C is the buffer cap 
and L_min = δ + K·stride sets a short training context (it is not a minimum legal streaming-buffer length). The chunk lengths 
vary so that normalization sees both short and long windows. Moryossef arm uses same `ChunkDataset` with L_min = C = its fixed 
chunk and its release input contract (`release`). The arms differ by model and input contract, and by the unit of the chunk: 
S1's is SECONDS (its head is banded in seconds and its Uni-Sign box depends on the window, like a stream buffer of at most C), 
the release's is FRAMES (1 1024-frame RoPE pass): under fps augmentation the release arm loads the native span that resamples 
to its fixed frame count, as the release's `load_and_augment` does. Fixed epoch budget can omit or repeat chunks from a draw 
(see ChunkDataset), so neither complete frame coverage nor exactly 1 expected visit per frame is guaranteed.

S1's train set may turn tiles into 2 other input types (`inputs`, bio_pretrain.yaml pretrain.inputs): restart chunks, which start
δ before a gold terminator of their record (FSM's buffer start after a correct commit), and stage-2 windows (data/sampler.py),
BIO only, with every frame whose receptive field reaches a window edge UNK. Moryossef arm and every dev set draw plain chunks.
A stage-2 arm's train rows use the same 3 types, the length law, restart points and edge mask of this file (data/sampler.py spec_for).
"""
from __future__ import annotations
import numpy as np

from torch.utils.data import Dataset
from data.loader import VideoRecord
from data.windowing import BIO, make_bio_labels
from moryossef26.dataset import pose_aspect, release_velocity, to_release_coords
from poses import apply_fps_aug, apply_frame_dropout, apply_hand_dropout, load_pose_window, normalize_keypoints_unisign, window_frames

from models.bio_head import BAND_SLACK_S
from infer.commit_gate import bio_complete_spans
from utils import seconds_to_frames

INPUT_TYPES = ("chunks", "windows", "restart_chunks")


def shortest_chunk_s(inference_cfg: dict) -> float:
    """L_min of the chunk length law U(L_min, C), C = buffer_cap_s: δ + K·stride, the shortest buffer the FSM decides on after a
    commit (δ of committed context + K strides). A short training context, not a minimum legal streaming-buffer length."""
    stability = inference_cfg["boundary_stability"]
    return float(stability["delta_s"]) + int(stability["hysteresis_strides"]) * float(inference_cfg["stride_s"])


def restart_chunk(rec: VideoRecord, terminator_s: float, delta_s: float, length_s: float) -> tuple[float, float]:
    # A restart chunk's edges: δ (on the record's frame grid) before the terminator, where the FSM starts its buffer after a correct
    # commit, then `length_s` (a draw of the chunk length law), inside the record.
    start = max(0.0, terminator_s - seconds_to_frames(delta_s, rec.pose.fps) / float(rec.pose.fps))
    return start, min(start + length_s, float(rec.pose.duration_s))


def committed_prefix(timestamps_s: np.ndarray, terminator_s: float) -> int:
    # The frames of a restart chunk that are UNK (absolute times): up to and including the terminator frame, the first kept frame at or
    # after the terminator time, which carries the terminator label on any grid (after resampling, an abutting successor's B moves
    # onto it). The FSM commits the frames before the terminator frame and decodes that frame at a known start (infer/stream.py);
    # its label is UNK too, so no supervised label sits at a fixed distance from this caption-placed edge.
    return int(np.searchsorted(timestamps_s, terminator_s - 1e-3)) + 1


def draw_chunks(
    records: list[VideoRecord], chunk_s: float, rng: np.random.Generator, min_chunk_s: float,
) -> list[tuple[VideoRecord, float, float]]:
    """Tile each record (1 clean segment) with chunks (record, start_s, end_s) of length U(min_chunk_s, C), in random order.

    The first chunk starts at a random phase; later cut positions follow the sampled lengths. No caption boundary is used to place a cut. 
    The first and last chunks are clipped by the record ends. A record shorter than C can still have several chunks when short lengths 
    are drawn. A clipped piece that holds no frame (load_pose_window's frame rule. A sliver shorter than FRAME_SLACK of a frame at a 
    record end) is no chunk: its frame is in the neighbouring chunk, and the rng draws do not change.
    """
    if not 0 < min_chunk_s <= chunk_s: raise ValueError(f"need 0 < min_chunk_s <= chunk_s, got {min_chunk_s}, {chunk_s}")
    chunks = []
    for rec in records:
        duration, fps = float(rec.pose.duration_s), float(rec.pose.fps)
        length = float(rng.uniform(min_chunk_s, chunk_s))
        start = -float(rng.uniform(0.0, length))
        while start < duration:
            a, b = max(start, 0.0), min(start + length, duration)
            first, end = window_frames(fps, a, b)
            if end > first: chunks.append((rec, a, b))
            start += length
            length = float(rng.uniform(min_chunk_s, chunk_s))
    rng.shuffle(chunks)  # a capped epoch (see ChunkDataset) drops a random tail, not the last videos
    return chunks


def restart_points(rec: VideoRecord) -> list[float]:
    """Times of the gold terminators of `rec` where the FSM restarts its buffer after a correct commit: on the record's gold tags
    (its native frame grid, labelled as a chunk is), the O or B frame that ends a complete span (infer.commit_gate.bio_complete_spans).
    Quarantined units are UNK, so they open no span; a unit that runs into UNK (a quarantined unit, an untrusted gap) or into the
    record end gives no point. Only gold places a restart point (P2)."""
    t = np.arange(rec.pose.total_frames, dtype=np.float32) / float(rec.pose.fps)
    labels = make_bio_labels(
        t, rec.sentences, 0.0, rec.pose.duration_s, video_duration_s=rec.pose.duration_s, untrusted=rec.untrusted_s
    )
    return [float(t[term]) for _, term in bio_complete_spans(labels) if labels[term] in (BIO["O"], BIO["B"])]


def edge_mask(timestamps_s: np.ndarray, reach: dict) -> np.ndarray:
    """Frames whose head output reads a frame at either window edge: `reach["frames"]` frames by index from the edge 
    (temporal convs), then `reach["depth"]` banded attention layers of `reach["radius_s"]` each, with the band's own 
    float32 rule and slack (models.bio_head.attention_mask). `timestamps_s` are the times the head reads (the item's 
    `timestamps_s`); train.bio_pretrain.edge_reach reads `reach` from the model's modules."""
    ts = np.asarray(timestamps_s, dtype=np.float32)
    mask = np.zeros(ts.shape[0], dtype=bool)
    if not mask.size: return mask
    band = np.float32(float(reach["radius_s"]) + BAND_SLACK_S)
    lo, hi = min(int(reach["frames"]), mask.size - 1), max(mask.size - 1 - int(reach["frames"]), 0)
    for _ in range(int(reach["depth"])):   # the reached frames stay a prefix (suffix): the frame nearest the rest decides
        lo = int(np.flatnonzero(ts - ts[lo] <= band)[-1])
        hi = int(np.flatnonzero(ts[hi] - ts <= band)[0])
    mask[: lo + 1] = mask[hi:] = True
    return mask


class ChunkDataset(Dataset):
    """Random whole-input chunks for a segmenter. Train: shared augmentation (poses/augmentation.py), chunks re-drawn every
    epoch (and the pooled records rotated through `records_for_epoch`). `deterministic=True` (dev): chunks drawn once from
    `seed`, no augmentation, so every epoch scores the identical set. `release` picks the input contract: None = S1's Uni-Sign 
    69-point normalization; {"stats": fitted table, "velocity": bool} = Moryossef release contract (moryossef26/dataset.py), 
    with velocity computed after the frame-removing augmentations.

    `inputs` (S1's train set only): the input types each tile can become, in equal shares, inside its own record, so each language
    keeps its share of the tiles. "chunks" keeps the tile. "restart_chunks" draws a chunk of same length law U(min_chunk_s, C) that 
    starts `delta_s` (on the record's frame grid) before one of the record's `restart_points`. frames up to and including terminator 
    frame (the first kept frame at or after terminator time, so after resampling too) are UNK, so no supervised label sits at a fixed 
    distance from a caption-placed edge. "windows" draws a stage-2 window (`window_sampler.window_spec`) for one of record's reliable 
    units, BIO only, and `edge_mask(edge_reach)` makes its edge frames UNK. A tile whose record fits none of listed types (no reliable 
    unit, no restart point) is dropped. A record that fits only some of them draws from those. The placement draws from its own rng, 
    so ("chunks",) is the plain tiling. Entries of `chunks` are (record, start_s, end_s, input type, terminator_s or None).

    Epoch length is fixed at 1st draw to keep the optimizer-step schedule fixed. A later draw with more chunks omits a shuffled tail. 
    One with fewer chunks repeats its head. Given M drawn chunks and an epoch budget of N, each chunk has N/M expected visits under 
    the shuffle. This need not be one. Pool rotation guarantees record selection over a full cycle, not that all frames of those 
    records reach the model. FPS augmentation and frame dropout also remove frames.
    """
    def __init__(
        self, records: list[VideoRecord], chunk_s: float, min_chunk_s: float, augmentation: dict | None = None, 
        seed: int = 42, records_for_epoch=None, deterministic: bool = False, release: dict | None = None,
        inputs=("chunks",), delta_s: float | None = None, window_sampler=None, edge_reach: dict | None = None,
    ):
        if not records: raise ValueError(f"{type(self).__name__} requires at least one record")
        self.records, self.chunk_s, self.min_chunk_s, self.seed = records, float(chunk_s), float(min_chunk_s), int(seed)
        self.deterministic = bool(deterministic)
        self._records_for_epoch = records_for_epoch
        self._epoch = 0
        self.augmentation = None if self.deterministic else augmentation
        self.release = release
        self.inputs = tuple(inputs)
        if not self.inputs or len(set(self.inputs)) != len(self.inputs) or not set(self.inputs) <= set(INPUT_TYPES): 
            raise ValueError(f"inputs must list each of {list(INPUT_TYPES)} at most once, got {list(inputs)}")
        if self.inputs != ("chunks",) and (release is not None or self.deterministic): raise ValueError(
            f"inputs {list(self.inputs)}: only S1's train set draws windows or restart chunks; "
            f"the Moryossef arm (`release`) and every dev set draw plain chunks"
        )
        if "restart_chunks" in self.inputs and delta_s is None: raise ValueError("restart_chunks need delta_s")
        if "windows" in self.inputs and (window_sampler is None or edge_reach is None): 
            raise ValueError("windows need window_sampler and edge_reach")
        self.delta_s, self.window_sampler, self.edge_reach = delta_s, window_sampler, edge_reach
        self._restart_points: dict[tuple[str, str], list[float]] = {}   # per record, labelled once
        self.chunks = self._draw(records, 0)
        self.epoch_size = len(self.chunks)

    def _draw(self, records: list[VideoRecord], epoch: int) -> list[tuple]:
        tiles = draw_chunks(records, self.chunk_s, np.random.default_rng((self.seed, epoch)), self.min_chunk_s)
        if self.inputs == ("chunks",): return [(rec, a, b, "chunks", None) for rec, a, b in tiles]
        rng, out = np.random.default_rng((self.seed, epoch, 1)), []
        for rec, a, b in tiles:
            anchors = [i for i, s in enumerate(rec.sentences) if s.reliable] if "windows" in self.inputs else []
            points = []
            if "restart_chunks" in self.inputs:
                key = (rec.language, rec.video_id)
                if key not in self._restart_points: self._restart_points[key] = restart_points(rec)
                points = self._restart_points[key]
            kinds = [k for k in self.inputs if k == "chunks" or (k == "windows" and anchors) or (k == "restart_chunks" and points)]
            if not kinds: continue
            kind, tau = kinds[int(rng.integers(len(kinds)))], None
            if kind == "windows":
                spec = self.window_sampler.window_spec(rec, anchors[int(rng.integers(len(anchors)))], rng)
                a, b = spec.start_s, spec.end_s
            elif kind == "restart_chunks":
                tau = points[int(rng.integers(len(points)))]
                a, b = restart_chunk(rec, tau, self.delta_s, float(rng.uniform(self.min_chunk_s, self.chunk_s)))
            out.append((rec, a, b, kind, tau))
        return out

    @property # data.loader.streaming_loader / LengthBucketSampler read chunk lengths through dataset.sampler.spec_frames
    def sampler(self): return self

    def __len__(self) -> int: return self.epoch_size

    def set_epoch(self, epoch: int) -> None:
        if self.deterministic: return
        self._epoch = int(epoch)
        if self._records_for_epoch is not None: self.records = self._records_for_epoch(self._epoch) or self.records
        self.chunks = self._draw(self.records, self._epoch)

    def spec_frames(self, index: int) -> int:
        # Upper bound on the chunk's frame count (load_pose_window framing; the augmentations only remove frames).
        rec, start_s, end_s, _, _ = self.chunks[int(index) % len(self.chunks)]
        fps = float(rec.pose.fps)
        return max(1, int(np.ceil(end_s * fps)) - int(np.floor(start_s * fps)) + 1)

    def __getitem__(self, index: int) -> dict:
        rec, start_s, end_s, kind, terminator_s = self.chunks[int(index) % len(self.chunks)]
        # 1 rng per (seed, epoch, index): augmentation is independent of the DataLoader worker layout and replays on resume.
        rng = np.random.default_rng((self.seed, self._epoch, int(index)))
        aug, rel, fps = self.augmentation, self.release, float(rec.pose.fps)
        # Density resampling to a share of record's own rate (augmentation.fps min_factor, max_factor), so it means same at any fps.
        target_fps = fps * float(rng.uniform(float(aug["fps"]["min_factor"]), float(aug["fps"]["max_factor"]))) if aug else fps
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
        labels = make_bio_labels(
            timestamps, rec.sentences, float(timestamps[0]), end_s, video_duration_s=rec.pose.duration_s, untrusted=rec.untrusted_s
        )
        # A restart chunk's committed prefix with its terminator frame (`committed_prefix`). A window's edge frames.
        if kind == "restart_chunks": labels[: committed_prefix(timestamps, terminator_s)] = BIO["UNK"]
        if kind == "windows": labels[edge_mask(timestamps - start_s, self.edge_reach)] = BIO["UNK"]
        if rel is not None and rel["velocity"]: poses = release_velocity(poses, timestamps)
        return {
            "poses": poses, "timestamps_s": timestamps - start_s, "bio_labels": labels,
            "spec": {"video_id": rec.video_id, "start_s": start_s, "end_s": end_s, "input": kind},
        }
