"""Whole-stream inference + standalone eval, shared by both segmenters.

Wrapper for faithful Moryossef segmenter (their landmarks + velocity) and the in-system BIO head as the `s1`
ablation (Uni-Sign features). Chunking lives INSIDE the model at train-time chunk size — the train-consistent
Moryossef inference. Per-model: the `velocity` flag (UNet only) and the logits container.
"""
from __future__ import annotations
import numpy as np
import torch

from data.loader import VideoRecord
from data.windowing import BIO, TRUSTED_GAP_S, make_bio_labels
from poses import load_pose_window
from models.bio_head import chunk_normalized_logits
from infer.duration_decode import DurationDecoder
from moryossef26.dataset import pose_aspect, release_velocity, to_release_coords
from metrics import Segment, bio_frame_metrics, moryossef_segment_metrics, signing_runs_with_b_splits


def bio_tags_to_segments(tags, timestamps_s) -> list[Segment]:
    """BIO argmax tags → time-domain phrase Segments (shared span-decode rule).

    Spans = contiguous signing runs (Moryossef's `likeliest_probs_to_segments` never needs a predicted `B`), split
    at interior `B`s to keep adjacent sentences separate. End = onset of the next frame, extrapolated one step past
    sequence end for open spans.
    """
    if not isinstance(tags, torch.Tensor): tags = torch.as_tensor(tags)
    times = [float(t) for t in (timestamps_s.tolist() if isinstance(timestamps_s, torch.Tensor) else timestamps_s)]
    if not times: return []

    step = (times[-1] - times[-2]) if len(times) > 1 else 0.04  # assume 25fps when a single frame
    segments: list[Segment] = []
    for seg in signing_runs_with_b_splits(tags):
        end_idx = int(seg["end"]) + 1
        end_t = times[end_idx] if end_idx < len(times) else times[-1] + step
        segments.append(Segment(times[int(seg["start"])], end_t))
    return segments


def _phrase_logits(
    model, poses_np: np.ndarray, timestamps_np: np.ndarray, 
    device: torch.device, velocity: bool
) -> torch.Tensor:
    if velocity: poses_np = release_velocity(poses_np, timestamps_np)
    poses = torch.as_tensor(poses_np, dtype=torch.float32, device=device).unsqueeze(0)
    timestamps = torch.as_tensor(timestamps_np, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.ones(poses.shape[:2], dtype=torch.bool, device=device)
    out = model(poses, frame_mask=mask, timestamps_s=timestamps)
    return out["phrase"] if isinstance(out, dict) else out.logits  # MoryossefSegmenter dict vs BIOHeadOutput


def whole_video_logits(model, record: VideoRecord, device: torch.device, velocity: bool, rope_chunk_s: float | None):
    """Phrase logits and timestamps. S1 uses one normalization per trained-context chunk.

    The Moryossef arm reads the RELEASED model's own input contract (moryossef26.dataset.to_release_coords):
    their 50 landmarks, their shoulder/standardise normalisation, no face, velocity after — the same representation
    at train and inference, and no video crop box.
    """
    if hasattr(model, "bio_head"):
        if rope_chunk_s is None: raise ValueError("S1 whole-video pass needs rope_chunk_s (the trained context)")
        raw, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=False)
        if raw.shape[0] == 0: return None, timestamps
        chunk = max(1, round(float(rope_chunk_s) * float(record.pose.fps)))
        model.bio_head.chunk_size, model.bio_head.chunk_overlap = chunk, True   # a chunk-length input is one RoPE pass
        forward = lambda p, m, t: model(p, frame_mask=m, timestamps_s=t).logits
        return chunk_normalized_logits(forward, raw, timestamps, chunk, device), timestamps
    raw, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=False)
    if raw.shape[0] == 0: return None, timestamps
    coords = to_release_coords(raw, getattr(model, "release_stats", None), aspect=pose_aspect(record.pose))
    return _phrase_logits(model, coords, timestamps, device, velocity), timestamps


@torch.no_grad()
def predict_phrase_segments(
    model, records: list[VideoRecord], device: torch.device, velocity: bool = True, 
    rope_chunk_s: float | None = None, duration=None
) -> dict[str, list[Segment]]:
    model.eval().to(device)
    predictions: dict[str, list[Segment]] = {}

    for record in records:
        logits, timestamps = whole_video_logits(model, record, device, velocity, rope_chunk_s)
        if logits is None:
            predictions[record.video_id] = []
            continue
        if duration is not None: tags = DurationDecoder(duration).decode(
            logits, torch.tensor([logits.shape[1]], device=logits.device), 
            timestamps_s=torch.as_tensor(timestamps, device=logits.device)[None]
        )[0]
        else: tags = logits.argmax(dim=-1)[0].detach().cpu()
        predictions[record.video_id] = bio_tags_to_segments(tags, timestamps.tolist())
    return predictions


def _segment_rows(logits: torch.Tensor, labels: torch.Tensor, tiou_thresholds: tuple[float, ...]) -> dict[str, float]:
    # 1 video's segment metrics, under the shared key convention.
    seg_keys = ("phrase_tiou_f1", "phrase_seg_precision", "phrase_seg_recall")
    row: dict[str, float] = {}
    per_key: dict[str, list[float]] = {k: [] for k in seg_keys}
    for t in tiou_thresholds:
        seg = moryossef_segment_metrics(logits, labels, prefix="phrase", tiou_threshold=float(t))
        row.setdefault("phrase_frame_f1", seg["phrase_frame_f1"])
        # Moryossef 2023's "%" metric: predicted/gold segment-count ratio, threshold-independent (counts come from the 
        # decode, not the matching). Reads over-/under-segmentation at a glance and is the cross-paper comparable pair to 
        # their (F1, %); the 1-to-1 tIoU F1 above stays the headline — count ratios alone are gameable (a right count 
        # with wrong placements scores 1.0).
        row.setdefault("phrase_segment_count_ratio", float(seg["phrase_n_pred"]) / max(1.0, float(seg["phrase_n_gold"])))
        for k in seg_keys:
            row[f"{k}@{t:g}"] = seg[k]
            per_key[k].append(seg[k])
    for k in seg_keys: row[f"{k}_avg"] = float(sum(per_key[k]) / len(per_key[k]))
    return row


@torch.no_grad()
def evaluate_moryossef_whole_video(
    model, records: list[VideoRecord], device: torch.device, velocity: bool = True, rope_chunk_s: float | None = None,
    trusted_gap_s: float | None = TRUSTED_GAP_S, tiou_thresholds: tuple[float, ...] = (0.5,), 
    return_segments: bool = False, duration=None,
) -> dict[str, float] | tuple[dict[str, float], dict[str, list[Segment]]]:
    """Moryossef evaluate.py-style STANDALONE eval: whole videos, GT phrase BIO from caption spans, frame-F1 and
    1-to-1 tIoU segment P/R/F1 averaged. Phrase only, no sign head; for `--segmenter-arch s1` it scores the
    pretrained in-system BIO head before joint fine-tuning.

    `rope_chunk_s` (S1 only): trained context in SECONDS (= buffer_cap_s) → frames per stream.
    `trusted_gap_s` defaults to TRUSTED_GAP_S to match training GT: uncaptioned stretches → UNK (excluded), not O,
    else the segmenter is penalized for firing on possibly-uncaptioned signing.
    `tiou_thresholds`: eval.yaml rq2.tiou_thresholds → comparable to the RQ2 segmentation block.

    The current window monitor pools complete-span counts at 1 threshold. Whole-video scoring uses different context, 
    units and aggregation. Compare checkpoints under both protocols before attributing a gap to training; a protocol 
    difference does not rule out a model or implementation failure.
    """
    model.eval().to(device)
    rows: list[dict[str, float]] = []
    # `return_segments`: the SAME decoded tags, as time-domain spans BEFORE the UNK ignore-region masking —
    # the caller feeds them to the RQ2 scoring path, whose quarantine rule (drop majority-quarantined events)
    # replaces the frame-level mask. 1 decode, 2 protocols.
    segments_by_video: dict[str, list[Segment]] = {}

    for record in records:
        logits, timestamps = whole_video_logits(model, record, device, velocity, rope_chunk_s)
        if logits is None: continue
        gold = make_bio_labels(
            timestamps, record.sentences, 0.0, float(record.pose.duration_s),
            trusted_gap_s=trusted_gap_s, video_duration_s=record.pose.duration_s,
        )
        logits = logits.detach().cpu()
        raw_logits = logits
        if duration is not None:
            tags = DurationDecoder(duration).decode(
                logits, torch.tensor([logits.shape[1]]), timestamps_s=torch.as_tensor(timestamps)[None]
            )[0]
            logits = torch.nn.functional.one_hot(tags.long(), num_classes=logits.shape[-1]).float().unsqueeze(0)
        if return_segments:
            raw_tags = logits[0].argmax(dim=-1)
            segments_by_video[record.video_id] = bio_tags_to_segments(raw_tags, timestamps.tolist())
        labels = torch.as_tensor(np.asarray(gold)).long().unsqueeze(0)
        # Ignore-region: where gold is UNK (quarantined chains / untrusted gaps) a prediction is neither right nor
        # wrong — force the pred to UNK there too, else phantom segments in no-GT regions deflate precision.
        unk_mask = labels[0] == BIO["UNK"]
        if unk_mask.any():
            logits = logits.clone()
            logits[0, unk_mask] = 0.0
            logits[0, unk_mask, BIO["UNK"]] = 10.0
        # prefix "bio" (trainer convention): bio_f1 is BINARY signing-vs-not F1; under "phrase" it would sit next
        # to phrase_frame_f1 (macro O/B/I, the §4.6 acceptance number) and read as the same thing.
        row = dict(bio_frame_metrics(raw_logits, labels, prefix="bio"))
        # Moryossef 2023's IoU (binary signing-vs-not, per video then averaged) = PR/(P+R−PR), algebraically F1/(2−F1). 
        # Reported for the cross-paper (F1, IoU, %) triple; weak alone — on dense corpora 1 all-signing span scores 
        # high IoU while resolving no boundaries, so the 1-to-1 tIoU F1 stays the headline.
        _p, _r = row["bio_precision"], row["bio_recall"]
        row["bio_iou"] = (_p * _r / (_p + _r - _p * _r)) if (_p + _r - _p * _r) > 0 else 0.0
        row.update(_segment_rows(logits, labels, tiou_thresholds))
        rows.append(row)
    metrics = {} if not rows else {k: float(sum(r[k] for r in rows) / len(rows)) for k in rows[0].keys()}
    return (metrics, segments_by_video) if return_segments else metrics
