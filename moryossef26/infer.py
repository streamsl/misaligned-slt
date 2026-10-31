"""Whole-stream inference + standalone eval, shared by both segmenters.

Wrapper for faithful Moryossef segmenter (their landmarks + velocity) and the in-system BIO head as the `s1`
ablation (Uni-Sign features). Chunking lives INSIDE the model at train-time chunk size — the train-consistent
Moryossef inference. Per-model: the `velocity` flag (UNet only) and the logits container.
"""
from __future__ import annotations
import numpy as np
import torch

from data.loader import VideoRecord
from data.windowing import BIO, make_bio_labels
from moryossef26.dataset import pose_aspect, release_velocity, to_release_coords

from poses import load_pose_window
from models.bio_head import chunk_normalized_logits
from infer.commit_gate import candidate_spans
from metrics import Segment, frame_scores
from utils import seconds_to_frames


def span_segments(tags, timestamps_s, fps: float, min_span_s: float) -> list[Segment]:
    """Decoded BIO tags of 1 whole stream → time-domain phrase Segments by the span rule of every offline row (the --offline row's):
    the FSM's complete spans of at least Λ_min (`min_span_s`, utils.lambda_min_s; infer.commit_gate.candidate_spans), each whole, with
    no buffer cap. So the offline no-text rows of the segmenters (eval.py --offline --text none), the cascades on their spans and the
    segmenters' val_video_f1 score the spans of a standard offline row. End = the terminator frame's onset."""
    times = [float(t) for t in (timestamps_s.tolist() if isinstance(timestamps_s, torch.Tensor) else timestamps_s)]
    spans = candidate_spans(torch.as_tensor(tags).cpu(), seconds_to_frames(min_span_s, fps))
    return [Segment(times[start], times[end]) for start, end in spans]


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
    """Phrase logits and timestamps. Both arms overlap-stitch their trained RoPE context 
    (models.bio_head.overlap_windows); S1 also normalizes per chunk.

    The Moryossef arm reads the RELEASED model's own input contract (moryossef26.dataset.to_release_coords):
    their 50 landmarks, their shoulder/standardise normalisation, no face, velocity after — the same representation
    at train and inference, and no video crop box.
    """
    if hasattr(model, "bio_head"):
        if rope_chunk_s is None: raise ValueError("S1 whole-video pass needs rope_chunk_s (the trained context)")
        raw, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=False)
        if raw.shape[0] == 0: return None, timestamps
        chunk = max(1, round(float(rope_chunk_s) * float(record.pose.fps)))
        forward = lambda p, m, t: model(p, frame_mask=m, timestamps_s=t).logits
        return chunk_normalized_logits(forward, raw, timestamps, chunk, device), timestamps
    raw, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=False)
    if raw.shape[0] == 0: return None, timestamps
    coords = to_release_coords(raw, getattr(model, "release_stats", None), aspect=pose_aspect(record.pose))
    return _phrase_logits(model, coords, timestamps, device, velocity), timestamps


@torch.no_grad()
def predict_phrase_segments(
    model, records: list[VideoRecord], device: torch.device, velocity: bool = True, 
    rope_chunk_s: float | None = None, decoder=None, *, min_span_s: float,
) -> dict[str, list[Segment]]:
    # 1 decode/stream: spans keyed by record id on the stream's own clock (evaluation.predictions.to_source maps them to the source
    # video). `decoder`: None, the most likely tag of each frame (argmax); an infer.duration_decode.DurationDecoder, the best legal path.
    model.eval().to(device)
    predictions: dict[str, list[Segment]] = {}

    for record in records:
        logits, timestamps = whole_video_logits(model, record, device, velocity, rope_chunk_s)
        if logits is None:
            predictions[record.video_id] = []
            continue
        if decoder is not None: tags = decoder.decode(
            logits, torch.tensor([logits.shape[1]], device=logits.device), 
            timestamps_s=torch.as_tensor(timestamps, device=logits.device)[None]
        )[0]
        else: tags = logits.argmax(dim=-1)[0].detach().cpu()
        predictions[record.video_id] = span_segments(tags, timestamps, float(record.pose.fps), min_span_s)
    return predictions


@torch.no_grad()
def evaluate_moryossef_whole_video(
    model, records: list[VideoRecord], device: torch.device, velocity: bool = True, rope_chunk_s: float | None = None,
    tiou_thresholds: tuple[float, ...] = (0.5,), return_segments: bool = False, decoder=None, min_span_s: float | None = None,
) -> dict[str, float] | tuple[dict[str, float], dict[str, list[Segment]]]:
    """Moryossef evaluate.py-style STANDALONE eval: whole streams, GT phrase BIO from caption spans, frame-F1 and
    1-to-1 tIoU segment P/R/F1 averaged over streams. A stream is a record: a whole video, or one clean segment of 
    a cut video (data.loader.split_clean_segments), so this mean is per stream, not per source video. Phrase only, 
    no sign head; for `--spans s1` it scores the pretrained in-system BIO head before joint fine-tuning.

    `rope_chunk_s` (S1 only): trained context in SECONDS (= buffer_cap_s) → frames per stream.
    Gold follows both arms' training label rule (TRUSTED_GAP_S): long uncaptioned stretches → UNK (excluded), not O,
    else the segmenter is penalized for firing on possibly-uncaptioned signing.
    `tiou_thresholds`: eval.yaml scoring.tiou_thresholds → the thresholds of the rows' tIoU columns and densevid scores.

    `decoder`: None, the most likely tag of each frame (argmax); an infer.duration_decode.DurationDecoder, the best legal BIO path (with
    no prior: viterbi; with the duration prior: the duration decode), decoded on the CPU. Each stream's scores are metrics.frame_scores:
    the bio_* frame scores read the raw argmax, the phrase_* scores the decoded tags.

    The segmenters' checkpoint monitor (`val_video_f1`, evaluation.segmenter_eval.segmenter_dev_scores) decodes the
    same whole-video pass under its config's segmenter_decode and scores it with the row scores of `eval.py --offline --text none`
    (evaluation.offline.run_offline, which saves this function's metrics as the row's `frame` block, the key set of the online rows'
    block too). The dev-chunk numbers it also logs use other context, units and aggregation and are diagnostics only.
    """
    model.eval().to(device)
    rows: list[dict[str, float]] = []
    # `return_segments`: the SAME decoded tags, as time-domain spans (keyed by record id, stream clock) BEFORE 
    # the UNK ignore-region masking — the caller feeds them to the scoring path of the rows, whose quarantine rule 
    # (drop majority-quarantined events) replaces the frame-level mask. 1 decode, 2 protocols.
    segments_by_video: dict[str, list[Segment]] = {}

    for record in records:
        logits, timestamps = whole_video_logits(model, record, device, velocity, rope_chunk_s)
        if logits is None: continue
        gold = make_bio_labels(
            timestamps, record.sentences, 0.0, float(record.pose.duration_s), 
            video_duration_s=record.pose.duration_s, untrusted=record.untrusted_s,
        )
        logits = logits.detach().cpu()
        raw_logits = logits
        if decoder is not None:
            tags = decoder.decode(logits, torch.tensor([logits.shape[1]]), timestamps_s=torch.as_tensor(timestamps)[None])[0]
            logits = torch.nn.functional.one_hot(tags.long(), num_classes=logits.shape[-1]).float().unsqueeze(0)
        if return_segments:
            raw_tags = logits[0].argmax(dim=-1)   # the offline span rule (`min_span_s` floor, no buffer cap) on the decoded tags
            segments_by_video[record.video_id] = span_segments(raw_tags, timestamps, float(record.pose.fps), min_span_s)
        labels = torch.as_tensor(np.asarray(gold)).long().unsqueeze(0)
        # Ignore-region: where gold is UNK (quarantined chains / untrusted gaps) a prediction is neither right 
        # nor wrong — force the pred to UNK there too, else phantom segments in no-GT regions deflate precision.
        unk_mask = labels[0] == BIO["UNK"]
        # A stream with no gold signing (all UNK, or O/UNK only: a clean segment with no reliable unit) gets no row: its F1
        # is undefined, and a 0 or 1 there would move the mean. Its spans are returned above and still reach the row scores
        # (main.edge_chunk, densevid).
        if not ((labels[0] == BIO["B"]) | (labels[0] == BIO["I"])).any(): continue
        if unk_mask.any():
            logits = logits.clone()
            logits[0, unk_mask] = 0.0
            logits[0, unk_mask, BIO["UNK"]] = 10.0
        rows.append(frame_scores(raw_logits, logits, labels, tiou_thresholds))
    metrics = {} if not rows else {k: float(sum(r[k] for r in rows) / len(rows)) for k in rows[0].keys()}
    return (metrics, segments_by_video) if return_segments else metrics
