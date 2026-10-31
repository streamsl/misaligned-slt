"""The decode of a segmenter's rows and the segmenter dev monitor.

The decode of a standalone segmenter (`segmenter_decode`), which its rows read (eval.py --spans moryossef|s1, offline and online, with or
without text) and analyze.py --stage segmenter-errors checks; the monitor (train/helpers.py) runs the whole-video pass of the offline
no-text row under its config's decode and its main score (the F1 of the tIoU x edge-limit grid, tau avg and tIoU avg) on all dev records
of each language for checkpoint selection.
"""
from __future__ import annotations
from contextlib import nullcontext
import argparse
import numpy as np

import torch
from data.loader import VideoRecord
from infer.duration_decode import DurationDecoder, DurationModel
from moryossef26.infer import predict_phrase_segments
from train import distributed as dist
from evaluation.predictions import gold_chunks, segment_events
from metrics import edge_chunk_scores
from evaluation.systems import source_language
from models.checkpointing import load_checkpoint_meta
from train.helpers import segmenter_config_decode
from utils import load_yaml


def segmenter_decode(args: argparse.Namespace, checkpoint: str) -> str:
    """The decode of a segmenter row (S1 or the Moryossef segmenter): --decode, else the decode that selected the checkpoint, as its
    `monitor_decode` stamp names it (train.helpers.segmenter_monitor_config): an argmax-selected checkpoint decodes with argmax, offline
    and online (argmax is the main decode, as greedy decoding is a translator's; viterbi and duration are the --decode rows of the decode
    comparison); any other stamp or none (semi_markov_viterbi, a file with no meta) with the duration decode. An untrained control (--segmenter-init released or random) was selected by no monitor: it decodes as the Moryossef config
    names (segmenter_decode)."""
    if getattr(args, "decode", None): return args.decode
    if getattr(args, "segmenter_init", "finetuned") != "finetuned":
        selected = segmenter_config_decode(load_yaml(args.moryossef_config, language=source_language(args)))
    else:   # plain: the stamp of argmax in the checkpoints trained before the decode names argmax and viterbi
        selected = "argmax" if load_checkpoint_meta(checkpoint).get("monitor_decode") in ("argmax", "plain") else "duration"
    return "argmax" if selected == "argmax" else "duration"


@torch.no_grad()
def segmenter_dev_scores(
    model, records: list[VideoRecord], device: torch.device, velocity: bool, 
    rope_chunk_s: float | None, priors: dict[str, DurationModel] | None, edge_cfg: dict, min_span_s: float,
) -> dict[str, float]:
    """Segmenter checkpoint selection (train/bio_pretrain.py, moryossef26/trainer.py): the whole-video pass and decode of
    `eval.py --offline --text none`: the duration decode with each language's own train-fitted prior (`priors`), or argmax (`priors` None),
    the span rule of every offline row (complete spans of at least `min_span_s`, no buffer cap), scored with the row's main score, the
    F1 of the tIoU x edge-limit grid at tau avg and tIoU avg (`edge_cfg` = eval.yaml scoring chunk_gap_s, edge_taus_s and tiou_thresholds).
    `video_f1` is the mean over languages, so a pooled run weights every language equally. Each rank decodes every world_size-th record and the 
    spans are merged, so every rank returns the same scores. The decode switches the model to eval mode; its mode is restored."""
    by_language: dict[str, list[VideoRecord]] = {}
    for record in records: by_language.setdefault(str(record.language), []).append(record)
    if not by_language: raise ValueError("The segmenter dev monitor needs at least 1 dev record.")
    was_training, scores = model.training, {}
    # Full precision, as `eval.py --offline --text none` decodes: the training loop calls its monitor under CUDA autocast, and bf16/fp16
    # logits can move a duration-Viterbi boundary. Autocast exists only on CUDA here (train.helpers.AmpHelper).
    full_precision = torch.autocast(device_type="cuda", enabled=False) if torch.device(device).type == "cuda" else nullcontext()
    try:
        with full_precision:
            for language, recs in sorted(by_language.items()):
                local = predict_phrase_segments(
                    model, recs[dist.rank()::dist.world_size()], device, velocity=velocity, 
                    rope_chunk_s=rope_chunk_s, decoder=None if priors is None else DurationDecoder(priors[language]), min_span_s=min_span_s,
                )
                events = segment_events(dist.merge_dicts(local), recs, tag=f"val-video {language}")
                scores[f"video_f1_{language}"] = float(edge_chunk_scores(
                    events, gold_chunks(recs, float(edge_cfg["chunk_gap_s"])), edge_cfg["edge_taus_s"], edge_cfg["tiou_thresholds"],
                    char_level=False,
                )["avg"]["f1"])
    finally: model.train(was_training)
    scores["video_f1"] = float(np.mean([scores[f"video_f1_{language}"] for language in by_language]))
    return scores
