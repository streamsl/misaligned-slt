"""The decode of a BIO head's rows and the segmenter dev monitor.

The decode of a standalone segmenter (`segmenter_decode`), which its rows read (eval.py --spans moryossef|s1, offline and online, with or
without text) and analyze.py --stage segmenter-errors checks, and of an arm's own head (`arm_decode`, eval.py --spans ar|dlm), by one
rule (`stamped_decode`, `require_config_decode`); the monitor (train/helpers.py) runs the whole-video pass of the offline no-text row
under its config's decode and its main score (the F1 of the tIoU match at tIoU avg, with no edge limit) on all dev records of each
language for checkpoint selection.
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


SEGMENTATION_DECODE = {"argmax": "bio_argmax", "viterbi": "legal_viterbi", "duration": "semi_markov_viterbi"}   # provenance names


def stamped_decode(meta: dict | None) -> str:
    # The decode that chose a checkpoint's epoch, as its `monitor_decode` stamp names it: argmax (plain is the stored name of argmax in
    # the checkpoints trained before the decodes were named argmax and viterbi), else the duration decode (semi_markov_viterbi, or no
    # stamp: a file with no meta, or an arm trained before the arms stamped their monitor's decode).
    return "argmax" if (meta or {}).get("monitor_decode") in ("argmax", "plain") else "duration"


def require_config_decode(meta: dict | None, cfg: dict, checkpoint, decode: str | None) -> None:
    """With no --decode (`decode`) a row decodes as the checkpoint's epoch was chosen (`stamped_decode`), so a stamped decode must be the
    one that the model's config `cfg` names for its rows (segmenter_decode: the segmenter config, or the arm's method config, which also
    applies to an S1 warm start in train-slt); else it is another monitor's choice. A file with no stamp passes: it decodes with the duration decode, the one that chose its epoch."""
    stamp = (meta or {}).get("monitor_decode")
    if stamp and not decode and stamped_decode(meta) != segmenter_config_decode(cfg): raise SystemExit(
        f"{checkpoint} chose its epoch under monitor_decode {stamp!r}, but its config names segmenter_decode "
        f"{segmenter_config_decode(cfg)!r}: train it again under that decode, or name the decode it was chosen under (segmenter_decode in "
        f"the config; an eval.py row also takes --decode)."
    )


def segmenter_decode(args: argparse.Namespace, checkpoint: str) -> str:
    """The decode of a segmenter row (S1 or the Moryossef segmenter): --decode, else the decode that selected the checkpoint
    (`stamped_decode`, train.helpers.segmenter_monitor_config): an argmax-selected checkpoint decodes with argmax, offline and online
    (argmax is the main decode, as greedy decoding is a translator's; viterbi and duration are the --decode rows of the decode
    comparison), any other one with the duration decode. An untrained control (--segmenter-init released or random) was selected by no
    monitor: it decodes as the Moryossef config names (segmenter_decode)."""
    if getattr(args, "decode", None): return args.decode
    if getattr(args, "segmenter_init", "finetuned") != "finetuned":
        return segmenter_config_decode(load_yaml(args.moryossef_config, language=source_language(args)))
    return stamped_decode(load_checkpoint_meta(checkpoint))


def arm_decode(args: argparse.Namespace, model, method_cfg: dict) -> str:
    """The decode of an arm's own head in its rows (eval.py --spans ar|dlm, offline and online), as a segmenter's: --decode, else the
    decode that chose the arm's epoch (`stamped_decode` of the file that build_eval_model loaded; the stage-2 monitor decodes as the
    method config's segmenter_decode names, train/slt.py), which must be the one that `method_cfg` names for the rows. An arm that no
    row monitor chose (checkpoint.monitor val_loss) decodes as its gate picked spans in training (`span_decode`, its stamp)."""
    meta, decode = getattr(model, "checkpoint_meta", None), getattr(args, "decode", None)
    require_config_decode(meta, method_cfg, getattr(model, "checkpoint_path", None), decode)
    return decode or (stamped_decode(meta) if (meta or {}).get("monitor_decode") else getattr(model, "span_decode", "duration"))


@torch.no_grad()
def segmenter_dev_scores(
    model, records: list[VideoRecord], device: torch.device, velocity: bool, 
    rope_chunk_s: float | None, priors: dict[str, DurationModel] | None, edge_cfg: dict, min_span_s: float,
) -> dict[str, float]:
    """Segmenter checkpoint selection (train/bio_pretrain.py, moryossef26/trainer.py): the whole-video pass and decode of
    `eval.py --offline --text none`: the duration decode with each language's own train-fitted prior (`priors`), or argmax (`priors` None),
    the span rule of every offline row (complete spans of at least `min_span_s`, no buffer cap), scored with the row's main score, the
    F1 of the tIoU match at tIoU avg (`metrics.edge_chunk_scores` `no_edge_limit`, with no edge-limit grid; `edge_cfg` = eval.yaml
    scoring chunk_gap_s and tiou_thresholds).
    `offline_f1` is the mean over languages, so a pooled run weights every language equally. Each rank decodes every world_size-th record and the 
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
                scores[f"offline_f1_{language}"] = float(edge_chunk_scores(
                    events, gold_chunks(recs, float(edge_cfg["chunk_gap_s"])), (), edge_cfg["tiou_thresholds"], char_level=False,
                )["no_edge_limit"]["avg"]["f1"])
    finally: model.train(was_training)
    scores["offline_f1"] = float(np.mean([scores[f"offline_f1_{language}"] for language in by_language]))
    return scores
