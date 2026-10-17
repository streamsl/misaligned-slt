"""Segmenter evaluation (eval.py --segmenter-eval) and the segmenter dev monitor.

The whole-video pass and duration decode of a segmenter, scored as RQ2 segmentation; the monitor (train/helpers.py) runs the same
protocol on the balanced dev records of each language for checkpoint selection."""
from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import argparse, json

import numpy as np
import torch
from data.loader import annotation_fingerprint, ANNOTATION_PROTOCOL, load_language_records, VideoRecord
from infer.duration_decode import DurationModel
from moryossef26.infer import evaluate_moryossef_whole_video, predict_phrase_segments
from train import distributed as dist
from utils import load_yaml, output_dir
from evaluation.scoring import _segment_rq2_rows
from evaluation.systems import _load_segmenter


def run_segmenter_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Whole-video segmentation eval (Moryossef evaluate.py protocol): frame-F1 + 1-to-1 tIoU segment P/R/F1. The acceptance check ("F1 
    within order-of-magnitude of published") and — same protocol for either --segmenter-arch — the only apples-to-apples Moryossef-vs-S1 
    comparison (training monitors use different window regimes). For `s1` it also scores the pretrained head before stage-2 fine-tuning.
    """
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run segmenter eval on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    model, device, velocity, rope_chunk_s, checkpoint = _load_segmenter(args)
    decode = args.segmenter_decode or "duration"
    duration = DurationModel.for_language(data_cfg, args.language) if decode == "duration" else None
    print(f"[segmenter-eval] {args.segmenter_arch} segmenter from {checkpoint} (decode={decode})", flush=True)

    thresholds = tuple(float(t) for t in (load_yaml(args.eval_config).get("rq2", {}) or {}).get("tiou_thresholds", [0.5, 0.7, 0.9]))
    metrics, segments_by_video = evaluate_moryossef_whole_video(
        model, records, device=device, velocity=velocity, rope_chunk_s=rope_chunk_s, 
        tiou_thresholds=thresholds, return_segments=True, duration=duration
    )
    # 2 protocols over 1 decode, both reported so the delta between standalone and RQ2 tables is a printed pair. `metrics` = Moryossef-comparable 
    #  protocol (per-stream mean of F1s; gold = the UNK-masked BIO label stream). `rq2_protocol` = SAME spans scored by SAME code path RQ2 uses 
    # (evaluate_predicted_events: caption-span gold per source video, quarantine-dropped events, F1 of macro-averaged P/R) — quote THIS one 
    # wherever localization is compared across tables; keep `metrics` for cross-paper comparability with Moryossef's published protocol.
    rq2_rows = _segment_rq2_rows(segments_by_video, records, thresholds, tag="segmenter-eval")
    rq2_protocol = {f"{r['tiou_threshold']:g}": {
        **r["segmentation"], "count_ratio": r["total_predictions"] / max(1, r["total_gold"])
    } for r in rq2_rows}
    for t in thresholds: print(
        f"[segmenter-eval] tIoU {t:g}: moryossef-protocol F1 {metrics.get(f'phrase_tiou_f1@{t:g}', float('nan')):.3f} | "
        f"rq2-protocol F1 {rq2_protocol[f'{t:g}']['f1']:.3f} (quote the rq2-protocol number across tables)", flush=True
    )
    # Average each reported statistic across thresholds; mean F1 is not F1 recomputed from mean P/R.
    rq2_protocol["avg"] = {k: float(np.mean([r["segmentation"][k] for r in rq2_rows])) for k in ("precision", "recall", "f1")}
    print("[segmenter-eval] rq2-protocol avg: " + " ".join(f"{k}={v:.4f}" for k, v in rq2_protocol["avg"].items()), flush=True)
    payload = {
        "language": args.language, "split": args.split, "segmenter_arch": args.segmenter_arch, "videos": len({r.pose.video_id for r in records}), 
        "streams": len(records), "metrics_mean_over": "streams", "rq2_protocol_mean_over": "videos", "checkpoint": checkpoint, 
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(records), 
        "frame_metrics_decode": "raw_argmax", "segmentation_decode": "semi_markov_viterbi" if duration else "bio_argmax", 
        "decode": decode, "duration_model": duration.to_dict() if duration else None, "tiou_thresholds": list(thresholds), 
        "metrics": metrics, "rq2_protocol": rq2_protocol, "segmenter_init": getattr(args, "segmenter_init", "finetuned"),
    }
    if payload["segmenter_init"] != "finetuned": payload["seed"] = int(args.seed)
    init = "" if payload["segmenter_init"] == "finetuned" \
              else f"_{payload['segmenter_init']}{args.seed if payload['segmenter_init'] == 'random' else ''}"
    output = Path(args.output) if args.output else \
             output_dir("segmenter") / f"segmenter_eval_{args.segmenter_arch}{init}_{args.language}_{args.split}.json"
    if init and init not in output.name: raise SystemExit(
        f"--segmenter-init {payload['segmenter_init']} must not overwrite a fine-tuned artifact: {output.name} "
        f"carries no '{init}' token. Drop --output, or put the token in the name."
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["output"] = str(output)
    return payload


@torch.no_grad()
def segmenter_dev_scores(
    model, records: list[VideoRecord], device: torch.device, velocity: bool, 
    rope_chunk_s: float | None, priors: dict[str, DurationModel], thresholds,
) -> dict[str, float]:
    """Segmenter checkpoint selection on RQ2 protocol (train/bio_pretrain.py, moryossef26/trainer.py): the whole-video pass and duration decode 
    of `--segmenter-eval`, each language under its own train-fitted prior (`priors`), scored as RQ2 segmentation (F1 averaged over `thresholds`). 
    `video_f1` is the mean over languages, so a pooled run weights every language equally. Each rank decodes every world_size-th record and the 
    spans are merged, so every rank returns the same scores. The decode switches the model to eval mode; its mode is restored."""
    by_language: dict[str, list[VideoRecord]] = {}
    for record in records: by_language.setdefault(str(record.language), []).append(record)
    if not by_language: raise ValueError("The segmenter dev monitor needs at least 1 dev record.")
    was_training, scores = model.training, {}
    # Full precision, as `--segmenter-eval` decodes: the training loop calls its monitor under CUDA autocast, and bf16/fp16
    # logits can move a duration-Viterbi boundary. Autocast exists only on CUDA here (train.helpers.AmpHelper).
    full_precision = torch.autocast(device_type="cuda", enabled=False) if torch.device(device).type == "cuda" else nullcontext()
    try:
        with full_precision:
            for language, recs in sorted(by_language.items()):
                local = predict_phrase_segments(
                    model, recs[dist.rank()::dist.world_size()], device, velocity=velocity, 
                    rope_chunk_s=rope_chunk_s, duration=priors[language],
                )
                rows = _segment_rq2_rows(dist.merge_dicts(local), recs, thresholds, tag=f"val-video {language}")
                scores[f"video_f1_{language}"] = float(np.mean([row["segmentation"]["f1"] for row in rows]))
    finally: model.train(was_training)
    scores["video_f1"] = float(np.mean([scores[f"video_f1_{language}"] for language in by_language]))
    return scores
