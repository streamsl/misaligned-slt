"""Segmenter evaluation (eval.py --segmenter-eval) and the segmenter dev monitor.

The whole-video pass and duration decode of a segmenter, scored with the segmentation scores of the rows (the edge chunk F1, the main score,
and the sentence tIoU F1); the monitor (train/helpers.py) runs the same pass and the edge chunk F1 on all dev records of each
language for checkpoint selection.
"""
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
from evaluation.predictions import corpus_token, gold_events, gold_chunks, segment_events
from metrics import edge_chunk_scores, evaluate_predicted_events
from evaluation.systems import load_segmenter, source_language
from models.checkpointing import load_checkpoint_meta
from utils import lambda_min_s, load_yaml, output_dir


def run_segmenter_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Whole-video segmentation eval: the Moryossef evaluate.py protocol (frame F1 + 1-to-1 tIoU segment P/R/F1 per stream; the acceptance
    check "F1 within order of magnitude of published"), and the segmentation scores of the rows on the same spans (the edge chunk F1, the main
    score, and the sentence tIoU F1), the same protocol for either --segmenter-arch. For `s1` it also scores the pretrained head before
    stage-2 fine-tuning.
    """
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run segmenter eval on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    source = source_language(args)   # the segmenter's corpus: its stamps and its duration prior; the records are --language's
    if args.output and source != args.language and corpus_token(args.language, source) not in Path(args.output).name: raise SystemExit(
        f"A zero-shot segmenter eval ({source} segmenter on {args.language}): put {corpus_token(args.language, source)!r} in --output "
        f"({args.output}), so report.py never reads it as an in-corpus one."
    )
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    model, device, velocity, rope_chunk_s, checkpoint = load_segmenter(args)
    decode = args.segmenter_decode or "duration"
    duration = DurationModel.for_language(data_cfg, source) if decode == "duration" else None
    print(f"[segmenter-eval] {args.segmenter_arch} segmenter from {checkpoint} (decode={decode})", flush=True)

    scoring_cfg, min_span_s = load_yaml(args.eval_config)["scoring"], lambda_min_s(load_yaml(args.inference_config))
    thresholds = tuple(float(t) for t in scoring_cfg["tiou_thresholds"])
    taus = [float(t) for t in scoring_cfg["edge_taus_s"]]
    metrics, segments_by_video = evaluate_moryossef_whole_video(
        model, records, device=device, velocity=velocity, rope_chunk_s=rope_chunk_s, 
        tiou_thresholds=thresholds, return_segments=True, duration=duration, min_span_s=min_span_s,
    )
    # 2 protocols over 1 decode. `metrics` = the Moryossef-comparable protocol (per-stream mean of F1s; gold = the UNK-masked BIO label
    # stream), for comparison with Moryossef's published numbers. `edge_chunk` and `tiou` = the SAME spans scored by the code path of the rows
    # (caption-unit gold per source video, quarantine-dropped events): the edge chunk F1 (the main score, which the segmenter monitor
    # reads) and the sentence tIoU F1 (F1 of macro-averaged P/R) — quote these wherever localization is compared across tables.
    events = segment_events(segments_by_video, records, tag="segmenter-eval")
    edge = edge_chunk_scores(events, gold_chunks(records, float(scoring_cfg["chunk_gap_s"])), taus, char_level=False)
    print(f"[segmenter-eval] MAIN edge chunk F1 avg {edge['f1_avg']:.4f} (tau " + " / ".join(f"{t:g}" for t in taus) + " s: " +
          " / ".join(f"{r['f1']:.4f}" for r in edge["taus"]) + f"; precision {edge['precision_avg']:.4f}, recall {edge['recall_avg']:.4f})", flush=True)
    tiou_rows = evaluate_predicted_events(events, gold_events(records), list(thresholds))["thresholds"]
    tiou = {f"{r['tiou_threshold']:g}": {
        **r["segmentation"], "count_ratio": r["total_predictions"] / max(1, r["total_gold"])
    } for r in tiou_rows}
    for t in thresholds: print(
        f"[segmenter-eval] tIoU {t:g}: moryossef-protocol F1 {metrics.get(f'phrase_tiou_f1@{t:g}', float('nan')):.3f} | "
        f"sentence tIoU F1 {tiou[f'{t:g}']['f1']:.3f}", flush=True
    )
    # Average each reported statistic across thresholds; mean F1 is not F1 recomputed from mean P/R.
    tiou["avg"] = {k: float(np.mean([r["segmentation"][k] for r in tiou_rows])) for k in ("precision", "recall", "f1")}
    print("[segmenter-eval] sentence tIoU F1 avg: " + " ".join(f"{k}={v:.4f}" for k, v in tiou["avg"].items()), flush=True)
    payload = {
        "language": args.language, "source_language": source, "split": args.split, "segmenter_arch": args.segmenter_arch,
        "videos": len({r.pose.video_id for r in records}),
        "streams": len(records), "metrics_mean_over": "streams", "scores_mean_over": "videos", "checkpoint": checkpoint, 
        # The segmenter's pretrain.inputs (None: a file with no stamp). An untrained control has no checkpoint file: `checkpoint` describes its weights.
        "segmenter_inputs": load_checkpoint_meta(checkpoint).get("pretrain_inputs") if getattr(args, "segmenter_init", "finetuned") == "finetuned" else None,
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(records), 
        "frame_metrics_decode": "raw_argmax", "segmentation_decode": "semi_markov_viterbi" if duration else "bio_argmax", 
        "decode": decode, "duration_model": duration.to_dict() if duration else None, "tiou_thresholds": list(thresholds), "edge_taus_s": taus,
        "metrics": metrics, "edge_chunk": edge, "tiou": tiou, "segmenter_init": getattr(args, "segmenter_init", "finetuned"),
        "min_span_s": min_span_s,   # the span floor of the scored spans (complete spans, no buffer cap: moryossef26.infer.span_segments)
    }
    if payload["segmenter_init"] != "finetuned": payload["seed"] = int(args.seed)
    init = "" if payload["segmenter_init"] == "finetuned" \
              else f"_{payload['segmenter_init']}{args.seed if payload['segmenter_init'] == 'random' else ''}"
    output = Path(args.output) if args.output else \
             output_dir("segmenter") / f"segmenter_eval_{args.segmenter_arch}{init}_{corpus_token(args.language, source)}_{args.split}.json"
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
    rope_chunk_s: float | None, priors: dict[str, DurationModel], edge_cfg: dict, min_span_s: float,
) -> dict[str, float]:
    """Segmenter checkpoint selection (train/bio_pretrain.py, moryossef26/trainer.py): the whole-video pass and duration decode of
    `--segmenter-eval`, each language under its own train-fitted prior (`priors`), the span rule of every offline row (complete spans
    of at least `min_span_s`, no buffer cap), scored with the edge chunk F1 avg, the main
    segmentation score (`edge_cfg` = eval.yaml scoring chunk_gap_s and edge_taus_s). 
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
                    rope_chunk_s=rope_chunk_s, duration=priors[language], min_span_s=min_span_s,
                )
                events = segment_events(dist.merge_dicts(local), recs, tag=f"val-video {language}")
                scores[f"video_f1_{language}"] = float(edge_chunk_scores(
                    events, gold_chunks(recs, float(edge_cfg["chunk_gap_s"])), [float(t) for t in edge_cfg["edge_taus_s"]], char_level=False
                )["f1_avg"])
    finally: model.train(was_training)
    scores["video_f1"] = float(np.mean([scores[f"video_f1_{language}"] for language in by_language]))
    return scores
