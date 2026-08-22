from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import json

import torch
from data.batch import collate_windows
from data.loader import load_language_records
from data.windowing import BIO

from models.checkpointing import save_model_checkpoint
from train import distributed as dist
from train.helpers import build_optimizer
from train.sampler import WindowSampler
from utils import checkpoint_dir, load_yaml, pick_device


def smoke_data(args: argparse.Namespace) -> dict:
    data_cfg = load_yaml(args.data_config)
    slt_cfg = load_yaml(args.slt_config)
    inference_cfg = load_yaml(args.inference_config)
    language = str(args.language or data_cfg.get("active_languages", ["asf"])[0])
    if language != slt_cfg.get("language"): slt_cfg = load_yaml(args.slt_config, language=language)
    records, splits = load_language_records(data_cfg, language, split=args.split)
    if not records: raise RuntimeError(f"No records loaded for language={language} split={args.split}")

    sampler = WindowSampler.from_slt_config(records, slt_cfg, inference_cfg)
    samples = [sampler.to_dict(sampler.sample(i)) for i in range(args.num_samples)]
    batch = collate_windows(samples)
    padded_labels = batch["bio_labels"][~batch["frame_mask"]]
    if padded_labels.numel() and not torch.all(padded_labels == BIO["UNK"]): raise AssertionError("Padding labels must be UNK, never O")

    modes = Counter(sample["spec"]["mode"] for sample in samples)
    mode2 = Counter(sample["spec"].get("subcase") for sample in samples if sample["spec"]["mode"] == "mode2")
    translated = sum(sample["translation_target"] is not None for sample in samples)
    return {
        "language": language, "split": args.split, "records": len(records), 
        "split_sizes": {k: len(v) for k, v in splits.items()}, "samples": args.num_samples, 
        "modes": dict(modes), "mode2_subcases": dict(mode2), "translation_targets": translated, 
        "batch_pose_shape": list(batch["poses"].shape), "padding_label": "UNK",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Misaligned-SLT training utilities")
    parser.add_argument("--stage", default="smoke-data", choices=["smoke-data", "train-slt", "train-bio", "train-moryossef"])
    parser.add_argument("--bio-config", default="configs/bio_pretrain.yaml")
    parser.add_argument("--moryossef-config", default="configs/moryossef26.yaml")
    parser.add_argument("--language", default=None, 
                        help="Override active language; default falls back to the config's language / data.yaml active_languages")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="continue from <checkpoint.dir>/latest.pt (full optimizer/scheduler state, epoch granularity)")
    parser.add_argument("--decoder", default=None, choices=["ar", "dlm"])
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--slt-config", default="configs/dlm.yaml")
    parser.add_argument("--baseline-config", default="configs/baseline_eval.yaml")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--device", default=None, help="override device; default cuda -> mps -> cpu")
    parser.add_argument("--output", default=None)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    # torchrun sets RANK/WORLD_SIZE/LOCAL_RANK; without them this is a no-op and everything runs as before.
    dist_device = dist.init_distributed()
    if args.stage == "smoke-data": result = smoke_data(args)
    elif args.stage == "train-bio":
        from train.bio_pretrain import build_bio_s1, train_bio_s1_epochs
        model, train_loader, dev_loader, cfg = build_bio_s1(
            data_config=args.data_config, config=args.bio_config,
            inference_config=args.inference_config, language=args.language,
        )
        device = dist_device or pick_device(args.device)
        epochs = int(args.epochs or cfg.get("epochs", 40))
        logs = train_bio_s1_epochs(model, train_loader, dev_loader, device, epochs=epochs, cfg=cfg, resume=args.resume)
        # Same meta as the save-on-best write (train/bio_pretrain.py), so both copies describe the same context.
        _inf = load_yaml(str(cfg.get("inference_config", "configs/inference.yaml")))
        _meta = {"rope_eval_chunk_s": float(cfg.get("rope_eval_chunk_s") or _inf.get("buffer_cap_s", 18.0)),
                 "buffer_cap_s": float(_inf.get("buffer_cap_s", 18.0)),
                 "bio_class_weights": cfg.get("bio_class_weights"), "language": cfg.get("language")}
        path = save_model_checkpoint(model, checkpoint_dir(cfg, default="checkpoints/bio_s1"), meta=_meta) if dist.is_main() else None
        result = {"stage": args.stage, "device": str(device), "checkpoint": str(path), "epochs": epochs, "log_rows": len(logs)}
    elif args.stage == "train-moryossef":
        # Faithful Moryossef external segmenter for error calibration + RQ2 cascade: raw keypoints + UNet.
        # from the FSM head. Standalone on whole-video chunks → checkpoints/moryossef, never bio_head_init.
        from moryossef26.trainer import build_segmenter, build_segmenter_loaders, train_segmenter_epochs
        train_loader, dev_loader, cfg = build_segmenter_loaders(args.data_config, args.moryossef_config, language=args.language)
        model = build_segmenter(args.moryossef_config)
        device = dist_device or pick_device(args.device)
        epochs = int(args.epochs or cfg.get("epochs", 50))
        logs = train_segmenter_epochs(model, train_loader, dev_loader, device, epochs=epochs, cfg=cfg, resume=args.resume)
        path = save_model_checkpoint(model, checkpoint_dir(cfg, default="checkpoints/moryossef")) if dist.is_main() else None
        result = {"stage": args.stage, "device": str(device), "checkpoint": str(path), "epochs": epochs, "log_rows": len(logs)}
    elif args.stage == "train-slt":
        from train.slt import build_slt_components, train_slt_epochs
        # --language re-points ${language} in checkpoint.dir for training and the save below.
        components = build_slt_components(
            data_config=args.data_config, slt_config=args.slt_config, inference_config=args.inference_config,
            decoder=args.decoder, include_dev=True, language=args.language,
        )
        slt_cfg = components.slt_cfg  # Carry corpus-measured values that a 2nd load_yaml would leave unresolved.
        epochs = int(args.epochs or slt_cfg.get("epochs", 1))
        device = dist_device or pick_device(args.device)
        # Skip frozen params; build_optimizer reads learning_rate/weight_decay (same keys every stage)
        optimizer = build_optimizer(slt_cfg, [p for p in components.model.parameters() if p.requires_grad])
        logs = train_slt_epochs(
            components.model, components.train_loader, optimizer, device=device, epochs=epochs, slt_cfg=slt_cfg, 
            dev_loader=components.dev_loader, resume=args.resume, checkpoint_meta=components.checkpoint_meta,
        )
        path = save_model_checkpoint(
            components.model, checkpoint_dir(slt_cfg, default="checkpoints/slt"), meta=components.checkpoint_meta,
        ) if dist.is_main() else None
        result = {"stage": args.stage, "device": str(device), "checkpoint": str(path), "epochs": epochs, "log_rows": len(logs)}
    else: raise ValueError(f"Unsupported stage: {args.stage}")

    if dist.is_main():
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(text + "\n", encoding="utf-8")
    dist.barrier()   # non-zero ranks must not exit before rank 0 finishes writing
    dist.cleanup()
