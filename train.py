from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import json

import torch
from data.batch import collate_windows
from data.loader import load_language_records
from data.windowing import BIO

from models.checkpointing import save_model_checkpoint, save_visual_backbone
from train.sampler import WindowSampler
from utils import load_yaml


def smoke_data(args: argparse.Namespace) -> dict:
    data_cfg = load_yaml(args.data_config)
    stage2_cfg = load_yaml(args.stage2_config)
    inference_cfg = load_yaml(args.inference_config)
    records, splits = load_language_records(data_cfg, args.language, split=args.split)
    if not records: raise RuntimeError(f"No records loaded for language={args.language} split={args.split}")

    sampler = WindowSampler.from_stage2_config(records, stage2_cfg, inference_cfg)
    samples = [sampler.to_dict(sampler.sample()) for _ in range(args.num_samples)]
    batch = collate_windows(samples)
    padded_labels = batch["bio_labels"][~batch["frame_mask"]]
    if padded_labels.numel() and not torch.all(padded_labels == BIO["UNK"]): raise AssertionError("Padding labels must be UNK, never O")

    modes = Counter(sample["spec"]["mode"] for sample in samples)
    mode2 = Counter(sample["spec"].get("subcase") for sample in samples if sample["spec"]["mode"] == "mode2")
    translated = sum(sample["translation_target"] is not None for sample in samples)
    return {
        "language": args.language, "split": args.split,
        "records": len(records), "split_sizes": {k: len(v) for k, v in splits.items()},
        "samples": args.num_samples, "modes": dict(modes), "mode2_subcases": dict(mode2),
        "translation_targets": translated, "batch_pose_shape": list(batch["poses"].shape), "padding_label": "UNK",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Misaligned-SLT training utilities")
    parser.add_argument(
        "--stage", default="smoke-data",
        choices=["trim-mbart", "smoke-data", "train-stage1", "train-baseline", "train-stage2", "train-segmenter"],
    )
    parser.add_argument("--language", default="asf")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--decoder", default=None, choices=["ar", "dlm"])
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--stage1-config", default="configs/stage1_vlp.yaml")
    parser.add_argument("--stage2-config", default="configs/stage2_dlm.yaml")
    parser.add_argument("--baseline-config", default="configs/stage2_baseline.yaml")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--segmenter-config", default="configs/segmenter.yaml")
    parser.add_argument("--output", default=None)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.stage == "trim-mbart":
        from trim_mbart import trim_MBartForConditionalGeneration
        data_cfg = load_yaml(args.data_config)
        stage1_cfg = load_yaml(args.stage1_config)
        language = str(args.language or stage1_cfg.get("language", data_cfg.get("active_languages", ["asf"])[0]))
        target_lang = data_cfg["languages"][language].get("target_lang", "en_XX")
        result = trim_MBartForConditionalGeneration(
            data_config=args.data_config, language=language,
            mbart_name=str(stage1_cfg.get("mbart_name", "facebook/mbart-large-cc25")), target_lang=target_lang,
            tokenizer_out=str(stage1_cfg.get("trimmed_tokenizer_dir", f"captioners/trimmed_mbart_{language}")),
            model_out=str(stage1_cfg.get("trimmed_mbart_dir", f"captioners/trimmed_mbart_{language}")),
        )
    elif args.stage == "smoke-data": result = smoke_data(args)

    elif args.stage == "train-stage1":
        from train.stage1 import build_stage1_components, train_stage1_epochs
        stage1_cfg = load_yaml(args.stage1_config)
        epochs = int(args.epochs or stage1_cfg.get("epochs", 1))
        components = build_stage1_components(
            data_config=args.data_config, stage1_config=args.stage1_config,
            max_items=args.num_samples if args.num_samples > 0 else None, include_dev=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.AdamW(
            components.model.parameters(), lr=float(stage1_cfg.get("learning_rate", 5e-4)),
            weight_decay=float(stage1_cfg.get("weight_decay", 1e-4)),
        )
        logs = train_stage1_epochs(
            components.model, components.train_loader, optimizer, device=device, 
            epochs=epochs, cfg=stage1_cfg, dev_loader=components.dev_loader,
        )
        output_dir = Path(stage1_cfg.get("output_dir", f"checkpoints/vlp/{stage1_cfg.get('language', args.language)}"))
        path = save_visual_backbone(components.model.visual, output_dir)
        print({"device": str(device), "visual_backbone": str(path), "logs": logs})

    elif args.stage == "train-baseline":
        from train.baseline import build_baseline_components, train_baseline_epochs
        base_cfg = load_yaml(args.baseline_config)
        epochs = int(args.epochs or base_cfg.get("epochs", 1))
        components = build_baseline_components(
            data_config=args.data_config, stage1_config=args.stage1_config, baseline_config=args.baseline_config,
            max_items=args.num_samples if args.num_samples > 0 else None, include_dev=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.AdamW(
            components.model.parameters(), lr=float(base_cfg.get("learning_rate", 5e-4)),
            weight_decay=float(base_cfg.get("weight_decay", 1e-4)),
        )
        logs = train_baseline_epochs(
            components.model, components.train_loader, optimizer, device=device, epochs=epochs, 
            cfg=base_cfg, dev_loader=components.dev_loader, tokenizer=components.tokenizer,
        )
        path = save_model_checkpoint(components.model, base_cfg.get("output_dir", "checkpoints/stage2_baseline"))
        print({"device": str(device), "checkpoint": str(path), "logs": logs})

    elif args.stage == "train-stage2":
        from train.stage2 import build_stage2_components, train_stage2_epochs
        stage2_cfg = load_yaml(args.stage2_config)
        segmenter_cfg = load_yaml(args.segmenter_config)
        epochs = int(args.epochs or stage2_cfg.get("epochs", 1))
        components = build_stage2_components(
            data_config=args.data_config,
            stage1_config=args.stage1_config,
            stage2_config=args.stage2_config,
            inference_config=args.inference_config,
            decoder=args.decoder,
            include_dev=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.AdamW(
            components.model.parameters(), lr=float(stage2_cfg.get("learning_rate", 3e-5)),
            weight_decay=float(stage2_cfg.get("weight_decay", 1e-4)),
        )
        logs = train_stage2_epochs(
            components.model, components.train_loader, optimizer, device=device, epochs=epochs,
            stage2_cfg=stage2_cfg, segmenter_cfg=segmenter_cfg, dev_loader=components.dev_loader,
        )
        path = save_model_checkpoint(components.model, stage2_cfg.get("output_dir", "checkpoints/stage2"))
        print({"device": str(device), "checkpoint": str(path), "logs": logs})

    elif args.stage == "train-segmenter":
        from moryossef26.trainer import build_segmenter, build_segmenter_loader, train_segmenter_epochs
        segmenter_cfg = load_yaml(args.segmenter_config)
        opt_cfg = segmenter_cfg.get("optimizer", {})
        epochs = int(args.epochs or opt_cfg.get("epochs", 1))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loader = build_segmenter_loader(data_config=args.data_config, segmenter_config=args.segmenter_config, split=args.split)
        dev_loader = build_segmenter_loader(data_config=args.data_config, segmenter_config=args.segmenter_config, split="dev")
        model = build_segmenter(args.segmenter_config)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(opt_cfg.get("lr", 1e-3)),
            weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
        )
        output = train_segmenter_epochs(
            model, loader, optimizer, device=device, epochs=epochs,
            dice_weight=float(segmenter_cfg.get("dice_loss_weight", 1.5)),
            cfg=segmenter_cfg, dev_loader=dev_loader,
        )
        path = save_model_checkpoint(model, segmenter_cfg.get("output_dir", "checkpoints/segmenter"))
        print({"device": str(device), "checkpoint": str(path), "logs": output.logs})
    else: raise ValueError(f"Unsupported stage: {args.stage}")

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
