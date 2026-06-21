from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.loader import load_language_records
from data.clean import CleanSentenceCollator, CleanSentenceDataset
from models.gfslt import make_gfslt_config, CleanARSLTModel, resolve_decoder_start_id
from models.checkpointing import load_visual_backbone_checked, save_model_checkpoint
from poses import pose_repr_for_backbone, build_pose_augmentor

from train.helpers import AmpHelper, TrainControl, TrainLogger, attach_save_best, build_scheduler, mean_logs
from metrics import compute_text_metrics, token_accuracy
from utils import load_yaml, mbart_trimmed_dir, vlp_checkpoint, backbone_name


@dataclass
class BaselineComponents:
    model: CleanARSLTModel
    tokenizer: Any
    train_loader: DataLoader
    dev_loader: DataLoader | None = None


def build_baseline_components(
    data_config: str = "configs/data.yaml",
    stage1_config: str = "configs/stage1_vlp.yaml",
    baseline_config: str = "configs/stage2_baseline.yaml",
    max_items: int | None = None, include_dev: bool = False,
) -> BaselineComponents:
    data_cfg = load_yaml(data_config)
    stage1_cfg = load_yaml(stage1_config)
    base_cfg = load_yaml(baseline_config)
    language = str(base_cfg.get("language", data_cfg.get("active_languages", ["phoenix"])[0]))
    target_lang = data_cfg["languages"][language].get("target_lang", "en_XX")

    tokenizer_dir = mbart_trimmed_dir(stage1_cfg)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, src_lang=target_lang, tgt_lang=target_lang)
    pose_repr = pose_repr_for_backbone(backbone_name(stage1_cfg))

    # Train-only pose augmentation (the baseline config's `augmentation:` block; falls back to stage-1's).
    aug_cfg = base_cfg.get("augmentation", stage1_cfg.get("augmentation"))
    augmentor = build_pose_augmentor(aug_cfg, np.random.default_rng(int(base_cfg.get("seed", 42)) + 991))
    records, _ = load_language_records(data_cfg, language, split="train")
    dataset = CleanSentenceDataset(records, max_items=max_items, pose_repr=pose_repr, augment=augmentor)
    collator = CleanSentenceCollator(
        tokenizer, max_text_tokens=int(base_cfg.get("max_text_tokens", 128)),
        visual_padding=str(base_cfg.get("visual_padding", stage1_cfg.get("visual_padding", "gfslt"))),
    )
    loader = DataLoader(
        dataset, batch_size=int(base_cfg.get("batch_size", 8)),
        shuffle=True, num_workers=0, collate_fn=collator,
    )
    dev_loader = None
    if include_dev:
        dev_records, _ = load_language_records(data_cfg, language, split="dev")
        dev_dataset = CleanSentenceDataset(dev_records, max_items=max_items, pose_repr=pose_repr)
        dev_loader = DataLoader(
            dev_dataset, batch_size=int(base_cfg.get("eval_batch_size", base_cfg.get("batch_size", 8))),
            shuffle=False, num_workers=0, collate_fn=collator,
        )
    model = CleanARSLTModel(
        make_gfslt_config(stage1_cfg),  # backbone inherited from the stage-1 VLP config
        decoder_start_token_id=resolve_decoder_start_id(tokenizer),
        label_smoothing=float(base_cfg.get("label_smoothing", 0.0)),
    )
    checkpoint = vlp_checkpoint(base_cfg)
    if checkpoint: load_visual_backbone_checked(model.visual, str(checkpoint), name="baseline", preserve_decoder_io=True)
    else: print("baseline | WARNING: no checkpoint.from_vlp set; visual backbone trains from scratch.", flush=True)

    if bool(base_cfg.get("freeze_backbone", False)):
        n = model.visual.freeze_pose_backbone(freeze_projection=bool(base_cfg.get("freeze_projection", False)))
        print(f"baseline | froze pose backbone ({n / 1e6:.2f}M params; spec §4.1)", flush=True)
    return BaselineComponents(model=model, tokenizer=tokenizer, train_loader=loader, dev_loader=dev_loader)


@torch.no_grad()
def evaluate_baseline(
    model: CleanARSLTModel, loader: DataLoader,
    device: torch.device, tokenizer=None, cfg: dict | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    cfg = cfg or {}
    validation_cfg = cfg.get("validation", {})
    # <= 0 means generate/score translation on ALL dev sentences (default).
    max_translation_samples = int(validation_cfg.get("max_translation_samples", 0) or 0)
    rows: list[dict[str, float]] = []
    pred_texts: list[str] = []
    ref_texts: list[str] = []
    for batch in loader:
        tokens = batch["text_tokens"]
        out = model(
            poses=batch["poses"].to(device),
            frame_mask=batch["frame_mask"].to(device),
            timestamps_s=batch["timestamps_s"].to(device),
            labels=tokens["labels"].to(device),
        )
        row = {"loss": float(out.loss.detach().cpu().item())}
        row.update(token_accuracy(out.logits.detach().cpu(), tokens["labels"], prefix="translation"))
        rows.append(row)

        cap_reached = max_translation_samples > 0 and len(pred_texts) >= max_translation_samples
        if tokenizer is not None and not cap_reached:
            take = batch["poses"].shape[0]
            if max_translation_samples > 0: take = min(max_translation_samples - len(pred_texts), take)
            generated = model.generate(
                poses=batch["poses"][:take].to(device),
                frame_mask=batch["frame_mask"][:take].to(device),
                timestamps_s=batch["timestamps_s"][:take].to(device),
                max_new_tokens=int(validation_cfg.get("max_text_tokens", cfg.get("max_text_tokens", 128))),
                num_beams=int(validation_cfg.get("num_beams", 4)),  # GFSLT-VLP dev BLEU is beam-4 (train_slt.py)
            )
            pred_texts.extend(tokenizer.batch_decode(generated.detach().cpu(), skip_special_tokens=True))
            ref_texts.extend([str(text) for text in batch["texts"][:take]])

    if was_training: model.train()
    metrics = mean_logs(rows, prefix="val")
    if pred_texts:
        metrics.update(compute_text_metrics(pred_texts, ref_texts, prefix="val_translation"))
        # metrics["val_translation_samples"] = float(len(pred_texts))
    return metrics


def train_baseline_epochs(
    model: CleanARSLTModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device,
    epochs: int, cfg: dict | None = None, dev_loader: DataLoader | None = None, tokenizer=None,
) -> list[dict[str, float]]:
    model.to(device)
    model.train()
    logs: list[dict[str, float]] = []
    cfg = cfg or {}

    scheduler = build_scheduler(optimizer, cfg, epochs=epochs, steps_per_epoch=len(loader))
    amp = AmpHelper.from_config(cfg, device)
    control = TrainControl.from_config(cfg, default_monitor="val_loss", default_mode="min")
    attach_save_best(control, cfg, "baseline", save_model_checkpoint)
    logger = TrainLogger("baseline", cfg, epochs=int(epochs), steps_per_epoch=len(loader), monitor=control.monitor)
    
    for epoch in range(1, int(epochs) + 1):
        epoch_logs: list[dict[str, float]] = []
        for step, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            tokens = batch["text_tokens"]
            with amp.autocast():
                out = model(
                    poses=batch["poses"].to(device),
                    frame_mask=batch["frame_mask"].to(device),
                    timestamps_s=batch["timestamps_s"].to(device),
                    labels=tokens["labels"].to(device),
                )
            amp.backward(out.loss)
            amp.clip_and_step(optimizer, model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
            scheduler.step_batch()
            row = {
                "epoch": float(epoch), "step": float(step),
                "baseline_ce_loss": float(out.loss.detach().cpu().item()),
                "lr": scheduler.lr(optimizer),
            }
            epoch_logs.append(row)
            logs.append(row)
            logger.log_step(epoch, step, row)

        scheduler.step_epoch()
        train_means = mean_logs(epoch_logs)
        if dev_loader is not None and control.should_eval(epoch, epochs):
            metrics = evaluate_baseline(model, dev_loader, device, tokenizer=tokenizer, cfg=cfg)
            improved = control.update(model, metrics, epoch)
            logger.epoch_summary(epoch, train=train_means, val=metrics, is_best=improved, saved_path=control.last_saved_path)
            logs.append({"epoch": float(epoch), **train_means, **metrics, **control.summary()})
            if control.stopped_early:
                print(f"baseline | early stop at epoch {epoch} (best {control.monitor}={control.best_value})", flush=True)
                break
        else: logger.epoch_summary(epoch, train=train_means)

    control.restore(model)
    logger.finish()
    return logs
