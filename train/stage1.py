from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.clean import CleanSentenceCollator, CleanSentenceDataset
from data.loader import load_language_records
from models.gfslt import GFSLTConfig
from models.vlp import PoseTextCLIP
from models.checkpointing import save_visual_backbone
from train.helpers import AmpHelper, TrainControl, TrainLogger, attach_save_best, build_scheduler, mean_logs
from utils import load_yaml, mbart_trimmed_dir


@dataclass
class Stage1Components:
    model: PoseTextCLIP
    tokenizer: Any
    train_loader: DataLoader
    dev_loader: DataLoader | None = None


def build_stage1_components(
    data_config: str = "configs/data.yaml", stage1_config: str = "configs/stage1_vlp.yaml",
    max_items: int | None = None, include_dev: bool = False,
) -> Stage1Components:
    data_cfg = load_yaml(data_config)
    cfg = load_yaml(stage1_config)
    language = str(cfg.get("language", data_cfg.get("active_languages", ["phoenix"])[0]))
    target_lang = data_cfg["languages"][language].get("target_lang", "en_XX")
    tokenizer_dir = mbart_trimmed_dir(cfg)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, src_lang=target_lang, tgt_lang=target_lang)

    records, _ = load_language_records(data_cfg, language, split="train")
    dataset = CleanSentenceDataset(records, max_items=max_items)
    cmlm_cfg = cfg.get("cmlm", {})
    cmlm_enabled = bool(cmlm_cfg.get("enabled", True))

    collator = CleanSentenceCollator(
        tokenizer=tokenizer,
        max_text_tokens=int(cfg.get("max_text_tokens", 128)),
        visual_padding=str(cfg.get("visual_padding", "gfslt")),
        cmlm=cmlm_enabled,
        noise_rate=float(cmlm_cfg.get("noise_rate", 0.15)),
        noise_type=str(cmlm_cfg.get("noise_type", "omit_last")),
        random_shuffle=bool(cmlm_cfg.get("random_shuffle", False)),
        seed=int(cfg.get("seed", 42)),
    )
    loader = DataLoader(
        dataset, batch_size=int(cfg.get("batch_size", 16)),
        shuffle=True, num_workers=0, collate_fn=collator,
    )
    dev_loader = None
    if include_dev:
        dev_records, _ = load_language_records(data_cfg, language, split="dev")
        dev_dataset = CleanSentenceDataset(dev_records, max_items=max_items)
        dev_collator = CleanSentenceCollator(
            tokenizer=tokenizer,
            max_text_tokens=int(cfg.get("max_text_tokens", 128)),
            visual_padding=str(cfg.get("visual_padding", "gfslt")),
            cmlm=cmlm_enabled,
            noise_rate=float(cmlm_cfg.get("noise_rate", 0.15)),
            noise_type=str(cmlm_cfg.get("noise_type", "omit_last")),
            random_shuffle=bool(cmlm_cfg.get("random_shuffle", False)),
            seed=int(cfg.get("seed", 42)) + 1,
        )
        dev_loader = DataLoader(
            dev_dataset, batch_size=int(cfg.get("eval_batch_size", cfg.get("batch_size", 16))),
            shuffle=False, num_workers=0, collate_fn=dev_collator,
        )
    gfslt_cfg = GFSLTConfig(
        embed_dim=int(cfg.get("embed_dim", 1024)),
        hidden_size=int(cfg.get("hidden_size", 1024)),
        temporal_kernel=int(cfg.get("temporal_kernel", 3)),
        mbart_name=mbart_trimmed_dir(cfg),  # one trimmed mBART for text encoder + visual side
        use_temporal_conv=bool(cfg.get("use_temporal_conv", False)),
    )
    model = PoseTextCLIP(
        gfslt_cfg, projection_dim=int(cfg.get("embed_dim", 1024)),
        cmlm_lambda=float(cmlm_cfg.get("lambda", 1.0)) if cmlm_enabled else 0.0,
        cmlm_label_smoothing=float(cmlm_cfg.get("label_smoothing", 0.2)),
    )
    return Stage1Components(model=model, tokenizer=tokenizer, train_loader=loader, dev_loader=dev_loader)


@torch.no_grad()
def evaluate_stage1(model: PoseTextCLIP, loader: DataLoader, device: torch.device) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    image_correct, text_correct, total = 0, 0, 0

    for batch in loader:
        poses = batch["poses"].to(device)
        frame_mask = batch["frame_mask"].to(device)
        timestamps = batch["timestamps_s"].to(device)
        tokens = batch["text_tokens"]
        masked = batch.get("masked_text_tokens")
        out = model(
            poses=poses, frame_mask=frame_mask, timestamps_s=timestamps,
            text_input_ids=tokens["input_ids"].to(device),
            text_attention_mask=tokens["attention_mask"].to(device),
            masked_text_input_ids=masked["input_ids"].to(device) if masked is not None else None,
            masked_text_attention_mask=masked["attention_mask"].to(device) if masked is not None else None,
        )
        losses.append(float(out.loss.detach().cpu().item()))
        labels = torch.arange(out.logits_per_image.shape[0], device=device)
        image_correct += int((out.logits_per_image.argmax(dim=1) == labels).sum().item())
        text_correct += int((out.logits_per_text.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())

    if was_training: model.train()
    loss = sum(losses) / max(1, len(losses))
    return {
        "val_loss": loss,
        "val_image_retrieval_acc": image_correct / max(1, total),
        "val_text_retrieval_acc": text_correct / max(1, total),
    }


def train_stage1_epochs(
    model: PoseTextCLIP, loader: DataLoader, optimizer: torch.optim.Optimizer,
    device: torch.device, epochs: int, cfg: dict | None = None, dev_loader: DataLoader | None = None,
) -> list[dict[str, float]]:
    model.to(device)
    model.train()
    logs: list[dict[str, float]] = []
    cfg = cfg or {}

    scheduler = build_scheduler(optimizer, cfg, epochs=epochs, steps_per_epoch=len(loader))
    amp = AmpHelper.from_config(cfg, device)
    control = TrainControl.from_config(cfg, default_monitor="val_loss", default_mode="min")
    # Save-on-best writes the VLP deliverable (visual backbone — what stage 2 / the baseline load).
    attach_save_best(control, cfg, "stage1-vlp", lambda model, out_dir: save_visual_backbone(model.visual, out_dir))
    logger = TrainLogger("stage1-vlp", cfg, epochs=int(epochs), steps_per_epoch=len(loader), monitor=control.monitor)
    
    for epoch in range(1, int(epochs) + 1):
        epoch_logs: list[dict[str, float]] = []
        for step, batch in enumerate(loader, start=1):
            poses = batch["poses"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            timestamps = batch["timestamps_s"].to(device)
            tokens = batch["text_tokens"]
            masked = batch.get("masked_text_tokens")
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast():
                out = model(
                    poses=poses, frame_mask=frame_mask, timestamps_s=timestamps,
                    text_input_ids=tokens["input_ids"].to(device),
                    text_attention_mask=tokens["attention_mask"].to(device),
                    masked_text_input_ids=masked["input_ids"].to(device) if masked is not None else None,
                    masked_text_attention_mask=masked["attention_mask"].to(device) if masked is not None else None,
                )
            amp.backward(out.loss)
            amp.clip_and_step(optimizer, model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
            scheduler.step_batch()

            row = {
                "epoch": float(epoch), "step": float(step),
                "vlp_loss": float(out.loss.detach().cpu().item()),
                "contrastive_loss": float(out.contrastive_loss.detach().cpu().item())
                if out.contrastive_loss is not None else float("nan"),
                "cmlm_loss": float(out.cmlm_loss.detach().cpu().item()) if out.cmlm_loss is not None else float("nan"),
                "lr": scheduler.lr(optimizer),
            }
            epoch_logs.append(row)
            logs.append(row)
            logger.log_step(epoch, step, row)

        scheduler.step_epoch()
        train_means = mean_logs(epoch_logs)
        if dev_loader is not None and control.should_eval(epoch, epochs):
            metrics = evaluate_stage1(model, dev_loader, device)
            improved = control.update(model, metrics, epoch)
            logger.epoch_summary(epoch, train=train_means, val=metrics, is_best=improved, saved_path=control.last_saved_path)
            logs.append({"epoch": float(epoch), **train_means, **metrics, **control.summary()})
            if control.stopped_early:
                print(f"stage1-vlp | early stop at epoch {epoch} (best {control.monitor}={control.best_value})", flush=True)
                break
        else: logger.epoch_summary(epoch, train=train_means)

    control.restore(model)
    logger.finish()
    return logs
