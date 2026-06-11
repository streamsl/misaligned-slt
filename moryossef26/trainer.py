from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader

from data.windowing import BIO
from data.loader import load_language_records
from moryossef26.dataset import SegmenterChunkDataset, collate_segmenter_chunks
from moryossef26.model import MoryossefSegmenter
from models.checkpointing import save_model_checkpoint

from train.losses import bio_class_weight_tensor, bio_nll_dice_loss
from train.helpers import AmpHelper, TrainControl, TrainLogger, attach_save_best, build_scheduler, mean_logs
from metrics import bio_frame_metrics, moryossef_segment_metrics
from utils import load_yaml


@dataclass
class SegmenterTrainOutput:
    logs: list[dict[str, float]]


def build_segmenter_loader(
    data_config: str = "configs/data.yaml",
    segmenter_config: str = "configs/segmenter.yaml",
    split: str = "train",
) -> DataLoader:
    data_cfg = load_yaml(data_config)
    seg_cfg = load_yaml(segmenter_config)
    language = str(seg_cfg.get("language", data_cfg.get("active_languages", ["asf"])[0]))
    records, _ = load_language_records(data_cfg, language, split=split)
    fps_cfg = seg_cfg.get("fps_aug", {})
    dataset = SegmenterChunkDataset(
        records=records,
        num_frames=int(seg_cfg.get("num_frames", 1024)),
        fps_aug_enabled=bool(fps_cfg.get("enabled", True)),
        fps_aug_min=float(fps_cfg.get("min_fps", 25.0)),
        fps_aug_max=float(fps_cfg.get("max_fps", 50.0)),
        velocity=bool(seg_cfg.get("velocity", True)),
        training=split == "train",
        frame_dropout=float(seg_cfg.get("frame_dropout", 0.15)),
        body_part_dropout=float(seg_cfg.get("body_part_dropout", 0.1)),
        seed=int(seg_cfg.get("seed", 42)),
    )
    return DataLoader(
        dataset, batch_size=int(seg_cfg.get("batch_size", 8)),
        shuffle=False, num_workers=0, collate_fn=collate_segmenter_chunks,
    )


def build_segmenter(segmenter_config: str = "configs/segmenter.yaml") -> MoryossefSegmenter:
    cfg = load_yaml(segmenter_config)
    pose_dim = 6 if bool(cfg.get("velocity", True)) else 3
    return MoryossefSegmenter(
        pose_dims=(77, pose_dim),
        hidden_dim=int(cfg.get("hidden_dim", 384)),
        encoder_depth=int(cfg.get("encoder_depth", 4)),
        attn_nhead=int(cfg.get("attn_nhead", 8)),
        attn_ff_mult=int(cfg.get("attn_ff_mult", 2)),
        attn_dropout=float(cfg.get("attn_dropout", 0.1)),
        num_frames=int(cfg.get("num_frames", 1024)),
    )


@torch.no_grad()
def evaluate_segmenter(
    model: MoryossefSegmenter, loader: DataLoader,
    device: torch.device, dice_weight: float = 1.5, class_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, float]] = []
    for batch in loader:
        poses = batch["poses"].to(device)
        timestamps = batch["timestamps_s"].to(device)
        labels = batch["phrase_bio"].to(device)
        outputs = model(poses, timestamps_s=timestamps)
        loss = bio_nll_dice_loss(outputs["phrase"], labels, dice_weight=dice_weight, class_weights=class_weights)
        row = {
            "loss": float(loss.detach().cpu().item()),
            **bio_frame_metrics(outputs["phrase"], labels),
            # decode="bio": B-required segments, identical to inference — so an all-I collapse
            # (no predicted B) scores seg_iou≈0 here and early stopping cannot select it.
            **moryossef_segment_metrics(outputs["phrase"], labels, prefix="phrase", decode="bio"),
        }
        rows.append(row)
    if was_training: model.train()
    return mean_logs(rows, prefix="val")


def train_segmenter_epochs(
    model: MoryossefSegmenter, loader: DataLoader, optimizer: torch.optim.Optimizer,
    device: torch.device, epochs: int, dice_weight: float = 1.5,
    cfg: dict | None = None, dev_loader: DataLoader | None = None,
) -> SegmenterTrainOutput:
    model.to(device)
    model.train()
    logs: list[dict[str, float]] = []
    cfg = cfg or {}
    class_weights = bio_class_weight_tensor(cfg.get("bio_class_weights"))

    scheduler = build_scheduler(optimizer, cfg, epochs=epochs, steps_per_epoch=len(loader))
    amp = AmpHelper.from_config(cfg, device)
    control = TrainControl.from_config(cfg, default_monitor="val_phrase_tiou_f1", default_mode="max")
    attach_save_best(control, cfg, "segmenter", save_model_checkpoint)
    logger = TrainLogger("segmenter", cfg, epochs=int(epochs), steps_per_epoch=len(loader), monitor=control.monitor)
    
    for epoch in range(1, int(epochs) + 1):
        epoch_logs: list[dict[str, float]] = []
        for step, batch in enumerate(loader, start=1):
            poses = batch["poses"].to(device)
            timestamps = batch["timestamps_s"].to(device)
            labels = batch["phrase_bio"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                outputs = model(poses, timestamps_s=timestamps)
                loss = bio_nll_dice_loss(outputs["phrase"], labels, dice_weight=dice_weight, class_weights=class_weights)

            amp.backward(loss)
            amp.clip_and_step(optimizer, model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
            scheduler.step_batch()
            row = {
                "epoch": float(epoch), "step": float(step),
                "phrase_bio_loss": float(loss.detach().cpu().item()), "lr": scheduler.lr(optimizer),
            }
            epoch_logs.append(row)
            logs.append(row)
            logger.log_step(epoch, step, row)

        scheduler.step_epoch()
        train_means = mean_logs(epoch_logs)
        if dev_loader is not None and control.should_eval(epoch, epochs):
            metrics = evaluate_segmenter(model, dev_loader, device, dice_weight=dice_weight, class_weights=class_weights)
            improved = control.update(model, metrics, epoch)
            logger.epoch_summary(epoch, train=train_means, val=metrics, is_best=improved, saved_path=control.last_saved_path)
            logs.append({"epoch": float(epoch), **train_means, **metrics, **control.summary()})
            if control.stopped_early:
                print(f"segmenter | early stop at epoch {epoch} (best {control.monitor}={control.best_value})", flush=True)
                break
        else: logger.epoch_summary(epoch, train=train_means)

    control.restore(model)
    logger.finish()
    return SegmenterTrainOutput(logs=logs)
