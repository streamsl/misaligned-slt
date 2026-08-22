"""Faithful Moryossef 2026 external segmenter for calibration and the RQ2 cascade.

Raw keypoints (+ velocity) → UNet CNN → RoPE Transformer → phrase BIO head: a different input space from the
in-system head's Uni-Sign features. Standalone on whole-video chunks, never the FSM head's bio_head_init.
"""
from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler
from data.windowing import BIO, TRUSTED_GAP_S
from data.loader import load_language_records
from moryossef26.dataset import SegmenterChunkDataset, collate_segmenter_chunks
from moryossef26.model import MoryossefSegmenter, load_moryossef_pretrained

from train import distributed as dist
from train.losses import bio_class_weight_tensor, bio_nll_dice_loss, resolve_bio_class_weights
from train.helpers import build_optimizer, eval_mode, mean_logs, run_epoch_loop
from metrics import bio_frame_metrics, moryossef_segment_metrics
from utils import load_yaml


def build_segmenter_loaders(data_config: str, moryossef_config: str, language: str | None = None) -> tuple[DataLoader, DataLoader, dict]:
    data_cfg = load_yaml(data_config)
    cfg = load_yaml(moryossef_config)
    # CLI --language > config language > active_languages; reload so ${language} in checkpoint.dir re-points.
    language = str(language or cfg.get("language") or data_cfg.get("active_languages", ["asf"])[0])
    if language != cfg.get("language"): cfg = load_yaml(moryossef_config, language=language)
    aug_cfg = cfg.get("augmentation", {}) or {}
    fps_cfg = aug_cfg.get("fps", {})
    trusted_gap = data_cfg.get("subtitles", {}).get("trusted_gap_s", TRUSTED_GAP_S)
    common = dict(
        num_frames=int(cfg.get("num_frames", 1024)),
        fps_aug_enabled=bool(fps_cfg.get("enabled", True)),
        fps_aug_min=float(fps_cfg.get("min_fps", 15.0)), fps_aug_max=float(fps_cfg.get("max_fps", 30.0)),
        velocity=bool(cfg.get("velocity", True)),
        frame_dropout=float(aug_cfg.get("frame_dropout", 0.15)), body_part_dropout=float(aug_cfg.get("body_part_dropout", 0.1)),
        seed=int(cfg.get("seed", 42)), trusted_gap_s=None if trusted_gap is None else float(trusted_gap),
    )
    train_records, _ = load_language_records(data_cfg, language, split="train")
    resolve_bio_class_weights(cfg, train_records)
    dev_records, _ = load_language_records(data_cfg, language, split="dev")
    dev_steps = sum(len(r.sentences) for r in dev_records)
    train_ds = SegmenterChunkDataset(train_records, steps_per_epoch=cfg.get("steps_per_epoch"), training=True, **common)
    dev_ds = SegmenterChunkDataset(dev_records, steps_per_epoch=max(dev_steps, 1), training=False, **common)
    bs = dist.per_rank_batch_size(int(cfg.get("batch_size", 8)))
    def _sampler(ds):
        return DistributedSampler(ds, num_replicas=dist.world_size(), rank=dist.rank(), shuffle=False) if dist.is_distributed() else None
        
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, sampler=_sampler(train_ds), collate_fn=collate_segmenter_chunks)
    dev_loader = DataLoader(dev_ds, batch_size=bs, sampler=_sampler(dev_ds), collate_fn=collate_segmenter_chunks)
    return train_loader, dev_loader, cfg


def build_segmenter(moryossef_config: str) -> MoryossefSegmenter:
    cfg = load_yaml(moryossef_config)
    pose_dim = 6 if bool(cfg.get("velocity", True)) else 3  # +velocity doubles the per-keypoint channel dim
    model = MoryossefSegmenter(
        pose_dims=(int(cfg.get("pose_joints", 69)), pose_dim),
        hidden_dim=int(cfg.get("hidden_dim", 384)), encoder_depth=int(cfg.get("encoder_depth", 4)),
        attn_nhead=int(cfg.get("attn_nhead", 8)), attn_ff_mult=int(cfg.get("attn_ff_mult", 2)),
        attn_dropout=float(cfg.get("attn_dropout", 0.1)), num_frames=int(cfg.get("num_frames", 1024)),
    )
    # Optional cross-modality warm-start from the released Moryossef 2026 weights (vs random init). See
    # load_moryossef_pretrained: not zero-shot — train-segmenter still fine-tunes on our DWPose data.
    pretrained = (cfg.get("checkpoint", {}) or {}).get("from_pretrained")
    if pretrained and Path(pretrained).exists(): load_moryossef_pretrained(model, pretrained)
    elif pretrained: print(f"segmenter | WARNING: from_pretrained {pretrained} not found — random init", flush=True)
    return model


@torch.no_grad()
def evaluate_segmenter(model, loader, device, dice_weight, class_weights) -> dict[str, float]:
    rows = []
    with eval_mode(model):
        for batch in loader:
            poses = batch["poses"].to(device)
            timestamps = batch["timestamps_s"].to(device)
            labels = batch["phrase_bio"].to(device)
            logits = model(poses, timestamps_s=timestamps)["phrase"]
            row = {"bio_loss": float(bio_nll_dice_loss(logits, labels, dice_weight=dice_weight, class_weights=class_weights))}
            row.update(bio_frame_metrics(logits, labels, prefix="bio"))
            row.update(moryossef_segment_metrics(logits, labels, prefix="phrase"))
            alli = torch.zeros_like(logits); alli[..., BIO["I"]] = 1.0
            row["alli_tiou_f1"] = moryossef_segment_metrics(alli, labels, prefix="alli")["alli_tiou_f1"]
            rows.append(row)
    return mean_logs(rows, prefix="val")


def train_segmenter_epochs(model, train_loader, dev_loader, device, epochs, cfg, resume: bool = False) -> list[dict]:
    dice_weight = float(cfg.get("dice_loss_weight", 1.5))
    class_weights = bio_class_weight_tensor(cfg.get("bio_class_weights"))
    if class_weights is not None: class_weights = class_weights.to(device)
    optimizer = build_optimizer(cfg, model.parameters())  # end-to-end: UNet + RoPE + head all train

    def step_fn(batch, epoch):
        logits = model(batch["poses"], timestamps_s=batch["timestamps_s"])["phrase"]
        loss = bio_nll_dice_loss(logits, batch["phrase_bio"], dice_weight=dice_weight, class_weights=class_weights)
        return loss, {"bio_loss": float(loss.detach())}

    return run_epoch_loop(
        name="segmenter", model=model, loader=train_loader, optimizer=optimizer, 
        device=device, epochs=epochs, cfg=cfg, step_fn=step_fn,
        evaluate_fn=lambda epoch: evaluate_segmenter(model, dev_loader, device, dice_weight, class_weights),
        default_monitor="val_phrase_tiou_f1", default_mode="max", dev_loader=dev_loader, resume=resume,
    )
