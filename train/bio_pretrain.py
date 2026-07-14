"""S1 — in-system BIO head pretrain on FROZEN Uni-Sign pose features (docs/membership_gate.md §1.4 "competence
before coupling").

This is the DEPLOYED FSM head, pretrained then jointly fine-tuned — NOT the Moryossef analysis segmenter
(moryossef26/, a separate raw-keypoint instrument for Analysis A/B + the RQ2 cascade). The two are deliberately
distinct: gate-doc §1.4 keeps the analysis segmenter on raw keypoints (a different input space) so Analysis A's
calibration is non-circular, while this head reads Uni-Sign features and lives inside the system.

Two jobs, both about the coupling:
  1. **membership-gate warm start**: S2 starts from a sharp head, so the gate couples on-policy from step one with
     `membership_gate.warmup_epochs: 0` (no garbage-conditioning phase).
  2. **in-system BIO head init**: S2 loads `bio_head.*` from this checkpoint (`checkpoint.bio_head_init` in dlm.yaml),
     then JOINTLY fine-tunes it under the gate (gate-doc §1.4 S2; the frozen-BIO alternative is an ablation only).

Recipe (gate-doc §1.4 S1): the SAME SLT window distribution the head deploys under — StreamingWindowDataset +
WindowSampler — with Dice(1.5) + plain CE, fps_aug, RoPE relative time, the pose encoder FROZEN (released Uni-Sign
checkpoint; BatchNorm held in eval so S1 features == S2 initial features). Training on the window distribution (not
Moryossef chunks) is train-what-inference-sees applied to the head, and gives S1↔S2 parity so S2 trains exactly one
new thing: the coupling.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backbones import UniSignPoseEncoder
from data.batch import WindowCollator
from data.loader import StreamingWindowDataset, load_language_records
from models.bio_head import RoPEBIOHead

from metrics import bio_frame_metrics, moryossef_segment_metrics
from train.helpers import build_optimizer, eval_mode, mean_logs, run_epoch_loop
from train.losses import bio_class_weight_tensor, bio_nll_dice_loss
from utils import load_yaml, pretrained_checkpoint


class BioS1Model(nn.Module):
    """Frozen Uni-Sign pose encoder + trainable RoPE BIO head.

    Attribute name `bio_head` matches `MisalignedSLTModel.bio_head`, so this checkpoint's `bio_head.*` keys
    load directly as the SLT-model init. The pose encoder is frozen AND kept in eval mode (`train()` override):
    its ST-GCN BatchNorm running stats must stay exactly the released checkpoint's — SLT training shares that init,
    so S1 features == S2 initial features and the head transfers without an input-distribution jump.
    """
    def __init__(self, pose_hidden_dim: int = 256, feat_dim: int = 768, bio_hidden_dim: int = 384,
                 bio_depth: int = 4, bio_nhead: int = 8, bio_dropout: float = 0.1, bio_conv_stem_layers: int = 2):
        super().__init__()
        self.pose_encoder = UniSignPoseEncoder(hidden_dim=int(pose_hidden_dim), out_dim=int(feat_dim))
        for p in self.pose_encoder.parameters(): p.requires_grad_(False)
        self.bio_head = RoPEBIOHead(
            input_dim=int(feat_dim), hidden_dim=int(bio_hidden_dim), depth=int(bio_depth), nhead=int(bio_nhead),
            dropout=float(bio_dropout), num_classes=4, conv_stem_layers=int(bio_conv_stem_layers),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.pose_encoder.eval()  # frozen: BN running stats pinned to the released checkpoint
        return self

    def load_unisign_pose(self, ckpt_path: str | Path) -> int:
        blob = torch.load(str(ckpt_path), map_location="cpu")
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        pose_sd = {k: v for k, v in sd.items() if not k.startswith("mt5_model.")}
        self.pose_encoder.load_state_dict(pose_sd, strict=True)
        return len(pose_sd)

    def forward(self, poses, frame_mask, timestamps_s=None):
        with torch.no_grad():
            feats = self.pose_encoder(poses, frame_mask)
        return self.bio_head(feats, timestamps_s=timestamps_s)


def build_bio_s1_model(cfg: dict) -> BioS1Model:
    """Construct BioS1Model from a config and load the FROZEN Uni-Sign pose encoder.

    SINGLE source of truth for the head shape: both training (build_bio_s1) and inference (analyze.segmenter_infer) go through 
    here, so the S1 checkpoint always strict-loads and SLT model's `bio_head_init` shape parity is guaranteed by construction.
    """
    model = BioS1Model(
        pose_hidden_dim=int(cfg.get("pose_hidden_dim", 256)), feat_dim=int(cfg.get("feat_dim", 768)),
        bio_hidden_dim=int(cfg.get("bio_hidden_dim", 384)), bio_depth=int(cfg.get("bio_depth", 4)),
        bio_nhead=int(cfg.get("bio_nhead", 8)), bio_dropout=float(cfg.get("bio_dropout", 0.1)),
        bio_conv_stem_layers=int(cfg.get("bio_conv_stem_layers", 2)),
    )
    pose_ckpt = pretrained_checkpoint(cfg, default="checkpoints/csl_daily_pose_only_slt.pth")
    n = model.load_unisign_pose(pose_ckpt)
    print(f"bio_s1 | frozen pose encoder from {pose_ckpt} ({n} tensors); "
          f"trainable head: {sum(p.numel() for p in model.bio_head.parameters()) / 1e6:.2f}M params", flush=True)
    return model


def build_bio_s1(
    data_config: str = "configs/data.yaml", config: str = "configs/bio_pretrain.yaml", 
    inference_config: str = "configs/inference.yaml",
) -> tuple[BioS1Model, DataLoader, DataLoader, dict]:
    cfg = load_yaml(config)
    data_cfg = load_yaml(data_config)
    inference_cfg = load_yaml(inference_config)
    language = str(cfg.get("language", data_cfg.get("active_languages", ["csl"])[0]))

    train_records, _ = load_language_records(data_cfg, language, split="train")
    train_dataset = StreamingWindowDataset(
        train_records, slt_cfg=cfg, inference_cfg=inference_cfg, pose_augment_cfg=cfg.get("augmentation"),
    )
    dev_records, _ = load_language_records(data_cfg, language, split="dev")
    dev_steps = sum(len(r.sentences) for r in dev_records)
    dev_dataset = StreamingWindowDataset(
        dev_records, slt_cfg=cfg, inference_cfg=inference_cfg,
        steps_per_epoch=max(dev_steps, 1), deterministic=True,
    )
    collator = WindowCollator(tokenizer=None)  # BIO-only: no text tokenization
    num_workers = int(cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset, batch_size=int(cfg.get("batch_size", 8)), shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0, collate_fn=collator
    )
    dev_loader = DataLoader(dev_dataset, batch_size=int(cfg.get("batch_size", 8)), collate_fn=collator)

    model = build_bio_s1_model(cfg)
    return model, train_loader, dev_loader, cfg


@torch.no_grad()
def evaluate_bio_s1(
    model: BioS1Model, loader: DataLoader, device: torch.device,
    dice_weight: float, class_weights: torch.Tensor | None
) -> dict[str, float]:
    rows = []
    with eval_mode(model):
        for batch in loader:
            poses = batch["poses"].to(device)
            mask = batch["frame_mask"].to(device)
            ts = batch["timestamps_s"].to(device)
            labels = batch["bio_labels"].to(device)
            out = model(poses, mask, timestamps_s=ts)
            row = {"bio_loss": float(bio_nll_dice_loss(out.logits, labels, dice_weight=dice_weight, class_weights=class_weights))}
            row.update(bio_frame_metrics(out.logits, labels, prefix="bio"))
            row.update(moryossef_segment_metrics(out.logits, labels, prefix="phrase"))
            rows.append(row)
    return mean_logs(rows, prefix="val")


def train_bio_s1_epochs(
    model: BioS1Model, train_loader: DataLoader, dev_loader: DataLoader, device: torch.device, epochs: int, cfg: dict
) -> list[dict[str, float]]:
    dice_weight = float(cfg.get("dice_loss_weight", 1.5))
    class_weights = bio_class_weight_tensor(cfg.get("bio_class_weights"))
    if class_weights is not None: class_weights = class_weights.to(device)
    optimizer = build_optimizer(cfg, model.bio_head.parameters())  # only the head trains; the pose encoder is frozen

    def step_fn(batch, epoch: int):
        out = model(batch["poses"], batch["frame_mask"], timestamps_s=batch["timestamps_s"])
        loss = bio_nll_dice_loss(out.logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=class_weights)
        return loss, {"bio_loss": float(loss.detach())}

    return run_epoch_loop(
        name="bio_s1", model=model, loader=train_loader, optimizer=optimizer, device=device,
        epochs=epochs, cfg=cfg, step_fn=step_fn,
        evaluate_fn=lambda epoch: evaluate_bio_s1(model, dev_loader, device, dice_weight, class_weights),
        default_monitor="val_phrase_tiou_f1", default_mode="max", dev_loader=dev_loader,
    )
