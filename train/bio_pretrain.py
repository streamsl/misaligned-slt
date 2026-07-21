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
from data.loader import StreamingWindowDataset, load_language_records, streaming_loader
from models.bio_head import RoPEBIOHead

from metrics import bio_frame_metrics, moryossef_segment_metrics
from train.helpers import build_optimizer, eval_mode, mean_logs, run_epoch_loop
from train.losses import bio_class_weight_tensor, bio_nll_dice_loss
from utils import load_yaml, pretrained_checkpoint, resolve_pretrained


class BioS1Model(nn.Module):
    """Uni-Sign pose encoder + trainable RoPE BIO head.

    Attribute name `bio_head` matches `MisalignedSLTModel.bio_head`, so this checkpoint's `bio_head.*` keys
    load directly as the SLT-model init.

    `freeze_encoder=True` (default, the gate-doc §1.4 recipe): the pose encoder is frozen AND pinned to eval
    (`train()` override) so its ST-GCN BatchNorm running stats stay exactly the released checkpoint's — S1
    features == S2 initial features and the head transfers without an input-distribution jump.
    `freeze_encoder=False` (`freeze_backbone: false`): the encoder trains too, adapting the translation-optimized
    features to the segmentation objective. NB this does NOT recover back-to-back sentence boundaries on YouTube
    corpora — that signal is absent from the captions themselves (docs/implementation_notes.md: the boundary-head +
    unfrozen-encoder recovery attempts all plateaued at val_phrase_tiou_f1 ≈ 0.55, an established data limit, not a
    head/loss/encoder deficiency); it is kept as a general feature-adaptation lever, not a boundary fix.
    """
    def __init__(
        self, pose_hidden_dim: int = 256, feat_dim: int = 768, bio_hidden_dim: int = 384, bio_depth: int = 4, 
        bio_nhead: int = 8, bio_dropout: float = 0.1, bio_conv_stem_layers: int = 2, freeze_encoder: bool = True,
    ):
        super().__init__()
        self.freeze_encoder = bool(freeze_encoder)
        self.pose_encoder = UniSignPoseEncoder(hidden_dim=int(pose_hidden_dim), out_dim=int(feat_dim))
        if self.freeze_encoder:
            for p in self.pose_encoder.parameters(): p.requires_grad_(False)
        self.bio_head = RoPEBIOHead(
            input_dim=int(feat_dim), hidden_dim=int(bio_hidden_dim), depth=int(bio_depth),
            nhead=int(bio_nhead), dropout=float(bio_dropout), num_classes=4,
            conv_stem_layers=int(bio_conv_stem_layers),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder: self.pose_encoder.eval()  # frozen: BN running stats pinned to the released checkpoint
        return self

    def load_unisign_pose(self, ckpt_path: str | Path) -> int:
        blob = torch.load(str(ckpt_path), map_location="cpu")
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        pose_sd = {k: v for k, v in sd.items() if not k.startswith("mt5_model.")}
        self.pose_encoder.load_state_dict(pose_sd, strict=True)
        return len(pose_sd)

    def forward(self, poses, frame_mask, timestamps_s=None):
        if self.freeze_encoder:
            with torch.no_grad(): feats = self.pose_encoder(poses, frame_mask)
        else: feats = self.pose_encoder(poses, frame_mask)
        return self.bio_head(feats, timestamps_s=timestamps_s, frame_mask=frame_mask)


def build_bio_s1_model(cfg: dict, pretrained_path: str | None = None) -> BioS1Model:
    """Construct BioS1Model from a config and load the FROZEN Uni-Sign pose encoder.

    SINGLE source of truth for the head shape: both training (build_bio_s1) and inference (analyze.segmenter_infer) go through
    here, so the S1 checkpoint always strict-loads and SLT model's `bio_head_init` shape parity is guaranteed by construction.
    `pretrained_path` overrides which released checkpoint the frozen pose encoder loads (per-language warm-start; the
    caller resolves it). At inference the trained bio_s1 checkpoint later overwrites these weights, so it only bites
    at train time — but keeping it per-language avoids depending on the CSL file on an English-only machine.
    """
    # Reuse the inherited dlm.yaml `freeze_backbone` key (same concept: freeze the Uni-Sign pose encoder). dlm's
    # value is for STAGE 2 (false → joint-train); bio_pretrain.yaml overrides it to true for the frozen-S1 recipe.
    freeze_encoder = bool(cfg.get("freeze_backbone", True))
    model = BioS1Model(
        pose_hidden_dim=int(cfg.get("pose_hidden_dim", 256)), feat_dim=int(cfg.get("feat_dim", 768)),
        bio_hidden_dim=int(cfg.get("bio_hidden_dim", 384)), bio_depth=int(cfg.get("bio_depth", 4)),
        bio_nhead=int(cfg.get("bio_nhead", 8)), bio_dropout=float(cfg.get("bio_dropout", 0.1)),
        bio_conv_stem_layers=int(cfg.get("bio_conv_stem_layers", 2)), freeze_encoder=freeze_encoder,
    )
    pose_ckpt = pretrained_path or pretrained_checkpoint(cfg, default="checkpoints/csl_daily_pose_only_slt.pth")
    n = model.load_unisign_pose(pose_ckpt)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"bio_s1 | pose encoder from {pose_ckpt} ({n} tensors); encoder {'FROZEN' if freeze_encoder else 'TRAINABLE'}; "
          f"trainable params: {trainable:.2f}M", flush=True)
    return model


def build_bio_s1(
    data_config: str = "configs/data.yaml", config: str = "configs/bio_pretrain.yaml",
    inference_config: str = "configs/inference.yaml", language: str | None = None,
) -> tuple[BioS1Model, DataLoader, DataLoader, dict]:
    data_cfg = load_yaml(data_config)
    cfg = load_yaml(config)
    # Effective language: CLI --language > config's own language: > data.yaml active_languages. Reload with the
    # override only when it changes, so ${language} in checkpoint.dir re-points to the right dataset dir.
    language = str(language or cfg.get("language") or data_cfg.get("active_languages", ["csl"])[0])
    if language != cfg.get("language"): cfg = load_yaml(config, language=language)
    inference_cfg = load_yaml(inference_config)

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
    # streaming_loader: num_workers is safe at any value — the anchor is index-driven (exact per-epoch coverage) and
    # each worker reseeds its rng to decorrelate window draws (see data.loader.streaming_loader).
    train_loader = streaming_loader(train_dataset, int(cfg.get("batch_size", 8)), collator, num_workers=num_workers)
    dev_loader = streaming_loader(dev_dataset, int(cfg.get("batch_size", 8)), collator, num_workers=num_workers)

    # Frozen pose encoder from the language's released Uni-Sign checkpoint (OpenASL for English asf/bfi, CSL for
    # Chinese csl) — this MUST match the SLT model's warm-start so S1 features == S2 initial features (§1.4).
    pretrained = resolve_pretrained(cfg, data_cfg, language, default="checkpoints/csl_daily_pose_only_slt.pth")
    model = build_bio_s1_model(cfg, pretrained_path=pretrained)
    return model, train_loader, dev_loader, cfg


@torch.no_grad()
def evaluate_bio_s1(
    model: BioS1Model, loader: DataLoader, device: torch.device,
    dice_weight: float, class_weights: torch.Tensor | None,
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
            # Per-mode tIoU diagnostic. This is a SEGMENTATION metric (predicted BIO spans vs the GT BIO spans),
            # well-defined for EVERY mode because the head is supervised on all modes' bio_labels — independent of
            # translation ("complete-sentence") supervision, which only some modes carry (OPUT on 1/3, CB on 2a,
            # none on 2b/2c/4). So the split is legitimate for modes 2 and 4; only the INTERPRETATION differs:
            #   mode1/3 = span boundary quality; mode2 = truncated-fragment localization;
            #   mode4 (gaps) = PHANTOM-AVOIDANCE — gold has 0 spans (all-O; long all-UNK gaps are skipped), so
            #                  tiou_f1 is 1.0 iff the head stays silent and 0.0 if it fires. It scores ABSENCE, not overlap.
            # A capped headline average mixing these is uninterpretable without the split (a low val_phrase_tiou_f1
            # driven by mode2 fragments is a metric-granularity property of misaligned windows, not head incompetence).
            modes = batch.get("mode_names") or []
            for mode in set(modes):
                idx = [i for i, m in enumerate(modes) if m == mode]
                sub = moryossef_segment_metrics(out.logits[idx], labels[idx], prefix=mode)
                row[f"{mode}_tiou_f1"] = sub[f"{mode}_tiou_f1"]
            rows.append(row)
    return mean_logs(rows, prefix="val")


def train_bio_s1_epochs(
    model: BioS1Model, train_loader: DataLoader, dev_loader: DataLoader, device: torch.device, epochs: int, cfg: dict
) -> list[dict[str, float]]:
    dice_weight = float(cfg.get("dice_loss_weight", 1.5))
    class_weights = bio_class_weight_tensor(cfg.get("bio_class_weights"))
    if class_weights is not None: class_weights = class_weights.to(device)
    # Frozen encoder → optimize the head only; unfrozen (freeze_backbone: false) → optimize everything that needs grad.
    params = model.bio_head.parameters() if model.freeze_encoder else (p for p in model.parameters() if p.requires_grad)
    optimizer = build_optimizer(cfg, params)

    def step_fn(batch, epoch: int):
        out = model(batch["poses"], batch["frame_mask"], timestamps_s=batch["timestamps_s"])
        loss = bio_nll_dice_loss(out.logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=class_weights)
        return loss, {"bio_loss": float(loss.detach())}

    return run_epoch_loop(
        name="bio_s1", model=model, loader=train_loader, optimizer=optimizer,
        device=device, epochs=epochs, cfg=cfg, step_fn=step_fn,
        evaluate_fn=lambda epoch: evaluate_bio_s1(model, dev_loader, device, dice_weight, class_weights),
        default_monitor="val_phrase_tiou_f1", default_mode="max", dev_loader=dev_loader,
    )
