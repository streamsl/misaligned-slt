"""Faithful Moryossef 2026 external segmenter for calibration and the RQ2 cascade.

Their 50 landmarks (+ velocity) → UNet CNN → RoPE Transformer → phrase BIO head: their own input contract, not the
in-system head's Uni-Sign features. Standalone on whole-video chunks, never the FSM head's bio_head_init.
"""
from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler
from data.windowing import BIO, TRUSTED_GAP_S
from data.loader import ANNOTATION_PROTOCOL, PooledEpochRecords, annotation_fingerprint, assert_pool_safe, resolve_pretrain_records 
from moryossef26.dataset import MoryossefChunkDataset, collate_moryossef_chunks, fit_release_stats
from moryossef26.model import MoryossefSegmenter, load_moryossef_pretrained

from train import distributed as dist
from train.losses import bio_class_weight_tensor, bio_nll_dice_loss, resolve_bio_class_weights
from train.helpers import build_optimizer, eval_mode, mean_logs, run_epoch_loop
from metrics import bio_frame_metrics, moryossef_segment_metrics
from utils import load_yaml, checkpoint_dir, pool_key


def build_moryossef_loaders(
    data_config: str, moryossef_config: str, language: str | None = None
) -> tuple[DataLoader, DataLoader, dict]:
    data_cfg = load_yaml(data_config)
    cfg = load_yaml(moryossef_config)
    # CLI --language > config language > active_languages; reload so ${language} in checkpoint.dir re-points.
    _requested_language = language   # raw CLI value, before defaulting (pooled runs refuse it)
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
    # Same pretraining pool as S1 when configured: if our segmenter sees several languages, the baseline
    # segmenter must too, or the cascade comparison is a data-advantage comparison.
    assert_pool_safe(cfg)
    train_records, pretrain_mix = resolve_pretrain_records(cfg, data_cfg, language, "train", requested=_requested_language)
    if pretrain_mix:
        cfg["pretrain_mix"] = pretrain_mix
        # Same rule as S1: a pooled run is ONE language-agnostic model, so `dir: .../${language}` would write the
        # identical model once per --language under different names. Pool-named dir; --language is ignored here.
        ckpt = dict(cfg.get("checkpoint", {}) or {})
        ckpt["dir"] = checkpoint_dir(cfg, default="checkpoints/moryossef")
        cfg["checkpoint"] = ckpt
        print(f"segmenter | multilingual pretraining -> {ckpt['dir']} (--language ignored)", flush=True)
    # Same trusted_gap_s as the labels this arm trains under (`common` feeds MoryossefChunkDataset's make_bio_labels): with the 
    # data.yaml override set, defaulting here would measure O/UNK marginal under 1 gap rule and scale a loss computed under another.
    resolve_bio_class_weights(cfg, train_records, trusted_gap_s=common["trusted_gap_s"])
    # The released model's standardisation table, fitted ONCE on these records (moryossef26.dataset.fit_release_stats):
    # a fixed table is what makes a training chunk and the whole-video inference pass agree, exactly as theirs does.
    release_stats = fit_release_stats(train_records, seed=int(cfg.get("seed", 42)))
    common["release_stats"] = release_stats
    # Provenance stamp, same keys S1 writes: without it a pooled baseline checkpoint is indistinguishable
    # from a monolingual one at load time and eval.py's pool assertion can never fire for this arm.
    cfg["checkpoint_meta"] = {
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(train_records),
        "language": cfg.get("language"), "bio_class_weights": cfg.get("bio_class_weights"),
        "pretrain_pool": pool_key(cfg), "pretrain_mix": cfg.get("pretrain_mix"), "release_stats": release_stats,
        "initialization": (cfg.get("checkpoint", {}) or {}).get("from_pretrained"),
    }
    # Dev is balanced by the same rule as train (see train/bio_pretrain.py for the argument) and never rotates.
    # Stamped for the same reason S1 stamps it: the two arms' monitors are only comparable next to their dev sets.
    dev_records, dev_mix = resolve_pretrain_records(cfg, data_cfg, language, "dev")
    if dev_mix: cfg["pretrain_dev_mix"] = cfg["checkpoint_meta"]["pretrain_dev_mix"] = dev_mix
    dev_steps = sum(sum(1 for sp in r.sentences if getattr(sp, 'reliable', True)) for r in dev_records)
    
    # Same rotation contract as S1 (train/bio_pretrain.py): a pooled run re-draws its balanced sub-sample each
    # epoch so both arms see the same data exposure and the cascade compares METHODS, not data.
    train_ds = MoryossefChunkDataset(train_records, steps_per_epoch=cfg.get("steps_per_epoch"), training=True,
        records_for_epoch=PooledEpochRecords(cfg, data_cfg, language) if pretrain_mix else None, **common)
    dev_ds = MoryossefChunkDataset(dev_records, steps_per_epoch=max(dev_steps, 1), training=False, **common)

    bs = dist.per_rank_batch_size(int(cfg.get("batch_size", 8)))
    _sampler = lambda ds: DistributedSampler(
        ds, num_replicas=dist.world_size(), rank=dist.rank(), shuffle=False
    ) if dist.is_distributed() else None
        
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, sampler=_sampler(train_ds), collate_fn=collate_moryossef_chunks)
    dev_loader = DataLoader(dev_ds, batch_size=bs, sampler=_sampler(dev_ds), collate_fn=collate_moryossef_chunks)
    return train_loader, dev_loader, cfg


def build_moryossef(moryossef_config: str, pretrained: bool = True, seed: int | None = None) -> MoryossefSegmenter:
    # `pretrained=False` + an explicit `seed` is random-init control (eval --segmenter-init random): same architecture and same input 
    # contract with nothing transferred, which is the range a released-weights score has to be read against.
    cfg = load_yaml(moryossef_config)
    if seed is not None: torch.manual_seed(int(seed))
    pose_dim = 6 if bool(cfg.get("velocity", True)) else 3  # +velocity doubles the per-keypoint channel dim
    model = MoryossefSegmenter(
        pose_dims=(int(cfg.get("pose_joints", 50)), pose_dim),
        hidden_dim=int(cfg.get("hidden_dim", 384)), encoder_depth=int(cfg.get("encoder_depth", 4)),
        attn_nhead=int(cfg.get("attn_nhead", 8)), attn_ff_mult=int(cfg.get("attn_ff_mult", 2)),
        attn_dropout=float(cfg.get("attn_dropout", 0.1)), num_frames=int(cfg.get("num_frames", 1024)),
    )
    # Optional warm start from the released Moryossef 2026 weights (vs random init). See load_moryossef_pretrained:
    # not zero-shot — train-moryossef fine-tunes after, and the untrained control is eval --segmenter-init released.
    weights = (cfg.get("checkpoint", {}) or {}).get("from_pretrained") if pretrained else None
    if weights and Path(weights).exists(): load_moryossef_pretrained(model, weights)
    elif weights: print(f"segmenter | WARNING: from_pretrained {weights} not found — random init", flush=True)
    elif not pretrained: print(f"segmenter | RANDOM init (seed {seed}), no released weights", flush=True)
    return model


@torch.no_grad()
def evaluate_moryossef(model, loader, device, dice_weight, class_weights) -> dict[str, float]:
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


def train_moryossef_epochs(model, train_loader, dev_loader, device, epochs, cfg, resume: bool = False) -> int:
    dice_weight = float(cfg.get("dice_loss_weight", 1.5))
    class_weights = bio_class_weight_tensor(cfg.get("bio_class_weights"))
    if class_weights is not None: class_weights = class_weights.to(device)
    optimizer = build_optimizer(cfg, model.parameters())  # end-to-end: UNet + RoPE + head all train

    def step_fn(batch, _epoch):
        logits = model(batch["poses"], timestamps_s=batch["timestamps_s"])["phrase"]
        loss = bio_nll_dice_loss(logits, batch["phrase_bio"], dice_weight=dice_weight, class_weights=class_weights)
        return loss, {"bio_loss": float(loss.detach())}

    return run_epoch_loop(
        name="moryossef", model=model, loader=train_loader, optimizer=optimizer, device=device, epochs=epochs, cfg=cfg, 
        step_fn=step_fn, evaluate_fn=lambda epoch: evaluate_moryossef(model, dev_loader, device, dice_weight, class_weights),
        default_monitor="val_phrase_tiou_f1", default_mode="max", dev_loader=dev_loader, resume=resume,
        # Same wiring as S1 (train/bio_pretrain.py): without it the SAVE-ON-BEST model.pt and latest.pt carry no meta, 
        # so eval.py's pool-provenance assertion could never fire for this arm and --resume could not detect config drift. 
        checkpoint_meta=cfg.get("checkpoint_meta")
    )
