"""Faithful Moryossef 2026 external segmenter for the RQ2 cascade.

Their 50 landmarks (+ velocity) → UNet CNN → RoPE Transformer → phrase BIO head: their own input contract, not the
in-system head's Uni-Sign features. Standalone on whole-video chunks, never the FSM head's bio_head_init. Everything else 
is S1's (train/bio_pretrain.py): pool and rotation, chunk tiling, augmentation, labels, dev monitor and best-epoch floor.
"""
from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from data.chunks import ChunkDataset
from data.loader import ANNOTATION_PROTOCOL, PooledEpochRecords, annotation_fingerprint, resolve_pretrain_records, streaming_loader
from moryossef26.dataset import collate_moryossef_chunks, fit_release_stats
from moryossef26.model import MoryossefSegmenter, load_moryossef_pretrained

from train import distributed as dist
from train.losses import bio_class_weight_tensor, bio_nll_dice_loss, resolve_bio_class_weights
from train.helpers import build_optimizer, evaluate_bio_chunks, run_epoch_loop
from utils import NATIVE_POSE_FPS, load_yaml, checkpoint_dir, pool_key


def build_moryossef_loaders(
    data_config: str, moryossef_config: str, language: str | None = None
) -> tuple[DataLoader, DataLoader, dict]:
    data_cfg = load_yaml(data_config)
    cfg = load_yaml(moryossef_config)
    # CLI --language > config language > active_languages; reload so ${language} in checkpoint.dir re-points.
    _requested_language = language   # raw CLI value, before defaulting (pooled runs refuse it)
    language = str(language or cfg.get("language") or data_cfg.get("active_languages", ["asf"])[0])
    if language != cfg.get("language"): cfg = load_yaml(moryossef_config, language=language)
    # Same pretraining pool as S1 when configured: if our segmenter sees several languages, the baseline
    # segmenter must too, or the cascade comparison is a data-advantage comparison.
    train_records, pretrain_mix = resolve_pretrain_records(cfg, data_cfg, language, "train", requested=_requested_language)
    if pretrain_mix:
        cfg["pretrain_mix"] = pretrain_mix
        # Same rule as S1: a pooled run is ONE language-agnostic model, so `dir: .../${language}` would write the
        # identical model once per --language under different names. Pool-named dir; --language is ignored here.
        ckpt = dict(cfg.get("checkpoint", {}) or {})
        ckpt["dir"] = checkpoint_dir(cfg, default="checkpoints/moryossef")
        cfg["checkpoint"] = ckpt
        print(f"segmenter | multilingual pretraining -> {ckpt['dir']} (--language ignored)", flush=True)
        
    resolve_bio_class_weights(cfg, train_records)   # labels and their class weights: S1's rule (data.chunks.ChunkDataset)
    seed = int(cfg.get("seed", 42))
    # The released model's standardisation table, fitted ONCE on these records (moryossef26.dataset.fit_release_stats):
    # a fixed table is what makes a training chunk and the whole-video inference pass agree, exactly as theirs does.
    release_stats = fit_release_stats(train_records, seed=seed)
    # Same rotation contract and best-epoch floor as S1 (train/bio_pretrain.py): a pooled run re-draws its balanced 
    # sub-sample each epoch, and no epoch before the rotation's first full cycle can become the best checkpoint.
    pool = PooledEpochRecords(cfg, data_cfg, language) if pretrain_mix else None
    # Provenance stamp, same keys S1 writes: without it a pooled baseline checkpoint is indistinguishable
    # from a monolingual one at load time and eval.py's pool assertion can never fire for this arm.
    cfg["checkpoint_meta"] = {
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(train_records),
        "language": cfg.get("language"), "bio_class_weights": cfg.get("bio_class_weights"),
        "pretrain_pool": pool_key(cfg), "pretrain_mix": cfg.get("pretrain_mix"), "release_stats": release_stats,
        "initialization": (cfg.get("checkpoint", {}) or {}).get("from_pretrained"),
        "learning_rate": float(cfg["learning_rate"]),  # --resume refuses a drifted rate the optimizer load would restore
        "augmentation": cfg.get("augmentation"),       # and a changed augmentation, as S1 does
        "best_epoch_floor": pool.cycle_epochs() if pool else 0,
    }
    # Dev is balanced by the same rule as train (see train/bio_pretrain.py for the argument) and never rotates.
    # Stamped for the same reason S1 stamps it: the two arms' monitors are only comparable next to their dev sets.
    dev_records, dev_mix = resolve_pretrain_records(cfg, data_cfg, language, "dev")
    if dev_mix: cfg["pretrain_dev_mix"] = cfg["checkpoint_meta"]["pretrain_dev_mix"] = dev_mix
    # One fixed chunk length, C = L_min = (num_frames - 1) / fps: a floored-and-ceiled window loads at most num_frames frames,
    # so a training chunk is one RoPE pass. The release contract (50 landmarks, fitted table, velocity) replaces Uni-Sign's.
    chunk_s = (int(cfg.get("num_frames", 1024)) - 1) / NATIVE_POSE_FPS
    release = {"stats": release_stats, "velocity": bool(cfg.get("velocity", True))}
    train_ds = ChunkDataset(
        train_records, chunk_s, chunk_s, augmentation=cfg["augmentation"], seed=seed, records_for_epoch=pool, release=release
    )
    dev_ds = ChunkDataset(dev_records, chunk_s, chunk_s, seed=seed, deterministic=True, release=release)
    print(f"segmenter | {len(train_ds)} train chunks/epoch, {len(dev_ds)} dev chunks ({train_ds.chunk_s:g} s)", flush=True)
    num_workers = int(cfg.get("num_workers", 0))
    bs = dist.per_rank_batch_size(int(cfg.get("batch_size", 8)))
    train_loader = streaming_loader(
        train_ds, bs, collate_moryossef_chunks, num_workers=num_workers,
        bucket_by_length=True, bucket_seed=seed,
    )
    dev_loader = streaming_loader(dev_ds, bs, collate_moryossef_chunks, num_workers=num_workers)
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


def train_moryossef_epochs(model, train_loader, dev_loader, device, epochs, cfg, resume: bool = False) -> int:
    dice_weight = float(cfg.get("dice_loss_weight", 1.5))
    class_weights = bio_class_weight_tensor(cfg.get("bio_class_weights"))
    if class_weights is not None: class_weights = class_weights.to(device)
    optimizer = build_optimizer(cfg, model.parameters())  # end-to-end: UNet + RoPE + head all train

    def step_fn(batch, _epoch):
        logits = model(batch["poses"], timestamps_s=batch["timestamps_s"])["phrase"]
        loss = bio_nll_dice_loss(logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=class_weights)
        return loss, {"bio_loss": float(loss.detach())}

    return run_epoch_loop(
        name="moryossef", model=model, loader=train_loader, optimizer=optimizer, device=device, epochs=epochs, cfg=cfg, 
        step_fn=step_fn, evaluate_fn=lambda epoch: evaluate_bio_chunks(model, dev_loader, device, dice_weight, class_weights),
        default_monitor="val_phrase_tiou_f1", default_mode="max", dev_loader=dev_loader, resume=resume,
        # Same wiring as S1 (train/bio_pretrain.py): without it the SAVE-ON-BEST model.pt and latest.pt carry no meta, 
        # so eval.py's pool-provenance assertion could never fire for this arm and --resume could not detect config drift. 
        checkpoint_meta=cfg["checkpoint_meta"], best_epoch_floor=cfg["checkpoint_meta"]["best_epoch_floor"],
    )
