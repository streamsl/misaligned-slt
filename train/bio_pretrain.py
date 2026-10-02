"""S1 — multilingual segmentation pretraining for the in-system pose encoder and BIO head.

This is the deployed FSM head, not the external Moryossef segmenter. Their input spaces and checkpoints differ.

Two jobs:
  1. **gate warm start**: S2 starts from a sharp head, so the gate couples on-policy from step 0 (stage 2 
     refuses to run without `checkpoint.bio_head_init`).
  2. **BIO head init**: S2 loads `bio_head.*` (`checkpoint.bio_head_init` in dlm.yaml) & JOINTLY 
     fine-tunes it under the gate (§1.4 S2).

Recipe (§1.4 S1): whole-input random chunks, length U(δ + K·stride, C) with C = pretrain_geometry.buffer_cap_s (data/chunks.py), 
BIO only, on the balanced pooled corpus; Dice(1.5) + balanced CE, the augmentation recipe shared with Moryossef arm, RoPE time and 
the banded attention of dlm.yaml bio_attention_radius_s. The pose encoder trains at `backbone_lr`, the BIO head at `learning_rate`.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.batch import collate_windows
from data.chunks import ChunkDataset
from data.loader import (ANNOTATION_PROTOCOL, PooledEpochRecords, resolve_pretrain_records, streaming_loader, 
                         annotation_fingerprint, segmenter_language_fingerprints)
from backbones import UniSignPoseEncoder
from models.bio_head import RoPEBIOHead
from models.unisign import released_layout_state
from models.checkpointing import _load_state, load_checkpoint_meta

from train import distributed as dist
from train.helpers import build_optimizer, evaluate_bio_chunks, run_epoch_loop
from train.losses import bio_class_weight_tensor, bio_nll_dice_loss, resolve_bio_class_weights
from utils import NATIVE_POSE_FPS, cfg_get, checkpoint_dir, load_yaml, pool_key

class BioS1Model(nn.Module): # Pose backbone and temporal BIO classifier shared with the joint model's segmentation branch.
    def __init__(
        self, pose_hidden_dim: int = 256, feat_dim: int = 768, bio_hidden_dim: int = 384, bio_depth: int = 4, bio_nhead: int = 8, 
        bio_dropout: float = 0.1, bio_conv_stem_layers: int = 2, bio_attention_radius_s: float | None = None,
    ):
        super().__init__()
        self.pose_encoder = UniSignPoseEncoder(hidden_dim=int(pose_hidden_dim), out_dim=int(feat_dim))
        self.bio_head = RoPEBIOHead(
            input_dim=int(feat_dim), hidden_dim=int(bio_hidden_dim), depth=int(bio_depth), nhead=int(bio_nhead), 
            dropout=float(bio_dropout), num_classes=4, conv_stem_layers=int(bio_conv_stem_layers),
            attention_radius_s=bio_attention_radius_s,
        )

    def load_pretrained(self, ckpt_path: str | Path) -> int:
        sd = _load_state(ckpt_path)
        if any(k.startswith("bio_head.") for k in sd):
            require_attention_radius(load_checkpoint_meta(str(ckpt_path)) or {}, self.bio_head.attention_radius_s, str(ckpt_path))
            self.load_state_dict(sd, strict=True)
            return len(sd)
        sd = released_layout_state(sd)
        pose_sd = {k: v for k, v in sd.items() if not k.startswith("mt5_model.")}
        self.pose_encoder.load_state_dict(pose_sd, strict=True)
        return len(pose_sd)

    def forward(self, poses, frame_mask, timestamps_s=None):
        return self.bio_head(self.pose_encoder(poses, frame_mask), timestamps_s=timestamps_s, frame_mask=frame_mask)


def require_attention_radius(meta: dict, expected: float | None, source: str) -> None:
    """Refuse an S1 head trained under another attention band. The band is a mask, not a weight, so a strict state_dict
    load cannot detect it; a head read at another radius sees contexts it never trained on. No stamp = full attention."""
    trained = meta.get("bio_attention_radius_s")
    if trained is None and expected is None: return
    if trained is not None and expected is not None and abs(float(trained) - float(expected)) < 1e-9: return
    raise SystemExit(
        f"{source} was trained with bio_attention_radius_s={trained} (None = full attention), but the config expects {expected}. "
        f"Retrain S1 at that radius, or set bio_attention_radius_s to the trained value."
    )


def require_segmenter_data(meta: dict, language: str, train_fingerprint: str, buffer_cap_s: float, source: str) -> None:
    """Refuse a segmenter (S1 or Moryossef) whose labels are not the current ones. A re-convert, a person-counts re-measure or 
    a caption rule changes the train labels; a strict state_dict load cannot see that, and the stale model would load silently.
    `train_fingerprint` is the annotation fingerprint of `language`'s whole train split. A language outside the stamped pool is
    zero-shot for this segmenter, so it has nothing to compare. S1 must also have trained at the deployed cap or longer."""
    if meta.get("annotation_protocol") != ANNOTATION_PROTOCOL: raise SystemExit(
        f"{source} was trained under annotation protocol {meta.get('annotation_protocol')!r}, not {ANNOTATION_PROTOCOL!r}: retrain it."
    )
    stamped = meta.get("language_fingerprints")
    if stamped is None: raise SystemExit(f"{source} stamps no language_fingerprints (trained before the data guard): retrain it.")
    if language in stamped and stamped[language] != train_fingerprint: raise SystemExit(
        f"{source} was trained on other {language} train labels "
        f"(a re-convert, re-measured cuts or a caption rule changed them): retrain it."
    )
    trained_chunk = meta.get("rope_eval_chunk_s")
    if trained_chunk is not None and float(trained_chunk) + 1e-6 < float(buffer_cap_s): raise SystemExit(
        f"{source} trained on chunks of at most {float(trained_chunk):.2f}s, "
        f"below the deployed buffer_cap_s {float(buffer_cap_s):.2f}s: retrain it."
    )


def build_bio_s1_model(cfg: dict, pretrained_path: str | None = None) -> BioS1Model:
    # Construct S1; optionally initialize from released pose weights or a complete S1 checkpoint. The pose 
    # encoder trains with the head, so `bio_head_init` carries an ADAPTED segmentation encoder into stage 2.
    model = BioS1Model(
        pose_hidden_dim=int(cfg.get("pose_hidden_dim", 256)), feat_dim=int(cfg.get("feat_dim", 768)),
        bio_hidden_dim=int(cfg.get("bio_hidden_dim", 384)), bio_depth=int(cfg.get("bio_depth", 4)),
        bio_nhead=int(cfg.get("bio_nhead", 8)), bio_dropout=float(cfg.get("bio_dropout", 0.1)),
        bio_conv_stem_layers=int(cfg.get("bio_conv_stem_layers", 2)), bio_attention_radius_s=cfg.get("bio_attention_radius_s"),
    )
    if pretrained_path:
        n = model.load_pretrained(pretrained_path)
        print(f"bio_s1 | initialized from {pretrained_path} ({n} tensors)", flush=True)
    return model


def build_bio_s1(
    data_config: str = "configs/data.yaml", config: str = "configs/bio_pretrain.yaml",
    inference_config: str = "configs/inference.yaml", language: str | None = None,
) -> tuple[BioS1Model, DataLoader, DataLoader, dict]:
    data_cfg = load_yaml(data_config)
    cfg = load_yaml(config)
    # Precedence: CLI --language > config `language:` > data.yaml active_languages. Reload only when it changes, 
    # so ${language} in checkpoint.dir re-points to the right dataset dir.
    _requested_language = language   # raw CLI value, before defaulting (pooled runs refuse it)
    language = str(language or cfg.get("language") or data_cfg.get("active_languages", ["asf"])[0])
    if language != cfg.get("language"): cfg = load_yaml(config, language=language)
    inference_cfg = load_yaml(inference_config)

    # Segmentation is language-agnostic (boundaries are prosodic), so S1 may pretrain on a pool of languages;
    # translation stays monolingual in stage 2. `pretrain_languages: null` = the target language alone.
    train_records, pretrain_mix = resolve_pretrain_records(cfg, data_cfg, language, "train", requested=_requested_language)
    if pretrain_mix:
        cfg["pretrain_mix"] = pretrain_mix   # recorded into the run config for the paper
        # A multilingual S1 is ONE language-agnostic model: the pooled data does not depend on --language, so
        # `checkpoint.dir: .../${language}` would train the identical model once per language under different
        # names. Re-point it at a pool-named directory so one run serves every target language, and so a
        # multilingual checkpoint can never be mistaken for a monolingual one.
        ckpt = dict(cfg.get("checkpoint", {}) or {})
        ckpt["dir"] = checkpoint_dir(cfg, default="checkpoints/bio_s1")
        cfg["checkpoint"] = ckpt
        print(f"bio_s1 | multilingual pretraining -> {ckpt['dir']} (--language ignored)", flush=True)
        
    resolve_bio_class_weights(cfg, train_records)
    cfg["language_fingerprints"] = segmenter_language_fingerprints(cfg, data_cfg, language, train_records)
    # Chunk length C is the longest context the head trains on. Checkpoint metadata and whole-video evaluation use this 
    # stamped value, not a target inference file that may change later. Stage-2 head and FSM run at the one deployed cap 
    # (inference.yaml buffer_cap_s), so S1 must have trained at least that context.
    cfg["training_buffer_cap_s"] = chunk_s = float(cfg["pretrain_geometry"]["buffer_cap_s"])
    if chunk_s + 1e-6 < float(inference_cfg["buffer_cap_s"]): raise SystemExit(
        f"pretrain_geometry.buffer_cap_s {chunk_s:.2f}s is below the deployed buffer_cap_s "
        f"{float(inference_cfg['buffer_cap_s']):.2f}s ({inference_config}); raise it and train S1 at that context."
    )
    # Shortest interior chunk = the shortest buffer the FSM decides on after a commit: δ of committed context + K strides.
    stability = inference_cfg["boundary_stability"]
    cfg["training_min_chunk_s"] = min_chunk_s = float(stability["delta_enc_frames"]) / NATIVE_POSE_FPS + \
                                                int(stability["hysteresis_strides"]) * float(inference_cfg["stride_s"])
    seed = int(cfg.get("seed", 42))
    # A pooled run re-draws its balanced sub-sample each epoch, so the videos a sub-sampled corpus contributes
    # ROTATE and the whole corpus is covered across epochs. Monolingual runs pass no provider.
    pool = PooledEpochRecords(cfg, data_cfg, language) if pretrain_mix else None
    # No epoch before the rotation's first full cycle can become the best checkpoint (train.helpers.TrainControl).
    cfg["best_epoch_floor"] = pool.cycle_epochs() if pool else 0
    train_dataset = ChunkDataset(
        train_records, chunk_s, min_chunk_s, augmentation=cfg["augmentation"], seed=seed, records_for_epoch=pool,
    )
    # Dev is drawn via the SAME balancing rule as train (`load_multilingual_records`), so the monitor measures what training optimises 
    # rather than the corpus-size prior. A pooled dev taken AS-IS would be ~84% ase, and best-checkpoint selection would then pick the 
    # best-for-ase head out of a run whose whole point is a language-agnostic one. It is a balanced SUB-SAMPLE of dev, not all of dev: 
    # the realised counts are logged and stamped into the checkpoint (`pretrain_dev_mix`) because a monitor is only interpretable next 
    # to its dev set. Its chunks are drawn once from the seed and never rotate, so every epoch is scored on the identical chunks.
    dev_records, dev_mix = resolve_pretrain_records(cfg, data_cfg, language, "dev")
    if dev_mix: cfg["pretrain_dev_mix"] = dev_mix
    dev_dataset = ChunkDataset(dev_records, chunk_s, min_chunk_s, seed=seed, deterministic=True)

    # The pool balances records (clean segments); training exposure is time. Stamp the hours each language contributes to an epoch.
    hours: dict[str, float] = {}
    for rec, a, b in train_dataset.chunks: hours[rec.language] = hours.get(rec.language, 0.0) + (b - a) / 3600.0
    cfg["pretrain_hours"] = {k: round(v, 2) for k, v in sorted(hours.items())}
    print(f"bio_s1 | {len(train_dataset)} train chunks/epoch, {len(dev_dataset)} dev chunks "
          f"(U({min_chunk_s:g}, {chunk_s:g}) s), train hours {cfg['pretrain_hours']}", flush=True)
    num_workers = int(cfg.get("num_workers", 0))
    batch_size = dist.per_rank_batch_size(int(cfg.get("batch_size", 8)))
    train_loader = streaming_loader(
        train_dataset, batch_size, collate_windows, num_workers=num_workers,
        # Group same-length chunks (short videos) so a batch is not padded to a much longer neighbour 
        # (data.loader LengthBucketSampler). Same indices, same coverage — only the grouping changes.
        bucket_by_length=True, bucket_seed=seed,
    )
    dev_loader = streaming_loader(dev_dataset, batch_size, collate_windows, num_workers=num_workers)
    # Every S1 recipe has an explicit initialization, independent of the clean translator's per-language re-root.
    pretrained = cfg_get(cfg, "checkpoint", "from_pretrained", default="checkpoints/openasl_pose_only_slt.pth")
    model = build_bio_s1_model(cfg, pretrained_path=pretrained)
    return model, train_loader, dev_loader, cfg


def train_bio_s1_epochs(
    model: BioS1Model, train_loader: DataLoader, dev_loader: DataLoader, 
    device: torch.device, epochs: int, cfg: dict, resume: bool = False,
) -> int:
    dice_weight = float(cfg.get("dice_loss_weight", 1.5))
    class_weights = bio_class_weight_tensor(cfg.get("bio_class_weights"))
    if class_weights is not None: class_weights = class_weights.to(device)
    print("bio_s1 | monitor: complete-span F1@0.5, pooled dev-chunk counts, BIO Viterbi without duration scores; " \
          "deployment: semi-Markov Viterbi with the train-fitted duration prior", flush=True)
    # Head at learning_rate, the pretrained pose encoder at backbone_lr (both required keys of bio_pretrain.yaml).
    optimizer = build_optimizer(cfg, model.bio_head.parameters(), backbone_params=model.pose_encoder.parameters())

    def step_fn(batch, _epoch: int):
        out = model(batch["poses"], batch["frame_mask"], timestamps_s=batch["timestamps_s"])
        loss = bio_nll_dice_loss(out.logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=class_weights)
        return loss, {"bio_loss": float(loss.detach())}

    # Save the trained context and monitor decode with the S1 weights.
    training_cap_s = float(cfg["training_buffer_cap_s"])
    meta = {
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(train_loader.dataset.records),
        "language_fingerprints": cfg.get("language_fingerprints"),  # whole train split per language: require_segmenter_data
        "monitor_decode": "bio_viterbi", "monitor_protocol": "complete_spans_micro_at_0.5", 
        "rope_eval_chunk_s": training_cap_s, "buffer_cap_s": training_cap_s, 
        "initialization": cfg_get(cfg, "checkpoint", "from_pretrained", default="checkpoints/openasl_pose_only_slt.pth"),
        "bio_class_weights": cfg.get("bio_class_weights"), "language": cfg.get("language"),
        "bio_attention_radius_s": cfg.get("bio_attention_radius_s"),  # a mask, invisible to a strict load: require_attention_radius
        "pretrain_pool": pool_key(cfg), "pretrain_mix": cfg.get("pretrain_mix"), "pretrain_dev_mix": cfg.get("pretrain_dev_mix"),
        "pretrain_hours": cfg.get("pretrain_hours"), "min_chunk_s": cfg.get("training_min_chunk_s"),
        # --resume refuses a drifted rate or augmentation: the optimizer load would otherwise restore the saved rates silently.
        "learning_rate": float(cfg["learning_rate"]), "backbone_lr": float(cfg["backbone_lr"]), 
        "augmentation": cfg.get("augmentation"), "best_epoch_floor": int(cfg["best_epoch_floor"]),
    }
    # The end-of-training save in train.py reuses THIS dict. A second, independently-built meta drops pretrain_pool/pretrain_mix 
    # (disarming eval.py's provenance assertion) and re-derives rope_eval_chunk_s from the live inference.yaml, which may change
    # after training; the stamp records the context the head actually trained on.
    cfg["checkpoint_meta"] = meta
    return run_epoch_loop(
        name="bio_s1", model=model, loader=train_loader, optimizer=optimizer, device=device, epochs=epochs, cfg=cfg, 
        step_fn=step_fn, evaluate_fn=lambda e: evaluate_bio_chunks(model, dev_loader, device, dice_weight, class_weights),
        default_monitor="val_phrase_tiou_f1", default_mode="max", dev_loader=dev_loader, resume=resume, checkpoint_meta=meta,
        best_epoch_floor=meta["best_epoch_floor"],
    )
