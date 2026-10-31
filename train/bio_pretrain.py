"""S1 — multilingual segmentation pretraining for the in-system pose encoder and BIO head.

This is the deployed FSM head, not the external Moryossef segmenter. Their input spaces and checkpoints differ.

Two jobs:
  1. **gate warm start**: S2 starts from a sharp head, so the gate couples on-policy from step 0 (stage 2 
     refuses to run without `checkpoint.bio_head_init`).
  2. **BIO head init**: S2 loads `bio_head.*` (`checkpoint.bio_head_init` in dlm.yaml) & JOINTLY 
     fine-tunes it under the gate (S2).

Recipe (S1): whole-input random chunks, length U(δ + K·stride, C) with C = inference.yaml buffer_cap_s (data/chunks.py), 
and other input types that pretrain.inputs lists (restart chunks, edge-masked stage-2 windows), BIO only, on the balanced
pooled corpus. Dice(1.5) + balanced CE, the augmentation recipe shared with Moryossef arm, RoPE time and a banded attention
derived from the K-stride horizon and slowest train frame rate (`models.bio_head.band_radius_s`). The pose encoder trains 
at `backbone_lr`, the BIO head at `learning_rate`.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.batch import collate_windows
from data.chunks import ChunkDataset, shortest_chunk_s
from data.loader import (ANNOTATION_PROTOCOL, PooledEpochRecords, resolve_pretrain_records, streaming_loader, 
                         annotation_fingerprint, segmenter_language_fingerprints, cached_language_records)
from data.sampler import WindowSampler
from backbones import UniSignPoseEncoder
from models.bio_head import RoPEBIOHead, band_frames, band_radius_s
from models.unisign import released_layout_state
from models.checkpointing import load_state, load_checkpoint_meta, warn_pose_geometry
from poses.pose_io import POSE_GEOMETRY

from train.losses import bio_class_weight_tensor, bio_nll_dice_loss, resolve_bio_class_weights
from train.helpers import build_optimizer, evaluate_segmenter, run_epoch_loop, segmenter_monitor_config
from utils import cfg_get, checkpoint_dir, load_yaml, pool_key

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
        sd = load_state(ckpt_path)
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


def edge_reach(encoder: nn.Module, head: RoPEBIOHead) -> dict:
    """How far an input edge reaches into the head's outputs, read from the modules (data.chunks.edge_mask): `frames` by index
    through the temporal convs (the widest ST-GCN part chain of the Uni-Sign pose encoder the head reads, then the conv stem; each
    chain's convs run in series, and the parallel paths of an ST-GCN block, its residual and its gcn, have temporal kernel 1), then
    `depth` banded RoPE layers of `radius_s` each (None: the head has no band yet)."""
    def conv_frames(module: nn.Module) -> int:
        return sum((m.kernel_size[0] - 1) // 2 * m.dilation[0] for m in module.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d)))
    frames = max(conv_frames(encoder.gcn_modules[p]) + conv_frames(encoder.fusion_gcn_modules[p]) for p in encoder.gcn_modules)
    return {"frames": frames + conv_frames(head.conv_stem), "radius_s": head.attention_radius_s, "depth": len(head.layers)}


def require_attention_radius(meta: dict, expected: float | None, source: str) -> None:
    """Refuse an S1 warm start trained under another attention band. The band is a mask, not a weight, so a strict state_dict
    load cannot detect it; a head read at another radius sees contexts it never trained on. No stamp = full attention."""
    trained = meta.get("bio_attention_radius_s")
    if trained is None and expected is None: return
    if trained is not None and expected is not None and abs(float(trained) - float(expected)) < 1e-9: return
    raise SystemExit(
        f"{source} was trained with bio_attention_radius_s={trained} (None = full attention), but this run derives {expected} "
        f"(models.bio_head.band_radius_s). Start S1 from a file trained under the same band, or from the released weights."
    )


def slowest_train_fps(cfg: dict, data_cfg: dict, train_records: list) -> float:
    """The slowest native frame rate of the train streams the head serves: the whole train split of every pooled language (a pooled
    epoch draws a sub-sample), else the target language's. It is frames / a duration_s rounded to 1 ms, so it is exact to a few
    thousandths of a frame per second on a video of some seconds; to 0.01 fps it reads the nominal rate."""
    langs = cfg_get(cfg, "pretrain", "languages")
    records = [r for l in langs for r in cached_language_records(data_cfg, str(l), "train")] if langs else train_records
    return round(min(float(r.pose.fps) for r in records), 2)


def require_segmenter_data(meta: dict, known: dict[str, str], buffer_cap_s: float, source: str) -> None:
    """Refuse a segmenter (S1 or Moryossef) trained under another annotation protocol or on chunks shorter than the deployed cap;
    warn when its train labels are not the current ones. A re-convert, a person-counts re-measure or a caption rule changes the
    labels, and a strict state_dict load cannot see that. `known`: the current fingerprints of the corpora that the run reads anyway
    (stage 2: its target; an evaluation: its source corpus); only those stamps are compared, so no other corpus of the pool loads, and
    a machine that holds only the run's corpus can run it. A label change of another pooled corpus is not seen here: train the
    segmenter again after any label change of its pool (README "After a change")."""
    if meta.get("annotation_protocol") != ANNOTATION_PROTOCOL: raise SystemExit(
        f"{source} was trained under annotation protocol {meta.get('annotation_protocol')!r}, not {ANNOTATION_PROTOCOL!r}: retrain it."
    )
    stamped = meta.get("language_fingerprints")
    if stamped is None: print(f"WARNING: {source} stamps no language_fingerprints, so its train labels cannot be compared.", flush=True)
    elif stale := [corpus for corpus, now in known.items() if corpus in stamped and stamped[corpus] != now]: print(
        f"WARNING: {source} was trained on other {', '.join(stale)} train labels (a re-convert, re-measured "
        f"cuts or a caption rule changed them). Retrain it before a run you report.", flush=True
    )
    trained_chunk = meta.get("rope_eval_chunk_s")
    if trained_chunk is not None and float(trained_chunk) + 1e-6 < float(buffer_cap_s): raise SystemExit(
        f"{source} trained on chunks of at most {float(trained_chunk):.2f}s, "
        f"below the deployed buffer_cap_s {float(buffer_cap_s):.2f}s: retrain it."
    )


def build_bio_s1_model(cfg: dict, attention_radius_s: float | None, pretrained_path: str | None = None) -> BioS1Model:
    # Construct S1 with the head's attention band (derived in build_bio_s1, or a checkpoint's stamp); optionally initialize from
    # released pose weights or a complete S1 checkpoint. The pose encoder trains with the head, so `bio_head_init` carries an
    # ADAPTED segmentation encoder into stage 2.
    model = BioS1Model(
        pose_hidden_dim=int(cfg.get("pose_hidden_dim", 256)), feat_dim=int(cfg.get("feat_dim", 768)),
        bio_hidden_dim=int(cfg["bio"]["hidden_dim"]), bio_depth=int(cfg["bio"]["depth"]),
        bio_nhead=int(cfg["bio"]["nhead"]), bio_dropout=float(cfg["bio"]["dropout"]),
        bio_conv_stem_layers=int(cfg["bio"]["conv_stem_layers"]), bio_attention_radius_s=attention_radius_s,
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
    # translation stays monolingual in stage 2. `pretrain.languages: null` = the target language alone.
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
    # Chunk length C is the longest context the head trains on: the deployed cap (inference.yaml buffer_cap_s), at which the 
    # stage-2 head and the FSM run. Checkpoint metadata and whole-video evaluation use this stamped value, not a target 
    # inference file that may change later.
    cfg["training_buffer_cap_s"] = chunk_s = float(inference_cfg["buffer_cap_s"])
    # Shortest interior chunk = the shortest buffer the FSM decides on after a commit (data.chunks.shortest_chunk_s, the law stage 2's
    # chunks share). A chunk is a time span over pooled records, so δ stays in seconds here (FSM rounds it up to whole frames of each stream).
    stability = inference_cfg["boundary_stability"]
    cfg["training_min_chunk_s"] = min_chunk_s = shortest_chunk_s(inference_cfg)
    seed = int(cfg.get("seed", 42))
    # A pooled run re-draws its balanced sub-sample each epoch, so the videos a sub-sampled corpus contributes
    # ROTATE and the whole corpus is covered across epochs. Monolingual runs pass no provider.
    pool = PooledEpochRecords(cfg, data_cfg, language) if pretrain_mix else None
    # No epoch before the rotation's first full cycle can become the best checkpoint (train.helpers.TrainControl).
    cfg["best_epoch_floor"] = pool.cycle_epochs() if pool else 0
    # Every S1 recipe has an explicit initialization, independent of the clean translator's per-language re-root.
    pretrained = cfg_get(cfg, "checkpoint", "from_pretrained", default="checkpoints/openasl_pose_only_slt.pth")
    # The attention band (spec §1.4 rule 16): the widest whole-frame band whose reach, through THIS model's temporal convs and
    # bio_depth banded layers, stays within the (K - 1) strides before a commit at the slowest train rate (faster rates fit too).
    # A model built only to read its modules gives that reach; the checkpoint stamps the band, and stage 2 and eval read the stamp.
    probe = build_bio_s1_model(cfg, None)
    reach = edge_reach(probe.pose_encoder, probe.bio_head)
    fps = slowest_train_fps(cfg, data_cfg, train_records)
    horizon_s = (int(stability["hysteresis_strides"]) - 1) * float(inference_cfg["stride_s"])
    cfg["bio_attention_radius_s"] = radius_s = band_radius_s(reach["frames"], reach["depth"], horizon_s, fps)
    print(f"bio_s1 | attention band +-{radius_s:.4f} s ({band_frames(radius_s, fps)} frames at the slowest train rate {fps:g} fps): "
          f"reach {reach['frames']} conv frames + {reach['depth']} bands <= {horizon_s:g} s", flush=True)
    model = build_bio_s1_model(cfg, radius_s, pretrained_path=pretrained)
    # The train set's input types (pretrain.inputs, data.chunks.ChunkDataset): windows follow the stage-2 window law of dlm.yaml
    # and mask the frames that read a window edge, as far as THIS model's modules reach; restart chunks start δ before a terminator.
    inputs = list(cfg["pretrain"]["inputs"])
    windows = "windows" in inputs
    train_dataset = ChunkDataset(
        train_records, chunk_s, min_chunk_s, augmentation=cfg["augmentation"], seed=seed, records_for_epoch=pool, inputs=inputs,
        delta_s=float(stability["delta_s"]), edge_reach=edge_reach(model.pose_encoder, model.bio_head) if windows else None,
        window_sampler=WindowSampler.from_slt_config(train_records, cfg, inference_cfg) if windows else None,
    )
    # Dev holds every record of each dev corpus (pretrain.dev_languages, default the pool; `load_multilingual_records` pools dev as-is).
    # Checkpoint selection averages the per-language main score (`val_video_f1`), so the largest corpus does not outweigh the others; the
    # dev-chunk diagnostics (`val_bio_*`,
    # `val_phrase_*`) pool all chunks, so ase dominates them. The realised counts are stamped (`pretrain_dev_mix`), because a monitor
    # is only interpretable next to its dev set. Its chunks are drawn once from the seed and never rotate, so every epoch is scored on
    # the identical chunks.
    dev_records, dev_mix = resolve_pretrain_records(cfg, data_cfg, language, "dev")
    if dev_mix: cfg["pretrain_dev_mix"] = dev_mix
    dev_dataset = ChunkDataset(dev_records, chunk_s, min_chunk_s, seed=seed, deterministic=True)
    # Checkpoint selection scores these dev records as whole streams under the scoring of the rows and the decode of segmenter_decode
    # (train.helpers.evaluate_segmenter).
    cfg["video_monitor"] = segmenter_monitor_config(cfg, data_cfg, dev_records)

    # The pool balances hours (data.loader._pool_targets). Stamp the hours each language contributes to an epoch as the tiling made them.
    hours: dict[str, float] = {}
    for rec, a, b, _, _ in train_dataset.chunks: hours[rec.language] = hours.get(rec.language, 0.0) + (b - a) / 3600.0
    cfg["pretrain_hours"] = {k: round(v, 2) for k, v in sorted(hours.items())}
    kinds = Counter(kind for _, _, _, kind, _ in train_dataset.chunks)
    print(f"bio_s1 | {len(train_dataset)} train inputs/epoch {dict(kinds)}, {len(dev_dataset)} dev chunks "
          f"(U({min_chunk_s:g}, {chunk_s:g}) s), train hours {cfg['pretrain_hours']}", flush=True)
    num_workers = int(cfg.get("num_workers", 0))
    train_loader = streaming_loader(   # batch_size: the global batch (data.loader.streaming_loader)
        train_dataset, int(cfg.get("batch_size", 8)), collate_windows, num_workers=num_workers,
        # Group same-length chunks (short videos) so a batch is not padded to a much longer neighbour 
        # (data.loader LengthBucketSampler). Same indices, same coverage — only the grouping changes.
        bucket_by_length=True, bucket_seed=seed,
    )
    dev_loader = streaming_loader( # checkpoint.dev_batch_size; every rank reads all dev chunks (data.loader.streaming_loader)
        dev_dataset, int(cfg["checkpoint"]["dev_batch_size"]), collate_windows, num_workers=num_workers
    )
    return model, train_loader, dev_loader, cfg


def train_bio_s1_epochs(
    model: BioS1Model, train_loader: DataLoader, dev_loader: DataLoader, 
    device: torch.device, epochs: int, cfg: dict, resume: bool = False,
) -> int:
    dice_weight = float(cfg["bio"]["dice_loss_weight"])
    class_weights = bio_class_weight_tensor(cfg["bio"]["class_weights"])
    if class_weights is not None: class_weights = class_weights.to(device)
    edge = cfg["video_monitor"]["edge"]
    print("bio_s1 | monitor: val_video_f1, the main score of a row on whole dev streams (decode "
          f"{cfg['video_monitor']['decode']}): the F1 of the tIoU x edge-limit grid at tau avg (edge taus {edge['edge_taus_s']} s) and tIoU "
          f"avg (thresholds {edge['tiou_thresholds']}), then the mean over languages", flush=True)
    # Head at learning_rate, the pretrained pose encoder at backbone_lr (both required keys of bio_pretrain.yaml).
    optimizer = build_optimizer(cfg, model.bio_head.parameters(), backbone_params=model.pose_encoder.parameters())

    def step_fn(batch, _epoch: int):
        out = model(batch["poses"], batch["frame_mask"], timestamps_s=batch["timestamps_s"])
        loss = bio_nll_dice_loss(out.logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=class_weights)
        return loss, {"bio_loss": float(loss.detach())}

    # Save the trained context and monitor decode with the S1 weights.
    training_cap_s = float(cfg["training_buffer_cap_s"])
    init = cfg_get(cfg, "checkpoint", "from_pretrained", default="checkpoints/openasl_pose_only_slt.pth")   # build_bio_s1 loaded it
    warn_pose_geometry(load_checkpoint_meta(init) if init else {}, init)   # a trained warm start that read another geometry
    meta = {
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(train_loader.dataset.records),
        "language_fingerprints": cfg.get("language_fingerprints"),  # whole train split per language: require_segmenter_data
        # The frame geometry of the poses it read (poses.pose_io.load_pose_frames); eval warns on any other value.
        "pose_geometry": POSE_GEOMETRY,
        "monitor_decode": cfg["video_monitor"]["decode"], "monitor_protocol": "edge_chunk_grid_f1_complete_spans", "monitor_edge": cfg["video_monitor"]["edge"],
        "rope_eval_chunk_s": training_cap_s, "buffer_cap_s": training_cap_s, 
        "initialization": init,
        "bio_class_weights": cfg["bio"]["class_weights"], "language": cfg.get("language"),
        "bio_attention_radius_s": cfg["bio_attention_radius_s"],  # derived; a mask, invisible to a strict load: stage 2 and eval read it
        "pretrain_pool": pool_key(cfg), "pretrain_mix": cfg.get("pretrain_mix"), "pretrain_dev_mix": cfg.get("pretrain_dev_mix"),
        "pretrain_hours": cfg.get("pretrain_hours"), "min_chunk_s": cfg.get("training_min_chunk_s"),
        # The pool's share rule and the corpora that choose the checkpoint (data.loader.resolve_pretrain_records).
        "pretrain_weights": cfg_get(cfg, "pretrain", "weights"), "pretrain_dev_languages": cfg_get(cfg, "pretrain", "dev_languages"),
        # The train set's input types, and with windows their window law: --resume refuses a change, eval prints the list.
        "pretrain_inputs": list(cfg["pretrain"]["inputs"]),
        "pretrain_window_law": {k: cfg[k] for k in ("mode_ratios", "mode2_subcase_weights", "jitter")}
                               if "windows" in cfg["pretrain"]["inputs"] else None,
        # --resume refuses a drifted rate or augmentation: the optimizer load would otherwise restore the saved rates silently.
        "learning_rate": float(cfg["learning_rate"]), "backbone_lr": float(cfg["backbone_lr"]),
        "augmentation": cfg.get("augmentation"), "best_epoch_floor": int(cfg["best_epoch_floor"]), "seed": cfg.get("seed"),
    }
    # The end-of-training save in train.py reuses THIS dict. A second, independently-built meta drops pretrain_pool/pretrain_mix 
    # (disarming the provenance assertion in evaluation/systems.py) and re-derives rope_eval_chunk_s from the live inference.yaml, 
    # which may change after training; the stamp records the context the head actually trained on.
    cfg["checkpoint_meta"] = meta
    return run_epoch_loop(
        name="bio_s1", model=model, loader=train_loader, optimizer=optimizer, device=device, epochs=epochs, cfg=cfg, 
        step_fn=step_fn, evaluate_fn=lambda e: evaluate_segmenter(
            model, dev_loader, device, dice_weight, class_weights, cfg["video_monitor"], 
            velocity=False, rope_chunk_s=training_cap_s,
        ),
        default_monitor="val_video_f1", default_mode="max", dev_loader=dev_loader, resume=resume, checkpoint_meta=meta,
        best_epoch_floor=meta["best_epoch_floor"],
    )
