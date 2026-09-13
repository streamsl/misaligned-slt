"""S1 — multilingual segmentation pretraining for the in-system pose encoder and BIO head.

This is the deployed FSM head, not the external Moryossef segmenter. Their input spaces and checkpoints differ.

Two jobs:
  1. **gate warm start**: S2 starts from sharp head, so the gate couples on-policy from step 1 with
     `membership_gate.warmup_epochs: 0` (no garbage-conditioning warmup).
  2. **BIO head init**: S2 loads `bio_head.*` (`checkpoint.bio_head_init` in dlm.yaml) & JOINTLY 
     fine-tunes it under the gate (§1.4 S2).

Recipe (§1.4 S1): use StreamingWindowDataset, the designed pooled corruption distribution, Dice(1.5) + balanced
CE, fps augmentation and RoPE time. The pose encoder trains at a lower learning rate than the BIO head.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.batch import WindowCollator
from data.loader import (
    ANNOTATION_PROTOCOL, StreamingWindowDataset, annotation_fingerprint, 
    assert_pool_safe, resolve_pretrain_records, sentence_p99_s, streaming_loader
)
from backbones import UniSignPoseEncoder
from models.bio_head import RoPEBIOHead
from models.unisign import released_layout_state
from infer.duration_decode import DurationDecoder
from models.checkpointing import _load_state

from train import distributed as dist
from train.helpers import build_optimizer, eval_mode, mean_logs, run_epoch_loop
from train.losses import bio_class_weight_tensor, bio_nll_dice_loss, resolve_bio_class_weights
from metrics import bio_frame_metrics, CompleteSpanMetrics
from utils import checkpoint_dir, load_yaml, pool_key, pretrained_checkpoint, resolve_inference

PRETRAIN_CONTEXT_MARGIN_S = 1.0  # floor for the δ/fps term below, so a language with no measured δ still gets headroom

class BioS1Model(nn.Module): # Pose backbone and temporal BIO classifier shared with the joint model's segmentation branch.
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
            input_dim=int(feat_dim), hidden_dim=int(bio_hidden_dim), depth=int(bio_depth), nhead=int(bio_nhead), 
            dropout=float(bio_dropout), num_classes=4, conv_stem_layers=int(bio_conv_stem_layers),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder: self.pose_encoder.eval()  # BN running stats pinned to the released checkpoint
        return self

    def load_pretrained(self, ckpt_path: str | Path) -> int:
        sd = _load_state(ckpt_path)
        if any(k.startswith("bio_head.") for k in sd):
            self.load_state_dict(sd, strict=True)
            return len(sd)
        sd = released_layout_state(sd)
        pose_sd = {k: v for k, v in sd.items() if not k.startswith("mt5_model.")}
        self.pose_encoder.load_state_dict(pose_sd, strict=True)
        return len(pose_sd)

    def forward(self, poses, frame_mask, timestamps_s=None):
        if self.freeze_encoder:
            with torch.no_grad(): feats = self.pose_encoder(poses, frame_mask)
        else: feats = self.pose_encoder(poses, frame_mask)
        return self.bio_head(feats, timestamps_s=timestamps_s, frame_mask=frame_mask)


def resolve_pretrain_context(cfg: dict, data_cfg: dict, inference_cfg: dict, language: str | None = None) -> dict[str, float] | None:
    """`pretrain_geometry.buffer_cap_s: auto` -> max over S1 languages (the pool, or `language` alone) of the DEPLOYED cap rule.

    The head is never run beyond its trained RoPE context, and `analyze.py --stage buffer-cap` refuses a cap above it, so this must 
    compute SAME terms that stage writes: train p99 + stride + δ/fps per language (utils.lambda_min_frames' δ, read from resolved 
    geometry). A fixed 1s margin silently under-covers any language whose δ exceeds 1s. Return the per-language terms, or None when 
    the cap is numeric (an explicit design override).
    """
    geometry = dict(cfg.get("pretrain_geometry") or {})
    langs = cfg.get("pretrain_languages") or ([language] if language else None)
    if not langs or str(geometry.get("buffer_cap_s", "")).lower() != "auto": return None
    stride_s = float(inference_cfg.get("stride_s", 1.0))
    languages = (data_cfg.get("languages") or {})
    deltas = (inference_cfg.get("boundary_stability", {}) or {}).get("delta_enc_frames", {})

    def margin(lang: str) -> float:
        delta = deltas.get(lang) if isinstance(deltas, dict) else deltas
        if delta is None: return PRETRAIN_CONTEXT_MARGIN_S  # δ not measured yet: the floor is the only headroom
        fps = float(((languages.get(lang) or {}).get("pose") or {}).get("fps", 24.0))
        return max(PRETRAIN_CONTEXT_MARGIN_S, float(delta) / max(fps, 1.0))
    
    caps = {lang: round(p99 + stride_s + margin(lang), 2)
            for lang, p99 in sentence_p99_s(data_cfg, [str(x) for x in langs], split="train").items()}
    geometry["buffer_cap_s"] = max(caps.values())
    cfg["pretrain_geometry"] = geometry
    return caps


def build_bio_s1_model(cfg: dict, pretrained_path: str | None = None) -> BioS1Model:
    # Construct S1; optionally initialize from released pose weights or a complete S1 checkpoint.
    # Inherited dlm.yaml `freeze_backbone`. The SHIPPED S1 recipe is false — the encoder trains, so `bio_head_init`
    # can carry an ADAPTED encoder into stage 2. The `True` default here is for a config that omits the key entirely
    # (the frozen-encoder ablation), not for bio_pretrain.yaml, which sets it explicitly.
    freeze_encoder = bool(cfg.get("freeze_backbone", True))
    model = BioS1Model(
        pose_hidden_dim=int(cfg.get("pose_hidden_dim", 256)), feat_dim=int(cfg.get("feat_dim", 768)),
        bio_hidden_dim=int(cfg.get("bio_hidden_dim", 384)), bio_depth=int(cfg.get("bio_depth", 4)),
        bio_nhead=int(cfg.get("bio_nhead", 8)), bio_dropout=float(cfg.get("bio_dropout", 0.1)),
        bio_conv_stem_layers=int(cfg.get("bio_conv_stem_layers", 2)), freeze_encoder=freeze_encoder,
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
    inference_cfg = resolve_inference(load_yaml(inference_config), language, strict=False)

    # Segmentation is language-agnostic (boundaries are prosodic), so S1 may pretrain on a pool of languages;
    # translation stays monolingual in stage 2. `pretrain_languages: null` = the target language alone.
    assert_pool_safe(cfg)
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
        
    context_caps = resolve_pretrain_context(cfg, data_cfg, inference_cfg, language)
    if context_caps:
        cfg["pretrain_context_caps"] = context_caps   # per-language train p99 + stride + margin, recorded for the paper
        print(f"bio_s1 | pretrain_geometry.buffer_cap_s auto -> {cfg['pretrain_geometry']['buffer_cap_s']:.2f}s "
              f"(train p99 + stride + {PRETRAIN_CONTEXT_MARGIN_S:g} s per language: {context_caps})", flush=True)
    resolve_bio_class_weights(cfg, train_records)
    # A pooled run re-draws its balanced sub-sample each epoch, so the videos a sub-sampled corpus contributes
    # ROTATE and the whole corpus is covered across epochs. Monolingual runs pass no provider and are unchanged.
    train_dataset = StreamingWindowDataset(
        train_records, slt_cfg=cfg, inference_cfg=inference_cfg, pose_augment_cfg=cfg.get("augmentation"),
        records_for_epoch=(lambda e: resolve_pretrain_records(cfg, data_cfg, language, "train", epoch=e)[0]) if pretrain_mix else None
    )
    # Record the sampler's resolved geometry once. Checkpoint metadata and whole-video evaluation must use the
    # context the head actually trained on, not a target inference file that may change later.
    cfg["training_buffer_cap_s"] = float(train_dataset.sampler.buffer_cap_s)
    cfg["training_min_span_frames"] = int(train_dataset.sampler.min_span_frames)
    # Dev is drawn via the SAME balancing rule as train (`load_multilingual_records`), so the monitor measures what training optimises 
    # rather than the corpus-size prior. A pooled dev taken AS-IS would be ~84% ase, and best-checkpoint selection would then pick the 
    # best-for-ase head out of a run whose whole point is a language-agnostic one. It is a balanced SUB-SAMPLE of dev, not all of dev: 
    # the realised counts are logged and stamped into the checkpoint (`pretrain_dev_mix`) because a monitor is only interpretable next 
    # to its dev set. It never rotates (no `records_for_epoch`, and dev datasets are deterministic), so every epoch is scored on the
    # identical windows — a rotating dev would make "best epoch" partly a draw.
    dev_records, dev_mix = resolve_pretrain_records(cfg, data_cfg, language, "dev")
    if dev_mix: cfg["pretrain_dev_mix"] = dev_mix
    dev_steps = sum(sum(1 for sp in r.sentences if getattr(sp, 'reliable', True)) for r in dev_records)
    dev_dataset = StreamingWindowDataset(
        dev_records, slt_cfg=cfg, inference_cfg=inference_cfg, steps_per_epoch=max(dev_steps, 1), deterministic=True,
    )
    collator = WindowCollator(tokenizer=None)  # BIO-only: no text tokenization
    num_workers = int(cfg.get("num_workers", 0))
    
    train_loader = streaming_loader(
        train_dataset, dist.per_rank_batch_size(int(cfg.get("batch_size", 8))), collator, num_workers=num_workers,
        # Group same-length windows so a batch is not padded to a much longer neighbour (data.loader
        # LengthBucketSampler). Same indices, same once-per-epoch coverage — only the grouping changes.
        bucket_by_length=bool(cfg.get("bucket_by_length", True)), bucket_seed=int(cfg.get("seed", 42))
    )
    dev_loader = streaming_loader(
        dev_dataset, dist.per_rank_batch_size(int(cfg.get("batch_size", 8))), collator, num_workers=num_workers
    )
    # Every S1 recipe has an explicit initialization, independent of the clean translator's per-language re-root.
    pretrained = pretrained_checkpoint(cfg, default="checkpoints/openasl_pose_only_slt.pth")
    model = build_bio_s1_model(cfg, pretrained_path=pretrained)
    return model, train_loader, dev_loader, cfg


@torch.no_grad()
def evaluate_bio_s1( # Evaluate frame losses and the untuned legal-path monitor before duration calibration.
    model: BioS1Model, loader: DataLoader, device: torch.device, dice_weight: float, class_weights: torch.Tensor | None, 
) -> dict[str, float]:
    rows, per_mode = [], {}
    spans = CompleteSpanMetrics()
    with eval_mode(model):
        for batch in loader:
            poses, mask = batch["poses"].to(device), batch["frame_mask"].to(device)
            ts, labels = batch["timestamps_s"].to(device), batch["bio_labels"].to(device)
            out = model(poses, mask, timestamps_s=ts)
            lengths = mask.long().sum(1)
            tags = DurationDecoder().decode(out.logits, lengths)
            row = {"bio_loss": float(bio_nll_dice_loss(out.logits, labels, dice_weight=dice_weight, class_weights=class_weights))}
            row.update(bio_frame_metrics(out.logits, labels, prefix="bio"))
            rows.append(row)
            spans.update(tags, labels, lengths)
            modes = batch.get("mode_names") or []
            for mode in set(modes) - {"mode4"}:
                idx = [i for i, m in enumerate(modes) if m == mode]
                per_mode.setdefault(mode, CompleteSpanMetrics()).update(tags[idx], labels[idx], lengths[idx])
    result = mean_logs(rows, prefix="val")
    result.update(spans.compute(prefix="val_phrase"))
    for mode, score in per_mode.items(): result[f"val_{mode}_tiou_f1"] = score.compute()["phrase_tiou_f1"]
    return result


def train_bio_s1_epochs(
    model: BioS1Model, train_loader: DataLoader, dev_loader: DataLoader, 
    device: torch.device, epochs: int, cfg: dict, resume: bool = False,
) -> list[dict[str, float]]:
    dice_weight = float(cfg.get("dice_loss_weight", 1.5))
    class_weights = bio_class_weight_tensor(cfg.get("bio_class_weights"))
    if class_weights is not None: class_weights = class_weights.to(device)
    # Frozen encoder → head only. Unfrozen → head at learning_rate, the pretrained encoder at backbone_lr.
    print("bio_s1 | monitor: complete-span F1@0.5, pooled window counts, BIO Viterbi without duration scores; " \
          "deployment: calibrated semi-Markov Viterbi", flush=True)
    if model.freeze_encoder: optimizer = build_optimizer(cfg, model.bio_head.parameters())
    else: optimizer = build_optimizer(cfg, model.bio_head.parameters(), backbone_params=model.pose_encoder.parameters())

    def step_fn(batch, _epoch: int):
        out = model(batch["poses"], batch["frame_mask"], timestamps_s=batch["timestamps_s"])
        loss = bio_nll_dice_loss(out.logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=class_weights)
        return loss, {"bio_loss": float(loss.detach())}

    # Save the trained context and monitor decode with the S1 weights.
    training_cap_s = float(cfg["training_buffer_cap_s"])
    meta = {
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(train_loader.dataset.records),
        "monitor_decode": "bio_viterbi", "monitor_protocol": "complete_spans_micro_at_0.5", "rope_eval_chunk_s": training_cap_s, 
        "buffer_cap_s": training_cap_s, "initialization": pretrained_checkpoint(cfg, default="checkpoints/openasl_pose_only_slt.pth"),
        "bio_class_weights": cfg.get("bio_class_weights"), "language": cfg.get("language"),
        "pretrain_pool": pool_key(cfg), "pretrain_mix": cfg.get("pretrain_mix"), "pretrain_dev_mix": cfg.get("pretrain_dev_mix")
    }
    # The end-of-training save in train.py reuses THIS dict. A second, independently-built meta drops pretrain_pool/pretrain_mix 
    # (disarming eval.py's provenance assertion) and re-derives rope_eval_chunk_s from the live inference.yaml — which is the value 
    # the stamp exists to override, since `analyze --stage buffer-cap --write-config` rewrites buffer_cap_s after training.
    cfg["checkpoint_meta"] = meta
    return run_epoch_loop(
        name="bio_s1", model=model, loader=train_loader, optimizer=optimizer, device=device, epochs=epochs, cfg=cfg, step_fn=step_fn, 
        evaluate_fn=lambda e: evaluate_bio_s1(model, dev_loader, device, dice_weight, class_weights),
        default_monitor="val_mode3_tiou_f1", default_mode="max", dev_loader=dev_loader, resume=resume, checkpoint_meta=meta
    )
