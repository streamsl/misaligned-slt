"""S1 — multilingual segmentation pretraining for the in-system pose encoder and BIO head.

This is the deployed FSM head, not the external Moryossef segmenter. Their input spaces and checkpoints differ.

Two jobs:
  1. **gate warm start**: S2 starts from a sharp head, so the gate couples on-policy from step one with
     `membership_gate.warmup_epochs: 0` (no garbage-conditioning warmup).
  2. **BIO head init**: S2 loads `bio_head.*` (`checkpoint.bio_head_init` in dlm.yaml) and JOINTLY fine-tunes it
     under the gate (§1.4 S2; frozen-BIO is an ablation only).

Recipe (§1.4 S1): use StreamingWindowDataset, the designed pooled corruption distribution, Dice(1.5) + balanced
CE, fps augmentation and RoPE time. The pose encoder trains at a lower learning rate than the BIO head.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.windowing import BIO
from data.batch import WindowCollator
from data.loader import StreamingWindowDataset, assert_pool_safe, resolve_pretrain_records, streaming_loader
from backbones import UniSignPoseEncoder
from models.bio_head import RoPEBIOHead
from models.unisign import released_layout_state

from train import distributed as dist
from train.helpers import build_optimizer, eval_mode, mean_logs, run_epoch_loop
from train.losses import bio_class_weight_tensor, bio_nll_dice_loss, resolve_bio_class_weights
from infer.duration_decode import deployed_decode_tags, duration_decode_params, fit_duration_prior
from metrics import bio_frame_metrics, moryossef_segment_metrics
from utils import checkpoint_dir, load_yaml, pool_key, pretrained_checkpoint, resolve_inference, resolve_pretrained


class BioS1Model(nn.Module):
    """Uni-Sign pose encoder + trainable RoPE BIO head.

    `bio_head` matches `MisalignedSLTModel.bio_head`, so this checkpoint's keys load directly as the SLT-model init.

    `freeze_encoder=False` (`freeze_backbone: false` — the SHIPPED recipe): the encoder trains too, adapting translation-optimized 
    features to segmentation. This is what makes multilingual pretraining a REPRESENTATION and not just a ~6.7M-parameter head, 
    and stage 2 loads this encoder alongside the head, so "S1 features == S2 initial features" still holds. It does NOT recover 
    back-to-back sentence boundaries on YouTube corpora — that signal is absent from the captions themselves. It also means the 
    gated arms and the clean floor no longer share a pose encoder — deliberately: the arms load this one so their BIO head meets 
    the features it trained on, while the floor stays the faithful Uni-Sign transfer it is supposed to anchor.

    `freeze_encoder=True` (the ablation): encoder frozen AND pinned to eval (`train()` override) so ST-GCN BatchNorm running stats 
    stay the released checkpoint's — no input-distribution jump, and 2 arms then share an encoder by construction.
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
        if self.freeze_encoder: self.pose_encoder.eval()  # BN running stats pinned to the released checkpoint
        return self

    def load_unisign_pose(self, ckpt_path: str | Path) -> int:
        blob = torch.load(str(ckpt_path), map_location="cpu")
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        sd = released_layout_state(sd)  # accepts released blobs AND trainer model.pt (baseline_train re-rooting)
        pose_sd = {k: v for k, v in sd.items() if not k.startswith("mt5_model.")}
        self.pose_encoder.load_state_dict(pose_sd, strict=True)
        return len(pose_sd)

    def forward(self, poses, frame_mask, timestamps_s=None):
        if self.freeze_encoder:
            with torch.no_grad(): feats = self.pose_encoder(poses, frame_mask)
        else: feats = self.pose_encoder(poses, frame_mask)
        return self.bio_head(feats, timestamps_s=timestamps_s, frame_mask=frame_mask)


def build_bio_s1_model(cfg: dict, pretrained_path: str | None = None) -> BioS1Model:
    """Construct BioS1Model and load the Uni-Sign pose encoder.

    SINGLE source of truth for the head shape: training (build_bio_s1) and inference (analyze.segmenter_infer) both route here, 
    so the S1 checkpoint always strict-loads and `bio_head_init` shape parity holds by construction. `pretrained_path` overrides 
    which released checkpoint the frozen encoder loads (per-language warm-start, resolved by the caller).
    """
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
    pose_ckpt = pretrained_path or pretrained_checkpoint(cfg, default="checkpoints/openasl_pose_only_slt.pth")
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
    # Precedence: CLI --language > config `language:` > data.yaml active_languages. Reload only when it changes, 
    # so ${language} in checkpoint.dir re-points to the right dataset dir.
    _requested_language = language   # raw CLI value, before defaulting (pooled runs refuse it)
    language = str(language or cfg.get("language") or data_cfg.get("active_languages", ["asf"])[0])
    if language != cfg.get("language"): cfg = load_yaml(config, language=language)
    inference_cfg = resolve_inference(load_yaml(inference_config), language, strict=False)
    # train_bio_s1_epochs re-reads these for the monitor's duration prior; record the CLI paths so a run with
    # non-default configs monitors under the same decode/data as the sampler it just built.
    cfg["inference_config"], cfg["data_config"] = str(inference_config), str(data_config)

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
    if cfg.get("pretrain_mix"): # Multilingual pool: warm-start from Uni-Sign released checkpoint, not a per-language fine-tune.
        # resolve_pretrained is per-language, and after the re-root, it points at baseline_train/<lang> — an encoder already 
        # adapted to ONE target language, which would (a) privilege that language inside a language-agnostic pretraining and (b) 
        # make the pool checkpoint depend on which --language launched it while writing to the same multi_* directory. S1 features 
        # == S2 initial features is preserved by bio_head_init carrying THIS encoder into stage 2, not by matching warm-starts.
        pretrained = pretrained_checkpoint(cfg, default="checkpoints/openasl_pose_only_slt.pth")
    else: # Monolingual: the language's configured Uni-Sign checkpoint, matching the SLT model's warm-start.
        pretrained = resolve_pretrained(cfg, data_cfg, language, default="checkpoints/openasl_pose_only_slt.pth")
    model = build_bio_s1_model(cfg, pretrained_path=pretrained)
    return model, train_loader, dev_loader, cfg


@torch.no_grad()
def evaluate_bio_s1(
    model: BioS1Model, loader: DataLoader, device: torch.device,
    dice_weight: float, class_weights: torch.Tensor | None, duration_prior=None,
) -> dict[str, float]:
    """`duration_prior` (inference.yaml duration_decode): score the DEPLOYED decoder, not raw argmax.

    Without it the monitor measures argmax's split rate — the very quantity the semi-Markov decode REPLACES with the duration 
    prior at deployment (infer/duration_decode.py). A checkpoint that argmax-splits more then ranks higher while contributing 
    nothing downstream, which is how a rising monitor coexists with falling whole-video test numbers. Same rule already 
    applied to `analyze --stage delta-enc`: measure under the decode you deploy.
    """
    rows = []
    with eval_mode(model):
        for batch in loader:
            poses = batch["poses"].to(device)
            mask = batch["frame_mask"].to(device)
            ts = batch["timestamps_s"].to(device)
            labels = batch["bio_labels"].to(device)
            out = model(poses, mask, timestamps_s=ts)
            # Loss + FRAME metrics read raw logits (they score the objective and per-frame quality); SEGMENT metrics read `seg_logits`, 
            # deployed decode's tags, so the monitor ranks what deployment produces. Segment metrics score DEPLOYED tags (same function 
            # the FSM and gate use); loss + frame metrics keep raw logits. One-hot so every metric call is unchanged — they all argmax.
            seg_logits = out.logits
            if duration_prior is not None:
                tags = deployed_decode_tags(out.logits, mask.long().sum(dim=1), duration_prior, ts, batch.get("commit_mask"))
                seg_logits = torch.nn.functional.one_hot(tags.clamp(min=0), num_classes=out.logits.shape[-1]).to(out.logits.dtype)

            row = {"bio_loss": float(bio_nll_dice_loss(out.logits, labels, dice_weight=dice_weight, class_weights=class_weights))}
            row.update(bio_frame_metrics(out.logits, labels, prefix="bio"))
            row.update(moryossef_segment_metrics(seg_logits, labels, prefix="phrase"))
            # Collapse floor: same segment metric on a CONSTANT all-I prediction. Under the symmetric run-decode any
            # single-span window is near-free, so a monitor within noise of this floor measures the window mix, not
            # the head. Logged every epoch so saturation is visible mid-run.
            alli = torch.zeros_like(seg_logits); alli[..., BIO["I"]] = 1.0
            row["alli_tiou_f1"] = moryossef_segment_metrics(alli, labels, prefix="alli")["alli_tiou_f1"]
            # Per-mode tIoU: a SEGMENTATION metric (predicted vs GT BIO spans), valid for EVERY mode since the head
            # is supervised on all modes' bio_labels — independent of translation supervision, which only some modes
            # carry (OPUT on 1/3, CB on 2a, none on 2b/2c/4). Only the INTERPRETATION differs:
            #   mode1/3 = span boundary quality; mode2 = truncated-fragment localization;
            #   mode4 (gaps) = PHANTOM-AVOIDANCE — gold has 0 spans (all-O; long all-UNK gaps skipped), so tiou_f1
            #                  is 1.0 iff the head stays silent, 0.0 if it fires. Scores ABSENCE, not overlap.
            # The headline average needs this split: a low val_phrase_tiou_f1 driven by mode2 fragments is metric
            # granularity on misaligned windows, not head incompetence.
            modes = batch.get("mode_names") or []
            for mode in set(modes):
                idx = [i for i, m in enumerate(modes) if m == mode]
                sub = moryossef_segment_metrics(seg_logits[idx], labels[idx], prefix=mode)
                row[f"{mode}_tiou_f1"] = sub[f"{mode}_tiou_f1"]
                # Per-mode floor. The POOLED alli above cannot interpret a per-mode number: the floors differ by a factor of ~4 
                # across modes because they are set by gold-span COUNT, not by difficulty (an all-I tagger emits ONE span, so 
                # F1 = 2/(N+1) when that span covers >half the window). The head can sit BELOW a constant on the majority slices 
                # while looking healthy pooled. Every slice must carry its own floor or none of these numbers is interpretable.
                row[f"{mode}_alli_tiou_f1"] = moryossef_segment_metrics(alli[idx], labels[idx], prefix=mode)[f"{mode}_tiou_f1"]
            rows.append(row)
    return mean_logs(rows, prefix="val")


def train_bio_s1_epochs(
    model: BioS1Model, train_loader: DataLoader, dev_loader: DataLoader, 
    device: torch.device, epochs: int, cfg: dict, resume: bool = False,
) -> list[dict[str, float]]:
    dice_weight = float(cfg.get("dice_loss_weight", 1.5))
    class_weights = bio_class_weight_tensor(cfg.get("bio_class_weights"))
    if class_weights is not None: class_weights = class_weights.to(device)
    # Frozen encoder → head only. Unfrozen → head at learning_rate, the pretrained encoder at backbone_lr.
    # Pooled pretraining monitors plain argmax. A monolingual ablation can monitor an already-pinned target decode.
    duration_prior = None
    _dd_cfg = load_yaml(str(cfg.get("inference_config", "configs/inference.yaml")))
    # Semi-Markov decoding is language-agnostic, but it can recover segments from an all-I/weak-B head using the
    # duration prior alone. That makes it a poor checkpoint selector for transferable SEGMENTATION pretraining.
    # Pooled training therefore ranks the learned BIO head with plain argmax. Target-specific decode calibration
    # happens after pretraining. A monolingual ablation can monitor an already-pinned target decode.
    _corpus = None if cfg.get("pretrain_mix") else cfg.get("language")
    _dd = duration_decode_params(_dd_cfg, _corpus) if _corpus else None
    if _dd is not None:
        _recs, _ = resolve_pretrain_records(
            cfg, load_yaml(str(cfg.get("data_config", "configs/data.yaml"))), str(cfg.get("language")), "train",
        )
        duration_prior = fit_duration_prior(_recs, **_dd)
        print(f"bio_s1 | monitor decode: duration (monolingual corpus {_corpus}); prior from {len(_recs)} videos", flush=True)
    else: print(f"bio_s1 | monitor decode: plain argmax "
                f"({'pooled pretraining' if cfg.get('pretrain_mix') else 'untuned corpus'})", flush=True)
    if model.freeze_encoder: optimizer = build_optimizer(cfg, model.bio_head.parameters())
    else: optimizer = build_optimizer(cfg, model.bio_head.parameters(), backbone_params=model.pose_encoder.parameters())

    def step_fn(batch, epoch: int):
        out = model(batch["poses"], batch["frame_mask"], timestamps_s=batch["timestamps_s"])
        loss = bio_nll_dice_loss(out.logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=class_weights)
        return loss, {"bio_loss": float(loss.detach())}

    # The head's RoPE context is set by the windows it trains on, which the sampler clamps to buffer_cap_s. Later stages re-measure 
    # and rewrite that cap, so eval must read the cap from HERE, not from the live config. monitor_decode: the triple best-epoch 
    # selection ran under. On a corpus's FIRST pass duration_decode_s1.<lang> is still unpinned (tune-decode needs this checkpoint), 
    # so selection happens with the decode off and deployed decode differs — recorded here so the gap is visible instead of inferred.
    training_cap_s = float(cfg["training_buffer_cap_s"])
    meta = {"rope_eval_chunk_s": training_cap_s, "buffer_cap_s": training_cap_s, "monitor_decode": _dd,
            "bio_class_weights": cfg.get("bio_class_weights"), "language": cfg.get("language"),
            "pretrain_pool": pool_key(cfg), "pretrain_mix": cfg.get("pretrain_mix"), "pretrain_dev_mix": cfg.get("pretrain_dev_mix")}
    # The end-of-training save in train.py reuses THIS dict. A second, independently-built meta drops pretrain_pool/pretrain_mix 
    # (disarming eval.py's provenance assertion) and re-derives rope_eval_chunk_s from the live inference.yaml — which is the value 
    # the stamp exists to override, since `analyze --stage buffer-cap --write-config` rewrites buffer_cap_s after training.
    cfg["checkpoint_meta"] = meta
    return run_epoch_loop(
        name="bio_s1", model=model, loader=train_loader, optimizer=optimizer, device=device, epochs=epochs, cfg=cfg, step_fn=step_fn, 
        evaluate_fn=lambda e: evaluate_bio_s1(model, dev_loader, device, dice_weight, class_weights, duration_prior=duration_prior), 
        default_monitor="val_mode3_tiou_f1", default_mode="max", dev_loader=dev_loader, resume=resume, checkpoint_meta=meta
    )
