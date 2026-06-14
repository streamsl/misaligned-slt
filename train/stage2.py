from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.batch import WindowCollator
from data.loader import StreamingWindowDataset, load_language_records
from models.gfslt import GFSLTConfig
from models.streaming_slt import MisalignedSLTModel, Stage2LossOutput
from models.checkpointing import load_visual_backbone, save_model_checkpoint

from train.helpers import AmpHelper, TrainControl, TrainLogger, attach_save_best, build_scheduler, mean_logs
from train.losses import bio_class_weight_tensor
from metrics import bio_frame_metrics, compute_text_metrics
from utils import load_yaml, mbart_trimmed_dir, vlp_checkpoint


@dataclass
class Stage2Components:
    model: MisalignedSLTModel
    tokenizer: Any
    train_loader: DataLoader
    dev_loader: DataLoader | None

def _move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor): return value.to(device)
    if isinstance(value, dict): return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list): return [_move_to_device(v, device) for v in value]
    return value

def _optional_int(value) -> int | None:
    return None if value is None else int(value)

def _optional_float(value) -> float | None:
    return None if value is None else float(value)

def build_gfslt_config(stage1_cfg: dict, stage2_cfg: dict) -> GFSLTConfig:
    return GFSLTConfig(
        embed_dim=int(stage1_cfg.get("embed_dim", 1024)),
        hidden_size=int(stage1_cfg.get("hidden_size", 1024)),
        temporal_kernel=int(stage1_cfg.get("temporal_kernel", 3)),
        mbart_name=mbart_trimmed_dir(stage1_cfg),  # same trimmed mBART the VLP stage trained
        use_temporal_conv=bool(stage2_cfg.get("use_temporal_conv", stage1_cfg.get("use_temporal_conv", False))),
    )

def build_stage2_components(
    data_config: str = "configs/data.yaml",
    stage1_config: str = "configs/stage1_vlp.yaml",
    stage2_config: str = "configs/stage2_dlm.yaml",
    inference_config: str = "configs/inference.yaml",
    decoder: str | None = None,
    include_dev: bool = False,
) -> Stage2Components:
    data_cfg = load_yaml(data_config)
    stage1_cfg = load_yaml(stage1_config)
    stage2_cfg = load_yaml(stage2_config)
    inference_cfg = load_yaml(inference_config)
    language = str(stage2_cfg.get("language", data_cfg.get("active_languages", ["phoenix"])[0]))

    tokenizer_dir = mbart_trimmed_dir(stage1_cfg)
    target_lang = data_cfg["languages"][language].get("target_lang", "en_XX")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, src_lang=target_lang, tgt_lang=target_lang)

    train_records, _ = load_language_records(data_cfg, language, split="train")
    train_dataset = StreamingWindowDataset(train_records, stage2_cfg=stage2_cfg, inference_cfg=inference_cfg)
    collator = WindowCollator(
        tokenizer, max_text_tokens=int(stage2_cfg.get("max_text_tokens", 128)),
        visual_padding=str(stage2_cfg.get("visual_padding", stage1_cfg.get("visual_padding", "gfslt"))),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(stage2_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    dev_loader = None
    if include_dev:
        dev_records, _ = load_language_records(data_cfg, language, split="dev")
        dev_dataset = StreamingWindowDataset(
            dev_records, stage2_cfg=stage2_cfg, inference_cfg=inference_cfg,
            steps_per_epoch=len(dev_records), deterministic=True,  # fixed dev windows across epochs
        )
        dev_loader = DataLoader(dev_dataset, batch_size=int(stage2_cfg.get("batch_size", 4)), collate_fn=collator)

    model = MisalignedSLTModel(
        gfslt_config=build_gfslt_config(stage1_cfg, stage2_cfg), tokenizer=tokenizer,
        decoder=decoder or str(stage2_cfg.get("decoder", "dlm")),
        bio_hidden_dim=int(stage2_cfg.get("bio_hidden_dim", 384)),
        block_size=int(stage2_cfg.get("block_size", 8)),
        bio_conv_stem_layers=int(stage2_cfg.get("bio_conv_stem_layers", 2)),
    )
    checkpoint = vlp_checkpoint(stage2_cfg)
    if checkpoint:
        try: load_visual_backbone(model.visual, checkpoint, strict=False)
        except (FileNotFoundError, OSError): pass
    return Stage2Components(model=model, tokenizer=tokenizer, train_loader=train_loader, dev_loader=dev_loader)


@torch.no_grad()
def evaluate_stage2(
    model: MisalignedSLTModel, loader: DataLoader, device: torch.device,
    stage2_cfg: dict, segmenter_cfg: dict | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, float]] = []
    
    confidence_cfg = stage2_cfg.get("confidence_bound", {})
    dcd_cfg = stage2_cfg.get("dcd", {})
    oput_cfg = stage2_cfg.get("oput", {})
    spd_cfg = stage2_cfg.get("spd", {})

    dice_weight = float((segmenter_cfg or {}).get("dice_loss_weight", 1.5))
    validation_cfg = stage2_cfg.get("validation", {})
    # <= 0 means evaluate translation on ALL supervised dev windows (default).
    max_translation_samples = int(validation_cfg.get("max_translation_samples", 0) or 0)

    pred_texts: list[str] = []
    ref_texts: list[str] = []
    for batch in loader:
        batch = _move_to_device(batch, device)
        output: Stage2LossOutput = model.forward_loss(
            batch, lambda_trans=float(stage2_cfg.get("lambda_trans", 1.0)), dice_weight=dice_weight,
            bio_class_weights=bio_class_weight_tensor(stage2_cfg.get("bio_class_weights")),
            oput_t_low=float(oput_cfg.get("t_low", 0.3)),
            oput_t_high=float(oput_cfg.get("t_high", 0.8)),
            oput_sample_rollout=bool(oput_cfg.get("sample_rollout", False)),
            confidence_bound_enabled=bool(confidence_cfg.get("enabled", True)),
            confidence_bound_active=True,
            confidence_bound_tau=float(confidence_cfg.get("tau_cb", 0.75)),
            cb_lambda=float(confidence_cfg.get("lambda", 0.3)),
            verified_full_evidence_gate=bool(confidence_cfg.get("verified_full_evidence_gate", True)),
            cb_decode_steps=int(stage2_cfg.get("diffusion_steps", 64)),
            cb_dcd_window_length=int(dcd_cfg.get("initial_window_length", stage2_cfg.get("block_size", 8))),
            cb_dcd_max_window_length=int(dcd_cfg.get("max_window_length", 64)),
            cb_dcd_window_type=str(dcd_cfg.get("window_type", "sliding")),
            cb_dcd_decode_algo=str(dcd_cfg.get("decode_algo", "threshold")),
            cb_dcd_decode_param=dcd_cfg.get("decode_param", confidence_cfg.get("tau_cb", 0.75)),
            cb_dcd_sample_top_k=_optional_int(dcd_cfg.get("top_k")),
            cb_dcd_top_p=_optional_float(dcd_cfg.get("top_p")),
            cb_dcd_cache_type=str(dcd_cfg.get("cache_type", "none")),
            cb_dcd_refresh_count=int(dcd_cfg.get("refresh_count", 16)),
            cb_spd_top_k=int(spd_cfg.get("top_k", 1)),
            cb_spd_renormalize=bool(spd_cfg.get("renormalize", True)),
            cb_spd_revision=bool(spd_cfg.get("revision", True)),
            cb_temperature=float(dcd_cfg.get("temperature", 0.0)),
        )
        row = {k: float(v.detach().cpu().item()) for k, v in output.logs.items() if v.numel() == 1}
        row["loss"] = float(output.loss.detach().cpu().item())
        post_vlp, mask, timestamps = model.visual.extract_post_vlp(batch["poses"], batch["frame_mask"], batch.get("timestamps_s"))
        bio_logits = model.bio_head(post_vlp, timestamps_s=timestamps).logits
        row.update(bio_frame_metrics(bio_logits, batch["bio_labels"], prefix="bio"))
        rows.append(row)

        cap_reached = max_translation_samples > 0 and len(pred_texts) >= max_translation_samples
        if not cap_reached:
            supervised = batch.get("translation_supervised")
            targets = batch.get("translation_targets", [])

            if isinstance(supervised, torch.Tensor) and supervised.any():
                idx = supervised.nonzero(as_tuple=False).flatten()
                if max_translation_samples > 0: idx = idx[: max_translation_samples - len(pred_texts)]
                if idx.numel() > 0:
                    _, tokens, _ = model.generate_from_poses(
                        poses=batch["poses"][idx], frame_mask=batch["frame_mask"][idx],
                        timestamps_s=batch.get("timestamps_s", None)[idx] if batch.get("timestamps_s") is not None else None,
                        max_text_tokens=int(stage2_cfg.get("max_text_tokens", 128)),
                        diffusion_steps=int(validation_cfg.get("diffusion_steps", stage2_cfg.get("diffusion_steps", 64))),
                        tau_dec=float(dcd_cfg.get("tau_dec", confidence_cfg.get("tau_cb", 0.75))),
                        spd_top_k=int(spd_cfg.get("top_k", 1)),
                        spd_renormalize=bool(spd_cfg.get("renormalize", True)),
                        spd_revision=bool(spd_cfg.get("revision", True)),
                        temperature=float(dcd_cfg.get("temperature", 0.0)),
                        dcd_window_length=int(dcd_cfg.get("initial_window_length", stage2_cfg.get("block_size", 8))),
                        dcd_max_window_length=int(dcd_cfg.get("max_window_length", 64)),
                        dcd_window_type=str(dcd_cfg.get("window_type", "sliding")),
                        dcd_decode_algo=str(dcd_cfg.get("decode_algo", "threshold")),
                        dcd_decode_param=dcd_cfg.get("decode_param", confidence_cfg.get("tau_cb", 0.75)),
                        dcd_sample_top_k=_optional_int(dcd_cfg.get("top_k")),
                        dcd_top_p=_optional_float(dcd_cfg.get("top_p")),
                        dcd_cache_type=str(dcd_cfg.get("cache_type", "none")),
                        dcd_refresh_count=int(dcd_cfg.get("refresh_count", 16)),
                    )
                    pred_texts.extend(model.tokenizer.batch_decode(tokens.detach().cpu(), skip_special_tokens=True))
                    for item_idx in idx.detach().cpu().tolist():
                        target = targets[int(item_idx)]
                        if isinstance(target, dict): ref_texts.append(str(target.get("text", "")))
                        else: ref_texts.append(str(getattr(target, "text", "")))

    if was_training: model.train()
    metrics = mean_logs(rows, prefix="val")
    if pred_texts:
        metrics.update(compute_text_metrics(pred_texts, ref_texts, prefix="val_translation"))
        metrics["val_translation_samples"] = float(len(pred_texts))
    return metrics


def train_stage2_epochs(
    model: MisalignedSLTModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epochs: int,
    stage2_cfg: dict, segmenter_cfg: dict | None = None, dev_loader: DataLoader | None = None,
) -> list[dict[str, float]]:
    model.to(device)
    model.train()
    logs: list[dict[str, float]] = []

    confidence_cfg = stage2_cfg.get("confidence_bound", {})
    dcd_cfg = stage2_cfg.get("dcd", {})
    oput_cfg = stage2_cfg.get("oput", {})
    spd_cfg = stage2_cfg.get("spd", {})

    dice_weight = float((segmenter_cfg or {}).get("dice_loss_weight", 1.5))
    scheduler = build_scheduler(optimizer, stage2_cfg, epochs=epochs, steps_per_epoch=len(loader))
    amp = AmpHelper.from_config(stage2_cfg, device)
    decoder_name = getattr(model, "decoder_type", "dlm")
    control = TrainControl.from_config(stage2_cfg, default_monitor="val_loss", default_mode="min")
    attach_save_best(control, stage2_cfg, f"stage2-{decoder_name}", save_model_checkpoint)

    # OPUT warmup: hold the confidence-bound term off until the model's own full-evidence decode is trustworthy. The prior de-risk
    # found that w/o warmup the term collapses early training (the f==r gate fixes collapse at convergence; warmup fixes early instability).
    cb_warmup_epochs = int(confidence_cfg.get("warmup_epochs", 1))
    cb_lambda = float(confidence_cfg.get("lambda", 0.3))
    logger = TrainLogger(f"stage2-{decoder_name}", stage2_cfg, epochs=int(epochs), steps_per_epoch=len(loader), monitor=control.monitor)
    
    for epoch in range(1, int(epochs) + 1):
        cb_active = epoch > cb_warmup_epochs
        epoch_logs: list[dict[str, float]] = []
        for step, batch in enumerate(loader, start=1):
            batch = _move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(): output: Stage2LossOutput = model.forward_loss(
                batch, lambda_trans=float(stage2_cfg.get("lambda_trans", 1.0)), dice_weight=dice_weight,
                bio_class_weights=bio_class_weight_tensor(stage2_cfg.get("bio_class_weights")),
                oput_t_low=float(oput_cfg.get("t_low", 0.3)),
                oput_t_high=float(oput_cfg.get("t_high", 0.8)),
                oput_sample_rollout=bool(oput_cfg.get("sample_rollout", False)),
                oput_rollout_eval_mode=bool(oput_cfg.get("rollout_eval_mode", True)),
                oput_eos_supervision=int(oput_cfg.get("eos_supervision_tokens", 32)),
                confidence_bound_enabled=bool(confidence_cfg.get("enabled", True)),
                confidence_bound_active=cb_active,
                confidence_bound_tau=float(confidence_cfg.get("tau_cb", 0.75)),
                cb_lambda=cb_lambda,
                verified_full_evidence_gate=bool(confidence_cfg.get("verified_full_evidence_gate", True)),
                cb_decode_steps=int(stage2_cfg.get("diffusion_steps", 64)),
                cb_dcd_window_length=int(dcd_cfg.get("initial_window_length", stage2_cfg.get("block_size", 8))),
                cb_dcd_max_window_length=int(dcd_cfg.get("max_window_length", 64)),
                cb_dcd_window_type=str(dcd_cfg.get("window_type", "sliding")),
                cb_dcd_decode_algo=str(dcd_cfg.get("decode_algo", "threshold")),
                cb_dcd_decode_param=dcd_cfg.get("decode_param", confidence_cfg.get("tau_cb", 0.75)),
                cb_dcd_sample_top_k=_optional_int(dcd_cfg.get("top_k")),
                cb_dcd_top_p=_optional_float(dcd_cfg.get("top_p")),
                cb_dcd_cache_type=str(dcd_cfg.get("cache_type", "none")),
                cb_dcd_refresh_count=int(dcd_cfg.get("refresh_count", 16)),
                cb_spd_top_k=int(spd_cfg.get("top_k", 1)),
                cb_spd_renormalize=bool(spd_cfg.get("renormalize", True)),
                cb_spd_revision=bool(spd_cfg.get("revision", True)),
                cb_temperature=float(dcd_cfg.get("temperature", 0.0)),
            )
            amp.backward(output.loss)
            amp.clip_and_step(optimizer, model.parameters(), float(stage2_cfg.get("max_grad_norm", 1.0)))
            scheduler.step_batch()
            row = {k: float(v.detach().cpu().item()) for k, v in output.logs.items() if v.numel() == 1}
            row.update({"epoch": float(epoch), "step": float(step), "lr": scheduler.lr(optimizer), "cb_active": float(cb_active)})
            epoch_logs.append(row)
            logs.append(row)
            logger.log_step(epoch, step, row)

        scheduler.step_epoch()
        train_means = mean_logs(epoch_logs)
        if dev_loader is not None and control.should_eval(epoch, epochs):
            metrics = evaluate_stage2(model, dev_loader, device, stage2_cfg=stage2_cfg, segmenter_cfg=segmenter_cfg)
            improved = control.update(model, metrics, epoch)
            logger.epoch_summary(epoch, train=train_means, val=metrics, is_best=improved, saved_path=control.last_saved_path)
            logs.append({"epoch": float(epoch), **train_means, **metrics, **control.summary()})
            if control.stopped_early:
                print(f"stage2-{decoder_name} | early stop at epoch {epoch} (best {control.monitor}={control.best_value})", flush=True)
                break
        else: logger.epoch_summary(epoch, train=train_means)

    control.restore(model)
    logger.finish()
    return logs
