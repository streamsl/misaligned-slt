from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.batch import WindowCollator
from data.loader import StreamingWindowDataset, load_language_records
from models.unisign import UniSignMT5FrontEnd, UniSignMBartFrontEnd, prompt_lang_for_target
from models.streaming_slt import MisalignedSLTModel, SLTLossOutput
from models.checkpointing import save_model_checkpoint

from train.losses import bio_class_weight_tensor
from train.helpers import eval_mode, mean_logs, move_to_device, run_epoch_loop
from metrics import bio_frame_metrics, compute_text_metrics, moryossef_segment_metrics
from utils import load_yaml, language_model_name, pretrained_checkpoint


@dataclass
class SLTComponents:
    model: MisalignedSLTModel
    tokenizer: Any
    train_loader: DataLoader
    dev_loader: DataLoader | None

def _assert_gate_inference_consistency(slt_cfg: dict, inference_cfg: dict) -> None:
    """The membership gate's δ and Λ_min are trained here and DEPLOYED from inference.yaml — they must be the
    SAME geometry, or the gate the decoder learned differs from the one the FSM runs. The two configs have no
    inheritance link, so nothing else enforces it; check at load (only when the gate is on)."""
    gate = slt_cfg.get("membership_gate", {})
    if not gate.get("enabled", False): return
    pairs = [
        ("membership_gate.delta", int(gate.get("delta", 3)),
         "boundary_stability.delta_enc_frames", int(inference_cfg.get("boundary_stability", {}).get("delta_enc_frames", 3))),
        ("membership_gate.min_span_frames", int(gate.get("min_span_frames", 0)),
         "span_selection.min_span_frames", int(inference_cfg.get("span_selection", {}).get("min_span_frames", 0))),
    ]
    for s_key, s_val, i_key, i_val in pairs:
        if s_val != i_val: raise ValueError(
            f"Membership-gate geometry mismatch: slt config {s_key}={s_val} but inference.yaml {i_key}={i_val}. "
            f"The gate trains and deploys under the SAME δ/Λ_min — reconcile the two configs.")

def _optional_int(value) -> int | None:
    return None if value is None else int(value)

def _optional_float(value) -> float | None:
    return None if value is None else float(value)

def build_slt_components(
    data_config: str = "configs/data.yaml",
    slt_config: str = "configs/dlm.yaml",
    inference_config: str = "configs/inference.yaml",
    decoder: str | None = None,
    include_dev: bool = False,
) -> SLTComponents:
    data_cfg = load_yaml(data_config)
    slt_cfg = load_yaml(slt_config)
    inference_cfg = load_yaml(inference_config)
    _assert_gate_inference_consistency(slt_cfg, inference_cfg)
    language = str(slt_cfg.get("language", data_cfg.get("active_languages", ["phoenix"])[0]))

    target_lang = data_cfg["languages"][language].get("target_lang", "en_XX")
    # Uni-Sign front end; the LANGUAGE MODEL (and its tokenizer) is selected by this config's language_model.name:
    # mT5 (Path A default) or mBART (the mT5-vs-mBART ablation). Same pose encoder + prompt either way — only the LM differs.
    lm_name = language_model_name(slt_cfg)
    prompt_lang = prompt_lang_for_target(target_lang)
    if "mbart" in lm_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(lm_name, src_lang=target_lang, tgt_lang=target_lang)
        front_end = UniSignMBartFrontEnd(
            mbart_name=lm_name, prompt_lang=prompt_lang, target_lang=target_lang, tokenizer=tokenizer,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(lm_name, legacy=False)
        front_end = UniSignMT5FrontEnd(mt5_name=lm_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)

    pose_augment_cfg = slt_cfg.get("augmentation")  # train-only spatial aug; dev dataset below passes None
    train_records, _ = load_language_records(data_cfg, language, split="train")
    train_dataset = StreamingWindowDataset(
        train_records, slt_cfg=slt_cfg, 
        inference_cfg=inference_cfg, pose_augment_cfg=pose_augment_cfg
    )
    collator = WindowCollator(
        tokenizer, max_text_tokens=int(slt_cfg.get("max_text_tokens", 128)),
        visual_padding=str(slt_cfg.get("visual_padding", "none")),
    )
    # num_workers>0 parallelizes pose load + window sampling + tokenization (the CPU bottleneck that otherwise
    # stalls the GPU). 0 is safest on Windows; set num_workers in the config for Colab/Linux.
    num_workers = int(slt_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset, batch_size=int(slt_cfg.get("batch_size", 4)), shuffle=False, 
        num_workers=num_workers, persistent_workers=num_workers > 0, collate_fn=collator,
    )
    dev_loader = None
    if include_dev:
        dev_records, _ = load_language_records(data_cfg, language, split="dev")
        # Dev scoring should cover the same experimental unit as standard SLT training: 1 sentence anchor, not 1 video. 
        # With len(dev_records), validation sampled only 1 fixed window per video and could miss most sentences.
        dev_steps = sum(len(record.sentences) for record in dev_records)
        dev_dataset = StreamingWindowDataset(
            dev_records, slt_cfg=slt_cfg, inference_cfg=inference_cfg,
            steps_per_epoch=max(dev_steps, 1), deterministic=True,  # fixed dev windows across epochs
        )
        dev_loader = DataLoader(dev_dataset, batch_size=int(slt_cfg.get("batch_size", 4)), collate_fn=collator)

    # `pretrained_path` is loaded inside MisalignedSLTModel BEFORE the DLM [MASK]-token extension, so the
    # block-diffusion decoder inherits the released Uni-Sign pose + LM weights (pose always; mT5 also loads the LM).
    model = MisalignedSLTModel(
        front_end=front_end, tokenizer=tokenizer,
        decoder=decoder or str(slt_cfg.get("decoder", "dlm")),
        block_size=int(slt_cfg.get("block_size", 8)),
        # BIO-head shape MUST match S1 (train/bio_pretrain.py) exactly, or `bio_head_init` fails to strict-load.
        # Read the SAME keys build_bio_s1 reads (bio_pretrain.yaml `extends` this file, so they share values).
        bio_hidden_dim=int(slt_cfg.get("bio_hidden_dim", 384)),
        bio_depth=int(slt_cfg.get("bio_depth", 4)),
        bio_nhead=int(slt_cfg.get("bio_nhead", 8)),
        bio_dropout=float(slt_cfg.get("bio_dropout", 0.1)),
        bio_conv_stem_layers=int(slt_cfg.get("bio_conv_stem_layers", 2)),
        pretrained_path=pretrained_checkpoint(slt_cfg),
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model initialized with {total_params / 1e6:.2f}M parameters')
    # S1 BIO init (docs/membership_gate.md §1.4 "competence before coupling"): load the pre-trained head from
    # train-bio so S2 trains exactly one new thing — the coupling — and membership_gate.warmup_epochs can be 0.
    bio_init = slt_cfg.get("checkpoint", {}).get("bio_head_init")
    if bio_init and Path(bio_init).exists():
        blob = torch.load(str(bio_init), map_location="cpu")
        sd = blob.get("model", blob) if isinstance(blob, dict) else blob
        head_sd = {k[len("bio_head."):]: v for k, v in sd.items() if k.startswith("bio_head.")}
        model.bio_head.load_state_dict(head_sd, strict=True)
        print(f"slt | loaded S1 BIO head init from {bio_init} ({len(head_sd)} tensors)", flush=True)
    elif bio_init:
        print(f"slt | WARNING: bio_head_init {bio_init} not found — BIO head starts FRESH; keep "
              f"membership_gate.warmup_epochs > 0 (the gate must not couple to an untrained head)", flush=True)
    if bool(slt_cfg.get("freeze_backbone", False)):
        n = model.front_end.freeze_pose_backbone(freeze_projection=bool(slt_cfg.get("freeze_projection", False)))
        print(f"slt | froze pose backbone ({n / 1e6:.2f}M parameters)", flush=True)
    return SLTComponents(model=model, tokenizer=tokenizer, train_loader=train_loader, dev_loader=dev_loader)


@torch.no_grad()
def evaluate_slt(
    model: MisalignedSLTModel, loader: DataLoader, device: torch.device, slt_cfg: dict, gate_active: bool | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, float]] = []

    confidence_cfg = slt_cfg.get("confidence_bound", {})
    dcd_cfg = slt_cfg.get("dcd", {})
    oput_cfg = slt_cfg.get("oput", {})
    spd_cfg = slt_cfg.get("spd", {})
    gate_cfg = slt_cfg.get("membership_gate", {})

    # Dev must be scored under the SAME gate state the current training epoch uses (during gate warmup the
    # decoder has never seen Ω — evaluating it gated would report a conditioning it wasn't trained under).
    gate_on = bool(gate_cfg.get("enabled", False)) if gate_active is None else bool(gate_active)
    gate_kwargs = dict(
        gate_enabled=gate_on, gate_delta=int(gate_cfg.get("delta", 3)), gate_eps=float(gate_cfg.get("eps", 1e-4)),
        gate_min_span_frames=int(gate_cfg.get("min_span_frames", 0)),
    )
    dice_weight = float(slt_cfg.get("dice_loss_weight", 1.5))
    validation_cfg = slt_cfg.get("validation", {})
    # <= 0 means evaluate translation on ALL supervised dev windows (default).
    max_translation_samples = int(validation_cfg.get("max_translation_samples", 0) or 0)

    pred_texts: list[str] = []
    ref_texts: list[str] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        output: SLTLossOutput = model.forward_loss(
            batch, lambda_trans=float(slt_cfg.get("lambda_trans", 1.0)), dice_weight=dice_weight,
            bio_class_weights=bio_class_weight_tensor(slt_cfg.get("bio_class_weights")),
            oput_t_low=float(oput_cfg.get("t_low", 0.3)),
            oput_t_high=float(oput_cfg.get("t_high", 0.8)),
            oput_sample_rollout=bool(oput_cfg.get("sample_rollout", False)),
            confidence_bound_enabled=bool(confidence_cfg.get("enabled", True)),
            confidence_bound_active=True,
            confidence_bound_tau=float(confidence_cfg.get("tau_cb", 0.75)),
            cb_lambda=float(confidence_cfg.get("lambda", 0.3)),
            verified_full_evidence_gate=bool(confidence_cfg.get("verified_full_evidence_gate", True)),
            cb_decode_steps=int(confidence_cfg.get("decode_steps", 16)),
            cb_dcd_window_length=int(dcd_cfg.get("initial_window_length", slt_cfg.get("block_size", 8))),
            cb_dcd_max_window_length=int(dcd_cfg.get("max_window_length", 64)),
            cb_dcd_window_type=str(confidence_cfg.get("window_type", dcd_cfg.get("window_type", "sliding"))),
            cb_dcd_decode_algo=str(dcd_cfg.get("decode_algo", "threshold")),
            cb_dcd_decode_param=dcd_cfg.get("decode_param", confidence_cfg.get("tau_cb", 0.75)),
            cb_dcd_sample_top_k=_optional_int(dcd_cfg.get("top_k")),
            cb_dcd_top_p=_optional_float(dcd_cfg.get("top_p")),
            cb_dcd_cache_type=str(confidence_cfg.get("cache_type", dcd_cfg.get("cache_type", "none"))),
            cb_dcd_refresh_count=int(dcd_cfg.get("refresh_count", 16)),
            cb_spd_top_k=int(spd_cfg.get("top_k", 1)),
            cb_spd_renormalize=bool(spd_cfg.get("renormalize", True)),
            cb_spd_revision=bool(spd_cfg.get("revision", True)),
            cb_temperature=float(dcd_cfg.get("temperature", 0.0)),
            **gate_kwargs,
        )
        row = {k: float(v.detach().cpu().item()) for k, v in output.logs.items() if v.numel() == 1}
        bio_tap, _, timestamps = model.front_end.extract_bio_tap(batch["poses"], batch["frame_mask"], batch.get("timestamps_s"))
        bio_logits = model.bio_head(bio_tap, timestamps_s=timestamps).logits
        row.update(bio_frame_metrics(bio_logits, batch["bio_labels"], prefix="bio"))
        # Moryossef-style segmentation metrics on the in-model BIO head (frame macro-F1, frame-IoU, overlap
        # segment-F1, one-to-one tIoU-matched segment-F1) under the inference decode (runs split at interior Bs),
        # so SLT dev tracks span quality — what RQ2 streaming depends on — not just per-frame BIO accuracy.
        row.update(moryossef_segment_metrics(bio_logits, batch["bio_labels"], prefix="phrase"))
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
                        max_text_tokens=int(slt_cfg.get("max_text_tokens", 128)),
                        diffusion_steps=int(validation_cfg.get("diffusion_steps", slt_cfg.get("diffusion_steps", 64))),
                        tau_dec=float(dcd_cfg.get("tau_dec", confidence_cfg.get("tau_cb", 0.75))),
                        spd_top_k=int(spd_cfg.get("top_k", 1)),
                        spd_renormalize=bool(spd_cfg.get("renormalize", True)),
                        spd_revision=bool(spd_cfg.get("revision", True)),
                        temperature=float(dcd_cfg.get("temperature", 0.0)),
                        dcd_window_length=int(dcd_cfg.get("initial_window_length", slt_cfg.get("block_size", 8))),
                        dcd_max_window_length=int(dcd_cfg.get("max_window_length", 64)),
                        dcd_window_type=str(dcd_cfg.get("window_type", "sliding")),
                        dcd_decode_algo=str(dcd_cfg.get("decode_algo", "threshold")),
                        dcd_decode_param=dcd_cfg.get("decode_param", confidence_cfg.get("tau_cb", 0.75)),
                        dcd_sample_top_k=_optional_int(dcd_cfg.get("top_k")),
                        dcd_top_p=_optional_float(dcd_cfg.get("top_p")),
                        dcd_cache_type=str(dcd_cfg.get("cache_type", "none")),
                        dcd_refresh_count=int(dcd_cfg.get("refresh_count", 16)),
                        **gate_kwargs,
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
    return metrics


def train_slt_epochs(
    model: MisalignedSLTModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epochs: int,
    slt_cfg: dict, dev_loader: DataLoader | None = None,
) -> list[dict[str, float]]:
    confidence_cfg = slt_cfg.get("confidence_bound", {})
    dcd_cfg = slt_cfg.get("dcd", {})
    oput_cfg = slt_cfg.get("oput", {})
    spd_cfg = slt_cfg.get("spd", {})
    gate_cfg = slt_cfg.get("membership_gate", {})
    dice_weight = float(slt_cfg.get("dice_loss_weight", 1.5))
    decoder_name = getattr(model, "decoder_type", "dlm")

    # OPUT warmup holds the confidence-bound term off until the model's full-evidence decode is trustworthy.
    # Membership-gate warmup holds Ω off while a fresh BIO head sharpens on Dice (prefer a real S1 pretrain, so
    # gate.warmup_epochs: 0 when bio_head_init is present). Both are per-epoch flags feeding step AND eval.
    cb_warmup_epochs = int(confidence_cfg.get("warmup_epochs", 1))
    cb_lambda = float(confidence_cfg.get("lambda", 0.3))
    gate_enabled_cfg = bool(gate_cfg.get("enabled", False))
    gate_warmup_epochs = int(gate_cfg.get("warmup_epochs", 0))

    def _gate_active(epoch: int) -> bool:
        return gate_enabled_cfg and epoch > gate_warmup_epochs

    def step_fn(batch, epoch: int):
        output: SLTLossOutput = model.forward_loss(
            batch, lambda_trans=float(slt_cfg.get("lambda_trans", 1.0)), dice_weight=dice_weight,
            bio_class_weights=bio_class_weight_tensor(slt_cfg.get("bio_class_weights")),
            oput_t_low=float(oput_cfg.get("t_low", 0.3)),
            oput_t_high=float(oput_cfg.get("t_high", 0.8)),
            oput_sample_rollout=bool(oput_cfg.get("sample_rollout", False)),
            oput_rollout_eval_mode=bool(oput_cfg.get("rollout_eval_mode", True)),
            oput_eos_supervision=int(oput_cfg.get("eos_supervision_tokens", 32)),
            confidence_bound_enabled=bool(confidence_cfg.get("enabled", True)),
            confidence_bound_active=epoch > cb_warmup_epochs,
            confidence_bound_tau=float(confidence_cfg.get("tau_cb", 0.75)),
            cb_lambda=cb_lambda,
            verified_full_evidence_gate=bool(confidence_cfg.get("verified_full_evidence_gate", True)),
            cb_decode_steps=int(confidence_cfg.get("decode_steps", 16)),
            cb_dcd_window_length=int(dcd_cfg.get("initial_window_length", slt_cfg.get("block_size", 8))),
            cb_dcd_max_window_length=int(dcd_cfg.get("max_window_length", 64)),
            cb_dcd_window_type=str(confidence_cfg.get("window_type", dcd_cfg.get("window_type", "sliding"))),
            cb_dcd_decode_algo=str(dcd_cfg.get("decode_algo", "threshold")),
            cb_dcd_decode_param=dcd_cfg.get("decode_param", confidence_cfg.get("tau_cb", 0.75)),
            cb_dcd_sample_top_k=_optional_int(dcd_cfg.get("top_k")),
            cb_dcd_top_p=_optional_float(dcd_cfg.get("top_p")),
            cb_dcd_cache_type=str(confidence_cfg.get("cache_type", dcd_cfg.get("cache_type", "none"))),
            cb_dcd_refresh_count=int(dcd_cfg.get("refresh_count", 16)),
            cb_spd_top_k=int(spd_cfg.get("top_k", 1)),
            cb_spd_renormalize=bool(spd_cfg.get("renormalize", True)),
            cb_spd_revision=bool(spd_cfg.get("revision", True)),
            cb_temperature=float(dcd_cfg.get("temperature", 0.0)),
            gate_enabled=_gate_active(epoch),
            # δ must match the inference commit gate's delta_enc_frames (configs/inference.yaml) — 
            # the gate's ramp/band/tolerance is the SAME δ at train and inference.
            gate_delta=int(gate_cfg.get("delta", 3)),
            gate_eps=float(gate_cfg.get("eps", 1e-4)),
            gate_min_span_frames=int(gate_cfg.get("min_span_frames", 0)),
            gate_iou_veto=float(gate_cfg.get("iou_veto", 0.5)),
            gate_gt_anchored=bool(gate_cfg.get("gt_anchored", False)),
        )
        return output.loss, {k: float(v.detach().cpu().item()) for k, v in output.logs.items() if v.numel() == 1}

    def evaluate_fn(epoch: int):
        # Dev must be scored under the SAME gate state the epoch trained under (see evaluate_slt).
        return evaluate_slt(model, dev_loader, device, slt_cfg=slt_cfg, gate_active=_gate_active(epoch))

    return run_epoch_loop(
        name=f"slt-{decoder_name}", model=model, loader=loader, optimizer=optimizer, device=device,
        epochs=epochs, cfg=slt_cfg, step_fn=step_fn, evaluate_fn=evaluate_fn,
        default_monitor="val_loss", default_mode="min", dev_loader=dev_loader,
    )
