from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Tokenizer

from data.batch import WindowCollator
from data.loader import StreamingWindowDataset, load_language_records, streaming_loader
from models.unisign import UniSignMT5FrontEnd, UniSignMBartFrontEnd, prompt_lang_for_target
from models.streaming_slt import MisalignedSLTModel, SLTLossOutput

from train import distributed as dist
from train.losses import bio_class_weight_tensor, resolve_bio_class_weights
from train.helpers import mean_logs, move_to_device, run_epoch_loop
from infer.duration_decode import duration_decode_params, fit_duration_prior
from metrics import bio_frame_metrics, compute_text_metrics, moryossef_segment_metrics
from utils import load_yaml, language_model_name, resolve_pretrained


@dataclass
class SLTComponents:
    model: MisalignedSLTModel
    tokenizer: Any
    train_loader: DataLoader
    dev_loader: DataLoader | None
    slt_cfg: dict
    checkpoint_meta: dict

def _assert_gate_inference_consistency(slt_cfg: dict, inference_cfg: dict) -> None:
    """Gate δ/Λ_min train here but deploy from inference.yaml, and nothing links 2 configs — 
    mismatched geometry silently gives the decoder a different gate than the FSM runs."""
    gate = slt_cfg.get("membership_gate", {})
    if not gate.get("enabled", False): return
    if "delta" not in gate or "min_span_frames" not in gate: 
        raise ValueError(f"membership_gate is enabled but missing delta/min_span_frames: {sorted(gate)}")
    delta_i = int(inference_cfg.get("boundary_stability", {}).get("delta_enc_frames", 3))
    lam_i = inference_cfg.get("span_selection", {}).get("min_span_frames")
    lam_i = delta_i + 1 if lam_i is None else int(lam_i)  # infer/stream.py's derivation
    pairs = [
        ("membership_gate.delta", int(gate["delta"]), "boundary_stability.delta_enc_frames", delta_i),
        ("membership_gate.min_span_frames", int(gate["min_span_frames"]), "span_selection.min_span_frames", lam_i),
    ]
    for s_key, s_val, i_key, i_val in pairs:
        if s_val != i_val: raise ValueError(
            f"Membership-gate geometry mismatch: slt config {s_key}={s_val} but inference.yaml {i_key}={i_val}. "
            f"The gate trains and deploys under the SAME δ/Λ_min — reconcile 2 configs.")

def _training_meta(slt_cfg: dict, inference_cfg: dict, language: str) -> dict:
    """The config this stage-2 run is parameterized by, travelling with the weights.

    δ/Λ_min are re-measured by `analyze --stage delta-enc`, buffer_cap_s by tail-benefit, the decode triple by tune-decode, and jitter 
    by segmenter-error analysis. Resuming across such a change trains 2 halves under different objectives, and without this record 
    nothing in the artifacts shows it (models/checkpointing.save_model_checkpoint makes the same argument for S1's chunk size).
    """
    gate_cfg = slt_cfg.get("membership_gate", {}) or {}
    return {
        "language": str(language), "decoder": str(slt_cfg.get("decoder", "dlm")),
        "gate": {k: gate_cfg.get(k) for k in ("enabled", "delta", "min_span_frames", "eps", "gamma")},
        "buffer_cap_s": inference_cfg.get("buffer_cap_s"), "duration_decode": duration_decode_params(inference_cfg, language),
        "mode_ratios": slt_cfg.get("mode_ratios"), "jitter": slt_cfg.get("jitter"),
        "bio_class_weights": slt_cfg.get("bio_class_weights"),  # resolved list, not the "balanced" string
        "lambda_bio": float(slt_cfg.get("lambda_bio", 1.0)), "lambda_trans": float(slt_cfg.get("lambda_trans", 1.0)),
        "batch_size": slt_cfg.get("batch_size"), "learning_rate": slt_cfg.get("learning_rate"),
    }

def _optional_int(value) -> int | None:
    return None if value is None else int(value)

def _optional_float(value) -> float | None:
    return None if value is None else float(value)

def build_slt_components(
    data_config: str = "configs/data.yaml", slt_config: str = "configs/dlm.yaml", inference_config: str = "configs/inference.yaml",
    decoder: str | None = None, include_dev: bool = False, language: str | None = None,
) -> SLTComponents:
    data_cfg = load_yaml(data_config)
    slt_cfg = load_yaml(slt_config)
    # Precedence: --language > config `language:` > active_languages. Reload on override so ${language} 
    # in checkpoint.dir (+ ar/baseline children) re-points at the right dataset.
    language = str(language or slt_cfg.get("language") or data_cfg.get("active_languages", ["asf"])[0])
    if language != slt_cfg.get("language"): slt_cfg = load_yaml(slt_config, language=language)
    inference_cfg = load_yaml(inference_config)
    _assert_gate_inference_consistency(slt_cfg, inference_cfg)

    target_lang = data_cfg["languages"][language].get("target_lang", "en_XX")
    # Uni-Sign front end. language_model.name picks the LM + tokenizer: mT5 (Path A default) or mBART
    # (mT5-vs-mBART ablation); same pose encoder + prompt either way.
    lm_name = language_model_name(slt_cfg)
    prompt_lang = prompt_lang_for_target(target_lang)
    if "mbart" in lm_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(lm_name, src_lang=target_lang, tgt_lang=target_lang)
        front_end = UniSignMBartFrontEnd(
            mbart_name=lm_name, prompt_lang=prompt_lang, target_lang=target_lang, tokenizer=tokenizer,
        )
    else:
        tokenizer = T5Tokenizer.from_pretrained(lm_name, legacy=False)
        front_end = UniSignMT5FrontEnd(mt5_name=lm_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)

    pose_augment_cfg = slt_cfg.get("augmentation")  # train-only spatial aug; dev passes None
    train_records, _ = load_language_records(data_cfg, language, split="train")
    resolve_bio_class_weights(slt_cfg, train_records)
    train_dataset = StreamingWindowDataset(
        train_records, slt_cfg=slt_cfg, 
        inference_cfg=inference_cfg, pose_augment_cfg=pose_augment_cfg
    )
    collator = WindowCollator(
        tokenizer, max_text_tokens=int(slt_cfg.get("max_text_tokens", 128)), visual_padding=str(slt_cfg.get("visual_padding", "none")),
        # `pad_text_to_max_length: false` sizes the text canvas to the batch instead of max_text_tokens. Captions
        # are ~15 tokens against a 128 canvas and every decoder forward runs the whole width, so this is the
        # largest single throughput lever; the collator keeps the EOS-supervision tail and block alignment intact.
        pad_to_max_length=bool(slt_cfg.get("pad_text_to_max_length", True)), block_size=int(slt_cfg.get("block_size", 8)),
        # Default must MATCH the loss path's (block_size, see forward_loss kwargs below): the collator reserves the canvas tail 
        # the EOS supervision writes into — a 0 default here with block_size there starves that tail under dynamic padding.
        eos_supervision_tokens=int((slt_cfg.get("oput", {}) or {}).get("eos_supervision_tokens", slt_cfg.get("block_size", 8))),
    )
    # num_workers is pure throughput: anchors are index-driven (each realized once per epoch regardless of worker
    # split) and workers reseed their rng (data.loader.streaming_loader / WindowSampler.configure_worker).
    num_workers = int(slt_cfg.get("num_workers", 0))
    train_loader = streaming_loader(
        train_dataset, dist.per_rank_batch_size(int(slt_cfg.get("batch_size", 4))), collator, num_workers=num_workers
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
        dev_loader = streaming_loader(
            dev_dataset, dist.per_rank_batch_size(int(slt_cfg.get("batch_size", 4))), collator, num_workers=num_workers
        )
    # `pretrained_path` is loaded inside MisalignedSLTModel BEFORE the DLM [MASK]-token extension, so the
    # block-diffusion decoder inherits the released Uni-Sign pose + LM weights (pose always; mT5 also loads the LM).
    model = MisalignedSLTModel(
        front_end=front_end, tokenizer=tokenizer,
        decoder=decoder or str(slt_cfg.get("decoder", "dlm")),
        block_size=int(slt_cfg.get("block_size", 8)),
        # Shape MUST match S1 (train/bio_pretrain.py) or `bio_head_init` fails to strict-load — 
        # same keys build_bio_s1 reads (bio_pretrain.yaml `extends` this file).
        bio_hidden_dim=int(slt_cfg.get("bio_hidden_dim", 384)),
        bio_depth=int(slt_cfg.get("bio_depth", 4)),
        bio_nhead=int(slt_cfg.get("bio_nhead", 8)),
        bio_dropout=float(slt_cfg.get("bio_dropout", 0.1)),
        bio_conv_stem_layers=int(slt_cfg.get("bio_conv_stem_layers", 2)),
        pretrained_path=resolve_pretrained(slt_cfg, data_cfg, language, default="checkpoints/openasl_pose_only_slt.pth"),
    )
    # S1 BIO init (docs/membership_gate.md §1.4 "competence before coupling"): load the pre-trained head from
    # train-bio so S2 trains exactly one new thing — the coupling — and membership_gate.warmup_epochs can be 0.
    bio_init = slt_cfg.get("checkpoint", {}).get("bio_head_init")
    if float(slt_cfg.get("lambda_bio", 1.0)) == 0.0:
        if bool(slt_cfg.get("membership_gate", {}).get("enabled", False)): raise SystemExit(
            "lambda_bio: 0 with membership_gate.enabled: true is incoherent — the gate reads the BIO head's posteriors, but "
            "lambda_bio: 0 skips the head's forward and leaves it untrained/frozen. Either train the head (lambda_bio > 0) "
            "or disable the gate (the clean-floor recipe does both)."
        )
        # Clean-floor recipe (lambda_bio: 0 — baseline_train.yaml): no BIO branch. Skip the S1 init entirely, head
        # AND its pose encoder (the baseline must start from the RELEASED weights only), and freeze the head so
        # the optimizer never sees it. forward_loss skips its forward, so the branch costs nothing.
        for p in model.bio_head.parameters(): p.requires_grad_(False)
        # generate_from_poses skips the head's forward too: with the gate off nobody reads frozen-random logits.
        model.bio_branch_off = True
        print("slt | lambda_bio=0: BIO branch OFF — head frozen at random init, forward SKIPPED in training and decode; "
              + ("bio_head_init IGNORED (clean-floor recipe trains the released front end only)" \
                if bio_init else "no bio_head_init configured"), flush=True)

    elif bio_init and Path(bio_init).exists():
        blob = torch.load(str(bio_init), map_location="cpu")
        sd = blob.get("model", blob) if isinstance(blob, dict) else blob
        head_sd = {k[len("bio_head."):]: v for k, v in sd.items() if k.startswith("bio_head.")}
        # S1's head reads feat_dim; this bio_head reads front_end.bio_tap_dim (LM d_model: 768 mT5 / 1024 mBART). Must match for 
        # "S1 features == S2 initial features" and the strict-load. Released Uni-Sign checkpoints (which seed train-bio) are mT5-768, 
        # so mBART arm needs S1 at bio_pretrain feat_dim=1024 + a 1024 pose encoder; no 1024 release exists to warm-start from.
        s1_dim = head_sd.get("input_proj.weight")
        if s1_dim is not None and int(s1_dim.shape[1]) != int(front_end.bio_tap_dim): raise ValueError(
            f"bio_head_init dim mismatch: S1 head reads {int(s1_dim.shape[1])}-d features but this SLT model's "
            f"bio_tap is {int(front_end.bio_tap_dim)}-d ({lm_name}). Retrain train-bio with feat_dim="
            f"{int(front_end.bio_tap_dim)} (bio_pretrain.yaml), or use the mT5 arm (the released checkpoints are mT5-768)."
        )
        model.bio_head.load_state_dict(head_sd, strict=True)
        # Carry S1's pose encoder so the head meets the features it trained on. No-op under S1 freeze_backbone:true 
        # (== the released weights above); under false the head trained on the ADAPTED encoder and would meet features 
        # it never saw (silent warm-start corruption).
        pose_sd = {k[len("pose_encoder."):]: v for k, v in sd.items() if k.startswith("pose_encoder.")}
        if pose_sd: model.front_end.pose_encoder.load_state_dict(pose_sd, strict=True)
        enc_note = (f"S1 pose encoder OVERRIDES the released warm-start ({len(pose_sd)} tensors) — deliberate: the head must "
                    "meet the features it trained on (S1 features == S2 initial features); identical to the released weights "
                    "when S1 ran freeze_backbone: true") if pose_sd else "S1 checkpoint carries no pose encoder"
        print(f"slt | loaded S1 BIO head init from {bio_init} ({len(head_sd)} tensors); {enc_note}", flush=True)
    elif bio_init:
        # Fail loud, mirroring the mode_ratios.source guard: bio_head_init is cwd-relative, so a wrong-cwd launch
        # (Colab default dir) would otherwise silently train the gate against a random-init head.
        if bool(slt_cfg.get("membership_gate", {}).get("enabled", False)): raise FileNotFoundError(
            f"bio_head_init {bio_init} not found while membership_gate.enabled: true — the gate must not couple to an untrained head. "
            f"Fix path/cwd, or set bio_head_init: null AND membership_gate.warmup_epochs >= 2 to train from a fresh head deliberately."
        )
        print(f"slt | WARNING: bio_head_init {bio_init} not found — BIO head starts FRESH", flush=True)
    if bool(slt_cfg.get("freeze_backbone", False)):
        n = model.front_end.freeze_pose_backbone(freeze_projection=bool(slt_cfg.get("freeze_projection", False)))
        print(f"slt | froze pose backbone ({n / 1e6:.2f}M parameters)", flush=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"slt | model: {total_params / 1e6:.2f}M parameters ({trainable_params / 1e6:.2f}M trainable, "
          f"{(total_params - trainable_params) / 1e6:.2f}M frozen)", flush=True)
          
    # Gate-side duration decode (inference.yaml duration_decode — the switch the streaming FSM reads):
    # build_gate_omega re-splits merged back-to-back runs, so stage-2 trains on the tag stream deployment gates
    # on (§1.3 on-policy symmetry). Prior fit on TRAIN captions only; no-op when the gate is off.
    if bool(slt_cfg.get("membership_gate", {}).get("enabled", False)):
        params = duration_decode_params(inference_cfg, language)
        if params is not None:
            model.duration_prior = fit_duration_prior(train_records, **params)
            if model.duration_prior is not None:
                p = model.duration_prior
                print(f"slt | gate duration decode ON: lognormal({p.mu_log_s:.2f},{p.sd_log_s:.2f}) cap={p.cap_s:.0f}s "
                      f"split_bias={p.split_bias:g} snap_radius_s={p.snap_radius_s:g}", flush=True)
    return SLTComponents(
        model=model, tokenizer=tokenizer, train_loader=train_loader, dev_loader=dev_loader, slt_cfg=slt_cfg,
        checkpoint_meta=_training_meta(slt_cfg, inference_cfg, language),
    )


@torch.no_grad()
def evaluate_slt(
    model: MisalignedSLTModel, loader: DataLoader, device: torch.device, slt_cfg: dict,
    gate_active: bool | None = None, cb_active: bool | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, float]] = []

    confidence_cfg = slt_cfg.get("confidence_bound", {})
    dcd_cfg = slt_cfg.get("dcd", {})
    oput_cfg = slt_cfg.get("oput", {})
    spd_cfg = slt_cfg.get("spd", {})
    gate_cfg = slt_cfg.get("membership_gate", {})

    # Score dev under the SAME gate/CB state the epoch trained under: during warmup the decoder has never seen Ω
    # or the CB self-target, so either one on reports an untrained objective and burns a CB decode on
    # untrustworthy targets. cb_active=None (standalone eval) = full objective.
    gate_on = bool(gate_cfg.get("enabled", False)) if gate_active is None else bool(gate_active)
    cb_on = True if cb_active is None else bool(cb_active)
    # Every gate knob training passes — otherwise dev loss is computed under forward_loss's DEFAULTS (iou_veto 0.5,
    # gt_anchored False) while training used the config, so val_loss is not comparable to train_loss and the
    # gt_anchored ablation's dev numbers are silently ungated.
    # Split by what the knob NEEDS, not by convenience. `iou_veto` and `gt_anchored` both read bio_labels, so they
    # exist only where GT exists — the loss. Inference selects the anchor on-policy from the head's own argmax
    # (bio_labels=None, no veto), so passing them to a generate path is meaningless and lands them in
    # **decode_kwargs -> generate_from_bio_tap -> TypeError.
    gate_kwargs = dict(
        gate_enabled=gate_on, gate_delta=int(gate_cfg.get("delta", 3)), gate_eps=float(gate_cfg.get("eps", 1e-4)),
        gate_min_span_frames=int(gate_cfg.get("min_span_frames", 0)),
    )
    gate_loss_kwargs = dict(
        gate_kwargs,
        gate_iou_veto=float(gate_cfg.get("iou_veto", 0.5)),
        gate_gt_anchored=bool(gate_cfg.get("gt_anchored", False)),
    )
    dice_weight = float(slt_cfg.get("dice_loss_weight", 1.5))
    validation_cfg = slt_cfg.get("validation", {})
    # <= 0: translate ALL supervised dev windows (default).
    max_translation_samples = int(validation_cfg.get("max_translation_samples", 0) or 0)

    pred_texts: list[str] = []
    ref_texts: list[str] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        output: SLTLossOutput = model.forward_loss(
            batch, lambda_trans=float(slt_cfg.get("lambda_trans", 1.0)), lambda_bio=float(slt_cfg.get("lambda_bio", 1.0)), 
            dice_weight=dice_weight, bio_class_weights=bio_class_weight_tensor(slt_cfg.get("bio_class_weights")),
            oput_t_low=float(oput_cfg.get("t_low", 0.3)), oput_t_high=float(oput_cfg.get("t_high", 0.8)),
            oput_sample_rollout=bool(oput_cfg.get("sample_rollout", False)),
            oput_label_smoothing=float(oput_cfg.get("label_smoothing", 0.0)),
            oput_rollout_eval_mode=bool(oput_cfg.get("rollout_eval_mode", True)),
            oput_eos_supervision=int(oput_cfg.get("eos_supervision_tokens", slt_cfg.get("block_size", 8))),
            confidence_bound_enabled=bool(confidence_cfg.get("enabled", True)), confidence_bound_active=cb_on,
            confidence_bound_tau=float(confidence_cfg.get("tau_cb", 0.75)), cb_lambda=float(confidence_cfg.get("lambda", 1.0)),
            verified_full_evidence_gate=bool(confidence_cfg.get("verified_full_evidence_gate", True)),
            cb_decode_steps=int(confidence_cfg.get("decode_steps", 16)),
            cb_dcd_window_length=int(dcd_cfg.get("initial_window_length", slt_cfg.get("block_size", 8))),
            cb_dcd_max_window_length=int(dcd_cfg.get("max_window_length", 64)),
            cb_dcd_window_type=str(confidence_cfg.get("window_type", dcd_cfg.get("window_type", "sliding"))),
            cb_dcd_decode_algo=str(dcd_cfg.get("decode_algo", "threshold")),
            cb_dcd_decode_param=dcd_cfg.get("decode_param", confidence_cfg.get("tau_cb", 0.75)),
            cb_dcd_sample_top_k=_optional_int(dcd_cfg.get("top_k")), cb_dcd_top_p=_optional_float(dcd_cfg.get("top_p")),
            cb_dcd_cache_type=str(confidence_cfg.get("cache_type", dcd_cfg.get("cache_type", "none"))),
            cb_spd_top_k=int(spd_cfg.get("top_k", 1)), cb_spd_renormalize=bool(spd_cfg.get("renormalize", True)),
            cb_spd_revision=bool(confidence_cfg.get("revision", spd_cfg.get("revision", True))),
            cb_temperature=float(dcd_cfg.get("temperature", 0.0)),
            **gate_loss_kwargs,
        )
        row = {k: float(v.detach().cpu().item()) for k, v in output.logs.items() if v.numel() == 1}
        if float(slt_cfg.get("lambda_bio", 1.0)) != 0.0 and output.bio_logits is not None:
            # Reuse forward_loss's own head forward (identical inputs, eval mode, no_grad) — a 2nd extract_bio_tap + bio_head pass here 
            # was pure recompute. forward_loss already passes frame_mask, so padded frames never enter conv stem / RoPE as real frames.
            bio_logits = output.bio_logits
            row.update(bio_frame_metrics(bio_logits, batch["bio_labels"], prefix="bio"))
            # Moryossef-style span metrics (frame + segment F1/IoU) under inference decode (runs split at interior
            # Bs), so dev tracks span quality — what RQ2 streaming needs — not just per-frame BIO accuracy.
            # Skipped at lambda_bio=0: the head is frozen at random init, so its metrics are noise.
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
                    _commit_mask = batch.get("commit_mask")
                    _, tokens, _, _ = model.generate_from_poses(
                        poses=batch["poses"][idx], frame_mask=batch["frame_mask"][idx],
                        commit_mask=_commit_mask[idx] if _commit_mask is not None else None,
                        timestamps_s=batch.get("timestamps_s", None)[idx] if batch.get("timestamps_s") is not None else None,
                        max_text_tokens=int(slt_cfg.get("max_text_tokens", 128)),
                        diffusion_steps=int(validation_cfg.get("diffusion_steps", slt_cfg.get("diffusion_steps", 64))),
                        tau_dec=float(dcd_cfg.get("tau_dec", 0.9)),  # same fallback as eval.py
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
        # Hyp/ref length ratio (BLEU brevity-penalty input, char-level for CJK): early-EOS diagnostic — < 1 and FALLING across epochs 
        # means the decode commits EOS ever earlier (eos_supervision / commit-threshold pressure), which BLEU/CIDEr punish as brevity.
        # WORD tokens, matching BLEU's BP. Characters disagree with it materially, so char ratio reads healthy while BLEU is penalised.
        total_ref = sum(len(r.split()) for r in ref_texts)
        metrics["val_translation_len_ratio"] = float(sum(len(p.split()) for p in pred_texts)) / max(1, total_ref)
    return metrics


def train_slt_epochs(
    model: MisalignedSLTModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epochs: int, 
    slt_cfg: dict, dev_loader: DataLoader | None = None, resume: bool = False, checkpoint_meta: dict | None = None,
) -> list[dict[str, float]]:
    confidence_cfg = slt_cfg.get("confidence_bound", {})
    dcd_cfg = slt_cfg.get("dcd", {})
    oput_cfg = slt_cfg.get("oput", {})
    spd_cfg = slt_cfg.get("spd", {})
    gate_cfg = slt_cfg.get("membership_gate", {})
    dice_weight = float(slt_cfg.get("dice_loss_weight", 1.5))
    decoder_name = getattr(model, "decoder_type", "dlm")

    # OPUT warmup holds the confidence-bound term off until full-evidence decode is trustworthy; gate warmup holds Ω off while a fresh 
    # BIO head sharpens on Dice (0 when bio_head_init is present — prefer a real S1 pretrain). Per-epoch flags, feeding step AND eval.
    cb_warmup_epochs = int(confidence_cfg.get("warmup_epochs", 1))
    cb_lambda = float(confidence_cfg.get("lambda", 1.0))
    gate_enabled_cfg = bool(gate_cfg.get("enabled", False))
    gate_warmup_epochs = int(gate_cfg.get("warmup_epochs", 0))
    # There is no safe default: warmup 0 is only correct when a trained S1 head was loaded.
    if gate_enabled_cfg and gate_warmup_epochs == 0 and not slt_cfg.get("checkpoint", {}).get("bio_head_init"): raise ValueError(
        "membership_gate.enabled with warmup_epochs: 0 and no bio_head_init couples the gate to an untrained head "
        "from epoch 1 — set warmup_epochs >= 2 or provide checkpoint.bio_head_init."
    )

    def _gate_active(epoch: int) -> bool:
        return gate_enabled_cfg and epoch > gate_warmup_epochs

    def step_fn(batch, epoch: int):
        output: SLTLossOutput = model.forward_loss(
            batch, lambda_trans=float(slt_cfg.get("lambda_trans", 1.0)), lambda_bio=float(slt_cfg.get("lambda_bio", 1.0)), 
            dice_weight=dice_weight, bio_class_weights=bio_class_weight_tensor(slt_cfg.get("bio_class_weights")),
            oput_t_low=float(oput_cfg.get("t_low", 0.3)), oput_t_high=float(oput_cfg.get("t_high", 0.8)),
            oput_sample_rollout=bool(oput_cfg.get("sample_rollout", False)),
            oput_label_smoothing=float(oput_cfg.get("label_smoothing", 0.0)),
            oput_rollout_eval_mode=bool(oput_cfg.get("rollout_eval_mode", True)),
            oput_eos_supervision=int(oput_cfg.get("eos_supervision_tokens", slt_cfg.get("block_size", 8))),
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
            cb_spd_top_k=int(spd_cfg.get("top_k", 1)),
            cb_spd_renormalize=bool(spd_cfg.get("renormalize", True)),
            cb_spd_revision=bool(confidence_cfg.get("revision", spd_cfg.get("revision", True))),
            cb_temperature=float(dcd_cfg.get("temperature", 0.0)),
            gate_enabled=_gate_active(epoch),
            # Same δ as the inference commit gate's delta_enc_frames (configs/inference.yaml).
            gate_delta=int(gate_cfg.get("delta", 3)),
            gate_eps=float(gate_cfg.get("eps", 1e-4)),
            gate_min_span_frames=int(gate_cfg.get("min_span_frames", 0)),
            gate_iou_veto=float(gate_cfg.get("iou_veto", 0.5)),
            gate_gt_anchored=bool(gate_cfg.get("gt_anchored", False)),
        )
        return output.loss, {k: float(v.detach().cpu().item()) for k, v in output.logs.items() if v.numel() == 1}

    def evaluate_fn(epoch: int): # Same gate/CB warmup state the epoch trained under (see evaluate_slt).
        return evaluate_slt(model, dev_loader, device, slt_cfg=slt_cfg,
                            gate_active=_gate_active(epoch), cb_active=epoch > cb_warmup_epochs)

    return run_epoch_loop(
        name=f"slt-{decoder_name}", model=model, loader=loader, optimizer=optimizer, device=device, epochs=epochs,
        cfg=slt_cfg, step_fn=step_fn, evaluate_fn=evaluate_fn, default_monitor="val_loss", default_mode="min",
        dev_loader=dev_loader, resume=resume, checkpoint_meta=checkpoint_meta,
    )
