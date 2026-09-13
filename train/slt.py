from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Tokenizer

from data.batch import WindowCollator
from data.loader import ANNOTATION_PROTOCOL, StreamingWindowDataset, annotation_fingerprint, load_language_records, streaming_loader 
from models.unisign import UniSignMT5FrontEnd, UniSignMBartFrontEnd, prompt_lang_for_target
from models.streaming_slt import MisalignedSLTModel, SLTLossOutput
from infer.duration_decode import DurationModel, DurationDecoder

from train import distributed as dist
from train.losses import bio_class_weight_tensor, resolve_bio_class_weights
from train.helpers import build_optimizer, mean_logs, move_to_device, resolve_lrs, run_epoch_loop
from metrics import char_level_for_target, bio_frame_metrics, compute_text_metrics, CompleteSpanMetrics
from utils import checkpoint_dir, lambda_min_frames, load_yaml, language_model_name, pool_key, resolve_inference, resolve_pretrained


# DEFAULT S1 config. `checkpoint.bio_head_init: auto` resolves the S1 checkpoint through it, and the pool-provenance check reads its 
# `pretrain_languages`. `train.py --bio-config` overrides it, so stage 1 and 2 read SAME S1 recipe when a run uses a non-default one.
BIO_S1_CONFIG = "configs/bio_pretrain.yaml"
# Gate options; delta and minimum eligible length come from resolved inference geometry.
GATE_CONFIG_KEYS = frozenset({"enabled", "eps", "warmup_epochs"})


@dataclass
class SLTComponents:
    model: MisalignedSLTModel
    tokenizer: Any
    train_loader: DataLoader
    dev_loader: DataLoader | None
    slt_cfg: dict
    checkpoint_meta: dict

def _optional_int(value) -> int | None:
    return None if value is None else int(value)

def _optional_float(value) -> float | None:
    return None if value is None else float(value)

def _inject_gate_geometry(slt_cfg: dict, inference_cfg: dict) -> None:
    # Match the sampler and FSM's first eligible target. Short units remain legal paths;
    # they are skipped by this target-selection rule, not removed from the path distribution.
    gate = slt_cfg.setdefault("membership_gate", {})
    # Every reader below uses .get() defaults, so a removed or misspelled key would silently train a different objective.
    unknown = set(gate) - GATE_CONFIG_KEYS
    if unknown: raise ValueError(f"membership_gate: unknown key(s) {sorted(unknown)}; accepted: {sorted(GATE_CONFIG_KEYS)}")
    gate["delta"] = int(inference_cfg.get("boundary_stability", {}).get("delta_enc_frames", 3))
    gate["min_span_frames"] = lambda_min_frames(inference_cfg)


def _training_meta(slt_cfg: dict, inference_cfg: dict, language: str) -> dict:
    """The config this stage-2 run is parameterized by, travelling with the weights.

    δ/Λ_min are re-measured by `analyze --stage delta-enc`, buffer_cap_s by buffer-cap, and jitter by segmenter-error analysis. 
    Resuming across such a change trains 2 halves under different objectives, and without this record nothing in the artifacts 
    shows it (models/checkpointing.save_model_checkpoint makes the same argument for S1's chunk size).
    """
    gate_cfg = slt_cfg.get("membership_gate", {}) or {}
    learning_rate, backbone_lr = resolve_lrs(slt_cfg)  # effective rates, whichever key spelling the config used
    return {
        # A changed generated-dev protocol invalidates cached best scores, not the learned weights.
        **({"validation_conditioning": "predicted_from_window"} if gate_cfg.get("enabled") else {}),
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": slt_cfg.get("annotation_fingerprint"),
        "language": str(language), "decoder": str(slt_cfg.get("decoder", "dlm")),
        "architecture": "shared_temporal_slt" if float(slt_cfg.get("lambda_bio", 1.0)) else "clean_translation",
        "bio_objective": "ce_dice", "dice_loss_weight": float(slt_cfg.get("dice_loss_weight", 1.5)),
        "gate": {k: gate_cfg.get(k) for k in ("enabled", "delta", "min_span_frames", "eps")},
        "buffer_cap_s": inference_cfg.get("buffer_cap_s"), 
        "segmentation_decode": "semi_markov_viterbi" if slt_cfg.get("duration_model") else "none",
        "duration_model": slt_cfg.get("duration_model"), "confidence_bound": slt_cfg.get("confidence_bound", {}), 
        "oput": slt_cfg.get("oput", {}), "spd": slt_cfg.get("spd", {}), "dcd": slt_cfg.get("dcd", {}), 
        "gate_warmup_epochs": int(gate_cfg.get("warmup_epochs", 0)),
        "mode_ratios": slt_cfg.get("mode_ratios"), "jitter": slt_cfg.get("jitter"),
        "bio_class_weights": slt_cfg.get("bio_class_weights"),  # resolved list, not the "balanced" string
        "lambda_bio": float(slt_cfg.get("lambda_bio", 1.0)), "lambda_trans": float(slt_cfg.get("lambda_trans", 1.0)),
        "batch_size": slt_cfg.get("batch_size"), "learning_rate": learning_rate, "backbone_lr": backbone_lr,
    }


def build_slt_optimizer(slt_cfg: dict, model) -> torch.optim.Optimizer:
    """Stage-2 optimizer: the warm-started pose encoder and, when it was loaded from S1, the BIO head at `backbone_lr`;
    everything else (a random-init head included) at `learning_rate`.

    Discriminative fine-tuning (same rule as stage 1): the LM-scale rate would overwrite what the segmentation objective
    already adapted. Frozen parameters (freeze_backbone, lambda_bio 0) are in no group."""
    head_from_s1 = bool(getattr(model, "bio_head_from_s1", False))
    mods = [model.front_end.pose_encoder] + ([model.bio_head] if head_from_s1 else [])
    pretrained = {id(p) for m in mods for p in m.parameters()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    backbone = [p for p in trainable if id(p) in pretrained]
    main = [p for p in trainable if id(p) not in pretrained]
    lr, backbone_lr = resolve_lrs(slt_cfg)
    label = "S1-pretrained (pose_encoder+bio_head)" if head_from_s1 else "warm-started pose_encoder (bio_head random init, full lr)"
    print(f"slt | optimizer: main {sum(p.numel() for p in main) / 1e6:.2f}M @ lr={lr:g} | {label} "
          f"{sum(p.numel() for p in backbone) / 1e6:.2f}M @ backbone_lr={backbone_lr:g}", flush=True)
    return build_optimizer(slt_cfg, main, backbone_params=backbone)


def assert_targets_fit(records, tokenizer, max_text_tokens: int, buffer_cap_s: float, split: str) -> None:
    """Refuse to start if any unit that can become a complete translation target exceeds the text canvas.

    The collator never truncates a complete-caption target (truncated reference silently rewrites the task), so an over-long target 
    would crash mid-epoch instead. Only a unit that fits streaming buffer can be a complete anchor (train/sampler.py `_clip_window`); 
    this mirrors that predicate and reports capacity to configure, same way buffer_cap_s is sized to data rather than data to constant.
    """
    longest, culprit = 0, None
    for record in records:
        for span in record.sentences:
            if not getattr(span, "reliable", True) or span.duration_s + 1.0 / record.pose.fps > float(buffer_cap_s): continue
            n = len(tokenizer(span.text)["input_ids"])
            if n > longest: longest, culprit = n, span
    if longest > int(max_text_tokens): raise ValueError(
        f"{split}: a complete caption target needs {longest} tokens but max_text_tokens is {max_text_tokens} "
        f"({culprit.video_id} {culprit.start_s:.1f}-{culprit.end_s:.1f}s). Set max_text_tokens >= {longest}: the collator refuses "
        f"to truncate a complete target, and would raise on this unit the first time it is sampled."
    )


def build_slt_components(
    data_config: str = "configs/data.yaml", slt_config: str = "configs/dlm.yaml", inference_config: str = "configs/inference.yaml",
    decoder: str | None = None, include_dev: bool = False, language: str | None = None, bio_config: str = BIO_S1_CONFIG,
) -> SLTComponents:
    data_cfg = load_yaml(data_config)
    slt_cfg = load_yaml(slt_config)
    # Precedence: --language > config `language:` > active_languages. Reload on override so ${language} 
    # in checkpoint.dir (+ ar/baseline children) re-points at the right dataset.
    language = str(language or slt_cfg.get("language") or data_cfg.get("active_languages", ["asf"])[0])
    if language != slt_cfg.get("language"): slt_cfg = load_yaml(slt_config, language=language)
    inference_cfg = resolve_inference(load_yaml(inference_config), language)
    _inject_gate_geometry(slt_cfg, inference_cfg)
    
    slt_cfg["decoder"] = decoder or str(slt_cfg.get("decoder", "dlm"))
    duration = DurationModel.from_config(inference_cfg, language) if float(slt_cfg.get("lambda_bio", 1.)) else None
    slt_cfg["duration_model"] = duration.to_dict() if duration else None
    train_records, _ = load_language_records(data_cfg, language, split="train")
    slt_cfg["annotation_fingerprint"] = annotation_fingerprint(train_records)
    if duration:
        duration.require_annotations(train_records)
        duration.require_calibration(inference_cfg, language)

    target_lang = data_cfg["languages"][language].get("target_lang", "en_XX")
    slt_cfg["target_lang"] = target_lang  # metric scoring level is declared, not sniffed (metrics.char_level_for_target)
    # Uni-Sign front end. language_model.name picks the LM + tokenizer: mT5 (Path A default) or mBART
    # (mT5-vs-mBART ablation); same pose encoder + prompt either way.
    lm_name = language_model_name(slt_cfg)
    prompt_lang = prompt_lang_for_target(target_lang)
    if "mbart" in lm_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(lm_name, src_lang=target_lang, tgt_lang=target_lang)
        front_end = UniSignMBartFrontEnd(mbart_name=lm_name, prompt_lang=prompt_lang, target_lang=target_lang, tokenizer=tokenizer)
    else:
        tokenizer = T5Tokenizer.from_pretrained(lm_name, legacy=False)
        front_end = UniSignMT5FrontEnd(mt5_name=lm_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)

    pose_augment_cfg = slt_cfg.get("augmentation")  # train-only spatial aug; dev passes None
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
    assert_targets_fit(train_records, tokenizer, collator.max_text_tokens, inference_cfg["buffer_cap_s"], f"{language}/train")
    # num_workers is pure throughput: anchors are index-driven (each realized once per epoch regardless of worker
    # split) and workers reseed their rng (data.loader.streaming_loader / WindowSampler.configure_worker).
    num_workers = int(slt_cfg.get("num_workers", 0))
    train_loader = streaming_loader(
        train_dataset, dist.per_rank_batch_size(int(slt_cfg.get("batch_size", 4))), collator, num_workers=num_workers,
        # See train/bio_pretrain.py: length bucketing, same coverage, fewer padded frames.
        bucket_by_length=bool(slt_cfg.get("bucket_by_length", True)), bucket_seed=int(slt_cfg.get("seed", 42)),
    )
    dev_loader = None
    if include_dev:
        dev_records, _ = load_language_records(data_cfg, language, split="dev")
        assert_targets_fit(dev_records, tokenizer, collator.max_text_tokens, inference_cfg["buffer_cap_s"], f"{language}/dev")
        # Dev scoring should cover the same experimental unit as standard SLT training: 1 sentence anchor, not 1 video.
        # With len(dev_records), validation sampled only 1 fixed window per video and could miss most sentences.
        dev_steps = sum(sum(1 for sp in record.sentences if getattr(sp, 'reliable', True)) for record in dev_records)
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
        front_end=front_end,
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
        shared_temporal=float(slt_cfg.get("lambda_bio", 1.0)) != 0.,
    )
    model.duration_model = duration
    # S1 BIO init (docs/membership_gate.md §1.4 "competence before coupling"): load the pre-trained head from
    # train-bio so S2 trains exactly one new thing — the coupling — and membership_gate.warmup_epochs can be 0.
    bio_init = slt_cfg.get("checkpoint", {}).get("bio_head_init")
    bio_cfg = load_yaml(bio_config, language=language)
    # `auto` DERIVES the path via the same resolver every other consumer uses, so pooled S1 is found by its pool key instead 
    # of a literal copied into this config. Hardcoded `checkpoints/bio_s1/multi_<pool>/model.pt` goes stale the moment 
    # `pretrain_languages` changes, and stage 2 would silently initialise the gate from another pool's head.
    if str(bio_init).lower() == "auto": bio_init = str(Path(checkpoint_dir(bio_cfg, default="checkpoints/bio_s1")) / "model.pt")
    if float(slt_cfg.get("lambda_bio", 1.0)) == 0.0:
        if bool(slt_cfg.get("membership_gate", {}).get("enabled", False)): raise SystemExit(
            "lambda_bio: 0 with membership_gate.enabled: true is incoherent — the gate reads the BIO head's posteriors, but "
            "lambda_bio: 0 skips the head's forward and leaves it untrained/frozen. Either train the head (lambda_bio > 0) "
            "or disable the gate (the clean-floor recipe does both)."
        )
        # Clean-floor recipe (lambda_bio: 0 — baseline_train.yaml): no BIO branch. Skip S1 init entirely, head AND its pose 
        # encoder, and freeze the head so the optimizer never sees it. forward_loss skips its forward, so the branch costs nothing. 
        # The floor must stay the PRIOR-ART recipe: S1's encoder is adapted by SEGMENTATION objective, and transplanting it into 
        # translation-only baseline would neither match the arms (which need it only so their BIO head meets features it trained on) 
        # nor keep this row a faithful Uni-Sign transfer. The mono-vs-multi S1 ablation is where the pool's contribution is measured.
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
        # PROVENANCE, same rule as eval.py: the checkpoint records which pool trained it. Stage 2 must not warm-start
        # the gate from a head trained on a different pool — that is a different model, and the failure is silent.
        _s1_meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
        _want = pool_key(bio_cfg)
        if "pretrain_pool" in _s1_meta and _s1_meta.get("pretrain_pool") != _want: raise SystemExit(
            f"{bio_init} was trained on pool {_s1_meta.get('pretrain_pool')!r}, but bio_pretrain.yaml now expects "
            f"{_want!r}. Retrain S1 for this pool, or point checkpoint.bio_head_init at the matching model."
        )
        # Context-coverage guard: this language's deployment cap must not exceed the S1 init's TRAINED RoPE context, 
        # or every over-context S2 window stretches the pretrained range (the coverage rule pretrain_geometry documents). 
        # Checked against the CHECKPOINT's stamp, not a config that can drift.
        _s1_ctx = _s1_meta.get("rope_eval_chunk_s")
        _cap = float(inference_cfg.get("buffer_cap_s", 0.0) or 0.0)
        if _s1_ctx and _cap > float(_s1_ctx) + 1e-6: print(
            f"slt | WARNING: buffer_cap_s {_cap:.2f}s for language {language!r} exceeds S1 init's trained context "
            f"{float(_s1_ctx):.2f}s — raise pretrain_geometry.buffer_cap_s to cover it & retrain S1, or S2 trains "
            f"the head beyond its pretrained RoPE range.", flush=True
        )
        model.bio_head.load_state_dict(head_sd, strict=True)
        model.bio_head_from_s1 = True   # build_slt_optimizer: the head takes backbone_lr only when loaded from S1
        # Carry S1's pose encoder so the head meets the features it trained on. A no-op only when S1 froze an encoder
        # warm-started from the SAME checkpoint as this run; with a trained S1 encoder (the shipped pooled recipe) the LM
        # encoder meets pose features adapted by segmentation, a real init shift — stated in the log line below.
        pose_sd = {k[len("pose_encoder."):]: v for k, v in sd.items() if k.startswith("pose_encoder.")}
        if pose_sd: model.front_end.pose_encoder.load_state_dict(pose_sd, strict=True)
        enc_note = (f"S1 pose encoder OVERRIDES the warm-start ({len(pose_sd)} tensors) — deliberate: the head must meet the "
                    "features it trained on (S1 features == S2 initial features); the LM encoder starts on these adapted "
                    "features, not the ones it was fine-tuned on") if pose_sd else "S1 checkpoint carries no pose encoder"
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
    spans = CompleteSpanMetrics()

    confidence_cfg = slt_cfg.get("confidence_bound", {})
    dcd_cfg = slt_cfg.get("dcd", {})
    oput_cfg = slt_cfg.get("oput", {})
    spd_cfg = slt_cfg.get("spd", {})
    gate_cfg = slt_cfg.get("membership_gate", {})

    # Score dev under SAME gate/CB state the epoch trained under: during warmup decoder has never seen Ω or the CB self-target, so either 
    # one on reports an untrained objective & burns a CB decode on untrustworthy targets. cb_active=None (standalone eval) = full objective.
    gate_on = bool(gate_cfg.get("enabled", False)) if gate_active is None else bool(gate_active)
    cb_on = True if cb_active is None else bool(cb_active)
    gate_kwargs = dict(
        gate_enabled=gate_on, gate_eps=float(gate_cfg.get("eps", 1e-4)), gate_min_span_frames=int(gate_cfg.get("min_span_frames", 0))
    )
    gate_loss_kwargs = gate_kwargs
    dice_weight = float(slt_cfg.get("dice_loss_weight", 1.5))
    validation_cfg = slt_cfg.get("validation", {})
    max_translation_samples = int(validation_cfg.get("max_translation_samples", 0) or 0) # <= 0: translate ALL supervised dev windows.
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
            cb_enabled=bool(confidence_cfg.get("enabled", True)), cb_active=cb_on,
            cb_tau=float(confidence_cfg.get("tau_cb", 0.75)), cb_lambda=float(confidence_cfg.get("lambda", 1.0)),
            cb_verified_gate=bool(confidence_cfg.get("verified_full_evidence_gate", True)),
            cb_belief_gap=bool(confidence_cfg.get("belief_gap", True)),
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
            cb_temperature=float(dcd_cfg.get("temperature", 0.0)), **gate_loss_kwargs,
        )
        row = {k: float(v.detach().cpu().item()) for k, v in output.logs.items() if v.numel() == 1}
        if float(slt_cfg.get("lambda_bio", 1.0)) != 0.0 and output.bio_logits is not None:
            # Reuse forward_loss's own head forward (identical inputs, eval mode, no_grad) — a 2nd extract_bio_tap + bio_head pass here 
            # was pure recompute. forward_loss already passes frame_mask, so padded frames never enter conv stem / RoPE as real frames.
            bio_logits = output.bio_logits
            row.update(bio_frame_metrics(bio_logits, batch["bio_labels"], prefix="bio"))
            # Score the visible window with calibrated duration scores; the gate separately excludes committed context.
            _lengths = batch["frame_mask"].long().sum(dim=1)
            tags = DurationDecoder(model.duration_model).decode(bio_logits, _lengths, timestamps_s=batch["timestamps_s"])
            spans.update(tags, batch["bio_labels"], _lengths)
        rows.append(row)

        cap_reached = max_translation_samples > 0 and len(pred_texts) >= max_translation_samples
        if not cap_reached:
            supervised = batch.get("translation_supervised")
            targets = batch.get("translation_targets", [])

            if isinstance(supervised, torch.Tensor) and supervised.any():
                idx = supervised.nonzero(as_tuple=False).flatten()
                if max_translation_samples > 0: idx = idx[: max_translation_samples - len(pred_texts)]
                if idx.numel() > 0: # A dev window has no observed commit history; sampler labels are supervision only.
                    _, tokens, _, _ = model.generate_from_poses(
                        poses=batch["poses"][idx], frame_mask=batch["frame_mask"][idx],
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
                        ref_texts.append(str(target.get("text", "")) if isinstance(target, dict) else str(getattr(target, "text", "")))

    if was_training: model.train()
    metrics = mean_logs(rows, prefix="val")
    if float(slt_cfg.get("lambda_bio", 1.)) != 0.: metrics.update(spans.compute(prefix="val_phrase"))
    if pred_texts:
        metrics.update(compute_text_metrics(
            pred_texts, ref_texts, prefix="val_translation", char_level=char_level_for_target(slt_cfg.get("target_lang"))
        ))
        # Hyp/ref length ratio (BLEU brevity-penalty input, char-level for CJK): early-EOS diagnostic — < 1 and FALLING across epochs 
        # means the decode commits EOS ever earlier (eos_supervision / commit-threshold pressure), which BLEU/CIDEr punish as brevity.
        # WORD tokens, matching BLEU's BP. Characters disagree with it materially, so char ratio reads healthy while BLEU is penalised.
        total_ref = sum(len(r.split()) for r in ref_texts)
        metrics["val_translation_len_ratio"] = float(sum(len(p.split()) for p in pred_texts)) / max(1, total_ref)
    # A dev-window trade-off score, not DVC and not a guarantee of retained localization.
    if "val_phrase_tiou_f1" in metrics and "val_translation_bleu4" in metrics:
        metrics["val_joint_score"] = float(metrics["val_phrase_tiou_f1"]) * float(metrics["val_translation_bleu4"])
    return metrics


def training_loss(model, batch: dict, slt_cfg: dict, epoch: int) -> SLTLossOutput:
    # The actual AR/DLM training objective, shared with initial loss-scale calibration.
    confidence_cfg = slt_cfg.get("confidence_bound", {})
    dcd_cfg = slt_cfg.get("dcd", {})
    oput_cfg = slt_cfg.get("oput", {})
    spd_cfg = slt_cfg.get("spd", {})
    gate_cfg = slt_cfg.get("membership_gate", {})
    dice_weight = float(slt_cfg.get("dice_loss_weight", 1.5))
    cb_warmup_epochs = int(confidence_cfg.get("warmup_epochs", 1))
    cb_lambda = float(confidence_cfg.get("lambda", 1.0))
    return model.forward_loss(
        batch, lambda_trans=float(slt_cfg.get("lambda_trans", 1.0)), lambda_bio=float(slt_cfg.get("lambda_bio", 1.0)),
        dice_weight=dice_weight, bio_class_weights=bio_class_weight_tensor(slt_cfg.get("bio_class_weights")),
        oput_t_low=float(oput_cfg.get("t_low", 0.3)), oput_t_high=float(oput_cfg.get("t_high", 0.8)),
        oput_sample_rollout=bool(oput_cfg.get("sample_rollout", False)),
        oput_label_smoothing=float(oput_cfg.get("label_smoothing", 0.0)),
        oput_rollout_eval_mode=bool(oput_cfg.get("rollout_eval_mode", True)),
        oput_eos_supervision=int(oput_cfg.get("eos_supervision_tokens", slt_cfg.get("block_size", 8))),
        cb_enabled=bool(confidence_cfg.get("enabled", True)),
        cb_active=epoch > cb_warmup_epochs,
        cb_tau=float(confidence_cfg.get("tau_cb", 0.75)),
        cb_lambda=cb_lambda,
        cb_verified_gate=bool(confidence_cfg.get("verified_full_evidence_gate", True)),
        cb_belief_gap=bool(confidence_cfg.get("belief_gap", True)),
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
        gate_enabled=bool(gate_cfg.get("enabled", False)) and epoch > int(gate_cfg.get("warmup_epochs", 0)),
        # Same δ as the inference commit gate's delta_enc_frames (configs/inference.yaml).
        gate_eps=float(gate_cfg.get("eps", 1e-4)),
        gate_min_span_frames=int(gate_cfg.get("min_span_frames", 0)),
    )


def train_slt_epochs(
    model: MisalignedSLTModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epochs: int,
    slt_cfg: dict, dev_loader: DataLoader | None = None, resume: bool = False, checkpoint_meta: dict | None = None,
) -> list[dict[str, float]]:
    confidence_cfg = slt_cfg.get("confidence_bound", {})
    gate_cfg = slt_cfg.get("membership_gate", {})
    decoder_name = getattr(model, "decoder_type", "dlm")
    if float(slt_cfg.get("lambda_bio", 1.)) != 0. and dist.is_main():
        print("slt | localization monitor: complete-span F1@0.5, pooled dev-window counts; RQ2 scores whole videos separately", flush=True)

    # OPUT warmup holds the confidence-bound term off until full-evidence decode is trustworthy; gate warmup holds Ω off while a fresh 
    # BIO head sharpens on Dice (0 when bio_head_init is present — prefer a real S1 pretrain). Per-epoch flags, feeding step AND eval.
    cb_warmup_epochs = int(confidence_cfg.get("warmup_epochs", 1))
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
        output = training_loss(model, batch, slt_cfg, epoch)
        return output.loss, {k: float(v.detach().cpu().item()) for k, v in output.logs.items() if v.numel() == 1}

    def evaluate_fn(epoch: int): # Same gate/CB warmup state the epoch trained under (see evaluate_slt).
        return evaluate_slt(model, dev_loader, device, slt_cfg=slt_cfg, gate_active=_gate_active(epoch), cb_active=epoch > cb_warmup_epochs)

    return run_epoch_loop(
        name=f"slt-{decoder_name}", model=model, loader=loader, optimizer=optimizer, device=device, epochs=epochs,
        cfg=slt_cfg, step_fn=step_fn, evaluate_fn=evaluate_fn, default_monitor="val_loss", default_mode="min",
        dev_loader=dev_loader, resume=resume, checkpoint_meta=checkpoint_meta,
    )
