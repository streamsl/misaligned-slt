from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Tokenizer

from data.batch import WindowCollator
from data.loader import ANNOTATION_PROTOCOL, StreamingWindowDataset, annotation_fingerprint, load_language_records, streaming_loader 
from models.unisign import UniSignMT5FrontEnd, UniSignMBartFrontEnd, PROMPT_LANG_BY_TARGET
from models.streaming_slt import MisalignedSLTModel, SLTLossOutput
from models.checkpointing import load_checkpoint_meta, require_fixed_constants
from infer.duration_decode import DurationModel, DurationDecoder

from train import distributed as dist
from train.bio_pretrain import require_attention_radius, require_segmenter_data
from train.losses import bio_class_weight_tensor, resolve_bio_class_weights
from train.helpers import mean_logs, move_to_device, run_epoch_loop
from metrics import char_level_for_target, bio_frame_metrics, compute_text_metrics, CompleteSpanMetrics
from utils import cfg_get, checkpoint_dir, lambda_min_frames, load_yaml, pool_key, resolve_pretrained


# DEFAULT S1 config. `checkpoint.bio_head_init: auto` resolves the S1 checkpoint through it, and the pool-provenance check reads its 
# `pretrain_languages`. `train.py --bio-config` overrides it, so stage 1 and 2 read SAME S1 recipe when a run uses a non-default one.
BIO_S1_CONFIG = "configs/bio_pretrain.yaml"
# Gate options; delta and the minimum eligible length come from the fixed inference.yaml constants.
GATE_CONFIG_KEYS = frozenset({"enabled", "eps", "detach_omega"})
SPD_CONFIG_KEYS = frozenset({"tau_dec", "top_k", "renormalize"})


@dataclass
class SLTComponents:
    model: MisalignedSLTModel
    tokenizer: Any
    train_loader: DataLoader
    dev_loader: DataLoader | None
    slt_cfg: dict
    checkpoint_meta: dict

def _file_stamp(path) -> dict | None: # Which file initialized a branch, with its size: a path alone doesn't show a later overwrite.
    if not path: return None
    f = Path(str(path))
    return {"path": str(path), "bytes": f.stat().st_size if f.exists() else None}

def _inject_gate_geometry(slt_cfg: dict, inference_cfg: dict) -> None:
    # Match the sampler and FSM's first eligible target. Short units remain legal paths;
    # they are skipped by this target-selection rule, not removed from the path distribution.
    gate = slt_cfg.setdefault("membership_gate", {})
    # Every reader below uses .get() defaults, so a removed or misspelled key would silently train a different objective.
    unknown = set(gate) - GATE_CONFIG_KEYS
    if unknown: raise ValueError(f"membership_gate: unknown key(s) {sorted(unknown)}; accepted: {sorted(GATE_CONFIG_KEYS)}")
    unknown = set(slt_cfg.get("spd", {}) or {}) - SPD_CONFIG_KEYS  # same reason: a misspelled tau_dec would decode at default
    if unknown: raise ValueError(f"spd: unknown key(s) {sorted(unknown)}; accepted: {sorted(SPD_CONFIG_KEYS)}")
    gate["delta"] = int(inference_cfg["boundary_stability"]["delta_enc_frames"])
    gate["min_span_frames"] = lambda_min_frames(inference_cfg)


def _training_meta(slt_cfg: dict, inference_cfg: dict, language: str) -> dict:
    """The config this stage-2 run is parameterized by, travelling with the weights.

    δ, Λ_min and buffer_cap_s come from inference.yaml and the duration prior from the train labels. Resuming across a change 
    of any of them trains 2 halves under different objectives, and without this record nothing in the artifacts shows it.
    """
    gate_cfg = slt_cfg.get("membership_gate", {}) or {}
    return {
        # A changed generated-dev protocol invalidates cached best scores, not the learned weights.
        **({"validation_conditioning": "predicted_from_window"} if gate_cfg.get("enabled") else {}),
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": slt_cfg.get("annotation_fingerprint"),
        "language": str(language), "decoder": str(slt_cfg.get("decoder", "dlm")),
        # two_stream_slt: separate segmentation/translation pose encoders, Ω on encoder keys and cross-attention. 
        # Eval refuses a stage-2 file with any other tag rather than load it into the wrong layout.
        "architecture": "two_stream_slt" if float(slt_cfg.get("lambda_bio", 1.0)) else "clean_translation",
        "bio_objective": "ce_dice", "dice_loss_weight": float(slt_cfg.get("dice_loss_weight", 1.5)),
        "gate": {k: gate_cfg.get(k) for k in ("enabled", "delta", "min_span_frames", "eps", "detach_omega")},
        # Which files initialized the translator and the segmentation branch, and the head's attention band.
        "resolved_inits": slt_cfg.get("resolved_inits"), "bio_attention_radius_s": slt_cfg.get("bio_attention_radius_s"),
        "early_stopping": slt_cfg.get("early_stopping"), "scheduler": slt_cfg.get("scheduler"), "epochs": slt_cfg.get("epochs"),
        "buffer_cap_s": inference_cfg.get("buffer_cap_s"), "duration_model": slt_cfg.get("duration_model"),
        "segmentation_decode": "semi_markov_viterbi" if slt_cfg.get("duration_model") else "none",
        "oput": slt_cfg.get("oput", {}), "spd": slt_cfg.get("spd", {}), 
        # Training geometry of the text canvas: the block-causal mask and the OPUT corruption read block_size, so eval
        # refuses a decoder built at another block (eval.py _build_eval_model), and the decode canvas must not shrink.
        "block_size": int(slt_cfg.get("block_size", 16)), "max_text_tokens": int(slt_cfg.get("max_text_tokens", 320)),
        "batch_size": slt_cfg.get("batch_size"), "learning_rate": float(slt_cfg["learning_rate"]),
        "mode_ratios": slt_cfg.get("mode_ratios"), "jitter": slt_cfg.get("jitter"), "augmentation": slt_cfg.get("augmentation"),
        "bio_class_weights": slt_cfg.get("bio_class_weights"),  # resolved list, not the "balanced" string
        "lambda_bio": float(slt_cfg.get("lambda_bio", 1.0)), "lambda_trans": float(slt_cfg.get("lambda_trans", 1.0)),
    }


def assert_targets_fit(records, tokenizer, max_text_tokens: int, buffer_cap_s: float, split: str) -> None:
    """Refuse to start if any unit that can become a complete translation target exceeds the text canvas.

    The collator never truncates a complete-caption target (truncated reference silently rewrites the task), so an over-long target 
    would crash mid-epoch instead. Only a unit that fits streaming buffer can be a complete anchor (data/sampler.py `_clip_window`); 
    this mirrors that predicate and reports capacity to configure, same way buffer_cap_s is sized to data rather than data to constant.
    """
    longest, culprit = 0, None
    for record in records:
        for span in record.sentences:
            if not getattr(span, "reliable", True) or span.duration_s + 1.0 / record.pose.fps > float(buffer_cap_s): continue
            n = len(tokenizer(span.text)["input_ids"])
            if n > longest: longest, culprit = n, span
    # +1: decode canvas holds BOS in slot 0, so a target of exactly max_text_tokens could be trained on but never emitted.
    if longest + 1 > int(max_text_tokens): raise ValueError(
        f"{split}: a complete caption target needs {longest} tokens plus the BOS slot but max_text_tokens is {max_text_tokens} "
        f"({culprit.video_id} {culprit.start_s:.1f}-{culprit.end_s:.1f}s). Set max_text_tokens >= {longest + 1}: the collator "
        f"refuses to truncate a complete target, and would raise on this unit the first time it is sampled."
    )


def require_translator_init(pretrained_path, segmentation_branch: bool, language: str, inference_cfg: dict, fingerprint) -> None:
    """The translator warm start of a train-slt run. The clean floor starts from the released Uni-Sign weights; an arm
    starts from THIS language's clean floor, trained on the current annotations under the fixed constants."""
    # A released Uni-Sign file while a trained clean translator exists for the language means 
    # the data.yaml re-root was forgotten: the arm would start elsewhere.
    local_floor = Path(f"checkpoints/baseline_train/{language}/model.pt")
    if segmentation_branch and Path(str(pretrained_path)).name.endswith("_pose_only_slt.pth") and local_floor.exists(): raise SystemExit(
        f"Stage 2 would warm-start the translator from the released {pretrained_path}, but {local_floor} exists. Set data.yaml "
        f"languages.{language}.pretrained_slt: {local_floor} (README B2b) so the arm starts at the clean translator."
    )
    # A trained warm start carries meta; the released Uni-Sign file carries none.
    init_meta = load_checkpoint_meta(pretrained_path) if Path(str(pretrained_path)).exists() else {}
    # After the B2b re-root, data.yaml pretrained_slt names the previous clean floor: warm-starting the floor from it would
    # silently fine-tune the old floor a second time.
    if not segmentation_branch and init_meta: raise SystemExit(
        f"The clean floor must warm-start from the released Uni-Sign weights, but {pretrained_path} is a trained checkpoint "
        f"(architecture={init_meta.get('architecture')!r}). Set checkpoint.from_pretrained to the released file "
        f"(baseline_train.yaml does), or point data.yaml languages.{language}.pretrained_slt back at it."
    )
    if segmentation_branch and init_meta:
        found = {"architecture": init_meta.get("architecture"), "language": init_meta.get("language"),
                 "annotation_fingerprint": init_meta.get("annotation_fingerprint")}
        want = {"architecture": "clean_translation", "language": str(language), "annotation_fingerprint": fingerprint}
        bad = [f"{k}={found[k]!r} (need {want[k]!r})" for k in want if found[k] != want[k]]
        if bad: raise SystemExit(
            f"Stage 2 warm-starts the translator from {pretrained_path}, but it is not this language's clean translator on the "
            f"current annotations: {'; '.join(bad)}. Retrain the clean baseline (README B2), then re-root it (B2b)."
        )
        require_fixed_constants(init_meta, inference_cfg, pretrained_path)   # the same rule eval applies to it

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
    # Stage 2 and the clean floor apply no pose augmentation; S1 has its own block in bio_pretrain.yaml.
    if slt_cfg.get("augmentation") is not None: raise SystemExit(
        f"{slt_config}: stage 2 applies no pose augmentation, but the config sets augmentation: {slt_cfg['augmentation']!r}. "
        f"Remove the block (S1 augmentation lives in bio_pretrain.yaml)."
    )
    inference_cfg = load_yaml(inference_config)
    _inject_gate_geometry(slt_cfg, inference_cfg)
    # Resolve the S1 init before any data loads, so a missing S1 fails in seconds. 
    # Both pretrained branches then adapt, with caption gradients coupled through Ω.
    bio_init = slt_cfg.get("checkpoint", {}).get("bio_head_init")
    bio_cfg = load_yaml(bio_config, language=language)
    # `auto` DERIVES the path via the same resolver every other consumer uses, so pooled S1 is found by its pool key instead 
    # of a literal copied into this config. Hardcoded `checkpoints/bio_s1/multi_<pool>/model.pt` goes stale the moment 
    # `pretrain_languages` changes, and stage 2 would silently initialise the gate from another pool's head.
    if str(bio_init).lower() == "auto": bio_init = str(Path(checkpoint_dir(bio_cfg, default="checkpoints/bio_s1")) / "model.pt")
    gate_on = bool(slt_cfg.get("membership_gate", {}).get("enabled", False))
    # The clean floor (baseline_train.yaml: no S1 init, gate off) has no segmentation branch; every other recipe has one.
    segmentation_branch = gate_on or bool(bio_init)
    if segmentation_branch != (float(slt_cfg.get("lambda_bio", 1.0)) != 0.0): raise SystemExit(
        "lambda_bio must be > 0 exactly when the recipe has a segmentation branch (gate on or bio_head_init set): "
        "without a branch there is no BIO loss, and a branch without BIO supervision is incoherent. The clean-floor "
        "recipe sets lambda_bio: 0, bio_head_init: null and membership_gate.enabled: false. The architecture stamp "
        "and the duration fit key on lambda_bio, so the two must agree."
    )
    # Every segmentation branch starts from S1: the gate is on from step 0, and coupled to an untrained head it would
    # condition the translator on noise. bio_head_init is cwd-relative, so a wrong-cwd launch (Colab default dir) lands here.
    if segmentation_branch and not (bio_init and Path(bio_init).exists()): raise SystemExit(
        f"checkpoint.bio_head_init {bio_init!r} not found: stage 2 needs a trained S1 for its segmentation branch. "
        f"Run `train.py --stage train-bio` first, or fix the path/cwd."
    )
    
    slt_cfg["decoder"] = decoder or str(slt_cfg.get("decoder", "dlm"))
    train_records, _ = load_language_records(data_cfg, language, split="train")
    slt_cfg["annotation_fingerprint"] = annotation_fingerprint(train_records)
    # Label-only duration prior of this language's train units; stamped in the checkpoint, so eval decodes with it.
    duration = DurationModel.fit(train_records) if float(slt_cfg.get("lambda_bio", 1.)) else None
    slt_cfg["duration_model"] = duration.to_dict() if duration else None

    target_lang = data_cfg["languages"][language].get("target_lang", "en_XX")
    slt_cfg["target_lang"] = target_lang  # metric scoring level is declared, not sniffed (metrics.char_level_for_target)
    # Uni-Sign front end. language_model.name picks the LM + tokenizer: mT5 (Path A default) or mBART
    # (mT5-vs-mBART ablation); same pose encoder + prompt either way.
    lm_name = str(cfg_get(slt_cfg, "language_model", "name", default="google/mt5-base"))
    prompt_lang = PROMPT_LANG_BY_TARGET.get(str(target_lang or ""), "English")
    if "mbart" in lm_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(lm_name, src_lang=target_lang, tgt_lang=target_lang)
        front_end = UniSignMBartFrontEnd(mbart_name=lm_name, prompt_lang=prompt_lang, target_lang=target_lang, tokenizer=tokenizer)
    else:
        tokenizer = T5Tokenizer.from_pretrained(lm_name, legacy=False)
        front_end = UniSignMT5FrontEnd(mt5_name=lm_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)

    resolve_bio_class_weights(slt_cfg, train_records)
    train_dataset = StreamingWindowDataset(train_records, slt_cfg=slt_cfg, inference_cfg=inference_cfg)
    collator = WindowCollator(
        tokenizer, max_text_tokens=int(slt_cfg.get("max_text_tokens", 320)),
        # The text canvas is sized to the batch, not to max_text_tokens: captions are ~15 tokens against a 320 
        # canvas and every decoder forward runs the whole width. The collator keeps the canvas block-aligned.
        block_size=int(slt_cfg.get("block_size", 16)),
    )
    assert_targets_fit(train_records, tokenizer, collator.max_text_tokens, inference_cfg["buffer_cap_s"], f"{language}/train")
    # num_workers is pure throughput: anchors are index-driven (each realized once per epoch regardless of worker split) 
    # and every draw reseeds its rng from the index (WindowSampler.spec_for), so no worker state matters.
    num_workers = int(slt_cfg.get("num_workers", 0))
    train_loader = streaming_loader(
        train_dataset, dist.per_rank_batch_size(int(slt_cfg.get("batch_size", 4))), collator, num_workers=num_workers,
        # See train/bio_pretrain.py: length bucketing, same coverage, fewer padded frames.
        bucket_by_length=True, bucket_seed=int(slt_cfg.get("seed", 42)),
    )
    dev_loader = None
    if include_dev:
        dev_records, _ = load_language_records(data_cfg, language, split="dev")
        assert_targets_fit(dev_records, tokenizer, collator.max_text_tokens, inference_cfg["buffer_cap_s"], f"{language}/dev")
        # Dev scores the same experimental unit as standard SLT training: 1 window per reliable sentence anchor 
        # (the dataset's length), not 1 per video. deterministic: the same dev windows every epoch.
        dev_dataset = StreamingWindowDataset(dev_records, slt_cfg=slt_cfg, inference_cfg=inference_cfg, deterministic=True)
        dev_loader = streaming_loader(
            dev_dataset, dist.per_rank_batch_size(int(slt_cfg.get("batch_size", 4))), collator, num_workers=num_workers
        )
    pretrained_path = resolve_pretrained(slt_cfg, data_cfg, language, default="checkpoints/openasl_pose_only_slt.pth")
    require_translator_init(pretrained_path, segmentation_branch, language, inference_cfg, slt_cfg.get("annotation_fingerprint"))
    # `pretrained_path` is loaded inside MisalignedSLTModel BEFORE the DLM [MASK]-token extension, so the
    # block-diffusion decoder inherits the released Uni-Sign pose + LM weights (pose always; mT5 also loads the LM).
    model = MisalignedSLTModel(
        front_end=front_end, decoder=decoder or str(slt_cfg.get("decoder", "dlm")), block_size=int(slt_cfg.get("block_size", 16)),
        # Shape MUST match S1 (train/bio_pretrain.py) or `bio_head_init` fails to strict-load — same keys build_bio_s1 reads.
        bio_hidden_dim=int(slt_cfg.get("bio_hidden_dim", 384)), bio_depth=int(slt_cfg.get("bio_depth", 4)),
        bio_nhead=int(slt_cfg.get("bio_nhead", 8)), bio_dropout=float(slt_cfg.get("bio_dropout", 0.1)),
        bio_conv_stem_layers=int(slt_cfg.get("bio_conv_stem_layers", 2)), bio_attention_radius_s=slt_cfg.get("bio_attention_radius_s"),
        pretrained_path=pretrained_path, segmentation_branch=segmentation_branch,
    )
    model.duration_model = duration
    slt_cfg["resolved_inits"] = {"translator": _file_stamp(pretrained_path), "segmentation": None}
    if not segmentation_branch: print(
        "slt | clean floor: no segmentation branch (lambda_bio 0, gate off, no bio_head_init); the released front end trains alone. "
        "S1's segmentation-adapted encoder stays out, so this row remains a faithful Uni-Sign transfer.", flush=True
    )
    else:
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
        if _s1_meta.get("pretrain_pool") != _want: raise SystemExit(
            f"{bio_init} was trained on pool {_s1_meta.get('pretrain_pool')!r}, but bio_pretrain.yaml now expects "
            f"{_want!r}. Retrain S1 for this pool, or point checkpoint.bio_head_init at the matching model."
        )
        # The band is an architecture property a strict state_dict load cannot see (no tensor carries it).
        require_attention_radius(_s1_meta, slt_cfg.get("bio_attention_radius_s"), str(bio_init))
        # Labels are not in any tensor either: an S1 trained before a re-convert or re-measured cuts would load silently.
        require_segmenter_data(_s1_meta, language, slt_cfg["annotation_fingerprint"], inference_cfg["buffer_cap_s"], str(bio_init))
        model.bio_head.load_state_dict(head_sd, strict=True)
        slt_cfg["resolved_inits"]["segmentation"] = _file_stamp(bio_init)
        # S1's pose encoder goes into SEGMENTATION branch; the translator keeps the clean translator's encoder. 
        # Each branch retains its own pretrained weights at initialization.
        pose_sd = {k[len("pose_encoder."):]: v for k, v in sd.items() if k.startswith("pose_encoder.")}
        if not pose_sd: raise ValueError(f"{bio_init} carries no pose encoder; the segmentation branch needs S1's encoder")
        model.bio_pose_encoder.load_state_dict(pose_sd, strict=True)
        print(f"slt | loaded S1 segmentation branch from {bio_init} ({len(pose_sd)} pose-encoder + "
              f"{len(head_sd)} head tensors); the translator keeps its own warm-start encoder", flush=True)

    print(f"slt | model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, all trained at 1 rate", flush=True)
    return SLTComponents(
        model=model, tokenizer=tokenizer, train_loader=train_loader, dev_loader=dev_loader, slt_cfg=slt_cfg,
        checkpoint_meta=_training_meta(slt_cfg, inference_cfg, language),
    )


@torch.no_grad()
def evaluate_slt(model: MisalignedSLTModel, loader: DataLoader, device: torch.device, slt_cfg: dict) -> dict[str, float]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, float]] = []
    spans = CompleteSpanMetrics()

    oput_cfg = slt_cfg.get("oput", {})
    spd_cfg = slt_cfg.get("spd", {})
    gate_cfg = slt_cfg.get("membership_gate", {})

    gate_kwargs = dict(
        gate_enabled=bool(gate_cfg.get("enabled", False)), gate_eps=float(gate_cfg.get("eps", 1e-4)),
        gate_min_span_frames=int(gate_cfg["min_span_frames"]),
    )
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
            oput_label_smoothing=float(oput_cfg.get("label_smoothing", 0.2)),
            gate_delta_frames=int(gate_cfg.get("delta", 12)), gate_detach_omega=bool(gate_cfg.get("detach_omega", False)),
            **gate_kwargs,
        )
        row = {k: float(v.detach().cpu().item()) for k, v in output.logs.items() if v.numel() == 1}
        if float(slt_cfg.get("lambda_bio", 1.0)) != 0.0 and output.bio_logits is not None:
            # Reuse forward_loss's own head forward (identical inputs, eval mode, no_grad): a 2nd segmentation pass here would only
            # recompute it. forward_loss already passes frame_mask, so padded frames never enter conv stem / RoPE as real frames.
            bio_logits = output.bio_logits
            row.update(bio_frame_metrics(bio_logits, batch["bio_labels"], prefix="bio"))
            # Score the visible window with the train-fitted duration prior; the gate separately excludes committed context.
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
                        max_text_tokens=int(slt_cfg.get("max_text_tokens", 320)), tau_dec=float(spd_cfg.get("tau_dec", 0.5)),
                        spd_top_k=int(spd_cfg.get("top_k", 1)), spd_renormalize=bool(spd_cfg.get("renormalize", True)),
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
        # means the decode commits EOS ever earlier (EOS supervision / commit-threshold pressure), which BLEU/CIDEr punish as brevity.
        # WORD tokens, matching BLEU's BP. Characters disagree with it materially, so char ratio reads healthy while BLEU is penalised.
        total_ref = sum(len(r.split()) for r in ref_texts)
        metrics["val_translation_len_ratio"] = float(sum(len(p.split()) for p in pred_texts)) / max(1, total_ref)
    # A dev-window trade-off score, not DVC and not a guarantee of retained localization.
    if "val_phrase_tiou_f1" in metrics and "val_translation_bleu4" in metrics:
        metrics["val_joint_score"] = float(metrics["val_phrase_tiou_f1"]) * float(metrics["val_translation_bleu4"])
    return metrics


def training_loss(model, batch: dict, slt_cfg: dict) -> SLTLossOutput: # The actual AR/DLM training objective.
    oput_cfg = slt_cfg.get("oput", {})
    gate_cfg = slt_cfg.get("membership_gate", {})
    dice_weight = float(slt_cfg.get("dice_loss_weight", 1.5))
    return model.forward_loss(
        batch, lambda_trans=float(slt_cfg.get("lambda_trans", 1.0)), lambda_bio=float(slt_cfg.get("lambda_bio", 1.0)),
        dice_weight=dice_weight, bio_class_weights=bio_class_weight_tensor(slt_cfg.get("bio_class_weights")),
        oput_t_low=float(oput_cfg.get("t_low", 0.3)), oput_t_high=float(oput_cfg.get("t_high", 0.8)),
        oput_label_smoothing=float(oput_cfg.get("label_smoothing", 0.2)),
        gate_enabled=bool(gate_cfg.get("enabled", False)),
        # Same δ as the inference commit gate's delta_enc_frames (configs/inference.yaml): the coverage tolerance of text routing.
        gate_eps=float(gate_cfg.get("eps", 1e-4)), gate_min_span_frames=int(gate_cfg["min_span_frames"]),
        gate_delta_frames=int(gate_cfg.get("delta", 12)), gate_detach_omega=bool(gate_cfg.get("detach_omega", False)),
    )


def train_slt_epochs(
    model: MisalignedSLTModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epochs: int,
    slt_cfg: dict, dev_loader: DataLoader | None = None, resume: bool = False, checkpoint_meta: dict | None = None,
) -> int:
    decoder_name = getattr(model, "decoder_type", "dlm")
    if float(slt_cfg.get("lambda_bio", 1.)) != 0. and dist.is_main():
        print("slt | localization monitor: complete-span F1@0.5, pooled dev-window counts; "
              "RQ2 scores whole videos separately", flush=True)

    def step_fn(batch, epoch: int):
        output = training_loss(model, batch, slt_cfg)
        return output.loss, {k: float(v.detach().cpu().item()) for k, v in output.logs.items() if v.numel() == 1}

    def evaluate_fn(epoch: int): return evaluate_slt(model, dev_loader, device, slt_cfg=slt_cfg)  # epoch 0 scores the init

    return run_epoch_loop(
        name=f"slt-{decoder_name}", model=model, loader=loader, optimizer=optimizer, device=device, epochs=epochs,
        cfg=slt_cfg, step_fn=step_fn, evaluate_fn=evaluate_fn, default_monitor="val_loss", default_mode="min",
        dev_loader=dev_loader, resume=resume, checkpoint_meta=checkpoint_meta,
    )
