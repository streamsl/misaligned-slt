from __future__ import annotations
import copy, random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

from data.batch import WindowCollator
from data.windowing import join_runs, joined_target
from data.loader import ANNOTATION_PROTOCOL, StreamingWindowDataset, annotation_fingerprint, load_language_records, streaming_loader 
from models.unisign import UniSignMT5FrontEnd, PROMPT_LANG_BY_TARGET
from models.streaming_slt import TEXT_ROUTING, MisalignedSLTModel, SLTLossOutput
from models.checkpointing import load_checkpoint_meta, require_fixed_constants
from infer.duration_decode import DurationModel

from train import distributed as dist
from train.bio_pretrain import require_attention_radius, require_segmenter_data
from train.losses import bio_class_weight_tensor, resolve_bio_class_weights
from train.helpers import mean_logs, move_to_device, run_epoch_loop
from metrics import char_level_for_target, bio_frame_metrics, compute_text_metrics
from utils import (
    UNIT_SEPARATOR, cfg_get, checkpoint_dir, lambda_min_s, load_yaml, 
    pick_device, pool_key, resolve_pretrained, seconds_to_frames
)

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
    inference_cfg: dict | None = None
    monitor_records: list | None = None

def _file_stamp(path) -> dict | None:   # Which file initialized a branch, with its size: a path alone doesn't show a later overwrite.
    if not path: return None
    f = Path(str(path))
    return {"path": str(path), "bytes": f.stat().st_size if f.exists() else None}

def _resolved_inits(translator, translator_meta: dict, segmentation, segmentation_meta: dict) -> dict:
    """The files that initialized the translator and the segmentation branch, and, under `annotation_signature`, the train labels each
    was trained on (its meta's annotation_fingerprint; S1's per-language language_fingerprints). As with the duration prior's signature,
    a --resume compares the files and not these labels (models.checkpointing._without_annotation); eval warns when they are not current 
    train labels (evaluation.systems.model_annotation). None where an init carries no stamp (the released Uni-Sign file)."""
    return {"translator": _file_stamp(translator), "segmentation": _file_stamp(segmentation), "annotation_signature": {
        "translator": translator_meta.get("annotation_fingerprint"), "segmentation": segmentation_meta.get("annotation_fingerprint"),
        "segmentation_languages": segmentation_meta.get("language_fingerprints"),
    }}

def _inject_gate_geometry(slt_cfg: dict, inference_cfg: dict) -> None:
    # Match the sampler and FSM's first eligible target. Short units remain legal paths;
    # they are skipped by this target-selection rule, not removed from the path distribution.
    gate = slt_cfg.setdefault("membership_gate", {})
    # Every reader below uses .get() defaults, so a removed or misspelled key would silently train a different objective.
    unknown = set(gate) - GATE_CONFIG_KEYS
    if unknown: raise ValueError(f"membership_gate: unknown key(s) {sorted(unknown)}; accepted: {sorted(GATE_CONFIG_KEYS)}")
    unknown = set(slt_cfg.get("spd", {}) or {}) - SPD_CONFIG_KEYS  # same reason: a misspelled tau_dec would decode at default
    if unknown: raise ValueError(f"spd: unknown key(s) {sorted(unknown)}; accepted: {sorted(SPD_CONFIG_KEYS)}")
    gate["delta_s"] = float(inference_cfg["boundary_stability"]["delta_s"])
    gate["min_span_s"] = lambda_min_s(inference_cfg)

def _gate_frames(gate_cfg: dict, batch: dict) -> dict:
    # Λ_min and δ in frames of each row's own pose grid: rows of 1 batch may have different frame rates.
    return {"gate_min_span_frames": [seconds_to_frames(gate_cfg["min_span_s"], fps) for fps in batch["fps"]],
            "gate_delta_frames": [seconds_to_frames(gate_cfg["delta_s"], fps) for fps in batch["fps"]]}

def online_monitor_records(dev_records: list, hours: float, seed: int) -> tuple[list, float]:
    """The online monitor's streams (checkpoint.monitor val_online_score): whole dev source videos (all their streams) in a seeded 
    random order, until pose time reaches `hours` (last video may pass it). The hours cap the cost of 1 evaluation on every corpus; 
    the random order keeps the subset unbiased (the first ids by sort are not a sample) and fixed across epochs and launches."""
    by_video: dict[str, list] = {}
    for record in dev_records: by_video.setdefault(record.pose.video_id, []).append(record)
    order = sorted(by_video)
    random.Random(int(seed)).shuffle(order)
    records, total_s = [], 0.0
    for vid in order:
        if total_s >= hours * 3600: break
        records += by_video[vid]; total_s += sum(r.pose.duration_s for r in by_video[vid])
    return records, total_s


def _training_meta(slt_cfg: dict, inference_cfg: dict, language: str) -> dict:
    """The config this stage-2 run is parameterized by, travelling with the weights.

    δ, Λ_min and buffer_cap_s come from inference.yaml and the duration prior from the train labels. Resuming across a change 
    of any of them trains 2 halves under different objectives, and without this record nothing in the artifacts shows it.
    """
    gate_cfg = slt_cfg.get("membership_gate", {}) or {}
    return {
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": slt_cfg.get("annotation_fingerprint"),
        "language": str(language), "decoder": str(slt_cfg.get("decoder", "dlm")),
        # two_stream_slt: separate segmentation/translation pose encoders, Ω on encoder keys and cross-attention; one_stream_slt:
        # one shared pose encoder (shared_pose_encoder). Eval builds the layout the tag names and refuses any other tag.
        "architecture": ("one_stream_slt" if slt_cfg.get("shared_pose_encoder") else "two_stream_slt")
                        if float(slt_cfg.get("lambda_bio", 1.0)) else "clean_translation",
        "bio_objective": "ce_dice", "dice_loss_weight": float(slt_cfg.get("dice_loss_weight", 1.5)),
        "gate": {k: gate_cfg.get(k) for k in ("enabled", "delta_s", "min_span_s", "eps", "detach_omega")},
        # Text routing: GT-anchored targets (one unit, or a run of units with gaps <= δ joined by UNIT_SEPARATOR) on 
        # a detached hard crop of the predicted span when it holds the target within δ, else on the δ-jittered GT span.
        "text_routing": TEXT_ROUTING, "unit_separator": UNIT_SEPARATOR, "bio_adapter": bool(slt_cfg.get("bio_adapter", False)), 
        # The checkpoint monitor; online_dev_hours only for the online monitor (the offline monitor reads the whole dev set).
        "monitor": slt_cfg.get("checkpoint", {}).get("monitor"), "seed": slt_cfg.get("seed"),
        "online_dev_hours": slt_cfg.get("checkpoint", {}).get("online_dev_hours")
                            if str(slt_cfg.get("checkpoint", {}).get("monitor", "")).startswith("val_online_") else None,
        # Which files initialized the translator and the segmentation branch, 
        # with the train labels of each (_resolved_inits), and the head's attention band.
        "resolved_inits": slt_cfg.get("resolved_inits"), "bio_attention_radius_s": slt_cfg.get("bio_attention_radius_s"),
        "early_stopping": slt_cfg.get("early_stopping"), "scheduler": slt_cfg.get("scheduler"), "epochs": slt_cfg.get("epochs"),
        "buffer_cap_s": inference_cfg.get("buffer_cap_s"), "duration_model": slt_cfg.get("duration_model"),
        "segmentation_decode": "semi_markov_viterbi" if slt_cfg.get("duration_model") else "none",
        "oput": slt_cfg.get("oput", {}), "spd": slt_cfg.get("spd", {}), 
        # Training geometry of the text canvas: the block-causal mask and the OPUT corruption read block_size, so eval refuses 
        # a decoder built at another block (evaluation/systems.py _build_eval_model), and the decode canvas must not shrink.
        "block_size": int(slt_cfg.get("block_size", 16)), "max_text_tokens": slt_cfg.get("max_text_tokens"),
        "batch_size": slt_cfg.get("batch_size"), "learning_rate": float(slt_cfg["learning_rate"]),
        "mode_ratios": slt_cfg.get("mode_ratios"), "jitter": slt_cfg.get("jitter"), "augmentation": slt_cfg.get("augmentation"),
        "bio_class_weights": slt_cfg.get("bio_class_weights"),  # resolved list, not the "balanced" string
        "lambda_bio": float(slt_cfg.get("lambda_bio", 1.0)), "lambda_trans": float(slt_cfg.get("lambda_trans", 1.0)),
    }


def assert_targets_fit(records, tokenizer, max_text_tokens: int, buffer_cap_s: float, delta_s: float, split: str) -> None:
    """Refuse to start if any text target the sampler can build exceeds the text canvas.

    The collator never truncates a complete-caption target (truncated reference silently rewrites the task), so an over-long target
    would crash mid-epoch instead. A target is a unit that fits the streaming buffer (data/sampler.py `_clip_window`), or a run of
    units with gaps of at most δ that fits it together (data.windowing.join_runs, joined by UNIT_SEPARATOR); this mirrors those
    predicates from every unit and reports the capacity to configure, as buffer_cap_s is sized to the data.
    """
    longest, culprit = 0, None
    # A complete unit's terminator frame lies inside the window, and _clip_window caps the window at buffer_cap_s, so a run that
    # spans more than buffer_cap_s can never be a target: buffer_cap_s is an upper bound that covers every reachable run.
    cap_s = float(buffer_cap_s)
    for record in records:
        units = sorted((u for u in record.sentences if getattr(u, "reliable", True)), key=lambda u: (u.start_s, u.end_s))
        gap_s = seconds_to_frames(delta_s, record.pose.fps) / record.pose.fps
        for i, unit in enumerate(units):
            if unit.duration_s > cap_s: continue
            target = joined_target(join_runs(units[i:], record.sentences, gap_s, max_span_s=cap_s)[0])
            n = len(tokenizer(target.text)["input_ids"])
            if n > longest: longest, culprit = n, target
    # +1: decode canvas holds BOS in slot 0, so a target of exactly max_text_tokens could be trained on but never emitted.
    if longest + 1 > int(max_text_tokens): raise ValueError(
        f"{split}: a complete caption target needs {longest} tokens plus the BOS slot but max_text_tokens is {max_text_tokens} "
        f"({culprit.video_id} {culprit.start_s:.1f}-{culprit.end_s:.1f}s). Set max_text_tokens >= {longest + 1}: the collator "
        f"refuses to truncate a complete target, and would raise on this target the first time it is sampled."
    )


def require_translator_init(pretrained_path, segmentation_branch: bool, language: str, inference_cfg: dict, fingerprint) -> dict:
    """Translator warm start of a train-slt run. Clean floor starts from the released Uni-Sign weights. An arm starts from THIS 
    language's clean floor, trained under fixed constants (other annotations only warn). Returns the warm start's meta."""
    # A trained warm start carries meta; the released Uni-Sign file carries none.
    init_meta = load_checkpoint_meta(pretrained_path) if Path(str(pretrained_path)).exists() else {}
    if segmentation_branch and not init_meta: raise SystemExit(
        f"Stage 2 must warm-start the translator from this language's trained clean floor, but {pretrained_path} "
        f"carries no training meta (the released file, or no file). Train the clean baseline, then set data.yaml "
        f"languages.{language}.pretrained_slt: checkpoints/baseline_train/{language}/model.pt."
    )
    # After the B2b re-root, data.yaml pretrained_slt names the previous clean floor: warm-starting the floor from it would
    # silently fine-tune the old floor a second time.
    if not segmentation_branch and init_meta: raise SystemExit(
        f"The clean floor must warm-start from the released Uni-Sign weights, but {pretrained_path} is a trained checkpoint "
        f"(architecture={init_meta.get('architecture')!r}). Set checkpoint.from_pretrained to the released file "
        f"(baseline_train.yaml does), or point data.yaml languages.{language}.pretrained_slt back at it."
    )
    if segmentation_branch and init_meta:
        found = {"architecture": init_meta.get("architecture"), "language": init_meta.get("language")}
        want = {"architecture": "clean_translation", "language": str(language)}
        bad = [f"{k}={found[k]!r} (need {want[k]!r})" for k in want if found[k] != want[k]]
        if bad: raise SystemExit(
            f"Stage 2 warm-starts the translator from {pretrained_path}, but it is not this language's clean translator: "
            f"{'; '.join(bad)}. Train the clean baseline (README B2), then re-root it (B2b)."
        )
        if init_meta.get("annotation_fingerprint") != fingerprint: print(
            f"WARNING: the translator warm start {pretrained_path} trained on other {language} "
            f"annotations ({init_meta.get('annotation_fingerprint')!r}, now {fingerprint!r}). "
            f"Retrain the clean baseline before a run you report.", flush=True
        )
        require_fixed_constants(init_meta, inference_cfg, pretrained_path)  # same rule eval applies to it
    return init_meta


@torch.no_grad()
def fold_s1_stitch(model: MisalignedSLTModel, s1_pose_sd: dict, batches, ridge: float = 1e-6) -> dict:
    """One-stream arm: pooled S1's head reads clean translator's encoder E through the affine least-squares map from E's tokens
    to S1's encoder tokens, folded exactly into `bio_head.input_proj` (W <- W M, b <- W m + b; no new parameter). S1's encoder 
    is only regression target on unlabeled train windows. It's not kept, and the head and E then train on both losses. Ridge
    1e-6·tr(XᵀX)/d on the weights, none on the offset. Returns frame count and the in-sample R² (pooled over dimensions)."""
    student = model.front_end.pose_encoder
    teacher = copy.deepcopy(student)
    teacher.load_state_dict(s1_pose_sd, strict=True)

    home, mode, device = next(student.parameters()).device, student.training, pick_device()
    student.to(device).eval(); teacher.to(device).eval()
    d = int(model.front_end.bio_tap_dim)
    gram, cross = torch.zeros(d + 1, d + 1, dtype=torch.float64), torch.zeros(d + 1, d, dtype=torch.float64)
    y_sum, y_sq, n = torch.zeros(d, dtype=torch.float64), 0.0, 0

    for batch in batches:
        poses, mask = batch["poses"].to(device), batch["frame_mask"].to(device)
        x, y = student(poses, mask)[mask].float().cpu().double(), teacher(poses, mask)[mask].float().cpu().double()
        x = torch.cat([x, x.new_ones(len(x), 1)], 1)
        gram += x.T @ x; cross += x.T @ y; y_sum += y.sum(0); y_sq += float((y * y).sum()); n += len(y)

    shrink = torch.zeros(d + 1, dtype=torch.float64)
    shrink[:d] = ridge * gram.diagonal()[:d].sum() / d
    w = torch.linalg.solve(gram + torch.diag(shrink), cross)   # S1 tokens ≈ [E tokens, 1] @ w
    rss = y_sq - 2 * float((w * cross).sum()) + float((w * (gram @ w)).sum())
    proj = model.bio_head.input_proj
    if not isinstance(proj, torch.nn.Linear): raise ValueError("the stitch folds into a Linear bio_head.input_proj")

    a = proj.weight.double()
    proj.bias.copy_((a @ w[d] + proj.bias.double()).to(proj.bias.dtype))
    proj.weight.copy_((a @ w[:d].T).to(proj.weight.dtype))
    student.to(home).train(mode)
    return {"frames": n, "r2": 1.0 - rss / (y_sq - float((y_sum * y_sum).sum()) / max(n, 1))}


def build_slt_components(
    data_config: str = "configs/data.yaml", slt_config: str = "configs/dlm.yaml", inference_config: str = "configs/inference.yaml",
    include_dev: bool = False, language: str | None = None, bio_config: str = BIO_S1_CONFIG,
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
    
    slt_cfg["decoder"] = str(slt_cfg.get("decoder", "dlm"))
    train_records, _ = load_language_records(data_cfg, language, split="train")
    slt_cfg["annotation_fingerprint"] = annotation_fingerprint(train_records)
    # Label-only duration prior of this language's train units; stamped in the checkpoint, so eval decodes with it.
    duration = DurationModel.fit(train_records) if float(slt_cfg.get("lambda_bio", 1.)) else None
    slt_cfg["duration_model"] = duration.to_dict() if duration else None

    target_lang = data_cfg["languages"][language].get("target_lang", "en_XX")
    slt_cfg["target_lang"] = target_lang  # metric scoring level is declared, not sniffed (metrics.char_level_for_target)
    # Uni-Sign front end. language_model.name picks the mT5 checkpoint and its tokenizer.
    lm_name = str(cfg_get(slt_cfg, "language_model", "name", default="google/mt5-base"))
    prompt_lang = PROMPT_LANG_BY_TARGET.get(str(target_lang or ""), "English")
    tokenizer = T5Tokenizer.from_pretrained(lm_name, legacy=False)
    front_end = UniSignMT5FrontEnd(mt5_name=lm_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)

    resolve_bio_class_weights(slt_cfg, train_records)
    train_dataset = StreamingWindowDataset(train_records, slt_cfg=slt_cfg, inference_cfg=inference_cfg)
    collator = WindowCollator(
        tokenizer, max_text_tokens=int(slt_cfg["max_text_tokens"]),
        # The text canvas is sized to the batch, not to max_text_tokens: captions are ~15 tokens, far below that canvas, 
        # and every decoder forward runs the whole width. The collator keeps the canvas block-aligned.
        block_size=int(slt_cfg.get("block_size", 16)),
    )
    assert_targets_fit(train_records, tokenizer, collator.max_text_tokens, inference_cfg["buffer_cap_s"],
                       float(inference_cfg["boundary_stability"]["delta_s"]), f"{language}/train")
    # num_workers is pure throughput: anchors are index-driven (each realized once per epoch regardless of worker split) 
    # and every draw reseeds its rng from the index (WindowSampler.spec_for), so no worker state matters.
    num_workers = int(slt_cfg.get("num_workers", 0))
    train_loader = streaming_loader(
        train_dataset, dist.per_rank_batch_size(int(slt_cfg.get("batch_size", 4))), collator, num_workers=num_workers,
        # No length bucketing (S1 and the Moryossef segmenter keep it): the global grad clip (max_grad_norm) gives every batch 
        # a full step, so a batch of short windows would weigh as much as a batch of long ones and over-weight short windows, 
        # B frames near window edges and text-free batches. Random batches keep each window near its frame share. The clean 
        # floor and the gated arms both build their train loader here. bucket_seed seeds the per-epoch shuffle.
        bucket_by_length=False, bucket_seed=int(slt_cfg.get("seed", 42)),
    )
    dev_loader, monitor_records = None, None
    if include_dev:
        dev_records, _ = load_language_records(data_cfg, language, split="dev")
        assert_targets_fit(dev_records, tokenizer, collator.max_text_tokens, inference_cfg["buffer_cap_s"],
                           float(inference_cfg["boundary_stability"]["delta_s"]), f"{language}/dev")
        # Dev scores the same experimental unit as standard SLT training: 1 window per reliable sentence anchor 
        # (the dataset's length), not 1 per video. deterministic: the same dev windows every epoch.
        dev_dataset = StreamingWindowDataset(dev_records, slt_cfg=slt_cfg, inference_cfg=inference_cfg, deterministic=True)
        dev_loader = streaming_loader(  # checkpoint.dev_batch_size: the global batch of every dev evaluation
            dev_dataset, dist.per_rank_batch_size(int(slt_cfg["checkpoint"]["dev_batch_size"])), 
            collator, num_workers=num_workers
        )
        hours = float(slt_cfg.get("checkpoint", {}).get("online_dev_hours") or 0.0)
        monitor = str(slt_cfg.get("checkpoint", {}).get("monitor", ""))
        if segmentation_branch and monitor.startswith("val_offline_"):
            monitor_records, total_s = dev_records, sum(r.pose.duration_s for r in dev_records)
        elif segmentation_branch and monitor.startswith("val_online_") and hours > 0:
            monitor_records, total_s = online_monitor_records(dev_records, hours, int(slt_cfg.get("seed", 42)))
        if monitor_records:
            slt_cfg["monitor_tiou_thresholds"] = [
                float(t) for t in load_yaml("configs/eval.yaml").get("rq2", {}).get("tiou_thresholds", [.5, .7, .9])
            ]
            if dist.is_main(): print(f"slt | dev monitor: {len({r.pose.video_id for r in monitor_records})} "
                                     f"dev videos, {len(monitor_records)} streams, {total_s / 3600:.2f} h", flush=True)
    pretrained_path = resolve_pretrained(slt_cfg, data_cfg, language, default="checkpoints/openasl_pose_only_slt.pth")
    init_meta = require_translator_init(
        pretrained_path, segmentation_branch, language, inference_cfg, slt_cfg.get("annotation_fingerprint")
    )
    # `pretrained_path` (the clean floor for an arm, the released Uni-Sign file for the floor itself) is loaded inside
    # MisalignedSLTModel BEFORE the DLM [MASK]-token extension, so the block-diffusion decoder inherits its pose + LM weights.
    model = MisalignedSLTModel(
        front_end=front_end, decoder=slt_cfg["decoder"], block_size=int(slt_cfg.get("block_size", 16)),
        # Shape MUST match S1 (train/bio_pretrain.py) or `bio_head_init` fails to strict-load — same keys build_bio_s1 reads.
        bio_hidden_dim=int(slt_cfg.get("bio_hidden_dim", 384)), bio_depth=int(slt_cfg.get("bio_depth", 4)),
        bio_nhead=int(slt_cfg.get("bio_nhead", 8)), bio_dropout=float(slt_cfg.get("bio_dropout", 0.1)),
        bio_conv_stem_layers=int(slt_cfg.get("bio_conv_stem_layers", 2)), 
        bio_attention_radius_s=slt_cfg.get("bio_attention_radius_s"),
        pretrained_path=pretrained_path, segmentation_branch=segmentation_branch, 
        bio_adapter=bool(slt_cfg.get("bio_adapter", False)),
        shared_pose_encoder=bool(slt_cfg.get("shared_pose_encoder", False)),
    )
    model.duration_model = duration
    slt_cfg["resolved_inits"] = _resolved_inits(pretrained_path, init_meta, None, {})
    if not segmentation_branch: print(
        "slt | clean floor: no segmentation branch (lambda_bio 0, gate off, no bio_head_init). The released front end trains alone. "
        "S1's segmentation-adapted encoder stays out, so this row remains a faithful Uni-Sign transfer.", flush=True
    )
    else:
        blob = torch.load(str(bio_init), map_location="cpu")
        sd = blob.get("model", blob) if isinstance(blob, dict) else blob
        head_sd = {k[len("bio_head."):]: v for k, v in sd.items() if k.startswith("bio_head.")}
        # S1's head reads feat_dim; this bio_head reads front_end.bio_tap_dim (mT5 d_model, 768). 
        # Must match for "S1 features == S2 initial features" and the strict-load.
        s1_dim = head_sd.get("input_proj.weight")
        if s1_dim is not None and int(s1_dim.shape[1]) != int(front_end.bio_tap_dim): raise ValueError(
            f"bio_head_init dim mismatch: S1 head reads {int(s1_dim.shape[1])}-d features but this SLT model's "
            f"bio_tap is {int(front_end.bio_tap_dim)}-d ({lm_name}). Retrain train-bio with feat_dim="
            f"{int(front_end.bio_tap_dim)} (bio_pretrain.yaml)."
        )
        _s1_meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
        # PROVENANCE, same rule as evaluation/systems.py _load_segmenter: the checkpoint records which pool trained it, 
        # and stage 2 must not warm-start the gate from a head of another pool (another model, a silent failure).
        if _s1_meta.get("pretrain_pool") != pool_key(bio_cfg): raise SystemExit(
            f"{bio_init} was trained on pool {_s1_meta.get('pretrain_pool')!r}, but bio_pretrain.yaml expects "
            f"{pool_key(bio_cfg)!r}. Retrain S1 for this pool, or point checkpoint.bio_head_init at the matching model."
        )
        # The band is an architecture property a strict state_dict load cannot see (no tensor carries it).
        require_attention_radius(_s1_meta, slt_cfg.get("bio_attention_radius_s"), str(bio_init))
        # Labels are not in any tensor either: an S1 trained before a re-convert or re-measured cuts would load silently.
        require_segmenter_data(_s1_meta, language, slt_cfg["annotation_fingerprint"], inference_cfg["buffer_cap_s"], str(bio_init))
        model.bio_head.load_state_dict(head_sd, strict=True)
        slt_cfg["resolved_inits"] = _resolved_inits(pretrained_path, init_meta, bio_init, _s1_meta)
        # Two streams: S1's pose encoder goes into the SEGMENTATION branch; the translator keeps the clean translator's encoder.
        # One-stream arm (shared_pose_encoder): ONE encoder, the clean translator's (the semantic stream owns it, as in TwInS); 
        # S1's head reads it through a least-squares map to S1's encoder tokens, folded into the head (fold_s1_stitch). Loading 
        # S1's encoder here instead would start the translator near 0 BLEU (docs/implementation_notes.md).
        pose_sd = {k[len("pose_encoder."):]: v for k, v in sd.items() if k.startswith("pose_encoder.")}
        if not pose_sd: raise ValueError(f"{bio_init} carries no pose encoder; the segmentation branch needs S1's encoder")
        if model.bio_pose_encoder is not None:
            model.bio_pose_encoder.load_state_dict(pose_sd, strict=True)
            print(f"slt | loaded S1 from {bio_init}: {len(head_sd)} head tensors and {len(pose_sd)} pose-encoder "
                  f"tensors into the segmentation branch; the translator keeps its own encoder", flush=True)
        else:
            picks = torch.linspace(0, len(train_dataset) - 1, 400).long().unique().tolist()
            fit = fold_s1_stitch(model, pose_sd, (collator(
                [train_dataset[i] for i in picks[k:k + 16]]) for k in range(0, len(picks), 16)
            ))
            print(f"slt | one stream: clean translator's encoder is shared. S1's head ({len(head_sd)} tensors, {bio_init}) reads "
                  f"it via a least-squares map to S1's encoder tokens folded into bio_head.input_proj ({fit['frames']} train "
                  f"frames, R^2 {fit['r2']:.3f}); S1's encoder is not kept", flush=True)

    print(f"slt | model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, all trained at 1 rate", flush=True)
    return SLTComponents(
        model=model, tokenizer=tokenizer, train_loader=train_loader, dev_loader=dev_loader, 
        slt_cfg=slt_cfg, inference_cfg=inference_cfg, monitor_records=monitor_records, 
        checkpoint_meta=_training_meta(slt_cfg, inference_cfg, language)
    )


@torch.no_grad()
def evaluate_slt(
    model: MisalignedSLTModel, loader: DataLoader, device: torch.device, 
    slt_cfg: dict, inference_cfg: dict | None = None, monitor_records: list | None = None,
) -> dict[str, float]:
    """Every run logs its dev-window losses (val_loss and its parts). Joint training logs RQ2 scores of whole dev videos for the 
    protocol that checkpoint.monitor names: val_offline_* for offline row (evaluation.rq2_offline.offline_dev_scores), val_online_* 
    for online row, the streaming FSM (evaluation.rq2_online.online_dev_scores). The monitor is F1 avg x DVC BLEU-4 avg, logged 
    with both parts, their video-bootstrap spread and events per gold unit."""
    was_training = model.training
    model.eval()
    rows: list[dict[str, float]] = []
    arm = float(slt_cfg.get("lambda_bio", 1.)) != 0.

    oput_cfg = slt_cfg.get("oput", {})
    spd_cfg = slt_cfg.get("spd", {})
    gate_cfg = slt_cfg.get("membership_gate", {})

    gate_kwargs = dict(gate_enabled=bool(gate_cfg.get("enabled", False)), gate_eps=float(gate_cfg.get("eps", 1e-4)))
    dice_weight = float(slt_cfg.get("dice_loss_weight", 1.5))
    pred_texts: list[str] = []
    ref_texts: list[str] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        frames = _gate_frames(gate_cfg, batch)
        output: SLTLossOutput = model.forward_loss(
            batch, lambda_trans=float(slt_cfg.get("lambda_trans", 1.0)), lambda_bio=float(slt_cfg.get("lambda_bio", 1.0)), 
            dice_weight=dice_weight, bio_class_weights=bio_class_weight_tensor(slt_cfg.get("bio_class_weights")),
            oput_t_low=float(oput_cfg.get("t_low", 0.3)), oput_t_high=float(oput_cfg.get("t_high", 0.8)),
            oput_label_smoothing=float(oput_cfg.get("label_smoothing", 0.2)),
            **gate_kwargs, **frames,
        )
        row = {k: float(v.detach().cpu().item()) for k, v in output.logs.items() if v.numel() == 1}
        # Frame metrics of forward_loss's own head pass feed mean_logs' B-collapse alarm only. They are not logged.
        if arm and output.bio_logits is not None: 
            row.update(bio_frame_metrics(output.bio_logits, batch["bio_labels"], prefix="bio"))
        rows.append(row)
        if arm: continue  # the arm's translation is scored on the deployed path below, not on dev windows

        # The clean floor translates every supervised dev window: dev BLEU selects its checkpoint.
        supervised = batch.get("translation_supervised")
        targets = batch.get("translation_targets", [])
        if isinstance(supervised, torch.Tensor) and supervised.any():
            idx = supervised.nonzero(as_tuple=False).flatten()
            _, tokens, _, _ = model.generate_from_poses(
                poses=batch["poses"][idx], frame_mask=batch["frame_mask"][idx],
                timestamps_s=batch.get("timestamps_s", None)[idx] if batch.get("timestamps_s") is not None else None,
                max_text_tokens=int(slt_cfg["max_text_tokens"]), tau_dec=float(spd_cfg.get("tau_dec", 0.5)),
                spd_top_k=int(spd_cfg.get("top_k", 1)), spd_renormalize=bool(spd_cfg.get("renormalize", True)),
                gate_min_span_frames=[frames["gate_min_span_frames"][i] for i in idx.tolist()], **gate_kwargs,
            )
            # Unit separator is scored as a space, as RQ1 and RQ2 score it: sacrebleu would split literal sentinel into 7 tokens.
            unjoin = lambda t: " ".join(str(t).replace(UNIT_SEPARATOR, " ").split())
            pred_texts.extend(unjoin(t) for t in model.tokenizer.batch_decode(tokens.detach().cpu(), skip_special_tokens=True))
            for item_idx in idx.detach().cpu().tolist():
                target = targets[int(item_idx)]
                ref_texts.append(unjoin(target.get("text", "") if isinstance(target, dict) else getattr(target, "text", "")))

    metrics = {k: v for k, v in mean_logs(rows, prefix="val").items() if k.endswith("loss")}
    if arm and monitor_records:  # whole dev videos, scored as RQ2: offline row (val_offline_*) or online row (val_online_*)
        if str(slt_cfg.get("checkpoint", {}).get("monitor", "")).startswith("val_offline_"):
            from evaluation.rq2_offline import offline_dev_scores as dev_scores
        else: from evaluation.rq2_online import online_dev_scores as dev_scores
        char_level = char_level_for_target(slt_cfg.get("target_lang"))
        scores = dev_scores(model, monitor_records, inference_cfg, slt_cfg, slt_cfg["monitor_tiou_thresholds"], char_level)
        metrics.update({f"val_{k}": v for k, v in scores.items()})

    if was_training: model.train()
    if pred_texts:
        text = compute_text_metrics(
            pred_texts, ref_texts, prefix="val_translation", bleurt_checkpoint=None, 
            char_level=char_level_for_target(slt_cfg.get("target_lang"))
        )
        metrics.update({k: v for k, v in text.items() if k == "val_translation_bleu4"})
        # Hyp/ref length ratio in WORD tokens, matching BLEU's brevity penalty: below 1 and falling 
        # across epochs means the decode ends ever earlier, which BLEU punishes as brevity.
        total_ref = sum(len(r.split()) for r in ref_texts)
        metrics["val_translation_len_ratio"] = float(sum(len(p.split()) for p in pred_texts)) / max(1, total_ref)
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
        # Same δ as the inference commit gate's delta_s (configs/inference.yaml): the coverage tolerance of text routing.
        gate_eps=float(gate_cfg.get("eps", 1e-4)), gate_detach_omega=bool(gate_cfg.get("detach_omega", False)), 
        **_gate_frames(gate_cfg, batch),
    )


def train_slt_epochs(
    model: MisalignedSLTModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, 
    epochs: int, slt_cfg: dict, inference_cfg: dict | None = None, dev_loader: DataLoader | None = None, 
    resume: bool = False, monitor_records: list | None = None, checkpoint_meta: dict | None = None,
) -> int:
    decoder_name = getattr(model, "decoder_type", "dlm")
    if float(slt_cfg.get("lambda_bio", 1.)) != 0. and dist.is_main():
        offline = str(slt_cfg.get("checkpoint", {}).get("monitor", "")).startswith("val_offline_")
        name, row = ("offline", "--offline row on whole dev set: each video segmented once, each span translated") \
                    if offline else ("online", "--online row, streaming FSM, on whole dev videos up to checkpoint.online_dev_hours")
        print(f"slt | checkpoint monitor: val_{name}_score = val_{name}_f1 x val_{name}_dvc_bleu4 ({row}, RQ2 protocol); "
              f"read an epoch-to-epoch change against val_{name}_score_sd", flush=True)

    def step_fn(batch, epoch: int):
        output = training_loss(model, batch, slt_cfg)
        return output.loss, {k: float(v.detach().cpu().item()) for k, v in output.logs.items() if v.numel() == 1}

    def evaluate_fn(epoch: int):
        return evaluate_slt(model, dev_loader, device, slt_cfg=slt_cfg, inference_cfg=inference_cfg, monitor_records=monitor_records)

    return run_epoch_loop(
        name=f"slt-{decoder_name}", model=model, loader=loader, optimizer=optimizer, device=device, epochs=epochs,
        cfg=slt_cfg, step_fn=step_fn, evaluate_fn=evaluate_fn, default_monitor="val_loss", default_mode="min",
        dev_loader=dev_loader, resume=resume, checkpoint_meta=checkpoint_meta,
    )
