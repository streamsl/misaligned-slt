"""The systems that evaluation runs, built from the run's arguments and configs.

The translator model of a row (method config, build, checkpoint checks, train-label provenance, the translator token in artifact names),
the segmenter (S1 or the Moryossef release), the streaming runner, and how a row reads and decodes a window (crop, pose normalization,
decode settings, the batched translate call).
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np

import torch
from transformers import T5Tokenizer
from data.batch import repeat_last_frame
from data.loader import _cached_language_records, annotation_fingerprint
from poses import load_pose_window, normalize_keypoints_unisign

from infer.duration_decode import DurationModel
from infer.stream import StreamingSLTRunner
from models.streaming_slt import MisalignedSLTModel, TEXT_ROUTING
from models.unisign import PROMPT_LANG_BY_TARGET, UniSignMT5FrontEnd
from models.checkpointing import _load_state, load_checkpoint_meta, load_model_checkpoint, require_fixed_constants, s1_layout_state
from utils import cfg_get, checkpoint_dir, lambda_min_s, load_yaml, pick_device, pool_key, resolve_pretrained, seconds_to_frames

METHOD_CONFIGS = {"baseline": "configs/baseline_eval.yaml", "ar": "configs/ar.yaml", "dlm": "configs/dlm.yaml"}

def model_annotation(model, data_cfg: dict, language: str) -> dict | None:
    """The train labels behind a built translator (an arm, or a trained clean floor; `_build_eval_model` keeps its meta) and behind
    the inits it stamps, beside the current `language` train labels: {train, model, translator_init, segmentation_init}, for the
    provenance. Warns, never refuses, when one differs. None for a file with no training meta (the released Uni-Sign weights)."""
    meta = getattr(model, "checkpoint_meta", None)
    if not meta: return None
    inits = (meta.get("resolved_inits") or {}).get("annotation_signature") or {}   # train/slt.py _resolved_inits
    found = {
        "model": meta.get("annotation_fingerprint"),
        "train": annotation_fingerprint(_cached_language_records(data_cfg, language, "train")), 
        # S1 trains on a pool: its own fingerprint hashes every pooled language, so the per-language stamp is the one to compare.
        "translator_init": inits.get("translator"), "segmentation_init": (inits.get("segmentation_languages") or {}).get(language),
    }
    warn_model_labels(found, getattr(model, "checkpoint_path", None))
    return found

def warn_model_labels(found: dict | None, source) -> bool:
    names = {"model": "the model", "translator_init": "its translator warm start", "segmentation_init": "its S1 init"}
    stale = [name for key, name in names.items() if (found or {}).get(key) not in (None, (found or {}).get("train"))]
    if stale: print(
        f"WARNING: {source}: {', '.join(stale)} trained on other train labels than {found.get('train')}, the train labels "
        f"at eval time. Its spans or text can follow the old labels: retrain before a run you report.", flush=True
    )
    return not stale

def _parse_grid(value: str | None, fallback: list[float]) -> list[float]:
    if not value: return [float(x) for x in fallback]
    return [float(x.strip()) for x in value.split(",") if x.strip()]

def _method_config_path(args: argparse.Namespace) -> str:
    return str(args.method_config) if args.method_config else METHOD_CONFIGS[args.method]

def _build_eval_model(
    method: str, checkpoint: str | None, language: str, data_cfg: dict, 
    method_cfg: dict, inference_cfg: dict, device: torch.device
):
    """Uni-Sign pose-only model. LM (`language_model.name`) and checkpoint (`checkpoint.from_pretrained` / `checkpoint.dir`) both come from  
    METHOD config — no separate stage-1 config. Plain scalars so analyze/visualize need no fake Namespace. The file actually loaded (for the
    baseline: --checkpoint or data.yaml pretrained_slt) passes fixed-constant check here, so every row that builds a model is covered once.
    The model keeps that file's meta; RQ1, the RQ2 rows and eval.py --resplit compare its train labels (model_annotation)."""
    if method != "baseline":
        name = checkpoint or checkpoint_dir(method_cfg, default="")
        if not name or not Path(name).exists(): raise FileNotFoundError(f"Missing checkpoint for {method}: {name!r}")
        ckpt = Path(name)
        meta = load_checkpoint_meta(ckpt)
        # one_stream_slt is the shared-pose-encoder ablation; the model is built in the layout the tag names.
        if meta.get("architecture") not in ("two_stream_slt", "one_stream_slt"): raise ValueError(
            f"{ckpt}: expected architecture two_stream_slt or one_stream_slt, found {meta.get('architecture')!r}"
        )
        expected = {"decoder": method, "language": language, "segmentation_decode": "semi_markov_viterbi", "text_routing": TEXT_ROUTING}
        # The block-causal mask is a training choice: a DLM decoded at another block runs under a mask it never saw.
        if method == "dlm": expected["block_size"] = int(method_cfg.get("block_size", 16))
        for key, value in expected.items():
            if meta.get(key) != value: raise ValueError(f"{ckpt}: expected {key}={value!r}, found {meta.get(key)!r}")
        require_fixed_constants(meta, inference_cfg, ckpt)   # after identity: a wrong model is named as such, not as a geometry drift
        duration = DurationModel(**meta["duration_model"])
        
    target_lang = data_cfg["languages"][language].get("target_lang")
    prompt_lang = PROMPT_LANG_BY_TARGET.get(str(target_lang or ""), "English")
    lm_name = str(cfg_get(method_cfg, "language_model", "name", default="google/mt5-base"))

    if method == "baseline":
        # Released Uni-Sign mT5 pose-only model, or the in-domain baseline_train checkpoint.
        # Same `MisalignedSLTModel(decoder="ar")` as ar, without a segmentation branch (no BIO head, no second pose encoder).
        tokenizer = T5Tokenizer.from_pretrained(lm_name, legacy=False)
        front_end = UniSignMT5FrontEnd(mt5_name=lm_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)
        model = MisalignedSLTModel(front_end=front_end, decoder="ar", segmentation_branch=False)
        # --checkpoint or data.yaml pretrained_slt ONLY (the released file, or this language's clean floor after the B2b re-root) 
        # — no `checkpoint.dir` fallback, so a concurrent train-slt run can't slip trained weights in.
        name = checkpoint or resolve_pretrained(method_cfg, data_cfg, language, default="")
        ckpt = Path(name or "")
        if not name or not ckpt.exists(): raise FileNotFoundError(
            f"Missing released Uni-Sign checkpoint for baseline (language '{language}'): {ckpt!s}. Set "
            f"languages.{language}.pretrained_slt in configs/data.yaml (or checkpoint.from_pretrained / --checkpoint)."
        )
        # A trained file must be THIS language's clean floor (not an arm, not another language's floor) under the fixed constants; 
        # the released Uni-Sign file carries no meta and passes both checks.
        meta = load_checkpoint_meta(ckpt)
        if meta and (meta.get("architecture") != "clean_translation" or str(meta.get("language")) != str(language)): raise ValueError(
            f"{ckpt}: the baseline needs the {language!r} clean translator, "
            f"found architecture={meta.get('architecture')!r}, language={meta.get('language')!r}"
        )
        require_fixed_constants(meta, inference_cfg, ckpt)
        rep = model.front_end.load_pretrained(ckpt, strict=True)
        print(f"[unisign] loaded {ckpt.name}: {rep['pose_tensors']} pose + {rep['mt5_tensors']} LM tensors (missing "
              f"{rep['pose_missing'] + rep['mt5_missing']}, unexpected {rep['pose_unexpected'] + rep['mt5_unexpected']})", flush=True)
        model.to(device); model.eval()
        model.checkpoint_path, model.checkpoint_meta = str(ckpt), meta   # the file actually loaded, for provenance (model_annotation)
        return model, tokenizer

    # Trained ar / dlm: the same pose encoder + prompt + mT5 front end as the baseline.
    tokenizer = T5Tokenizer.from_pretrained(lm_name, legacy=False)
    front_end = UniSignMT5FrontEnd(mt5_name=lm_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)
    model = MisalignedSLTModel(
        front_end=front_end, decoder="ar" if method == "ar" else "dlm",
        bio_hidden_dim=int(method_cfg.get("bio_hidden_dim", 384)), bio_depth=int(method_cfg.get("bio_depth", 4)),
        bio_nhead=int(method_cfg.get("bio_nhead", 8)), bio_dropout=float(method_cfg.get("bio_dropout", 0.1)),
        bio_conv_stem_layers=int(method_cfg.get("bio_conv_stem_layers", 2)), block_size=int(method_cfg.get("block_size", 16)),
        # The band is architecture a strict load cannot see: build it from the checkpoint's stamp, not the live config.
        bio_attention_radius_s=meta.get("bio_attention_radius_s"), bio_adapter=bool(meta.get("bio_adapter", False)),
        shared_pose_encoder=meta.get("architecture") == "one_stream_slt",
    )
    load_model_checkpoint(model, ckpt, strict=True)
    model.duration_model = duration
    model.to(device); model.eval()
    model.checkpoint_path, model.checkpoint_meta = str(ckpt), meta   # the file actually loaded, for provenance (model_annotation)
    return model, tokenizer


def rq2_translator_token(args: argparse.Namespace, arm_checkpoint: str | None = None) -> str:
    """The `translator` field: `clean`, `ar`, `dlm`, or an arm running a non-default method config.

    An ablation run is the same method under another config, so without the config stem it writes the main row's
    events, scores and stability file over the top and prints nothing.
    """
    cfg = getattr(args, "method_config", None)
    suffix = "" if not cfg or str(cfg) == METHOD_CONFIGS[args.method] \
                else "-" + Path(cfg).stem.removeprefix("ablation_").replace("_", "-")
    if args.method != "baseline" and arm_checkpoint: suffix += "".join(_arm_variant_tokens(args, arm_checkpoint))
    if getattr(args, "no_translate", False): return "none" + suffix
    return ("clean" if args.method == "baseline" else str(args.method)) + suffix


def _arm_variant_tokens(args: argparse.Namespace, arm_checkpoint: str) -> list[str]:
    # Name tokens for an arm whose stamps differ from the method config this run reads.
    meta = load_checkpoint_meta(arm_checkpoint)
    live = load_yaml(_method_config_path(args), language=getattr(args, "language", None))
    tokens = ["-onestream"] if meta.get("architecture") == "one_stream_slt" else []
    detach = (meta.get("gate") or {}).get("detach_omega")
    if detach is not None and bool(detach) != bool((live.get("membership_gate") or {}).get("detach_omega", False)):
        tokens.append("-detach" if detach else "-nodetach")

    adapter = meta.get("bio_adapter")   # the one-stream arm runs without the adapter by construction: its own token names it
    if meta.get("architecture") == "two_stream_slt" and adapter is not None and bool(adapter) != bool(live.get("bio_adapter", False)):
        tokens.append("-adapter" if adapter else "-noadapter")

    lt = meta.get("lambda_trans")
    if lt is not None and abs(float(lt) - float(live.get("lambda_trans", 1.0))) > 1e-9: tokens.append(f"-lt{float(lt):g}")
    seed = meta.get("seed")   # a repeat of an arm under another seed
    if seed is not None and int(seed) != int(live.get("seed", 42)): tokens.append(f"-seed{int(seed)}")
    return tokens


def _load_segmenter(args):
    """Trained segmenter by --segmenter-arch (shared by eval --segmenter-eval and analyze --stage segmenter-infer).

    moryossef (default): faithful Moryossef segmenter (UNet) on THEIR input contract — no preprocessing shared with the FSM head. 
    It supplies the RQ2 cascade spans. s1 is the in-system BIO head, isolating system design from segmentation competence. 
    rope_chunk_s is SECONDS (s1) or None (moryossef).
    """
    args.segmenter_arch = args.segmenter_arch or "moryossef"
    device = pick_device(args.device)
    # The untrained controls exist only for the arm whose input contract is a reconstruction of a RELEASED model's.
    if getattr(args, "segmenter_init", "finetuned") != "finetuned" and args.segmenter_arch != "moryossef": raise SystemExit(
        f"--segmenter-init {args.segmenter_init} is moryossef-only; --segmenter-arch {args.segmenter_arch} has no untrained control."
    )
    if args.segmenter_arch == "s1":
        from train.bio_pretrain import build_bio_s1_model
        cfg = load_yaml(args.bio_config, language=args.language)
        model = build_bio_s1_model(cfg)
        # A pooled (multilingual) S1 is ONE language-agnostic checkpoint; per-language analysis on any --language
        # reads the same pool directory. This is zero-shot only for a language excluded from all S1 training.
        ckpt_default = f"checkpoints/bio_s1/{args.language}"
        # Chunked RoPE at the head's TRAINED context, in SECONDS: S1 trains on chunks up to pretrain_geometry.buffer_cap_s 
        # (data/chunks.py), so eval chunks there too (wrapper converts to frames per stream fps). Larger chunks would attend 
        # over untrained context. The live cap is the start value; the checkpoint's stamp replaces it below.
        velocity, rope_chunk_s = False, float(load_yaml(args.inference_config)["buffer_cap_s"])
    elif getattr(args, "segmenter_init", "finetuned") in ("released", "random"):
        # 2 untrained controls on same geometry and same input contract the arm trains under (moryossef26.dataset.to_release_coords), 
        # so both isolate training from representation. `released` scores DGS weights with nothing trained; `random` is the range it 
        # has to be read against — a released score means nothing on its own, because an untrained network of this shape is not at 0.
        from moryossef26.trainer import build_moryossef
        cfg = load_yaml(args.moryossef_config, language=args.language)
        init = args.segmenter_init
        released = (cfg.get("checkpoint", {}) or {}).get("from_pretrained")
        if init == "released" and (not released or not Path(released).exists()):
            raise SystemExit(f"--segmenter-init released needs checkpoint.from_pretrained; {released!r} is missing.")
        model = build_moryossef(args.moryossef_config, pretrained=init == "released", seed=None if init == "released" else int(args.seed))
        stats_from = Path(args.checkpoint or Path(checkpoint_dir(cfg, default=f"checkpoints/moryossef/{args.language}")) / "model.pt")
        model.release_stats = (load_checkpoint_meta(str(stats_from)) or {}).get("release_stats")
        if model.release_stats is None: raise SystemExit(
            f"--segmenter-init {init} needs the arm's fitted release_stats, absent from {stats_from}. Train the "
            f"Moryossef arm first: a control must read the same coordinates as the arm it controls for."
        )
        model.eval().to(device)
        source = f"{released} (released weights, no fine-tune)" if init == "released" else f"random init (seed {args.seed})"
        return model, device, bool(cfg.get("velocity", True)), None, f"{source}; release_stats from {stats_from}"
    else:
        from moryossef26.trainer import build_moryossef
        cfg = load_yaml(args.moryossef_config, language=args.language)
        model = build_moryossef(args.moryossef_config, pretrained=False)   # every weight comes from the fine-tuned checkpoint below
        ckpt_default = f"checkpoints/moryossef/{args.language}"
        velocity, rope_chunk_s = bool(cfg.get("velocity", True)), None  # UNet chunks internally at num_frames

    ckpt_dir = checkpoint_dir(cfg, default=ckpt_default)
    checkpoint = args.checkpoint or str(Path(ckpt_dir) / "model.pt")
    if args.segmenter_arch == "s1":
        # A stage-2 checkpoint (the deployed arm) is accepted for the S1 arch: its segmentation branch is re-keyed 
        # to the S1 layout (models.checkpointing.s1_layout_state), so the joint head can be scored alone.
        model.load_state_dict(s1_layout_state(_load_state(checkpoint)), strict=True)
    else: load_model_checkpoint(model, checkpoint, strict=True)
    _meta = load_checkpoint_meta(checkpoint)
    print(f"segmenter | {args.segmenter_arch} weights from {checkpoint} (pool {_meta.get('pretrain_pool')!r})", flush=True)
    # S1's RoPE chunk is the buffer cap the head TRAINED under, which the checkpoint records. It wins over the live buffer_cap_s, because a 
    # cap edit would re-chunk a trained head over context it never saw. PROVENANCE: config says which pool this run expects. The checkpoint 
    # records which pool produced it. A mismatch means the wrong segmenter is about to be evaluated — silently, with a plausible-looking 
    # score — so it fails loud. A stage-2 arm scored through its S1-layout head stamps no `pretrain_pool` and is exempt.
    if "pretrain_pool" in _meta and _meta.get("pretrain_pool") != pool_key(cfg): raise SystemExit(
        f"{checkpoint} was trained on pool {_meta.get('pretrain_pool')!r}, but this config expects {pool_key(cfg)!r}. Point --checkpoint "
        f"at the matching model, or align `pretrain_languages` (a pooled checkpoint is a DIFFERENT model from a monolingual one)."
    )
    if "pretrain_pool" in _meta:  # a segmenter checkpoint (S1 or Moryossef); a stage-2 arm's head carries the arm's own provenance
        from train.bio_pretrain import require_segmenter_data
        train_labels = annotation_fingerprint(_cached_language_records(load_yaml(args.data_config), args.language, "train"))
        require_segmenter_data(_meta, args.language, train_labels, load_yaml(args.inference_config)["buffer_cap_s"], checkpoint)
    if args.segmenter_arch == "moryossef":
        model.release_stats = _meta.get("release_stats")
        if model.release_stats is None: raise SystemExit(
            f"{checkpoint} stamps no release_stats, so its training coordinates cannot be reproduced. Retrain the Moryossef arm."
        )
    if args.segmenter_arch == "s1": # Head's attention band isn't in any tensor: the model must carry the band the checkpoint trained with.
        from train.bio_pretrain import require_attention_radius
        require_attention_radius(_meta, model.bio_head.attention_radius_s, checkpoint)
    if args.segmenter_arch == "s1": # S1 stamps rope_eval_chunk_s. A arm stamps the cap it trained under (its head's context) as buffer_cap_s.
        trained_chunk = _meta.get("rope_eval_chunk_s") or _meta.get("buffer_cap_s")
        if trained_chunk:
            if abs(float(trained_chunk) - float(rope_chunk_s)) > 1e-6: print(
                f"segmenter | rope_eval_chunk_s {float(trained_chunk):.2f}s from the checkpoint "
                f"(buffer_cap_s says {float(rope_chunk_s):.2f}s); using the trained value.", flush=True
            )
            rope_chunk_s = float(trained_chunk)
    return model, device, velocity, rope_chunk_s, checkpoint


def _prep_window(poses_np: np.ndarray, timestamps_np: np.ndarray, start_s: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Uni-Sign uses raw windows: the encoder masks padding via attention mask, no boundary halos.
    poses = torch.as_tensor(poses_np, dtype=torch.float32)
    timestamps = torch.as_tensor(np.asarray(timestamps_np, dtype=np.float32) - float(start_s), dtype=torch.float32)
    return poses, timestamps, torch.ones(poses.shape[0], dtype=torch.bool)

def _commit_box_s(inference_cfg: dict) -> tuple[float, float]:
    """How far the FSM buffer that commits a span reaches past the span's crop, as (lead, tail) seconds. The buffer holds δ before the
    span (boundary_stability.delta_s), and a terminator commits after K stable strides (boundary_stability.hysteresis_strides, stride_s,
    read as _build_streaming_runner reads them), so at commit the buffer holds about (K - 1) strides after the terminator frame."""
    boundary = inference_cfg.get("boundary_stability", {})
    return float(boundary["delta_s"]), (int(boundary.get("hysteresis_strides", 3)) - 1) * float(inference_cfg.get("stride_s", 1.0))

def _box_normalized_window(pose, start_s: float, end_s: float, box_start_s: float, box_end_s: float) -> tuple[np.ndarray, np.ndarray]:
    """The frames and timestamps of load_pose_window(pose, start_s, end_s), normalized with ONE Uni-Sign body box over the raw window
    [box_start_s, box_end_s) around them. Only the body box differs from a crop normalized alone."""
    raw, timestamps = load_pose_window(pose, start_s, end_s, normalize=False)
    if raw.shape[0] == 0 or raw.shape[1:] != (133, 3): return raw, timestamps   # as load_pose_window, which normalizes raw DWPose only
    box, box_ts = load_pose_window(pose, min(box_start_s, start_s), max(box_end_s, end_s), normalize=False)
    first = int(round((float(timestamps[0]) - float(box_ts[0])) * float(pose.fps)))
    return normalize_keypoints_unisign(box)[first:first + raw.shape[0]], timestamps

def _decode_provenance(method: str, method_cfg: dict) -> dict:
    # The DLM's decode threshold as run (a fixed constant), recorded so every payload names the decode that made it.
    if method != "dlm": return {}
    spd_cfg = method_cfg.get("spd", {})
    return {"tau_dec": float(spd_cfg["tau_dec"]),
            "spd_top_k": int(spd_cfg.get("top_k", 1)), "spd_renormalize": bool(spd_cfg.get("renormalize", True))}

def _generation_kwargs(method: str, inference_cfg: dict, method_cfg: dict, max_tokens: int, fps: list[float]) -> dict:
    # Main AR rows are greedy; only DLM uses the block-decode params. `fps`: 1 pose rate per row of the batch.
    if method == "baseline":
        num_beams = int(method_cfg.get("validation", {}).get("num_beams", 1))
        return {"max_text_tokens": max_tokens, "num_beams": num_beams}

    spd_cfg = method_cfg.get("spd", {})
    return {
        "max_text_tokens": max_tokens, "num_beams": 1, "tau_dec": float(spd_cfg.get("tau_dec", 0.5)),
        "spd_top_k": int(spd_cfg.get("top_k", 1)), "spd_renormalize": bool(spd_cfg.get("renormalize", True)),
        # Membership gate at RQ1: same Ω the decoder trained with (on-policy span, no GT/χ). BOTH arms gated — 
        # DLM injects Ω in its manual decode, AR via HF cross-attention hooks (front_end.ar_generate).
        "gate_enabled": bool(method_cfg.get("membership_gate", {}).get("enabled", False)),
        # Same first-eligible-unit rule as the training sampler and streaming runner.
        "gate_eps": float(method_cfg.get("membership_gate", {}).get("eps", 1e-4)),
        "gate_min_span_frames": [seconds_to_frames(lambda_min_s(inference_cfg), rate) for rate in fps],
    }

@torch.no_grad()
def _translate_windows(
    model, tokenizer, method: str, items: list[tuple[np.ndarray, np.ndarray, float, float]], device: torch.device,
    inference_cfg: dict, method_cfg: dict, batch_size: int | None = None, stream_start: bool = False,
    commit_frontier_s: list[float] | None = None, anchor_frames: list[tuple[int, int]] | None = None,
) -> list[tuple[str, float, bool]]:
    """Pre-trimmed pose windows (poses, timestamps, start_s, fps) -> [(text, mean_token_confidence, gate_would_skip)]. 
    The loop-decode entry point: ONE `generate_from_poses` path for every method.

    Confidence is the real softmax prob of the model's own tokens, so "confidently wrong" is measured. Rows are right-padded and 
    `frame_mask`-masked, so each row's result matches translating it alone. `batch_size` None = 1 batch padded to its longest row; 
    mixed-length corpora should pass --batch-size."""
    if not items: return []
    if batch_size is not None and int(batch_size) < len(items):
        out: list[tuple[str, float, bool]] = []
        for i in range(0, len(items), max(1, int(batch_size))): out.extend(_translate_windows(
            model, tokenizer, method, items[i:i + max(1, int(batch_size))], device, inference_cfg, method_cfg, stream_start=stream_start,
            commit_frontier_s=commit_frontier_s[i:i + max(1, int(batch_size))] if commit_frontier_s else None,
            anchor_frames=anchor_frames[i:i + max(1, int(batch_size))] if anchor_frames else None,
        ))
        return out
    prepped = [_prep_window(p, ts, start_s) for (p, ts, start_s, _) in items]
    max_t = max(int(p.shape[0]) for p, _, _ in prepped)
    
    # Same repeat-last-frame pad + frame-mask contract as the training collator (data.batch).
    poses = torch.stack([repeat_last_frame(p, max_t - int(p.shape[0])) for p, _, _ in prepped]).to(device)
    timestamps = torch.stack([torch.nn.functional.pad(ts, (0, max_t - int(ts.shape[0]))) for _, ts, _ in prepped]).to(device)
    frame_mask = torch.stack([torch.nn.functional.pad(m, (0, max_t - int(m.shape[0]))) for _, _, m in prepped]).to(device)

    max_tokens = int(inference_cfg["translation"]["max_text_tokens"])  # the decode canvas, as in the RQ2 runner
    gen_kwargs = _generation_kwargs(method, inference_cfg, method_cfg, max_tokens, [fps for *_, fps in items])
    # Whether frame 0 is a genuine sentence ONSET is a property of how the caller cut this window, not a constant:
    # a window that starts BEFORE the sentence opens inside the predecessor, and minting a B there would anchor Ω
    # on the predecessor's tail and floor the sentence being scored. Baseline is ungated, so the flag is inert.
    if gen_kwargs.get("gate_enabled"):
        gen_kwargs["gate_stream_start"] = bool(stream_start)
        # Without anchors the gate selects its own first span (duration decode); with them Ω reads the caller's span on every row.
        if anchor_frames is not None:  # window-relative (start, terminator) frames; terminator -1 = the span runs to the window end
            gen_kwargs["gate_anchor"] = torch.tensor([list(a) for a in anchor_frames], dtype=torch.long, device=device)
        # commit_frontier_s (per item, window-relative): frames strictly before it are the already-handled predecessor — the FSM's χ. 
        # Without it a window opening MID-SIGNING has an unopenable leading I-run (buffer-start I never opens; no χ mint fires) and Ω 
        # anchors on a shifted fragment or the NEXT sentence.
        if commit_frontier_s is not None: gen_kwargs["commit_mask"] = timestamps < torch.tensor(
            commit_frontier_s, dtype=timestamps.dtype, device=timestamps.device
        ).unsqueeze(1)
    _, tokens, confidence, gate_skip = model.generate_from_poses(poses=poses, frame_mask=frame_mask, timestamps_s=timestamps, **gen_kwargs)
    tok = tokens.detach().cpu()
    conf = confidence.detach().float().cpu()
    texts = [t.strip() for t in tokenizer.batch_decode(tok, skip_special_tokens=True)]

    # Mean prob over REAL tokens: both arms return produced tokens only (synthetic start slots stripped at the decode boundary), so mask 
    # pads and everything past 1st EOS — batched DLM rows have no per-row trim, and post-EOS filler dilutes the "confidently wrong" signal.
    n = min(tok.shape[1], conf.shape[1])
    tok, conf = tok[:, :n], conf[:, :n]
    valid = tok != int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        cum = (tok == int(tokenizer.eos_token_id)).cumsum(dim=1)
        valid &= (cum == 0) | ((cum == 1) & (tok == int(tokenizer.eos_token_id)))  # keep the EOS slot itself
    confs = ((conf * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)).tolist()
    # gate_skip: the deployed FSM would never decode this window (no span — all-gap or headless fragment). Text is still reported, 
    # but RQ1 must split "translation failed" from "policy is to skip", else the gate's refusal scores as garbage BLEU.
    skips = [bool(s) for s in gate_skip.tolist()]
    return list(zip(texts, [float(c) for c in confs], skips))


def _build_streaming_runner(
    model, inference_cfg: dict, method_cfg: dict, translate: bool = True, cascade_model=None, 
    record_trace: bool = False, record_shadow: bool = False,
) -> StreamingSLTRunner:
    spd = method_cfg.get("spd", {})
    boundary = inference_cfg.get("boundary_stability", {})

    return StreamingSLTRunner(
        model, stride_s=float(inference_cfg.get("stride_s", 1.0)),
        buffer_cap_s=float(inference_cfg["buffer_cap_s"]),
        delta_s=float(boundary["delta_s"]),
        hysteresis_strides=int(boundary.get("hysteresis_strides", 3)),
        min_span_s=lambda_min_s(inference_cfg),
        # Ω from the method config's membership_gate; χ from the runner's commit log.
        gate_enabled=bool(method_cfg.get("membership_gate", {}).get("enabled", False)),
        gate_eps=float(method_cfg.get("membership_gate", {}).get("eps", 1e-4)),
        max_text_tokens=int(inference_cfg["translation"]["max_text_tokens"]),
        tau_dec=float(spd.get("tau_dec", 0.5)),  # same source and default as _generation_kwargs
        spd_top_k=int(spd.get("top_k", 1)), spd_renormalize=bool(spd.get("renormalize", True)),
        translate=bool(translate), cascade_model=cascade_model, 
        record_trace=bool(record_trace), record_shadow=bool(record_shadow),
    )
