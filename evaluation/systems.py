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
from data.loader import cached_language_records, annotation_fingerprint
from poses import load_pose_window, normalize_keypoints_unisign

from infer.duration_decode import DurationModel
from infer.stream import S1RunnerAdapter, StreamingSLTRunner
from models.bio_head import RoPEBIOHead
from models.streaming_slt import MisalignedSLTModel, TEXT_ROUTING
from models.unisign import PROMPT_LANG_BY_TARGET, UniSignMT5FrontEnd
from models.checkpointing import (load_state, load_checkpoint_meta, load_model_checkpoint, require_fixed_constants, s1_layout_state,
                                  warn_pose_geometry)
from utils import checkpoint_dir, lambda_min_s, load_yaml, pick_device, pool_key, resolve_pretrained, seconds_to_frames

METHOD_CONFIGS = {"baseline": "configs/baseline_eval.yaml", "ar": "configs/ar.yaml", "dlm": "configs/dlm.yaml"}
SEGMENTERS, ARMS = ("moryossef", "s1"), ("ar", "dlm")   # eval.py --spans: the standalone segmenters, and the arms whose head cuts the spans
# eval.py --text -> the key of METHOD_CONFIGS that a translator's build, decode and provenance read (pretrimmed: the pretrimmed translator).
TEXT_METHODS = {"pretrimmed": "baseline", "ar": "ar", "dlm": "dlm"}

def source_language(args) -> str:
    """The corpus the run's models trained on: eval.py / analyze.py --source-language, default --language. Their method config,
    stamp checks, translator warm start (pretrained_slt), train-label warnings and duration prior are this corpus's; the data, gold,
    test guard and text scoring stay --language's (a zero-shot run when the 2 differ, evaluation.predictions.corpus_token)."""
    return getattr(args, "source_language", None) or args.language

def model_annotation(model, data_cfg: dict, language: str) -> dict | None:
    """The train labels behind a built translator (an arm, or a trained pretrimmed translator; `build_eval_model` keeps its meta) and behind
    the inits it stamps, beside the current `language` train labels (the corpus the model trained on, `source_language`): {train,
    model, translator_init, segmentation_init}, for the provenance. Warns, never refuses, when one differs. None for a file with no
    training meta (the released Uni-Sign weights)."""
    meta = getattr(model, "checkpoint_meta", None)
    if not meta: return None
    inits = (meta.get("resolved_inits") or {}).get("annotation_signature") or {}   # train/slt.py _resolved_inits
    s1_langs = inits.get("segmentation_languages") or {}
    found = {
        "model": meta.get("annotation_fingerprint"),
        "train": annotation_fingerprint(cached_language_records(data_cfg, language, "train")),
        # S1 trains on a pool: its own fingerprint hashes every pooled language, so the per-language stamp is the one to compare.
        "translator_init": inits.get("translator"), "segmentation_init": s1_langs.get(language),
    }
    # A corpus that trains its pretrimmed translators on another corpus (data.yaml floor_corpus) compares them with that corpus's labels:
    # the pretrimmed translator itself (a --text pretrimmed row) and an arm's translator warm start.
    floor = ((data_cfg.get("languages") or {}).get(language) or {}).get("floor_corpus")
    if floor and floor != language: found["floor_train"] = annotation_fingerprint(cached_language_records(data_cfg, floor, "train"))
    warn_model_labels(found, getattr(model, "checkpoint_path", None))
    return found

def warn_model_labels(found: dict | None, source) -> bool:
    names = {"model": "the model", "translator_init": "its translator warm start", "segmentation_init": "its S1 init"}
    found = found or {}
    # With a floor corpus (floor_train, model_annotation), the translator warm start follows that corpus's labels, and the model may be
    # that pretrimmed translator itself.
    floor = found.get("floor_train")
    refs = {"model": {found.get("train"), floor}, "translator_init": {floor or found.get("train")}, "segmentation_init": {found.get("train")}}
    stale = [name for key, name in names.items() if found.get(key) is not None and found.get(key) not in refs[key]]
    if stale: print(
        f"WARNING: {source}: {', '.join(stale)} trained on other train labels than {found.get('train')}, the train labels "
        f"at eval time. Its spans or text can follow the old labels: retrain before a run you report.", flush=True
    )
    return not stale

def parse_grid(value: str | None, fallback: list[float]) -> list[float]:
    if not value: return [float(x) for x in fallback]
    return [float(x.strip()) for x in value.split(",") if x.strip()]

def method_config_path(args: argparse.Namespace, method: str) -> str:
    # The config of `method` (a key of METHOD_CONFIGS): --method-config, else its own file. evaluation.rows gives the span pass of a
    # cascade no --method-config, so it replaces the config of the row's translator, or with no text, of the model that cuts the spans.
    return str(args.method_config) if getattr(args, "method_config", None) else METHOD_CONFIGS[method]

def build_eval_model(
    method: str, checkpoint: str | None, language: str, data_cfg: dict, 
    method_cfg: dict, inference_cfg: dict, device: torch.device
):
    """Uni-Sign pose-only model. LM (`language_model.name`) and checkpoint (`checkpoint.from_pretrained` / `checkpoint.dir`) both come from  
    METHOD config — no separate stage-1 config. Plain scalars so analyze/visualize need no fake Namespace. The file actually loaded (for the
    pretrimmed translator: --pretrimmed-checkpoint or data.yaml pretrained_slt) passes fixed-constant check here, so every row that builds a
    model is covered once.
    The model keeps that file's meta; the boundary sweep and the rows compare its train labels (model_annotation)."""
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
        if method == "dlm": expected["block_size"] = int(method_cfg["block_size"])
        for key, value in expected.items():
            if meta.get(key) != value: raise ValueError(f"{ckpt}: expected {key}={value!r}, found {meta.get(key)!r}")
        require_fixed_constants(meta, inference_cfg, ckpt)   # after identity: a wrong model is named as such, not as a geometry drift
        warn_pose_geometry(meta, ckpt)
        duration = DurationModel(**meta["duration_model"])
        
    target_lang = data_cfg["languages"][language].get("target_lang")
    prompt_lang = PROMPT_LANG_BY_TARGET.get(str(target_lang or ""), "English")
    lm_name = str(method_cfg["language_model"]["name"])

    if method == "baseline":
        # Released Uni-Sign mT5 pose-only model (AR), or an in-domain pretrimmed translator of the decoder it stamps (translator_decoder).
        # Same MisalignedSLTModel as the arms, without a segmentation branch (no BIO head, no second pose encoder).
        # --pretrimmed-checkpoint or data.yaml pretrained_slt ONLY (the released file, or this language's pretrimmed translator after
        # the B2b re-root) — no `checkpoint.dir` fallback, so a concurrent train-slt run can't slip trained weights in.
        name = checkpoint or resolve_pretrained(method_cfg, data_cfg, language, default="")
        ckpt = Path(name or "")
        if not name or not ckpt.exists(): raise FileNotFoundError(
            f"Missing the checkpoint of the pretrimmed translator (language '{language}'): {ckpt!s}. Set languages.{language}."
            f"pretrained_slt.<decoder> in configs/data.yaml (or checkpoint.from_pretrained / --pretrimmed-checkpoint)."
        )
        # A trained file must be the pretrimmed translator (not an arm) of THIS language, or of the corpus that it trains its pretrimmed
        # translators on (data.yaml floor_corpus), under the fixed constants; the released Uni-Sign file carries no meta and passes both
        # checks.
        meta = load_checkpoint_meta(ckpt)
        floor = str(((data_cfg.get("languages") or {}).get(language) or {}).get("floor_corpus") or language)
        if meta and (meta.get("architecture") != "clean_translation" or str(meta.get("language")) != floor): raise ValueError(
            f"{ckpt}: --text pretrimmed needs the {floor!r} pretrimmed translator, "
            f"found architecture={meta.get('architecture')!r}, language={meta.get('language')!r}"
        )
        require_fixed_constants(meta, inference_cfg, ckpt)
        warn_pose_geometry(meta, ckpt)   # silent for the released file, which has no meta
        tokenizer = T5Tokenizer.from_pretrained(lm_name, legacy=False)
        front_end = UniSignMT5FrontEnd(mt5_name=lm_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)
        if translator_decoder("baseline", ckpt) == "ar":
            model = MisalignedSLTModel(front_end=front_end, decoder="ar", segmentation_branch=False)
            rep = model.front_end.load_pretrained(ckpt, strict=True)
            print(f"[unisign] loaded {ckpt.name}: {rep['pose_tensors']} pose + {rep['mt5_tensors']} LM tensors (missing "
                  f"{rep['pose_missing'] + rep['mt5_missing']}, unexpected {rep['pose_unexpected'] + rep['mt5_unexpected']})", flush=True)
        else:
            # A pretrimmed DLM translator, as the DLM arm: decoded at the block it trained with (block_size), and its block decode has no beam
            # search (a num_beams above 1 would be recorded but not run). It loads whole and strict, so a missing canvas fails.
            if meta["block_size"] != int(method_cfg["block_size"]): raise ValueError(
                f"{ckpt}: expected block_size={int(method_cfg['block_size'])!r}, found {meta['block_size']!r}"
            )
            if int(method_cfg["validation"]["num_beams"]) != 1: raise SystemExit(
                f"{ckpt} is a pretrimmed DLM translator: its block decode has no beam search, so num_beams must be 1."
            )
            model = MisalignedSLTModel(front_end=front_end, decoder="dlm", block_size=int(method_cfg["block_size"]), segmentation_branch=False)
            load_model_checkpoint(model, ckpt, strict=True)
        model.to(device); model.eval()
        model.checkpoint_path, model.checkpoint_meta = str(ckpt), meta   # the file actually loaded, for provenance (model_annotation)
        return model, tokenizer

    # Trained ar / dlm: the same pose encoder + prompt + mT5 front end as the baseline.
    tokenizer = T5Tokenizer.from_pretrained(lm_name, legacy=False)
    front_end = UniSignMT5FrontEnd(mt5_name=lm_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)
    model = MisalignedSLTModel(
        front_end=front_end, decoder="ar" if method == "ar" else "dlm",
        bio_hidden_dim=int(method_cfg["bio"]["hidden_dim"]), bio_depth=int(method_cfg["bio"]["depth"]),
        bio_nhead=int(method_cfg["bio"]["nhead"]), bio_dropout=float(method_cfg["bio"]["dropout"]),
        bio_conv_stem_layers=int(method_cfg["bio"]["conv_stem_layers"]), block_size=int(method_cfg["block_size"]),
        # The band is architecture a strict load cannot see: build it from the checkpoint's stamp, not the live config.
        bio_attention_radius_s=meta.get("bio_attention_radius_s"), bio_adapter=bool(meta.get("bio_adapter", False)),
        shared_pose_encoder=meta.get("architecture") == "one_stream_slt",
    )
    load_model_checkpoint(model, ckpt, strict=True)
    model.duration_model = duration
    model.to(device); model.eval()
    model.checkpoint_path, model.checkpoint_meta = str(ckpt), meta   # the file actually loaded, for provenance (model_annotation)
    return model, tokenizer


def translator_decoder(method: str, checkpoint) -> str:
    # The decoder of a row's translator: an arm's own method, or the decoder a pretrimmed translator's file stamps. The released Uni-Sign file
    # stamps nothing and is an AR mT5, and so is a missing file (the build then refuses it): the one place of that rule.
    if method != "baseline": return str(method)
    meta = load_checkpoint_meta(checkpoint) if checkpoint and Path(str(checkpoint)).exists() else {}
    return str(meta["decoder"]) if meta else "ar"


def translator_token(args: argparse.Namespace, method: str, checkpoint: str | None = None, translate: bool = True) -> str:
    """The name of the model of `method` that a row loaded (`checkpoint`: the arm, or the pretrimmed translator): `pretrimmed` (the
    pretrimmed AR translator or the released file), `pretrimmed-dlm` (a pretrimmed DLM translator), `ar` or `dlm`; `none` with the
    model's tokens when it translates nothing (`translate` False: the `text` field of a no-text row). A non-default --method-config
    adds its stem, and an arm its variant tokens.

    An ablation run is the same method under another config, so without the config stem it writes the main row's
    events, scores and stability file over the top and prints nothing.
    """
    cfg = getattr(args, "method_config", None)
    suffix = "" if not cfg or str(cfg) == METHOD_CONFIGS[method] \
                else "-" + Path(cfg).stem.removeprefix("ablation_").replace("_", "-")
    if method != "baseline" and checkpoint: suffix += "".join(arm_variant_tokens(args, method, checkpoint))
    if not translate: return "none" + suffix
    if method != "baseline": return str(method) + suffix
    return ("pretrimmed-dlm" if translator_decoder("baseline", checkpoint) == "dlm" else "pretrimmed") + suffix


def arm_variant_tokens(args: argparse.Namespace, method: str, arm_checkpoint: str) -> list[str]:
    # Name tokens for an arm whose stamps differ from the method config this run reads.
    meta = load_checkpoint_meta(arm_checkpoint)
    live = load_yaml(method_config_path(args, method), language=getattr(args, "language", None))
    tokens = ["-onestream"] if meta.get("architecture") == "one_stream_slt" else []
    detach = (meta.get("gate") or {}).get("detach_omega")
    if detach is not None and bool(detach) != bool(live["membership_gate"]["detach_omega"]):
        tokens.append("-detach" if detach else "-nodetach")

    adapter = meta.get("bio_adapter")   # the one-stream arm runs without the adapter by construction: its own token names it
    if meta.get("architecture") == "two_stream_slt" and adapter is not None and bool(adapter) != bool(live["bio"]["adapter"]):
        tokens.append("-adapter" if adapter else "-noadapter")

    lt = meta.get("lambda_trans")
    if lt is not None and abs(float(lt) - float(live["lambda_trans"])) > 1e-9: tokens.append(f"-lt{float(lt):g}")
    seed = meta.get("seed")   # a repeat of an arm under another seed
    if seed is not None and int(seed) != int(live["seed"]): tokens.append(f"-seed{int(seed)}")
    return tokens


def load_segmenter(args):
    """The standalone segmenter of eval.py --spans (moryossef or s1), from --segmenter-checkpoint, else its pooled folder.

    moryossef: faithful Moryossef segmenter (UNet) on THEIR input contract — no preprocessing shared with the FSM head. 
    It supplies the spans of the cascade rows. s1 is the in-system BIO head, isolating system design from segmentation competence. 
    rope_chunk_s is SECONDS (s1) or None (moryossef).
    """
    arch = args.spans
    source = source_language(args)   # the corpus the segmenter trained on: its config, default folder and stamp checks
    device = pick_device(args.device)
    # The untrained controls (--segmenter-init, the Moryossef segmenter only; evaluation.rows refuses another) reconstruct a RELEASED model.
    if arch == "s1":
        cfg = load_yaml(args.bio_config, language=source)
        # A pooled (multilingual) S1 is ONE language-agnostic checkpoint; per-language analysis on any --language
        # reads the same pool directory. This is zero-shot only for a language excluded from all S1 training.
        ckpt_default = f"checkpoints/bio_s1/{source}"
        # Chunked RoPE at the head's TRAINED context, in SECONDS: S1 trains on chunks up to the cap it trained under
        # (inference.yaml buffer_cap_s at train time; data/chunks.py), so eval chunks there too (wrapper converts to frames per stream fps). Larger chunks would attend 
        # over untrained context. The live cap is the start value; the checkpoint's stamp replaces it below.
        velocity, rope_chunk_s = False, float(load_yaml(args.inference_config)["buffer_cap_s"])
    elif getattr(args, "segmenter_init", "finetuned") in ("released", "random"):
        # 2 untrained controls on same geometry and same input contract the arm trains under (moryossef26.dataset.to_release_coords), 
        # so both isolate training from representation. `released` scores DGS weights with nothing trained; `random` is the range it 
        # has to be read against — a released score means nothing on its own, because an untrained network of this shape is not at 0.
        from moryossef26.trainer import build_moryossef
        cfg = load_yaml(args.moryossef_config, language=source)
        init = args.segmenter_init
        released = (cfg.get("checkpoint", {}) or {}).get("from_pretrained")
        if init == "released" and (not released or not Path(released).exists()):
            raise SystemExit(f"--segmenter-init released needs checkpoint.from_pretrained; {released!r} is missing.")
        model = build_moryossef(args.moryossef_config, pretrained=init == "released", seed=None if init == "released" else int(args.seed))
        stats_from = Path(args.segmenter_checkpoint or Path(checkpoint_dir(cfg, default=f"checkpoints/moryossef/{source}")) / "model.pt")
        model.release_stats = (load_checkpoint_meta(str(stats_from)) or {}).get("release_stats")
        if model.release_stats is None: raise SystemExit(
            f"--segmenter-init {init} needs the arm's fitted release_stats, absent from {stats_from}. Train the "
            f"Moryossef arm first: a control must read the same coordinates as the arm it controls for."
        )
        model.eval().to(device)
        source = f"{released} (released weights, no fine-tune)" if init == "released" else f"random init (seed {args.seed})"
        return model, device, bool(cfg["velocity"]), None, f"{source}; release_stats from {stats_from}"
    else:
        from moryossef26.trainer import build_moryossef
        cfg = load_yaml(args.moryossef_config, language=source)
        model = build_moryossef(args.moryossef_config, pretrained=False)   # every weight comes from the fine-tuned checkpoint below
        ckpt_default = f"checkpoints/moryossef/{source}"
        velocity, rope_chunk_s = bool(cfg["velocity"]), None  # UNet chunks internally at num_frames

    ckpt_dir = checkpoint_dir(cfg, default=ckpt_default)
    checkpoint = args.segmenter_checkpoint or str(Path(ckpt_dir) / "model.pt")
    _meta = load_checkpoint_meta(checkpoint)
    # With no --decode a row decodes as the checkpoint's epoch was chosen (monitor_decode; plain is the stored name of argmax), so that
    # stamp must be the decode that the config names for the segmenter's rows (segmenter_decode): else it is another monitor's choice.
    from train.helpers import segmenter_config_decode
    stamp = _meta.get("monitor_decode")
    if stamp and not getattr(args, "decode", None) and (stamp == "semi_markov_viterbi") != (segmenter_config_decode(cfg) == "duration"):
        raise SystemExit(
            f"{checkpoint} chose its epoch under monitor_decode {stamp!r}, but its config names segmenter_decode "
            f"{segmenter_config_decode(cfg)!r} for the rows of this segmenter: train it again, or name a decode with --decode."
        )
    if arch == "s1":
        # The head's attention band is in no tensor: the model is built with the band the checkpoint stamps (S1 derives it, an arm
        # stamps its S1's; no stamp = full attention). A stage-2 checkpoint (the deployed arm) is accepted for the S1 arch: its
        # segmentation branch is re-keyed to the S1 layout (models.checkpointing.s1_layout_state). The rows refuse it (evaluation.rows):
        # the rows of an arm's head read the whole arm (--spans ar|dlm).
        from train.bio_pretrain import build_bio_s1_model
        model = build_bio_s1_model(cfg, _meta.get("bio_attention_radius_s"))
        model.load_state_dict(s1_layout_state(load_state(checkpoint)), strict=True)
    else: load_model_checkpoint(model, checkpoint, strict=True)
    print(f"segmenter | {arch} weights from {checkpoint} (pool {_meta.get('pretrain_pool')!r}, "
          f"train inputs {_meta.get('pretrain_inputs')!r})", flush=True)   # the segmenter's pretrain.inputs; None: a file trained before the stamp (plain chunks for the Moryossef segmenter)
    # Only S1 (or an arm's head) reads the Uni-Sign geometry. The Moryossef arm stamps none: its shoulder normalisation removes the
    # reader's x map (moryossef26.dataset.pose_aspect), so a file trained before the map reads the same coordinates, except in the rare
    # frames with one visible shoulder, which the normalisation does not scale.
    if arch == "s1": warn_pose_geometry(_meta, checkpoint)
    # S1's RoPE chunk is the buffer cap the head TRAINED under, which the checkpoint records. It wins over the live buffer_cap_s, because a 
    # cap edit would re-chunk a trained head over context it never saw. PROVENANCE: config says which pool this run expects. The checkpoint 
    # records which pool produced it. A mismatch means the wrong segmenter is about to be evaluated — silently, with a plausible-looking 
    # score — so it fails loud. A stage-2 arm scored through its S1-layout head stamps no `pretrain_pool` and is exempt.
    if "pretrain_pool" in _meta and _meta.get("pretrain_pool") != pool_key(cfg): raise SystemExit(
        f"{checkpoint} was trained on pool {_meta.get('pretrain_pool')!r}, but this config expects {pool_key(cfg)!r}. Point --segmenter-checkpoint "
        f"at the matching model, or align `pretrain.languages` (a pooled checkpoint is a DIFFERENT model from a monolingual one)."
    )
    if "pretrain_pool" in _meta:  # a segmenter checkpoint (S1 or Moryossef); a stage-2 arm's head carries the arm's own provenance
        from train.bio_pretrain import require_segmenter_data
        # Only the corpus whose labels this run reads (the source corpus) is compared, so no other corpus of the pool loads.
        source, stamped = source_language(args), _meta.get("language_fingerprints") or {}
        known = {source: annotation_fingerprint(cached_language_records(load_yaml(args.data_config), source, "train"))} if source in stamped else {}
        require_segmenter_data(_meta, known, load_yaml(args.inference_config)["buffer_cap_s"], checkpoint)
    if arch == "moryossef":
        model.release_stats = _meta.get("release_stats")
        if model.release_stats is None: raise SystemExit(
            f"{checkpoint} stamps no release_stats, so its training coordinates cannot be reproduced. Retrain the Moryossef arm."
        )
    if arch == "s1": # S1 stamps rope_eval_chunk_s. A arm stamps the cap it trained under (its head's context) as buffer_cap_s.
        trained_chunk = _meta.get("rope_eval_chunk_s") or _meta.get("buffer_cap_s")
        if trained_chunk:
            if abs(float(trained_chunk) - float(rope_chunk_s)) > 1e-6: print(
                f"segmenter | rope_eval_chunk_s {float(trained_chunk):.2f}s from the checkpoint "
                f"(buffer_cap_s says {float(rope_chunk_s):.2f}s); using the trained value.", flush=True
            )
            rope_chunk_s = float(trained_chunk)
    return model, device, velocity, rope_chunk_s, checkpoint


def prep_window(poses_np: np.ndarray, timestamps_np: np.ndarray, start_s: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Uni-Sign uses raw windows: the encoder masks padding via attention mask, no boundary halos.
    poses = torch.as_tensor(poses_np, dtype=torch.float32)
    timestamps = torch.as_tensor(np.asarray(timestamps_np, dtype=np.float32) - float(start_s), dtype=torch.float32)
    return poses, timestamps, torch.ones(poses.shape[0], dtype=torch.bool)

def commit_box_s(inference_cfg: dict) -> tuple[float, float]:
    """How far the FSM buffer that commits a span reaches past the span's crop, as (lead, tail) seconds. The buffer holds δ before the
    span (boundary_stability.delta_s), and a terminator commits after K stable strides (boundary_stability.hysteresis_strides, stride_s,
    read as build_streaming_runner reads them), so at commit the buffer holds about (K - 1) strides after the terminator frame. A cap cut
    commits at the cap instead, so `run_cascade` ends the box of an online cap-cut span at the cap of a buffer that starts δ before it."""
    boundary = inference_cfg["boundary_stability"]
    return float(boundary["delta_s"]), (int(boundary["hysteresis_strides"]) - 1) * float(inference_cfg["stride_s"])

def box_normalized_window(pose, start_s: float, end_s: float, box_start_s: float, box_end_s: float) -> tuple[np.ndarray, np.ndarray]:
    """The frames and timestamps of load_pose_window(pose, start_s, end_s), normalized with ONE Uni-Sign body box over the raw window
    [box_start_s, box_end_s) around them. Only the body box differs from a crop normalized alone."""
    raw, timestamps = load_pose_window(pose, start_s, end_s, normalize=False)
    if raw.shape[0] == 0 or raw.shape[1:] != (133, 3): return raw, timestamps   # as load_pose_window, which normalizes raw DWPose only
    box, box_ts = load_pose_window(pose, min(box_start_s, start_s), max(box_end_s, end_s), normalize=False)
    first = int(round((float(timestamps[0]) - float(box_ts[0])) * float(pose.fps)))
    return normalize_keypoints_unisign(box)[first:first + raw.shape[0]], timestamps

def decode_provenance(decoder: str, method_cfg: dict) -> dict:
    # A DLM's decode threshold as run (a fixed constant; an arm or a pretrimmed DLM translator), recorded so every payload names the decode
    # that made it. `decoder` is the row's translator decoder (translator_decoder).
    if decoder != "dlm": return {}
    spd_cfg = method_cfg["spd"]
    return {"tau_dec": float(spd_cfg["tau_dec"]),
            "spd_top_k": int(spd_cfg["top_k"]), "spd_renormalize": bool(spd_cfg["renormalize"])}

def generation_kwargs(method: str, inference_cfg: dict, method_cfg: dict, fps: list[float]) -> dict:
    # Main AR rows are greedy; only a DLM (an arm or a pretrimmed DLM translator) reads the block-decode params, which the AR generate
    # ignores. `fps`: 1 pose rate per row of the batch. The decode canvas is the training canvas of the method config
    # (max_text_tokens): a shorter one would truncate exactly the long units the arm was trained to emit, and an unreached slot
    # costs nothing, as the block decode stops once every row has emitted EOS.
    max_tokens = int(method_cfg["max_text_tokens"])
    spd_cfg = method_cfg["spd"]
    spd = {"tau_dec": float(spd_cfg["tau_dec"]), "spd_top_k": int(spd_cfg["top_k"]), "spd_renormalize": bool(spd_cfg["renormalize"])}
    if method == "baseline":
        num_beams = int(method_cfg["validation"]["num_beams"])
        return {"max_text_tokens": max_tokens, "num_beams": num_beams, **spd}

    return {
        "max_text_tokens": max_tokens, "num_beams": 1, **spd,
        # Membership gate in the boundary sweep: same Ω the decoder trained with (on-policy span, no GT/χ). BOTH arms gated — 
        # DLM injects Ω in its manual decode, AR via HF cross-attention hooks (front_end.ar_generate).
        "gate_enabled": bool(method_cfg["membership_gate"]["enabled"]),
        # Same first-eligible-unit rule as the training sampler and streaming runner.
        "gate_eps": float(method_cfg["membership_gate"]["eps"]),
        "gate_min_span_frames": [seconds_to_frames(lambda_min_s(inference_cfg), rate) for rate in fps],
    }

@torch.no_grad()
def translate_windows(
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
        for i in range(0, len(items), max(1, int(batch_size))): out.extend(translate_windows(
            model, tokenizer, method, items[i:i + max(1, int(batch_size))], device, inference_cfg, method_cfg, stream_start=stream_start,
            commit_frontier_s=commit_frontier_s[i:i + max(1, int(batch_size))] if commit_frontier_s else None,
            anchor_frames=anchor_frames[i:i + max(1, int(batch_size))] if anchor_frames else None,
        ))
        return out
    prepped = [prep_window(p, ts, start_s) for (p, ts, start_s, _) in items]
    max_t = max(int(p.shape[0]) for p, _, _ in prepped)
    
    # Same repeat-last-frame pad + frame-mask contract as the training collator (data.batch).
    poses = torch.stack([repeat_last_frame(p, max_t - int(p.shape[0])) for p, _, _ in prepped]).to(device)
    timestamps = torch.stack([torch.nn.functional.pad(ts, (0, max_t - int(ts.shape[0]))) for _, ts, _ in prepped]).to(device)
    frame_mask = torch.stack([torch.nn.functional.pad(m, (0, max_t - int(m.shape[0]))) for _, _, m in prepped]).to(device)

    gen_kwargs = generation_kwargs(method, inference_cfg, method_cfg, [fps for *_, fps in items])
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
    # but the boundary sweep must split "translation failed" from "policy is to skip", else the gate's refusal scores as garbage BLEU.
    skips = [bool(s) for s in gate_skip.tolist()]
    return list(zip(texts, [float(c) for c in confs], skips))


def build_streaming_runner(
    model, inference_cfg: dict, method_cfg: dict, translate: bool = True, cascade_model=None, 
    record_trace: bool = False, argmax: bool = False,
) -> StreamingSLTRunner:
    # argmax: the argmax decode of a standalone segmenter's row (eval.py --online --decode argmax, infer.duration_decode.argmax_tags);
    # else the legal path search under the model's duration_model (None: viterbi). An arm's head always takes the legal path.
    spd = method_cfg["spd"]
    boundary = inference_cfg["boundary_stability"]
    # The BIO head's reach, which the runner checks against each stream's fps (spec §1.4 rule 16): an S1 (the online cascade's
    # adapter holds it) or an arm, whose segmentation pose encoder has the translator's architecture. None: the Moryossef
    # segmenter, which has no RoPE BIO head. A RoPE BIO head with full attention (no band stamp) breaks rule 16: the FSM refuses it.
    seg = model.s1 if isinstance(model, S1RunnerAdapter) else model
    head, band_reach = getattr(seg, "bio_head", None), None
    if isinstance(head, RoPEBIOHead):
        if head.attention_radius_s is None: raise SystemExit(
            "The FSM's BIO head has full attention (its checkpoint stamps no bio_attention_radius_s), so its logits read the whole "
            "buffer ahead (spec §1.4 rule 16): retrain S1 (train-bio derives the band), then the arm."
        )
        from train.bio_pretrain import edge_reach
        band_reach = edge_reach(seg.pose_encoder if seg is not model else model.front_end.pose_encoder, head)

    return StreamingSLTRunner(
        model, stride_s=float(inference_cfg["stride_s"]),
        buffer_cap_s=float(inference_cfg["buffer_cap_s"]),
        delta_s=float(boundary["delta_s"]),
        hysteresis_strides=int(boundary["hysteresis_strides"]),
        min_span_s=lambda_min_s(inference_cfg),
        # Ω from the method config's membership_gate; χ from the runner's commit log.
        gate_enabled=bool(method_cfg["membership_gate"]["enabled"]),
        gate_eps=float(method_cfg["membership_gate"]["eps"]),
        max_text_tokens=int(method_cfg["max_text_tokens"]),   # the decode canvas, as in generation_kwargs
        tau_dec=float(spd["tau_dec"]),  # same source as generation_kwargs
        spd_top_k=int(spd["top_k"]), spd_renormalize=bool(spd["renormalize"]),
        translate=bool(translate), cascade_model=cascade_model, 
        record_trace=bool(record_trace), band_reach=band_reach, argmax=bool(argmax),
    )
