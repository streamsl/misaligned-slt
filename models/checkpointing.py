from __future__ import annotations
from pathlib import Path
import os, pickle, random
import torch
import torch.nn as nn
import numpy as np

_ANNOTATION_KEYS = ("annotation_fingerprint", "language_fingerprints")   # label hashes: a change warns, never refuses

def without_annotation(value):
    # The duration prior stamps the hash of the labels it was fitted on; its fitted numbers still count as config.
    return {k: v for k, v in value.items() if k != "annotation_signature"} if isinstance(value, dict) else value

def _resolve_checkpoint_file(path: str | Path) -> Path:
    # A checkpoint argument may be a DIRECTORY (checkpoint_dir output); resolve it to the file inside.
    # Shared by the state and meta readers so no caller can hit IsADirectoryError on one but not the other.
    path = Path(path)
    if path.is_dir():
        for name in ("pytorch_model.bin", "model.pt", "checkpoint.pt"):
            if (path / name).exists(): return path / name
        raise FileNotFoundError(f"No checkpoint file found in {path}")
    if not path.exists(): raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return path

def load_state(path: str | Path) -> dict:
    path = _resolve_checkpoint_file(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict): state = state["model"]
    if not isinstance(state, dict): raise TypeError(f"Checkpoint at {path} did not contain a state dict")
    return state


def _atomic_torch_save(obj: dict, path: Path) -> Path:
    """torch.save with write-then-rename atomicity + fsync durability.

    A multi-GB `torch.save` straight onto the target truncates the previous checkpoint if interrupted, and
    can sit unflushed in the page cache. Temp file in the SAME directory + fsync + os.replace leaves the target
    as the old or the new complete checkpoint, never a torso.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # PID-unique temp name: a shared `model.pt.tmp` lets two writers of the same dir interleave into ONE
    # published file — exactly the corruption this prevents. Unique names degrade concurrency to
    # last-writer-wins over WHOLE files.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:  # BaseException, not Exception: KeyboardInterrupt is the motivating case
        tmp.unlink(missing_ok=True)  # never leave a multi-GB torso behind (Colab disk quota)
        raise

    try: # fsync the DIRECTORY so the rename is durable across a crash, not just the file's bytes.
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(dir_fd)
        finally: os.close(dir_fd)
    except OSError: pass  # filesystems that reject directory fsync: the rename is still atomic
    return path


def save_train_state(
    path: str | Path, *, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler_state: dict | None, 
    scaler_state: dict | None, control_state: dict, epoch: int, epochs: int, meta: dict | None = None,
) -> Path:
    """Full resumable snapshot (latest.pt): weights + optimizer moments + scheduler/scaler + best-tracking.

    Written every epoch so a preemption (Colab session death) loses at most one epoch. The best-model file
    (model.pt / best.json) is separate and unchanged — this file is operational state, never a deliverable.
    """
    state = {
        "epoch": int(epoch),
        "epochs": int(epochs),  # schedule horizon: total_steps is baked into the scheduler state, so resume must match
        # The run's training-critical config, compared on resume: configs are live files that can change between
        # sessions, so without this a resumed run changes objective mid-training and nothing records it.
        "meta": dict(meta or {}),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state,
        "scaler": scaler_state,
        "control": control_state,
        "rng": { # Best-effort determinism: epoch-boundary RNG for the main process. Worker RNGs reseed per epoch anyway.
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()
        },
    }
    return _atomic_torch_save(state, Path(path))


def _refuse_meta_drift(path, saved: dict, expected: dict) -> None:
    """Refuse a resume whose training-critical config moved. The configs (inference.yaml, the stage config) are 
    live files that parameterize the run: resuming across a change trains 2 halves under different objectives, 
    visible afterwards only as an unexplained break in the loss curve. A changed annotation fingerprint only warns."""
    keys = set(saved) | set(expected)
    labels = sorted(k for k in keys if k in _ANNOTATION_KEYS and saved.get(k) != expected.get(k))
    if labels: print(
        f"WARNING: --resume across changed annotations ({', '.join(labels)}): {path} trained on other labels.", flush=True
    )
    drift = sorted(
        k for k in keys - set(_ANNOTATION_KEYS) 
        if without_annotation(saved.get(k)) != without_annotation(expected.get(k))
    )
    if not drift: return
    raise SystemExit(
        f"--resume: this run started under different training-critical config; {', '.join(drift)} changed "
        + "; ".join(f"{k}: {saved.get(k)!r} -> {expected.get(k)!r}" for k in drift[:4])
        + f". Restore those values to resume, or start a fresh run (move {path}) "
        f"— resuming across the change trains 2 halves under different objectives."
    )


def load_train_state(
    path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, expected_meta: dict | None = None,
) -> dict:
    """Load a latest.pt snapshot into model+optimizer; returns the raw state for the caller to finish
    (scheduler/scaler/control/rng), since those objects live in the training loop.

    `expected_meta` is compared BEFORE the weights are loaded. A config edit that changes which parameters exist
    makes `load_state_dict` raise a raw shape/key error, and a guard placed after it can never report the cause.
    Drift that does not change the layout (a loss weight, a rate) has no other detector at all."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    state["meta"] = dict(state.get("meta") or {})
    if expected_meta: _refuse_meta_drift(path, state["meta"], expected_meta)
    model.load_state_dict(state["model"])
    saved_groups, live_groups = len(state["optimizer"]["param_groups"]), len(optimizer.param_groups)
    if saved_groups != live_groups: raise SystemExit(
        f"--resume: {path} holds optimizer state for {saved_groups} param group(s) but this run builds "
        f"{live_groups} (the optimizer layout changed); moments cannot be mapped across layouts. "
        f"Start a fresh run, or warm-start from model.pt."
    )
    optimizer.load_state_dict(state["optimizer"])
    rng = state.get("rng") or {}
    if rng.get("torch") is not None: torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        try: torch.cuda.set_rng_state_all(rng["cuda"])
        except RuntimeError: pass  # resumed on a machine with a different GPU count
    if rng.get("numpy") is not None: np.random.set_state(rng["numpy"])
    if rng.get("python") is not None: random.setstate(rng["python"])
    return state


def save_model_checkpoint(
    module: nn.Module, output_dir: str | Path, filename: str = "model.pt", meta: dict | None = None,
) -> Path:
    """Weights, plus optional `meta` describing the CONTEXT the weights were trained under.

    Anything eval must reproduce but cannot re-derive belongs here rather than in a config: a config is a live
    file that can change after training, so a value read from it at eval time need not be what this checkpoint
    trained under. Travelling with the weights is the only binding that cannot drift.
    """
    payload = {"model": module.state_dict()}
    if meta: payload["meta"] = dict(meta)
    return _atomic_torch_save(payload, Path(output_dir) / filename)


def s1_layout_state(state: dict) -> dict:
    """A stage-2 (MisalignedSLTModel) state dict re-keyed to the S1 layout (BioS1Model: pose_encoder.* + bio_head.*).

    The deployed FSM head is the ARM's segmentation branch after joint training (`bio_pose_encoder` + `bio_head`), so joint 
    head can be scored alone: eval.py --segmenter-eval --segmenter-arch s1 --checkpoint checkpoints/{ar,dlm}/<lang>/model.pt.
    A one-stream arm (no `bio_pose_encoder`) reads the shared `front_end.pose_encoder`. The LM and decoder tensors (and a 
    two-stream arm's translator encoder) are dropped. An S1-layout dict passes through unchanged. An arm with `bio_adapter` 
    is refused: its head also reads the translator's features, which the S1 layout has no place for, so the head alone is 
    not the deployed head (score it with `eval.py --online --no-translate`).
    """
    if any(k.startswith("bio_adapter.") for k in state): raise ValueError(
        "This arm's BIO head reads the translator's features via bio_adapter, so it can't be scored in the S1 layout. "
        "Score its segmentation with eval.py --online --no-translate (or --offline) instead."
    )
    if any(k.startswith("bio_pose_encoder.") for k in state): encoder = "bio_pose_encoder."
    elif any(k.startswith("bio_head.") for k in state) and any(k.startswith("front_end.pose_encoder.") for k in state):
        encoder = "front_end.pose_encoder."
    else: return state
    out = {}
    for k, v in state.items():
        if k.startswith(encoder): out["pose_encoder." + k[len(encoder):]] = v
        elif k.startswith("bio_head."): out[k] = v
    return out


def load_model_checkpoint(module: nn.Module, checkpoint: str | Path, strict: bool = False) -> tuple[list[str], list[str]]:
    raw = load_state(checkpoint)
    missing, unexpected = module.load_state_dict(raw, strict=strict)
    return list(missing), list(unexpected)


def load_checkpoint_meta(checkpoint: str | Path) -> dict:
    # `meta` written by save_model_checkpoint; {} when the file carries none. mmap: the tensors are not read into memory.
    # weights_only first, so reading the third-party released Uni-Sign file never runs pickled code (it loads this way).
    # Only our own training-state files (latest.pt: optimizer and RNG state) need full unpickler, as load_train_state does.
    path = str(_resolve_checkpoint_file(checkpoint))
    try: raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except pickle.UnpicklingError: raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return dict(raw.get("meta") or {}) if isinstance(raw, dict) else {}


def warn_pose_geometry(meta: dict, source) -> bool:
    """Warn when a trained Uni-Sign pose model (S1, a clean floor, an arm) read its poses, itself or through a warm start, in another
    frame geometry than poses.pose_io.load_pose_frames returns now (a file trained before the reader mapped x to a 16:9 frame stamps
    none; train.slt and train.bio_pretrain call it on a trained warm start at the start of a run). A file with no meta (the released
    Uni-Sign checkpoint) is silent. A warning, not a stop, like every label check: retrain before a run you report.
    False when it warned."""
    from poses.pose_io import POSE_GEOMETRY   # lazy, as in require_fixed_constants below
    stale = bool(meta) and meta.get("pose_geometry") != POSE_GEOMETRY
    if stale: print(
        f"WARNING: {source} stamps pose_geometry={meta.get('pose_geometry')!r}, not {POSE_GEOMETRY!r}: it, or a warm start it trained "
        f"from, read poses in another frame geometry (poses.pose_io.load_pose_frames). Retrain it before a run you report.", flush=True
    )
    return not stale


def require_fixed_constants(meta: dict, inference_cfg: dict, source) -> None:
    """Refuse a trained checkpoint whose labels or FSM constants are not the current ones. inference.yaml holds ONE design
    value per constant (buffer cap, δ, Λ_min) and train-slt stamps the values it trained with, so a stamp that differs means
    the file trained under other constants. Eval (every model build) and the stage-2 warm-start guard both call this, so they
    refuse the same files. A file with no meta (the released Uni-Sign checkpoint) passes."""
    if not meta: return
    from data.loader import ANNOTATION_PROTOCOL   # lazy: data.loader is heavy, and this module is imported early
    from utils import lambda_min_s
    if meta.get("annotation_protocol") != ANNOTATION_PROTOCOL: raise SystemExit(
        f"{source} stamps annotation_protocol={meta.get('annotation_protocol')!r}, not {ANNOTATION_PROTOCOL!r}, "
        f"so it predates the current labels. Retrain it."
    )
    gate = meta.get("gate") or {}
    # δ and Λ_min are seconds; a frame count is a time only at 1 fps, so a file that stamps frames is refused.
    if frame_keys := sorted({"delta", "min_span_frames"} & set(gate)): raise SystemExit(
        f"{source} stamps {'/'.join(frame_keys)} in frames; δ and Λ_min are seconds "
        f"(inference.yaml delta_s, min_span_s). Retrain it under the current constants."
    )
    live = {
        "buffer_cap_s": inference_cfg["buffer_cap_s"],
        "delta_s": inference_cfg["boundary_stability"]["delta_s"],
        "min_span_s": lambda_min_s(inference_cfg)
    }
    stamped = { # δ is a training parameter only of a gated checkpoint; a clean floor stamps it but never reads it.
        "buffer_cap_s": meta.get("buffer_cap_s"), 
        "delta_s": gate.get("delta_s") if gate.get("enabled") else None,
        "min_span_s": gate.get("min_span_s")
    }
    drift = [
        f"{k} {stamped[k]} (inference.yaml {live[k]})" for k in live
        if stamped[k] is not None and live[k] is not None and abs(float(stamped[k]) - float(live[k])) > 1e-6
    ]
    if drift: raise SystemExit(
        f"{source} was trained with other FSM constants than configs/inference.yaml: "
        f"{'; '.join(drift)}. Retrain it under the fixed constants."
    )
