from __future__ import annotations
from pathlib import Path
import os
import torch
import torch.nn as nn


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


def _load_state(path: str | Path) -> dict:
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
    scaler_state: dict | None, control_state: dict, epoch: int, epochs: int, step: int = 0, meta: dict | None = None,
) -> Path:
    """Full resumable snapshot (latest.pt): weights + optimizer moments + scheduler/scaler + best-tracking.

    Written every epoch so a preemption (Colab session death) loses at most one epoch. The best-model file
    (model.pt / best.json) is separate and unchanged — this file is operational state, never a deliverable.
    """
    import numpy as np, random
    state = {
        "epoch": int(epoch),
        # step > 0 = MID-epoch snapshot: that many batches of `epoch` are applied, resume re-enters the SAME epoch
        # and fast-forwards; 0 = epoch boundary (the default).
        "step": int(step),
        "epochs": int(epochs),  # schedule horizon: total_steps is baked into the scheduler state, so resume must match
        # The run's training-critical config, compared on resume: analysis stages rewrite those configs between
        # sessions, so without this a resumed run changes objective mid-training and nothing records it.
        "meta": dict(meta or {}),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state,
        "scaler": scaler_state,
        "control": control_state,
        "rng": { # Best-effort determinism: epoch-boundary RNG for the main process. Worker RNGs reseed per epoch anyway.
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    return _atomic_torch_save(state, Path(path))


def load_train_state(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    """Load a latest.pt snapshot into model+optimizer; returns the raw state for the caller to finish
    (scheduler/scaler/control/rng), since those objects live in the training loop."""
    import numpy as np, random
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    saved_groups, live_groups = len(state["optimizer"]["param_groups"]), len(optimizer.param_groups)
    if saved_groups != live_groups: raise SystemExit(
        f"--resume: {path} holds optimizer state for {saved_groups} param group(s) but this run builds {live_groups} "
        f"(the optimizer layout changed, e.g. the main/backbone_lr split); moments cannot be mapped across layouts. "
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
    file that later stages rewrite, so a value read from it at eval time is whatever the last stage wrote, not
    what this checkpoint trained under. Travelling with the weights is the only binding that cannot drift.
    """
    payload = {"model": module.state_dict()}
    if meta: payload["meta"] = dict(meta)
    return _atomic_torch_save(payload, Path(output_dir) / filename)


def s1_layout_state(state: dict) -> dict:
    """A stage-2 (MisalignedSLTModel) state dict re-keyed to the S1 layout (BioS1Model: pose_encoder.* + bio_head.*).

    The deployed FSM head is the ARM's head after joint training, so post-training FSM constants that are not training
    inputs (the commit lag) may be re-selected on it: tune-stream --checkpoint checkpoints/{ar,dlm}/<lang>/model.pt.
    LM and decoder tensors are dropped. An S1-layout dict passes through unchanged."""
    if not any(k.startswith("front_end.pose_encoder.") for k in state): return state
    out = {}
    for k, v in state.items():
        if k.startswith("front_end.pose_encoder."): out["pose_encoder." + k[len("front_end.pose_encoder."):]] = v
        elif k.startswith("bio_head."): out[k] = v
    return out


def load_model_checkpoint(module: nn.Module, checkpoint: str | Path, strict: bool = False) -> tuple[list[str], list[str]]:
    raw = _load_state(checkpoint)
    missing, unexpected = module.load_state_dict(raw, strict=strict)
    return list(missing), list(unexpected)


def load_checkpoint_meta(checkpoint: str | Path) -> dict:
    # `meta` written by save_model_checkpoint; {} for checkpoints saved before it existed.
    raw = torch.load(str(_resolve_checkpoint_file(checkpoint)), map_location="cpu", weights_only=False)
    return dict(raw.get("meta") or {}) if isinstance(raw, dict) else {}