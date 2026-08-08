from __future__ import annotations
from pathlib import Path
import os
import torch
import torch.nn as nn


def _load_state(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        found = None
        for name in ("pytorch_model.bin", "model.pt", "checkpoint.pt"):
            candidate = path / name
            if candidate.exists():
                found = candidate
                break
        if found is None: raise FileNotFoundError(f"No checkpoint file found in {path}")
        path = found
    if not path.exists(): raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict): state = state["model"]
    if not isinstance(state, dict): raise TypeError(f"Checkpoint at {path} did not contain a state dict")
    return state


def save_model_checkpoint(module: nn.Module, output_dir: str | Path, filename: str = "model.pt") -> Path:
    """torch.save with write-then-rename atomicity + fsync durability.

    A multi-GB `torch.save` straight onto the target is a data-loss hazard in exactly the situations that matter: 
    an interrupt DURING the write truncates the previous best checkpoint, and on network/FUSE mounts (Colab + Drive)
    the large write can sit unflushed in the page/FUSE cache while a tiny sibling file (best.json) syncs. Writing to 
    a temp file in the SAME directory, fsync'ing, then os.replace guarantees the target is always either the old 
    complete checkpoint or the new complete checkpoint, never a torso.
    """
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save({"model": module.state_dict()}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def load_model_checkpoint(module: nn.Module, checkpoint: str | Path, strict: bool = False) -> tuple[list[str], list[str]]:
    raw = _load_state(checkpoint)
    missing, unexpected = module.load_state_dict(raw, strict=strict)
    return list(missing), list(unexpected)
