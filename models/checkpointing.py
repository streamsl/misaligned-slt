from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn


def _load_state(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        found = None
        for name in ("visual_backbone.pt", "pytorch_model.bin", "model.pt", "checkpoint.pt"):
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


def _strip_prefix(state: dict, prefix: str) -> dict:
    out = {}
    needle = prefix + "."
    for key, value in state.items():
        if key.startswith(needle):
            out[key[len(needle) :]] = value
    return out


def save_visual_backbone(module: nn.Module, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "visual_backbone.pt"
    torch.save(module.state_dict(), path)
    return path


def save_model_checkpoint(module: nn.Module, output_dir: str | Path, filename: str = "model.pt") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    torch.save({"model": module.state_dict()}, path)
    return path


def load_visual_backbone(module: nn.Module, checkpoint: str | Path, strict: bool = False) -> tuple[list[str], list[str]]:
    raw = _load_state(checkpoint)
    candidates = [
        raw,
        _strip_prefix(raw, "visual"),
        _strip_prefix(raw, "base_module.visual"),
        _strip_prefix(raw, "model.visual"),
    ]
    target_keys = set(module.state_dict().keys())
    best_state = max(candidates, key=lambda state: len(target_keys.intersection(state.keys())))
    missing, unexpected = module.load_state_dict(best_state, strict=strict)
    return list(missing), list(unexpected)


def load_model_checkpoint(module: nn.Module, checkpoint: str | Path, strict: bool = False) -> tuple[list[str], list[str]]:
    raw = _load_state(checkpoint)
    missing, unexpected = module.load_state_dict(raw, strict=strict)
    return list(missing), list(unexpected)
