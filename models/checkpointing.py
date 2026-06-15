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


def load_visual_backbone(
    module: nn.Module, checkpoint: str | Path, 
    strict: bool = False, preserve_decoder_io: bool = False,
) -> tuple[list[str], list[str]]:
    raw = _load_state(checkpoint)
    candidates = [raw, _strip_prefix(raw, "visual"), _strip_prefix(raw, "base_module.visual"), _strip_prefix(raw, "model.visual")]

    # GFSLT-VLP downstream transfer (`train_slt.py`) loads VLP visual/decoder layers but restores decoder token + position embeddings 
    # from the base transformer checkpoint, and it never transfers stage-1 Text_Decoder lm_head. Because this project saves 1 shared 
    # mBART, all tied token/output aliases are dropped for downstream baseline/stage-2 loading when preserve_decoder_io=True.
    if preserve_decoder_io: candidates = [{key: value for key, value in state.items() if key not in {
        "mbart.model.shared.weight",
        "mbart.model.encoder.embed_tokens.weight",
        "mbart.model.decoder.embed_tokens.weight",
        "mbart.model.decoder.embed_positions.weight",
        "mbart.lm_head.weight",
        "mbart.final_logits_bias",
    }} for state in candidates]
    
    target_keys = set(module.state_dict().keys())
    best_state = max(candidates, key=lambda state: len(target_keys.intersection(state.keys())))
    missing, unexpected = module.load_state_dict(best_state, strict=strict)
    return list(missing), list(unexpected)


def load_visual_backbone_checked(
    module: nn.Module, checkpoint: str | Path, 
    name: str = "model", preserve_decoder_io: bool = False,
) -> bool:
    """Load the VLP visual backbone and LOUDLY report how much actually loaded.

    The call sites used to swallow FileNotFoundError silently, so a missing/misnamed VLP checkpoint — or a
    key-prefix mismatch that matches almost nothing — left the visual backbone RANDOMLY INITIALISED with no
    signal at all. That is a prime suspect for weak features / overfitting / low BLEU. Returns True iff a
    substantial fraction of the target params were populated.
    """
    target = len(module.state_dict())
    try: missing, unexpected = load_visual_backbone(module, checkpoint, strict=False, preserve_decoder_io=preserve_decoder_io)
    except (FileNotFoundError, OSError) as exc:
        print(f"{name} | WARNING: VLP checkpoint NOT loaded ({type(exc).__name__}: {exc}); the visual backbone is "
              f"RANDOMLY INITIALISED — expect weak features and low BLEU. Path: {checkpoint}", flush=True)
        return False
        
    loaded = target - len(missing)
    if loaded <= max(1, target // 2):
        print(f"{name} | WARNING: VLP load matched only {loaded}/{target} params (key-prefix mismatch?); most of the "
              f"visual backbone is RANDOM. unexpected={len(unexpected)}", flush=True)
        return False
        
    reset_note = " reset_decoder_io=True" if preserve_decoder_io else ""
    print(f"{name} | VLP backbone loaded: {loaded}/{target} params "
          f"(missing={len(missing)}, unexpected={len(unexpected)}{reset_note}).", flush=True)
    return True


def load_model_checkpoint(module: nn.Module, checkpoint: str | Path, strict: bool = False) -> tuple[list[str], list[str]]:
    raw = _load_state(checkpoint)
    missing, unexpected = module.load_state_dict(raw, strict=strict)
    return list(missing), list(unexpected)
