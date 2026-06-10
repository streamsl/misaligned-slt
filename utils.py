from __future__ import annotations
from pathlib import Path
import yaml

def cfg_get(cfg: dict, *path: str, default=None):
    cur = cfg
    for key in path: # Nested lookup: cfg_get(cfg, "checkpoint", "dir"). Returns `default` if any level is missing.
        if not isinstance(cur, dict) or key not in cur or cur[key] is None: return default
        cur = cur[key]
    return cur

# Single source of truth for the consolidated `checkpoint:` / `mbart:` config blocks (§12).
# Each reader falls back to the pre-consolidation flat keys so older configs keep working.
def checkpoint_dir(cfg: dict, default: str | None = None) -> str | None:
    return cfg_get(cfg, "checkpoint", "dir", default=cfg.get("output_dir", default))

def vlp_checkpoint(cfg: dict, default: str | None = None) -> str | None:
    return cfg_get(cfg, "checkpoint", "from_vlp", default=cfg.get("checkpoint_vlp", default))

def save_best_enabled(cfg: dict, default: bool = True) -> bool:
    return bool(cfg_get(cfg, "checkpoint", "save_best", default=default))

def mbart_name(cfg: dict) -> str:
    return str(cfg_get(cfg, "mbart", "name", default=cfg.get("mbart_name", "facebook/mbart-large-cc25")))

def mbart_trimmed_dir(cfg: dict) -> str:
    # trim_mbart writes one directory holding both the trimmed model and tokenizer.
    fallback = cfg.get("trimmed_mbart_dir", cfg.get("trimmed_tokenizer_dir", mbart_name(cfg)))
    return str(cfg_get(cfg, "mbart", "trimmed_dir", default=fallback))

def _deep_merge(base: dict, override: dict) -> dict: # Recursively merge `override` onto `base` (override wins; nested dicts merged).
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict): out[key] = _deep_merge(out[key], value)
        else: out[key] = value
    return out

def load_yaml(path: str | Path) -> dict:
    """Load a YAML config, resolving an optional `extends:` key.

    `extends` may be a path (or list of paths), relative to the child file, to a parent config that is loaded first and deep-merged under the child. 
    This lets e.g. `stage2_ar.yaml` inherit the entire `stage2_dlm.yaml` recipe (sampler, Analysis-A ratios/jitter, confidence-bound, optimization) 
    and override only the decoder + output dir — so the AR-vs-DLM comparison isolates the decoder alone.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    extends = cfg.pop("extends", None)
    if not extends: return cfg
    if isinstance(extends, str): extends = [extends]
    merged: dict = {}
    for parent in extends:
        parent_path = Path(parent)
        if not parent_path.is_absolute(): parent_path = path.parent / parent_path
        merged = _deep_merge(merged, load_yaml(parent_path))
    return _deep_merge(merged, cfg)
