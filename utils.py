from __future__ import annotations
from pathlib import Path
import math, yaml, re
import torch

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
LAMBDA_MIN_S = 0.5  # label-domain floor that rejects flicker; genuine shorter units can't be emitted.


def seconds_to_frames(seconds: float, fps) -> int:
    """`seconds` as a frame count at `fps`, at least 1: the one conversion of every time constant (Λ_min, δ, the run rules).

    Every constant is in seconds. `fps` may be a list, 1 rate/row of a batch: rows must then give 2 count, because a batch 
    has 1 frame floor. The 0.01-frame slack absorbs loader's per-video fps (frames / a duration_s rounded to 1 ms, within 
    0.002 frame of 24 fps on SignVerse), so 0.5s at that rate stays 12 frames.
    """
    rates = fps if isinstance(fps, (list, tuple)) else [fps]
    counts = {max(1, math.ceil(float(seconds) * float(rate) - 0.01)) for rate in rates}
    if len(counts) != 1: raise ValueError(
        f"{seconds}s is {sorted(counts)} frames across the rows of 1 batch (fps {sorted({float(r) for r in rates})}); "
        f"a batch has 1 frame floor, so its rows must share a frame rate"
    )
    return counts.pop()


def lambda_min_s(inference_cfg: dict) -> float:
    # Lambda_min: the shortest span the FSM may commit. A LABEL-domain floor, so it does not move with a noise measurement.
    return float(inference_cfg["span_selection"]["min_span_s"])

def pick_device(preferred: str | None = None):
    # TF32 matmuls for the fp32 residue outside AMP autocast (Ampere+; no-op elsewhere). 
    # Set once, at the one chokepoint every entry point already routes through.
    torch.set_float32_matmul_precision("high")
    if preferred: return torch.device(preferred)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def cfg_get(cfg: dict, *path: str, default=None):
    cur = cfg
    for key in path: # Nested lookup: cfg_get(cfg, "checkpoint", "dir"); `default` if any level is missing.
        if not isinstance(cur, dict) or key not in cur or cur[key] is None: return default
        cur = cur[key]
    return cur

def pool_key(cfg: dict) -> str | None: # Accessors for the consolidated `checkpoint:` / `language_model:` config blocks.
    # Corpus key naming a multilingual segmentation-pretraining pool, or None for a monolingual run.
    langs = (cfg or {}).get("pretrain_languages") or None
    return f"multi_{'-'.join(sorted(str(x) for x in langs))}" if langs else None


def checkpoint_dir(cfg: dict, default: str | None = None) -> str | None:
    # Resolved `checkpoint.dir`. Shipped segmentation configs template it with `${corpus}` (load_yaml), so it is
    # already pool-correct here; the last-segment substitution below is the safety net for a `${language}`-templated
    # config that sets pretrain_languages — without it a pooled run would silently write to one language's dir.
    resolved = cfg_get(cfg, "checkpoint", "dir", default=default)
    key = pool_key(cfg)
    if not resolved or not key: return resolved
    parts = str(resolved).rstrip("/").split("/")
    langs = {str(x) for x in (cfg.get("pretrain_languages") or [])}
    if parts and (parts[-1] in langs or parts[-1] == str(cfg.get("language", ""))): parts[-1] = key
    return "/".join(parts)   # an explicit, non-language-templated dir is the caller's choice — never rewritten

def target_language(data_cfg: dict, language: str, default: str = "en_XX") -> str:
    # Declared TEXT language of a dataset language's captions (`data.yaml languages.<lang>.target_lang`).
    return str(((data_cfg.get("languages", {}) or {}).get(language, {}) or {}).get("target_lang") or default)

def _deep_merge(base: dict, override: dict) -> dict: # `override` wins; nested dicts merged recursively.
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict): out[key] = _deep_merge(out[key], value)
        else: out[key] = value
    return out


def resolve_pretrained(model_cfg: dict, data_cfg: dict, language: str, default: str | None = None) -> str | None:
    # Resolve the warm start: method override, then language config, then default.
    explicit = cfg_get(model_cfg, "checkpoint", "from_pretrained", default=None)
    if explicit: return explicit
    lang_ckpt = ((data_cfg.get("languages", {}) or {}).get(language, {}) or {}).get("pretrained_slt")
    return lang_ckpt or default


def resolve_placeholders(cfg: dict) -> dict:
    # Substitute `${key}` in string values from the config's own TOP-LEVEL scalar keys.
    scalars = {k: v for k, v in cfg.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
    if not scalars: return cfg

    def walk(obj):
        if isinstance(obj, dict): return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list): return [walk(v) for v in obj]
        if isinstance(obj, str): 
            return _PLACEHOLDER_RE.sub(lambda m: str(scalars[m.group(1)]) if m.group(1) in scalars else m.group(0), obj)
        return obj

    return walk(cfg)


def load_yaml(path: str | Path, language: str | None = None) -> dict:
    """Load a YAML config, resolving an optional `extends:` key and `${key}` placeholders.

    `extends` (path or list, relative to the child) is deep-merged under the child by `_load_yaml_raw`: `ar.yaml` inherits 
    the whole `dlm.yaml` recipe and overrides only the decoder + output dir, so AR-vs-DLM comparison isolates the decoder 
    alone. `${key}` then resolves from the merged config's own top-level scalars.

    `language` overrides the config's own `language:` BEFORE resolution, so one `--language asf` re-points BOTH the active 
    dataset AND every `${language}`-templated path without editing the shared configs.

    `${corpus}` names TRAINING CORPUS a checkpoint is function of: pool key on pooled segmentation run (`pretrain_languages` 
    set), else the language. Segmentation-trainer configs template checkpoint/wandb paths with it, so 1 file is correct for 
    the pooled and monolingual recipe without path rewriting in code. Derived before resolution, so it always agrees with 
    the run's actual `pretrain_languages`; with neither pool nor language, `${corpus}` stays literal and fails visibly.
    """
    merged = _load_yaml_raw(Path(path))
    if language is not None: merged["language"] = str(language)
    if "corpus" not in merged:
        corpus = pool_key(merged) or merged.get("language")
        if corpus is not None: merged["corpus"] = str(corpus)
    return resolve_placeholders(merged)


def _load_yaml_raw(path: str | Path) -> dict:
    # load_yaml without placeholder resolution (used for `extends` parents; the child resolves the merge).
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
        merged = _deep_merge(merged, _load_yaml_raw(parent_path))
    return _deep_merge(merged, cfg)
