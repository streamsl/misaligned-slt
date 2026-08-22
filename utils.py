from __future__ import annotations
from pathlib import Path
import yaml, re
import torch

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")

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

# Accessors for the consolidated `checkpoint:` / `language_model:` config blocks (§12).
def checkpoint_dir(cfg: dict, default: str | None = None) -> str | None:
    return cfg_get(cfg, "checkpoint", "dir", default=default)

def pretrained_checkpoint(cfg: dict, default: str | None = None) -> str | None:
    # Start weights: released Uni-Sign pose-only checkpoint (mBART ablation uses only its pose encoder; LM starts from base).
    return cfg_get(cfg, "checkpoint", "from_pretrained", default=default)

def save_best_enabled(cfg: dict, default: bool = True) -> bool:
    return bool(cfg_get(cfg, "checkpoint", "save_best", default=default))

def language_model_name(cfg: dict) -> str:
    # ONE key for the text model regardless of family: google/mt5-base OR facebook/mbart-large-cc25.
    return str(cfg_get(cfg, "language_model", "name", default="google/mt5-base"))

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

    def sub(s: str) -> str:
        return _PLACEHOLDER_RE.sub(lambda m: str(scalars[m.group(1)]) if m.group(1) in scalars else m.group(0), s)

    def walk(obj):
        if isinstance(obj, dict): return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list): return [walk(v) for v in obj]
        return sub(obj) if isinstance(obj, str) else obj

    return walk(cfg)


def load_yaml(path: str | Path, language: str | None = None) -> dict:
    """Load a YAML config, resolving an optional `extends:` key and `${key}` placeholders.

    `extends` (path or list, relative to the child) is deep-merged under the child by `_load_yaml_raw`: `ar.yaml`
    inherits the whole `dlm.yaml` recipe and overrides only the decoder + output dir, so the AR-vs-DLM comparison
    isolates the decoder alone. `${key}` then resolves from the merged config's own top-level scalars.

    `language` overrides the config's own `language:` BEFORE resolution, so one `--language asf` re-points BOTH
    the active dataset AND every `${language}`-templated path without editing the shared configs.
    """
    merged = _load_yaml_raw(Path(path))
    if language is not None: merged["language"] = str(language)
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


def update_yaml_scalar(path: str | Path, key_path: tuple[str, ...] | list[str], value) -> bool:
    """Replace one scalar in a YAML file in place, preserving layout and comments.

    Analysis persists the measured buffer cap / delta_enc into configs/inference.yaml (the spec requires 
    freezing the constant). Line-targeted: walks the indentation stack to `key_path`, rewriting only that 
    value and keeping any inline comment.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    target = tuple(key_path)
    stack: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        match = re.match(r"^(\s*)([A-Za-z0-9_]+):(.*)$", line)
        if not match: continue

        indent = len(match.group(1))
        while stack and stack[-1][0] >= indent: stack.pop()
        stack.append((indent, match.group(2)))
        if tuple(key for _, key in stack) != target: continue

        rest = match.group(3)
        comment = f"  #{rest.split('#', 1)[1]}" if "#" in rest else ""
        lines[i] = f"{match.group(1)}{match.group(2)}: {value}{comment}"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True
    return False
