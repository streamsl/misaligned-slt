from __future__ import annotations
from pathlib import Path
import yaml, re

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")

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
    # trim_mbart writes one directory holding the trimmed tokenizer and the (depth-trimmed)
    # model used by every stage — text encoder, visual encoder, and AR/DLM decoder all load it.
    fallback = cfg.get("trimmed_mbart_dir", cfg.get("trimmed_tokenizer_dir", mbart_name(cfg)))
    return str(cfg_get(cfg, "mbart", "trimmed_dir", default=fallback))

def backbone_name(cfg: dict) -> str:
    # Pose backbone selector from the stage config's `backbone:` block (the architecture source of truth).
    return str(cfg_get(cfg, "backbone", "name", default="cosign"))

def _deep_merge(base: dict, override: dict) -> dict: # Recursively merge `override` onto `base` (override wins; nested dicts merged).
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict): out[key] = _deep_merge(out[key], value)
        else: out[key] = value
    return out


def resolve_placeholders(cfg: dict) -> dict:
    """Substitute `${key}` in string values using the config's own TOP-LEVEL scalar keys.

    Makes configs dataset-agnostic: with `language: phoenix`, every `checkpoints/vlp/${language}`, `stage1_vlp_${language}`, 
    `outputs/a_mode_ratios_${language}.json` etc. resolves to the phoenix path, and switching to another corpus is a one-line `language:` 
    change (no other dataset to template here — roots/target_lang live per-entry in data.yaml). Unknown placeholders are left untouched 
    (no key, no substitution), so this is a no-op for any config that uses none.
    """
    scalars = {k: v for k, v in cfg.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
    if not scalars: return cfg

    def sub(s: str) -> str:
        return _PLACEHOLDER_RE.sub(lambda m: str(scalars[m.group(1)]) if m.group(1) in scalars else m.group(0), s)

    def walk(obj):
        if isinstance(obj, dict): return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list): return [walk(v) for v in obj]
        return sub(obj) if isinstance(obj, str) else obj

    return walk(cfg)


def load_yaml(path: str | Path) -> dict:
    """Load a YAML config, resolving an optional `extends:` key and `${key}` placeholders.

    `extends` may be a path (or list of paths), relative to the child file, to a parent config that is loaded first and deep-merged 
    under the child. This lets e.g. `stage2_ar.yaml` inherit the entire `stage2_dlm.yaml` recipe (sampler, Analysis-A ratios/jitter, 
    confidence-bound, optimization) and override only the decoder + output dir — so the AR-vs-DLM comparison isolates the decoder alone.
    `${key}` placeholders are then resolved from the merged config's own top-level scalars (see resolve_placeholders) — chiefly `${language}`.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    extends = cfg.pop("extends", None)
    if not extends: return resolve_placeholders(cfg)
    if isinstance(extends, str): extends = [extends]
    merged: dict = {}
    for parent in extends:
        parent_path = Path(parent)
        if not parent_path.is_absolute(): parent_path = path.parent / parent_path
        # Parents are loaded raw (placeholder resolution deferred to the merged child, so the child's
        # `language` governs inherited templated paths — e.g. stage2_ar inherits stage2_dlm's templates).
        merged = _deep_merge(merged, _load_yaml_raw(parent_path))
    return resolve_placeholders(_deep_merge(merged, cfg))


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

    Used by Analysis to write the measured buffer cap / delta_enc into configs/inference.yaml (the spec requires the analysis 
    to persist the frozen constant). Line-targeted: walks the indentation stack to find `key_path` (e.g. ("buffer_cap_s",) or 
    ("boundary_stability", "delta_enc_frames")) and rewrites only that line's value, keeping any inline comment.
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