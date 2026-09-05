from __future__ import annotations
from pathlib import Path
import yaml, re
import torch

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
GEOMETRY_KEY_PATHS = (
    ("buffer_cap_s",), ("boundary_stability", "delta_enc_frames"), 
    ("span_selection", "min_span_frames"), ("boundary_stability", "commit_lag_s")
)
# Per-language rows with a valid code default: resolved when present, dropped when missing, so the stage  that writes them 
# can run first. commit_lag_s = 0.0 means "commit as soon as hysteresis passes".
SOFT_KEY_PATHS = frozenset({("boundary_stability", "commit_lag_s")})

def lambda_min_frames(inference_cfg: dict) -> int:
    # Lambda_min: span_selection.min_span_frames, else delta+1 (infer/stream.py's own derivation).
    ss = (inference_cfg or {}).get("span_selection") or {}
    bs = (inference_cfg or {}).get("boundary_stability") or {}
    return int(ss.get("min_span_frames", int(bs.get("delta_enc_frames", 3)) + 1))

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

def pretrained_checkpoint(cfg: dict, default: str | None = None) -> str | None:
    # Start weights: released Uni-Sign pose-only checkpoint (mBART ablation uses only its pose encoder; LM starts from base).
    return cfg_get(cfg, "checkpoint", "from_pretrained", default=default)

def save_best_enabled(cfg: dict, default: bool = True) -> bool:
    return bool(cfg_get(cfg, "checkpoint", "save_best", default=default))

def language_model_name(cfg: dict) -> str:
    # ONE key for the text model regardless of family: google/mt5-base OR facebook/mbart-large-cc25.
    return str(cfg_get(cfg, "language_model", "name", default="google/mt5-base"))

def target_language(data_cfg: dict, language: str, default: str = "en_XX") -> str:
    # Declared TEXT language of a dataset language's captions (`data.yaml languages.<lang>.target_lang`).
    return str(((data_cfg.get("languages", {}) or {}).get(language, {}) or {}).get("target_lang") or default)

def _deep_merge(base: dict, override: dict) -> dict: # `override` wins; nested dicts merged recursively.
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict): out[key] = _deep_merge(out[key], value)
        else: out[key] = value
    return out


def resolve_inference(cfg: dict, language: str, strict: bool = True) -> dict:
    """Resolve inference.yaml's PER-LANGUAGE measured geometry to flat scalars for one language.

    buffer_cap_s / boundary_stability.delta_enc_frames / span_selection.min_span_frames resolves ONCE, right after load; 
    downstream readers keep seeing plain scalars. A scalar value passes through unchanged (language-independent pins,
    synthetic test configs).
    """
    out = dict(cfg)
    for key_path in GEOMETRY_KEY_PATHS:
        parent, node = out, out
        for key in key_path[:-1]:
            if not isinstance(node.get(key), dict): node = None; break
            parent[key] = dict(node[key])   # copy-on-write down the path; never mutate the caller's dict
            parent, node = parent[key], parent[key]

        if node is None: continue
        leaf = node.get(key_path[-1])
        if not isinstance(leaf, dict): continue   # scalar or absent: already resolved / code defaults apply
        if str(language) not in {str(k) for k in leaf}:
            # strict=False: bootstrap/smoke mode — drop the unresolved leaf so flat `.get(..., default)`
            # fallbacks engage (delta-enc's FIRST pass on a new language has no Λ_min row yet, by construction).
            if not strict or key_path in SOFT_KEY_PATHS:
                node.pop(key_path[-1], None)
                continue
            writer = "--stage buffer-cap" if key_path == ("buffer_cap_s",) else "--stage delta-enc"
            raise SystemExit(
                f"inference config has no {'.'.join(key_path)} entry for language {language!r} (has: "
                f"{sorted(map(str, leaf))}). Run `analyze.py {writer} --language {language} --write-config` "
                f"first — borrowing another language's measured geometry would be silent miscalibration."
            )
        node[key_path[-1]] = {str(k): v for k, v in leaf.items()}[str(language)]
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


def update_yaml_scalar(path: str | Path, key_path: tuple[str, ...] | list[str], value) -> bool:
    """Replace one scalar in a YAML file in place, preserving layout and comments.

    Analysis persists the measured buffer cap / delta_enc into configs/inference.yaml. Line-targeted: walks the indentation 
    stack to `key_path`, rewriting only that value and keeping any inline comment.

    If the FINAL key is missing but its parent mapping exists, the key is INSERTED as a new child line — this is how a new 
    language gets its per-language geometry row without hand-editing the file. Parents are never created.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    target = tuple(str(k) for k in key_path)

    def walk(want: tuple) -> int | None:
        stack: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            match = re.match(r"^(\s*)([A-Za-z0-9_]+):(.*)$", line)
            if not match: continue
            indent = len(match.group(1))
            while stack and stack[-1][0] >= indent: stack.pop()
            stack.append((indent, match.group(2)))
            if tuple(key for _, key in stack) == want: return i
        return None

    i = walk(target)
    if i is not None:
        match = re.match(r"^(\s*)([A-Za-z0-9_]+):(.*)$", lines[i])
        rest = match.group(3)
        comment = f"  #{rest.split('#', 1)[1]}" if "#" in rest else ""
        lines[i] = f"{match.group(1)}{match.group(2)}: {value}{comment}"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True

    if len(target) >= 2:
        j = walk(target[:-1])
        if j is not None:
            parent_match = re.match(r"^(\s*)([A-Za-z0-9_]+):(.*)$", lines[j])
            rest = parent_match.group(3).split("#", 1)[0].strip()
            if rest: return False   # parent holds a scalar/flow value, not a block mapping — refuse to corrupt it
            child_indent = parent_match.group(1) + "    "
            lines.insert(j + 1, f"{child_indent}{target[-1]}: {value}")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return True
    return False
