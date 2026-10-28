from __future__ import annotations
from pathlib import Path
import math, yaml, re
import torch

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
# The keys of the `bio:` block (the BIO head and its loss, configs/dlm.yaml); load_yaml refuses any other key there or at the top level.
BIO_CONFIG_KEYS = frozenset({"hidden_dim", "depth", "nhead", "dropout", "conv_stem_layers", "adapter", "class_weights", "dice_loss_weight"})
# Joins the texts of caption units that follow each other with gaps of at most δ into 1 text target 
# ("A <sep> B"). An unused mT5 sentinel: the tokenizer keeps it as 1 piece, and decoding keeps it in 
# the text, where `infer.stream.join_unit_text` reads it as a space.
UNIT_SEPARATOR = "<extra_id_0>"

# A time within this many frames of a frame time is that frame: it absorbs the float noise of a seconds round trip
# (a frame time saved as seconds, a per-video fps from a duration rounded to 1 ms).
FRAME_SLACK = 0.01

# Artifact folders under outputs/, one per kind: the offline and online rows (each events file with its scores), 
# segmenter files (dev/test evaluation and span files), analyses (segmenter error types, boundary sweeps), the display 
# replays of --stability (policy scores and provisional texts) and report.py's reports and debug views.
OUTPUT_KINDS = ("offline", "online", "segmenter", "analysis", "stability", "report")

def seconds_to_frames(seconds: float, fps: float) -> int:
    # `seconds` as a frame count at `fps`, at least 1: the one conversion of every time constant (Λ_min, δ, the run rules).
    return max(1, math.ceil(float(seconds) * float(fps) - FRAME_SLACK))

def lambda_min_s(inference_cfg: dict) -> float:
    # Lambda_min: the shortest span the FSM may commit, its one source. A LABEL-domain floor that rejects flicker, 
    # so it does not move with a noise measurement; genuine shorter units can't be emitted.
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
    stale = [k for k in ("pretrain_languages", "pretrain_temperature", "pretrain_geometry") if k in (cfg or {})]
    if stale: raise SystemExit(
        f"config keys {stale} are no longer read: put them in the `pretrain:` block (languages, temperature), "
        f"as configs/bio_pretrain.yaml does; else the run would be monolingual with no error"
    )
    langs = cfg_get(cfg or {}, "pretrain", "languages")
    return f"multi_{'-'.join(sorted(str(x) for x in langs))}" if langs else None

def checkpoint_dir(cfg: dict, default: str | None = None) -> str | None:
    # Resolved `checkpoint.dir`. Shipped segmentation configs template it with `${corpus}` (load_yaml), so it is
    # already pool-correct here; the last-segment substitution below is the safety net for a `${language}`-templated
    # config that sets pretrain.languages — without it a pooled run would silently write to one language's dir.
    resolved = cfg_get(cfg, "checkpoint", "dir", default=default)
    key = pool_key(cfg)
    if not resolved or not key: return resolved
    parts = str(resolved).rstrip("/").split("/")
    langs = {str(x) for x in (cfg_get(cfg, "pretrain", "languages") or [])}
    if parts and (parts[-1] in langs or parts[-1] == str(cfg.get("language", ""))): parts[-1] = key
    return "/".join(parts)   # an explicit, non-language-templated dir is the caller's choice — never rewritten

def output_dir(kind: str) -> Path:
    # The only resolver of artifact paths: outputs/<kind>/, created on first use (a fresh clone or VM has no outputs/).
    if kind not in OUTPUT_KINDS: raise ValueError(f"unknown output kind {kind!r}; known: {', '.join(OUTPUT_KINDS)}")
    path = Path("outputs") / kind
    path.mkdir(parents=True, exist_ok=True)
    return path

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
    # Resolve the warm start. The clean floor's own run (a method config with checkpoint.from_pretrained) starts from the language's
    # own released translator (data.yaml released_slt) when it has one, else from that file. Any other run reads the language's
    # data.yaml pretrained_slt entry of the config's decoder (each decoder has its own clean floor), then default.
    corpus = (data_cfg.get("languages", {}) or {}).get(language, {}) or {}
    explicit = cfg_get(model_cfg, "checkpoint", "from_pretrained", default=None)
    if explicit: return corpus.get("released_slt") or explicit
    entry = corpus.get("pretrained_slt")
    if entry is None: return default
    if not isinstance(entry, dict): raise SystemExit(
        f"data.yaml languages.{language}.pretrained_slt must map each decoder to its warm start, "
        f"{{ar: <path>, dlm: <path>}} (README B2b), but it is {entry!r}."
    )
    return entry.get(str(model_cfg["decoder"])) or default

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

    `${corpus}` names TRAINING CORPUS a checkpoint is function of: pool key on pooled segmentation run (`pretrain.languages` 
    set), else the language. Segmentation-trainer configs template checkpoint/wandb paths with it, so 1 file is correct for 
    the pooled and monolingual recipe without path rewriting in code. Derived before resolution, so it always agrees with 
    the run's actual `pretrain.languages`. With neither pool nor language, `${corpus}` stays literal and fails visibly.
    """
    merged = _load_yaml_raw(Path(path))
    # A BIO key outside the `bio:` block, or an unknown one in it, is read by no stage: the run would keep the block's values with no error.
    stale = sorted(k for k in merged if k.startswith("bio_") or k == "dice_loss_weight")
    unknown = sorted(set(merged.get("bio") or {}) - BIO_CONFIG_KEYS)
    if stale or unknown: raise SystemExit(
        f"{path}: config keys {stale + [f'bio.{k}' for k in unknown]} are not read: the BIO keys are {sorted(BIO_CONFIG_KEYS)} in the "
        f"`bio:` block, as configs/dlm.yaml sets them"
    )
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
