from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal
from pathlib import Path
import copy, json, time
import torch


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _fmt_metric(value) -> str:
    # Fixed-shape number for table columns: 3 decimals in the normal range, 3-sig sci at the
    # extremes (tiny LRs, huge sums). Within a column the metric magnitude is stable, so the
    # chosen form is consistent down the column → decimal points line up.
    try: v = float(value)
    except (TypeError, ValueError): return str(value)
    if v != v: return "nan"  # NaN
    if v == 0: return "0.000"
    av = abs(v)
    if av >= 1e5 or av < 1e-3: return f"{v:.3e}"
    return f"{v:.3f}"


# Short column headers for the step table (full names stay on W&B). Falls back to stripping `_loss`.
_HEADER_ABBREV = {
    "total_loss": "total", "bio_loss": "bio", "translation_loss": "transl",
    "confidence_bound_loss": "cb", "oput_loss": "oput", "baseline_ce_loss": "ce",
    "vlp_loss": "vlp", "contrastive_loss": "contr", "cmlm_loss": "cmlm", "phrase_bio_loss": "bio",
}
def _short_header(key: str) -> str:
    if key in _HEADER_ABBREV: return _HEADER_ABBREV[key]
    return key[:-5] if key.endswith("_loss") else key


# Step-line keys hidden from the console when no explicit console_keys list is given.
# These are per-mode diagnostics that belong on W&B, not in the terminal.
_CONSOLE_HIDDEN_SUBSTRINGS = ("_count", "masked_fraction", "mask_loss", "pred_loss", "cb_active")
_NUM_COL_W = 10   # numeric column width: holds "2.800e-05" / "-12.300" / "0.000"
_VAL_COL_W = 11   # value column width for the epoch key/value summary table
SchedulerInterval = Literal["step", "epoch", "none"]


class TrainLogger:
    """Unified console + Weights & Biases logger for the training loops.

    Console: a per-epoch banner, then a **fixed-width table** — a header + separator printed once when
    the epoch starts, then one right-aligned row every `log_every` steps. Columns are `step`, the headline
    metrics (`console_keys` argument or `console_keys:` in the config; everything else stays on W&B), `lr`,
    `s/it`, and an epoch `eta`. Numbers use a consistent per-column form (3 decimals normally, 3-sig sci at
    the extremes) so decimal points line up down each column. Epoch summaries ([train]/[val]) print as an
    aligned key/value table. ASCII only (Windows consoles choke on box-drawing characters).

    W&B: every step (and per epoch) under a `<stage>/` namespace, with ALL numeric keys. Enable via the
    config `wandb:` block (`enabled`, `project`, `name`, `group`, `mode`). Login uses wandb's own
    interactive console flow: when no API key is configured, `wandb.init` prompts in the terminal
    (create account / paste existing API key via browser / don't visualize) instead of silently
    degrading. Only a missing wandb *installation* degrades to console-only.
    """
    def __init__(
        self, stage: str, cfg: dict | None = None, epochs: int = 0, steps_per_epoch: int = 0, 
        log_every: int | None = None, console_keys: list[str] | None = None,
    ):
        self.stage = str(stage)
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        cfg = cfg or {}
        wandb_cfg = dict(cfg.get("wandb", {}) or {})

        self.log_every = int(log_every if log_every is not None else cfg.get("log_every", wandb_cfg.get("log_every", 10)) or 1)
        self.console_keys = [str(k) for k in (console_keys or cfg.get("console_keys") or [])]
        self._global_step = 0
        self._epoch: int | None = None
        self._epoch_t0 = 0.0
        self._epoch_step0 = 0

        self._step_keys: list[str] = []
        self._step_specs_cache: list[tuple[str, str, int]] = []
        self._wandb = None
        
        if bool(wandb_cfg.get("enabled", False)):
            try: import wandb
            except ImportError:
                print(f"{self.stage} | wandb enabled in config but not installed (pip install wandb); console only", flush=True)
            else:
                if wandb.run is None: wandb.init(
                    project=str(wandb_cfg.get("project", "misaligned-slt")), name=str(wandb_cfg.get("name") or self.stage),
                    group=wandb_cfg.get("group"), mode=str(wandb_cfg.get("mode", "online")), config=cfg,
                )
                self._wandb = wandb

    @staticmethod
    def _numeric(row: dict) -> dict:
        return {k: v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

    def _resolve_step_keys(self, numeric: dict) -> list[str]:
        # Freeze the step table's metric columns at the start of each epoch (stable headers).
        skip = {"epoch", "step", "lr"}
        if self.console_keys: return [k for k in self.console_keys if k not in skip]
        return [k for k in numeric if k not in skip and not any(h in k for h in _CONSOLE_HIDDEN_SUBSTRINGS)]

    def _step_specs(self) -> list[tuple[str, str, int]]:
        # (value_key, header_label, width). Trailing __rate/__eta are computed, not metrics.
        step_w = max(len("step"), len(f"{self.steps_per_epoch}/{self.steps_per_epoch}"))
        specs = [("step", "step", step_w)]
        for key in self._step_keys:
            head = _short_header(key)
            specs.append((key, head, max(len(head), _NUM_COL_W)))
        specs.append(("lr", "lr", _NUM_COL_W))
        specs.append(("__rate", "s/it", 8))
        specs.append(("__eta", "eta", 7))
        return specs

    def _print_table_header(self, specs: list[tuple[str, str, int]]) -> None:
        print("  " + "  ".join(f"{head:>{w}}" for _, head, w in specs), flush=True)
        print("  " + "  ".join("-" * w for _, _, w in specs), flush=True)

    def log_step(self, epoch: int, step: int, row: dict) -> None:
        self._global_step += 1
        numeric = self._numeric(row)
        if self._wandb is not None: self._wandb.log({f"{self.stage}/{k}": v for k, v in numeric.items()}, step=self._global_step)

        now = time.monotonic()
        if epoch != self._epoch:
            self._epoch = int(epoch)
            self._epoch_t0 = now
            self._epoch_step0 = int(step)
            self._step_keys = self._resolve_step_keys(numeric)
            self._step_specs_cache = self._step_specs()
            print(f"\n{self.stage}  epoch {epoch}/{self.epochs}  ({self.steps_per_epoch} steps)", flush=True)
            self._print_table_header(self._step_specs_cache)

        if self.log_every > 0 and (step % self.log_every == 0 or step == self.steps_per_epoch):
            done = step - self._epoch_step0
            rate = (now - self._epoch_t0) / done if done > 0 else None
            values: dict[str, str] = {k: _fmt_metric(v) for k, v in numeric.items()}
            values["step"] = f"{step}/{self.steps_per_epoch}"
            values["__rate"] = f"{rate:.2f}s" if rate else "--"
            values["__eta"] = _fmt_duration((self.steps_per_epoch - step) * rate) if rate else "--"
            cells = [f"{values.get(key, ''):>{w}}" for key, _, w in self._step_specs_cache]
            print("  " + "  ".join(cells), flush=True)

    def log_epoch(self, epoch: int, metrics: dict, tag: str = "epoch") -> None:
        numeric = self._numeric(metrics)
        if self._wandb is not None:
            self._wandb.log({f"{self.stage}/{tag}/{k}": v for k, v in numeric.items()}, step=self._global_step)

        suffix = ""
        if tag == "train" and self._epoch == epoch and self._epoch_t0:
            suffix = f"  (took {_fmt_duration(time.monotonic() - self._epoch_t0)})"
        print(f"{self.stage}  epoch {epoch}/{self.epochs}  [{tag}]{suffix}", flush=True)
        if not numeric: return

        # Aligned key/value table, sorted so related metrics (best_*, val_*) group and scan cleanly.
        name_width = max(len(k) for k in numeric)
        for key in sorted(numeric):
            print(f"    {key:<{name_width}}  {_fmt_metric(numeric[key]):>{_VAL_COL_W}}", flush=True)

    def finish(self) -> None:
        if self._wandb is not None and self._wandb.run is not None:
            try: self._wandb.finish()
            except Exception: pass # noqa: BLE001


def mean_logs(rows: list[dict[str, float]], prefix: str = "train") -> dict[str, float]:
    if not rows: return {}
    keys = sorted({key for row in rows for key in row if key not in {"epoch", "step", "lr"}})
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if values: out[f"{prefix}_{key}"] = sum(values) / len(values)
    return out


@dataclass
class TrainControl:
    eval_every_epochs: int = 0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    monitor: str = "val_loss"
    monitor_mode: Literal["min", "max"] = "min"
    restore_best: bool = True
    best_value: float | None = None
    best_epoch: int = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs: int = 0
    stopped_early: bool = False
    # Save-on-best: when set, `update()` calls this with the model the moment a new best monitor value is found and writes a 
    # best.json sidecar — so a crash/preemption (Colab) never loses the best weights. The end-of-training save in train.py 
    # still runs (after restore_best it rewrites the same state). Wire via `attach_save_best`.
    name: str = "train"
    save_fn: Callable[[torch.nn.Module], Path | None] | None = None

    @classmethod
    def from_config(cls, cfg: dict, default_monitor: str = "val_loss", default_mode: Literal["min", "max"] = "min") -> "TrainControl":
        validation = cfg.get("validation", {})
        early = cfg.get("early_stopping", {})
        return cls(
            eval_every_epochs=int(cfg.get("eval_every_epochs", validation.get("eval_every_epochs", 0)) or 0),
            early_stopping_patience=int(early.get("patience", cfg.get("patience", 0)) or 0),
            early_stopping_min_delta=float(early.get("min_delta", 0.0)),
            monitor=str(early.get("monitor", default_monitor)),
            monitor_mode=str(early.get("mode", default_mode)),  # type: ignore[arg-type]
            restore_best=bool(early.get("restore_best", True)),
        )

    def should_eval(self, epoch: int, epochs: int) -> bool:
        if self.eval_every_epochs <= 0: return False
        return epoch % self.eval_every_epochs == 0 or epoch == int(epochs)

    def _is_better(self, value: float) -> bool:
        if self.best_value is None: return True
        if self.monitor_mode == "max": return value > self.best_value + self.early_stopping_min_delta
        return value < self.best_value - self.early_stopping_min_delta

    def update(self, model: torch.nn.Module, metrics: dict[str, float], epoch: int) -> bool:
        if self.monitor not in metrics: return False
        value = float(metrics[self.monitor])
        if self._is_better(value):
            previous = self.best_value
            self.best_value = value
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            if self.restore_best: self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if self.save_fn is not None:
                path = self.save_fn(model)
                prev = "first eval" if previous is None else f"prev {previous:.6g}"
                where = f" -> saved {path}" if path is not None else ""
                print(f"{self.name} | * new best {self.monitor}={value:.6g} @ epoch {epoch} ({prev}){where}", flush=True)
                if path is not None:
                    meta = {
                        "monitor": self.monitor, "mode": self.monitor_mode,
                        "value": value, "epoch": int(epoch), "checkpoint": str(path),
                    }
                    Path(path).parent.joinpath("best.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            return True
        self.bad_epochs += 1
        if self.early_stopping_patience > 0 and self.bad_epochs >= self.early_stopping_patience: self.stopped_early = True
        return False

    def restore(self, model: torch.nn.Module) -> None:
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(copy.deepcopy(self.best_state))

    def summary(self) -> dict[str, float | int | bool | None | str]:
        return {
            "monitor": self.monitor, "monitor_mode": self.monitor_mode,
            "best_value": self.best_value, "best_epoch": self.best_epoch, "stopped_early": self.stopped_early,
        }


def attach_save_best(
    control: TrainControl, cfg: dict, name: str,
    saver: Callable[[torch.nn.Module, str | Path], Path],
) -> TrainControl:
    """Wire save-on-best into a TrainControl from the config's `checkpoint:` block.

    `checkpoint.dir` (falling back to `output_dir`) is where the best checkpoint and `best.json`
    land; `checkpoint.save_best: false` disables mid-training saves (end-of-training save only).
    `saver(model, dir)` performs the stage-appropriate write (full model vs visual backbone).
    """
    control.name = str(name)
    ckpt_cfg = dict(cfg.get("checkpoint", {}) or {})
    ckpt_dir = ckpt_cfg.get("dir") or cfg.get("output_dir")
    if bool(ckpt_cfg.get("save_best", True)) and ckpt_dir: control.save_fn = lambda model: saver(model, ckpt_dir)
    return control


@dataclass
class SchedulerBundle:
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    interval: SchedulerInterval = "none"

    def step_batch(self) -> None:
        if self.scheduler is not None and self.interval == "step":
            self.scheduler.step()

    def step_epoch(self) -> None:
        if self.scheduler is not None and self.interval == "epoch":
            self.scheduler.step()

    def lr(self, optimizer: torch.optim.Optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, epochs: int, steps_per_epoch: int) -> SchedulerBundle:
    sched_cfg = cfg.get("scheduler", {})
    sched_type = str(sched_cfg.get("type", "none")).lower()
    if sched_type in {"", "none", "constant"}: return SchedulerBundle()

    total_steps = max(1, int(epochs) * max(1, int(steps_per_epoch)))
    if sched_type in {"onecycle", "one_cycle", "adamw-onecycle"}:
        max_lr = float(sched_cfg.get("max_lr", optimizer.param_groups[0]["lr"]))
        pct_start = float(sched_cfg.get("pct_start", 0.1))
        div_factor = float(sched_cfg.get("div_factor", 25.0))
        final_div_factor = float(sched_cfg.get("final_div_factor", 1e4))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps,
            pct_start=pct_start, div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
        return SchedulerBundle(scheduler=scheduler, interval="step")

    if sched_type in {"cosine", "cosine_annealing"}:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(epochs)),
            eta_min=float(sched_cfg.get("eta_min", 0.0)),
        )
        return SchedulerBundle(scheduler=scheduler, interval="epoch")
    raise ValueError(f"Unsupported scheduler type: {sched_type}")
