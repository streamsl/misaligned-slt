from __future__ import annotations
import os
import copy
import torch
from dataclasses import dataclass, field
from typing import Literal

SchedulerInterval = Literal["step", "epoch", "none"]


def _fmt_num(value) -> str: # Compact number formatting for console lines (no scientific noise for ints).
    if isinstance(value, bool): return str(value)
    if isinstance(value, int): return str(value)
    try: return f"{float(value):.4g}"
    except (TypeError, ValueError): return str(value)


class TrainLogger:
    """Unified console + Weights & Biases logger for the training loops.

    Console: one compact line every `log_every` steps and an epoch-summary line — no timestamps. W&B: every step (and per epoch) 
    under a `<stage>/` namespace. Enable W&B via a `wandb:` block in the config (`enabled`, `project`, `name`, `group`, `mode`); 
    if W&B is unavailable or unconfigured the logger degrades to console-only with a one-line notice rather than failing the run.
    """
    def __init__(
        self, stage: str, cfg: dict | None = None,
        epochs: int = 0, steps_per_epoch: int = 0, log_every: int | None = None,
    ):
        self.stage = str(stage)
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        cfg = cfg or {}
        wandb_cfg = dict(cfg.get("wandb", {}) or {})
        self.log_every = int(log_every if log_every is not None else cfg.get("log_every", wandb_cfg.get("log_every", 10)) or 1)
        self._global_step = 0
        self._wandb = None
        if bool(wandb_cfg.get("enabled", False)):
            try:
                os.environ.setdefault("WANDB_SILENT", "true")
                import wandb

                if wandb.run is None:
                    wandb.init(
                        project=str(wandb_cfg.get("project", "misaligned-slt")),
                        name=str(wandb_cfg.get("name") or self.stage),
                        group=wandb_cfg.get("group"),
                        mode=str(wandb_cfg.get("mode", "online")),
                        config=cfg,
                    )
                self._wandb = wandb
            except Exception as exc:  # noqa: BLE001 - never let logging kill training
                print(f"{self.stage} | wandb disabled ({type(exc).__name__}: {exc}); console only", flush=True)

    @staticmethod
    def _numeric(row: dict) -> dict:
        return {k: v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

    def log_step(self, epoch: int, step: int, row: dict) -> None:
        self._global_step += 1
        numeric = self._numeric(row)
        if self._wandb is not None: self._wandb.log({f"{self.stage}/{k}": v for k, v in numeric.items()}, step=self._global_step)
        if self.log_every > 0 and (step % self.log_every == 0 or step == self.steps_per_epoch):
            body = "  ".join(f"{k} {_fmt_num(v)}" for k, v in numeric.items() if k not in {"epoch", "step"})
            print(f"{self.stage} | ep {epoch}/{self.epochs} | step {step}/{self.steps_per_epoch} | {body}", flush=True)

    def log_epoch(self, epoch: int, metrics: dict, tag: str = "epoch") -> None:
        numeric = self._numeric(metrics)
        if self._wandb is not None:
            self._wandb.log({f"{self.stage}/{tag}/{k}": v for k, v in numeric.items()}, step=self._global_step)
        body = "  ".join(f"{k} {_fmt_num(v)}" for k, v in numeric.items())
        print(f"{self.stage} | ep {epoch}/{self.epochs} [{tag}] | {body}", flush=True)

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
            self.best_value = value
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            if self.restore_best: self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
