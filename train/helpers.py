from __future__ import annotations
from typing import Any, Callable, Iterable, Literal
from dataclasses import dataclass, field

import copy, csv, json, sys, time
from pathlib import Path
from tqdm.auto import tqdm
from contextlib import contextmanager, nullcontext

import torch
from torch.utils.data import DataLoader
from models.checkpointing import save_model_checkpoint
from utils import checkpoint_dir, save_best_enabled


def move_to_device(value, device: torch.device):
    # Recursively move tensors in a (possibly nested) batch container onto `device`.
    if isinstance(value, torch.Tensor): return value.to(device)
    if isinstance(value, dict): return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list): return [move_to_device(v, device) for v in value]
    return value


def build_optimizer(cfg: dict, params) -> torch.optim.Optimizer:
    """AdamW from a config, reading the SAME keys for every stage.

    Prefers top-level `learning_rate` / `weight_decay` (the slt/bio convention); falls back to nested
    `optimizer.lr` / `optimizer.weight_decay` so older configs keep working. One place, one convention.
    """
    opt = cfg.get("optimizer", {}) or {}
    lr = float(cfg.get("learning_rate", opt.get("lr", 1e-4)))
    weight_decay = float(cfg.get("weight_decay", opt.get("weight_decay", 1e-4)))
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


@contextmanager
def eval_mode(model: torch.nn.Module):
    # `model.eval()` for the block, restoring the prior train/eval state on exit. Shared by every evaluator.
    was_training = model.training
    model.eval()
    try: yield
    finally:
        if was_training: model.train()


class AmpHelper:
    """Mixed-precision wrapper shared by all training loops.

    `mixed_precision:` config values: "auto" (default — bf16 when the GPU supports it, else fp16 with loss scaling), "bf16", 
    "fp16", or "none". CPU always runs fp32. bf16 needs no GradScaler (same exponent range as fp32); fp16 uses one to prevent 
    gradient underflow. F.cross_entropy / softmax run in fp32 under autocast (PyTorch autocast promote list), so the 1/t-weighted 
    BD3LM loss and SPD/DCD confidences keep full precision.
    """
    def __init__(self, mode: str = "auto", device: torch.device | str = "cpu"):
        device_type = torch.device(device).type
        mode = str(mode or "auto").lower()
        if device_type != "cuda" or mode in {"none", "off", "fp32", "float32"}: self.dtype = None
        elif mode in {"bf16", "bfloat16"}: self.dtype = torch.bfloat16
        elif mode in {"fp16", "float16"}: self.dtype = torch.float16
        elif mode == "auto": self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else: raise ValueError(f"Unknown mixed_precision mode: {mode}")
        self.device_type = device_type
        self.scaler = torch.amp.GradScaler(device_type, enabled=self.dtype == torch.float16)

    @classmethod
    def from_config(cls, cfg: dict | None, device: torch.device | str) -> "AmpHelper":
        return cls(mode=str((cfg or {}).get("mixed_precision", "auto")), device=device)

    def autocast(self):
        if self.dtype is None: return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)

    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler.is_enabled(): self.scaler.scale(loss).backward()
        else: loss.backward()

    def clip_and_step(self, optimizer: torch.optim.Optimizer, parameters, max_grad_norm: float) -> None:
        # fp16: gradients must be unscaled before clipping or the norm is measured on scaled values.
        if self.scaler.is_enabled():
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, float(max_grad_norm))
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(parameters, float(max_grad_norm))
            optimizer.step()


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

def _fmt_metric(value) -> str:
    # Fixed-shape number for table columns: 3 decimals in the normal range, 3-sig sci at the extremes (tiny LRs, huge sums). 
    # Within a column the metric magnitude is stable, so the chosen form is consistent down the column → decimal points line up.
    try: v = float(value)
    except (TypeError, ValueError): return str(value)
    if v != v: return "nan"  # NaN
    if v == 0: return "0.000"
    av = abs(v)
    if av >= 1e5 or av < 1e-3: return f"{v:.3e}"
    return f"{v:.3f}"


class TrainLogger: # Unified console + Weights & Biases logger for the training loops.
    def __init__(
        self, stage: str, cfg: dict | None = None, epochs: int = 0, 
        steps_per_epoch: int = 0, monitor: str = "val_loss",
    ):
        self.stage = str(stage)
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        cfg = cfg or {}
        wandb_cfg = dict(cfg.get("wandb", {}) or {})

        self.monitor = str(monitor)  # surfaced first among the val columns of the epoch table
        self._global_step = 0
        self._epoch: int | None = None
        self._epoch_t0 = 0.0
        self._rows: list[dict] = []
        self._progress = None
        self._progress_total = max(0, self.epochs * self.steps_per_epoch)
        self._wandb = None
        self._history_csv: Path | None = None
        self._configure_history_paths(cfg)

        if bool(wandb_cfg.get("enabled", False)):
            try: import wandb
            except ImportError:
                print(f"{self.stage} | wandb enabled in config but not installed (pip install wandb); console only", flush=True)
            else: self._init_wandb(wandb, wandb_cfg, cfg)


    def _configure_history_paths(self, cfg: dict) -> None:
        root = checkpoint_dir(cfg)
        logging_cfg = dict(cfg.get("logging", {}) or {})
        if logging_cfg.get("history", True) is False: return
        out_dir = logging_cfg.get("dir") or root
        if not out_dir: return
        out_dir = Path(out_dir)
        self._history_csv = out_dir / "history.csv"

    def _init_wandb(self, wandb, wandb_cfg: dict, cfg: dict) -> None:
        if wandb.run is not None: self._wandb = wandb; return
        init_kwargs = dict(
            project=str(wandb_cfg.get("project", "misaligned-slt")),
            name=str(wandb_cfg.get("name") or self.stage),
            group=wandb_cfg.get("group"), config=cfg,
        )
        mode = wandb_cfg.get("mode")
        if mode and str(mode) != "online": init_kwargs["mode"] = str(mode)
        try:
            wandb.init(**init_kwargs)
            self._wandb = wandb
        except Exception as exc:  # noqa: BLE001 - login declined / unavailable → continue without W&B
            print(f"{self.stage} | wandb not active ({type(exc).__name__}: {exc}); console table only", flush=True)
            self._wandb = None
            try: wandb.finish()
            except Exception: pass  # noqa: BLE001

    @staticmethod
    def _numeric(row: dict) -> dict:
        return {k: v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

    @staticmethod
    def _public_key(key: str) -> str:
        return {
            "train_total_loss": "train_loss",
            "train_phrase_bio_loss": "train_loss",
            "train_baseline_ce_loss": "train_loss",
            "train_vlp_loss": "train_loss",
        }.get(key, key)

    @classmethod
    def _public_metrics(cls, row: dict) -> dict:
        out = {}
        for key, value in row.items():
            public = cls._public_key(str(key))
            if public in out and public != key: continue # Prefer an explicitly named target over an alias collision.
            out[public] = value
        return out

    def log_step(self, epoch: int, step: int, row: dict) -> None:
        self._global_step += 1
        numeric = self._numeric(row)
        if self._wandb is not None: self._wandb.log({f"{self.stage}/{k}": v for k, v in numeric.items()}, step=self._global_step)
        if epoch != self._epoch:
            self._epoch = int(epoch)
            self._epoch_t0 = time.monotonic()
        self._update_progress(epoch, step, numeric)


    def _update_progress(self, epoch: int, step: int, numeric: dict) -> None:
        if self._progress is None:
            self._progress = tqdm(
                total=(self._progress_total or None), desc=f"{self.stage} train", 
                unit="step", dynamic_ncols=True, leave=True, file=sys.stdout,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )

        absolute_step = (int(epoch) - 1) * self.steps_per_epoch + int(step) if self.steps_per_epoch > 0 else self._global_step
        delta = max(0, absolute_step - int(self._progress.n))
        if delta: self._progress.update(delta)

        postfix = {}
        for key in ("loss", "total_loss", "phrase_bio_loss", "vlp_loss", "baseline_ce_loss"):
            if key in numeric:
                postfix["loss"] = _fmt_metric(numeric[key])
                break

        if "lr" in numeric: postfix["lr"] = _fmt_metric(numeric["lr"])
        if postfix: self._progress.set_postfix(postfix, refresh=False)


    def epoch_summary(
        self, epoch: int, train: dict, val: dict | None = None, 
        is_best: bool = False, saved_path: str | None = None
    ) -> None:
        """Record one epoch and append one comma-separated line.

        ALL train/val metrics become columns; epochs without an eval show `-`. `ckpt` records a save event, and `eta` 
        estimates remaining training time from the mean completed-epoch duration. The best epoch is available from
        `best.json`, so it is not duplicated in the console/history rows.
        """
        took = (time.monotonic() - self._epoch_t0) if (self._epoch == epoch and self._epoch_t0) else None
        train_num, val_num = self._numeric(train or {}), self._numeric(val or {})
        if self._wandb is not None:
            payload = {f"{self.stage}/epoch/{k}": v for k, v in {**train_num, **val_num}.items()}
            if payload: self._wandb.log(payload, step=self._global_step)

        elapsed_epochs = [float(r["took"]) for r in self._rows if isinstance(r.get("took"), (int, float))]
        if took is not None: elapsed_epochs.append(float(took))
        mean_epoch_s = sum(elapsed_epochs) / max(1, len(elapsed_epochs))
        remaining_s = max(0, int(self.epochs) - int(epoch)) * mean_epoch_s

        row: dict = {"epoch": epoch, **self._public_metrics({**train_num, **val_num})}
        if took is not None: row["took"] = took
        row["eta"] = remaining_s
        row["ckpt"] = "saved" if saved_path else ""
        self._rows.append(row)

        # Console line = readable `key=value` fields (each value self-labelled), same column order and cell
        # formatting as history.csv. epoch shows progress `n/total`; an empty `ckpt` (no save) is omitted.
        fields = []
        for col in self._order_columns(row.keys()):
            if col == "epoch": fields.append(f"epoch={epoch}/{self.epochs}")
            elif col == "ckpt":
                if row.get(col): fields.append(f"ckpt={row[col]}")
            else: fields.append(f"{col}={self._format_cell(row.get(col), col)}")

        text = ", ".join(fields)
        if self._progress is not None: self._progress.write(text)
        else: print(text, flush=True)
        self._save_history_files()
        
        if self._progress is not None:
            postfix = {"epoch": f"{epoch}/{self.epochs}"}
            if val_num and self.monitor in val_num: postfix[self.monitor] = _fmt_metric(val_num[self.monitor])
            self._progress.set_postfix(postfix, refresh=False)


    def _order_columns(self, keys) -> list[str]:
        keys = list(keys)
        train = sorted(k for k in keys if k.startswith("train_"))
        val = sorted(k for k in keys if k.startswith("val_"))

        # Pin *_loss first within each group, but only when present: train-only epochs (no dev_loader, or a
        # non-eval epoch) emit no val_* columns, so an unconditional remove() would crash _save_history_files.
        if "train_loss" in train: train.remove("train_loss"); train.insert(0, "train_loss")
        if "val_loss" in val: val.remove("val_loss"); val.insert(0, "val_loss")
        tail = [k for k in ("took", "eta", "ckpt") if k in keys]
        other = [k for k in keys if k not in {"epoch", *train, *val, *tail}]
        return ["epoch", *train, *val, *other, *tail]


    def _format_cell(self, value, column: str) -> str:
        if value is None: return "-"
        if column in {"epoch", "ckpt"}: return str(value) if str(value) else "-"
        if column in {"took", "eta"}:
            try: return _fmt_duration(float(value))
            except (TypeError, ValueError): return str(value)
        if isinstance(value, (int, float)): return _fmt_metric(value)
        return str(value)


    def _save_history_files(self) -> None:
        if not self._rows or self._history_csv is None: return
        self._history_csv.parent.mkdir(parents=True, exist_ok=True)
        columns = self._order_columns({key for row in self._rows for key in row})

        with self._history_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in self._rows: writer.writerow([self._format_cell(row.get(col, "-"), col) for col in columns])


    def finish(self) -> None:
        if self._progress is not None:
            try: self._progress.close()
            except Exception: pass  # noqa: BLE001
            self._progress = None
        if self._wandb is not None and self._wandb.run is not None:
            try: self._wandb.finish()
            except Exception: pass # noqa: BLE001


def run_epoch_loop(
    *, name: str, model: torch.nn.Module, loader: DataLoader, 
    optimizer: torch.optim.Optimizer, device: torch.device, epochs: int, cfg: dict, 
    step_fn: Callable[[dict, int], tuple[torch.Tensor, dict[str, float]]], 
    evaluate_fn: Callable[[int], dict[str, float]] | None = None,
    default_monitor: str = "val_loss", default_mode: Literal["min", "max"] = "min",
    dev_loader: DataLoader | None = None,
) -> list[dict[str, float]]:
    """The one training loop every trainer shares (slt / bio_s1).

    Owns the whole skeleton — scheduler, AMP, best-checkpoint selection, early stop, logging, restore —
    so a stage only supplies what genuinely differs:
      step_fn(batch, epoch)  -> (loss_tensor, scalar_log_dict). Runs under this loop's amp.autocast; the
                                loop does zero_grad / backward / grad-clip(model.parameters()) / step.
      evaluate_fn(epoch)     -> dev metrics dict (called only on control.should_eval epochs), or None.

    Grad clipping is over model.parameters() for all stages: frozen params carry no grad, so clipping a
    superset that includes them is identical to clipping only the trainable subset.
    """
    model.to(device)
    model.train()
    logs: list[dict[str, float]] = []

    scheduler = build_scheduler(optimizer, cfg, epochs=epochs, steps_per_epoch=len(loader))
    amp = AmpHelper.from_config(cfg, device)
    control = TrainControl.from_config(cfg, default_monitor=default_monitor, default_mode=default_mode)
    attach_save_best(control, cfg, name, save_model_checkpoint)
    logger = TrainLogger(name, cfg, epochs=int(epochs), steps_per_epoch=len(loader), monitor=control.monitor)
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    for epoch in range(1, int(epochs) + 1):
        epoch_logs: list[dict[str, float]] = []
        for step, batch in enumerate(loader, start=1):
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                loss, step_logs = step_fn(batch, epoch)

            amp.backward(loss)
            amp.clip_and_step(optimizer, model.parameters(), max_grad_norm)
            scheduler.step_batch()
            row = {"epoch": float(epoch), "step": float(step), "lr": scheduler.lr(optimizer), **step_logs}
            epoch_logs.append(row); logs.append(row)
            logger.log_step(epoch, step, row)

        scheduler.step_epoch()
        train_means = mean_logs(epoch_logs)
        if evaluate_fn is not None and dev_loader is not None and control.should_eval(epoch, epochs):
            metrics = evaluate_fn(epoch)
            improved = control.update(model, metrics, epoch)
            logger.epoch_summary(epoch, train=train_means, val=metrics, is_best=improved, saved_path=control.last_saved_path)
            logs.append({"epoch": float(epoch), **train_means, **metrics, **control.summary()})
            if control.stopped_early:
                print(f"{name} | early stop at epoch {epoch} (best {control.monitor}={control.best_value})", flush=True)
                break
        else: logger.epoch_summary(epoch, train=train_means)

    control.restore(model)
    logger.finish()
    return logs


def mean_logs(rows: list[dict[str, float]], prefix: str = "train") -> dict[str, float]:
    if not rows: return {}
    sums, counts = {}, {}
    for row in rows:
        for key, value in row.items():
            if key in {"epoch", "step", "lr"}: continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {f"{prefix}_{key}": sums[key] / counts[key] for key in sums if counts.get(key, 0) > 0}


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
    last_saved_path: str | None = None  # Set on the epoch a new best is saved (for the logger to report)
    _warned_missing_monitor: bool = False

    @classmethod
    def from_config(cls, cfg: dict, default_monitor: str = "val_loss", default_mode: Literal["min", "max"] = "min") -> "TrainControl":
        # Two separated concerns:
        #   checkpoint: monitor / mode / restore_best  -> best-model SELECTION (always active when a dev set exists)
        #   early_stopping: enabled / patience / min_delta -> TERMINATION only (opt-in; removing the block disables it)
        validation = cfg.get("validation", {})
        ckpt = cfg.get("checkpoint", {})
        early = cfg.get("early_stopping", {})
        early_enabled = bool(early.get("enabled", False))
        return cls(
            eval_every_epochs=int(cfg.get("eval_every_epochs", validation.get("eval_every_epochs", 0)) or 0),
            early_stopping_patience=int(early.get("patience", 0) or 0) if early_enabled else 0,
            early_stopping_min_delta=float(early.get("min_delta", 0.0)),
            monitor=str(ckpt.get("monitor", default_monitor)),
            monitor_mode=str(ckpt.get("mode", default_mode)),  # type: ignore[arg-type]
            restore_best=bool(ckpt.get("restore_best", True)),
        )

    def should_eval(self, epoch: int, epochs: int) -> bool:
        if self.eval_every_epochs <= 0: return False
        return epoch % self.eval_every_epochs == 0 or epoch == int(epochs)

    def _is_better(self, value: float) -> bool:
        if self.best_value is None: return True
        if self.monitor_mode == "max": return value > self.best_value + self.early_stopping_min_delta
        return value < self.best_value - self.early_stopping_min_delta

    def update(self, model: torch.nn.Module, metrics: dict[str, float], epoch: int) -> bool:
        if self.monitor not in metrics:
            # Loudly surface a monitor/metric-key mismatch once. This was previously silent: a wrong
            # monitor name (e.g. segmenter "val_phrase_f1" when eval emits "val_phrase_tiou_f1") left
            # best_epoch=0 and never saved a best checkpoint, with no warning.
            if not self._warned_missing_monitor:
                self._warned_missing_monitor = True
                available = ", ".join(sorted(metrics)) or "(no metrics returned)"
                print(f"{self.name} | WARNING: monitor '{self.monitor}' not in eval metrics — best-checkpoint "
                      f"tracking and early stopping are DISABLED. Available keys: {available}", flush=True)
            return False

        value = float(metrics[self.monitor])
        self.last_saved_path = None
        if self._is_better(value):
            self.best_value = value
            self.best_epoch = int(epoch)
            self.bad_epochs = 0

            if self.restore_best: self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if self.save_fn is not None:
                path = self.save_fn(model)
                if path is not None:
                    self.last_saved_path = str(path)
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
    """Wire save-on-best into a TrainControl from the consolidated `checkpoint:` block.

    `checkpoint.dir` is where the best checkpoint and `best.json`
    land; `checkpoint.save_best: false` disables mid-training saves (end-of-training save only).
    `saver(model, dir)` performs the stage-appropriate write (full model vs visual backbone).
    """
    control.name = str(name)
    ckpt_dir = checkpoint_dir(cfg)
    if save_best_enabled(cfg) and ckpt_dir: control.save_fn = lambda model: saver(model, ckpt_dir)
    return control


SchedulerInterval = Literal["step", "epoch", "none"]

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
