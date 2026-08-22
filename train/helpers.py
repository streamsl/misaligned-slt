from __future__ import annotations
from typing import Callable, Literal
from dataclasses import dataclass

import copy, csv, json, sys, time
from pathlib import Path
from tqdm.auto import tqdm
from contextlib import contextmanager, nullcontext

import torch
from torch.utils.data import DataLoader
from models.checkpointing import load_train_state, save_train_state, save_model_checkpoint
from utils import checkpoint_dir, save_best_enabled
from train import distributed as dist


def move_to_device(value, device: torch.device):
    # Recursive tensor move for nested batch containers.
    if isinstance(value, torch.Tensor): return value.to(device)
    if isinstance(value, dict): return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list): return [move_to_device(v, device) for v in value]
    return value


def build_optimizer(cfg: dict, params, backbone_params=None) -> torch.optim.Optimizer:
    """AdamW from a config, reading the SAME keys for every stage.

    Prefers top-level `learning_rate` / `weight_decay`; falls back to `optimizer.lr` / `optimizer.weight_decay`
    so older configs keep working.

    `backbone_params`: optional second param set at `backbone_lr` (default learning_rate × 0.1) — discriminative
    fine-tuning for a PRETRAINED encoder unfrozen under a head-scale learning_rate.
    """
    opt = cfg.get("optimizer", {}) or {}
    lr = float(cfg.get("learning_rate", opt.get("lr", 1e-4)))
    weight_decay = float(cfg.get("weight_decay", opt.get("weight_decay", 1e-4)))

    # No weight decay on biases / 1-D params (LayerNorm, RMSNorm): decaying norm gains regularizes the wrong
    # thing. Same split as Uni-Sign (timm create_optimizer filter_bias_and_bn=True) and standard HF practice.
    def wd_split(ps, group_lr):
        ps = [p for p in ps if p.requires_grad]
        return [g for g in ({"params": [p for p in ps if p.ndim > 1], "weight_decay": weight_decay, "lr": group_lr},
                            {"params": [p for p in ps if p.ndim <= 1], "weight_decay": 0.0, "lr": group_lr}) if g["params"]]

    groups = wd_split(params, lr)
    if backbone_params is not None:
        groups += wd_split(backbone_params, float(cfg.get("backbone_lr", opt.get("backbone_lr", lr * 0.1))))
    # fused=True on CUDA: one multi-tensor kernel per step instead of a Python loop over ~800M trainable params.
    return torch.optim.AdamW(groups, lr=lr, fused=torch.cuda.is_available())


@contextmanager
def eval_mode(model: torch.nn.Module):
    # `model.eval()` for the block, restoring the prior mode on exit.
    was_training = model.training
    model.eval()
    try: yield
    finally:
        if was_training: model.train()


class AmpHelper:
    """Mixed-precision wrapper shared by all training loops.

    `mixed_precision:` "auto" (default — bf16 if supported, else fp16), "bf16", "fp16", "none". CPU always fp32. bf16 needs no 
    GradScaler (same exponent range as fp32); fp16 uses one against gradient underflow. F.cross_entropy / softmax run fp32 under 
    autocast (PyTorch promote list), so the 1/t-weighted BD3LM loss and SPD/DCD confidences keep full precision.
    """
    def __init__(self, mode: str = "auto", device: torch.device | str = "cpu"):
        device_type = torch.device(device).type
        mode = str(mode or "auto").lower()
        if device_type != "cuda" or mode in {"none", "off", "fp32", "float32"}: self.dtype = None
        elif mode in {"bf16", "bfloat16"}: self.dtype = torch.bfloat16
        elif mode in {"fp16", "float16"}: self.dtype = torch.float16
        # including_emulation=False: torch's default reports bf16 "supported" on Turing (Colab's T4) via EMULATION —
        # no tensor cores, and the fp16 GradScaler path stays disabled, so "auto" picks the slowest option there.
        elif mode == "auto": self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
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
        # fp16: unscale before clipping or the norm is measured on scaled values.
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
    # Fixed-shape number for table columns: 3 decimals normally, 3-sig sci at the extremes (tiny LRs, huge sums).
    # Magnitude is stable within a column, so the form stays consistent → decimal points line up.
    try: v = float(value)
    except (TypeError, ValueError): return str(value)
    if v != v: return "nan"  # NaN
    if v == 0: return "0.000"
    av = abs(v)
    if av >= 1e5 or av < 1e-3: return f"{v:.3e}"
    return f"{v:.3f}"


class TrainLogger: # Console + Weights & Biases logger for the training loops.
    def __init__(
        self, stage: str, cfg: dict | None = None, epochs: int = 0,
        steps_per_epoch: int = 0, monitor: str = "val_loss", enabled: bool = True, resumed: bool = False,
    ):
        # `enabled=False` (non-zero ranks under torchrun) makes every method a no-op: no console, no progress bar,
        # no history.csv, no wandb — one writer per artifact, one readable stream on stdout.
        self.enabled = bool(enabled)
        self.stage = str(stage)
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        cfg = dict(cfg or {}) if self.enabled else {}
        wandb_cfg = dict(cfg.get("wandb", {}) or {})

        self.monitor = str(monitor)  # surfaced first among the val columns
        self._global_step = 0
        self._epoch: int | None = None
        self._epoch_t0 = 0.0
        self._rows: list[dict] = []
        self._resumed = bool(resumed)  # only a resume may carry forward prior rows (see _save_history_files)
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
        except Exception as exc:
            print(f"{self.stage} | wandb not active ({type(exc).__name__}: {exc}); console table only", flush=True)
            self._wandb = None
            try: wandb.finish()
            except Exception: pass

    @staticmethod
    def _numeric(row: dict) -> dict:
        return {k: v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


    def log_step(self, epoch: int, step: int, row: dict) -> None:
        if not self.enabled: return
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
        self, epoch: int, train: dict, val: dict | None = None, is_best: bool = False, saved_path: str | None = None
    ) -> None:
        if not self.enabled: return
        """Record one epoch and append one comma-separated line.

        ALL train/val metrics become columns; non-eval epochs show `-`. `ckpt` records a save; `eta` estimates
        remaining time from the mean completed-epoch duration. Best epoch lives in `best.json`, not duplicated here.
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

        row: dict = {"epoch": epoch, **train_num, **val_num}
        if took is not None: row["took"] = took
        row["eta"] = remaining_s
        row["ckpt"] = "saved" if saved_path else ""
        self._rows.append(row)

        # Console line = `key=value` fields, same column order and cell formatting as history.csv. 
        # epoch shows `n/total`; empty `ckpt` omitted.
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

        def group(prefix: str) -> list[str]:
            # Tuple key ranks bare `{prefix}loss` < other `*_loss` (A–Z) < the rest (A–Z), so val reads val_loss,
            # val_bio_loss, val_translation_loss, … mirroring the train row instead of scattering component losses
            # through the alphabetical metrics.
            return sorted((k for k in keys if k.startswith(prefix)),
                          key=lambda k: (k != f"{prefix}loss", not k.endswith("_loss"), k))

        train, val = group("train_"), group("val_")
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
        # A RESUMED run starts with an empty _rows, so rewriting would drop epochs 1..k of the curve — the only
        # record of them when wandb is off. Carry those forward, but ONLY on a resume: a fresh run in the same
        # checkpoint dir would otherwise inherit the tail of a longer previous run (its epochs beyond this run's
        # last), producing a file that reads as one curve with an unexplained jump. `_resumed` is set by the
        # training loop; without it, a fresh run truncates the file to its own rows.
        rows = self._rows
        if self._resumed and self._history_csv.exists():
            with self._history_csv.open("r", encoding="utf-8", newline="") as f:
                prior = [r for r in csv.DictReader(f) if r.get("epoch") not in {str(r2.get("epoch")) for r2 in rows}]
            rows = prior + rows
        columns = self._order_columns({key for row in rows for key in row})

        with self._history_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in rows: writer.writerow([self._format_cell(row.get(col, "-"), col) for col in columns])


    def finish(self) -> None:
        if not self.enabled: return
        if self._progress is not None:
            try: self._progress.close()
            except Exception: pass
            self._progress = None
        if self._wandb is not None and self._wandb.run is not None:
            try: self._wandb.finish()
            except Exception: pass


def run_epoch_loop(
    *, name: str, model: torch.nn.Module, loader: DataLoader, 
    optimizer: torch.optim.Optimizer, device: torch.device, epochs: int, cfg: dict, 
    step_fn: Callable[[dict, int], tuple[torch.Tensor, dict[str, float]]],
    evaluate_fn: Callable[[int], dict[str, float]] | None = None,
    default_monitor: str = "val_loss", default_mode: Literal["min", "max"] = "min",
    dev_loader: DataLoader | None = None, resume: bool = False, checkpoint_meta: dict | None = None,
) -> list[dict[str, float]]:
    """The one training loop every trainer shares (slt / bio_s1).

    Owns scheduler, AMP, best-checkpoint selection, early stop, logging, restore; a stage supplies only:
      step_fn(batch, epoch)  -> (loss_tensor, scalar_log_dict), run under this loop's amp.autocast; the loop does
                                zero_grad / backward / grad-clip(model.parameters()) / step.
      evaluate_fn(epoch)     -> dev metrics dict (only on control.should_eval epochs), or None.

    Grad clipping covers model.parameters() everywhere: frozen params carry no grad, so clipping the superset is
    identical to clipping the trainable subset.
    """
    model.to(device)
    model.train()
    logs: list[dict[str, float]] = []

    scheduler = build_scheduler(optimizer, cfg, epochs=epochs, steps_per_epoch=len(loader))
    amp = AmpHelper.from_config(cfg, device)
    control = TrainControl.from_config(cfg, default_monitor=default_monitor, default_mode=default_mode)
    # Side effects (checkpoint writes, wandb, history.csv, progress bars) are rank 0's alone: concurrent writers
    # would race on one path, and N copies of the same numbers make the console unreadable. Every rank still runs
    # the identical control logic on ALL-REDUCED metrics, so their best/early-stop decisions never diverge.
    if dist.is_main(): attach_save_best(control, cfg, name, save_model_checkpoint, meta=checkpoint_meta)
    logger = TrainLogger(name, cfg, epochs=int(epochs), steps_per_epoch=len(loader), monitor=control.monitor,
                         enabled=dist.is_main(), resumed=bool(resume))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    if dist.is_distributed() and amp.scaler.is_enabled(): raise SystemExit(
        "fp16 AMP is unsafe under multi-GPU here: each rank keeps its OWN GradScaler, so ranks can disagree about "
        "the loss scale (and about skipping a step) and drift apart. Use amp: bf16 (A100/H100) or amp: none."
    )
    # Preemption safety (Colab): latest.pt = full resumable state, rewritten every epoch; loses at most one epoch.
    ckpt_dir = checkpoint_dir(cfg)
    latest_path = Path(ckpt_dir) / "latest.pt" if ckpt_dir else None
    start_epoch = 1
    # Re-running the same command without --resume (the Colab reflex after a session dies) would overwrite latest.pt
    # AND the best model.pt at the end of epoch 1, discarding the whole prior run. Refuse instead.
    if not resume and latest_path is not None and latest_path.exists(): raise SystemExit(
        f"{latest_path} exists: a previous run is resumable. Pass --resume to continue it, or move/delete that file "
        f"to start over (this would otherwise overwrite it and the best model.pt after one epoch)."
    )
    if resume:
        if latest_path is None or not latest_path.exists():
            raise SystemExit(f"--resume: no resumable state at {latest_path} (need checkpoint.dir + a prior epoch)")
        
        state = load_train_state(latest_path, model, optimizer)
        # Analysis stages rewrite inference.yaml and the jitter artifact between sessions. Both parameterize this run. Resuming across such 
        # a change trains the 2nd half under a different objective — visible afterwards only as an unexplained discontinuity in loss curves.
        saved_meta = dict(state.get("meta") or {})
        if saved_meta and checkpoint_meta:
            drift = sorted(k for k in set(saved_meta) | set(checkpoint_meta) if saved_meta.get(k) != checkpoint_meta.get(k))
            if drift: raise SystemExit(
                f"--resume: this run started under different training-critical config; {', '.join(drift)} changed "
                + "; ".join(f"{k}: {saved_meta.get(k)!r} -> {checkpoint_meta.get(k)!r}" for k in drift[:4])
                + f". Restore those values to resume, or start a fresh run (move {latest_path}) — resuming across "
                f"the change trains the two halves under different objectives."
            )
        saved_epochs = int(state.get("epochs", epochs))
        if saved_epochs != int(epochs) and scheduler.scheduler is not None: raise SystemExit(
            f"--resume: run was launched with epochs={saved_epochs} but this invocation says {epochs}; the LR schedule horizon is baked "
            f"into the scheduler state, so resume with the SAME --epochs (extend a finished run by warm-starting from model.pt instead)."
        )
        if scheduler.scheduler is not None and state.get("scheduler") is not None: scheduler.scheduler.load_state_dict(state["scheduler"])
        if amp.scaler.is_enabled() and state.get("scaler") is not None: amp.scaler.load_state_dict(state["scaler"])

        control.load_state_dict(state.get("control") or {})
        control.best_checkpoint_path = str(Path(ckpt_dir) / "model.pt") if ckpt_dir else None
        start_epoch = int(state["epoch"]) + 1
        if start_epoch > int(epochs): raise SystemExit(f"--resume: {latest_path} already at epoch {state['epoch']} of {epochs}")
        if dist.is_main(): print(f"{name} | resumed {latest_path} -> starting epoch {start_epoch}/{epochs} "
                                 f"(best {control.monitor}={control.best_value} @ epoch {control.best_epoch})", flush=True)

    for epoch in range(start_epoch, int(epochs) + 1):
        # Re-shuffle the shard boundaries per epoch where the sampler supports it (DistributedSampler).
        for ld in (loader, dev_loader):
            sampler = getattr(ld, "sampler", None) if ld is not None else None
            if hasattr(sampler, "set_epoch"): sampler.set_epoch(epoch)
        epoch_logs: list[dict[str, float]] = []
        for step, batch in enumerate(loader, start=1):
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                loss, step_logs = step_fn(batch, epoch)

            amp.backward(loss)
            # Average gradients BEFORE clipping so every rank clips the same (global) gradient and therefore
            # applies the identical update — clipping per-rank first would make the clip threshold rank-dependent.
            dist.average_gradients(model.parameters())
            # Clip only trainable params: iterating all ~1B (frozen included) per step is pure overhead.
            amp.clip_and_step(optimizer, [p for p in model.parameters() if p.requires_grad], max_grad_norm)
            scheduler.step_batch()
            row = {"epoch": float(epoch), "step": float(step), "lr": scheduler.lr(optimizer), **step_logs}
            epoch_logs.append(row); logs.append(row)
            logger.log_step(epoch, step, row)

        scheduler.step_epoch()
        train_means = dist.reduce_metrics(mean_logs(epoch_logs))
        if evaluate_fn is not None and dev_loader is not None and control.should_eval(epoch, epochs):
            with amp.autocast():  # dev eval in the same precision as training steps; fp32 eval was ~2x slower
                eval_metrics = evaluate_fn(epoch)
            metrics = dist.reduce_metrics(eval_metrics)
            improved = control.update(model, metrics, epoch)
            logger.epoch_summary(epoch, train=train_means, val=metrics, is_best=improved, saved_path=control.last_saved_path)
            logs.append({"epoch": float(epoch), **train_means, **metrics, **control.summary()})
            if control.stopped_early:
                if dist.is_main(): print(f"{name} | early stop at epoch {epoch} (best {control.monitor}={control.best_value})", flush=True)
                break
        else: logger.epoch_summary(epoch, train=train_means)
        if dist.is_main() and latest_path is not None: save_train_state(
            latest_path, model=model, optimizer=optimizer,
            scheduler_state=scheduler.scheduler.state_dict() if scheduler.scheduler is not None else None,
            scaler_state=amp.scaler.state_dict() if amp.scaler.is_enabled() else None,
            control_state=control.state_dict(), epoch=epoch, epochs=int(epochs), meta=checkpoint_meta,
        )
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
    out = {f"{prefix}_{key}": sums[key] / counts[key] for key in sums if counts.get(key, 0) > 0}
    # Two silent-failure alarms the logged numbers already contained but nobody compared:
    #   B-class collapse — signing-vs-not P/R/F1 stay high while B never fires, so only its own rate shows it.
    #   all-I control — a segmenter that cannot beat "call every frame signing and chop by the duration prior"
    #   has learned nothing a reviewer will credit; both quantities were being computed already.
    b_rate = next((v for k, v in out.items() if k.endswith("_pred_b_rate")), None)
    gold_rate = next((v for k, v in out.items() if k.endswith("_gold_b_rate")), None)
    if b_rate is not None and gold_rate and b_rate < 0.1 * gold_rate: print(
        f"[{prefix}] WARNING: B-class collapse — predicted B rate {b_rate:.5f} vs gold {gold_rate:.5f}. "
        f"Check bio_class_weights (train/losses.py documents this failure).", flush=True
    )
    tiou = next((v for k, v in out.items() if k.endswith("phrase_tiou_f1")), None)
    alli = next((v for k, v in out.items() if k.endswith("alli_tiou_f1")), None)
    if tiou is not None and alli is not None and tiou <= alli: print(
        f"[{prefix}] WARNING: tIoU-F1 {tiou:.4f} does not beat the all-I control {alli:.4f} — the head is "
        f"contributing nothing over chopping by the duration prior.", flush=True
    )
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
    # Save-on-best: `update()` calls this the moment a new best monitor value appears and writes a best.json
    # sidecar, so a crash/preemption (Colab) never loses the best weights. train.py's end-of-training save still
    # runs (after restore_best it rewrites the same state). Wire via `attach_save_best`.
    name: str = "train"
    save_fn: Callable[[torch.nn.Module], Path | None] | None = None
    last_saved_path: str | None = None  # Set on the epoch a new best is saved (for the logger)
    best_checkpoint_path: str | None = None  # Disk fallback for restore() after a resume (best_state is in-memory only)
    _warned_missing_monitor: bool = False

    @classmethod
    def from_config(cls, cfg: dict, default_monitor: str = "val_loss", default_mode: Literal["min", "max"] = "min") -> "TrainControl":
        # Separate concerns:
        #   checkpoint: monitor / mode / restore_best      -> best-model SELECTION (active whenever a dev set exists)
        #   early_stopping: enabled / patience / min_delta -> TERMINATION only (opt-in; drop the block to disable)
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
            # Surface a monitor/metric-key mismatch once: a wrong monitor name (e.g. "val_phrase_f1" when eval
            # emits "val_phrase_tiou_f1") silently left best_epoch=0 and never saved a best checkpoint.
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
        if not self.restore_best: return
        if self.best_state is not None: model.load_state_dict(copy.deepcopy(self.best_state))
        elif self.best_checkpoint_path and Path(self.best_checkpoint_path).exists() and self.best_epoch > 0:
            # Resumed run whose best epoch predates the resume: the in-memory copy died with the old process,
            # but the save-on-best file survived — restore from disk.
            state = torch.load(self.best_checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"] if "model" in state else state)

    def state_dict(self) -> dict:
        return {"best_value": self.best_value, "best_epoch": self.best_epoch, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict) -> None:
        self.best_value = state.get("best_value")
        self.best_epoch = int(state.get("best_epoch", 0))
        self.bad_epochs = int(state.get("bad_epochs", 0))

    def summary(self) -> dict[str, float | int | bool | None | str]:
        return {
            "monitor": self.monitor, "monitor_mode": self.monitor_mode,
            "best_value": self.best_value, "best_epoch": self.best_epoch, "stopped_early": self.stopped_early,
        }


def attach_save_best(
    control: TrainControl, cfg: dict, name: str, saver: Callable[[torch.nn.Module, str | Path], Path], meta: dict | None = None,
) -> TrainControl:
    """Wire save-on-best into a TrainControl from the `checkpoint:` block.

    `checkpoint.dir`: where the best checkpoint and `best.json` land. `save_best: false` disables mid-training
    saves. `saver(model, dir)` does the stage-appropriate write (full model vs visual backbone).
    """
    control.name = str(name)
    ckpt_dir = checkpoint_dir(cfg)
    if save_best_enabled(cfg) and ckpt_dir: control.save_fn = lambda model: saver(model, ckpt_dir, meta=meta)
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
        # Optional linear warmup (scheduler.warmup_epochs, fractional ok): every A2D/OPUT reference warms up when
        # retraining a pretrained model (dllm 0.1 of steps, DMax 0.03) — full LR from step one knocks a converged
        # checkpoint out of its basin. Stepped per-STEP: a per-epoch step over a short horizon is a staircase.
        warmup_epochs = float(sched_cfg.get("warmup_epochs", 0.0))
        warmup_steps = int(round(warmup_epochs * max(1, int(steps_per_epoch))))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=float(sched_cfg.get("eta_min", 0.0)),
        )
        if warmup_steps <= 0: return SchedulerBundle(scheduler=cosine, interval="step")
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
        return SchedulerBundle(scheduler=scheduler, interval="step")
    raise ValueError(f"Unsupported scheduler type: {sched_type}")
