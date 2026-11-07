from __future__ import annotations
import bisect, copy, os, random, time, unicodedata
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from transformers import T5Tokenizer

from data.batch import WindowCollator
from data.windowing import join_runs, joined_target
from data.loader import ANNOTATION_PROTOCOL, RankSlice, StreamingWindowDataset, annotation_fingerprint, load_language_records, streaming_loader
from models.unisign import UniSignMT5FrontEnd, PROMPT_LANG_BY_TARGET
from models.streaming_slt import TEXT_ROUTING, MisalignedSLTModel, SLTLossOutput
from models.checkpointing import load_checkpoint_meta, require_fixed_constants, warn_pose_geometry
from poses.pose_io import POSE_GEOMETRY
from infer.duration_decode import DurationModel
from infer.stream import join_unit_text

from train import distributed as dist
from train.bio_pretrain import edge_reach, encoder_reach, require_segmenter_data
from train.losses import bio_class_weight_tensor, resolve_bio_class_weights
from train.helpers import BestFile, mean_logs, move_to_device, require_resume_state, run_epoch_loop, segmenter_config_decode
from metrics import bio_frame_metrics, text_level, text_scores
from utils import UNIT_SEPARATOR, cfg_get, checkpoint_dir, lambda_min_s, load_yaml, pool_key, resolve_pretrained, seconds_to_frames

# DEFAULT S1 config. `checkpoint.bio_head_init: auto` resolves the S1 checkpoint through it, and the pool-provenance check reads its 
# `pretrain.languages`. `train.py --bio-config` overrides it, so stage 1 and 2 read SAME S1 recipe when a run uses a non-default one.
BIO_S1_CONFIG = "configs/bio_pretrain.yaml"
# Gate options; delta and the minimum eligible length come from the fixed inference.yaml constants.
GATE_CONFIG_KEYS = frozenset({"enabled", "eps", "detach_omega"})
SPD_CONFIG_KEYS = frozenset({"tau_dec", "top_k", "renormalize"})


@dataclass
class SLTComponents:
    model: MisalignedSLTModel
    tokenizer: Any
    train_loader: DataLoader
    dev_loader: DataLoader | None
    slt_cfg: dict
    checkpoint_meta: dict
    inference_cfg: dict | None = None
    monitor_records: list | None = None

def _file_stamp(path) -> dict | None:   # Which file initialized a branch, with its size: a path alone doesn't show a later overwrite.
    if not path: return None
    f = Path(str(path))
    return {"path": str(path), "bytes": f.stat().st_size if f.exists() else None}

def _resolved_inits(translator, translator_meta: dict, segmentation, segmentation_meta: dict) -> dict:
    """The files that initialized the translator and the segmentation branch, and, under `annotation_signature`, the train labels each
    was trained on (its meta's annotation_fingerprint; S1's per-language language_fingerprints). As with the duration prior's signature,
    a --resume compares the files and not these labels (models.checkpointing.without_annotation); eval warns when they are not current 
    train labels (evaluation.systems.model_annotation). None where an init carries no stamp (the released Uni-Sign file)."""
    return {"translator": _file_stamp(translator), "segmentation": _file_stamp(segmentation), "annotation_signature": {
        "translator": translator_meta.get("annotation_fingerprint"), "segmentation": segmentation_meta.get("annotation_fingerprint"),
        "segmentation_languages": segmentation_meta.get("language_fingerprints"),
    }}

def _inject_gate_geometry(slt_cfg: dict, inference_cfg: dict) -> None:
    # Match the sampler and FSM's first eligible target. Short units remain legal paths;
    # they are skipped by this target-selection rule, not removed from the path distribution.
    gate = slt_cfg.setdefault("membership_gate", {})
    # The readers index every key, so a missing key fails there; an unknown (for example misspelled) key would be ignored, so it fails here.
    unknown = set(gate) - GATE_CONFIG_KEYS
    if unknown: raise ValueError(f"membership_gate: unknown key(s) {sorted(unknown)}; accepted: {sorted(GATE_CONFIG_KEYS)}")
    unknown = set(slt_cfg.get("spd", {}) or {}) - SPD_CONFIG_KEYS  # same reason
    if unknown: raise ValueError(f"spd: unknown key(s) {sorted(unknown)}; accepted: {sorted(SPD_CONFIG_KEYS)}")
    gate["delta_s"] = float(inference_cfg["boundary_stability"]["delta_s"])
    gate["min_span_s"] = lambda_min_s(inference_cfg)

def _gate_frames(gate_cfg: dict, batch: dict) -> dict:
    # Λ_min and δ in frames of each row's own pose grid (WindowSample.fps: the record's fps, or the realized rate of an augmented
    # train window): rows of 1 batch may have different frame rates.
    return {"gate_min_span_frames": [seconds_to_frames(gate_cfg["min_span_s"], fps) for fps in batch["fps"]],
            "gate_delta_frames": [seconds_to_frames(gate_cfg["delta_s"], fps) for fps in batch["fps"]]}

def online_monitor_records(dev_records: list, hours: float, seed: int) -> tuple[list, float]:
    """The online monitor's streams (checkpoint.monitor val_online_score): whole dev source videos (all their streams) in a seeded 
    random order, until pose time reaches `hours` (last video may pass it). The hours cap the cost of 1 evaluation on every corpus; 
    the random order keeps the subset unbiased (the first ids by sort are not a sample) and fixed across epochs and launches."""
    by_video: dict[str, list] = {}
    for record in dev_records: by_video.setdefault(record.pose.video_id, []).append(record)
    order = sorted(by_video)
    random.Random(int(seed)).shuffle(order)
    records, total_s = [], 0.0
    for vid in order:
        if total_s >= hours * 3600: break
        records += by_video[vid]; total_s += sum(r.pose.duration_s for r in by_video[vid])
    return records, total_s


def _training_meta(slt_cfg: dict, inference_cfg: dict, language: str) -> dict:
    """The config this stage-2 run is parameterized by, travelling with the weights.

    δ, Λ_min and buffer_cap_s come from inference.yaml and the duration prior from the train labels. Resuming across a change 
    of any of them trains 2 halves under different objectives, and without this record nothing in the artifacts shows it.
    """
    gate_cfg = slt_cfg.get("membership_gate", {}) or {}
    # two_stream_slt: separate segmentation/translation pose encoders, Ω on encoder keys and cross-attention; one_stream_slt:
    # one shared pose encoder (shared_pose_encoder). Eval builds the layout the tag names and refuses any other tag.
    architecture = ("one_stream_slt" if slt_cfg.get("shared_pose_encoder") else "two_stream_slt") \
                   if float(slt_cfg["lambda_bio"]) else "clean_translation"
    return {
        "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": slt_cfg.get("annotation_fingerprint"),
        "language": str(language), "decoder": str(slt_cfg.get("decoder", "dlm")),
        # The frame geometry of the poses it read (poses.pose_io.load_pose_frames); eval warns on any other value
        # (models.checkpointing.warn_pose_geometry), and --resume refuses a latest.pt with another one (one written before the map stamps none).
        "pose_geometry": slt_cfg.get("pose_geometry"), "architecture": architecture,
        # The hidden size of the one-stream arm's stitch (stitch.hidden_dim), which eval builds it with; None for every other layout,
        # so their --resume reads no drift. Its fit's results are not stamped: a resumed run skips the fit.
        "stitch_hidden_dim": int(slt_cfg["stitch"]["hidden_dim"]) if architecture == "one_stream_slt" else None,
        "bio_objective": "ce_dice", "dice_loss_weight": float(slt_cfg["bio"]["dice_loss_weight"]),
        "gate": {k: gate_cfg.get(k) for k in ("enabled", "delta_s", "min_span_s", "eps", "detach_omega")},
        # Text routing: GT-anchored targets (one unit, or a run of units with gaps <= δ joined by UNIT_SEPARATOR) on
        # a detached hard crop of the predicted span when it holds the target within δ, else on the δ-jittered GT span; the
        # end term on targets with a pause after them, from the B frame of their last unit.
        "text_routing": TEXT_ROUTING, "unit_separator": UNIT_SEPARATOR, "bio_adapter": bool(slt_cfg["bio"]["adapter"]),
        # The train rows' input types (train_inputs, data/sampler.py spec_for), so --resume refuses a run that started under other ones.
        "train_inputs": slt_cfg.get("train_inputs"),
        # The checkpoint monitor; online_dev_hours only for the online monitor (the offline monitor reads the whole dev set).
        "monitor": slt_cfg.get("checkpoint", {}).get("monitor"), "seed": slt_cfg.get("seed"),
        # What a row monitor computes (`evaluation.offline.dev_monitor_scores`), the settings of its main score (eval.yaml scoring: chunk
        # gap, tIoU thresholds) and the decode of the head's spans (segmenter_decode, named as S1 stamps its own: argmax or
        # semi_markov_viterbi; eval decodes the arm's rows as its epoch was chosen, evaluation.segmenter_eval.arm_decode), so --resume
        # refuses a run selected under another one. The inference settings a monitor reads (inference.yaml stride_s and K, and with them
        # the cap cut window, the FSM and span rules) are not stamped: a change can select another epoch, and --resume does not refuse a
        # run across it (README "After a change"). The clean floor's monitor (val_translation_bleu4) stamps none of them.
        **({"monitor_protocol": "chunk_tiou_f1_x_dvc_bleu4", "monitor_edge": slt_cfg.get("monitor_edge"),
            "monitor_decode": "semi_markov_viterbi" if segmenter_config_decode(slt_cfg) == "duration" else "argmax"}
           if str(slt_cfg.get("checkpoint", {}).get("monitor", "")).startswith(("val_offline_", "val_online_")) else
           {"monitor_protocol": None, "monitor_edge": None, "monitor_decode": None}),
        "online_dev_hours": slt_cfg.get("checkpoint", {}).get("online_dev_hours")
                            if str(slt_cfg.get("checkpoint", {}).get("monitor", "")).startswith("val_online_") else None,
        # Which files initialized the translator and the segmentation branch, 
        # with the train labels of each (_resolved_inits), and the head's attention band.
        "resolved_inits": slt_cfg.get("resolved_inits"), "bio_attention_radius_s": slt_cfg.get("bio_attention_radius_s"),
        "early_stopping": slt_cfg.get("early_stopping"), "scheduler": slt_cfg.get("scheduler"), "epochs": slt_cfg.get("epochs"),
        "buffer_cap_s": inference_cfg.get("buffer_cap_s"), "duration_model": slt_cfg.get("duration_model"),
        # The decode of the gate's span pick in training (the text routing), the monitor's (segmenter_decode); eval builds the arm with it.
        "segmentation_decode": ("bio_argmax" if segmenter_config_decode(slt_cfg) == "argmax" else "semi_markov_viterbi")
                               if slt_cfg.get("duration_model") else "none",
        "oput": slt_cfg["oput"], "spd": slt_cfg["spd"], 
        # Training geometry of the text canvas: the block-causal mask and the OPUT corruption read block_size, so eval refuses 
        # a decoder built at another block (evaluation/systems.py build_eval_model), and the decode canvas must not shrink.
        "block_size": int(slt_cfg["block_size"]), "max_text_tokens": slt_cfg.get("max_text_tokens"),
        "batch_size": slt_cfg.get("batch_size"), "learning_rate": float(slt_cfg["learning_rate"]),
        "mode_ratios": slt_cfg.get("mode_ratios"), "jitter": slt_cfg.get("jitter"), "augmentation": slt_cfg.get("augmentation"),
        "bio_class_weights": slt_cfg["bio"]["class_weights"],  # resolved list, not the "balanced" string
        "lambda_bio": float(slt_cfg["lambda_bio"]), "lambda_trans": float(slt_cfg["lambda_trans"]),
    }


def assert_targets_fit(records, tokenizer, max_text_tokens: int, buffer_cap_s: float, delta_s: float, split: str) -> None:
    """Refuse to start if any text target the sampler can build exceeds the text canvas.

    The collator never truncates a complete-caption target (truncated reference silently rewrites the task), so an over-long target
    would crash mid-epoch instead. A target is a unit that fits the streaming buffer (data/sampler.py `_clip_window`), or a run of
    units with gaps of at most δ that fits it together (data.windowing.join_runs, joined by UNIT_SEPARATOR); this mirrors those
    predicates from every unit and reports the capacity to configure, as buffer_cap_s is sized to the data.
    """
    targets = []
    # A complete unit's terminator frame lies inside the window, and _clip_window caps the window at buffer_cap_s, so a run that
    # spans more than buffer_cap_s can never be a target: buffer_cap_s is an upper bound that covers every reachable run.
    cap_s = float(buffer_cap_s)
    for record in records:
        units = sorted((u for u in record.sentences if getattr(u, "reliable", True)), key=lambda u: (u.start_s, u.end_s))
        gap_s = seconds_to_frames(delta_s, record.pose.fps) / record.pose.fps
        # The target from unit i is the first run of join_runs, and only what lies in its cap window [start, start + cap] shapes that
        # run: a unit that starts later cannot fit, and a sentence wholly outside cannot lie between 2 of its units. So join_runs reads
        # that window, not the rest of the record (every unit against every sentence made ase train take minutes).
        starts = [u.start_s for u in units]
        by_start = sorted(record.sentences, key=lambda u: u.start_s)
        sentence_starts, longest_s = [u.start_s for u in by_start], max((u.duration_s for u in by_start), default=0.0)
        for i, unit in enumerate(units):
            if unit.duration_s > cap_s: continue
            window_end = unit.start_s + cap_s
            near = [u for u in by_start[bisect.bisect_left(sentence_starts, unit.start_s - longest_s):
                                        bisect.bisect_left(sentence_starts, window_end)] if u.end_s > unit.start_s]
            targets.append(joined_target(join_runs(units[i:bisect.bisect_right(starts, window_end)], near, gap_s, max_span_s=cap_s)[0]))
    # A text never has more tokens than the UTF-8 bytes of its NFKC form + 2: each SentencePiece token covers at least 1 byte of the
    # normalized text (a byte-fallback token exactly 1), plus the leading "▁" and EOS. So only the targets whose bound reaches the
    # longest count so far are tokenized, the largest bound first: the same count, and the same culprit (the first target in record
    # order with it), as tokenizing every target (ase train: about 330,000).
    bound = [len(unicodedata.normalize("NFKC", t.text).encode("utf-8")) + 2 for t in targets]
    tokens, longest = {}, 0
    for k in sorted(range(len(targets)), key=lambda k: -bound[k]):
        if bound[k] < longest: break
        tokens[k] = len(tokenizer(targets[k].text)["input_ids"])
        longest = max(longest, tokens[k])
    culprit = targets[min(k for k, n in tokens.items() if n == longest)] if tokens else None
    # +1: decode canvas holds BOS in slot 0, so a target of exactly max_text_tokens could be trained on but never emitted.
    if longest + 1 > int(max_text_tokens): raise ValueError(
        f"{split}: a complete caption target needs {longest} tokens plus the BOS slot but max_text_tokens is {max_text_tokens} "
        f"({culprit.video_id} {culprit.start_s:.1f}-{culprit.end_s:.1f}s). Set max_text_tokens >= {longest + 1}: the collator "
        f"refuses to truncate a complete target, and would raise on this target the first time it is sampled."
    )


def require_translator_init(
    pretrained_path, segmentation_branch: bool, language: str, inference_cfg: dict, fingerprint, decoder: str, block_size: int,
    augmentation: dict,
) -> dict:
    """Translator warm start of a train-slt run. Clean floor starts from released Uni-Sign weights (no training meta): the method's
    checkpoint.from_pretrained, for every language (utils.resolve_pretrained). An arm starts
    from the clean floor of its own decoder (a DLM floor at its own block size) of THIS language, trained under fixed constants
    (another augmentation or other annotations only warn). Returns the warm start's meta."""
    # A trained warm start carries meta; the released Uni-Sign file carries none.
    init_meta = load_checkpoint_meta(pretrained_path) if Path(str(pretrained_path)).exists() else {}
    key = f"data.yaml languages.{language}.pretrained_slt.{decoder}"
    if segmentation_branch and not init_meta: raise SystemExit(
        f"Stage 2 must warm-start the translator from this language's trained pretrimmed {decoder} translator, but {pretrained_path} "
        f"carries no training meta (the released file, or no file). Train that pretrimmed translator (README B2a), then set {key} to "
        f"it (B2b)."
    )
    # After the B2b re-root, data.yaml pretrained_slt names the previous clean floor: warm-starting the floor from it would
    # silently fine-tune the old floor a second time.
    if not segmentation_branch and init_meta: raise SystemExit(
        f"The pretrimmed translator must warm-start from released Uni-Sign weights, but {pretrained_path} is a trained checkpoint "
        f"(architecture={init_meta.get('architecture')!r}). Point checkpoint.from_pretrained (baseline_train.yaml) at a released "
        f"file."
    )
    if segmentation_branch and init_meta:
        # The decoder must match: the AR arm would read a DLM floor's untrained mT5 lm_head, and the DLM arm would start its
        # canvas from scratch on an AR floor. A DLM decoder must also keep its block-causal mask (eval refuses another block).
        found = {"architecture": init_meta.get("architecture"), "language": init_meta.get("language"),
                 "decoder": init_meta.get("decoder"), "block_size": init_meta.get("block_size")}
        want = {"architecture": "clean_translation", "language": str(language), "decoder": str(decoder)}
        if decoder == "dlm": want["block_size"] = int(block_size)
        bad = [f"{k}={found[k]!r} (need {want[k]!r})" for k in want if found[k] != want[k]]
        if bad: raise SystemExit(
            f"Stage 2 warm-starts the translator from {pretrained_path}, but it is not the pretrimmed {decoder} translator of "
            f"{language}: {'; '.join(bad)}. Train that pretrimmed translator (README B2a), then set {key} to it (B2b)."
        )
        if init_meta.get("annotation_fingerprint") != fingerprint: print(
            f"WARNING: the translator warm start {pretrained_path} trained on other {language} "
            f"annotations ({init_meta.get('annotation_fingerprint')!r}, now {fingerprint!r}). "
            f"Retrain the pretrimmed translator before a run you report.", flush=True
        )
        require_fixed_constants(init_meta, inference_cfg, pretrained_path)  # same rule eval applies to it
    return init_meta


def fit_stitch(
    stitch: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, rows: torch.Tensor, held_from: int, cfg: dict, seed: int, device,
) -> dict:
    """Fit the one-stream arm's stitch (models.bio_head.EncoderStitch, `cfg` = dlm.yaml stitch) from `x` (the translator encoder's
    features) to `y` (S1's encoder features): the kept frames of every train row and then of every dev window of the run, in the
    1-process row order, float16 on the CPU, with `rows` the row of each: the train rows are 0, 1, ... and the dev windows `held_from`
    on. The dev windows are held out (whole dev videos that the fit never reads, so their R² is a dev estimate); the train rows fit.
    The buffers are the
    fit frames' moments: x's mean and std per dimension (std at least stitch.std_floor), y's mean per dimension, and y_scale, the
    root mean square of y's deviation over the fit frames and the dimensions. The 2 Linear layers start from torch's init under
    `seed` and train with AdamW on mean(((stitch(x) - y) / y_scale)^2), in batches of stitch.batch_frames frames from a seeded random
    order each epoch, under ReduceLROnPlateau on the held-out loss RSS / TSS (pooled over the dimensions, TSS around the held-out
    frames' own mean: 1 - R²), for at most stitch.max_epochs epochs, until stitch.patience epochs bring no gain above
    stitch.min_delta; the best epoch's weights stay. float32 on `device`. Returns the report."""
    batch, held = int(cfg["batch_frames"]), rows >= int(held_from)
    fit_idx, held_idx = (~held).nonzero().squeeze(1), held.nonzero().squeeze(1)
    if not len(fit_idx) or not len(held_idx): raise ValueError(
        f"The stitch fit needs train frames to fit and dev frames to hold out, but it got {len(fit_idx)} train frames and "
        f"{len(held_idx)} dev frames."
    )

    def moments(t: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # float64 mean and population std per dimension over the frames `idx`, a batch at a time (no copy of all of them)
        s = ss = 0.
        for j in idx.split(batch):
            c = t[j].double()
            s, ss = s + c.sum(0), ss + (c * c).sum(0)
        mean = s / len(idx)
        return mean, (ss / len(idx) - mean * mean).clamp(min=0).sqrt()

    home = next(stitch.parameters()).device
    stitch.cpu()
    with torch.no_grad():
        x_mean, x_std = moments(x, fit_idx)
        y_mean, y_std = moments(y, fit_idx)
        stitch.x_mean.copy_(x_mean); stitch.x_std.copy_(x_std.clamp(min=float(cfg["std_floor"])))
        stitch.y_mean.copy_(y_mean); stitch.y_scale.fill_(float(y_std.pow(2).mean().sqrt()))
        tss = len(held_idx) * float(moments(y, held_idx)[1].pow(2).sum())
    with torch.random.fork_rng(devices=[]):   # the init draws as the 2 layers built right after torch.manual_seed(seed)
        torch.default_generator.manual_seed(int(seed))
        for layer in stitch.net:
            if isinstance(layer, torch.nn.Linear): layer.reset_parameters()
    stitch.to(device)
    opt = torch.optim.AdamW(stitch.net.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=float(cfg["plateau_factor"]), patience=int(cfg["plateau_patience"]))
    order = torch.Generator().manual_seed(int(seed))
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        for j in fit_idx[torch.randperm(len(fit_idx), generator=order)].split(batch):
            loss = ((stitch(x[j].to(device).float()) - y[j].to(device).float()) / stitch.y_scale).pow(2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            held_loss = sum(float((stitch(x[j].to(device).float()).cpu().double() - y[j].double()).pow(2).sum())
                            for j in held_idx.split(batch)) / tss
        plateau.step(held_loss)
        if held_loss < best - float(cfg["min_delta"]):
            best, best_epoch, bad = held_loss, epoch, 0
            best_state = {k: v.clone() for k, v in stitch.state_dict().items()}
        else: bad += 1
        if bad >= int(cfg["patience"]): break
    if best_state is None: raise RuntimeError("The stitch fit gave no finite held-out loss: its training diverged.")
    stitch.load_state_dict(best_state)
    stitch.to(home)
    unique = torch.unique(rows)
    return {"fit_rows": int((unique < held_from).sum()), "held_rows": int((unique >= held_from).sum()), "fit_frames": len(fit_idx),
            "held_frames": len(held_idx), "r2": 1.0 - best, "best_epoch": best_epoch, "epochs": epoch}


def fit_s1_stitch(
    model: MisalignedSLTModel, s1_pose_sd: dict, batches, held_batches, row_frames, held_from: int, stride: int, cfg: dict, seed: int,
    device,
) -> dict | None:
    """One-stream arm: fit `model.bio_stitch`, through which pooled S1's head reads the clean translator's encoder E_T, without labels
    before stage 2 trains. S1's encoder E_S1 (built from `s1_pose_sd`) only gives the fit its targets and is then dropped. `batches`
    are this rank's train rows r, r + N, ... in order (rank r of N; 1 process: every row), `held_batches` its rows r, r + N, ... of
    the run's dev windows, numbered from `held_from` (the number of train rows) and held out by the fit, and `row_frames` holds the
    most real frames that each of them can have, the train rows first. Both encoders read each row in eval mode, and of a row's real frames the fit keeps every
    `stride`-th, from frame (row % stride) % real frames, so every row gives frames (the caller passes 2 x the encoder's reach + 1:
    the kept frames have disjoint receptive fields), as float16 on the CPU, in 1 buffer per tensor sized by `row_frames` (no copy
    of them; the pages past the frames kept are never written). Under torchrun the kept frames go to rank 0, each straight to its
    place in the 1-process row order (train.distributed.gather_to_main), rank 0 fits them (`fit_stitch`, on `device`) while the
    other ranks wait (train.distributed.rank0_alone), and every rank then takes rank 0's stitch, so N ranks give the 1-process fit
    up to float rounding. Returns rank 0's report, with `minutes` for the extraction, the gather and the fit (None on the others)."""
    started = time.perf_counter()
    capacity = sum(-(-int(n) // stride) for n in row_frames)   # a row of n real frames keeps at most ceil(n / stride)
    # Rank 0 holds x and y (float16) and the row id of every kept frame for the whole fit. Check that bound against this machine's
    # memory now, not after hours of extraction (under torchrun on 1 machine the ranks' own shares also stay until the gather ends).
    total = int(dist.all_sum(torch.tensor([capacity], device=dist.reduction_device()))) if dist.is_distributed() else capacity
    need, ram = total * (4 * int(model.front_end.bio_tap_dim) + 8), os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if dist.is_main(): print(
        f"slt | stitch fit: at most {total} kept frames, up to {need / 2**30:.1f} GiB of CPU memory on rank 0"
        + (" (up to twice that on 1 machine during the gather)" if dist.is_distributed() else "")
        + f"; this machine has {ram / 2**30:.1f} GiB", flush=True)
    if need > ram: raise SystemExit(
        f"The stitch fit keeps up to {total} frames, up to {need / 2**30:.1f} GiB of CPU memory on rank 0, but this machine has "
        f"{ram / 2**30:.1f} GiB: run the one-stream arm's train-slt on a machine with more memory."
    )
    student, stitch = model.front_end.pose_encoder, model.bio_stitch
    teacher = copy.deepcopy(student)
    teacher.load_state_dict(s1_pose_sd, strict=True)
    home, mode = next(student.parameters()).device, student.training
    student.to(device).eval(); teacher.to(device).eval()
    x, y = (torch.empty(capacity, int(model.front_end.bio_tap_dim), dtype=torch.float16) for _ in range(2))
    rows, kept, seen = torch.empty(capacity, dtype=torch.long), 0, 0
    with torch.no_grad():
        for first, part in ((0, batches), (int(held_from), held_batches)):   # the train rows, then the dev windows
            seen = 0
            for batch in part:
                poses, mask = batch["poses"].to(device), batch["frame_mask"].to(device).bool()
                ids = first + (seen + torch.arange(len(mask), device=device)) * dist.world_size() + dist.rank()
                seen += len(mask)
                index = mask.long().cumsum(1) - 1   # each frame's place among the real frames of its row
                start = ((ids % stride) % mask.sum(1).clamp(min=1))[:, None]
                keep = mask & (index >= start) & ((index - start) % stride == 0)
                n = int(keep.sum())
                if kept + n > capacity: raise RuntimeError(
                    f"The stitch fit keeps more frames than its bound ({capacity}): a row has more real frames than its row_frames "
                    f"bound (data.sampler.WindowSampler.spec_frames in build_slt_components)."
                )
                x[kept:kept + n] = student(poses, mask)[keep].half(); y[kept:kept + n] = teacher(poses, mask)[keep].half()
                rows[kept:kept + n] = ids[:, None].expand_as(keep)[keep]
                kept += n
    student.to(home).train(mode)
    del teacher
    x, y, rows = x[:kept], y[:kept], rows[:kept]
    if dist.is_distributed():
        # Rank 0 takes every rank's frames, the rows first: a stable sort by row gives the 1-process order (each rank holds its rows
        # in order, and no row is on 2 ranks), and x and y go straight to their places in it, 1 at a time. Each rank frees its own.
        rows = dist.gather_to_main(rows)
        place = None
        if rows is not None:
            order = torch.argsort(rows, stable=True)
            place = torch.empty_like(order)
            place[order] = torch.arange(len(order))
            rows = rows[order]
        x = dist.gather_to_main(x, place)
        y = dist.gather_to_main(y, place)
    with dist.rank0_alone():
        report = {**fit_stitch(stitch, x, y, rows, held_from, cfg, seed, device), "stride": int(stride),
                  "minutes": (time.perf_counter() - started) / 60} if dist.is_main() else None
    del x, y, rows
    if dist.is_distributed():   # every rank leaves with rank 0's stitch, its buffers included (they move for the collective)
        stitch_home = next(stitch.parameters()).device
        stitch.to(dist.reduction_device()); dist.broadcast_state(stitch); stitch.to(stitch_home)
    return report


def build_slt_components(
    data_config: str = "configs/data.yaml", slt_config: str = "configs/dlm.yaml", inference_config: str = "configs/inference.yaml",
    include_dev: bool = False, language: str | None = None, bio_config: str = BIO_S1_CONFIG, resume: bool = False,
    device: torch.device | str = "cpu", dev_split: str = "dev",
) -> SLTComponents:
    # `device`: where the one-stream arm's stitch fit runs (train.py passes the run's device); the model itself stays on the CPU here.
    data_cfg = load_yaml(data_config)
    slt_cfg = load_yaml(slt_config)
    # Precedence: --language > config `language:` > active_languages. Reload on override so ${language} 
    # in checkpoint.dir (+ ar/baseline children) re-points at the right dataset.
    language = str(language or slt_cfg.get("language") or data_cfg.get("active_languages", ["asf"])[0])
    if language != slt_cfg.get("language"): slt_cfg = load_yaml(slt_config, language=language)
    # Stage 2 and the clean floor train on augmented windows (data/sampler.py materialize), with dlm.yaml's block, S1's recipe, or on plain
    # windows with augmentation.enabled false (None here, utils.load_yaml). Checked here, before any data loads: the sampler reads these keys
    # first in a DataLoader worker.
    aug = slt_cfg.get("augmentation", "no block")
    if aug is not None and not (isinstance(aug, dict) and set(aug) == {"fps", "frame_dropout", "body_part_dropout"}
                                and isinstance(aug["fps"], dict) and set(aug["fps"]) == {"min_factor", "max_factor"}): raise SystemExit(
        f"{slt_config}: augmentation must be dlm.yaml's block {{enabled, fps: {{min_factor, max_factor}}, frame_dropout, body_part_dropout}}, "
        f"but it is {aug!r}."
    )
    # Training decodes greedy (the pretrimmed translator's dev BLEU-4 and both stage-2 monitors), so no epoch is chosen under a beam search
    # that no stamp records; the offline monitor reads validation.num_beams (evaluation.systems.generation_kwargs). Beam rows are eval only.
    if int(slt_cfg["validation"]["num_beams"]) != 1: raise SystemExit(
        f"{slt_config}: validation.num_beams is {slt_cfg['validation']['num_beams']}, but training decodes greedy: set it to 1 (dlm.yaml). "
        f"eval.py --num-beams gives one offline row of an AR translator a beam size."
    )
    inference_cfg = load_yaml(inference_config)
    _inject_gate_geometry(slt_cfg, inference_cfg)
    # Resolve the S1 init before any data loads, so a missing S1 fails in seconds. 
    # Both pretrained branches then adapt, with caption gradients coupled through Ω.
    bio_init = slt_cfg.get("checkpoint", {}).get("bio_head_init")
    bio_cfg = load_yaml(bio_config, language=language)
    # `auto` DERIVES the path via the same resolver every other consumer uses, so pooled S1 is found by its pool key instead 
    # of a literal copied into this config. Hardcoded `checkpoints/bio_s1/multi_<pool>/model.pt` goes stale the moment 
    # `pretrain.languages` changes, and stage 2 would silently initialise the gate from another pool's head.
    if str(bio_init).lower() == "auto": bio_init = str(Path(checkpoint_dir(bio_cfg, default="checkpoints/bio_s1")) / "model.pt")
    gate_on = bool(slt_cfg["membership_gate"]["enabled"])
    # The clean floor (baseline_train.yaml: no S1 init, gate off) has no segmentation branch; every other recipe has one.
    segmentation_branch = gate_on or bool(bio_init)
    if segmentation_branch != (float(slt_cfg["lambda_bio"]) != 0.0): raise SystemExit(
        "lambda_bio must be > 0 exactly when the recipe has a segmentation branch (gate on or bio_head_init set): "
        "without a branch there is no BIO loss, and a branch without BIO supervision is incoherent. The pretrimmed-translator "
        "recipe sets lambda_bio: 0, bio_head_init: null and membership_gate.enabled: false. The architecture stamp "
        "and the duration fit key on lambda_bio, so the two must agree."
    )
    # Every segmentation branch starts from S1: the gate is on from step 0, and coupled to an untrained head it would
    # condition the translator on noise. bio_head_init is cwd-relative, so a wrong-cwd launch (Colab default dir) lands here.
    if segmentation_branch and not (bio_init and Path(bio_init).exists()): raise SystemExit(
        f"checkpoint.bio_head_init {bio_init!r} not found: stage 2 needs a trained S1 for its segmentation branch. "
        f"Run `train.py --stage train-bio` first, or fix the path/cwd."
    )
    if segmentation_branch:   # the gate picks spans with segmenter_decode from step 0, so S1 must have chosen its epoch under it too
        from evaluation.segmenter_eval import require_config_decode
        require_config_decode(load_checkpoint_meta(str(bio_init)), slt_cfg, bio_init, None)
    
    slt_cfg["decoder"] = str(slt_cfg.get("decoder", "dlm"))
    # The latest.pt check of run_epoch_loop, before the data load and the one-stream arm's stitch fit, so a repeated command without
    # --resume stops in seconds.
    require_resume_state(slt_cfg, resume)
    train_records, _ = load_language_records(data_cfg, language, split="train")
    slt_cfg["annotation_fingerprint"] = annotation_fingerprint(train_records)
    # Label-only duration prior of this language's train units; stamped in the checkpoint, so eval decodes with it.
    duration = DurationModel.fit(train_records) if float(slt_cfg["lambda_bio"]) else None
    slt_cfg["duration_model"] = duration.to_dict() if duration else None

    target_lang = data_cfg["languages"][language].get("target_lang", "en_XX")
    # The dev scores' text level is declared, not sniffed (metrics.text_level): the target language, and lowercase on both sides when
    # data.yaml sets lowercase_references, as every row of this corpus scores. Not stamped: a new meta key would refuse every --resume.
    slt_cfg["target_lang"] = target_lang
    slt_cfg["lowercase_references"] = bool(data_cfg["languages"][language].get("lowercase_references", False))
    if slt_cfg["lowercase_references"] and dist.is_main(): print(
        f"slt | dev text scores in lowercase (data.yaml languages.{language}.lowercase_references)", flush=True)
    # Uni-Sign front end. language_model.name picks the mT5 checkpoint and its tokenizer.
    lm_name = str(cfg_get(slt_cfg, "language_model", "name", default="google/mt5-base"))
    prompt_lang = PROMPT_LANG_BY_TARGET.get(str(target_lang or ""), "English")
    tokenizer = T5Tokenizer.from_pretrained(lm_name, legacy=False)
    front_end = UniSignMT5FrontEnd(mt5_name=lm_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)

    resolve_bio_class_weights(slt_cfg, train_records)
    collator = WindowCollator(
        tokenizer, max_text_tokens=int(slt_cfg["max_text_tokens"]),
        # The text canvas is sized to the batch, not to max_text_tokens: captions are ~15 tokens, far below that canvas, 
        # and every decoder forward runs the whole width. The collator keeps the canvas block-aligned.
        block_size=int(slt_cfg["block_size"]),
    )
    assert_targets_fit(train_records, tokenizer, collator.max_text_tokens, inference_cfg["buffer_cap_s"],
                       float(inference_cfg["boundary_stability"]["delta_s"]), f"{language}/train")
    # num_workers is pure throughput: anchors are index-driven (each realized once per epoch regardless of worker split)
    # and every draw reseeds its rng from the index (WindowSampler.spec_for), so no worker state matters.
    num_workers = int(slt_cfg.get("num_workers", 0))
    dev_dataset, dev_loader, monitor_records = None, None, None
    if include_dev:
        dev_records, _ = load_language_records(data_cfg, language, split=dev_split)
        assert_targets_fit(dev_records, tokenizer, collator.max_text_tokens, inference_cfg["buffer_cap_s"],
                           float(inference_cfg["boundary_stability"]["delta_s"]), f"{language}/{dev_split}")
        # Dev scores the same experimental unit as standard SLT training: 1 window per reliable sentence anchor 
        # (the dataset's length), not 1 per video. deterministic: the same dev windows every epoch.
        dev_dataset = StreamingWindowDataset(dev_records, slt_cfg=slt_cfg, inference_cfg=inference_cfg, deterministic=True)
        dev_loader = streaming_loader(  # checkpoint.dev_batch_size, a GLOBAL batch: each rank loads its part of it (evaluate_slt)
            dev_dataset, int(slt_cfg["checkpoint"]["dev_batch_size"]), collator, num_workers=num_workers, split_dev=True
        )
        hours = float(slt_cfg.get("checkpoint", {}).get("online_dev_hours") or 0.0)
        monitor = str(slt_cfg.get("checkpoint", {}).get("monitor", ""))
        if segmentation_branch and monitor.startswith("val_offline_"):
            monitor_records, total_s = dev_records, sum(r.pose.duration_s for r in dev_records)
        elif segmentation_branch and monitor.startswith("val_online_") and hours > 0:
            monitor_records, total_s = online_monitor_records(dev_records, hours, int(slt_cfg.get("seed", 42)))
        if monitor_records:
            scoring_cfg = load_yaml("configs/eval.yaml")["scoring"]   # the monitor reads a row's main score: no edge limit
            slt_cfg["monitor_edge"] = {"chunk_gap_s": float(scoring_cfg["chunk_gap_s"]),
                                       "tiou_thresholds": [float(t) for t in scoring_cfg["tiou_thresholds"]]}
            if dist.is_main(): print(f"slt | dev monitor: {len({r.pose.video_id for r in monitor_records})} "
                                     f"dev videos, {len(monitor_records)} streams, {total_s / 3600:.2f} h, "
                                     f"decode {segmenter_config_decode(slt_cfg)} (segmenter_decode)", flush=True)
    pretrained_path = resolve_pretrained(slt_cfg, data_cfg, language, default="checkpoints/openasl_pose_only_slt.pth")
    init_meta = require_translator_init(
        pretrained_path, segmentation_branch, language, inference_cfg, slt_cfg.get("annotation_fingerprint"),
        slt_cfg["decoder"], int(slt_cfg["block_size"]), slt_cfg["augmentation"],
    )
    # `pretrained_path` (the clean floor of this decoder for an arm, the released Uni-Sign file for the floor itself) is loaded inside
    # MisalignedSLTModel BEFORE the DLM [MASK]-token extension, so the block-diffusion decoder inherits its pose + LM weights; a DLM
    # arm then also loads its DLM floor's own canvas ([MASK] row, input table and output head).
    # The head's attention band is S1's (derived when S1 trained, from the K-stride horizon and its slowest train rate; no tensor
    # carries it). The arm stamps it, and eval builds the arm with it. No segmentation branch = no head. An S1 with no stamp has full
    # attention, which breaks spec §1.4 rule 16 (no commit lag), so it is refused.
    slt_cfg["bio_attention_radius_s"] = load_checkpoint_meta(str(bio_init)).get("bio_attention_radius_s") if segmentation_branch else None
    if segmentation_branch and slt_cfg["bio_attention_radius_s"] is None: raise SystemExit(
        f"{bio_init} stamps no bio_attention_radius_s (a full-attention head, spec §1.4 rule 16): retrain S1 (train-bio derives the band)."
    )
    model = MisalignedSLTModel(
        front_end=front_end, decoder=slt_cfg["decoder"], block_size=int(slt_cfg["block_size"]),
        # Shape MUST match S1 (train/bio_pretrain.py) or `bio_head_init` fails to strict-load — same keys build_bio_s1 reads.
        bio_hidden_dim=int(slt_cfg["bio"]["hidden_dim"]), bio_depth=int(slt_cfg["bio"]["depth"]),
        bio_nhead=int(slt_cfg["bio"]["nhead"]), bio_dropout=float(slt_cfg["bio"]["dropout"]),
        bio_conv_stem_layers=int(slt_cfg["bio"]["conv_stem_layers"]), bio_attention_radius_s=slt_cfg["bio_attention_radius_s"],
        pretrained_path=pretrained_path, segmentation_branch=segmentation_branch, 
        bio_adapter=bool(slt_cfg["bio"]["adapter"]),
        shared_pose_encoder=bool(slt_cfg.get("shared_pose_encoder", False)),
        # The one-stream arm's stitch (dlm.yaml stitch); only that arm reads the block.
        bio_stitch_hidden_dim=int(slt_cfg["stitch"]["hidden_dim"]) if slt_cfg.get("shared_pose_encoder") else None,
    )
    model.duration_model = duration
    # The gate picks the span that the text reads with the decode of the monitors and the rows (segmenter_decode); only the end term
    # reads the duration model's posterior (models.streaming_slt.MisalignedSLTModel.span_decoder).
    if segmentation_branch: model.span_decode = segmenter_config_decode(slt_cfg)
    # The train rows are the listed input types at each anchor (train_inputs: dlm.yaml all 3, the clean floor windows; data/sampler.py
    # spec_for). An arm's window hides every frame whose head output reads a window edge, as far as the arm's own modules reach
    # (train.bio_pretrain.edge_reach; the translator's encoder that the adapter adds has the same reach). The clean floor has no head.
    reach = edge_reach(model.bio_pose_encoder if model.bio_pose_encoder is not None else model.front_end.pose_encoder, model.bio_head) \
            if segmentation_branch else None
    train_dataset = StreamingWindowDataset(train_records, slt_cfg=slt_cfg, inference_cfg=inference_cfg, edge_reach=reach,
                                           inputs=slt_cfg["train_inputs"])
    slt_cfg["train_inputs"] = list(train_dataset.sampler.inputs)
    if segmentation_branch and dist.is_main(): print(f"slt | {len(train_dataset)} train rows/epoch, 1 per reliable unit: " + str(dict(Counter(
        train_dataset.sampler.spec_for(i)[1].input for i in range(len(train_dataset))))) + " (the draw made when the data are built; each epoch draws again)", flush=True)
    train_loader = streaming_loader(
        train_dataset, int(slt_cfg.get("batch_size", 4)), collator, num_workers=num_workers,   # the global batch
        # No length bucketing (S1 keeps it): the global grad clip (max_grad_norm) gives every batch
        # a full step, so a batch of short windows would weigh as much as a batch of long ones and over-weight short windows,
        # B frames near window edges and text-free batches. Random batches keep each window near its frame share. The clean
        # floor and the gated arms both build their train loader here. bucket_seed seeds the per-epoch shuffle.
        bucket_by_length=False, bucket_seed=int(slt_cfg.get("seed", 42)),
    )
    slt_cfg["resolved_inits"] = _resolved_inits(pretrained_path, init_meta, None, {})
    # The reader's frame geometry. A trained warm start (the clean floor here, S1 below) that read poses in another one warns now;
    # the released Uni-Sign file has no meta and is silent.
    warn_pose_geometry(init_meta, pretrained_path)
    slt_cfg["pose_geometry"] = POSE_GEOMETRY
    if not segmentation_branch: print(
        "slt | pretrimmed translator: no segmentation branch (lambda_bio 0, gate off, no bio_head_init). The released front end trains alone. "
        "S1's segmentation-adapted encoder stays out, so this row remains a faithful Uni-Sign transfer.", flush=True
    )
    else:
        blob = torch.load(str(bio_init), map_location="cpu")
        sd = blob.get("model", blob) if isinstance(blob, dict) else blob
        head_sd = {k[len("bio_head."):]: v for k, v in sd.items() if k.startswith("bio_head.")}
        # S1's head reads feat_dim; this bio_head reads front_end.bio_tap_dim (the LM's d_model). 
        # Must match for "S1 features == S2 initial features" and the strict-load.
        s1_dim = head_sd.get("input_proj.weight")
        if s1_dim is not None and int(s1_dim.shape[1]) != int(front_end.bio_tap_dim): raise ValueError(
            f"bio_head_init dim mismatch: S1 head reads {int(s1_dim.shape[1])}-d features but this SLT model's "
            f"bio_tap is {int(front_end.bio_tap_dim)}-d ({lm_name}). Retrain train-bio with feat_dim="
            f"{int(front_end.bio_tap_dim)} (bio_pretrain.yaml)."
        )
        _s1_meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
        # PROVENANCE, same rule as evaluation/systems.py load_segmenter: the checkpoint records which pool trained it, 
        # and stage 2 must not warm-start the gate from a head of another pool (another model, a silent failure).
        if _s1_meta.get("pretrain_pool") != pool_key(bio_cfg): raise SystemExit(
            f"{bio_init} was trained on pool {_s1_meta.get('pretrain_pool')!r}, but bio_pretrain.yaml expects "
            f"{pool_key(bio_cfg)!r}. Retrain S1 for this pool, or point checkpoint.bio_head_init at the matching model."
        )
        # Labels are not in any tensor either: an S1 trained before a re-convert or re-measured cuts would load silently. Stage 2 compares
        # only the target's own stamp, so no other corpus loads and it runs on a machine that holds only its target.
        require_segmenter_data(_s1_meta, {language: slt_cfg["annotation_fingerprint"]}, inference_cfg["buffer_cap_s"], str(bio_init))
        warn_pose_geometry(_s1_meta, str(bio_init))
        model.bio_head.load_state_dict(head_sd, strict=True)
        slt_cfg["resolved_inits"] = _resolved_inits(pretrained_path, init_meta, bio_init, _s1_meta)
        # Two streams: S1's pose encoder goes into the SEGMENTATION branch; the translator keeps the clean translator's encoder.
        # One-stream arm (shared_pose_encoder): ONE encoder, the clean translator's (the semantic stream owns it, as in TwInS); S1's
        # head, unchanged, reads it through the stitch, an MLP fitted to map its tokens to S1's encoder tokens (fit_s1_stitch).
        # Loading S1's encoder here instead would lower the translator's starting BLEU a lot (docs/implementation_notes.md).
        pose_sd = {k[len("pose_encoder."):]: v for k, v in sd.items() if k.startswith("pose_encoder.")}
        if not pose_sd: raise ValueError(f"{bio_init} carries no pose encoder; the segmentation branch needs S1's encoder")
        if model.bio_pose_encoder is not None:
            model.bio_pose_encoder.load_state_dict(pose_sd, strict=True)
            print(f"slt | loaded S1 from {bio_init}: {len(head_sd)} head tensors and {len(pose_sd)} pose-encoder "
                  f"tensors into the segmentation branch; the translator keeps its own encoder", flush=True)
        else:
            # The fit reads every train row once, as drawn when the dataset is built (each training epoch draws again), and holds out
            # the run's dev windows (whole dev videos), at every (2 x the encoder's reach + 1)-th frame; rank r reads the rows
            # r, r + N, ... of each and rank 0 fits on all of them (fit_s1_stitch). A resumed run skips it: latest.pt holds the trained
            # stitch, which its load would write over the fit. No row has more real frames than its window spec, read without poses
            # (augmentation only drops frames): the bound of the fit's buffers.
            fit = None
            if not resume:
                if dev_dataset is None: raise ValueError(
                    "The one-stream arm's stitch fit holds out the run's dev windows: build it with include_dev=True (train.py does).")
                rows, held = (Subset(d, range(dist.rank(), len(d), dist.world_size())) for d in (train_dataset, dev_dataset))
                fit = fit_s1_stitch(model, pose_sd, *(DataLoader(
                    part, batch_size=dist.per_rank_batch_size(int(slt_cfg["checkpoint"]["dev_batch_size"])), collate_fn=collator,
                    num_workers=num_workers,
                ) for part in (rows, held)), [train_dataset.sampler.spec_frames(train_dataset.effective_index(i)) for i in rows.indices]
                   + [dev_dataset.sampler.spec_frames(i) for i in held.indices], len(train_dataset),
                   2 * encoder_reach(model.front_end.pose_encoder) + 1, slt_cfg["stitch"], int(slt_cfg["seed"]), device)
            width = f"{int(front_end.bio_tap_dim)}-{int(slt_cfg['stitch']['hidden_dim'])}-{int(front_end.bio_tap_dim)}"
            if dist.is_main(): print(
                f"slt | one stream: 1 pose encoder, the pretrimmed translator's, feeds the translator and the BIO head, and both losses "
                f"train it. The head is S1's ({len(head_sd)} tensors, {bio_init}); it reads this encoder through the stitch, an MLP "
                f"{width} that trains with every other module. " + (
                    "Resumed run: the stitch comes from latest.pt, so no new fit." if fit is None else
                    f"Fit without labels on {fit['fit_rows']} train rows, held out on {fit['held_rows']} dev windows: "
                    f"{fit['fit_frames'] + fit['held_frames']} frames (1 of every {fit['stride']} real frames of a row), R^2 "
                    f"{fit['r2']:.3f} on the dev windows (whole dev videos that the fit never reads) at epoch {fit['best_epoch']} of "
                    f"{fit['epochs']}, {fit['minutes']:.1f} min for the extraction, the gather and the fit.") + " S1's own pose encoder only gives the fit its targets: it is not in the model",
                flush=True)

    print(f"slt | model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, all trained at 1 rate", flush=True)
    return SLTComponents(
        model=model, tokenizer=tokenizer, train_loader=train_loader, dev_loader=dev_loader, 
        slt_cfg=slt_cfg, inference_cfg=inference_cfg, monitor_records=monitor_records, 
        checkpoint_meta=_training_meta(slt_cfg, inference_cfg, language)
    )


@torch.no_grad()
def evaluate_slt(
    model: MisalignedSLTModel, loader: DataLoader, device: torch.device, 
    slt_cfg: dict, inference_cfg: dict | None = None, monitor_records: list | None = None,
) -> dict[str, float]:
    """Every run logs its dev-window losses (val_loss and its parts). Joint training logs the row scores of whole dev videos for the 
    protocol that checkpoint.monitor names: val_offline_* for offline row (evaluation.offline.offline_dev_scores), val_online_* 
    for online row, the streaming FSM (evaluation.online.online_dev_scores). The monitor is the main score of the row, the F1 of the
    tIoU match at tIoU avg (no edge limit), x the DVC BLEU-4 of the pairs that count (the same avg), logged with both parts and
    events per gold unit."""
    was_training = model.training
    model.eval()
    arm = float(slt_cfg["lambda_bio"]) != 0.

    spd_cfg = slt_cfg["spd"]
    gate_cfg = slt_cfg["membership_gate"]

    gate_kwargs = dict(gate_enabled=bool(gate_cfg["enabled"]), gate_eps=float(gate_cfg["eps"]))
    # Under torchrun each rank computes its part of every global dev batch (data.loader.RankSlice, as in training), its losses by the
    # training rule for a global batch (training_loss in train_step: their mean over the ranks is the 1-process value), and the rows
    # and texts are merged in the 1-process order below. A last batch with fewer windows than ranks reaches every rank whole and is
    # computed on rank 0 alone.
    parts = getattr(loader, "batch_sampler", None)
    whole_last = isinstance(parts, RankSlice) and parts.short
    rows_by_part: dict[tuple[int, int], dict[str, float]] = {}
    texts_by_part: dict[tuple[int, int], tuple[list[str], list[str]]] = {}
    for b, batch in enumerate(loader):
        whole = whole_last and b == len(parts) - 1
        if whole and dist.rank(): continue
        batch = move_to_device(batch, device)
        frames = _gate_frames(gate_cfg, batch)
        with nullcontext() if whole else dist.train_step(): output = training_loss(model, batch, slt_cfg)
        row = {k: float(v.detach().cpu().item()) for k, v in output.logs.items() if v.numel() == 1}
        # Frame metrics of forward_loss's own head pass feed mean_logs' B-collapse alarm only. They are not logged.
        if arm and output.bio_logits is not None:
            row.update(bio_frame_metrics(output.bio_logits, batch["bio_labels"], prefix="bio"))
        rows_by_part[(b, dist.rank())] = row
        if arm: continue  # the arm's translation is scored on the deployed path below, not on dev windows

        # The clean floor translates every supervised dev window: dev BLEU selects its checkpoint.
        supervised = batch.get("translation_supervised")
        targets = batch.get("translation_targets", [])
        if isinstance(supervised, torch.Tensor) and supervised.any():
            idx = supervised.nonzero(as_tuple=False).flatten()
            _, tokens, _, _ = model.generate_from_poses(
                poses=batch["poses"][idx], frame_mask=batch["frame_mask"][idx],
                timestamps_s=batch.get("timestamps_s", None)[idx] if batch.get("timestamps_s") is not None else None,
                max_text_tokens=int(slt_cfg["max_text_tokens"]), tau_dec=float(spd_cfg["tau_dec"]),
                spd_top_k=int(spd_cfg["top_k"]), spd_renormalize=bool(spd_cfg["renormalize"]),
                gate_min_span_frames=[frames["gate_min_span_frames"][i] for i in idx.tolist()], **gate_kwargs,
            )
            # Unit separator is scored as a space, as the boundary sweep and the rows score it: sacrebleu would split literal sentinel into 7 tokens.
            preds = [join_unit_text(t) for t in model.tokenizer.batch_decode(tokens.detach().cpu(), skip_special_tokens=True)]
            refs = [join_unit_text(t.get("text", "") if isinstance(t, dict) else getattr(t, "text", ""))
                    for t in (targets[int(i)] for i in idx.detach().cpu().tolist())]
            texts_by_part[(b, dist.rank())] = (preds, refs)

    rows_by_part, texts_by_part = dist.merge_dicts(rows_by_part), dist.merge_dicts(texts_by_part)
    by_batch: dict[int, list[dict[str, float]]] = {}
    for (b, _), row in sorted(rows_by_part.items()): by_batch.setdefault(b, []).append(row)
    rows = [{k: sum(r[k] for r in part if k in r) / sum(k in r for r in part) for k in {k for r in part for k in r}}
            for part in (by_batch[b] for b in sorted(by_batch))]   # 1 row per batch: the mean over the ranks that computed it
    pred_texts = [p for key in sorted(texts_by_part) for p in texts_by_part[key][0]]
    ref_texts = [r for key in sorted(texts_by_part) for r in texts_by_part[key][1]]
    metrics = {k: v for k, v in mean_logs(rows, prefix="val").items() if k.endswith("loss")}
    level = text_level(slt_cfg.get("target_lang"), slt_cfg.get("lowercase_references"))   # the corpus's text level, as its rows score
    if arm and monitor_records:  # whole dev videos, scored as a row: offline row (val_offline_*) or online row (val_online_*)
        if str(slt_cfg.get("checkpoint", {}).get("monitor", "")).startswith("val_offline_"):
            from evaluation.offline import offline_dev_scores as dev_scores
        else: from evaluation.online import online_dev_scores as dev_scores
        scores = dev_scores(model, monitor_records, inference_cfg, slt_cfg, slt_cfg["monitor_edge"], level)
        metrics.update({f"val_{k}": v for k, v in scores.items()})

    if was_training: model.train()
    if pred_texts:
        metrics["val_translation_bleu4"] = text_scores(
            pred_texts, ref_texts, level, keys=("bleu4",)
        )["bleu4"]
        # Hyp/ref length ratio in WORD tokens, matching BLEU's brevity penalty: below 1 and falling 
        # across epochs means the decode ends ever earlier, which BLEU punishes as brevity.
        total_ref = sum(len(r.split()) for r in ref_texts)
        metrics["val_translation_len_ratio"] = float(sum(len(p.split()) for p in pred_texts)) / max(1, total_ref)
    return metrics


def gradient_probe(model, bio_part: torch.Tensor, text_part: torch.Tensor) -> dict[str, float]:
    """The gradients of the 2 weighted losses of 1 batch (`bio_part` = lambda_bio x L_BIO, `text_part` = lambda_trans x L_text, the 2
    summands of the step's loss) on the weights that the BIO loss trains: the segmentation branch (in the one-stream arm the shared
    encoder, the stitch and the head). Adam scales each weight's gradient by its own running size, so the loss weights act only where
    both losses reach the same weights; the text loss reaches these through the end-side term (and the one-stream arm's shared encoder).
    Logs their norms, the text norm over the BIO norm and, when both are non-zero, their cosine. It reads the batch's graph and
    changes no gradient that the step applies. Under torchrun both gradients are averaged over the ranks, as the step averages its
    gradient, and the text backward also runs the BIO graph at weight 0, so every rank runs the same BatchNorm collectives in the
    same order. fp16 autocast: the losses are scaled before the backward (a fixed scale against underflow); a non-finite result logs
    nothing. Empty when the BIO loss trains no weight."""
    params = [p for p in model.parameters() if p.requires_grad]
    fp16 = torch.cuda.is_available() and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
    scale = 2.0 ** 12 if fp16 else 1.0
    bio = torch.autograd.grad(bio_part * scale, params, retain_graph=True, allow_unused=True)
    shared = [p for p, g in zip(params, bio) if g is not None]
    if not shared: return {}
    text = torch.autograd.grad((text_part + 0.0 * bio_part) * scale, shared, retain_graph=True, allow_unused=True)
    flat = lambda grads: torch.cat([(g if g is not None else torch.zeros_like(p)).to(torch.promote_types(p.dtype, torch.float32)).reshape(-1)
                                    for g, p in zip(grads, shared)])   # at least float32, so a float64 model keeps its precision
    b, t = flat([g for g in bio if g is not None]) / scale, flat(text) / scale
    if dist.is_distributed(): b, t = dist.all_sum(b) / dist.world_size(), dist.all_sum(t) / dist.world_size()
    nb, nt = b.norm(), t.norm()
    if not (bool(torch.isfinite(nb)) and bool(torch.isfinite(nt))) or float(nb) == 0.0: return {}
    out = {"grad_bio_norm": float(nb), "grad_text_norm": float(nt), "grad_text_over_bio": float(nt / nb)}
    if float(nt) > 0.0: out["grad_cosine"] = float(torch.dot(b, t) / (nb * nt))
    return out


def training_loss(model, batch: dict, slt_cfg: dict) -> SLTLossOutput: # The actual AR/DLM training objective.
    """`model.forward_loss` with the run's settings. Under torchrun, in a training step, its text loss (a mean over this rank's target
    tokens) is weighted to the mean over the global batch's target tokens: world_size x this rank's token sum / all ranks' tokens. A
    rank with no text row then adds 0, not a full-weight mean. The BIO loss already reads the global batch (train.losses)."""
    oput_cfg = slt_cfg["oput"]
    gate_cfg = slt_cfg["membership_gate"]
    dice_weight = float(slt_cfg["bio"]["dice_loss_weight"])
    output = model.forward_loss(
        batch, lambda_trans=float(slt_cfg["lambda_trans"]), lambda_bio=float(slt_cfg["lambda_bio"]),
        dice_weight=dice_weight, bio_class_weights=bio_class_weight_tensor(slt_cfg["bio"]["class_weights"]),
        oput_t_low=float(oput_cfg["t_low"]), oput_t_high=float(oput_cfg["t_high"]),
        oput_label_smoothing=float(oput_cfg["label_smoothing"]),
        gate_enabled=bool(gate_cfg["enabled"]),
        # Same δ as the inference commit gate's delta_s (configs/inference.yaml): the coverage tolerance of text routing.
        gate_eps=float(gate_cfg["eps"]), gate_detach_omega=bool(gate_cfg["detach_omega"]), 
        **_gate_frames(gate_cfg, batch),
    )
    if not dist.in_step(): return output
    supervised, labels = batch.get("translation_supervised"), (batch.get("target_tokens") or {}).get("labels")
    tokens = torch.zeros((), dtype=torch.float64, device=output.loss.device)   # fp64: a float32 count ratio rounds the weight
    if supervised is not None and labels is not None and bool(supervised.any()):
        # The positions that the text loss averages over: the target tokens (AR, models.unisign ar_loss), or the DLM canvas positions,
        # padding to the block edge as EOS included (models.block_diffusion _prepare_x0).
        labels = labels[supervised.to(labels.device)]
        valid = model.dlm_decoder._prepare_x0(labels)[1] if model.decoder_type == "dlm" else labels != -100
        tokens = valid.sum().to(tokens)
    weight = dist.world_size() * tokens / dist.all_sum(tokens.clone()).clamp(min=1)
    text = output.translation_loss * weight.to(output.translation_loss.dtype)
    loss = output.loss + float(slt_cfg["lambda_trans"]) * (text - output.translation_loss)
    logs = {**output.logs, "translation_loss": text.detach(), "loss": loss.detach()}   # their mean over ranks is the global value
    return replace(output, loss=loss, translation_loss=text, logs=logs)


def train_slt_epochs(
    model: MisalignedSLTModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, 
    epochs: int, slt_cfg: dict, inference_cfg: dict | None = None, dev_loader: DataLoader | None = None, 
    resume: bool = False, monitor_records: list | None = None, checkpoint_meta: dict | None = None,
) -> int:
    decoder_name = getattr(model, "decoder_type", "dlm")
    if float(slt_cfg["lambda_bio"]) != 0. and dist.is_main():
        offline = str(slt_cfg.get("checkpoint", {}).get("monitor", "")).startswith("val_offline_")
        name, row = ("offline", "--offline row on whole dev set: each video segmented once, each span translated") \
                    if offline else ("online", "--online row, streaming FSM, on whole dev videos up to checkpoint.online_dev_hours")
        print(f"slt | checkpoint monitor: val_{name}_score = val_{name}_f1 (the main score: the F1 of the tIoU match at tIoU avg, "
              f"no edge limit) x val_{name}_bleu4 (the DVC BLEU-4 of the pairs that count, the same avg) ({row})", flush=True)

    # Every grad_probe_every-th training batch (dlm.yaml; 0: never) also logs the gradient probe (gradient_probe); the count is the
    # same on every rank, so all ranks probe the same batches.
    probe_every, batches = int(slt_cfg.get("grad_probe_every") or 0), [0]

    def step_fn(batch, epoch: int):
        output = training_loss(model, batch, slt_cfg)
        logs = {k: float(v.detach().cpu().item()) for k, v in output.logs.items() if v.numel() == 1}
        batches[0] += 1
        if probe_every and float(slt_cfg["lambda_bio"]) and batches[0] % probe_every == 0: logs.update(gradient_probe(
            model, float(slt_cfg["lambda_bio"]) * output.bio_loss, float(slt_cfg["lambda_trans"]) * output.translation_loss))
        return output.loss, logs

    def evaluate_fn(epoch: int):
        return evaluate_slt(model, dev_loader, device, slt_cfg=slt_cfg, inference_cfg=inference_cfg, monitor_records=monitor_records)

    # An arm writes 4 files: model.pt, the best of the monitor (F1 x BLEU-4, both tasks), the best of each factor, and the last epoch. The
    # BLEU-4 file reads the DVC BLEU-4 of the counted pairs, so it also moves with the F1. The pretrimmed translator has 1 task and 1 file.
    monitor = str(slt_cfg["checkpoint"]["monitor"])
    prefix = monitor[:-len("_score")] if monitor.endswith("_score") else None
    return run_epoch_loop(
        name=f"slt-{decoder_name}", model=model, loader=loader, optimizer=optimizer, device=device, epochs=epochs,
        cfg=slt_cfg, step_fn=step_fn, evaluate_fn=evaluate_fn, default_monitor="val_loss", default_mode="min",
        dev_loader=dev_loader, resume=resume, checkpoint_meta=checkpoint_meta,
        also_best=(BestFile(f"{prefix}_f1", "model_seg.pt", "bestseg"), BestFile(f"{prefix}_bleu4", "model_text.pt", "besttext")) if prefix else (),
    )
