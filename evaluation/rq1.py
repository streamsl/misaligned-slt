# RQ1: the controlled boundary sweep (eval.py --rq 1).
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from tqdm import tqdm
import argparse, json

import numpy as np
import pandas as pd
from poses import load_pose_window
from data.loader import annotation_fingerprint, ANNOTATION_PROTOCOL, load_language_records, VideoRecord
from infer.stream import split_unit_text
from metrics import char_level_for_target, compute_text_metrics
from evaluation.systems import (
    _arm_variant_tokens, _build_eval_model, _decode_provenance, _method_config_path, 
    _parse_grid, _translate_windows, model_annotation, rq2_translator_token
)
from utils import load_yaml, pick_device, target_language

@dataclass(frozen=True)
class ControlledWindow:
    video_id: str
    reference: str
    gt_start_s: float
    gt_end_s: float
    window_start_s: float
    window_end_s: float
    delta_head_s: float     # realized offset in seconds (= grid_head * duration if relative)
    delta_tail_s: float
    grid_head: float = 0.0  # grid coordinate for grouping (fraction of duration if relative)
    grid_tail: float = 0.0


def controlled_windows(
    records: list[VideoRecord], grid: list[float], relative: bool = True, max_sentences: int | None = None, 
    tail_grid: list[float] | None = None, drop_counts: dict[tuple[float, float], int] | None = None,
) -> list[ControlledWindow]:
    """RQ1 signed-offset boundaries: start = gt_start + delta_head, end = gt_end + delta_tail. The loaded input
    includes 1 frame after the requested end so a zero-offset target contains its terminator.

    `relative=True` (default): grid values are FRACTIONS of sentence duration — absolute seconds mix regimes (0.3s
    destroys a 1s sentence, not a 10s one) at 1 x-point.

    `tail_grid` (default `grid`) decouples the axes: extension (head < 0 / tail > 0) needs a continuous timeline
    and clamps on pre-trimmed clips; truncation (head ≥ 0, tail ≤ 0) does not.
    """
    windows: list[ControlledWindow] = []
    count = 0
    tail_values = grid if tail_grid is None else tail_grid
    for record in records:
        for span in record.sentences:
            if not getattr(span, "reliable", True): continue  # quarantined: no correct target exists
            if max_sentences is not None and count >= int(max_sentences): return windows
            count += 1
            duration = max(1e-6, float(span.end_s) - float(span.start_s))
            for gh in grid:
                for gt in tail_values:
                    dh = float(gh) * duration if relative else float(gh)
                    dt = float(gt) * duration if relative else float(gt)
                    start_s = max(0.0, float(span.start_s) + dh)
                    boundary_end_s = min(float(record.pose.duration_s), float(span.end_s) + dt)
                    # The boundary is at boundary_end_s. The input also needs the following frame, which carries
                    # O or the next B and makes a zero-offset target complete under the FSM rule.
                    end_s = min(float(record.pose.duration_s), boundary_end_s + 1.0 / float(record.pose.fps))
                    if boundary_end_s <= start_s:
                        # Fully truncated: dropped but COUNTED — dropping silently biases the row 
                        # toward longer sentences (dropped_fraction exposes it).
                        if drop_counts is not None: 
                            drop_counts[(float(gh), float(gt))] = drop_counts.get((float(gh), float(gt)), 0) + 1
                        continue
                    windows.append(ControlledWindow(
                        video_id=record.video_id, reference=span.text,
                        gt_start_s=float(span.start_s), gt_end_s=float(span.end_s), 
                        window_start_s=start_s, window_end_s=end_s,
                        # REALIZED offsets, not requested dh/dt: the request overstates severity where pre-trimmed
                        # data clamps. grid_head/grid_tail keep the request, for grouping.
                        delta_head_s=start_s - float(span.start_s), delta_tail_s=boundary_end_s - float(span.end_s),
                        grid_head=float(gh), grid_tail=float(gt),
                    ))
    return windows


def run_rq1(args: argparse.Namespace) -> "pd.DataFrame":
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run RQ1 on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    method_cfg = load_yaml(_method_config_path(args), language=args.language)
    # Optional per-run search override. Main comparisons use the greedy default.
    if getattr(args, "num_beams", None): method_cfg.setdefault("validation", {})["num_beams"] = int(args.num_beams)
    _rq1_beams = int(method_cfg.get("validation", {}).get("num_beams", 1))
    if args.method == "baseline" and _rq1_beams != 1: print(
        f"[rq1] WARNING: baseline decodes with num_beams={_rq1_beams} while the ar/dlm arms are greedy — "
        f"this run is not search-budget matched.", flush=True
    )
    records, _ = load_language_records(data_cfg, args.language, split=args.split)

    rq_cfg = eval_cfg.get("rq1", {})
    mode = str(args.severity_mode or rq_cfg.get("severity_mode", "relative"))
    relative = mode == "relative"
    if relative: default_key = "smoke_grid_rel" if args.smoke else "severity_grid_rel"
    else: default_key = "smoke_grid_s" if args.smoke else "severity_grid_s"
    grid = _parse_grid(args.severity_grid_s, rq_cfg.get(default_key, [0.0]))
    
    # Per-axis grids, falling back to the shared one (controlled_windows: which signs a corpus supports).
    head_grid = _parse_grid(args.severity_grid_head, grid)
    tail_grid = _parse_grid(args.severity_grid_tail, grid)
    max_sentences = args.num_sentences
    if max_sentences is None and args.smoke: max_sentences = int(rq_cfg.get("smoke_num_sentences", 10))
    drop_counts: dict[tuple[float, float], int] = {}
    windows = controlled_windows(
        records, head_grid, relative=relative, max_sentences=max_sentences, 
        tail_grid=tail_grid, drop_counts=drop_counts
    )
    records_by_id = {record.video_id: record for record in records}
    device = pick_device(args.device)
    inference_cfg = load_yaml(args.inference_config)
    model, tokenizer = _build_eval_model(
        args.method, args.checkpoint, args.language, data_cfg, method_cfg, inference_cfg, device
    )
    labels = model_annotation(model, data_cfg, args.language)
    arm_token = rq2_translator_token(args, getattr(model, "checkpoint_path", None))
    variant = _arm_variant_tokens(args, model.checkpoint_path) \
                if args.method != "baseline" and getattr(model, "checkpoint_path", None) else []

    if args.output and (missing := [t.lstrip("-") for t in variant if t.lstrip("-") not in Path(args.output).name]): 
        raise SystemExit(
            f"{model.checkpoint_path} is a variant arm ({''.join(variant)}): put {', '.join(map(repr, missing))} "
            f"in --output ({args.output}) so it cannot overwrite the main arm's sweep."
        )
    batch_size = max(1, int(rq_cfg.get("batch_size", 16)))
    grouped: dict[tuple[float, float], dict[str, list]] = {}
    rows = []

    # Length-sorted batches keep padding minimal, but the sort key is PREDICTED from window's own floor/ceil framing 
    # (data/sampler.spec_frames) rather than from a loaded array: severity grid emits 1 window/sentence per grid point, 
    # so materialising all of them before 1st translation holds the whole split's poses in ANONYMOUS memory. Loading 
    # inside the batch loop caps it at 1 batch and no extra I/O: every window is still read exactly once.
    def _predicted_frames(w: ControlledWindow) -> int:
        pose = records_by_id[w.video_id].pose
        fps = float(pose.fps)
        start_f = int(np.floor(max(0.0, w.window_start_s) * fps))
        end_f = int(np.ceil(min(w.window_end_s, pose.duration_s) * fps))
        return max(1, end_f - start_f + 1)

    ordered = sorted(windows, key=_predicted_frames)
    chunks = [ordered[s:s + batch_size] for s in range(0, len(ordered), batch_size)]
    # GT constructs and scores the controlled windows; the decoder receives no reference-derived frontier or onset flag.
    for spec_chunk in tqdm(chunks, desc="Translating windows"):
        chunk = []
        for window in spec_chunk:
            record = records_by_id[window.video_id]
            poses, timestamps = load_pose_window(record.pose, window.window_start_s, window.window_end_s, normalize=True)
            if poses.shape[0] == 0: continue
            chunk.append((window, poses, timestamps))

        if not chunk: continue
        results = _translate_windows(
            model=model, tokenizer=tokenizer, method=args.method, method_cfg=method_cfg, inference_cfg=inference_cfg, 
            device=device, items=[(poses, timestamps, w.window_start_s, float(records_by_id[w.video_id].pose.fps)) 
                                  for (w, poses, timestamps) in chunk]
        )
        # Sequential decoder passes of the chunk's decode (one decode per chunk). Rows decode together, so at
        # batch_size > 1 this is the chunk's count; measure latency at rq1.batch_size 1, where it is per window.
        passes = getattr(model, "last_decode_passes", None)
        for (window, _poses, _ts), (prediction, confidence, gate_skip) in zip(chunk, results):
            pose = records_by_id[window.video_id].pose
            key = (window.grid_head, window.grid_tail)  # group by grid coordinate (fraction in relative mode)
            grouped.setdefault(key, {
                "predictions": [], "references": [], "confidences": [], "gate_skips": [], "passes": [],
                "head_s": [], "tail_s": [], "req_head_s": [], "req_tail_s": [],
            })
            duration = window.gt_end_s - window.gt_start_s
            grouped[key]["predictions"].append(" ".join(split_unit_text(prediction)))
            grouped[key]["references"].append(window.reference)
            grouped[key]["confidences"].append(confidence)
            grouped[key]["gate_skips"].append(bool(gate_skip))

            # A window the deployed FSM would skip is decoded here only to report pessimistic text bound.
            if passes is not None and not gate_skip: grouped[key]["passes"].append(int(passes))
            grouped[key]["head_s"].append(window.delta_head_s)
            grouped[key]["tail_s"].append(window.delta_tail_s)
            grouped[key]["req_head_s"].append(window.grid_head * duration if relative else window.grid_head)
            grouped[key]["req_tail_s"].append(window.grid_tail * duration if relative else window.grid_tail)
            rows.append({
                **asdict(window), "prediction": prediction, "mean_confidence": confidence, 
                "gate_would_skip": bool(gate_skip), "decoder_passes": passes, "video_id": pose.video_id,
                **{k: getattr(window, k) + pose.offset_s 
                   for k in ("gt_start_s", "gt_end_s", "window_start_s", "window_end_s")},
            })

    severity = []
    _char_level = char_level_for_target(target_language(data_cfg, args.language))
    for (gh, gt), values in tqdm(sorted(grouped.items()), desc="Computing severity"):
        confs = values["confidences"]
        head_s, tail_s = values["head_s"], values["tail_s"]
        clamped = sum(1 for req, real in zip(values["req_head_s"], head_s) if abs(req - real) > 1e-3)
        clamped += sum(1 for req, real in zip(values["req_tail_s"], tail_s) if abs(req - real) > 1e-3)
        dropped = int(drop_counts.get((float(gh), float(gt)), 0))
        severity.append({
            "windows": len(values["predictions"]),
            "grid_head": gh, "grid_tail": gt,  # fraction of sentence duration in relative mode, else seconds
            "delta_head_s_mean": float(sum(head_s) / len(head_s)) if head_s else 0.0,
            "delta_tail_s_mean": float(sum(tail_s) / len(tail_s)) if tail_s else 0.0,
            "clamped_fraction": float(clamped) / max(1, 2 * len(head_s)),
            # High -> this row averages over a longer-sentence subset.
            "dropped_fraction": float(dropped) / max(1, dropped + len(values["predictions"])),
            "mean_translation_confidence": float(sum(confs) / len(confs)) if confs else 0.0,
            # DLM's parallelism as run: sequential decoder passes per decode (see the chunk loop). 
            # Clean point at batch_size 1 gives BLEU passes per sentence (the decode cost at fixed tau_dec); 
            # AR arm's count is its generate() length (N cached steps + confidence pass).
            "mean_decoder_passes": float(sum(values["passes"]) / len(values["passes"])) if values["passes"] else None,
            "text_metrics": compute_text_metrics(values["predictions"], values["references"], char_level=_char_level),
            # GATED methods only (baseline skip rate 0). The FSM SKIPS no-span windows by design; force-decoding
            # them against the inert Ω gives near-empty hallucinations whose brevity penalty tanks the cell's
            # corpus BLEU superlinearly. Read WITH gate_skip_rate; text_metrics above is the pessimistic bound.
            "gate_skip_rate": float(sum(values["gate_skips"])) / max(1, len(values["gate_skips"])),
        })
        if any(values["gate_skips"]):  # omitted when nothing is skipped: it would equal text_metrics exactly
            severity[-1]["text_metrics_decoded_only"] = compute_text_metrics(
                [p for p, s in zip(values["predictions"], values["gate_skips"]) if not s],
                [r for r, s in zip(values["references"], values["gate_skips"]) if not s],
                char_level=_char_level,
            )
    # Sweeps are expensive and the artifacts feed the paper plots.
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "rq": "1", "language": args.language, "split": args.split, "method": args.method, "translator": arm_token,
            "annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(records),
            # Per-axis grids as ACTUALLY swept: with head/tail overrides `grid` alone misdescribes the axes.
            "severity_mode": mode, "grid": grid, "grid_head": head_grid, "grid_tail": tail_grid,
            # Decode provenance, mirroring RQ2's events stamp, so 2 sweeps under different flags 
            # stay distinguishable and cross-method RQ1 comparisons never mix decode budgets.
            "provenance": {
                "num_beams": int(method_cfg.get("validation", {}).get("num_beams", 1)) if args.method == "baseline" else 1,
                **_decode_provenance(args.method, method_cfg),
                "gate": bool(method_cfg.get("membership_gate", {}).get("enabled", False)),
                "checkpoint": getattr(model, "checkpoint_path", None),   # the file actually loaded
                "model_annotation": labels,   # its train labels and its inits', beside the train labels at eval time
                "segmentation_decode": "none" if args.method == "baseline" else "semi_markov_viterbi",
                "batch_size": batch_size,     # the unit of mean_decoder_passes: per window at 1, per chunk above
            },
            "windows": len(rows), "severity": severity, "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[rq1] wrote {out}", flush=True)
    return pd.json_normalize(severity, sep=".").T # 1 row per (grid_head, grid_tail) severity point
