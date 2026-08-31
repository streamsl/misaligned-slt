from __future__ import annotations
from dataclasses import asdict, dataclass
import argparse, json

from tqdm import tqdm
from typing import Any
from pathlib import Path
from IPython.display import display

import torch
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)         # show all columns
pd.set_option("display.max_colwidth", None)        # don't truncate long text in cells
pd.set_option("display.width", None)               # auto-detect terminal width
pd.set_option("display.expand_frame_repr", False)  # don't wrap columns into blocks

from poses import load_pose_window
from data.windowing import BIO, make_bio_labels
from data.loader import VideoRecord, load_language_records
from data.batch import frame_mask_for, repeat_last_frame

from transformers import T5Tokenizer, AutoTokenizer
from models.unisign import UniSignMT5FrontEnd, UniSignMBartFrontEnd, load_unisign_pretrained, prompt_lang_for_target
from models.streaming_slt import MisalignedSLTModel
from models.checkpointing import load_checkpoint_meta, load_model_checkpoint
from moryossef26.infer import duration_decode_tags, evaluate_segmenter_whole_video
from infer.duration_decode import duration_decode_params, fit_duration_prior
from infer.stability import TAU_GRID, group_tracks, build_policies, score_policy
from metrics import Segment, match_segments, moryossef_segment_metrics, segmentation_prf, compute_text_metrics, char_level_for_target
from utils import checkpoint_dir, load_yaml, language_model_name, pick_device, resolve_inference, resolve_pretrained, pool_key, target_language


@dataclass(frozen=True)
class PredictionEvent:
    video_id: str
    start_s: float
    end_s: float
    text: str | None = None
    flagged_partial: bool = False
    commit_time_s: float | None = None  # stride time the commit fired (streaming only)

    @property
    def segment(self) -> Segment:
        return Segment(self.start_s, self.end_s)


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


def save_prediction_file(predictions: dict[str, list[Segment]], path: str | Path, provenance: dict | None = None) -> Path:
    # Predicted spans + WHO produced them. Without the stamp a predictions file is just spans, and the arch/decode
    # that made it is unrecoverable — calibration could silently inherit the wrong decode.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"video_id": video_id, "segments": [{"start_s": float(s.start_s), "end_s": float(s.end_s)} for s in segments]}
        for video_id, segments in sorted(predictions.items())
    ]
    payload = {"provenance": dict(provenance), "predictions": rows} if provenance else rows
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_prediction_file(path: str | Path) -> dict[str, list[Segment]]:
    # Predicted-segments JSON: dict {vid: [{start_s,end_s}]} or list-of-rows form.
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "predictions" in raw: raw = raw["predictions"]  # stamped form
    if isinstance(raw, dict):
        return {str(vid): [Segment(float(r["start_s"]), float(r["end_s"])) for r in rows] for vid, rows in raw.items()}
    predictions: dict[str, list[Segment]] = {}
    for row in raw:
        if "video_id" in row and "segments" in row:
            predictions[str(row["video_id"])] = [Segment(float(i["start_s"]), float(i["end_s"])) for i in row["segments"]]
        elif "video_id" in row and "start_s" in row and "end_s" in row:
            predictions.setdefault(str(row["video_id"]), []).append(Segment(float(row["start_s"]), float(row["end_s"])))
        else: raise ValueError(f"Unsupported prediction row format: {row}")
    return predictions


def _drop_quarantined_predictions(predicted: dict[str, list[PredictionEvent]], records: list[VideoRecord]) -> dict[str, list[PredictionEvent]]:
    """Ignore-region protocol, prediction side. Gold already excludes quarantined spans (reliable=False); a prediction majority-inside one must 
    not count as unmatched either — the region is unscoreable, not empty. Majority-overlap, not any-overlap: a span merely grazing a quarantine 
    edge is still a real, scoreable event."""
    zones = {r.video_id: [(s.start_s, s.end_s) for s in r.sentences if not getattr(s, "reliable", True)] for r in records}
    out: dict[str, list[PredictionEvent]] = {}
    for vid, events in predicted.items():
        keep = []
        for ev in events:
            dur = max(1e-9, float(ev.end_s) - float(ev.start_s))
            inside = sum(max(0.0, min(ev.end_s, b) - max(ev.start_s, a)) for a, b in zones.get(vid, []))
            if inside / dur <= 0.5: keep.append(ev)
        out[vid] = keep
    return out


def _gold_events(records: list[VideoRecord]) -> dict[str, list[PredictionEvent]]:
    return {record.video_id: [PredictionEvent(
        video_id=record.video_id, start_s=float(span.start_s), end_s=float(span.end_s), text=span.text,
    ) for span in record.sentences if getattr(span, "reliable", True)] for record in records}


def write_gold_segments(records: list[VideoRecord], path: str | Path) -> Path:
    # GT sentence spans in the `--segments` schema (load_prediction_file dict form).

    # Feeds the RQ2 oracle-input (ceiling) rows: `--rq 2 --segments <this>` is RQ1 @ delta=0, framed for RQ2.
    rows = {record.video_id: [
        {"start_s": float(span.start_s), "end_s": float(span.end_s), "text": span.text}
        for span in record.sentences if getattr(span, "reliable", True)
    ] for record in tqdm(records, desc="Extracting gold segments")}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _segment_from_any(value: Any, video_id: str | None = None) -> PredictionEvent:
    if isinstance(value, dict):
        vid = str(value.get("video_id", video_id or ""))
        text = value.get("text", value.get("prediction", value.get("translation")))
        commit = value.get("commit_time_s")
        return PredictionEvent(
            video_id=vid, start_s=float(value["start_s"]), end_s=float(value["end_s"]),
            text=str(text) if text is not None else None,
            flagged_partial=bool(value.get("flagged_partial", False)),
            commit_time_s=float(commit) if commit is not None else None,
        )
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        if video_id is None: raise ValueError("Tuple/list prediction rows require an enclosing video_id")
        return PredictionEvent(video_id=video_id, start_s=float(value[0]), end_s=float(value[1]))
    raise TypeError(f"Unsupported prediction segment row: {value!r}")


def load_event_predictions(path: str | Path) -> dict[str, list[PredictionEvent]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "events" in data: data = data["events"]  # stamped form (see _write_events_json)
    out: dict[str, list[PredictionEvent]] = {}
    if isinstance(data, dict):
        for video_id, segments in data.items():
            out[str(video_id)] = [_segment_from_any(row, video_id=str(video_id)) for row in segments]
        return out

    if not isinstance(data, list): raise TypeError("Prediction file must be a dict or list JSON payload")
    for row in data:
        if isinstance(row, dict) and "segments" in row:
            video_id = str(row["video_id"])
            out[video_id] = [_segment_from_any(seg, video_id=video_id) for seg in row["segments"]]
        else:
            event = _segment_from_any(row)
            out.setdefault(event.video_id, []).append(event)
    return out


def controlled_windows(
    records: list[VideoRecord], grid: list[float], relative: bool = True, max_sentences: int | None = None, 
    tail_grid: list[float] | None = None, drop_counts: dict[tuple[float, float], int] | None = None,
) -> list[ControlledWindow]:
    """RQ1 signed-offset boundaries: start = gt_start + delta_head, end = gt_end + delta_tail. The loaded input
    includes one frame after the requested end so a zero-offset target contains its terminator.

    `relative=True` (default): grid values are FRACTIONS of sentence duration — absolute seconds mix regimes (0.3s
    destroys a 1s sentence, not a 10s one) at one x-point.

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
                        if drop_counts is not None: drop_counts[(float(gh), float(gt))] = drop_counts.get((float(gh), float(gt)), 0) + 1
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


def evaluate_predicted_events(
    predicted: dict[str, list[PredictionEvent]], gold: dict[str, list[PredictionEvent]], thresholds: list[float], char_level: bool = False
) -> dict[str, Any]:
    """RQ2 scoring, SODA-style (Fujita et al. 2020) with translations in place of captions. Per tIoU threshold: predictions and gold are 
    matched 1-to-1 by tIoU (metrics.match_segments); segmentation block reports that localization P/R/F1; TEXT block is a localization-aware 
    F1 — per-pair sentence scores summed, then precision = Σ/n_pred and recall = Σ/n_gold (`compute_text_metrics(localization_aware=True)`), 
    macro-averaged over videos.

    Matched-pairs-only text (a mean over survivors) is not used: it conditions translation quality on localization success, so a higher-tIoU 
    column scores an easier survivor subset and an over-generating cascade is scored only on spans it happens to localize while its spurious 
    spans go uncharged. The SODA fusion charges spurious predictions (n_pred) and missed gold (n_gold) in the SAME number. densevid_eval's 
    garbage-ref alternative is avoided: with corpus BLEU it rewards over-generation via the brevity penalty, and it charges only predictions,
    not misses. `threshold_average` means over the grid. Ours on top: emission latency + best-match tIoU."""
    results = []
    all_video_ids = sorted(set(predicted) | set(gold))
    pair_memo: dict[tuple[str, str], dict[str, float]] = {}  # (hyp, ref) -> sentence scores; reused across thresholds
    for threshold in thresholds:
        per_video: list[dict[str, float]] = []
        vid_prec, vid_rec, latencies, best_tious = [], [], [], []
        n_pairs = n_unmatched_pred = n_missed_gold = total_pred = total_gold = 0
        for video_id in all_video_ids:
            pred_events, gold_events = predicted.get(video_id, []), gold.get(video_id, [])
            if not pred_events and not gold_events: continue
            total_pred += len(pred_events); total_gold += len(gold_events)

            # SEGMENTATION: the codebase's 1 canonical segment metric — greedy 1-to-1 tIoU matching, shared verbatim with segmenter-eval 
            # and the BIO monitor. densevid's own proposal P/R is coverage-based (many-to-many), which lets duplicate spans all "match" 
            # 1 gold and keeps precision at 1.0; 1-to-1 charges the duplicate, so it is the defensible number for a segmentation claim.
            pred_segs, gold_segs = [ev.segment for ev in pred_events], [gt.segment for gt in gold_events]
            prf = segmentation_prf(pred_segs, gold_segs, tiou_threshold=float(threshold))
            if pred_events: vid_prec.append(prf["precision"])
            if gold_events: vid_rec.append(prf["recall"])
            one_to_one = match_segments(pred_segs, gold_segs, threshold=float(threshold))

            # Counts come from SAME 1-to-1 matching as P/R, so they never contradict it: a merged span covering 2 gold sentences matches 
            # 1 and leaves the other missed — under many-to-many bookkeeping it would read "0 missed" against that same recall.
            n_missed_gold += len(gold_events) - len(one_to_one)
            for pi, gi, iou in one_to_one:
                best_tious.append(float(iou))
                if pred_events[pi].commit_time_s is not None: # Emission latency.
                    latencies.append(float(pred_events[pi].commit_time_s) - float(gold_events[gi].end_s))

            # TEXT: per-pair sentence scores over the 1-to-1 matched pairs, memoised across thresholds (BLEURT is a model forward). 
            # The video's F1 normalises by ITS n_pred/n_gold, so an empty-match video scores 0 and is charged — not skipped.
            hyps, refs = [], []
            for pi, gi, _ in one_to_one:
                if pred_events[pi].text is None or gold_events[gi].text is None: continue
                hyps.append(pred_events[pi].text); refs.append(gold_events[gi].text)
            n_pairs += len(hyps)
            n_unmatched_pred += len(pred_events) - len(one_to_one)
            per_video.append(compute_text_metrics(
                hyps, refs, localization_aware=True, n_pred=len(pred_events), n_gold=len(gold_events), memo=pair_memo, char_level=char_level
            ))

        # Macro over every video with predictions OR gold (empty-match videos contribute 0, charging the miss).
        text_metrics = {
            k: float(np.mean([r[k] for r in per_video])) for k in per_video[0]
        } if per_video else compute_text_metrics([], [], localization_aware=True, n_pred=0, n_gold=0, char_level=char_level)
        precision = float(np.mean(vid_prec)) if vid_prec else 0.0
        recall = float(np.mean(vid_rec)) if vid_rec else 0.0
        
        latency_block, tiou_block = {}, {}
        if latencies:
            arr = np.sort(np.asarray(latencies, dtype=np.float64))
            latency_block = {
                "median_latency_s": float(np.median(arr)),
                "p90_latency_s": float(arr[min(len(arr) - 1, int(round(0.9 * (len(arr) - 1))))]),
                "n": len(arr)
            }
        if best_tious:
            arr = np.asarray(best_tious, dtype=np.float64)
            tiou_block = {
                "min": float(np.min(arr)), "p25": float(np.percentile(arr, 25)),
                "median": float(np.median(arr)), "p75": float(np.percentile(arr, 75)),
                "max": float(np.max(arr))
            }
        results.append({
            "tiou_threshold": float(threshold),
            "segmentation": {
                "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            },
            "text_metrics": text_metrics,
            # scored_pairs feed the text score; unmatched_predictions support `precision`, missed_gold `recall`.
            "scored_pairs": n_pairs, "unmatched_predictions": n_unmatched_pred, "missed_gold": n_missed_gold,
            "total_predictions": total_pred, "total_gold": total_gold, 
            "mean_matched_tiou": float(np.mean(best_tious)) if best_tious else 0.0,
            "matched_tiou_distribution": tiou_block, "emission_latency": latency_block,
        })
    keys = results[0]["text_metrics"] if results else {}
    summary = {k: float(np.mean([r["text_metrics"][k] for r in results])) for k in keys}
    summary["segmentation_f1"] = float(np.mean([r["segmentation"]["f1"] for r in results]))
    return {"thresholds": results, "threshold_average": summary}


def _parse_grid(value: str | None, fallback: list[float]) -> list[float]:
    if not value: return [float(x) for x in fallback]
    return [float(x.strip()) for x in value.split(",") if x.strip()]

METHOD_CONFIGS = { # method -> default config; shared with visualize.
    "baseline": "configs/baseline_eval.yaml",
    "ar": "configs/ar.yaml",
    "dlm": "configs/dlm.yaml",
}
def _method_config_path(args: argparse.Namespace) -> str:
    return str(args.method_config) if args.method_config else METHOD_CONFIGS[args.method]


def _attach_duration_prior(model, language: str, data_cfg: dict, inference_cfg: dict | None) -> None:
    """Attach the deployed semi-Markov duration prior (inference.yaml `duration_decode`, per language).

    `build_gate_omega` re-splits tags with `model.duration_prior` before picking the Ω anchor, so None here while
    segmentation duration-decodes makes the GATE off-policy: Ω anchored on raw-argmax merged runs, emitted spans
    duration-split. Every gated RQ2 row routes through here; no-op if the switch is off or the attr is absent."""
    if not hasattr(model, "duration_prior"): return
    dd = duration_decode_params(inference_cfg or {}, language)
    if dd is None:
        model.duration_prior = None
        return
    train_records, _ = load_language_records(data_cfg, language, split="train")
    model.duration_prior = fit_duration_prior(train_records, **dd)


def _build_eval_model(
    method: str, checkpoint: str | None, language: str, data_cfg: dict, method_cfg: dict, 
    device: torch.device, inference_cfg: dict | None = None,
):
    """Uni-Sign pose-only model. LM (`language_model.name`) and checkpoint (`checkpoint.from_pretrained` /
    `checkpoint.dir`) both come from the METHOD config — no separate stage-1 config. Plain scalars so
    analyze/visualize need no fake Namespace. `inference_cfg` gives the gate the deployed duration prior."""
    target_lang = data_cfg["languages"][language].get("target_lang")
    prompt_lang = prompt_lang_for_target(target_lang)
    lm_name = language_model_name(method_cfg)

    if method == "baseline":
        # Released Uni-Sign mT5 pose-only model, or the in-domain baseline_train checkpoint.
        # Same `MisalignedSLTModel(decoder="ar")` as ar, released weights, BIO head unused.
        mt5_name = lm_name if "mt5" in lm_name.lower() else "google/mt5-base"
        tokenizer = T5Tokenizer.from_pretrained(mt5_name, legacy=False)
        front_end = UniSignMT5FrontEnd(mt5_name=mt5_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)
        model = MisalignedSLTModel(
            front_end=front_end, tokenizer=tokenizer, decoder="ar", 
            bio_hidden_dim=int(method_cfg.get("bio_hidden_dim", 384))
        )
        # Released ckpt ONLY — no `checkpoint.dir` fallback, so a concurrent train-slt run can't slip trained
        # weights in. Retained languages use OpenASL or their in-domain clean baseline.
        ckpt = Path(checkpoint or resolve_pretrained(method_cfg, data_cfg, language, default="") or "")
        if not ckpt.exists(): raise FileNotFoundError(
            f"Missing released Uni-Sign checkpoint for baseline (language '{language}'): {ckpt!s}. Set "
            f"languages.{language}.pretrained_slt in configs/data.yaml (or checkpoint.from_pretrained / --checkpoint)."
        )
        rep = load_unisign_pretrained(model, ckpt, strict=True)
        print(f"[unisign] loaded {ckpt.name}: {rep['pose_tensors']} pose + {rep['mt5_tensors']} LM tensors (missing "
              f"{rep['pose_missing'] + rep['mt5_missing']}, unexpected {rep['pose_unexpected'] + rep['mt5_unexpected']})", flush=True)
        model.to(device); model.eval()
        _attach_duration_prior(model, language, data_cfg, inference_cfg)
        return model, tokenizer

    # Trained ar / dlm: same pose encoder + prompt, only the LM differs (clean mT5-vs-mBART ablation).
    if "mbart" in lm_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(lm_name, src_lang=target_lang, tgt_lang=target_lang)
        front_end = UniSignMBartFrontEnd(mbart_name=lm_name, prompt_lang=prompt_lang, target_lang=target_lang, tokenizer=tokenizer)
    else:
        tokenizer = T5Tokenizer.from_pretrained(lm_name, legacy=False)
        front_end = UniSignMT5FrontEnd(mt5_name=lm_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)
    # EVERY head knob from config, exactly as train/slt.py builds it. Passing only bio_hidden_dim left the other
    # four at code defaults, so any config change silently gives eval a different architecture than training —
    # and the strict=False load below would absorb the mismatch as "missing keys" without a word.
    model = MisalignedSLTModel(
        front_end=front_end, tokenizer=tokenizer,
        decoder="ar" if method == "ar" else "dlm",
        bio_hidden_dim=int(method_cfg.get("bio_hidden_dim", 384)),
        bio_depth=int(method_cfg.get("bio_depth", 4)),
        bio_nhead=int(method_cfg.get("bio_nhead", 8)),
        bio_dropout=float(method_cfg.get("bio_dropout", 0.1)),
        bio_conv_stem_layers=int(method_cfg.get("bio_conv_stem_layers", 2)),
        block_size=int(method_cfg.get("block_size", 8)),
    )
    ckpt = Path(checkpoint or checkpoint_dir(method_cfg, default="") or "")
    if not ckpt.exists(): raise FileNotFoundError(f"Missing checkpoint for {method}: {ckpt}. Train the method first or pass --checkpoint.")

    # strict=False is deliberate (an ar checkpoint has no dlm decoder and vice versa), but the report must be READ:
    # a silently half-loaded model evaluates a partly-random network and reports it as a result.
    missing, unexpected = load_model_checkpoint(model, ckpt, strict=False)
    core_missing = [k for k in missing if k.startswith(("bio_head.", "front_end.pose_encoder."))]
    if core_missing: raise RuntimeError(
        f"{ckpt} did not supply {len(core_missing)} core tensor(s) (e.g. {core_missing[:3]}). The eval model's "
        f"shape disagrees with the checkpoint — check the bio_* / block_size keys in the method config."
    )
    if missing or unexpected: print(
        f"[eval] {ckpt.name}: {len(missing)} missing, {len(unexpected)} unexpected tensor(s) "
        f"(expected across decoder arms; core pose/BIO tensors verified present).", flush=True
    )
    model.to(device); model.eval()
    _attach_duration_prior(model, language, data_cfg, inference_cfg)
    return model, tokenizer


def _prep_window(
    poses_np: np.ndarray, timestamps_np: np.ndarray, start_s: float, visual_padding: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Uni-Sign uses raw windows (`visual_padding: none`): the encoder masks padding via attention mask, no boundary halos.
    poses = torch.as_tensor(poses_np, dtype=torch.float32)
    timestamps = torch.as_tensor(np.asarray(timestamps_np, dtype=np.float32) - float(start_s), dtype=torch.float32)
    return poses, timestamps, frame_mask_for(poses.shape[0], visual_padding)


def _generation_kwargs(method: str, inference_cfg: dict, method_cfg: dict, max_tokens: int) -> dict:
    # Main AR rows are greedy; only DLM uses SPD/DCD params.
    if method == "baseline":
        num_beams = int(method_cfg.get("validation", {}).get("num_beams", method_cfg.get("num_beams", 1)))
        return {"max_text_tokens": max_tokens, "num_beams": num_beams}

    trans_cfg = inference_cfg.get("translation", {})
    # PER-KEY merge, not whole-block fallback: method config supplies the trained defaults, inference.yaml
    # overrides per key at deployment. Block-level fallback shadowed dlm.yaml's entire dcd block the moment
    # inference.yaml defined one — setting dcd.temperature in dlm.yaml then silently did nothing at eval.
    dcd_cfg = {**method_cfg.get("dcd", {}), **trans_cfg.get("dcd", {})}
    spd_cfg = method_cfg.get("spd", {})
    return {
        "max_text_tokens": max_tokens, "num_beams": 1,
        "diffusion_steps": int(trans_cfg.get("diffusion_steps", method_cfg.get("diffusion_steps", 64))),
        "tau_dec": float(dcd_cfg.get("tau_dec", 0.9)),  # DCD default; commit_confidence_tau is FSM gate's knob, not a decode threshold
        "spd_top_k": int(spd_cfg.get("top_k", 1)),
        "spd_renormalize": bool(spd_cfg.get("renormalize", True)),
        "spd_revision": bool(spd_cfg.get("revision", True)),
        "temperature": float(dcd_cfg.get("temperature", 0.0)),
        "dcd_window_length": int(dcd_cfg.get("initial_window_length", method_cfg.get("block_size", 8))),
        "dcd_max_window_length": int(dcd_cfg.get("max_window_length", 64)),
        "dcd_window_type": str(dcd_cfg.get("window_type", "sliding")),
        "dcd_decode_algo": str(dcd_cfg.get("decode_algo", "threshold")),
        "dcd_decode_param": dcd_cfg.get("decode_param", 0.9),
        "dcd_sample_top_k": None if dcd_cfg.get("top_k") is None else int(dcd_cfg.get("top_k")),
        "dcd_top_p": None if dcd_cfg.get("top_p") is None else float(dcd_cfg.get("top_p")),
        "dcd_cache_type": str(dcd_cfg.get("cache_type", "none")),
        # Membership gate at RQ1: same Ω the decoder trained with (on-policy span, no GT/χ). BOTH arms gated — 
        # DLM injects Ω in its manual decode, AR via HF cross-attention hooks (front_end.ar_generate).
        "gate_enabled": bool(method_cfg.get("membership_gate", {}).get("enabled", False)),
        # Fallbacks mirror the RQ2 runner: delta from the measured noise floor, Lambda_min from it (delta+1),
        # never 0 — a 0 floor re-admits 1-frame flicker spans in exactly one of the two eval paths.
        "gate_delta": int(method_cfg.get("membership_gate", {}).get("delta",
                          inference_cfg.get("boundary_stability", {}).get("delta_enc_frames", 3))),
        "gate_eps": float(method_cfg.get("membership_gate", {}).get("eps", 1e-4)),
        "gate_min_span_frames": int(method_cfg.get("membership_gate", {}).get("min_span_frames",
                                    inference_cfg.get("span_selection", {}).get("min_span_frames",
                                    int(inference_cfg.get("boundary_stability", {}).get("delta_enc_frames", 3)) + 1))),
    }


@torch.no_grad()
def _translate_windows(
    model, tokenizer, method: str, items: list[tuple[np.ndarray, np.ndarray, float]], device: torch.device,
    inference_cfg: dict, method_cfg: dict, batch_size: int | None = None, stream_start: bool = False,
    commit_frontier_s: list[float] | None = None, use_duration_prior: bool | None = None,
) -> list[tuple[str, float, bool]]:
    """Pre-trimmed pose windows -> [(text, mean_token_confidence, gate_would_skip)]. The loop-decode entry point:
    ONE `generate_from_poses` path for every method.

    Confidence is the real softmax prob of the model's own tokens, so "confidently wrong" is measured. Rows are right-padded and 
    `frame_mask`-masked, so each row's result matches translating it alone. `batch_size` None = 1 batch padded to its longest row; 
    mixed-length corpora should pass --batch-size."""
    if not items: return []
    if batch_size is not None and int(batch_size) < len(items):
        out: list[tuple[str, float, bool]] = []
        for i in range(0, len(items), max(1, int(batch_size))): out.extend(_translate_windows(
            model, tokenizer, method, items[i:i + max(1, int(batch_size))], device, 
            inference_cfg, method_cfg, stream_start=stream_start, use_duration_prior=use_duration_prior,
            commit_frontier_s=commit_frontier_s[i:i + max(1, int(batch_size))] if commit_frontier_s else None,
        ))
        return out
    visual_padding = str(method_cfg.get("visual_padding", "none"))
    prepped = [_prep_window(p, ts, start_s, visual_padding) for (p, ts, start_s) in items]
    max_t = max(int(p.shape[0]) for p, _, _ in prepped)
    # Same repeat-last-frame pad + frame-mask contract as the training collator (data.batch).
    poses = torch.stack([repeat_last_frame(p, max_t - int(p.shape[0])) for p, _, _ in prepped]).to(device)
    timestamps = torch.stack([torch.nn.functional.pad(ts, (0, max_t - int(ts.shape[0]))) for _, ts, _ in prepped]).to(device)
    frame_mask = torch.stack([torch.nn.functional.pad(m, (0, max_t - int(m.shape[0]))) for _, _, m in prepped]).to(device)

    # inference.yaml overrides the method default — the same per-key precedence as the dcd block and the RQ2 runner.
    max_tokens = int(inference_cfg.get("translation", {}).get("max_text_tokens", method_cfg.get("max_text_tokens", 128)))
    gen_kwargs = _generation_kwargs(method, inference_cfg, method_cfg, max_tokens)
    # Whether frame 0 is a genuine sentence ONSET is a property of how the caller cut this window, not a constant:
    # a window that starts BEFORE the sentence opens inside the predecessor, and minting a B there would anchor Ω
    # on the predecessor's tail and floor the sentence being scored. Baseline is ungated, so the flag is inert.
    if gen_kwargs.get("gate_enabled"):
        gen_kwargs["gate_stream_start"] = bool(stream_start)
        gen_kwargs["gate_use_duration_prior"] = (not bool(stream_start)) if use_duration_prior is None else bool(use_duration_prior)
        # commit_frontier_s (per item, window-relative): frames strictly before it are the already-handled predecessor — the FSM's χ. Without 
        # it a window opening MID-SIGNING has an unopenable leading I-run (buffer-start I never opens; no χ mint fires) and Ω anchors on a 
        # shifted fragment or the NEXT sentence. Measured on the offline row: the whole stream-vs-offline gap sat in back-to-back windows.
        if commit_frontier_s is not None: gen_kwargs["commit_mask"] = timestamps < torch.tensor(
            commit_frontier_s, dtype=timestamps.dtype, device=timestamps.device
        ).unsqueeze(1)
    _, tokens, confidence, gate_skip = model.generate_from_poses(poses=poses, frame_mask=frame_mask, timestamps_s=timestamps, **gen_kwargs)
    tok = tokens.detach().cpu()
    conf = confidence.detach().float().cpu()
    texts = [t.strip() for t in tokenizer.batch_decode(tok, skip_special_tokens=True)]

    # Mean prob over REAL tokens: both arms now return produced tokens only (synthetic start slots stripped at the decode boundary), so mask 
    # pads and everything past 1st EOS — batched DLM rows have no per-row trim, and post-EOS filler dilutes the "confidently wrong" signal.
    n = min(tok.shape[1], conf.shape[1])
    tok, conf = tok[:, :n], conf[:, :n]
    valid = tok != int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        cum = (tok == int(tokenizer.eos_token_id)).cumsum(dim=1)
        valid &= (cum == 0) | ((cum == 1) & (tok == int(tokenizer.eos_token_id)))  # keep the EOS slot itself
    confs = ((conf * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)).tolist()
    # gate_skip: the deployed FSM would never decode this window (no span — all-gap or headless fragment). Text is still reported, but RQ1 
    # must split "translation failed" from "policy is to skip", else the gate's refusal scores as garbage BLEU.
    skips = [bool(s) for s in gate_skip.tolist()]
    return list(zip(texts, [float(c) for c in confs], skips))


def run_rq1(args: argparse.Namespace) -> "pd.DataFrame":
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run RQ1 on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    method_cfg = load_yaml(_method_config_path(args), language=args.language)
    # Optional per-run search override. Main comparisons use the greedy default.
    if getattr(args, "num_beams", None): method_cfg.setdefault("validation", {})["num_beams"] = int(args.num_beams)
    if getattr(args, "gate", None): method_cfg.setdefault("membership_gate", {})["enabled"] = (args.gate == "on")
    _rq1_beams = int(method_cfg.get("validation", {}).get("num_beams", method_cfg.get("num_beams", 1)))
    if args.method == "baseline" and _rq1_beams != 1: print(
        f"[rq1] WARNING: baseline decodes with num_beams={_rq1_beams} while the dlm/ar arms are greedy — "
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
        records, head_grid, relative=relative, max_sentences=max_sentences, tail_grid=tail_grid, drop_counts=drop_counts
    )
    records_by_id = {record.video_id: record for record in records}
    device = pick_device(args.device)
    inference_cfg = resolve_inference(load_yaml(args.inference_config), args.language)
    model, tokenizer = _build_eval_model(args.method, args.checkpoint, args.language, data_cfg, method_cfg, device, inference_cfg)
    
    # Length-sorted batches keep padding minimal.
    materialized: list[tuple[ControlledWindow, np.ndarray, np.ndarray]] = []
    for window in windows:
        record = records_by_id[window.video_id]
        poses, timestamps = load_pose_window(record.pose, window.window_start_s, window.window_end_s, normalize=True)
        if poses.shape[0] == 0: continue
        materialized.append((window, poses, timestamps))

    batch_size = max(1, int(rq_cfg.get("batch_size", 16)))
    grouped: dict[tuple[float, float], dict[str, list]] = {}
    rows = []

    # Mint the stream-start onset only when frame 0 is the realized sentence onset. A positive head offset has removed that onset; 
    # a negative offset starts in leading context. Partition before batching so this flag is row-invariant.
    _rq1_stream_start = lambda delta_head_s: abs(float(delta_head_s)) <= 1e-6
    partitions = [
        ([m for m in materialized if  _rq1_stream_start(m[0].delta_head_s)], True),
        ([m for m in materialized if not _rq1_stream_start(m[0].delta_head_s)], False),
    ]
    chunks = []
    for part, is_stream_start in partitions:
        part.sort(key=lambda wp: int(wp[1].shape[0]))
        chunks += [(part[s : s + batch_size], is_stream_start) for s in range(0, len(part), batch_size)]

    for chunk, is_stream_start in tqdm(chunks, desc="Translating windows"):
        results = _translate_windows(
            model=model, tokenizer=tokenizer, method=args.method,
            items=[(poses, timestamps, w.window_start_s) for (w, poses, timestamps) in chunk], device=device,
            inference_cfg=inference_cfg, method_cfg=method_cfg, stream_start=is_stream_start, use_duration_prior=False,
            # chi for a lead-in window: frames before the sentence onset are the predecessor's tail, already handled — the same commit frontier
            # run_offline supplies. Without it a lead-in window on back-to-back pair has an unopenable leading I-run (buffer-start I never opens,
            # no chi mint), so Omega anchors on the merged predecessor+target run and the gated arms are scored on a mis-anchored decode while
            # ungated baseline is untouched. delta_head_s < 0 means the window opens that far before the onset.
            commit_frontier_s=[max(0.0, -float(w.delta_head_s)) for (w, _p, _t) in chunk],
        )
        for (window, _poses, _ts), (prediction, confidence, gate_skip) in zip(chunk, results):
            key = (window.grid_head, window.grid_tail)  # group by grid coordinate (fraction in relative mode)
            grouped.setdefault(key, {
                "predictions": [], "references": [], "confidences": [], "gate_skips": [],
                "head_s": [], "tail_s": [], "req_head_s": [], "req_tail_s": [],
            })
            duration = window.gt_end_s - window.gt_start_s
            grouped[key]["predictions"].append(prediction)
            grouped[key]["references"].append(window.reference)
            grouped[key]["confidences"].append(confidence)
            grouped[key]["gate_skips"].append(bool(gate_skip))
            grouped[key]["head_s"].append(window.delta_head_s)
            grouped[key]["tail_s"].append(window.delta_tail_s)
            grouped[key]["req_head_s"].append(window.grid_head * duration if relative else window.grid_head)
            grouped[key]["req_tail_s"].append(window.grid_tail * duration if relative else window.grid_tail)
            rows.append({**asdict(window), "prediction": prediction, "mean_confidence": confidence, "gate_would_skip": bool(gate_skip)})

    severity = []
    _char_level = char_level_for_target(target_language(data_cfg, args.language))
    for (gh, gt), values in tqdm(sorted(grouped.items()), desc="Computing severity"):
        confs = values["confidences"]
        head_s, tail_s = values["head_s"], values["tail_s"]
        # REQUESTED offsets clamped by the timeline. High -> corpus lacks this grid point's context and the realized means sit inside it.
        clamped = sum(1 for req, real in zip(values["req_head_s"], head_s) if abs(req - real) > 1e-3)
        clamped += sum(1 for req, real in zip(values["req_tail_s"], tail_s) if abs(req - real) > 1e-3)
        dropped = int(drop_counts.get((float(gh), float(gt)), 0))
        severity.append({
            "windows": len(values["predictions"]),
            "grid_head": gh, "grid_tail": gt,  # fraction of sentence duration in relative mode, else seconds
            "delta_head_s_mean": float(sum(head_s) / len(head_s)) if head_s else 0.0,  # realized offset (relative -> varies per sentence)
            "delta_tail_s_mean": float(sum(tail_s) / len(tail_s)) if tail_s else 0.0,
            "clamped_fraction": float(clamped) / max(1, 2 * len(head_s)),
            # High -> this row averages over a longer-sentence subset.
            "dropped_fraction": float(dropped) / max(1, dropped + len(values["predictions"])),
            "mean_translation_confidence": float(sum(confs) / len(confs)) if confs else 0.0,
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
            "rq": "1", "language": args.language, "split": args.split, "method": args.method,
            # Per-axis grids as ACTUALLY swept: with head/tail overrides `grid` alone misdescribes the axes (kept for old readers).
            "severity_mode": mode, "grid": grid, "grid_head": head_grid, "grid_tail": tail_grid,
            # Decode provenance, mirroring RQ2's events stamp: two sweeps under different flags were previously
            # indistinguishable JSONs, so cross-method RQ1 comparisons could silently mix decode budgets.
            "provenance": {
                "num_beams": int(method_cfg.get("validation", {}).get("num_beams", method_cfg.get("num_beams", 1)))
                             if args.method == "baseline" else 1,
                "gate": bool(method_cfg.get("membership_gate", {}).get("enabled", False)),
                "checkpoint": getattr(args, "checkpoint", None),
                "duration_decode": duration_decode_params(load_yaml(args.inference_config), args.language),
                # Held constant across the sweep (see the _translate_windows call above); recorded because a grid
                # swept under a varying gate configuration is not a degradation curve.
                "gate_use_duration_prior": False,
            },
            "windows": len(rows), "severity": severity, "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[rq1] wrote {out}", flush=True)
    return pd.json_normalize(severity, sep=".").T # one row per (grid_head, grid_tail) severity point


def _build_streaming_runner(model, inference_cfg: dict, method_cfg: dict, duration_prior=None):
    from infer.stream import StreamingSLTRunner
    trans = inference_cfg.get("translation", {})
    dcd = {**method_cfg.get("dcd", {}), **trans.get("dcd", {})}  # same per-key merge as _generation_kwargs
    spd = method_cfg.get("spd", {})
    boundary = inference_cfg.get("boundary_stability", {})

    return StreamingSLTRunner(
        model,
        stride_s=float(inference_cfg.get("stride_s", 1.0)),
        buffer_cap_s=float(inference_cfg.get("buffer_cap_s", 18.0)),
        delta_enc_frames=int(boundary.get("delta_enc_frames", 3)),
        hysteresis_strides=int(boundary.get("hysteresis_strides", 3)),
        token_confidence_tau=float(trans.get("commit_confidence_tau", 0.3)),
        # None (missing key) → runner derives Λ_min = δ+1; a 0-fallback would re-admit 1-frame flicker spans
        # (Λ_min is a duration noise floor, NOT the re-emission guard — that is select_target_span's χ filter).
        min_span_frames=inference_cfg.get("span_selection", {}).get("min_span_frames"),
        forced_tail_policy=str(inference_cfg.get("forced_tail_policy", "skip")),
        # Ω from the method config's membership_gate; χ from the runner's commit log.
        gate_enabled=bool(method_cfg.get("membership_gate", {}).get("enabled", False)),
        gate_delta=int(method_cfg.get("membership_gate", {}).get("delta", boundary.get("delta_enc_frames", 3))),
        gate_eps=float(method_cfg.get("membership_gate", {}).get("eps", 1e-4)),
        max_text_tokens=int(trans.get("max_text_tokens", method_cfg.get("max_text_tokens", 128))),
        diffusion_steps=int(trans.get("diffusion_steps", method_cfg.get("diffusion_steps", 64))),
        tau_dec=float(dcd.get("tau_dec", 0.9)),
        spd_top_k=int(spd.get("top_k", 1)),
        spd_renormalize=bool(spd.get("renormalize", True)),
        spd_revision=bool(spd.get("revision", True)),
        temperature=float(dcd.get("temperature", 0.0)),
        # block_size is a MODEL property -> METHOD config (same source as _generation_kwargs). inference.yaml.translation 
        # has no block_size; a trans fallback would force 8 when initial_window_length is unset.
        dcd_window_length=int(dcd.get("initial_window_length", method_cfg.get("block_size", 8))),
        dcd_max_window_length=int(dcd.get("max_window_length", 64)),
        dcd_window_type=str(dcd.get("window_type", "sliding")),
        dcd_decode_algo=str(dcd.get("decode_algo", "threshold")),
        dcd_decode_param=dcd.get("decode_param", 0.9),
        dcd_cache_type=str(dcd.get("cache_type", "none")),
        # Same two sampling knobs the RQ1 path reads (_generation_kwargs); omitting them here left the FSM at
        # None while single-window RQ1 honoured the config — the two rows would decode under different policies.
        dcd_sample_top_k=None if dcd.get("top_k") is None else int(dcd.get("top_k")),
        dcd_top_p=None if dcd.get("top_p") is None else float(dcd.get("top_p")),
        decode_conditioning=str(trans.get("decode_conditioning", "window")),
        duration_prior=duration_prior,
    )


@torch.no_grad()
def run_streaming(args: argparse.Namespace) -> dict[str, list[PredictionEvent]]:
    """Drive the sawtooth FSM end-to-end over each video → committed events.

    This is the *usable inference engine* for RQ2: Our own BIO head + commit gate, recompute-each-stride, no cross-stride decoder state. 
    FSM methods (dlm/ar) only; the clean baseline's RQ2 is the segment-then-translate cascade (--segments).
    """
    if args.method == "baseline": raise SystemExit("Streaming RQ2 uses the FSM (dlm/ar). For cascaded baselines, pass --segments.")
    data_cfg = load_yaml(args.data_config)
    method_cfg = load_yaml(_method_config_path(args), language=args.language)
    if getattr(args, "num_beams", None): method_cfg.setdefault("validation", {})["num_beams"] = int(args.num_beams)
    if getattr(args, "gate", None): method_cfg.setdefault("membership_gate", {})["enabled"] = (args.gate == "on")
    inference_cfg = resolve_inference(load_yaml(args.inference_config), args.language)
    device = pick_device(args.device)
    model, tokenizer = _build_eval_model(args.method, args.checkpoint, args.language, data_cfg, method_cfg, device, inference_cfg)
    # Opt-in buffer-level semi-Markov duration decode (inference.yaml duration_decode: true, or per-language tuned
    # {split_bias, snap_radius_s} from `analyze --stage tune-decode`).
    duration_prior = None
    dd = duration_decode_params(inference_cfg, args.language)
    if dd is not None:
        train_records, _ = load_language_records(data_cfg, args.language, split="train")
        duration_prior = fit_duration_prior(train_records, **dd)
        if duration_prior is not None: print(f"[streaming] buffer duration decode ON (prior from {len(train_records)} train videos"
                                            + (f"; tuned {dd}" if dd else "") + ")", flush=True)
        else: print(f"[streaming] duration_decode requested but <10 usable train captions — decoding plain argmax", flush=True)
    runner = _build_streaming_runner(model, inference_cfg, method_cfg, duration_prior=duration_prior)
    if getattr(args, "stability", False): runner.trace = []
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    stability_tracks: list = []

    predicted: dict[str, list[PredictionEvent]] = {}
    fsm_bio_rows: list[dict[str, float]] = []
    for record in tqdm(records, desc="Processing records"):
        poses, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=True)
        if poses.shape[0] == 0:
            predicted[record.video_id] = []
            continue

        if runner.trace is not None: runner.trace.clear()
        events = runner.run(torch.as_tensor(poses, dtype=torch.float32), fps=float(record.pose.fps))
        if runner.trace: stability_tracks.extend(group_tracks(runner.trace, delta_s=runner.gate_delta / max(1.0, float(record.pose.fps))))
        predicted[record.video_id] = [PredictionEvent(
            video_id=record.video_id, start_s=float(ev.start_s), end_s=float(ev.end_s),
            text=tokenizer.decode(ev.token_ids.tolist(), skip_special_tokens=True).strip(),
            flagged_partial=bool(ev.flagged_partial), commit_time_s=float(ev.commit_time_s),
        ) for ev in events]

        # Moryossef-style BIO P/R/F1 on the runner's stitched per-frame argmax vs GT captions: 
        # the deployed head as the FSM saw it (latest estimate per frame).
        if runner._bio_timeline is not None:
            gold = make_bio_labels(timestamps, record.sentences, 0.0, float(record.pose.duration_s), video_duration_s=record.pose.duration_s)
            tags = runner._bio_timeline.clone()
            # Ignore-region: where gold is UNK (quarantine / untrusted gap) the pred is neither right nor wrong —
            # mask it too, or every phantom "segment" there deflates precision for a region that has no GT.
            gold_t = torch.as_tensor(np.asarray(gold)).long()
            tags[gold_t == BIO["UNK"]] = BIO["UNK"]
            logits_1hot = torch.nn.functional.one_hot(tags.clamp(min=0), num_classes=4).float().unsqueeze(0) * 10.0
            fsm_bio_rows.append(moryossef_segment_metrics(logits_1hot, torch.as_tensor(np.asarray(gold)).long().unsqueeze(0), prefix="fsm_bio"))

    # Why-did-it-(not)-commit: low streaming recall with near-perfect frame BIO = the gate suppressed emission;
    # boundary_ok / translation_ok (of spans_seen, complete spans) say which signal blocks.
    s = runner.gate_stats
    seen = s.get("spans_seen", 0)
    if seen: print(
        f"[stream] gate: spans_seen={seen} boundary_ok={s.get('boundary_ok',0)} translation_ok={s.get('translation_ok',0)} "
        f"committed={s.get('committed',0)} forced={s.get('forced_commit',0)} | translation_ok rate={s.get('translation_ok',0)/seen:.2f} "
        f"(if this is low, the commit gate's token-confidence floor is suppressing a weak decoder, not an eval bug)", flush=True
    )
    if stability_tracks:
        # Stable-prefix comparison on the SAME decodes the FSM already produced: how much earlier could text appear,
        # and what does freezing it early cost? commit_only is what ships today (latency ceiling, zero error).
        rows = {name: score_policy(stability_tracks, pol) for name, pol in build_policies().items()}
        print(f"[stability] {len(stability_tracks)} tracks | tau grid {TAU_GRID}", flush=True)
        # ANCHORS DIFFER, deliberately. RQ2 `emission_latency` = commit_time - GOLD SENTENCE END: when the SCORED translation is complete, 
        # under commit_only, the only policy the FSM actually runs. `latency_s` here is first_reveal - COMMIT: how much earlier a policy would put 
        # text on screen. Latency to first text = emission_latency + latency_s (negative). Keep them separate — quoting a prefix's latency against 
        # the full sentence's BLEU would describe 2 different objects.
        print("latency_s is relative to the COMMIT (negative = earlier); add RQ2 emission_latency for the gold-anchored number.", flush=True)
        print(f"{'policy':<30}{'latency_s':>11}{'revealed':>10}{'frozen_err':>12}{'rewrite_rate':>14}", flush=True)
        for name, r in rows.items(): print(
            f"{name:<30}{r['first_token_latency_s']:>11.3f}{r['revealed_fraction']:>10.3f}"
            f"{r['frozen_prefix_error']:>12.3f}{r['contradicted_track_rate']:>14.3f}", flush=True
        )
        Path(f"outputs/stability_{args.method}_{args.language}_{args.split}.json").write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if fsm_bio_rows:
        fsm_bio = {k: float(sum(r[k] for r in fsm_bio_rows) / len(fsm_bio_rows)) for k in fsm_bio_rows[0]}
        print("[stream] FSM BIO (stitched per-frame argmax vs GT): " + " ".join(f"{k}={v:.3f}" for k, v in sorted(fsm_bio.items())), flush=True)
        run_streaming.last_fsm_bio = fsm_bio  # run_rq2 picks this up for the output payload
    else: run_streaming.last_fsm_bio = None
    return predicted


@torch.no_grad()
def run_cascade(args: argparse.Namespace) -> dict[str, list[PredictionEvent]]:
    """Segment-then-translate cascade: predicted spans (analyze.py --stage segmenter-infer JSON, via --segments)
    are cut from the pose stream and translated offline. Scored by the same tIoU/translation harness as the streaming FSM.

    The prior-art FLOOR is an independent (Moryossef) segmenter's spans translated by the CLEAN baseline, so pass `--method baseline`.
    Translation uses `args.method` (NOT pinned), so `--method dlm` here is a DIFFERENT row (Moryossef spans + our DLM, offline) — legitimate,
    but not the floor. The method is encoded in the output filename so the two never collide; pass the --method for the row you want.
    """
    data_cfg = load_yaml(args.data_config)
    method_cfg = load_yaml(_method_config_path(args), language=args.language)
    if getattr(args, "num_beams", None): method_cfg.setdefault("validation", {})["num_beams"] = int(args.num_beams)
    if getattr(args, "gate", None): method_cfg.setdefault("membership_gate", {})["enabled"] = (args.gate == "on")
    inference_cfg = resolve_inference(load_yaml(args.inference_config), args.language)
    device = pick_device(args.device)
    model, tokenizer = _build_eval_model(args.method, args.checkpoint, args.language, data_cfg, method_cfg, device, inference_cfg)

    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    records_by_id = {record.video_id: record for record in records}
    segments = load_prediction_file(args.segments)

    # Split comes from --split, NOT the JSON filename; mismatched video_ids translate nothing and score all-zero,
    # so fail loud ("gold_*_test.json but forgot --split test").
    matched = set(segments) & set(records_by_id)
    if not matched: raise SystemExit(
        f"--segments has {len(segments)} video_ids, none in the {len(records_by_id)} '{args.split}' records — "
        f"nothing to translate (output would be all-zero). Pass --split test (+ --allow-test) for test gold spans."
    )
    if len(matched) < len(segments): print(
        f"[cascade] WARNING: {len(segments) - len(matched)}/{len(segments)} segment video_ids "
        f"are absent from --split {args.split}; scoring only the {len(matched)} that match.", flush=True)

    # Same single-sentence-unit rule as run_rq1: a cascade window IS the span to translate — the upstream segmenter already did the splitting,
    # so re-splitting inside it would anchor Ω on a fragment of the very span being scored. Gate-off methods (baseline) never read the prior.
    model.duration_prior = None
    predicted: dict[str, list[PredictionEvent]] = {}
    for video_id, spans in tqdm(segments.items(), desc="Processing segments"):
        record = records_by_id.get(video_id)
        if record is None: continue
        items, kept = [], []
        for span in spans:
            poses, timestamps = load_pose_window(record.pose, span.start_s, span.end_s, normalize=True)
            if poses.shape[0] == 0: continue
            items.append((poses, timestamps, float(span.start_s))); kept.append(span)
        results = _translate_windows( # Cascade windows are cut exactly at the predicted span start, so frame 0 is that span's onset.
            model, tokenizer, args.method, items, device, inference_cfg, method_cfg,
            batch_size=int(args.batch_size), stream_start=True,
        )
        predicted[video_id] = [
            PredictionEvent(video_id=video_id, start_s=float(s.start_s), end_s=float(s.end_s), text=text)
            for s, (text, _, _) in zip(kept, results)
        ]
    return predicted


@torch.no_grad()
def run_offline(args: argparse.Namespace) -> dict[str, list[PredictionEvent]]:
    """FINAL joint fine-tuned model OFFLINE: its OWN BIO head segments each video, the SAME model translates each
    span under the FSM's DECODE CONDITIONING — the buffer-shaped window (δ-frame lead + trailing context to buffer_cap) 
    with Ω anchoring the span — never a tight span crop. No FSM, no commit/hysteresis, no cross-stride refinement.

    Held against `--stream` at the SAME model to measure what streaming buys; it folds in the cost of causal (vs
    offline-bidirectional) segmentation, so it is a conservative baseline, not a pure-refinement control.
    """
    if args.method == "baseline": raise SystemExit(
        "Offline RQ2 uses the trained model's own BIO head (ar/dlm); the baseline has no "
        "trained head. For the external-segmenter cascade rows, pass --segments."
    )
    from moryossef26.infer import bio_tags_to_segments
    data_cfg = load_yaml(args.data_config)
    method_cfg = load_yaml(_method_config_path(args), language=args.language)
    if getattr(args, "num_beams", None): method_cfg.setdefault("validation", {})["num_beams"] = int(args.num_beams)
    if getattr(args, "gate", None): method_cfg.setdefault("membership_gate", {})["enabled"] = (args.gate == "on")
    inference_cfg = resolve_inference(load_yaml(args.inference_config), args.language)
    device = pick_device(args.device)

    model, tokenizer = _build_eval_model(args.method, args.checkpoint, args.language, data_cfg, method_cfg, device, inference_cfg)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    buffer_cap_s = float(inference_cfg.get("buffer_cap_s", 18.0))
    # SEGMENTATION context comes from the cap the head TRAINED under (checkpoint meta), same rule _load_segmenter applies to S1: `tail-benefit 
    # --write-config` rewrites buffer_cap_s after training, and re-chunking a trained head over context it never saw degrades it (measured on 
    # S1). The live cap still shapes TRANSLATION windows below — that is deployment geometry, not the head's trained attention span.
    _ckpt = args.checkpoint or checkpoint_dir(method_cfg, default="")
    _trained_cap = (load_checkpoint_meta(_ckpt) or {}).get("buffer_cap_s") if _ckpt else None
    seg_chunk_cap_s = float(_trained_cap or buffer_cap_s)
    if _trained_cap and abs(float(_trained_cap) - buffer_cap_s) > 1e-6: print(
        f"[offline] BIO chunking at the TRAINED cap {float(_trained_cap):.2f}s (live inference.yaml says "
        f"{buffer_cap_s:.2f}s); translation windows keep the live cap.", flush=True
    )
    min_span_frames = int((inference_cfg.get("span_selection", {}) or {}).get(
        "min_span_frames", int((inference_cfg.get("boundary_stability", {}) or {}).get("delta_enc_frames", 3)) + 1))
    # Same deployed decode as every other inference.yaml consumer — else offline argmaxes while streaming duration-decodes and the
    # rows compare decodes, not deployments. Whole-video input has ended, so nothing is right-censored (unlike FSM's "survival" buffers).
    dd = duration_decode_params(inference_cfg, args.language)
    duration_prior = None
    if dd is not None:
        train_records, _ = load_language_records(data_cfg, args.language, split="train")
        duration_prior = fit_duration_prior(train_records, **dd)
        print(f"[offline] whole-video duration decode ON (prior from {len(train_records)} train videos)", flush=True)

    predicted: dict[str, list[PredictionEvent]] = {}
    n_sub_lambda = 0
    for record in tqdm(records, desc="Offline (self-segment + translate)"):
        poses, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=True)
        if poses.shape[0] == 0:
            predicted[record.video_id] = []
            continue
        # Model's OWN BIO head, chunked at the trained buffer scale (Moryossef's train/infer lesson).
        poses_t = torch.as_tensor(poses, dtype=torch.float32, device=device).unsqueeze(0)
        ts_t = torch.as_tensor(timestamps, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.ones(poses_t.shape[:2], dtype=torch.bool, device=device)
        
        bio_tap, bio_mask, ts_out = model.front_end.extract_bio_tap(poses_t, mask_t, ts_t)
        model.bio_head.chunk_size = max(1, int(round(seg_chunk_cap_s * float(record.pose.fps))))
        model.bio_head.chunk_overlap = True
        bio_logits = model.bio_head(bio_tap, timestamps_s=ts_out, frame_mask=bio_mask).logits
        
        if duration_prior is not None: tags = duration_decode_tags(bio_logits, float(record.pose.fps), duration_prior).cpu()
        else: tags = bio_logits.argmax(dim=-1)[0].cpu()
        segments = bio_tags_to_segments(tags, timestamps.tolist())
        # Deployed span floor, offline too: Λ_min is POLICY (span_selection — a sub-floor span is unresolvable from
        # boundary evidence and the FSM never emits one), not streaming machinery. Without it this row emits flicker
        # spans the deployed system cannot produce, each charged by SODA as a spurious prediction plus a junk
        # translation — so (8−7) would partly measure the missing floor, not streaming.
        kept = [s for s in segments if (float(s.end_s) - float(s.start_s)) * float(record.pose.fps) >= float(min_span_frames)]
        n_sub_lambda += len(segments) - len(kept)
        segments = kept

        items, bounds, frontiers = [], [], []
        delta_lead_s = float(inference_cfg.get("boundary_stability", {}).get("delta_enc_frames", 3)) / float(record.pose.fps)
        for span in segments:
            # Decode the span inside a buffer-shaped window (δ lead + context to buffer_cap), gated. A tight crop is stream.py's 'span' mode: 
            # untrained, and it force-disables Ω because select_target_span would sub-anchor inside the sentence and mask part of it. 
            # Cap-length trailing context is DELIBERATE and measured: cutting it to the FSM's commit-time tail lowered this row.
            w_start = max(0.0, float(span.start_s) - delta_lead_s)
            w_end = min(float(record.pose.duration_s), w_start + buffer_cap_s)
            span_poses, span_ts = load_pose_window(record.pose, w_start, w_end, normalize=True)
            if span_poses.shape[0] == 0: continue
            items.append((span_poses, span_ts, w_start))
            bounds.append((float(span.start_s), min(float(span.end_s), w_end)))
            # χ for this window: the δ lead is the PREDECESSOR's tail — already handled, exactly the FSM's post-commit leftover. Without it, 
            # a back-to-back lead is mid-signing, the merged I-run cannot open a span, and Ω anchors on the NEXT sentence.
            frontiers.append(max(0.0, float(span.start_s) - w_start))
        results = _translate_windows( # Offline windows are extended by delta_lead_s before the span, so frame 0 precedes the sentence.
            model, tokenizer, args.method, items, device, inference_cfg, method_cfg,
            batch_size=int(args.batch_size), stream_start=False, commit_frontier_s=frontiers,
        )
        predicted[record.video_id] = [
            PredictionEvent(video_id=record.video_id, start_s=s0, end_s=s1, text=text)
            for (s0, s1), (text, _, _) in zip(bounds, results)
        ]
    if n_sub_lambda: print(f"[offline] dropped {n_sub_lambda} sub-Λ_min span(s) (< {min_span_frames} frames) — "
                           f"parity with the FSM's span selection, which can never commit them.", flush=True)
    return predicted


def _write_events_json(predicted: dict[str, list[PredictionEvent]], path: str | Path, provenance: dict | None = None,) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"events": {vid: [asdict(ev) for ev in evs] for vid, evs in predicted.items()}}
    if provenance: payload["provenance"] = dict(provenance)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _format_threshold_table(t: "pd.DataFrame") -> "pd.DataFrame":
    t = t.copy()
    avg = t.mean(axis=1)
    out = {}
    for idx, row in t.iterrows():
        vals = [float(v) for v in row]
        if all(v.is_integer() for v in vals): out[idx] = [f"{int(v):d}" for v in vals] + [f"{avg[idx]:.1f}"]
        else: out[idx] = [f"{v:.4f}" for v in vals] + [f"{avg[idx]:.4f}"]
    return pd.DataFrame.from_dict(out, orient="index", columns=[*t.columns, "avg"])


def run_rq2(args: argparse.Namespace) -> "pd.DataFrame":
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run RQ2 on test without --allow-test")
    if not args.predictions and not args.stream and not args.segments and not args.offline: raise SystemExit(
        "--rq 2 needs --stream (streaming FSM), --offline (final model self-segments offline), "
        "--segments (given-spans cascaded baselines), or --predictions (score an events JSON)"
    )
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    method_cfg = load_yaml(_method_config_path(args), language=args.language)
    if getattr(args, "num_beams", None): method_cfg.setdefault("validation", {})["num_beams"] = int(args.num_beams)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    thresholds = _parse_grid(args.tiou_thresholds, eval_cfg.get("rq2", {}).get("tiou_thresholds", [0.3, 0.5, 0.7, 0.9]))
    provenance = {
        "method": args.method, 
        "num_beams": int(method_cfg.get("validation", {}).get("num_beams", method_cfg.get("num_beams", 1))) if args.method == "baseline" else 1,
        "language": args.language, "split": args.split, "checkpoint": getattr(args, "checkpoint", None),
        "segments": getattr(args, "segments", None), "gate": bool(method_cfg.get("membership_gate", {}).get("enabled", False)),
    }
    # A deliberate beam override changes the search budget and cannot enter a main row delta.
    if args.method == "baseline" and int(provenance["num_beams"]) != 1: print(
        f"[rq2] WARNING: baseline decodes with num_beams={provenance['num_beams']} while the ar/dlm arms are greedy — "
        f"this row is not search-budget matched.", flush=True
    )
    if args.stream:
        predicted = run_streaming(args)
        _write_events_json(predicted, f"outputs/rq2_stream_events_{args.method}_{args.language}_{args.split}.json", provenance)
        fsm_bio = getattr(run_streaming, "last_fsm_bio", None)
        if fsm_bio:  # FSM-internal BIO metric, persisted alongside the events
            path = Path(f"outputs/rq2_fsm_bio_{args.method}_{args.language}_{args.split}.json")
            path.write_text(json.dumps(fsm_bio, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.offline:
        predicted = run_offline(args)
        _write_events_json(predicted, f"outputs/rq2_offline_events_{args.method}_{args.language}_{args.split}.json", provenance)
    elif args.segments:
        predicted = run_cascade(args)
        # Name by span source (--segments stem carries arch+lang+split) AND method, so cascade rows never collide.
        src = Path(args.segments).stem
        _write_events_json(predicted, f"outputs/rq2_cascade_{src}_{args.method}.json", provenance)
    else:
        predicted = load_event_predictions(args.predictions)
        gold_ids = {r.video_id for r in records}
        if predicted and not (set(predicted) & gold_ids): raise SystemExit(
            f"--predictions has {len(predicted)} video_ids, none in the {len(gold_ids)} '{args.split}' records — "
            f"wrong split or wrong events file (score would be silently near-zero)."
        )
        # Partial mismatch (same handling as run_cascade): a predicted video absent from this split has no gold, so it would be scored as pure 
        # phantoms — charging per-video precision 0 and a 0-F1 text row for events that belong to ANOTHER split, not being flagged as wrong file.
        foreign = set(predicted) - gold_ids
        if foreign:
            print(f"[rq2] WARNING: {len(foreign)}/{len(predicted)} --predictions video_ids are absent from "
                  f"--split {args.split}; scoring only the {len(predicted) - len(foreign)} that match.", flush=True)
            predicted = {vid: evs for vid, evs in predicted.items() if vid in gold_ids}
        
    predicted = _drop_quarantined_predictions(predicted, records)
    gold = _gold_events(records)
    summary = evaluate_predicted_events(
        predicted, gold, thresholds, char_level=char_level_for_target(target_language(data_cfg, args.language))
    ).get("thresholds", [])
    summary = pd.json_normalize(summary, sep=".")  # one row per tIoU threshold
    summary.set_index("tiou_threshold", inplace=True)
    return _format_threshold_table(summary.T)


def _load_segmenter(args):
    """Trained segmenter by --segmenter-arch (shared by eval --segmenter-eval and analyze --stage segmenter-infer).

    moryossef (default): faithful Moryossef segmenter (UNet) — a DIFFERENT input space from FSM head. It supplies calibration and RQ2 cascade 
    spans. s1 is the in-system BIO head, isolating system design from segmentation competence. rope_chunk_s is SECONDS (s1) or None (moryossef).
    """
    device = pick_device(args.device)
    if args.segmenter_arch == "s1":
        from train.bio_pretrain import build_bio_s1_model
        cfg = load_yaml(args.bio_config, language=args.language)
        pretrained = resolve_pretrained(cfg, load_yaml(args.data_config), args.language, default="checkpoints/openasl_pose_only_slt.pth")
        model = build_bio_s1_model(cfg, pretrained_path=pretrained)
        # A pooled (multilingual) S1 is ONE language-agnostic checkpoint; per-language analysis on any --language
        # reads the same pool directory (that IS the zero-shot protocol). checkpoint_dir does the substitution.
        ckpt_default = f"checkpoints/bio_s1/{args.language}"
        # Chunked RoPE at the head's TRAINED context, in SECONDS: training windows clamp to buffer_cap_s (sampler.py), so eval chunks there 
        # too (wrapper converts to frames per stream fps). Larger chunks would attend over untrained context.
        buffer_cap_s = float(resolve_inference(load_yaml(args.inference_config), args.language, strict=False).get("buffer_cap_s", 30.06))
        velocity, rope_chunk_s = False, float(cfg.get("rope_eval_chunk_s") or buffer_cap_s)
    else:
        from moryossef26.trainer import build_segmenter
        cfg = load_yaml(args.moryossef_config, language=args.language)
        model = build_segmenter(args.moryossef_config)
        ckpt_default = f"checkpoints/moryossef/{args.language}"
        velocity, rope_chunk_s = bool(cfg.get("velocity", True)), None  # UNet chunks internally at num_frames

    # checkpoint.dir expanded ${language} from the config's OWN `language:` key, so a differing CLI --language
    # would load another corpus's checkpoint.
    ckpt_dir = checkpoint_dir(cfg, default=ckpt_default)
    # A differing CLI --language would otherwise read another corpus's checkpoint (the config expanded ${language}
    # from its OWN key). Pooled runs are exempt: the pool IS language-agnostic, so every --language reads it.
    if (args.language and not pool_key(cfg) and str(cfg.get("language", args.language)) != str(args.language)): ckpt_dir = ckpt_default
    checkpoint = args.checkpoint or str(Path(ckpt_dir) / "model.pt")
    load_model_checkpoint(model, checkpoint, strict=True)
    # S1's RoPE chunk is the buffer cap the head TRAINED under, which the checkpoint records. It wins over both the
    # config pin and the live buffer_cap_s, because `analyze --stage buffer-cap --write-config` rewrites that cap
    # after training and following it re-chunks a trained head over context it never saw.
    # PROVENANCE: the config says which pool this run expects; the checkpoint records which pool produced it. A
    # mismatch means the wrong segmenter is about to be evaluated — silently, with a plausible-looking score — so
    # it fails loud. Checkpoints written before `pretrain_pool` existed carry no key and are exempt.
    _meta = load_checkpoint_meta(checkpoint)
    if "pretrain_pool" in _meta and _meta.get("pretrain_pool") != pool_key(cfg): raise SystemExit(
        f"{checkpoint} was trained on pool {_meta.get('pretrain_pool')!r}, but this config expects {pool_key(cfg)!r}. Point --checkpoint "
        f"at the matching model, or align `pretrain_languages` (a pooled checkpoint is a DIFFERENT model from a monolingual one)."
    )
    if args.segmenter_arch == "s1":
        trained_chunk = _meta.get("rope_eval_chunk_s")
        if trained_chunk:
            if abs(float(trained_chunk) - float(rope_chunk_s)) > 1e-6: print(
                f"segmenter | rope_eval_chunk_s {float(trained_chunk):.2f}s from the checkpoint (config/buffer_cap_s "
                f"says {float(rope_chunk_s):.2f}s); using the trained value.", flush=True)
            rope_chunk_s = float(trained_chunk)
    return model, device, velocity, rope_chunk_s, checkpoint


def run_segmenter_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Whole-video segmentation eval (Moryossef evaluate.py protocol): frame-F1 + 1-to-1 tIoU segment P/R/F1. The acceptance check ("F1 
    within order-of-magnitude of published") and — same protocol for either --segmenter-arch — the only apples-to-apples Moryossef-vs-S1 
    comparison (training monitors use different window regimes). For `s1` it also scores the pretrained head before stage-2 fine-tuning.
    """
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run segmenter eval on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    model, device, velocity, rope_chunk_s, checkpoint = _load_segmenter(args)
    
    # Per-arch default, same rule as analyze.py segmenter-infer: moryossef stays on its published argmax decode unless
    # --segmenter-decode duration is explicit — the config switch alone must not silently re-split the external baseline.
    dd = duration_decode_params(load_yaml(args.inference_config), args.language, arch=args.segmenter_arch)
    decode = args.segmenter_decode or ("duration" if (args.segmenter_arch == "s1" and dd is not None) else "plain")
    duration_prior = None
    if decode == "duration":
        train_records, _ = load_language_records(data_cfg, args.language, split="train")
        duration_prior = fit_duration_prior(train_records, **(dd or {}))
        if duration_prior is None: print("[segmenter-eval] WARNING: too few train captions to fit duration prior; plain decode", flush=True)
        elif dd is None: print(f"[segmenter-eval] NOTE: --segmenter-decode duration on a language with duration OFF/untuned in "
                               f"inference.yaml; using module-default params", flush=True)
    print(f"[segmenter-eval] {args.segmenter_arch} segmenter from {checkpoint} (decode={'duration' if duration_prior else 'plain'})", flush=True)
    decode = "duration" if duration_prior else "plain"

    # RQ2's tIoU grid so THRESHOLDS line up, but cells are NOT comparable — both are per-video MACRO, and they
    # still differ 2 ways: this stage averages per-video F1s while RQ2 builds F1 from macro-averaged P/R, and
    # gold here is the UNK-masked BIO stream vs RQ2's caption spans (gap predictions = false positives).
    thresholds = tuple(float(t) for t in (load_yaml(args.eval_config).get("rq2", {}) or {}).get("tiou_thresholds", [0.5]))
    metrics = evaluate_segmenter_whole_video(
        model, records, device=device, velocity=velocity, rope_chunk_s=rope_chunk_s, tiou_thresholds=thresholds, duration_prior=duration_prior,
    )
    payload = {
        "language": args.language, "split": args.split, "videos": len(records), "segmenter_arch": args.segmenter_arch, "checkpoint": checkpoint,
        "decode": decode, "tiou_thresholds": list(thresholds), "metrics": metrics,
        # Pinned pair is dev-SELECTED, so dev numbers with it are in-selection — quote held-out or test.
        "decode_hparams": ((dd or "module-defaults") if decode == "duration" else None),
    }
    output = Path(args.output or f"outputs/segmenter_eval_{args.segmenter_arch}_{args.language}_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["output"] = str(output)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Misaligned-SLT evaluation entry points")
    parser.add_argument("--rq", choices=["1", "2"], default=None)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--method-config", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--method", default="dlm", choices=["baseline", "ar", "dlm"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--severity-grid-s", default=None,
                        help="Comma-separated signed RQ1 grid (fractions of duration in relative mode, else seconds)")
    # Fall back to --severity-grid-s / eval.yaml. One shared grid's cross-product cannot express a truncation-only sweep.
    parser.add_argument("--severity-grid-head", default=None,
                        help="RQ1 head-axis grid override (use --severity-grid-head=-0.3,0 for negative-leading values)")
    parser.add_argument("--severity-grid-tail", default=None,
                        help="RQ1 tail-axis grid override (use --severity-grid-tail=-0.3,0 for negative-leading values)")
    parser.add_argument("--severity-mode", default=None, choices=["relative", "absolute"],  # None -> eval.yaml rq1.severity_mode
                        help="RQ1 perturbation: relative (fraction of sentence duration, default) or absolute seconds")
    parser.add_argument("--emit-gold-segments", default=None, 
                        help="Write GT spans JSON (for --rq 2 --segments oracle-input rows) and exit")
    parser.add_argument("--segmenter-eval", action="store_true",
                        help="Standalone whole-video segmentation eval (Moryossef protocol) for --segmenter-arch, then exit")
    parser.add_argument("--num-beams", type=int, default=None, help="AR search override for this run; main comparisons default to greedy")
    parser.add_argument("--gate", default=None, choices=["on", "off"],
                        help="membership-gate override for THIS run (default: method config). RQ1 measures translation under controlled boundary "
                             "severity, so gate-on vs gate-off is the ablation separating translation quality from gate's conditioning effect")
    parser.add_argument("--segmenter-decode", default=None, choices=["duration", "plain"],
                        help="whole-video decode; default: s1 -> duration (semi-Markov re-split), moryossef -> plain (Moryossef argmax)")
    parser.add_argument("--segmenter-arch", default="moryossef", choices=["moryossef", "s1"],
                        help="segmenter-eval backend: moryossef = external Moryossef segmenter, s1 = in-system head")
    parser.add_argument("--moryossef-config", default="configs/moryossef26.yaml", help="Moryossef segmenter config")
    parser.add_argument("--bio-config", default="configs/bio_pretrain.yaml", help="S1 (in-system head) config for --segmenter-arch s1")
    parser.add_argument("--num-sentences", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8, help="windows per translate/forward batch in loop-decode paths")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--predictions", default=None,
                        help="RQ2: re-score a saved events JSON (outputs/rq2_*_events_*.json) without re-running the model")
    parser.add_argument("--segments", default=None, help="Segmenter spans JSON (analyze --stage segmenter-infer)")
    parser.add_argument("--offline", action="store_true",
                        help="Final misaligned model self-segments each whole video offline (its own BIO head) and translates each span")
    parser.add_argument("--stream", action="store_true", help="Run the streaming FSM engine to produce events")
    parser.add_argument("--stability", action="store_true",
                        help="With --stream: also score stable-prefix display policies (LA-n vs confidence) from the per-stride candidate "
                             "decodes the FSM already computes. No extra decoding.")
    parser.add_argument("--tiou-thresholds", default=None, help="Comma-separated RQ2 tIoU thresholds")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-test", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    data_cfg = load_yaml(args.data_config)
    if args.language is None: args.language = str(data_cfg.get("active_languages", ["asf"])[0])
    if args.emit_gold_segments:
        records, _ = load_language_records(data_cfg, args.language, split=args.split)
        path = write_gold_segments(records, args.emit_gold_segments)
        result = {
            "emit_gold_segments": str(path), "videos": len(records), 
            "segments": sum(1 for r in records for sp in r.sentences if getattr(sp, "reliable", True))
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0)
    if args.segmenter_eval:
        print(json.dumps(run_segmenter_eval(args), indent=2, sort_keys=True))
        raise SystemExit(0)

    if args.rq == "1": result = run_rq1(args)
    elif args.rq == "2": result = run_rq2(args)
    else: raise SystemExit(
        "eval.py needs a mode: --rq 1 (controlled misalignment sweep), --rq 2 (end-to-end ladder), "
        "--segmenter-eval (whole-video segmentation), or --emit-gold-segments (write GT spans)."
    )
    display(result)
