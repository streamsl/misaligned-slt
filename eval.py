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
pd.set_option("display.float_format", "{:.4f}".format)

from poses import load_pose_window
from data.windowing import SentenceSpan
from data.loader import VideoRecord, load_language_records
from models.unisign import UniSignMT5FrontEnd, UniSignMBartFrontEnd, load_unisign_pretrained, prompt_lang_for_target
from models.streaming_slt import MisalignedSLTModel
from models.checkpointing import load_model_checkpoint
from metrics import Segment, match_segments, segmentation_prf, compute_text_metrics
from utils import checkpoint_dir, load_yaml, language_model_name, pretrained_checkpoint


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
    delta_head_s: float     # realized offset in seconds (= grid_head * duration in relative mode)
    delta_tail_s: float
    grid_head: float = 0.0  # grid coordinate used for grouping (a fraction of duration in relative mode)
    grid_tail: float = 0.0


def _load_segments(path: str | Path) -> list[Segment]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Segment(float(row["start_s"]), float(row["end_s"])) for row in rows]


def _gold_events(records: list[VideoRecord]) -> dict[str, list[PredictionEvent]]:
    return {record.video_id: [PredictionEvent(
        video_id=record.video_id, start_s=float(span.start_s), end_s=float(span.end_s), text=span.text,
    ) for span in record.sentences] for record in records}


def write_gold_segments(records: list[VideoRecord], path: str | Path) -> Path:
    """Emit GT sentence spans in the `--segments` schema (load_prediction_file dict form).

    Feeds the RQ2 oracle-input rows: `eval.py --rq 2 --segments <this> --method {stage2_baseline,stage2_dlm}`
    gives (clean AR | our DLM) translating GT-trimmed spans offline — the ceiling rows of the RQ2 ladder.
    Identical translation to RQ1 @ delta=0, but DVC-framed so it sits in the same table as the other rungs.
    """
    rows = {record.video_id: [
        {"start_s": float(span.start_s), "end_s": float(span.end_s), "text": span.text}
        for span in record.sentences
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
    # Load predicted segments, optionally with text, from common JSON layouts.
    data = json.loads(Path(path).read_text(encoding="utf-8"))
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
) -> list[ControlledWindow]:
    """Build RQ1 signed-offset windows.

    Analysis A defines deltas as predicted boundary minus GT boundary, so perturbed window uses start = gt_start + delta_head 
    and end = gt_end + delta_tail. Negative delta_tail is the intended right-truncation stress test.

    `relative=True` (default): each grid value is a FRACTION of the anchor sentence's own duration, so the realized offset scales 
    with sentence length (delta_s = grid * duration). Absolute-seconds offsets mix regimes — a 0.3s head cut destroys a 1s sentence 
    but barely touches a 10s one, so the curve would average 2 different stress levels at one x-point. Relative perturbation keeps 
    every sentence at the same proportional severity. `relative=False` restores absolute-seconds offsets.
    """
    windows: list[ControlledWindow] = []
    count = 0
    for record in records:
        for span in record.sentences:
            if max_sentences is not None and count >= int(max_sentences): return windows
            count += 1
            duration = max(1e-6, float(span.end_s) - float(span.start_s))
            for gh in grid:
                for gt in grid:
                    dh = float(gh) * duration if relative else float(gh)
                    dt = float(gt) * duration if relative else float(gt)
                    start_s = max(0.0, float(span.start_s) + dh)
                    end_s = min(float(record.pose.duration_s), float(span.end_s) + dt)
                    if end_s <= start_s: continue
                    windows.append(ControlledWindow(
                        video_id=record.video_id, reference=span.text,
                        gt_start_s=float(span.start_s), gt_end_s=float(span.end_s),
                        window_start_s=start_s, window_end_s=end_s,
                        delta_head_s=dh, delta_tail_s=dt, grid_head=float(gh), grid_tail=float(gt),
                    ))
    return windows


def evaluate_predicted_events(
    predicted: dict[str, list[PredictionEvent]],
    gold: dict[str, list[PredictionEvent]],
    thresholds: list[float],
) -> dict[str, Any]:
    results = []
    all_video_ids = sorted(set(predicted) | set(gold))
    for threshold in thresholds:
        total_pred, total_gold = 0, 0
        pred_texts: list[str] = []
        ref_texts: list[str] = []
        latencies: list[float] = []
        matched_pairs = 0

        matched_tious: list[float] = []
        # Recall-inclusive translation: every gold sentence contributes, missed ones as an empty hypothesis,
        # so the corpus metric reflects BOTH localisation (did we emit near it) and translation quality —
        # the matched-only number above hides the misses (it averages over the easy, matched subset only).
        matched_gold_text: dict[tuple[str, int], str] = {}
        for video_id in all_video_ids:
            pred_events = predicted.get(video_id, [])
            gold_events = gold.get(video_id, [])
            pred_by_idx = [event.segment for event in pred_events]
            gold_by_idx = [event.segment for event in gold_events]
            total_pred += len(pred_by_idx)
            total_gold += len(gold_by_idx)

            matches = match_segments(pred_by_idx, gold_by_idx, threshold=float(threshold))
            matched_pairs += len(matches)
            for pred_idx, gold_idx, match_tiou in matches:
                matched_tious.append(float(match_tiou))
                pred_event = pred_events[pred_idx]
                gold_text = gold_events[gold_idx].text
                if pred_event.text is not None and gold_text is not None:
                    pred_texts.append(pred_event.text)
                    ref_texts.append(gold_text)
                    matched_gold_text[(video_id, gold_idx)] = pred_event.text
                if pred_event.commit_time_s is not None: # Emission latency: commit time minus GT sentence end (spec §9.2).
                    latencies.append(float(pred_event.commit_time_s) - float(gold_events[gold_idx].end_s))

        ri_hyps: list[str] = []
        ri_refs: list[str] = []
        for video_id in all_video_ids:
            for gold_idx, gold_event in enumerate(gold.get(video_id, [])):
                if gold_event.text is None: continue
                ri_refs.append(gold_event.text)
                ri_hyps.append(matched_gold_text.get((video_id, gold_idx), ""))  # missed gold -> empty hypothesis

        precision = matched_pairs / total_pred if total_pred else 0.0
        recall = matched_pairs / total_gold if total_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        latency_block = {}
        if latencies:
            arr = np.sort(np.asarray(latencies, dtype=np.float64))
            latency_block = {
                "median_latency_s": float(np.median(arr)),
                "p90_latency_s": float(arr[min(len(arr) - 1, int(round(0.9 * (len(arr) - 1))))]),
                "n": len(arr),
            }
        tiou_block = {}
        if matched_tious:
            tiou_arr = np.asarray(matched_tious, dtype=np.float64)
            tiou_block = {
                "min": float(np.min(tiou_arr)),
                "p25": float(np.percentile(tiou_arr, 25)),
                "median": float(np.median(tiou_arr)),
                "p75": float(np.percentile(tiou_arr, 75)),
                "max": float(np.max(tiou_arr)),
            }
        results.append({
            "tiou_threshold": float(threshold),
            "segmentation": {"precision": precision, "recall": recall, "f1": f1, "matches": float(matched_pairs)},
            "matched_segments": matched_pairs,
            "unmatched_predicted": total_pred - matched_pairs,
            "unmatched_gold": total_gold - matched_pairs,
            "matched_translation_pairs": len(pred_texts),
            # Diagnostic: when this is ~1.0 the metrics are (correctly) identical across tIoU thresholds —
            # the predicted spans localize almost exactly onto gold, so every threshold keeps the same matches.
            # A flat severity/threshold curve there is a property of the predictions, not an eval bug.
            "mean_matched_tiou": float(sum(matched_tious) / len(matched_tious)) if matched_tious else 0.0,
            "matched_tiou_distribution": tiou_block,
            "translation_metrics": compute_text_metrics(pred_texts, ref_texts),
            # Recall-inclusive: missed gold sentences scored as empty hypotheses (LOWER than matched-only;
            # the gap to translation_metrics is exactly the cost of the sentences the FSM never emitted).
            "translation_metrics_recall_inclusive": compute_text_metrics(ri_hyps, ri_refs),
            "recall_inclusive_pairs": len(ri_refs),
            "emission_latency": latency_block,
        })
    return {"thresholds": results}


def _parse_grid(value: str | None, fallback: list[float]) -> list[float]:
    if not value: return [float(x) for x in fallback]
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _method_config_path(args: argparse.Namespace) -> str:
    if args.method_config: return str(args.method_config)
    defaults = {
        "stage2_baseline": "configs/stage2_baseline.yaml",
        "stage2_ar": "configs/stage2_ar.yaml",
        "stage2_dlm": "configs/stage2_dlm.yaml",
    }
    return defaults[args.method]


def _build_eval_model(args: argparse.Namespace, data_cfg: dict, stage1_cfg: dict, method_cfg: dict, device: torch.device):
    """Uni-Sign pose-only model. The language model is selected by `language_model.name`: mT5 (Path A default) or
    mBART (the mT5-vs-mBART ablation). The `stage2_baseline` method is the released Uni-Sign mT5 model (eval-only)."""
    from transformers import T5Tokenizer, AutoTokenizer
    target_lang = data_cfg["languages"][args.language].get("target_lang")
    prompt_lang = prompt_lang_for_target(target_lang)
    lm_name = language_model_name(stage1_cfg)

    if args.method == "stage2_baseline":
        # The baseline is the RELEASED Uni-Sign mT5 pose-only model (there is no released mBART SLT), evaluated
        # AR-only with beam search — the literature-comparison floor / clean point. It is the SAME
        # `MisalignedSLTModel(decoder="ar")` as stage2_ar, just with the released weights instead of trained ones
        # (its BIO head is unused for translation). `load_unisign_pretrained` strict-loads the front end (pose+mT5).
        mt5_name = lm_name if "mt5" in lm_name.lower() else "google/mt5-base"
        tokenizer = T5Tokenizer.from_pretrained(mt5_name, legacy=False)
        front_end = UniSignMT5FrontEnd(mt5_name=mt5_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)
        model = MisalignedSLTModel(front_end=front_end, tokenizer=tokenizer, decoder="ar",
                                   bio_hidden_dim=int(method_cfg.get("bio_hidden_dim", 384)))
        ckpt = Path(args.checkpoint or pretrained_checkpoint(stage1_cfg, default="")
                    or checkpoint_dir(method_cfg, default="") or "")
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing Uni-Sign checkpoint: {ckpt}. Set checkpoint.from_pretrained or pass --checkpoint.")
        rep = load_unisign_pretrained(model, ckpt, strict=True)
        print(f"[unisign] loaded {ckpt.name}: {rep['pose_tensors']} pose + {rep['mt5_tensors']} LM tensors (missing "
              f"{rep['pose_missing'] + rep['mt5_missing']}, unexpected {rep['pose_unexpected'] + rep['mt5_unexpected']})", flush=True)
        model.to(device); model.eval()
        return model, tokenizer

    # Trained stage2_ar / stage2_dlm: the unified MisalignedSLTModel on the Uni-Sign front end (mT5 or mBART) + the
    # trained checkpoint. Same pose encoder + prompt either way — only the LM differs (clean mT5-vs-mBART ablation).
    if "mbart" in lm_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(lm_name, src_lang=target_lang, tgt_lang=target_lang)
        front_end = UniSignMBartFrontEnd(mbart_name=lm_name, prompt_lang=prompt_lang, target_lang=target_lang, tokenizer=tokenizer)
    else:
        tokenizer = T5Tokenizer.from_pretrained(lm_name, legacy=False)
        front_end = UniSignMT5FrontEnd(mt5_name=lm_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)
    model = MisalignedSLTModel(
        front_end=front_end, tokenizer=tokenizer,
        decoder="ar" if args.method == "stage2_ar" else "dlm",
        bio_hidden_dim=int(method_cfg.get("bio_hidden_dim", 384)),
        block_size=int(method_cfg.get("block_size", 8)),
    )
    checkpoint = Path(args.checkpoint or checkpoint_dir(method_cfg, default="") or "")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for {args.method}: {checkpoint}. Train the method first or pass --checkpoint.")
    load_model_checkpoint(model, checkpoint, strict=False)
    model.to(device); model.eval()
    return model, tokenizer


def _prep_window(
    poses_np: np.ndarray, timestamps_np: np.ndarray, start_s: float, visual_padding: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Window pose tensor + window-relative timestamps + frame mask. Uni-Sign uses raw windows
    (`visual_padding: none`) — the mT5/mBART encoder masks padding via the attention mask; no boundary halos."""
    poses = torch.as_tensor(poses_np, dtype=torch.float32)
    ts = torch.as_tensor(np.asarray(timestamps_np, dtype=np.float32) - float(start_s), dtype=torch.float32)
    if visual_padding in {"none", "zero"}: mask = torch.ones(poses.shape[0], dtype=torch.bool)
    else: raise ValueError(f"Unsupported visual_padding={visual_padding!r} (Uni-Sign uses 'none')")
    return poses, ts, mask


def _generation_kwargs(method: str, inference_cfg: dict, method_cfg: dict, max_tokens: int) -> dict:
    """Build the `MisalignedSLTModel.generate_from_poses` kwargs for a method. Baseline = AR **beam search** (the
    literature-comparison floor); stage2_ar / stage2_dlm stay **greedy** (num_beams=1) so §9.3 is a clean
    AR-vs-DLM contrast — the DLM arm additionally uses the SPD/DCD params (the AR arms ignore them)."""
    if method == "stage2_baseline":
        num_beams = int(method_cfg.get("validation", {}).get("num_beams", method_cfg.get("num_beams", 4)))
        return {"max_text_tokens": max_tokens, "num_beams": num_beams}

    trans_cfg = inference_cfg.get("translation", {})
    dcd_cfg = trans_cfg.get("dcd", method_cfg.get("dcd", {}))
    spd_cfg = method_cfg.get("spd", {})
    return {
        "max_text_tokens": max_tokens, "num_beams": 1,
        "diffusion_steps": int(trans_cfg.get("diffusion_steps", method_cfg.get("diffusion_steps", 64))),
        "tau_dec": float(dcd_cfg.get("tau_dec", trans_cfg.get("commit_confidence_tau", 0.75))),
        "spd_top_k": int(spd_cfg.get("top_k", 1)),
        "spd_renormalize": bool(spd_cfg.get("renormalize", True)),
        "spd_revision": bool(spd_cfg.get("revision", True)),
        "temperature": float(dcd_cfg.get("temperature", 0.0)),
        "dcd_window_length": int(dcd_cfg.get("initial_window_length", method_cfg.get("block_size", 8))),
        "dcd_max_window_length": int(dcd_cfg.get("max_window_length", 64)),
        "dcd_window_type": str(dcd_cfg.get("window_type", "sliding")),
        "dcd_decode_algo": str(dcd_cfg.get("decode_algo", "threshold")),
        "dcd_decode_param": dcd_cfg.get("decode_param", trans_cfg.get("commit_confidence_tau", 0.75)),
        "dcd_sample_top_k": None if dcd_cfg.get("top_k") is None else int(dcd_cfg.get("top_k")),
        "dcd_top_p": None if dcd_cfg.get("top_p") is None else float(dcd_cfg.get("top_p")),
        "dcd_cache_type": str(dcd_cfg.get("cache_type", "none")),
        "dcd_refresh_count": int(dcd_cfg.get("refresh_count", 16)),
    }


@torch.no_grad()
def _translate_windows(
    model, tokenizer, method: str, items: list[tuple[np.ndarray, np.ndarray, float]],
    device: torch.device, inference_cfg: dict, method_cfg: dict,
) -> list[tuple[str, float]]:
    """Translate a batch of pre-trimmed pose windows -> [(text, mean_token_confidence)].

    ONE path for every method — each is `MisalignedSLTModel.generate_from_poses` (baseline = AR beam search on the
    released model; stage2_ar = AR greedy; stage2_dlm = SPD/DCD). It returns the REAL per-token confidence (the
    softmax prob the model assigns its own tokens), so the §9.1 "confidently wrong" claim is measured, not a
    placeholder. Variable-length windows are right-padded and masked via `frame_mask` (the LM encoder attends only
    to real frames; SPD/DCD reads per-row lengths), so each row's result is identical to translating it alone."""
    if not items: return []
    visual_padding = str(method_cfg.get("visual_padding", "none"))
    prepped = [_prep_window(p, ts, start_s, visual_padding) for (p, ts, start_s) in items]
    max_t = max(int(p.shape[0]) for p, _, _ in prepped)
    pose_shape = tuple(prepped[0][0].shape[1:])
    batch = len(items)
    
    poses = torch.zeros((batch, max_t, *pose_shape), dtype=torch.float32)
    timestamps = torch.zeros((batch, max_t), dtype=torch.float32)
    frame_mask = torch.zeros((batch, max_t), dtype=torch.bool)
    for i, (p_t, ts_t, m_t) in enumerate(prepped):
        n = int(p_t.shape[0])
        poses[i, :n] = p_t
        timestamps[i, :n] = ts_t
        frame_mask[i, :n] = m_t
    poses, timestamps, frame_mask = poses.to(device), timestamps.to(device), frame_mask.to(device)

    max_tokens = int(method_cfg.get("max_text_tokens", inference_cfg.get("translation", {}).get("max_text_tokens", 128)))
    gen_kwargs = _generation_kwargs(method, inference_cfg, method_cfg, max_tokens)
    _, tokens, confidence = model.generate_from_poses(poses=poses, frame_mask=frame_mask, timestamps_s=timestamps, **gen_kwargs)
    tok = tokens.detach().cpu()
    conf = confidence.detach().float().cpu()
    texts = [t.strip() for t in tokenizer.batch_decode(tok, skip_special_tokens=True)]

    # Per-row confidence = mean prob over the REAL produced tokens only — drop the decoder-start slot (its conf is
    # the placeholder 1.0) and any padding after EOS, so the §9.1 "confidently wrong" signal is not diluted by pads.
    n = min(tok.shape[1], conf.shape[1])
    tok, conf = tok[:, :n], conf[:, :n]
    valid = tok != int(tokenizer.pad_token_id)
    if n: valid[:, 0] = False
    confs = ((conf * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)).tolist()
    return list(zip(texts, [float(c) for c in confs]))


@torch.no_grad()
def _translate_window(
    model, tokenizer, method: str,
    poses_np: np.ndarray, timestamps_np: np.ndarray, start_s: float,
    device: torch.device, inference_cfg: dict, method_cfg: dict,
) -> tuple[str, float]:
    # Single-window convenience wrapper over `_translate_windows` (used by the RQ2 pipeline floor + analyze.py).
    return _translate_windows(model, tokenizer, method, [(poses_np, timestamps_np, start_s)], device, inference_cfg, method_cfg)[0]


def run_rq1(args: argparse.Namespace) -> dict[str, Any]:
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run RQ1 on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    stage1_cfg = load_yaml(args.stage1_config)
    method_cfg = load_yaml(_method_config_path(args))
    records, _ = load_language_records(data_cfg, args.language, split=args.split)

    rq_cfg = eval_cfg.get("rq1", {})
    mode = str(args.severity_mode or rq_cfg.get("severity_mode", "relative"))
    relative = mode == "relative"
    if relative: default_key = "smoke_grid_rel" if args.smoke else "severity_grid_rel"
    else: default_key = "smoke_grid_s" if args.smoke else "severity_grid_s"
    grid = _parse_grid(args.severity_grid_s, rq_cfg.get(default_key, [0.0]))
    max_sentences = args.num_sentences
    if max_sentences is None and args.smoke: max_sentences = int(rq_cfg.get("smoke_num_sentences", 10))
    windows = controlled_windows(records, grid, relative=relative, max_sentences=max_sentences)

    records_by_id = {record.video_id: record for record in records}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    inference_cfg = load_yaml(args.inference_config)
    model, tokenizer = _build_eval_model(args, data_cfg, stage1_cfg, method_cfg, device)

    # Materialize every non-empty window, then translate in length-sorted batches (sorting keeps each batch near-uniform length 
    # so padding — and wasted compute — is minimal). Padding is masked, so batching not change any per-window result, only throughput.
    materialized: list[tuple[ControlledWindow, np.ndarray, np.ndarray]] = []
    for window in windows:
        record = records_by_id[window.video_id]
        poses, timestamps = load_pose_window(record.pose, window.window_start_s, window.window_end_s, normalize=True)
        if poses.shape[0] == 0: continue
        materialized.append((window, poses, timestamps))

    materialized.sort(key=lambda wp: int(wp[1].shape[0]))
    batch_size = max(1, int(rq_cfg.get("batch_size", 16)))
    grouped: dict[tuple[float, float], dict[str, list]] = {}
    rows = []
    for start in tqdm(range(0, len(materialized), batch_size), desc="Translating windows"):
        chunk = materialized[start : start + batch_size]
        results = _translate_windows(
            model=model, tokenizer=tokenizer, method=args.method,
            items=[(poses, timestamps, w.window_start_s) for (w, poses, timestamps) in chunk],
            device=device, inference_cfg=inference_cfg, method_cfg=method_cfg,
        )
        for (window, _poses, _ts), (prediction, confidence) in zip(chunk, results):
            key = (window.grid_head, window.grid_tail)  # group by grid coordinate (fraction in relative mode)
            grouped.setdefault(key, {"predictions": [], "references": [], "confidences": [], "head_s": [], "tail_s": []})
            grouped[key]["predictions"].append(prediction)
            grouped[key]["references"].append(window.reference)
            grouped[key]["confidences"].append(confidence)
            grouped[key]["head_s"].append(window.delta_head_s)
            grouped[key]["tail_s"].append(window.delta_tail_s)
            rows.append({**asdict(window), "prediction": prediction, "mean_confidence": confidence})

    severity = []
    for (gh, gt), values in tqdm(sorted(grouped.items()), desc="Computing severity"):
        confs = values["confidences"]
        head_s, tail_s = values["head_s"], values["tail_s"]
        severity.append({
            # "severity_mode": mode,
            "windows": len(values["predictions"]),
            "grid_head": gh, "grid_tail": gt,  # fraction of sentence duration in relative mode, else seconds
            "delta_head_s_mean": float(sum(head_s) / len(head_s)) if head_s else 0.0,  # realized offset (relative -> varies per sentence)
            "delta_tail_s_mean": float(sum(tail_s) / len(tail_s)) if tail_s else 0.0,
            "mean_translation_confidence": float(sum(confs) / len(confs)) if confs else 0.0,
            "text_metrics": compute_text_metrics(values["predictions"], values["references"]),
        })
    # summary = {
    #     "rq": "1", "language": args.language, "split": args.split, "method": args.method,
    #     "severity_mode": mode, "grid": grid, "windows": len(rows), "severity": severity,
    # }
    return pd.json_normalize(severity, sep=".").T # one row per (grid_head, grid_tail) severity point


def _build_streaming_runner(model, inference_cfg: dict, method_cfg: dict):
    from infer.stream import StreamingSLTRunner
    trans = inference_cfg.get("translation", {})
    dcd = trans.get("dcd", {})
    spd = method_cfg.get("spd", {})
    boundary = inference_cfg.get("boundary_stability", {})

    return StreamingSLTRunner(
        model,
        stride_s=float(inference_cfg.get("stride_s", 1.0)),
        buffer_cap_s=float(inference_cfg.get("buffer_cap_s", 18.0)),
        delta_enc_frames=int(boundary.get("delta_enc_frames", 3)),
        hysteresis_strides=int(boundary.get("hysteresis_strides", 2)),
        token_confidence_tau=float(trans.get("commit_confidence_tau", 0.75)),
        max_text_tokens=int(trans.get("max_text_tokens", method_cfg.get("max_text_tokens", 128))),
        diffusion_steps=int(trans.get("diffusion_steps", method_cfg.get("diffusion_steps", 64))),
        tau_dec=float(dcd.get("tau_dec", trans.get("commit_confidence_tau", 0.75))),
        spd_top_k=int(spd.get("top_k", 1)),
        spd_renormalize=bool(spd.get("renormalize", True)),
        spd_revision=bool(spd.get("revision", True)),
        temperature=float(dcd.get("temperature", 0.0)),
        dcd_window_length=int(dcd.get("initial_window_length", trans.get("block_size", 8))),
        dcd_max_window_length=int(dcd.get("max_window_length", 64)),
        dcd_window_type=str(dcd.get("window_type", "sliding")),
        dcd_decode_algo=str(dcd.get("decode_algo", "threshold")),
        dcd_decode_param=dcd.get("decode_param", trans.get("commit_confidence_tau", 0.75)),
        dcd_cache_type=str(dcd.get("cache_type", "none")),
        dcd_refresh_count=int(dcd.get("refresh_count", 16)),
        decode_conditioning=str(trans.get("decode_conditioning", "window")),
    )


@torch.no_grad()
def run_streaming(args: argparse.Namespace) -> dict[str, list[PredictionEvent]]:
    """Drive the sawtooth FSM end-to-end over each video → committed events.

    This is the *usable inference engine* for RQ2: a raw pose stream in, committed `(start, end, text, flagged_partial, commit_time)` 
    events out — our own BIO head and commit gate, recompute-each-stride, no cross-stride decoder state (§7). Only valid for the FSM 
    methods (stage2_dlm / stage2_ar); the clean baseline's RQ2 is the segment-then-translate pipeline floor (use --predictions).
    """
    if args.method == "stage2_baseline":
        raise SystemExit("Streaming RQ2 uses the FSM (stage2_dlm/stage2_ar). For the baseline pipeline-floor, pass --predictions.")
    data_cfg = load_yaml(args.data_config)
    stage1_cfg = load_yaml(args.stage1_config)
    method_cfg = load_yaml(_method_config_path(args))
    inference_cfg = load_yaml(args.inference_config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer = _build_eval_model(args, data_cfg, stage1_cfg, method_cfg, device)
    runner = _build_streaming_runner(model, inference_cfg, method_cfg)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)

    predicted: dict[str, list[PredictionEvent]] = {}
    for record in tqdm(records, desc="Processing records"):
        poses, _ = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=True)
        if poses.shape[0] == 0:
            predicted[record.video_id] = []
            continue

        events = runner.run(torch.as_tensor(poses, dtype=torch.float32), fps=float(record.pose.fps))
        predicted[record.video_id] = [PredictionEvent(
            video_id=record.video_id, start_s=float(ev.start_s), end_s=float(ev.end_s),
            text=tokenizer.decode(ev.token_ids.tolist(), skip_special_tokens=True).strip(),
            flagged_partial=bool(ev.flagged_partial), commit_time_s=float(ev.commit_time_s),
        ) for ev in events]

    # Why-did-it-(not)-commit summary. Low streaming recall with near-perfect frame BIO is the gate
    # suppressing emission; this shows which signal blocks. spans_seen = complete spans the FSM saw;
    # boundary_ok / translation_ok = how many passed each gate signal; committed = actual emissions.
    s = runner.gate_stats
    seen = s.get("spans_seen", 0)
    if seen: print(
        f"[stream] gate: spans_seen={seen} boundary_ok={s.get('boundary_ok',0)} "
        f"translation_ok={s.get('translation_ok',0)} committed={s.get('committed',0)} "
        f"forced={s.get('forced_commit',0)} | translation_ok rate={s.get('translation_ok',0)/seen:.2f} "
        f"(if this is low, the commit gate's token-confidence floor is suppressing a weak decoder, not an eval bug)", flush=True)
    return predicted


@torch.no_grad()
def run_pipeline_floor(args: argparse.Namespace) -> dict[str, list[PredictionEvent]]:
    """Segment-then-translate pipeline floor: The retrained Moryossef segmenter's predicted spans (analyze.py --stage segmenter-infer JSON,
    via --segments) are cut from the pose stream and translated by clean-trained Uni-Sign baseline — natural pipeline with no robustness 
    training anywhere. Scored by the same tIoU/translation harness as the streaming FSM, so floor and method are directly comparable.
    """
    from moryossef26.infer import load_prediction_file
    data_cfg = load_yaml(args.data_config)
    stage1_cfg = load_yaml(args.stage1_config)
    method_cfg = load_yaml(_method_config_path(args))
    inference_cfg = load_yaml(args.inference_config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer = _build_eval_model(args, data_cfg, stage1_cfg, method_cfg, device)

    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    records_by_id = {record.video_id: record for record in records}
    segments = load_prediction_file(args.segments)

    # The split comes from --split, NOT the JSON filename. Mismatched video_ids silently translate nothing and
    # score all-zero, so fail loud instead (the classic "gold_*_test.json but forgot --split test" footgun).
    matched = set(segments) & set(records_by_id)
    if not matched: raise SystemExit(
        f"--segments has {len(segments)} video_ids, none in the {len(records_by_id)} '{args.split}' records — "
        f"nothing to translate (output would be all-zero). Pass --split test (+ --allow-test) for test gold spans."
    )
    if len(matched) < len(segments): print(
        f"[pipeline_floor] WARNING: {len(segments) - len(matched)}/{len(segments)} segment video_ids "
        f"are absent from --split {args.split}; scoring only the {len(matched)} that match.", flush=True)

    predicted: dict[str, list[PredictionEvent]] = {}
    for video_id, spans in tqdm(segments.items(), desc="Processing segments"):
        record = records_by_id.get(video_id)
        if record is None: continue
        events: list[PredictionEvent] = []
        for span in spans:
            poses, timestamps = load_pose_window(record.pose, span.start_s, span.end_s, normalize=True)
            if poses.shape[0] == 0: continue
            text, _ = _translate_window(
                model=model, tokenizer=tokenizer, method=args.method,
                poses_np=poses, timestamps_np=timestamps, start_s=span.start_s,
                device=device, inference_cfg=inference_cfg, method_cfg=method_cfg,
            )
            events.append(PredictionEvent(video_id=video_id, start_s=float(span.start_s), end_s=float(span.end_s), text=text))
        predicted[video_id] = events
    return predicted


def run_rq2(args: argparse.Namespace) -> dict[str, Any]:
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run RQ2 on test without --allow-test")
    if not args.predictions and not args.stream and not args.segments: raise SystemExit(
        "--rq 2 needs --stream (FSM engine), --segments (segment-then-translate pipeline floor), or --predictions (score an events JSON)"
    )
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    thresholds = _parse_grid(args.tiou_thresholds, eval_cfg.get("rq2", {}).get("tiou_thresholds", [0.3, 0.5, 0.7, 0.9]))

    if args.stream:
        predicted = run_streaming(args)
        events_path = Path(f"outputs/rq2_stream_events_{args.method}_{args.language}_{args.split}.json")
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text(json.dumps({
            vid: [asdict(ev) for ev in evs] for vid, evs in predicted.items()
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.segments:
        predicted = run_pipeline_floor(args)
        events_path = Path(f"outputs/rq2_pipeline_floor_events_{args.method}_{args.language}_{args.split}.json")
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text(json.dumps({
            vid: [asdict(ev) for ev in evs] for vid, evs in predicted.items()
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else: predicted = load_event_predictions(args.predictions)
        
    gold = _gold_events(records)
    # summary = {
    #     "rq": "2", "language": args.language, "split": args.split, "method": args.method,
    #     "predictions": source, "streamed": bool(args.stream),
    #     **evaluate_predicted_events(predicted, gold, thresholds),
    # }
    summary = evaluate_predicted_events(predicted, gold, thresholds).get("thresholds", [])
    summary = pd.json_normalize(summary, sep=".")  # one row per tIoU threshold
    summary.set_index("tiou_threshold", inplace=True)
    return summary.T


def run_segment_prf(args: argparse.Namespace) -> dict[str, Any]:
    if not args.pred or not args.gold: raise SystemExit("--pred and --gold JSON files are required when --rq is omitted")
    pred = _load_segments(args.pred)
    gold = _load_segments(args.gold)
    return segmentation_prf(pred, gold, tiou_threshold=args.tiou_threshold)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Misaligned-SLT evaluation entry points")
    parser.add_argument("--rq", choices=["1", "2"], default=None)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--stage1-config", default="configs/stage1_pretraining.yaml")
    parser.add_argument("--method-config", default=None)
    parser.add_argument("--language", default="phoenix")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--method", default="stage2_dlm", choices=["stage2_baseline", "stage2_ar", "stage2_dlm"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--severity-grid-s", default=None, 
        help="Comma-separated signed RQ1 grid (fractions of duration in relative mode, else seconds)"
    )
    parser.add_argument(
        "--severity-mode", default="relative", choices=["relative", "absolute"], 
        help="RQ1 perturbation: relative (fraction of sentence duration, default) or absolute seconds"
    )
    parser.add_argument("--emit-gold-segments", default=None, help="Write GT spans JSON (for --rq 2 --segments oracle-input rows) and exit")
    parser.add_argument("--num-sentences", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--predictions", default=None, help="RQ2 event JSON with start_s/end_s and optional text")
    parser.add_argument("--segments", default=None, help="RQ2 pipeline floor: segmenter spans JSON (analyze --stage segmenter-infer)")
    parser.add_argument("--stream", action="store_true", help="RQ2: run the streaming FSM engine to produce events")
    parser.add_argument("--events-out", default=None, help="Where to write streamed RQ2 events JSON")
    parser.add_argument("--tiou-thresholds", default=None, help="Comma-separated RQ2 tIoU thresholds")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--pred", default=None, help="Legacy segment-PRF predicted JSON list")
    parser.add_argument("--gold", default=None, help="Legacy segment-PRF gold JSON list")
    parser.add_argument("--tiou-threshold", type=float, default=0.1)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.emit_gold_segments:
        records, _ = load_language_records(load_yaml(args.data_config), args.language, split=args.split)
        path = write_gold_segments(records, args.emit_gold_segments)
        result = {"emit_gold_segments": str(path), "videos": len(records), "segments": sum(len(r.sentences) for r in records)}
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0)

    if args.rq == "1": result = run_rq1(args)
    elif args.rq == "2": result = run_rq2(args)
    else: result = run_segment_prf(args)
    display(result)
