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
from data.windowing import make_bio_labels
from data.loader import VideoRecord, load_language_records
from data.batch import frame_mask_for, repeat_last_frame

from transformers import T5Tokenizer, AutoTokenizer
from models.unisign import UniSignMT5FrontEnd, UniSignMBartFrontEnd, load_unisign_pretrained, prompt_lang_for_target
from models.streaming_slt import MisalignedSLTModel
from models.checkpointing import load_model_checkpoint
from moryossef26.infer import duration_decode_params, evaluate_segmenter_whole_video, fit_duration_prior
from metrics import Segment, match_segments, moryossef_segment_metrics, segmentation_prf, compute_text_metrics
from utils import checkpoint_dir, load_yaml, language_model_name, pick_device, resolve_pretrained


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


def save_prediction_file(predictions: dict[str, list[Segment]], path: str | Path) -> Path:
    # Write predicted phrase segments (the segmenter-infer / analysis artifact) as JSON.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"video_id": video_id, "segments": [{"start_s": float(s.start_s), "end_s": float(s.end_s)} for s in segments]}
        for video_id, segments in sorted(predictions.items())
    ]
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_prediction_file(path: str | Path) -> dict[str, list[Segment]]:
    # Read a predicted-segments JSON (dict {vid: [{start_s,end_s}]} or list-of-rows form).
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
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


def _gold_events(records: list[VideoRecord]) -> dict[str, list[PredictionEvent]]:
    return {record.video_id: [PredictionEvent(
        video_id=record.video_id, start_s=float(span.start_s), end_s=float(span.end_s), text=span.text,
    ) for span in record.sentences] for record in records}


def write_gold_segments(records: list[VideoRecord], path: str | Path) -> Path:
    """Emit GT sentence spans in the `--segments` schema (load_prediction_file dict form).

    Feeds the RQ2 oracle-input rows: `eval.py --rq 2 --segments <this> --method {baseline,dlm}`
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
    tail_grid: list[float] | None = None, drop_counts: dict[tuple[float, float], int] | None = None,
) -> list[ControlledWindow]:
    """Build RQ1 signed-offset windows.

    Analysis A defines deltas as predicted boundary minus GT boundary, so perturbed window uses start = gt_start + delta_head
    and end = gt_end + delta_tail. Negative delta_tail is the intended right-truncation stress test.

    `relative=True` (default): each grid value is a FRACTION of the anchor sentence's own duration, so the realized offset 
    scales with sentence length (delta_s = grid * duration). Absolute-seconds offsets mix regimes — a 0.3s head cut destroys a 
    1s sentence but barely touches a 10s one, so the curve would average 2 different stress levels at one x-point. Relative 
    perturbation keeps every sentence at the same proportional severity. `relative=False` restores absolute-seconds offsets.

    `tail_grid` (default: same as `grid`) decouples 2 axes. Needed because a corpus dictates which grid SIGNS are meaningful: 
    truncation points (head ≥ 0, tail ≤ 0) need no context beyond the sentence and run on any corpus incl. official pre-trimmed 
    releases; extension points (head < 0 / tail > 0) need a continuous timeline (streams) — pre-trimmed clips only carry thin 
    same-take rest-pose margins, so extensions there mostly clamp.
    """
    windows: list[ControlledWindow] = []
    count = 0
    tail_values = grid if tail_grid is None else tail_grid
    for record in records:
        for span in record.sentences:
            if max_sentences is not None and count >= int(max_sentences): return windows
            count += 1
            duration = max(1e-6, float(span.end_s) - float(span.start_s))
            for gh in grid:
                for gt in tail_values:
                    dh = float(gh) * duration if relative else float(gh)
                    dt = float(gt) * duration if relative else float(gt)
                    start_s = max(0.0, float(span.start_s) + dh)
                    end_s = min(float(record.pose.duration_s), float(span.end_s) + dt)
                    if end_s <= start_s:
                        # Fully-truncated window (mostly absolute mode on short sentences): unrepresentable, so it is dropped — but COUNTED, 
                        # because a severity row that silently sheds its short sentences averages over a longer-sentence subset than its 
                        # neighbours (selection bias the dropped_fraction column makes visible, mirroring clamped_fraction).
                        if drop_counts is not None: drop_counts[(float(gh), float(gt))] = drop_counts.get((float(gh), float(gt)), 0) + 1
                        continue
                    windows.append(ControlledWindow(
                        video_id=record.video_id, reference=span.text,
                        gt_start_s=float(span.start_s), gt_end_s=float(span.end_s),
                        window_start_s=start_s, window_end_s=end_s,
                        # REALIZED offsets after clamping to the available timeline — not the requested dh/dt. On pre-trimmed records the 
                        # context margin around a sentence is finite, so a large negative-head / positive-tail request clamps; reporting 
                        # the request as if it were realized would overstate severity exactly where the data limits it. grid_head/grid_tail 
                        # keep the requested coordinate for grouping.
                        delta_head_s=start_s - float(span.start_s), delta_tail_s=end_s - float(span.end_s),
                        grid_head=float(gh), grid_tail=float(gt),
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
                if pred_event.commit_time_s is not None: # Emission latency: commit time minus GT sentence end.
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

METHOD_CONFIGS = { # method -> its default config. Shared by eval + visualize so the map lives in one place.
    "baseline": "configs/baseline.yaml",
    "ar": "configs/ar.yaml",
    "dlm": "configs/dlm.yaml",
}
def _method_config_path(args: argparse.Namespace) -> str:
    return str(args.method_config) if args.method_config else METHOD_CONFIGS[args.method]


def _build_eval_model(method: str, checkpoint: str | None, language: str, data_cfg: dict, method_cfg: dict, device: torch.device):
    """Uni-Sign pose-only model. Everything (the language model via `language_model.name`, and the checkpoint via
    `checkpoint.from_pretrained` / `checkpoint.dir`) is read from the METHOD config — no separate stage-1 config.
    The `baseline` method is the released Uni-Sign mT5 model (eval-only). Takes plain scalars (method, checkpoint, 
    language), not an argparse.Namespace, so non-CLI callers (analyze/visualize) need no fake namespace."""
    target_lang = data_cfg["languages"][language].get("target_lang")
    prompt_lang = prompt_lang_for_target(target_lang)
    lm_name = language_model_name(method_cfg)

    if method == "baseline":
        # The baseline is the RELEASED Uni-Sign mT5 pose-only model (there is no released mBART SLT), evaluated
        # AR-only with beam search — the literature-comparison floor / clean point. It is the SAME
        # `MisalignedSLTModel(decoder="ar")` as ar, just with the released weights instead of trained ones
        # (its BIO head is unused for translation). `load_unisign_pretrained` strict-loads the front end (pose+mT5).
        mt5_name = lm_name if "mt5" in lm_name.lower() else "google/mt5-base"
        tokenizer = T5Tokenizer.from_pretrained(mt5_name, legacy=False)
        front_end = UniSignMT5FrontEnd(mt5_name=mt5_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)
        model = MisalignedSLTModel(
            front_end=front_end, tokenizer=tokenizer, decoder="ar", 
            bio_hidden_dim=int(method_cfg.get("bio_hidden_dim", 384))
        )
        # Load ONLY the released ckpt (NO fallback to a trained `checkpoint.dir`, so a concurrent/prior train-slt
        # run can never make the baseline silently pick up trained weights). Resolved PER LANGUAGE: the baseline
        # floor for asf/bfi is the released OpenASL (English) model, for csl the CSL (Chinese) one.
        ckpt = Path(checkpoint or resolve_pretrained(method_cfg, data_cfg, language, default="") or "")
        if not ckpt.exists(): raise FileNotFoundError(
            f"Missing released Uni-Sign checkpoint for baseline (language '{language}'): {ckpt!s}. Set "
            f"languages.{language}.pretrained_slt in configs/data.yaml (or checkpoint.from_pretrained / --checkpoint)."
        )
        rep = load_unisign_pretrained(model, ckpt, strict=True)
        print(f"[unisign] loaded {ckpt.name}: {rep['pose_tensors']} pose + {rep['mt5_tensors']} LM tensors (missing "
              f"{rep['pose_missing'] + rep['mt5_missing']}, unexpected {rep['pose_unexpected'] + rep['mt5_unexpected']})", flush=True)
        model.to(device); model.eval()
        return model, tokenizer

    # Trained ar / dlm: the unified MisalignedSLTModel on the Uni-Sign front end (mT5 or mBART) + the
    # trained checkpoint. Same pose encoder + prompt either way — only the LM differs (clean mT5-vs-mBART ablation).
    if "mbart" in lm_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(lm_name, src_lang=target_lang, tgt_lang=target_lang)
        front_end = UniSignMBartFrontEnd(mbart_name=lm_name, prompt_lang=prompt_lang, target_lang=target_lang, tokenizer=tokenizer)
    else:
        tokenizer = T5Tokenizer.from_pretrained(lm_name, legacy=False)
        front_end = UniSignMT5FrontEnd(mt5_name=lm_name, prompt_lang=prompt_lang, tokenizer=tokenizer, init_mt5_weights=False)
    model = MisalignedSLTModel(
        front_end=front_end, tokenizer=tokenizer,
        decoder="ar" if method == "ar" else "dlm",
        bio_hidden_dim=int(method_cfg.get("bio_hidden_dim", 384)),
        block_size=int(method_cfg.get("block_size", 8)),
    )
    ckpt = Path(checkpoint or checkpoint_dir(method_cfg, default="") or "")
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint for {method}: {ckpt}. Train the method first or pass --checkpoint.")
    load_model_checkpoint(model, ckpt, strict=False)
    model.to(device); model.eval()
    return model, tokenizer


def _prep_window(
    poses_np: np.ndarray, timestamps_np: np.ndarray, start_s: float, visual_padding: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Window pose tensor + window-relative timestamps + frame mask. Uni-Sign uses raw windows
    (`visual_padding: none`) — the mT5/mBART encoder masks padding via the attention mask; no boundary halos."""
    poses = torch.as_tensor(poses_np, dtype=torch.float32)
    timestamps = torch.as_tensor(np.asarray(timestamps_np, dtype=np.float32) - float(start_s), dtype=torch.float32)
    return poses, timestamps, frame_mask_for(poses.shape[0], visual_padding)


def _generation_kwargs(method: str, inference_cfg: dict, method_cfg: dict, max_tokens: int) -> dict:
    """Build the `MisalignedSLTModel.generate_from_poses` kwargs for a method. Baseline = AR **beam search** (literature-comparison 
    floor); ar / dlm stay **greedy** (num_beams=1) — the DLM arm additionally uses the SPD/DCD params (the AR arms ignore them)."""
    if method == "baseline":
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
        # Membership gate at RQ1: the DLM decodes under the same Ω conditioning it trained with (on-policy
        # span, no GT, no χ for single-window eval). BOTH arms are gated: the DLM injects Ω in its manual
        # decode; the AR arm receives the same Ω through HF cross-attention forward-hooks (front_end.ar_generate).
        "gate_enabled": bool(method_cfg.get("membership_gate", {}).get("enabled", False)),
        "gate_delta": int(method_cfg.get("membership_gate", {}).get("delta", 3)),
        "gate_eps": float(method_cfg.get("membership_gate", {}).get("eps", 1e-4)),
        "gate_min_span_frames": int(method_cfg.get("membership_gate", {}).get("min_span_frames",
                                    inference_cfg.get("span_selection", {}).get("min_span_frames", 0))),
    }


@torch.no_grad()
def _translate_windows(
    model, tokenizer, method: str, items: list[tuple[np.ndarray, np.ndarray, float]],
    device: torch.device, inference_cfg: dict, method_cfg: dict,
) -> list[tuple[str, float, bool]]:
    """Translate a batch of pre-trimmed pose windows -> [(text, mean_token_confidence, gate_would_skip)].

    ONE path for every method — each is `MisalignedSLTModel.generate_from_poses` (baseline = AR beam search on the released model; 
    ar = AR greedy; dlm = SPD/DCD). It returns REAL per-token confidence (the softmax prob the model assigns its own tokens), so 
    "confidently wrong" is measured, not a placeholder. Variable-length windows are right-padded and masked via `frame_mask` (LM 
    enc attends only to real frames; SPD/DCD reads per-row lengths), so each row's result is identical to translating it alone."""
    if not items: return []
    visual_padding = str(method_cfg.get("visual_padding", "none"))
    prepped = [_prep_window(p, ts, start_s, visual_padding) for (p, ts, start_s) in items]
    max_t = max(int(p.shape[0]) for p, _, _ in prepped)
    # Same repeat-last-frame pad + frame-mask contract as the training collator (data.batch), so an eval row is
    # padded exactly as it was in training and each row's result is identical to translating it alone.
    poses = torch.stack([repeat_last_frame(p, max_t - int(p.shape[0])) for p, _, _ in prepped]).to(device)
    timestamps = torch.stack([torch.nn.functional.pad(ts, (0, max_t - int(ts.shape[0]))) for _, ts, _ in prepped]).to(device)
    frame_mask = torch.stack([torch.nn.functional.pad(m, (0, max_t - int(m.shape[0]))) for _, _, m in prepped]).to(device)

    max_tokens = int(method_cfg.get("max_text_tokens", inference_cfg.get("translation", {}).get("max_text_tokens", 128)))
    gen_kwargs = _generation_kwargs(method, inference_cfg, method_cfg, max_tokens)
    _, tokens, confidence, gate_skip = model.generate_from_poses(poses=poses, frame_mask=frame_mask, timestamps_s=timestamps, **gen_kwargs)
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
    # gate_skip: the deployed FSM would never decode this window (no span — all-gap or headless fragment). The
    # decode above still ran (its text is reported), but RQ1 must be able to split "translation failed" from
    # "the system's policy is to defer/skip here" — conflating them charges the gate's refusal as garbage BLEU.
    skips = [bool(s) for s in gate_skip.tolist()]
    return list(zip(texts, [float(c) for c in confs], skips))


@torch.no_grad()
def _translate_window(
    model, tokenizer, method: str,
    poses_np: np.ndarray, timestamps_np: np.ndarray, start_s: float,
    device: torch.device, inference_cfg: dict, method_cfg: dict,
) -> tuple[str, float, bool]: # Single-window wrapper over `_translate_windows` (RQ2 pipeline floor + analyze.py).
    return _translate_windows(
        model, tokenizer, method, 
        [(poses_np, timestamps_np, start_s)], 
        device, inference_cfg, method_cfg
    )[0]


def run_rq1(args: argparse.Namespace) -> dict[str, Any]:
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run RQ1 on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    method_cfg = load_yaml(_method_config_path(args), language=args.language)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)

    rq_cfg = eval_cfg.get("rq1", {})
    mode = str(args.severity_mode or rq_cfg.get("severity_mode", "relative"))
    relative = mode == "relative"
    if relative: default_key = "smoke_grid_rel" if args.smoke else "severity_grid_rel"
    else: default_key = "smoke_grid_s" if args.smoke else "severity_grid_s"
    grid = _parse_grid(args.severity_grid_s, rq_cfg.get(default_key, [0.0]))
    # Optional per-axis grids: truncation-only sweeps (head ≥ 0, tail ≤ 0) are the valid sub-grid on
    # pre-trimmed corpora; extension points need a continuous corpus. Fall back to the shared grid.
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
    inference_cfg = load_yaml(args.inference_config)
    model, tokenizer = _build_eval_model(args.method, args.checkpoint, args.language, data_cfg, method_cfg, device)
    # Gate-side duration decode (same inference.yaml switch as streaming/training): generate_from_poses builds Ω
    # from the model's duration_prior, so the single-window gate selects the same duration-split anchors the
    # trained decoder was conditioned on. Only relevant when the gate is on; harmless otherwise.
    if bool(method_cfg.get("membership_gate", {}).get("enabled", False)):
        dd = duration_decode_params(inference_cfg)
        if dd is not None:
            train_records, _ = load_language_records(data_cfg, args.language, split="train")
            model.duration_prior = fit_duration_prior(train_records, **dd)

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
    for (gh, gt), values in tqdm(sorted(grouped.items()), desc="Computing severity"):
        confs = values["confidences"]
        head_s, tail_s = values["head_s"], values["tail_s"]
        # Fraction of windows whose REQUESTED offset was clamped by the record's timeline (start/end of the take). High values mean the corpus 
        # lacks the context this grid point asks for (pre-trimmed clips have thin rest-pose margins) — the realized means below then sit well 
        # inside the requested point and the row must not be read at face value.
        clamped = sum(1 for req, real in zip(values["req_head_s"], head_s) if abs(req - real) > 1e-3)
        clamped += sum(1 for req, real in zip(values["req_tail_s"], tail_s) if abs(req - real) > 1e-3)
        dropped = int(drop_counts.get((float(gh), float(gt)), 0))
        severity.append({
            "windows": len(values["predictions"]),
            "grid_head": gh, "grid_tail": gt,  # fraction of sentence duration in relative mode, else seconds
            "delta_head_s_mean": float(sum(head_s) / len(head_s)) if head_s else 0.0,  # realized offset (relative -> varies per sentence)
            "delta_tail_s_mean": float(sum(tail_s) / len(tail_s)) if tail_s else 0.0,
            "clamped_fraction": float(clamped) / max(1, 2 * len(head_s)),
            # Fully-truncated (unrepresentable) windows dropped at this grid point: a high value means this row
            # averages over a LONGER-sentence subset than its neighbours — read it like clamped_fraction.
            "dropped_fraction": float(dropped) / max(1, dropped + len(values["predictions"])),
            "mean_translation_confidence": float(sum(confs) / len(confs)) if confs else 0.0,
            "text_metrics": compute_text_metrics(values["predictions"], values["references"]),
            # Deployment-honest split for GATED methods (baseline: skip rate 0, decoded == all). A window with no span at all (headless 
            # left-truncation, all-gap) is a state the FSM SKIPS by design — force-decoding it against the inert no-span Ω yields near-empty 
            # hallucinations whose brevity penalty tanks the corpus BLEU of whole cell superlinearly. Read gated robustness as: decoded-only 
            # quality (windows the system would actually translate) TOGETHER WITH gate_skip_rate (its refusal/deferral rate); the all-windows 
            # text_metrics above stays as the coverage-pessimistic bound (mirrors RQ2's matched-only vs recall-inclusive dual).
            "gate_skip_rate": float(sum(values["gate_skips"])) / max(1, len(values["gate_skips"])),
            "text_metrics_decoded_only": compute_text_metrics(
                [p for p, s in zip(values["predictions"], values["gate_skips"]) if not s],
                [r for r, s in zip(values["references"], values["gate_skips"]) if not s],
            ),
        })
    # Persist the sweep (summary + every per-window prediction) — RQ1 sweeps are expensive and their artifacts feed the paper plots.
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "rq": "1", "language": args.language, "split": args.split, "method": args.method,
            # Per-axis grids as ACTUALLY swept — --severity-grid-head/--severity-grid-tail override the shared
            # grid, so persisting only `grid` would misdescribe the artifact's axes. `grid` kept for old readers.
            "severity_mode": mode, "grid": grid, "grid_head": head_grid, "grid_tail": tail_grid,
            "windows": len(rows), "severity": severity, "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[rq1] wrote {out}", flush=True)
    return pd.json_normalize(severity, sep=".").T # one row per (grid_head, grid_tail) severity point


def _build_streaming_runner(model, inference_cfg: dict, method_cfg: dict, duration_prior=None):
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
        hysteresis_strides=int(boundary.get("hysteresis_strides", 3)),
        token_confidence_tau=float(trans.get("commit_confidence_tau", 0.75)),
        # None (missing key) → the runner derives the minimal spec-safe Λ_min = 2δ+1; 0-fallback will silently
        # disabled the frozen Λ_min > 2δ invariant for any config lacking span_selection.
        min_span_frames=inference_cfg.get("span_selection", {}).get("min_span_frames"),
        forced_tail_policy=str(inference_cfg.get("forced_tail_policy", "skip")),
        # Membership gate: the RQ2 streaming decode runs under the SAME Ω conditioning the decoder trained
        # with (method config's membership_gate block); χ comes from the runner's own commit log.
        gate_enabled=bool(method_cfg.get("membership_gate", {}).get("enabled", False)),
        gate_delta=int(method_cfg.get("membership_gate", {}).get("delta", boundary.get("delta_enc_frames", 3))),
        gate_eps=float(method_cfg.get("membership_gate", {}).get("eps", 1e-4)),
        max_text_tokens=int(trans.get("max_text_tokens", method_cfg.get("max_text_tokens", 128))),
        diffusion_steps=int(trans.get("diffusion_steps", method_cfg.get("diffusion_steps", 64))),
        tau_dec=float(dcd.get("tau_dec", trans.get("commit_confidence_tau", 0.75))),
        spd_top_k=int(spd.get("top_k", 1)),
        spd_renormalize=bool(spd.get("renormalize", True)),
        spd_revision=bool(spd.get("revision", True)),
        temperature=float(dcd.get("temperature", 0.0)),
        # block_size is a MODEL property (the DLM decoder's block), so fall back to the METHOD config — same
        # source as _generation_kwargs. inference.yaml.translation has no block_size, so the old trans.block_size
        # fallback silently forced 8 whenever initial_window_length was unset.
        dcd_window_length=int(dcd.get("initial_window_length", method_cfg.get("block_size", 8))),
        dcd_max_window_length=int(dcd.get("max_window_length", 64)),
        dcd_window_type=str(dcd.get("window_type", "sliding")),
        dcd_decode_algo=str(dcd.get("decode_algo", "threshold")),
        dcd_decode_param=dcd.get("decode_param", trans.get("commit_confidence_tau", 0.75)),
        dcd_cache_type=str(dcd.get("cache_type", "none")),
        dcd_refresh_count=int(dcd.get("refresh_count", 16)),
        decode_conditioning=str(trans.get("decode_conditioning", "window")),
        duration_prior=duration_prior,
    )


@torch.no_grad()
def run_streaming(args: argparse.Namespace) -> dict[str, list[PredictionEvent]]:
    """Drive the sawtooth FSM end-to-end over each video → committed events.

    This is the *usable inference engine* for RQ2: raw pose stream in, committed `(start, end, text, flagged_partial, commit_time)` 
    events out — our own BIO head and commit gate, recompute-each-stride, no cross-stride decoder state. Only valid for the FSM 
    methods (dlm / ar); the clean baseline's RQ2 is the segment-then-translate pipeline floor (use --predictions).
    """
    if args.method == "baseline":
        raise SystemExit("Streaming RQ2 uses the FSM (dlm/ar). For the baseline pipeline-floor, pass --predictions.")
    data_cfg = load_yaml(args.data_config)
    method_cfg = load_yaml(_method_config_path(args), language=args.language)
    inference_cfg = load_yaml(args.inference_config)
    device = pick_device(args.device)
    model, tokenizer = _build_eval_model(args.method, args.checkpoint, args.language, data_cfg, method_cfg, device)
    # Opt-in buffer-level semi-Markov duration decode (inference.yaml duration_decode: true, or a mapping with the
    # per-language tuned {split_bias, snap_radius_s} from `analyze --stage tune-decode`; see infer/duration_decode.py).
    duration_prior = None
    dd = duration_decode_params(inference_cfg)
    if dd is not None:
        train_records, _ = load_language_records(data_cfg, args.language, split="train")
        duration_prior = fit_duration_prior(train_records, **dd)
        if duration_prior is not None: print(f"[streaming] buffer duration decode ON (prior from {len(train_records)} train videos"
                                            + (f"; tuned {dd}" if dd else "") + ")", flush=True)
        else: print(f"[streaming] duration_decode requested but <10 usable train captions — decoding plain argmax", flush=True)
    runner = _build_streaming_runner(model, inference_cfg, method_cfg, duration_prior=duration_prior)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)

    predicted: dict[str, list[PredictionEvent]] = {}
    fsm_bio_rows: list[dict[str, float]] = []
    for record in tqdm(records, desc="Processing records"):
        poses, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=True)
        if poses.shape[0] == 0:
            predicted[record.video_id] = []
            continue

        events = runner.run(torch.as_tensor(poses, dtype=torch.float32), fps=float(record.pose.fps))
        predicted[record.video_id] = [PredictionEvent(
            video_id=record.video_id, start_s=float(ev.start_s), end_s=float(ev.end_s),
            text=tokenizer.decode(ev.token_ids.tolist(), skip_special_tokens=True).strip(),
            flagged_partial=bool(ev.flagged_partial), commit_time_s=float(ev.commit_time_s),
        ) for ev in events]

        # "Moryossef-style BIO precision/recall/F1 on the FSM's internal BIO predictions": score the runner's
        # stitched per-frame argmax timeline against GT caption labels — the deployed head's segmentation quality,
        # as the FSM actually saw it stride-by-stride (latest estimate per frame).
        if runner._bio_timeline is not None:
            gold = make_bio_labels(
                timestamps, record.sentences, 0.0, float(record.pose.duration_s), video_duration_s=record.pose.duration_s,
            )
            tags = runner._bio_timeline
            logits_1hot = torch.nn.functional.one_hot(tags.clamp(min=0), num_classes=4).float().unsqueeze(0) * 10.0
            fsm_bio_rows.append(moryossef_segment_metrics(
                logits_1hot, torch.as_tensor(np.asarray(gold)).long().unsqueeze(0), prefix="fsm_bio",
            ))

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
    if fsm_bio_rows:
        fsm_bio = {k: float(sum(r[k] for r in fsm_bio_rows) / len(fsm_bio_rows)) for k in fsm_bio_rows[0]}
        print("[stream] FSM BIO (stitched per-frame argmax vs GT): "
              + " ".join(f"{k}={v:.3f}" for k, v in sorted(fsm_bio.items())), flush=True)
        run_streaming.last_fsm_bio = fsm_bio  # picked up by run_rq2 for the output payload
    else: run_streaming.last_fsm_bio = None
    return predicted


@torch.no_grad()
def run_pipeline_floor(args: argparse.Namespace) -> dict[str, list[PredictionEvent]]:
    """Segment-then-translate pipeline floor: predicted spans (analyze.py --stage segmenter-infer JSON, via --segments)
    are cut from the pose stream and translated offline. Scored by the same tIoU/translation harness as the streaming FSM.

    Pipeline FLOOR = an independent/external segmenter's spans translated by CLEAN baseline, so pass `--method baseline`. Translation uses 
    `args.method` (NOT pinned), so `--method dlm` here is a DIFFERENT ablation (external spans + our DLM, offline) — a legitimate RQ2 row, 
    but not the floor. The method is encoded in output filename so the two never collide; just pass the right --method for the row you want.
    """
    data_cfg = load_yaml(args.data_config)
    method_cfg = load_yaml(_method_config_path(args), language=args.language)
    inference_cfg = load_yaml(args.inference_config)
    device = pick_device(args.device)
    model, tokenizer = _build_eval_model(args.method, args.checkpoint, args.language, data_cfg, method_cfg, device)

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
            text, _, _ = _translate_window(
                model=model, tokenizer=tokenizer, method=args.method,
                poses_np=poses, timestamps_np=timestamps, start_s=span.start_s,
                device=device, inference_cfg=inference_cfg, method_cfg=method_cfg,
            )
            events.append(PredictionEvent(video_id=video_id, start_s=float(span.start_s), end_s=float(span.end_s), text=text))
        predicted[video_id] = events
    return predicted


@torch.no_grad()
def run_offline(args: argparse.Namespace) -> dict[str, list[PredictionEvent]]:
    """FINAL joint fine-tuned SLT model deployed OFFLINE. Its OWN jointly-trained BIO head segments each video — chunked at buffer_cap 
    so the head's attention span matches training, NOT one full-length pass — and the SAME model translates each resulting span, split 
    into <=buffer_cap sub-windows so no decode window exceeds training scale (a span merged across chunks can be longer than buffer_cap; 
    the FSM would force-commit it): no streaming FSM, no online commit/hysteresis, no cross-stride refinement.

    This is NOT the pipeline floor: here the boundaries come from the deployed model's own head. Held against streaming (`--stream`) at 
    the SAME model. This is to measure what the streaming machinery buys over offline deployment — folding in the cost of causal 
    (vs offline-bidirectional) segmentation, so this is a conservative baseline, not a pure-refinement control.
    """
    if args.method == "baseline": raise SystemExit(
        "Offline RQ2 (row 5) uses the trained model's own BIO head (dlm/ar); the baseline has no "
        "trained head. For the external-segmenter cascade floor, pass --segments."
    )
    from moryossef26.infer import bio_tags_to_segments
    data_cfg = load_yaml(args.data_config)
    method_cfg = load_yaml(_method_config_path(args), language=args.language)
    inference_cfg = load_yaml(args.inference_config)
    device = pick_device(args.device)

    model, tokenizer = _build_eval_model(args.method, args.checkpoint, args.language, data_cfg, method_cfg, device)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    buffer_cap_s = float(inference_cfg.get("buffer_cap_s", 18.0))

    predicted: dict[str, list[PredictionEvent]] = {}
    for record in tqdm(records, desc="Offline (self-segment + translate)"):
        poses, timestamps = load_pose_window(record.pose, 0.0, record.pose.duration_s, normalize=True)
        if poses.shape[0] == 0:
            predicted[record.video_id] = []
            continue
        # Segment the whole video with the model's OWN BIO head, chunked at the trained buffer scale (a full-length
        # single RoPE pass would attend over a context the head never trained on — Moryossef's train/infer lesson).
        poses_t = torch.as_tensor(poses, dtype=torch.float32, device=device).unsqueeze(0)
        ts_t = torch.as_tensor(timestamps, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.ones(poses_t.shape[:2], dtype=torch.bool, device=device)
        bio_tap, bio_mask, ts_out = model.front_end.extract_bio_tap(poses_t, mask_t, ts_t)
        model.bio_head.chunk_size = max(1, int(round(buffer_cap_s * float(record.pose.fps))))
        tags = model.bio_head(bio_tap, timestamps_s=ts_out, frame_mask=bio_mask).logits.argmax(dim=-1)[0].cpu()
        segments = bio_tags_to_segments(tags, timestamps.tolist())

        events: list[PredictionEvent] = []
        for span in segments:
            # The head can leave a span longer than buffer_cap (a continuous I-run merged across chunk boundaries).
            # Translating that in one shot feeds the decoder a window longer than ANY it trained on (windows are
            # clamped to buffer_cap by the sampler, and the streaming FSM force-commits at buffer_cap) — OOD RoPE +
            # attention. Mirror the FSM offline: split an over-long span into <=buffer_cap sub-windows, translate each.
            sub_start = float(span.start_s)
            while sub_start < float(span.end_s) - 1e-6:
                sub_end = min(float(span.end_s), sub_start + buffer_cap_s)
                span_poses, span_ts = load_pose_window(record.pose, sub_start, sub_end, normalize=True)
                if span_poses.shape[0] > 0:
                    text, _, _ = _translate_window(
                        model=model, tokenizer=tokenizer, method=args.method,
                        poses_np=span_poses, timestamps_np=span_ts, start_s=sub_start,
                        device=device, inference_cfg=inference_cfg, method_cfg=method_cfg,
                    )
                    events.append(PredictionEvent(video_id=record.video_id, start_s=sub_start, end_s=sub_end, text=text))
                sub_start = sub_end
        predicted[record.video_id] = events
    return predicted


def _write_events_json(predicted: dict[str, list[PredictionEvent]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {vid: [asdict(ev) for ev in evs] for vid, evs in predicted.items()}, 
        indent=2, sort_keys=True
    ) + "\n", encoding="utf-8")
    return out


def run_rq2(args: argparse.Namespace) -> dict[str, Any]:
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run RQ2 on test without --allow-test")
    if not args.predictions and not args.stream and not args.segments and not args.offline: raise SystemExit(
        "--rq 2 needs --stream (streaming FSM), --offline (final model self-segments offline), "
        "--segments (external-segmenter cascade floor, rows 3/4), or --predictions (score an events JSON)"
    )
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    thresholds = _parse_grid(args.tiou_thresholds, eval_cfg.get("rq2", {}).get("tiou_thresholds", [0.3, 0.5, 0.7, 0.9]))

    if args.stream:
        predicted = run_streaming(args)
        _write_events_json(predicted, f"outputs/rq2_stream_events_{args.method}_{args.language}_{args.split}.json")
        fsm_bio = getattr(run_streaming, "last_fsm_bio", None)
        if fsm_bio:  # FSM-internal BIO metric, persisted alongside the events
            path = Path(f"outputs/rq2_fsm_bio_{args.method}_{args.language}_{args.split}.json")
            path.write_text(json.dumps(fsm_bio, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.offline:
        predicted = run_offline(args)
        _write_events_json(predicted, f"outputs/rq2_offline_events_{args.method}_{args.language}_{args.split}.json")
    elif args.segments:
        predicted = run_pipeline_floor(args)
        # Name by the SPAN SOURCE (the --segments stem already carries arch+lang+split) AND method, so the cascade
        # rows never collide (external+dlm vs external+baseline, etc.).
        src = Path(args.segments).stem
        _write_events_json(predicted, f"outputs/rq2_pipeline_floor_{src}_{args.method}.json")
    else: predicted = load_event_predictions(args.predictions)
        
    gold = _gold_events(records)
    summary = evaluate_predicted_events(predicted, gold, thresholds).get("thresholds", [])
    summary = pd.json_normalize(summary, sep=".")  # one row per tIoU threshold
    summary.set_index("tiou_threshold", inplace=True)
    return summary.T


def run_segment_prf(args: argparse.Namespace) -> dict[str, Any]:
    if not args.pred or not args.gold: raise SystemExit("--pred and --gold JSON files are required when --rq is omitted")
    pred = _load_segments(args.pred)
    gold = _load_segments(args.gold)
    return segmentation_prf(pred, gold, tiou_threshold=args.tiou_threshold)


def _load_segmenter(args):
    """Build + load a trained segmenter by --segmenter-arch (shared by eval --segmenter-eval and
    analyze --stage segmenter-infer).

    external (default): the faithful Moryossef segmenter (raw keypoints + UNet; a DIFFERENT input space from the FSM
    head — the non-circular Analysis-A/B / RQ2-cascade instrument, gate-doc §1.4). s1: the in-system BIO head, an
    ablation that swaps the same Uni-Sign head in to isolate system design from segmentation competence.
    Returns (model, device, velocity, rope_chunk_s, checkpoint) — rope_chunk_s in SECONDS (S1) or None (external).
    """
    device = pick_device(args.device)
    if args.segmenter_arch == "s1":
        from train.bio_pretrain import build_bio_s1_model
        cfg = load_yaml(args.bio_config, language=args.language)
        pretrained = resolve_pretrained(cfg, load_yaml(args.data_config), args.language, default="checkpoints/csl_daily_pose_only_slt.pth")
        model = build_bio_s1_model(cfg, pretrained_path=pretrained)
        ckpt_default = f"checkpoints/bio_s1/{args.language}"
        # Uni-Sign features; whole-video chunked RoPE at the head's TRAINED context, in SECONDS: training windows are
        # clamped to buffer_cap_s (sampler.py), so eval chunks at buffer_cap_s — dataset-general (the wrapper converts
        # to frames at each stream's own fps). A larger chunk would attend over contexts the head never trained on.
        buffer_cap_s = float(load_yaml(args.inference_config).get("buffer_cap_s", 18.0))
        velocity, rope_chunk_s = False, float(cfg.get("rope_eval_chunk_s", buffer_cap_s))
    else:
        from moryossef26.trainer import build_segmenter
        cfg = load_yaml(args.segmenter_config, language=args.language)
        model = build_segmenter(args.segmenter_config)
        ckpt_default = f"checkpoints/segmenter/{args.language}"
        velocity, rope_chunk_s = bool(cfg.get("velocity", True)), None  # UNet chunks internally at num_frames

    # The config's checkpoint.dir had ${language} expanded from the config file's OWN `language:` key at load
    # time; when the CLI --language differs, that path points at another corpus's checkpoint — fall back to the
    # default (built from args.language) instead of silently loading e.g. the csl model for --language phoenix.
    ckpt_dir = checkpoint_dir(cfg, default=ckpt_default)
    if args.language and str(cfg.get("language", args.language)) != str(args.language): ckpt_dir = ckpt_default
    checkpoint = args.checkpoint or str(Path(ckpt_dir) / "model.pt")
    load_model_checkpoint(model, checkpoint, strict=True)
    return model, device, velocity, rope_chunk_s, checkpoint


def run_segmenter_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Standalone whole-video segmentation eval (Moryossef evaluate.py protocol): frame-F1 + one-to-one tIoU segment
    P/R/F1 on a split. The §4.6 acceptance check ("F1 within order-of-magnitude of published"), and — because it runs
    the SAME protocol for either --segmenter-arch — the ONLY apples-to-apples way to compare the Moryossef segmenter
    against the in-system S1 head: the two training monitors are NOT comparable (chunk windows vs misaligned windows).
    For `s1` it also scores the pretrained head on its own, without waiting for stage-2 joint fine-tuning.
    """
    if args.split == "test" and not args.allow_test: raise SystemExit("Refusing to run segmenter eval on test without --allow-test")
    data_cfg = load_yaml(args.data_config)
    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    model, device, velocity, rope_chunk_s, checkpoint = _load_segmenter(args)
    # Per-arch decode defaults: `s1` (OUR system's head) -> the semi-Markov duration re-split, this system's contribution 
    # (infer/duration_decode.py); `external` -> `plain` argmax. Override either way with --segmenter-decode.
    decode = args.segmenter_decode or ("duration" if args.segmenter_arch == "s1" else "plain")
    duration_prior = None
    if decode == "duration":
        # Enablement is per-arch/CLI (above), but the PARAMETERS come from inference.yaml's duration_decode mapping
        # when present — the tuned per-language pair must be shared by every consumer of the decode.
        dd = duration_decode_params(load_yaml(args.inference_config)) or {}
        train_records, _ = load_language_records(data_cfg, args.language, split="train")
        duration_prior = fit_duration_prior(train_records, **dd)
        if duration_prior is None: print("[segmenter-eval] WARNING: too few train captions to fit duration prior; plain decode", flush=True)
    print(f"[segmenter-eval] {args.segmenter_arch} segmenter from {checkpoint} (decode={'duration' if duration_prior else 'plain'})", flush=True)
    decode = "duration" if duration_prior else "plain"

    # Report at RQ2's tIoU grid so these standalone numbers line up 1:1 with the full-system segmentation block.
    thresholds = tuple(float(t) for t in (load_yaml(args.eval_config).get("rq2", {}) or {}).get("tiou_thresholds", [0.5]))
    metrics = evaluate_segmenter_whole_video(
        model, records, device=device, velocity=velocity, rope_chunk_s=rope_chunk_s, tiou_thresholds=thresholds, duration_prior=duration_prior,
    )
    payload = {
        "language": args.language, "split": args.split, "videos": len(records), "segmenter_arch": args.segmenter_arch, "checkpoint": checkpoint,
        "decode": decode, "tiou_thresholds": list(thresholds), "metrics": metrics,
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
    parser.add_argument(
        "--severity-grid-s", default=None,
        help="Comma-separated signed RQ1 grid (fractions of duration in relative mode, else seconds)"
    )
    # Per-axis overrides (fall back to --severity-grid-s / the eval.yaml grid). Truncation-only sweeps —
    # the valid sub-grid on pre-trimmed corpora — need head ≥ 0 with tail ≤ 0, which one shared grid's
    # cross-product cannot express: e.g. --severity-grid-head 0,0.1,0.2,0.3 --severity-grid-tail 0,-0.1,-0.2,-0.3
    parser.add_argument("--severity-grid-head", default=None,
                        help="RQ1 head-axis grid override (use --severity-grid-head=-0.3,0 for negative-leading values)")
    parser.add_argument("--severity-grid-tail", default=None,
                        help="RQ1 tail-axis grid override (use --severity-grid-tail=-0.3,0 for negative-leading values)")
    parser.add_argument("--severity-mode", default="relative", choices=["relative", "absolute"], 
                        help="RQ1 perturbation: relative (fraction of sentence duration, default) or absolute seconds")
    parser.add_argument("--emit-gold-segments", default=None, 
                        help="Write GT spans JSON (for --rq 2 --segments oracle-input rows) and exit")
    parser.add_argument("--segmenter-eval", action="store_true",
                        help="Standalone whole-video segmentation eval (Moryossef protocol) for --segmenter-arch, then exit")
    parser.add_argument(
        "--segmenter-decode", default=None, choices=["duration", "plain"],
        help="whole-video decode; default per arch: s1 -> duration (our semi-Markov re-split), external -> plain (faithful Moryossef argmax)"
    )
    parser.add_argument(
        "--segmenter-arch", default="external", choices=["external", "s1"],
        help="segmenter-eval backend: external = Moryossef analysis segmenter (default), s1 = in-system head"
    )
    parser.add_argument("--segmenter-config", default="configs/moryossef26.yaml", help="Moryossef segmenter config")
    parser.add_argument("--bio-config", default="configs/bio_pretrain.yaml", help="S1 (in-system head) config for --segmenter-arch s1")
    parser.add_argument("--num-sentences", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--predictions", default=None, help="RQ2 event JSON with start_s/end_s and optional text")
    parser.add_argument("--segments", default=None, help="RQ2 pipeline floor: segmenter spans JSON (analyze --stage segmenter-infer)")
    parser.add_argument("--stream", action="store_true", help="RQ2 row 6: run the streaming FSM engine to produce events")
    parser.add_argument(
        "--offline", action="store_true",
        help="Final misaligned model self-segments each whole video offline (its own BIO head) and translates each span"
    )
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
    data_cfg = load_yaml(args.data_config)
    if args.language is None: args.language = str(data_cfg.get("active_languages", ["csl"])[0])
    if args.emit_gold_segments:
        records, _ = load_language_records(data_cfg, args.language, split=args.split)
        path = write_gold_segments(records, args.emit_gold_segments)
        result = {"emit_gold_segments": str(path), "videos": len(records), "segments": sum(len(r.sentences) for r in records)}
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0)
    if args.segmenter_eval:
        print(json.dumps(run_segmenter_eval(args), indent=2, sort_keys=True))
        raise SystemExit(0)

    if args.rq == "1": result = run_rq1(args)
    elif args.rq == "2": result = run_rq2(args)
    else: result = run_segment_prf(args)
    display(result)
