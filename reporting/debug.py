"""Per-video debugging views: the pose stream a network is fed (`poses`) and one model's whole-video outputs
(`predict`). These are checks for one video, not results; `report.py results` is the results report."""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
import imageio.v2 as imageio
from data.windowing import BIO, make_bio_labels
from data.loader import load_language_records
from poses import UNISIGN_SELECTED_IDS, load_pose_window, normalize_keypoints_unisign
from metrics import (Segment, signing_runs_with_b_splits, sentence_bleu_scores, text_scores, text_level,
                     bio_frame_metrics, segmentation_prf, macro_frame_f1)
from evaluation.systems import (
    box_normalized_window, build_eval_model, commit_box_s, generation_kwargs, prep_window, METHOD_CONFIGS, TEXT_METHODS, model_annotation
)
from utils import FRAME_SLACK, cfg_get, load_yaml, output_dir, pick_device, target_language

# BIO row order for plotting (UNK at bottom .. I at top) and a stable colour per class.
_BIO_ROWS = ["UNK", "O", "B", "I"]
_BIO_COLOR = {"UNK": "#bdbdbd", "O": "#4c78a8", "B": "#e45756", "I": "#54a24b"}

# ── Skeleton topology over the 69 Uni-Sign keypoints (poses.normalize_keypoints_unisign layout) ─────────
# Tensor layout: body 0-8 (nose,lear,rear,l/r shoulder,l/r elbow,l/r wrist) | left hand 9-29 | right hand 30-50 |
# face_all 51-68 (scattered face landmarks — drawn as points, no chain).
NUM_VIZ_KPS = 69
_GROUP_RANGES = (("body", 0, 9), ("left", 9, 30), ("right", 30, 51), ("face_all", 51, 69))

def _hand_edges(base: int) -> list[tuple[int, int]]:
    edges = []  # COCO-WholeBody hand: 0 wrist, then thumb/index/middle/ring/pinky as 4-joint chains off the wrist.
    for finger_start in (1, 5, 9, 13, 17):
        edges.append((base + 0, base + finger_start))
        for j in range(finger_start, finger_start + 3):
            edges.append((base + j, base + j + 1))
    return edges

def _point_group(idx: int) -> str:
    for name, lo, hi in _GROUP_RANGES:
        if lo <= idx < hi: return name
    return "body"

_BODY_EDGES = [(0, 1), (0, 2), (3, 4), (3, 5), (5, 7), (4, 6), (6, 8)]
SKELETON_EDGES = _BODY_EDGES + _hand_edges(9) + _hand_edges(30)  # body + left/right hand (face_all = scattered points)
GROUP_COLOR = {"body": "#1f77b4", "left": "#d62728", "right": "#2ca02c", "face_all": "#ff7f0e"}
_POINT_COLORS = [GROUP_COLOR[_point_group(i)] for i in range(NUM_VIZ_KPS)]

def _draw_skeleton(ax, pts77: np.ndarray, conf77: np.ndarray, title: str) -> None:
    # Draw one frame. Only valid (conf>0) points/edges shown; y inverted (image convention). Edges/colors are the
    # 69-kp Uni-Sign layout, so on a different kp count (e.g. raw DWPose) out-of-range edges are skipped (points only).
    ax.clear()
    valid = conf77 > 0
    n = pts77.shape[0]
    color_of = lambda i: _POINT_COLORS[i] if i < len(_POINT_COLORS) else "#777777"
    for a, b in SKELETON_EDGES:
        if a < n and b < n and valid[a] and valid[b]: ax.plot(
            [pts77[a, 0], pts77[b, 0]], [pts77[a, 1], pts77[b, 1]],
            "-", color=color_of(a), lw=1.0, alpha=0.7
        )
    if valid.any(): ax.scatter(
        pts77[valid, 0], pts77[valid, 1], s=8,
        c=[color_of(i) for i in np.where(valid)[0]], zorder=3
    )
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # pixel/group coords are y-down
    ax.set_xticks([]); ax.set_yticks([])


# ── Shared timeline helpers (whole-video, seconds on x) ────────────────────────────────────────────────
def _draw_spans(ax, spans_s: list[tuple[float, float]], color: str, y0: float, y1: float, alpha: float = 0.30) -> None:
    for s0, s1 in spans_s:
        ax.axvspan(s0, s1, y0, y1, color=color, alpha=alpha, lw=0)

def _bio_track(ax, times_s: np.ndarray, tags: np.ndarray, title: str) -> None:
    # Step plot of a BIO tag sequence on a real-seconds x-axis (one coloured marker band per class row).
    ax.plot(times_s, tags, drawstyle="steps-post", color="#333333", lw=0.8, alpha=0.6)
    for name in _BIO_ROWS:
        sel = tags == BIO[name]
        if sel.any(): ax.scatter(times_s[sel], np.full(int(sel.sum()), BIO[name]), s=6, color=_BIO_COLOR[name], label=name)
    ax.set_yticks([BIO[n] for n in _BIO_ROWS]); ax.set_yticklabels(_BIO_ROWS)
    ax.set_ylim(-0.4, 3.4); ax.set_title(title, fontsize=9); ax.grid(alpha=0.25, axis="x")

def _segments_to_seconds(segs: list[dict], times_s: np.ndarray) -> list[tuple[float, float]]:
    # Frame-index runs ({start,end} inclusive) -> (start_s, end_s) using the per-frame timestamps.
    out = []
    n = len(times_s)
    for seg in segs:
        a, b = int(seg["start"]), min(int(seg["end"]), n - 1)
        if a < n: out.append((float(times_s[a]), float(times_s[b])))
    return out

def _load_whole_video(record) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Whole-video RAW (Uni-Sign-selected pixel) + NORMALIZED (unisign69) poses + per-frame timestamps (seconds).
    dur = float(record.pose.duration_s)
    raw, ts = load_pose_window(record.pose, 0.0, dur, normalize=False)
    norm, _ = load_pose_window(record.pose, 0.0, dur, normalize=True)
    if raw.shape[1:] != (133, 3): raise SystemExit(f"Expected raw (T,133,3), got {raw.shape}")
    return raw[:, UNISIGN_SELECTED_IDS, :], norm, ts


# ── Subcommand: poses (INPUTS, whole video) ────────────────────────────────────────────────────────────
def _render_skeleton_clip(raw77: np.ndarray, norm77: np.ndarray, out_stem: Path, fps: float, max_frames: int) -> dict:
    # Whole-video RAW vs NORMALIZED skeleton -> mp4 (or gif fallback). Capped at max_frames for file size.
    n = min(len(raw77), len(norm77))
    capped = n if max_frames <= 0 else min(n, max_frames)
    written: dict = {}
    try:
        frames = []
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
        for fi in range(capped):
            # Draw the normalized side only when it is the 69-kp Uni-Sign layout the skeleton edges expect.
            _draw_skeleton(axes[0], raw77[fi, :, :2], raw77[fi, :, 2], "raw (pixel)")
            if norm77.shape[1] == NUM_VIZ_KPS: _draw_skeleton(axes[1], norm77[fi, :, :2], norm77[fi, :, 2], "normalized")
            else: axes[1].clear(); axes[1].set_axis_off(); axes[1].set_title(f"normalized ({norm77.shape[1]}-kp repr)", fontsize=9)
            fig.suptitle(f"t={fi/max(fps,1e-6):.2f}s   frame {fi+1}/{n}", fontsize=9)
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
        try:
            mp4 = out_stem.with_name(out_stem.name + ".mp4")
            imageio.mimsave(mp4, frames, fps=max(1.0, fps)); written["video"] = str(mp4)
        except Exception:  # noqa: BLE001 - mp4 needs imageio-ffmpeg; gif always works (Pillow)
            gif = out_stem.with_name(out_stem.name + ".gif")
            imageio.mimsave(gif, frames, duration=1.0 / max(1.0, fps), loop=0); written["video"] = str(gif)
        if capped < n: written["video_note"] = f"clip capped at {capped}/{n} frames (--max-render-frames)"
    except Exception as exc: written["video_error"] = f"{type(exc).__name__}: {exc} (contact sheet still written)"
    return written


def cmd_poses(args: argparse.Namespace) -> dict:
    out = (Path(args.out) if args.out else output_dir("report")) / "poses"
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:  # Self-contained demo: a moving synthetic clip so rendering is verifiable with no dataset.
        rng = np.random.default_rng(0)
        T = int(args.frames)
        raw = np.zeros((T, 133, 3), dtype=np.float32)
        base = rng.uniform(400, 1500, size=(133, 2)).astype(np.float32)
        for t in range(T):
            raw[t, :, :2] = base + 20 * np.sin(t / 5.0 + np.arange(133)[:, None] * 0.1)
            raw[t, :, 2] = 0.9
        norm = normalize_keypoints_unisign(raw)
        res = _render_skeleton_clip(raw[:, UNISIGN_SELECTED_IDS, :], norm, out / "synthetic", float(args.fps), int(args.max_render_frames))
        return {"what": "poses", "synthetic": True, **res}

    records, _ = load_language_records(load_yaml(args.data_config), args.language, split=args.split)
    if not records: raise SystemExit("No records found for the requested language/split")
    record = next((r for r in records if args.video_id in (r.video_id, r.pose.video_id)), records[0])  # a source id picks its first stream

    raw77, norm, ts = _load_whole_video(record)
    fps = float(record.pose.fps)
    dur = float(record.pose.duration_s)
    spans_s = [(float(s.start_s), float(s.end_s)) for s in record.sentences]
    gt_bio = make_bio_labels(ts, record.sentences, 0.0, dur, video_duration_s=dur, untrusted=record.untrusted_s)

    stem = out / f"{record.video_id}_whole"
    written = _render_skeleton_clip(raw77, norm, stem, fps, int(args.max_render_frames))

    # Contact sheet sampled across the WHOLE video (raw row / normalized row).
    n = len(raw77)
    cols = min(10, max(1, n))
    idxs = np.linspace(0, n - 1, cols).round().astype(int)
    fig, axes = plt.subplots(2, cols, figsize=(1.9 * cols, 4.4), squeeze=False)
    for c, fi in enumerate(idxs):
        _draw_skeleton(axes[0][c], raw77[fi, :, :2], raw77[fi, :, 2], f"raw t={ts[fi]:.1f}s")
        if norm.shape[1] == NUM_VIZ_KPS: _draw_skeleton(axes[1][c], norm[fi, :, :2], norm[fi, :, 2], "norm")
        else: axes[1][c].set_axis_off()

    fig.suptitle(f"{record.video_id} — whole video ({n} frames @ {fps:.2f} fps, {dur:.1f}s)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    sheet = stem.with_name(stem.name + "_contact.png"); fig.savefig(sheet, dpi=110); plt.close(fig)
    written["contact_sheet"] = str(sheet)

    # Full-timeline strip: GT sentence spans + GT phrase-BIO track (what the BIO head is trained against).
    fig, (ax_sp, ax_bio) = plt.subplots(
        2, 1, figsize=(min(18, max(8, dur * 0.5)), 4.0), 
        sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )
    ax_sp.set_xlim(0, dur); ax_sp.set_ylim(0, 1); ax_sp.set_yticks([])
    _draw_spans(ax_sp, spans_s, "#54a24b", 0.15, 0.85)
    for s in record.sentences:
        ax_sp.text((s.start_s + s.end_s) / 2, 0.5, s.text[:22], fontsize=6, ha="center", va="center", rotation=0, clip_on=True)

    ax_sp.set_title(f"GT sentence spans ({len(record.sentences)} captions)", fontsize=9)
    _bio_track(ax_bio, ts, gt_bio, "GT phrase-BIO labels (UNK = no-loss / untrusted-O)")
    ax_bio.set_xlabel("seconds"); ax_bio.legend(fontsize=6, ncol=4, loc="upper right")
    fig.tight_layout()
    tl = stem.with_name(stem.name + "_timeline.png"); fig.savefig(tl, dpi=120); plt.close(fig)
    written["timeline"] = str(tl)
    return {"what": "poses", "video_id": record.video_id, "frames": int(n), "fps": fps,
            "duration_s": round(dur, 2), "sentences": len(record.sentences), **written}


def _bio_over_whole_video(model, raw_np: np.ndarray, ts_np: np.ndarray, device, chunk: int) -> np.ndarray:
    """Whole-video phrase-BIO logits (T, 4): raw (T,133,3) poses, each chunk pose-normalized on its own frames and stitched by the
    RoPE chunking's window rule (models.bio_head.chunk_normalized_logits), as every other whole-video S1 pass."""
    from models.bio_head import chunk_normalized_logits
    def _head_forward(p, m, t):
        return model.segment(p, m, t).logits
    return chunk_normalized_logits(_head_forward, raw_np, ts_np, int(chunk), device)[0].detach().cpu().numpy()


def _bio_tap_whole_video(model, poses_np, ts_np, device, chunk: int) -> np.ndarray:
    pieces = []
    n = len(poses_np)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        poses_t, ts_t, mask_t = prep_window(poses_np[a:b], ts_np[a:b], float(ts_np[a]))
        p, m = poses_t.unsqueeze(0).to(device), mask_t.unsqueeze(0).to(device)
        # The features the BIO head reads: the segmentation branch's own encoder in the two-stream arm, the shared encoder through the
        # stitch in the one-stream arm (MisalignedSLTModel.segment).
        seg_encoder, stitch = getattr(model, "bio_pose_encoder", None), getattr(model, "bio_stitch", None)
        bio_tap = seg_encoder(p, m) if seg_encoder is not None else model.front_end.extract_bio_tap(p, m, ts_t.unsqueeze(0).to(device))[0]
        if seg_encoder is None and stitch is not None: bio_tap = stitch(bio_tap)
        pieces.append(bio_tap[0, : b - a].detach().cpu())
    return torch.cat(pieces, dim=0).numpy()  # (T, d_model)


def _translate_sentence(model, tokenizer, method: str, poses_np, ts_np, start_s, fps, device, inference_cfg, method_cfg, anchor=None) -> tuple[str, np.ndarray]:
    # Decode ONE sentence window -> (text, PER-TOKEN confidence) for the dashboard's confidence strip. With the crop cmd_predict
    # cuts (an evaluation.offline.run_cascade row's crop and normalization) and evaluation.systems.generation_kwargs, the decode matches a cascade row
    # exactly (SPD params and all); the only reason this isn't evaluation.systems.translate_windows is that it collapses confidence to a mean.
    poses_t, ts_t, mask_t = prep_window(poses_np, ts_np, float(start_s))
    poses, ts, mask = poses_t.unsqueeze(0).to(device), ts_t.unsqueeze(0).to(device), mask_t.unsqueeze(0).to(device)
    gen = generation_kwargs(method, inference_cfg, method_cfg, [fps])   # the decode canvas and SPD params of evaluation/systems.py
    # As evaluation.offline.run_cascade: a joint arm's Ω reads the supplied span (window-relative (start, terminator) frames).
    if gen.get("gate_enabled") and anchor is not None: gen["gate_anchor"] = torch.tensor([list(anchor)], dtype=torch.long, device=device)
    _, tokens, conf, _ = model.generate_from_poses(poses=poses, frame_mask=mask, timestamps_s=ts, **gen)
    text = tokenizer.batch_decode(tokens.detach().cpu(), skip_special_tokens=True)[0].strip()
    row_tok = tokens[0].detach().cpu()
    row_conf = (conf[0] if conf.ndim == 2 else conf.reshape(-1)).detach().float().cpu()
    # Per-token confidence over the produced tokens, padding dropped (so the strip shows real tokens, not pads).
    keep = row_tok[: row_conf.shape[0]] != int(tokenizer.pad_token_id)
    return text, row_conf[keep[: row_conf.shape[0]]].numpy()


def cmd_predict(args: argparse.Namespace) -> dict:
    # --text as in eval.py: the pretrimmed translator (--pretrimmed-checkpoint, else data.yaml pretrained_slt) or an arm (--arm-checkpoint).
    method = TEXT_METHODS[args.text]
    data_cfg = load_yaml(args.data_config)
    # language= re-points ${language} paths (checkpoint.dir); without it the config's own `language:` key would
    # resolve another language's checkpoint and score it against this corpus.
    method_cfg = load_yaml(METHOD_CONFIGS[method], language=args.language)
    inference_cfg = load_yaml(args.inference_config)
    device = pick_device(args.device)
    model, tokenizer = build_eval_model(
        method, args.pretrimmed_checkpoint if method == "baseline" else args.arm_checkpoint, args.language, data_cfg, method_cfg,
        inference_cfg, device
    )
    labels = model_annotation(model, data_cfg, args.language)   # warns when the model trained on other train labels

    records, _ = load_language_records(data_cfg, args.language, split=args.split)
    # Declared, not sniffed from 1 video's captions: the corpus's scoring level, as every row reads it.
    char_level = text_level(target_language(data_cfg, args.language), cfg_get(data_cfg, "languages", args.language, "lowercase_references"))
    if not records: raise SystemExit("No records found for the requested language/split")
    record = next((r for r in records if args.video_id in (r.video_id, r.pose.video_id)), records[0])  # a source id picks its first stream
    dur = float(record.pose.duration_s)
    # The BIO pass chunks at the fixed cap the arm trained under (run_offline does the same).
    chunk = max(1, int(round(float(inference_cfg["buffer_cap_s"]) * float(record.pose.fps))))

    # Whole-video normalized poses + timestamps (what the network is fed).
    poses_np, ts_np = load_pose_window(record.pose, 0.0, dur, normalize=True)
    raw_np, _ = load_pose_window(record.pose, 0.0, dur, normalize=False)
    gt_bio = make_bio_labels(ts_np, record.sentences, 0.0, dur, video_duration_s=dur, untrusted=record.untrusted_s)
    has_bio = getattr(model, "bio_head", None) is not None   # the pretrimmed translator has no segmentation branch
    summary: dict = {"what": "predict", "video_id": record.video_id, "text": args.text,
                     "duration_s": round(dur, 2), "frames": int(len(poses_np)), "model_annotation": labels}

    # ── Per-sentence translation over the WHOLE video (every reliable caption, not one; a quarantined unit has no reference) ──
    sent_rows = []
    lead_s, tail_s = commit_box_s(inference_cfg) if method != "baseline" else (0., 0.)
    for s in record.sentences:
        if not s.reliable: continue
        # A gold span's cascade row (evaluation.offline.run_cascade): the crop [start, end + 1 frame]; the pretrimmed translator reads it
        # normalized alone, an arm under the body box of the buffer that would commit it, clipped to the stream.
        end_s = min(dur, float(s.end_s) + 1.0 / float(record.pose.fps))
        if method == "baseline": p_np, t_np = load_pose_window(record.pose, float(s.start_s), end_s, normalize=True)
        else: p_np, t_np = box_normalized_window(record.pose, float(s.start_s), end_s, max(0.0, float(s.start_s) - lead_s), min(dur, end_s + tail_s))
        if len(p_np) == 0: continue
        term = int(np.searchsorted(t_np, float(s.end_s) - FRAME_SLACK / float(record.pose.fps)))   # the terminator frame, -1 when the crop ends before it
        text, conf = _translate_sentence(
            model, tokenizer, method, p_np, t_np, float(s.start_s), float(record.pose.fps), device, inference_cfg, method_cfg,
            anchor=(0, term if term < len(p_np) else -1),
        )
        sent_rows.append({
            "start_s": float(s.start_s), "end_s": float(s.end_s), "ref": s.text, "pred": text, 
            "sentence_bleu4": round(sentence_bleu_scores([text], [s.text], char_level)[0], 2),
            "mean_conf": round(float(conf.mean()) if conf.size else 0.0, 3), "_conf": conf
        })
    summary["corpus_bleu4"] = round(text_scores([r["pred"] for r in sent_rows], [r["ref"] for r in sent_rows], char_level, keys=("bleu4",))["bleu4"], 2)
    summary["sentences"] = [{k: v for k, v in r.items() if k != "_conf"} for r in sent_rows]
    out = (Path(args.out) if args.out else output_dir("report")) / "predict"; out.mkdir(parents=True, exist_ok=True)

    # ── Dashboard figure ── (with a BIO head: metrics + 5 localization panels + translation; AR baseline: translation only)
    height_ratios = [1.0, 1.3, 1.0, 0.8, 0.9, 1.3, 2.2] if has_bio else [2.2]
    n_panels = len(height_ratios)
    fig = plt.figure(figsize=(min(20, max(10, dur * 0.6)), 2.0 * n_panels + 2))
    gs = fig.add_gridspec(n_panels, 1, height_ratios=height_ratios)
    row = 0
    if has_bio:
        bio_logits = _bio_over_whole_video(model, raw_np, ts_np, device, chunk)
        bio_pred = bio_logits.argmax(axis=-1)
        probs = torch.softmax(torch.as_tensor(bio_logits), dim=-1).numpy()
        bio_tap = _bio_tap_whole_video(model, poses_np, ts_np, device, chunk)

        # ── HONEST localization metrics (frame accuracy is inflated by class imbalance; segment tIoU is the
        # real signal). All computed whole-video against the GT spans, ignoring UNK (no-loss) frames. ──
        gl = torch.as_tensor(bio_logits).unsqueeze(0); gt = torch.as_tensor(gt_bio).unsqueeze(0)
        frame = bio_frame_metrics(gl, gt, prefix="bio")          # signing(B|I)-vs-O P/R/F1 + frame_acc (mask-based)
        valid = gt_bio != BIO["UNK"]
        macro_f1 = macro_frame_f1(torch.as_tensor(bio_pred[valid]), torch.as_tensor(gt_bio[valid]))  # O/B/I macro F1 over valid frames
        maj_class = int(np.bincount(gt_bio[valid], minlength=4).argmax()) if valid.any() else BIO["I"]
        maj_acc = float((gt_bio[valid] == maj_class).mean()) if valid.any() else 0.0
        # Score exactly like metrics.moryossef_segment_metrics: ONE decode rule (signing_runs_with_b_splits) on BOTH sides, and
        # predictions inside untrusted gaps (gold UNK) neutralized to UNK first — else a model correctly signing
        # through a real gap reads as a giant false-positive span (the asymmetry the monitor deliberately removes).
        pred_for_decode = np.where(gt_bio == BIO["UNK"], BIO["UNK"], bio_pred)
        pred_segs = _segments_to_seconds(signing_runs_with_b_splits(torch.as_tensor(pred_for_decode)), ts_np)
        gold_segs = _segments_to_seconds(signing_runs_with_b_splits(torch.as_tensor(gt_bio)), ts_np)
        p_objs = [Segment(a, b) for a, b in pred_segs]; g_objs = [Segment(a, b) for a, b in gold_segs]
        prf10 = segmentation_prf(p_objs, g_objs, 0.1); prf50 = segmentation_prf(p_objs, g_objs, 0.5)

        # B-boundary localization: each GT segment start vs the nearest predicted start.
        g_starts = [a for a, _ in gold_segs]; p_starts = [a for a, _ in pred_segs]
        offs = [min(abs(g - p) for p in p_starts) for g in g_starts] if (g_starts and p_starts) else []
        b_recall_04 = float(np.mean([o <= 0.4 for o in offs])) if offs else 0.0
        median_off = float(np.median(offs)) if offs else float("nan")
        summary["localization"] = {
            "frame_acc": round(frame["bio_frame_acc"], 3), "majority_class_frame_acc": round(maj_acc, 3),
            "macro_frame_f1_OBI": round(macro_f1, 3),
            "signing_precision": round(frame["bio_precision"], 3), "signing_recall": round(frame["bio_recall"], 3),
            "signing_f1": round(frame["bio_f1"], 3),
            "seg_tiou_f1@0.1": round(prf10["f1"], 3), "seg_tiou_f1@0.5": round(prf50["f1"], 3),
            "seg_precision@0.5": round(prf50["precision"], 3), "seg_recall@0.5": round(prf50["recall"], 3),
            "gt_segments": len(gold_segs), "pred_segments": len(pred_segs),
            "b_boundary_recall@0.4s": round(b_recall_04, 3),
            "median_boundary_offset_s": None if np.isnan(median_off) else round(median_off, 2),
        }
        # Panel 0: the honest metrics, with the caveat the eye needs to read "near-perfect" correctly.
        ax = fig.add_subplot(gs[row]); row += 1; ax.axis("off")
        gap = frame["bio_frame_acc"] - maj_acc
        verdict = ("frame acc ≈ majority baseline -> the high number is class imbalance, NOT localization; read segment tIoU"
                   if gap < 0.05 else "frame acc clears the majority baseline")
        loc_txt = (
            f"LOCALIZATION (whole video, UNK ignored)\n"
            f"  frame acc {frame['bio_frame_acc']:.1%} vs majority-class('{_BIO_ROWS[maj_class]}') baseline {maj_acc:.1%} (delta {gap:+.1%})\n"
            f"  macro O/B/I frame-F1 {macro_f1:.3f} signing P/R/F1 {frame['bio_precision']:.2f}/{frame['bio_recall']:.2f}/{frame['bio_f1']:.2f}\n"
            f"  segment tIoU-F1  @0.1={prf10['f1']:.2f} @0.5={prf50['f1']:.2f} (GT {len(gold_segs)} vs pred {len(pred_segs)} segments)\n"
            f"  B-boundary recall@0.4s {b_recall_04:.2f} median boundary offset {median_off:.2f}s\n-> {verdict}"
        )
        ax.text(0.0, 1.0, loc_txt, fontsize=8, family="monospace", va="top", ha="left", transform=ax.transAxes)

        ax = fig.add_subplot(gs[row]); row += 1
        _bio_track(ax, ts_np, bio_pred, f"Predicted phrase-BIO  (frame acc {frame['bio_frame_acc']:.1%}; red = wrong vs GT)")
        err = valid & (bio_pred != gt_bio)  # show WHERE it is wrong, not just the aggregate
        if err.any(): ax.scatter(ts_np[err], bio_pred[err], s=14, marker="x", color="red", zorder=5)
        ax.legend(fontsize=6, ncol=4, loc="upper right")

        ax = fig.add_subplot(gs[row]); row += 1   # GT BIO for direct comparison
        _bio_track(ax, ts_np, gt_bio, "GT phrase-BIO")

        ax = fig.add_subplot(gs[row]); row += 1   # class probabilities
        for name in ["O", "B", "I"]: ax.plot(ts_np, probs[:, BIO[name]], color=_BIO_COLOR[name], lw=1.0, label=f"P({name})")
        ax.set_ylim(0, 1); ax.set_title("BIO class probabilities", fontsize=9); ax.legend(fontsize=6, ncol=3); ax.grid(alpha=0.25)

        ax = fig.add_subplot(gs[row]); row += 1   # predicted vs GT segment spans + tIoU match lines
        ax.set_xlim(0, dur); ax.set_ylim(0, 1); ax.set_yticks([0.3, 0.7]); ax.set_yticklabels(["GT", "pred"], fontsize=7)
        _draw_spans(ax, gold_segs, "#54a24b", 0.05, 0.45); _draw_spans(ax, pred_segs, "#e45756", 0.55, 0.95)
        ax.set_title(f"Segments: GT={len(gold_segs)} vs pred={len(pred_segs)} | tIoU-F1 @0.1={prf10['f1']:.2f} @0.5={prf50['f1']:.2f}", fontsize=9)

        ax = fig.add_subplot(gs[row]); row += 1   # the per-frame features the BIO head reads (_bio_tap_whole_video)
        act = np.linalg.norm(bio_tap, axis=-1)
        dims = bio_tap[:, :: max(1, bio_tap.shape[1] // 64)][:, :64]  # 64 representative channels
        im = ax.imshow(dims.T, aspect="auto", cmap="magma", extent=[0, dur, 0, dims.shape[1]], origin="lower")
        ax.set_title(f"pose-token features the BIO head reads (64/{bio_tap.shape[1]} dims; mean |act| L2={act.mean():.2f})", fontsize=9)
        ax.set_ylabel("dim"); fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02)

    # ── Translation panel (always): per-sentence pred/ref + mean token confidence ──
    ax = fig.add_subplot(gs[row])
    ax.axis("off")
    header = f"{record.video_id} — {args.text} — corpus BLEU4 = {summary['corpus_bleu4']}"
    lines = [header, ""]
    for r in sent_rows[: int(args.max_sentences)]:
        lines.append(f"[{r['start_s']:5.1f}-{r['end_s']:5.1f}s] BLEU {r['sentence_bleu4']:5.1f} conf {r['mean_conf']:.2f}")
        lines.append(f"    REF : {r['ref'][:110]}")
        lines.append(f"    PRED: {r['pred'][:110]}")
    if len(sent_rows) > int(args.max_sentences): lines.append(f"... (+{len(sent_rows) - int(args.max_sentences)} more in JSON)")
    ax.text(0.0, 1.0, "\n".join(lines), fontsize=7, family="monospace", va="top", ha="left", transform=ax.transAxes)

    fig.suptitle(f"Whole-video outputs — {record.video_id} ({args.text})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    card = out / f"{record.video_id}_{args.text}_dashboard.png"; fig.savefig(card, dpi=120); plt.close(fig)
    summary["dashboard"] = str(card)

    # Separate per-token confidence strip (concatenated over sentences).
    if sent_rows:
        fig, ax = plt.subplots(figsize=(min(18, max(8, dur * 0.5)), 2.4))
        x = 0; ticks = []
        for r in sent_rows:
            c = r["_conf"]
            if c.size == 0: continue
            ax.bar(range(x, x + c.size), c, color="#4c78a8", width=1.0)
            ticks.append((x + c.size / 2, f"{r['start_s']:.0f}s"))
            x += c.size + 1
        ax.set_ylim(0, 1); ax.set_ylabel("token confidence"); ax.set_xlabel("output token (grouped by sentence)")
        if ticks: ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks], fontsize=6)
        ax.set_title("Per-token confidence over the whole video", fontsize=9)
        fig.tight_layout(); conf_png = out / f"{record.video_id}_{args.text}_confidence.png"
        fig.savefig(conf_png, dpi=120); plt.close(fig); summary["confidence_strip"] = str(conf_png)

    json_path = out / f"{record.video_id}_{args.text}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["json"] = str(json_path)
    return summary
