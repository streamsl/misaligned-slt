"""Misalignment-aware window sampler (spec §5). Turns GT sentence anchors into
real-timeline training windows across four modes whose mix is calibrated by
Analysis A's measured segmenter-error rates."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path

import json
import numpy as np
from data.jitter import JitterSampler, normalized_mode_ratios
from data.loader import VideoRecord
from data.windowing import (
    BIO, WindowSample, WindowSpec, classify_anchor_visibility,
    count_complete_spans, first_complete_span, make_bio_labels,
)
from poses import load_pose_window


class WindowSampler:
    """Emit one real-timeline training window per step (spec §5).

    Each step picks a GT sentence anchor and a mode (probabilities from Analysis A's measured segmenter-error rates), 
    then cuts a window on the *real* video timeline — neighbour content and gaps inside the jittered range are the actual 
    adjacent frames, never concatenated clips (avoids seam artifacts). The 4 modes mirror the inference-time buffer states:

    - **Mode 1** — anchor fully inside (jittered head/tail). OPUT target = anchor.
    - **Mode 2** — anchor truncated: `right` (no closing O, → confidence-bound),
      `left` (no B, → no translation loss, trains silence), `both` (interior, rare).
    - **Mode 3** — ≥2 complete sentences; target = earliest complete span (first-complete-span rule, identical at train and inference).
    - **Mode 4** — pure inter-sentence gap; BIO-only, trains the head to stay quiet.

    Truncation happens only here, by *window shaping* — never by relabelling text
    (premise P1). BIO labels come from GT boundaries; padding is masked, never `O`.
    """
    DEFAULT_MODE2_SUBCASE_WEIGHTS = {"right": 0.45, "left": 0.45, "both": 0.10}

    def __init__(
        self, records: list[VideoRecord], jitter: JitterSampler,
        mode_ratios: dict[str, float], buffer_cap_s: float,
        seed: int = 42, mode2_subcase_weights: dict[str, float] | None = None,
        fps_aug_enabled: bool = True, fps_aug_min: float = 25.0, fps_aug_max: float = 50.0,
    ):
        self.records = records
        self.jitter = jitter
        self.mode_ratios = normalized_mode_ratios(mode_ratios)
        self.buffer_cap_s = float(buffer_cap_s)
        # fps_aug is a Hard Rule (§1.4.4, Moryossef 2026: essential, 0.58→0.49 without). Applied to sampled training windows only — 
        # the Mode-2a full-evidence view stays at native fps so the no-grad self-target decode sees the same frame rate inference will.
        self.fps_aug_enabled = bool(fps_aug_enabled)
        self.fps_aug_min = float(fps_aug_min)
        self.fps_aug_max = float(fps_aug_max)
        self.rng = np.random.default_rng(seed)
        weights = dict(mode2_subcase_weights or self.DEFAULT_MODE2_SUBCASE_WEIGHTS)
        self._mode2_subcases = list(weights.keys())
        probs = np.asarray([float(weights[k]) for k in self._mode2_subcases], dtype=np.float64)
        self._mode2_subcase_probs = probs / probs.sum()
        self.anchors = [(ri, si) for ri, rec in enumerate(records) for si, _ in enumerate(rec.sentences)]
        if not self.anchors: raise ValueError("WindowSampler requires at least one sentence anchor.")

    @classmethod
    def from_stage2_config(cls, records: list[VideoRecord], stage2_cfg: dict, inference_cfg: dict) -> "WindowSampler":
        ratios_cfg = stage2_cfg.get("mode_ratios", {})
        source = ratios_cfg.get("source")
        if source and Path(source).exists():
            loaded = json.loads(Path(source).read_text(encoding="utf-8"))
            mode_ratios = loaded.get("mode_ratios", loaded)
        else: mode_ratios = ratios_cfg.get("fallback", {})

        fps_cfg = stage2_cfg.get("fps_aug", {})
        return cls(
            records=records, jitter=JitterSampler.from_config(stage2_cfg.get("jitter", {})),
            mode_ratios=mode_ratios, buffer_cap_s=float(inference_cfg.get("buffer_cap_s", 18.0)),
            seed=int(stage2_cfg.get("seed", 42)), mode2_subcase_weights=stage2_cfg.get("mode2_subcase_weights"),
            fps_aug_enabled=bool(fps_cfg.get("enabled", True)),
            fps_aug_min=float(fps_cfg.get("min_fps", 25.0)),
            fps_aug_max=float(fps_cfg.get("max_fps", 50.0)),
        )

    def _choose_mode(self) -> str:
        keys = list(self.mode_ratios.keys())
        probs = np.asarray([self.mode_ratios[k] for k in keys], dtype=np.float64)
        probs = probs / probs.sum()
        return str(self.rng.choice(keys, p=probs))

    def _choose_anchor(self) -> tuple[VideoRecord, int]:
        ridx, sidx = self.anchors[int(self.rng.integers(0, len(self.anchors)))]
        return self.records[ridx], sidx

    def _clip_window(self, rec: VideoRecord, start_s: float, end_s: float) -> tuple[float, float]:
        start_s = max(0.0, float(start_s))
        end_s = min(float(end_s), rec.pose.duration_s)
        if end_s - start_s > self.buffer_cap_s: end_s = start_s + self.buffer_cap_s
        if end_s <= start_s: end_s = min(rec.pose.duration_s, start_s + 1.0 / rec.pose.fps)
        return start_s, end_s


    def _mode1_spec(self, rec: VideoRecord, anchor_idx: int) -> WindowSpec: # Complete-anchor window
        # Jitter both edges, retry until anchor's B and closing O are both inside; fall back to a clean clip if jitter never fits.
        anchor = rec.sentences[anchor_idx]
        for _ in range(20):
            dh, dt = self.jitter.sample(self.rng)
            # Analysis A stores signed offsets as pred_boundary - gt_boundary.
            start_s, end_s = self._clip_window(rec, anchor.start_s + dh, anchor.end_s + dt)
            if classify_anchor_visibility(anchor, start_s, end_s) == "complete":
                return WindowSpec(rec.video_id, start_s, end_s, "mode1", anchor_idx)
        return WindowSpec(rec.video_id, *self._clip_window(rec, anchor.start_s, anchor.end_s + 1.0 / rec.pose.fps), "mode1", anchor_idx)


    def _mode2_spec(self, rec: VideoRecord, anchor_idx: int) -> WindowSpec: # Truncated-anchor window
        """`right` keeps the start, cuts before the end (B, no closing O); `left` cuts after the start, keeps 
        the end (no B); `both` is a strictly-interior slice (all I). Cut points are drawn on the timeline."""
        anchor = rec.sentences[anchor_idx]
        subcase = str(self.rng.choice(self._mode2_subcases, p=self._mode2_subcase_probs))
        eps = max(1.0 / rec.pose.fps, 1e-3)
        if subcase == "right":
            cut = float(self.rng.uniform(anchor.start_s + eps, max(anchor.start_s + eps, anchor.end_s - eps)))
            start_s, end_s = self._clip_window(rec, max(0.0, anchor.start_s - eps), cut)
        elif subcase == "left":
            cut = float(self.rng.uniform(anchor.start_s + eps, max(anchor.start_s + eps, anchor.end_s - eps)))
            start_s, end_s = self._clip_window(rec, cut, min(rec.pose.duration_s, anchor.end_s + eps))
        else:
            if anchor.duration_s <= 3 * eps:
                subcase = "right"
                start_s, end_s = self._clip_window(rec, anchor.start_s, anchor.start_s + eps)
            else:
                s = float(self.rng.uniform(anchor.start_s + eps, anchor.end_s - 2 * eps))
                e = float(self.rng.uniform(s + eps, anchor.end_s - eps))
                start_s, end_s = self._clip_window(rec, s, e)
        return WindowSpec(rec.video_id, start_s, end_s, "mode2", anchor_idx, subcase)  # type: ignore[arg-type]


    def _mode3_spec(self, rec: VideoRecord, anchor_idx: int) -> WindowSpec: # Multi-complete window
        """Span the anchor and its successor so ≥2 sentences are fully inside; degrade to Mode 1 if the pair 
        does not yield 2 complete spans. The translation target is later chosen by `first_complete_span`."""
        anchor = rec.sentences[anchor_idx]
        next_idx = min(anchor_idx + 1, len(rec.sentences) - 1)
        end_anchor = rec.sentences[next_idx]
        start_s, end_s = self._clip_window(rec, anchor.start_s, end_anchor.end_s + 1.0 / rec.pose.fps)
        if count_complete_spans(rec.sentences, start_s, end_s) < 2: return self._mode1_spec(rec, anchor_idx)
        return WindowSpec(rec.video_id, start_s, end_s, "mode3", anchor_idx)


    def _mode4_spec(self, rec: VideoRecord) -> WindowSpec: # All-gap window
        # Sample inside an inter-sentence gap (≥0.5s of no signing); falls back to a Mode-2 window if the video has no usable gap.
        gaps: list[tuple[float, float]] = []
        prev = 0.0
        for span in rec.sentences:
            if span.start_s > prev: gaps.append((prev, span.start_s))
            prev = max(prev, span.end_s)

        if prev < rec.pose.duration_s: gaps.append((prev, rec.pose.duration_s))
        gaps = [(s, e) for s, e in gaps if e - s >= 0.5]
        if not gaps:
            idx = int(self.rng.integers(0, len(rec.sentences)))
            return self._mode2_spec(rec, idx)

        gap = gaps[int(self.rng.integers(0, len(gaps)))]
        length = min(self.buffer_cap_s, max(0.5, gap[1] - gap[0]))
        start_s = float(self.rng.uniform(gap[0], max(gap[0], gap[1] - length)))
        end_s = min(gap[1], start_s + length)
        return WindowSpec(rec.video_id, start_s, end_s, "mode4")


    def sample(self) -> WindowSample:
        rec, anchor_idx = self._choose_anchor()
        mode = self._choose_mode()
        if mode == "mode1": spec = self._mode1_spec(rec, anchor_idx)
        elif mode == "mode2": spec = self._mode2_spec(rec, anchor_idx)
        elif mode == "mode3": spec = self._mode3_spec(rec, anchor_idx)
        elif mode == "mode4": spec = self._mode4_spec(rec)
        else: raise ValueError(f"Unknown mode {mode}")
        return self.materialize(rec, spec, fps_aug=self.fps_aug_enabled)


    def materialize(self, rec: VideoRecord, spec: WindowSpec, fps_aug: bool = False) -> WindowSample:
        """Realize a `WindowSpec` into tensors: load+normalize the pose window, build per-frame BIO labels from GT boundaries,
        pick the translation target (Mode 1/3 first-complete-span), and for Mode-2a attach the Mode-1-equivalent
        `full_evidence_spec` the confidence-bound term decodes under no_grad. `fps_aug=True` resamples the window to a
        random fps (Moryossef 2026 fps_aug; Hard Rule §1.4.4) and rebuilds the BIO labels from the resampled timestamps."""
        poses, timestamps = load_pose_window(rec.pose, spec.start_s, spec.end_s, normalize=True)
        if fps_aug and poses.shape[0] > 1:
            from moryossef26.dataset import apply_fps_aug
            poses, rel_timestamps, _ = apply_fps_aug(
                poses, source_fps=rec.pose.fps,
                min_fps=self.fps_aug_min, max_fps=self.fps_aug_max, rng=self.rng,
            )
            timestamps = spec.start_s + rel_timestamps

        frame_mask = np.ones((poses.shape[0],), dtype=bool)
        labels = make_bio_labels(timestamps, rec.sentences, spec.start_s, spec.end_s, frame_mask)
        anchor_span = rec.sentences[spec.anchor_index] if spec.anchor_index is not None else None
        target, full_evidence_spec = None, None

        if spec.mode in {"mode1", "mode3"}: target = first_complete_span(rec.sentences, spec.start_s, spec.end_s, 1.0 / rec.pose.fps)
        elif spec.mode == "mode2" and spec.subcase == "right" and spec.anchor_index is not None:
            target = None
            full_evidence_spec = self._mode1_spec(rec, spec.anchor_index)
        return WindowSample(
            spec=spec, poses=poses, timestamps_s=timestamps - spec.start_s,
            bio_labels=labels, frame_mask=frame_mask, spans=rec.sentences,
            translation_target=target, anchor_span=anchor_span, full_evidence_spec=full_evidence_spec,
        )

    @staticmethod
    def to_dict(sample: WindowSample) -> dict:
        return {
            "spec": asdict(sample.spec), "poses": sample.poses, "timestamps_s": sample.timestamps_s,
            "bio_labels": sample.bio_labels, "frame_mask": sample.frame_mask,
            "translation_target": asdict(sample.translation_target) if sample.translation_target else None,
            "anchor_span": asdict(sample.anchor_span) if sample.anchor_span else None,
            "full_evidence_spec": asdict(sample.full_evidence_spec) if sample.full_evidence_spec else None,
        }
