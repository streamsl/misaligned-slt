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
    BIO, TRUSTED_GAP_S, WindowSample, WindowSpec, classify_anchor_visibility,
    count_complete_spans, first_complete_span, make_bio_labels,
)
from poses import load_pose_window, build_pose_augmentor


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
        pose_augment_cfg: dict | None = None,
    ):
        self.records = records
        self.jitter = jitter
        self.mode_ratios = normalized_mode_ratios(mode_ratios)
        self.buffer_cap_s = float(buffer_cap_s)
        # Spatial pose augmentation (train only), with its own rng so it does not perturb the window-sampling
        # rng stream. None on dev (deterministic monitor) and never applied to the Mode-2a full-evidence view.
        self.pose_augmentor = build_pose_augmentor(pose_augment_cfg, np.random.default_rng(int(seed) + 997))

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

        # Anchor windowing (spec §5.0): every GT sentence is drawn once per pass through the permutation,
        # so with steps_per_epoch == len(anchors) each sentence anchors a window each epoch — random
        # sampling with replacement undersamples short sentences and wastes data. The mode is still drawn
        # independently per step. (Requires num_workers=0; each worker would otherwise hold its own cursor.)
        self._anchor_order = self.rng.permutation(len(self.anchors))
        self._anchor_cursor = 0


    @classmethod
    def from_stage2_config(
        cls, records: list[VideoRecord], stage2_cfg: dict, inference_cfg: dict,
        pose_augment_cfg: dict | None = None,
    ) -> "WindowSampler":
        ratios_cfg = stage2_cfg.get("mode_ratios", {})
        fallback_ratios = ratios_cfg.get("fallback", {})
        source = ratios_cfg.get("source")
        measured = None
        if source and Path(source).exists():
            loaded = json.loads(Path(source).read_text(encoding="utf-8"))
            measured = loaded.get("mode_ratios", loaded)

        jitter_cfg = dict(stage2_cfg.get("jitter", {}))
        mode_ratios = measured if measured is not None else fallback_ratios
        # Degenerate-measurement guard. On a clean corpus (e.g. PHOENIX) the retrained segmenter is
        # near-perfect, so Analysis A measures almost all Mode 1 (mode1~0.98) and a ~0-offset jitter CDF.
        # Training on that = no misalignment = the robustness method learns nothing — here faithfulness to
        # the spec's "derive ratios from Analysis A" becomes a bug, because that rule assumed a NONTRIVIAL
        # measured error distribution. When the measurement is degenerate we fall back to the DESIGNED
        # distribution (fallback ratios + fallback_laplace jitter) and warn; robustness is then evaluated
        # by the segmenter-agnostic RQ1 controlled sweep, which never depended on a noisy segmenter.
        threshold = float(ratios_cfg.get("degenerate_mode1_threshold", 0.9))
        if measured is not None and normalized_mode_ratios(measured).get("mode1", 0.0) >= threshold:
            print(
                f"[sampler] WARNING: measured Analysis-A mode ratios are degenerate "
                f"(mode1={normalized_mode_ratios(measured).get('mode1', 0.0):.3f} >= {threshold}); the "
                f"segmenter is too clean to yield a useful misalignment distribution. Using the DESIGNED "
                f"fallback ratios + jitter (configs/stage2_dlm.yaml: mode_ratios.fallback, jitter."
                f"fallback_laplace). Robustness is evaluated via the controlled RQ1 sweep.", flush=True,
            )
            mode_ratios = fallback_ratios
            jitter_cfg["source"] = None  # force the designed fallback_laplace jitter, not the ~0 measured CDF

        fps_cfg = (stage2_cfg.get("augmentation", {}) or {}).get("fps", {})
        return cls(
            records=records, jitter=JitterSampler.from_config(jitter_cfg),
            mode_ratios=mode_ratios, buffer_cap_s=float(inference_cfg.get("buffer_cap_s", 18.0)),
            seed=int(stage2_cfg.get("seed", 42)), mode2_subcase_weights=stage2_cfg.get("mode2_subcase_weights"),
            fps_aug_enabled=bool(fps_cfg.get("enabled", True)),
            fps_aug_min=float(fps_cfg.get("min_fps", 25.0)),
            fps_aug_max=float(fps_cfg.get("max_fps", 50.0)),
            pose_augment_cfg=pose_augment_cfg,
        )

    def _choose_mode(self) -> str:
        keys = list(self.mode_ratios.keys())
        probs = np.asarray([self.mode_ratios[k] for k in keys], dtype=np.float64)
        probs = probs / probs.sum()
        return str(self.rng.choice(keys, p=probs))

    def _choose_anchor(self) -> tuple[VideoRecord, int]:
        if self._anchor_cursor >= len(self._anchor_order):
            self._anchor_order = self.rng.permutation(len(self.anchors))
            self._anchor_cursor = 0
        ridx, sidx = self.anchors[int(self._anchor_order[self._anchor_cursor])]
        self._anchor_cursor += 1
        return self.records[ridx], sidx

    def _clip_window(self, rec: VideoRecord, start_s: float, end_s: float) -> tuple[float, float]:
        start_s = max(0.0, float(start_s))
        end_s = min(float(end_s), rec.pose.duration_s)
        if end_s - start_s > self.buffer_cap_s: end_s = start_s + self.buffer_cap_s
        if end_s <= start_s: end_s = min(rec.pose.duration_s, start_s + 1.0 / rec.pose.fps)
        return start_s, end_s

    def _cut_time(self, anchor, lo: float = 0.05, hi: float = 0.95) -> float:
        # Absolute time of a spurious internal cut, from Analysis A's over-seg cut-position distribution
        # (JitterSampler.sample_cut; uniform fallback). Clamped away from the exact edges.
        rel = min(max(self.jitter.sample_cut(self.rng), lo), hi)
        return float(anchor.start_s + rel * anchor.duration_s)


    def _mode1_spec(self, rec: VideoRecord, anchor_idx: int) -> WindowSpec: # Complete-anchor window
        # Jitter both edges, retry until anchor's B and closing O are both inside; fall back to a clean clip if jitter never fits.
        anchor = rec.sentences[anchor_idx]
        eps = 1.0 / rec.pose.fps
        for _ in range(20):
            dh, dt = self.jitter.sample(self.rng)
            # Analysis A stores signed offsets as pred_boundary - gt_boundary.
            start_s, end_s = self._clip_window(rec, anchor.start_s + dh, anchor.end_s + dt)
            # The end check mirrors first_complete_span(min_o_after_s=1/fps) in materialize(); a looser check here
            # would classify the window complete yet yield no translation target (silently unsupervised Mode 1).
            if classify_anchor_visibility(anchor, start_s, end_s) == "complete" and anchor.end_s + eps <= end_s:
                return WindowSpec(rec.video_id, start_s, end_s, "mode1", anchor_idx)
        return WindowSpec(rec.video_id, *self._clip_window(rec, anchor.start_s, anchor.end_s + eps), "mode1", anchor_idx)


    def _full_evidence_spec(self, rec: VideoRecord, anchor_idx: int) -> WindowSpec:
        """Mode-1-equivalent window for the §6.3 full-evidence decode, with 1 extra constraint: the anchor
        must be the window's FIRST complete span. The full-evidence decode has no explicit target — the model
        (trained on the first-complete-span rule, §5.3) translates the earliest complete sentence in its
        conditioning. A plain `_mode1_spec` window whose head jitter pulls in a complete earlier neighbour
        would therefore yield y_full for the *neighbour*, not the anchor the truncated view shows: the verified
        gate (f==r) then never fires (dead CB batch), and the unverified ablation would supervise toward the
        wrong sentence's text. Falls back to the clean anchor clip, where anchor-first is guaranteed (an
        earlier sentence cannot have its B inside a window that starts at the anchor's start)."""
        anchor = rec.sentences[anchor_idx]
        eps = 1.0 / rec.pose.fps
        for _ in range(20):
            dh, dt = self.jitter.sample(self.rng)
            start_s, end_s = self._clip_window(rec, anchor.start_s + dh, anchor.end_s + dt)
            if (classify_anchor_visibility(anchor, start_s, end_s) == "complete" and anchor.end_s + eps <= end_s
                and first_complete_span(rec.sentences, start_s, end_s, eps) is anchor):
                return WindowSpec(rec.video_id, start_s, end_s, "mode1", anchor_idx)
        return WindowSpec(rec.video_id, *self._clip_window(rec, anchor.start_s, anchor.end_s + eps), "mode1", anchor_idx)


    def _mode2_spec(self, rec: VideoRecord, anchor_idx: int) -> WindowSpec: # Truncated-anchor window
        """`right` keeps the start, cuts before the end (B, no closing O); `left` cuts after the start, keeps
        the end (no B); `both` is a strictly-interior slice (all I).

        The truncation depth — where the window cuts *inside* the anchor — is drawn from Analysis A's measured
        over-segmentation cut positions (`JitterSampler.sample_cut`), the empirical answer to "where does the
        segmenter split a sentence". The *surviving* outer edge carries ordinary boundary jitter (Δ_head/Δ_tail),
        so e.g. a right-truncated window's true start still wobbles like a real Started-Pre/Post-Signing event.
        Uniform interior cut is used only when no over-seg was measured (§5.0/§5.2; Hard Rule §1.4.5)."""
        anchor = rec.sentences[anchor_idx]
        subcase = str(self.rng.choice(self._mode2_subcases, p=self._mode2_subcase_probs))
        eps = max(1.0 / rec.pose.fps, 1e-3)
        dh, dt = self.jitter.sample(self.rng)

        if subcase == "both" and anchor.duration_s > 3 * eps:
            a, b = sorted((self._cut_time(anchor), self._cut_time(anchor)))
            start_s, end_s = self._clip_window(rec, max(anchor.start_s + eps, a), min(anchor.end_s - eps, max(a + eps, b)))
            if classify_anchor_visibility(anchor, start_s, end_s) == "both":
                return WindowSpec(rec.video_id, start_s, end_s, "mode2", anchor_idx, "both")
            subcase = "right"  # degenerate interior slice → fall through to right-trunc

        if subcase == "left":
            # Keep the true end, discard the head: window starts at the spurious cut. Tail jitter may only
            # push the end OUTWARD (max(dt,0)) so the closing O stays inside — otherwise the GT end leaves
            # the window and the labels no longer describe a left-truncation (P2: labels follow the window).
            cut = min(self._cut_time(anchor), anchor.end_s - eps)
            # End must sit strictly past the anchor end (classify_anchor_visibility uses end_s < window_end),
            # so the closing O is inside; tail jitter only extends it further out.
            start_s, end_s = self._clip_window(rec, cut, min(rec.pose.duration_s, anchor.end_s + max(dt, eps)))
        else:  # "right": keep the true start, cut before the end. Head jitter only pulls the start outward.
            cut = max(self._cut_time(anchor), anchor.start_s + eps)
            start_s, end_s = self._clip_window(rec, max(0.0, anchor.start_s + min(dh, 0.0)), cut)
        return WindowSpec(rec.video_id, start_s, end_s, "mode2", anchor_idx, subcase)  # type: ignore[arg-type]


    def _mode3_spec(self, rec: VideoRecord, anchor_idx: int) -> WindowSpec: # Multi-complete window
        """Span the anchor and its successor so ≥2 sentences are fully inside; degrade to Mode 1 if the pair
        does not yield 2 complete spans. The translation target is later chosen by `first_complete_span`.
        Edges carry Analysis-A jitter like every other mode — an exact [anchor B, successor end] window would
        train a boundary distribution (B at frame 0) the streaming buffer never produces."""
        anchor = rec.sentences[anchor_idx]
        next_idx = min(anchor_idx + 1, len(rec.sentences) - 1)
        end_anchor = rec.sentences[next_idx]
        eps = 1.0 / rec.pose.fps

        for _ in range(20):
            dh, dt = self.jitter.sample(self.rng)
            start_s, end_s = self._clip_window(rec, anchor.start_s + dh, end_anchor.end_s + dt)
            if count_complete_spans(rec.sentences, start_s, end_s, eps) >= 2:
                return WindowSpec(rec.video_id, start_s, end_s, "mode3", anchor_idx)

        start_s, end_s = self._clip_window(rec, anchor.start_s, end_anchor.end_s + eps)
        if count_complete_spans(rec.sentences, start_s, end_s, eps) < 2: return self._mode1_spec(rec, anchor_idx)
        return WindowSpec(rec.video_id, start_s, end_s, "mode3", anchor_idx)


    def _mode4_spec(self, rec: VideoRecord) -> WindowSpec: # All-gap window
        # Sample inside an inter-sentence gap (≥0.5s of no signing); falls back to a Mode-2 window if the video has no usable gap.
        gaps: list[tuple[float, float]] = []
        prev = 0.0
        for span in rec.sentences:
            if span.start_s > prev: gaps.append((prev, span.start_s))
            prev = max(prev, span.end_s)

        if prev < rec.pose.duration_s: gaps.append((prev, rec.pose.duration_s))
        # Trusted gaps only: long uncaptioned stretches (> TRUSTED_GAP_S, mostly intros/outros)
        # may contain uncaptioned signing — an "all-gap" window there could be all-signing, the
        # exact opposite of what Mode 4 trains (stay quiet on non-signing input).
        gaps = [(s, e) for s, e in gaps if 0.5 <= e - s <= TRUSTED_GAP_S]
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
        # Augment the sampled training window; the Mode-2a full-evidence view (materialized separately
        # with defaults) stays un-augmented so the no-grad self-target matches inference conditions.
        return self.materialize(rec, spec, fps_aug=self.fps_aug_enabled, augment=self.pose_augmentor is not None)


    def materialize(self, rec: VideoRecord, spec: WindowSpec, fps_aug: bool = False, augment: bool = False) -> WindowSample:
        """Realize a `WindowSpec` into tensors: load+normalize the pose window, build per-frame BIO labels from GT boundaries,
        pick the translation target (Mode 1/3 first-complete-span), and for Mode-2a attach the Mode-1-equivalent
        `full_evidence_spec` the confidence-bound term decodes under no_grad. `fps_aug=True` resamples the window to a
        random fps (Moryossef 2026 fps_aug; Hard Rule §1.4.4) and rebuilds the BIO labels from the resampled timestamps."""
        poses, timestamps = load_pose_window(
            rec.pose, spec.start_s, spec.end_s, normalize=True,
            augment=self.pose_augmentor if augment else None
        )
        if fps_aug and poses.shape[0] > 1:
            from moryossef26.dataset import apply_fps_aug
            poses, timestamps, _ = apply_fps_aug(
                poses, source_fps=rec.pose.fps,
                min_fps=self.fps_aug_min, max_fps=self.fps_aug_max, rng=self.rng,
                source_timestamps_s=timestamps,
            )

        frame_mask = np.ones((poses.shape[0],), dtype=bool)
        labels = make_bio_labels(
            timestamps, rec.sentences, spec.start_s, spec.end_s, frame_mask,
            video_duration_s=rec.pose.duration_s,  # long uncaptioned stretches -> UNK (see make_bio_labels)
        )
        anchor_span = rec.sentences[spec.anchor_index] if spec.anchor_index is not None else None
        target, full_evidence_spec = None, None

        if spec.mode in {"mode1", "mode3"}: target = first_complete_span(rec.sentences, spec.start_s, spec.end_s, 1.0 / rec.pose.fps)
        elif spec.mode == "mode2" and spec.subcase == "right" and spec.anchor_index is not None:
            target = None
            full_evidence_spec = self._full_evidence_spec(rec, spec.anchor_index)
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
