"""Misalignment-aware window sampler over four real-timeline window modes."""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path

import json
import numpy as np
from data.jitter import JitterSampler, normalized_mode_ratios
from data.loader import VideoRecord
from utils import pool_key
from data.windowing import (
    TRUSTED_GAP_S, WindowSample, WindowSpec, classify_anchor_visibility,
    count_complete_spans, first_complete_span, make_bio_labels,
)
from poses import load_pose_window, build_pose_augmentor, apply_fps_aug


class WindowSampler:
    """Emit one real-timeline training window per step.

    Each step picks a GT sentence anchor and a mode from the configured coverage mix,
    then cuts a window on the *real* video timeline — neighbour content and gaps inside the jittered range are the actual 
    adjacent frames, never concatenated clips (avoids seam artifacts). The 4 modes mirror the inference-time buffer states:

    - **Mode 1** — anchor fully inside (jittered head/tail). OPUT target = anchor.
    - **Mode 2** — anchor truncated: `right` (no terminator — the anchor's I-run reaches the window edge, → confidence-bound),
      `left` (no B, no translation loss), `both` (interior, rare).
    - **Mode 3** — ≥2 complete sentences; target = earliest complete span (first-complete-span rule, identical at train and inference).
    - **Mode 4** — pure inter-sentence gap; BIO-only, trains the head to stay quiet.

    Truncation happens only here, by *window shaping* — never by relabelling text (premise P1). 
    BIO labels come from GT boundaries; padding is masked, never `O`.
    """
    DEFAULT_MODE2_SUBCASE_WEIGHTS = {"right": 0.45, "left": 0.45, "both": 0.10}

    def __init__(
        self, records: list[VideoRecord], jitter: JitterSampler, mode_ratios: dict[str, float], buffer_cap_s: float,
        mode2_subcase_weights: dict[str, float] | None = None, mode3_span_counts: dict[int, float] | None = None,
        fps_aug_enabled: bool = True, fps_aug_min: float = 15.0, fps_aug_max: float = 30.0,
        pose_augment_cfg: dict | None = None, min_span_frames: int = 0, seed: int = 42
    ):
        self.records = records
        self.jitter = jitter
        self.mode_ratios = normalized_mode_ratios(mode_ratios)
        self.buffer_cap_s = float(buffer_cap_s)
        # Λ_min mirror of the commit gate's span_selection.min_span_frames: the training target/complete-span
        # rules must skip sub-Λ_min sentences exactly like the deployed select_target_span, or a window whose
        # earliest complete span is sub-Λ_min supervises a sentence the gate never anchors (wrong-sentence Ω).
        self.min_span_frames = int(min_span_frames)
        # Spatial pose augmentation (train only), with its own rng so it does not perturb the window-sampling
        # rng stream. None on dev (deterministic monitor) and never applied to the Mode-2a full-evidence view.
        self.pose_augmentor = build_pose_augmentor(pose_augment_cfg, np.random.default_rng(int(seed) + 997))

        # fps_aug is a Hard Rule (Moryossef 2026: essential, 0.58→0.49 without). Applied to sampled training windows only — 
        # the Mode-2a full-evidence view stays at native fps so the no-grad self-target decode sees the same frame rate inference will.
        self.fps_aug_enabled = bool(fps_aug_enabled)
        self.fps_aug_min = float(fps_aug_min)
        self.fps_aug_max = float(fps_aug_max)
        self.rng = np.random.default_rng(seed)
        # Base for index-seeded per-window draws (spec_for). Workers offset it, so parallel workers stay
        # decorrelated while each index still maps to ONE window within a given worker layout.
        self.draw_seed = int(seed)
        weights = dict(mode2_subcase_weights or self.DEFAULT_MODE2_SUBCASE_WEIGHTS)
        # _mode2_spec dispatches by string compare with a bare else -> a typo'd key would silently sample as 'right'.
        unknown = set(weights) - {"right", "left", "both"}
        if unknown: raise ValueError(f"mode2_subcase_weights has unknown keys {sorted(unknown)}; valid: right/left/both")
        self._mode2_subcases = list(weights.keys())

        probs = np.asarray([float(weights[k]) for k in self._mode2_subcases], dtype=np.float64)
        self._mode2_subcase_probs = probs / probs.sum()
        # Mode-3 span-count distribution {k: weight}: how many COMPLETE sentences a mode-3 window spans. FSM reaches k>=3 buffers (a commit 
        # retires 1 sentence per >=hysteresis strides while the buffer grows every stride), so a hard-wired 2 leaves that regime untrained.
        counts = {int(k): float(v) for k, v in (mode3_span_counts or {2: 1.0}).items()} # Default {2: 1.0}
        if any(k < 2 for k in counts): 
            raise ValueError(f"mode3_span_counts keys must be >= 2 (k=1 is mode1's job; labelling it mode3 corrupts the mix): {sorted(counts)}")
        
        cprobs = np.asarray([counts[k] for k in sorted(counts)], dtype=np.float64)
        if cprobs.sum() <= 0: raise ValueError(f"mode3_span_counts weights must sum > 0: {counts}")
        self._mode3_counts = sorted(counts)
        self._mode3_count_probs = cprobs / cprobs.sum()
        # Quarantined spans (reliable=False) occupy time for gap/window purposes but are never anchors: no correct supervision exists for them.
        self.anchors = [(ri, si) for ri, rec in enumerate(records) for si, sp in enumerate(rec.sentences) if getattr(sp, "reliable", True)]
        if not self.anchors: raise ValueError("WindowSampler requires at least one sentence anchor.")


    def configure_worker(self, seed: int) -> None:
        """Give a forked DataLoader worker its OWN rng so parallel workers don't replay identical mode/jitter streams.

        fork/spawn copies the sampler's numpy Generator(s) identically into every worker, and PyTorch's per-worker seeding never touches 
        a Generator stored on the dataset — so without this, W workers draw the SAME mode/jitter sequence. (The anchor is index-driven — 
        anchors[index % len] in `sample` — so each worker already visits DIFFERENT anchors via its round-robin index slice, and coverage 
        is exactly one pass per epoch regardless of how the DataLoader partitions indices across workers. This reseed only decorrelates 
        the per-window draws.) Called from data.loader.streaming_loader's worker_init_fn.
        """
        self.rng = np.random.default_rng(int(seed))
        # draw_seed is deliberately NOT reseeded: `spec_for` derives its rng from draw_seed + INDEX, so windows are
        # already decorrelated across indices without per-worker offsets — and a worker-varying base would make a
        # worker realise a different window than the length pre-pass predicted, silently disabling bucketing.
        if self.pose_augmentor is not None: self.pose_augmentor.rng = np.random.default_rng(int(seed) + 997)


    @classmethod
    def from_slt_config(
        cls, records: list[VideoRecord], slt_cfg: dict, inference_cfg: dict, pose_augment_cfg: dict | None = None,
    ) -> "WindowSampler":
        ratios_cfg = slt_cfg.get("mode_ratios", {})
        fallback_ratios = ratios_cfg.get("fallback", {})
        source = ratios_cfg.get("source")
        measured = None
        if source and Path(source).exists():
            loaded = json.loads(Path(source).read_text(encoding="utf-8"))
            if str(loaded.get("split", "dev")) == "test": raise SystemExit(
                f"mode_ratios.source {source!r} was measured on TEST split — refusing a test-measured training input; point at dev artifact."
            )
            measured = loaded.get("mode_ratios", loaded)
        elif source: # An explicit ablation source must exist. The main method uses source: null and the fixed mix.
            raise FileNotFoundError(
                f"mode_ratios.source is set but missing: {source!r} (cwd={Path.cwd()}). Run `analyze.py --stage segmenter-infer` + "
                f"`--stage segmenter-errors` for this language first, or set mode_ratios.source: null for the fixed mix."
            )
        print(f"[sampler] mode_ratios: {'MEASURED ' + str(source) if measured is not None else 'designed fallback'} "
              f"-> {normalized_mode_ratios(measured if measured is not None else fallback_ratios)}", flush=True)

        jitter_cfg = dict(slt_cfg.get("jitter", {}))
        pool = pool_key(slt_cfg) # Pooled segmentation run uses designed calibration, never target language's measured artifact
        if pool and jitter_cfg.get("source"): raise ValueError(
            f"pooled run ({pool}) cannot read measured jitter {jitter_cfg['source']!r}; "
            "set jitter.source: null to use the designed pretraining distribution"
        )
        mode_ratios = measured if measured is not None else fallback_ratios
        # Guard the optional event-mix ablation against a degenerate all-clean measurement.
        threshold = float(ratios_cfg.get("degenerate_mode1_threshold", 0.9))
        if measured is not None and normalized_mode_ratios(measured).get("mode1", 0.0) >= threshold:
            print(
                f"[sampler] WARNING: measured event-mix ratios are degenerate "
                f"(mode1={normalized_mode_ratios(measured).get('mode1', 0.0):.3f} >= {threshold}); the segmenter is too clean to yield a "
                f"useful misalignment distribution. Using the DESIGNED fallback ratios + jitter (configs/dlm.yaml: mode_ratios.fallback, "
                f"jitter.fallback_laplace). Robustness is evaluated via the controlled RQ1 sweep.", flush=True,
            )
            mode_ratios = fallback_ratios
            jitter_cfg["source"] = None  # force the designed fallback_laplace jitter, not the ~0 measured CDF

        if pool:
            geometry = slt_cfg.get("pretrain_geometry", {}) or {}
            missing = {"buffer_cap_s", "min_span_frames"} - set(geometry)
            if missing: raise ValueError(
                f"pooled run ({pool}) needs pretrain_geometry.{'/'.join(sorted(missing))}; "
                "target inference geometry must not leak into multilingual pretraining"
            )
            buffer_cap_s, min_span_frames = float(geometry["buffer_cap_s"]), int(geometry["min_span_frames"])
            if buffer_cap_s <= 0 or min_span_frames < 1: raise ValueError(
                f"invalid pretrain_geometry: buffer_cap_s={buffer_cap_s}, min_span_frames={min_span_frames}"
            )
        else:
            buffer_cap_s = float(inference_cfg.get("buffer_cap_s", 18.0))
            min_span_frames = int((inference_cfg.get("span_selection", {}) or {}).get(
                "min_span_frames", int((inference_cfg.get("boundary_stability", {}) or {}).get("delta_enc_frames", 3)) + 1,
            ))

        aug_cfg = slt_cfg.get("augmentation", {})
        fps_cfg = (aug_cfg or {}).get("fps", {})
        return cls(
            records=records, jitter=JitterSampler.from_config(jitter_cfg),
            mode_ratios=mode_ratios, buffer_cap_s=buffer_cap_s, min_span_frames=min_span_frames,
            mode2_subcase_weights=slt_cfg.get("mode2_subcase_weights"), mode3_span_counts=slt_cfg.get("mode3_span_counts"),
            fps_aug_enabled=bool((aug_cfg or {}).get("enabled", aug_cfg is not None) and fps_cfg.get("enabled", aug_cfg is not None)),
            fps_aug_min=float(fps_cfg.get("min_fps", 15.0)), fps_aug_max=float(fps_cfg.get("max_fps", 30.0)),
            pose_augment_cfg=pose_augment_cfg, seed=int(slt_cfg.get("seed", 42))
        )

    def _choose_mode(self) -> str:
        keys = list(self.mode_ratios.keys())
        probs = np.asarray([self.mode_ratios[k] for k in keys], dtype=np.float64)
        probs = probs / probs.sum()
        return str(self.rng.choice(keys, p=probs))

    def _choose_anchor(self, index: int) -> tuple[VideoRecord, int]:
        # Anchor is a DETERMINISTIC function of the global sample index: with steps_per_epoch == len(anchors) and shuffle=False, indices 
        # 0..N-1 map bijectively to anchors 0..N-1, so every GT sentence anchors exactly 1 window/epoch — with NO cross-call/cross-worker 
        # state. (Stateful-cursor sampling gave uneven coverage under num_workers>0, as DataLoader dispatches whole BATCHES round-robin, 
        # so worker's call count needn't equal any fixed anchor shard.) Mode/jitter stay random (drawn from self.rng, per-worker reseeded).
        ridx, sidx = self.anchors[int(index) % len(self.anchors)]
        return self.records[ridx], sidx

    def _clip_window(self, rec: VideoRecord, start_s: float, end_s: float) -> tuple[float, float]:
        # Clamp BOTH ends into the pose stream. start_s must leave room for >=1 frame — a right-truncated anchor's
        # cut time (Mode 2a) or a caption whose onset sits past the extracted poses can arrive > duration, and
        # clamping only end_s left start_s > end_s → load_pose_frames("Invalid frame range") on real data.
        dur = float(rec.pose.duration_s)
        start_s = min(max(0.0, float(start_s)), max(0.0, dur - 1.0 / rec.pose.fps))
        end_s = min(float(end_s), dur)
        if end_s - start_s > self.buffer_cap_s: end_s = start_s + self.buffer_cap_s
        if end_s <= start_s: end_s = min(dur, start_s + 1.0 / rec.pose.fps)
        return start_s, end_s

    def _cut_time(self, anchor, lo: float = 0.05, hi: float = 0.95) -> float:
        # Absolute time of a spurious internal cut, from segmenter-error analysis's over-seg cut-position distribution
        # (JitterSampler.sample_cut; uniform fallback). Clamped away from the exact edges.
        rel = min(max(self.jitter.sample_cut(self.rng), lo), hi)
        return float(anchor.start_s + rel * anchor.duration_s)


    def _mode1_spec(self, rec: VideoRecord, anchor_idx: int) -> WindowSpec: # Complete-anchor window
        # Jitter both edges, retry until anchor's B and terminator frame are both inside; fall back to a clean clip if jitter never fits.
        anchor = rec.sentences[anchor_idx]
        eps = 1.0 / rec.pose.fps
        for _ in range(20):
            dh, dt = self.jitter.sample(self.rng)
            # segmenter-error analysis stores signed offsets as pred_boundary - gt_boundary.
            start_s, end_s = self._clip_window(rec, anchor.start_s + dh, anchor.end_s + dt)
            # The end check mirrors first_complete_span(min_tail_s=1/fps) in materialize(); a looser check here
            # would classify the window complete yet yield no translation target (silently unsupervised Mode 1).
            if classify_anchor_visibility(anchor, start_s, end_s) == "complete" and anchor.end_s + eps <= end_s:
                return WindowSpec(rec.video_id, start_s, end_s, "mode1", anchor_idx)
        return WindowSpec(rec.video_id, *self._clip_window(rec, anchor.start_s, anchor.end_s + eps), "mode1", anchor_idx)


    def _full_evidence_spec(self, rec: VideoRecord, anchor_idx: int) -> WindowSpec:
        """Mode-1-equivalent window for full-evidence decode, with 1 extra constraint: the anchor must be the window's FIRST complete span. 
        The full-evidence decode has no explicit target — the model (trained on first-complete-span rule) translates the earliest complete 
        sentence in its conditioning. A plain `_mode1_spec` window whose head jitter pulls in a complete earlier neighbour would therefore 
        yield y_full for the *neighbour*, not the anchor the truncated view shows: the verified gate (f==r) then never fires (dead CB batch), 
        and the unverified ablation would supervise toward wrong sentence's text. Falls back to the clean anchor clip, where anchor-first 
        is guaranteed (an earlier sentence cannot have its B inside a window that starts at the anchor's start)."""
        anchor = rec.sentences[anchor_idx]
        eps = 1.0 / rec.pose.fps
        for _ in range(20):
            dh, dt = self.jitter.sample(self.rng)
            start_s, end_s = self._clip_window(rec, anchor.start_s + dh, anchor.end_s + dt)
            if (classify_anchor_visibility(anchor, start_s, end_s) == "complete" and anchor.end_s + eps <= end_s
                and first_complete_span(rec.sentences, start_s, end_s, eps, min_span_s=self.min_span_frames / rec.pose.fps) is anchor):
                return WindowSpec(rec.video_id, start_s, end_s, "mode1", anchor_idx)
        return WindowSpec(rec.video_id, *self._clip_window(rec, anchor.start_s, anchor.end_s + eps), "mode1", anchor_idx)


    def _mode2_spec(self, rec: VideoRecord, anchor_idx: int) -> WindowSpec: # Truncated-anchor window
        """`right` keeps the start, cuts before the end (B, no terminator); `left` cuts after the start, keeps the end + its terminator 
        frame (no B); `both` is a strictly-interior slice (all I).

        The truncation depth — where the window cuts *inside* the anchor — is drawn from measured over-segmentation cut positions 
        (`JitterSampler.sample_cut`), the empirical answer to "where does the segmenter split a sentence". The *surviving* outer edge 
        carries ordinary boundary jitter (Δ_head/Δ_tail), so e.g. a right-truncated window's true start still wobbles like a real 
        Started-Pre/Post-Signing event. Uniform interior cut is used only when no over-seg was measured."""
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
            # Keep true end, discard the head: window starts at the spurious cut. Tail jitter may only push the end OUTWARD (abs(dt), see below) 
            # so the terminator frame (O, or the next sentence's B) stays inside — otherwise the GT end leaves the window and labels no longer 
            # describe a left-truncation (P2: labels follow the window).
            cut = min(self._cut_time(anchor), anchor.end_s - eps)
            # End must sit strictly past the anchor end (classify_anchor_visibility uses end_s < window_end), so the terminator is 
            # inside; tail jitter only extends it further out — abs(dt), NOT max(dt, eps): with max(), half of a zero-loc jitter draw 
            # collapses to a window edge EXACTLY on GT terminator, a zero-error corner segmenter-error analysis measures at ~3% — the 
            # head learns "window edge = sentence edge", a shortcut that is false at deployment (FSM buffers start at terminator−δ; 
            # whole-video chunks on an arbitrary 18s grid) and that doubles B-at-frame-0 mass under the measured mix.
            start_s, end_s = self._clip_window(rec, cut, min(rec.pose.duration_s, anchor.end_s + max(abs(dt), eps)))
        else:  # "right": keep the true start, cut before the end. Head jitter only pulls the start outward.
            cut = max(self._cut_time(anchor), anchor.start_s + eps)
            start_lo = max(0.0, anchor.start_s - abs(dh))  # abs: same zero-error-corner removal as the tail above
            # A COMPLETE earlier sentence inside the right-truncated view is poison for the confidence-bound term: the decoder 
            # — correctly, per the shared first-complete-span rule — would translate the NEIGHBOUR, while y_full is anchored on 
            # the anchor (_full_evidence_spec enforces anchor-first), so the gate would penalize correct behaviour. Clamp the 
            # start past the predecessor's B: the neighbour can then only appear left-truncated (tail I-frames — never a selectable 
            # target), which is also exactly the post-commit leftover geometry streaming produces.
            prev_starts = [s.start_s for s in rec.sentences if s.start_s < anchor.start_s]
            if prev_starts: start_lo = max(start_lo, max(prev_starts) + eps)
            start_s, end_s = self._clip_window(rec, start_lo, cut)
        return WindowSpec(rec.video_id, start_s, end_s, "mode2", anchor_idx, subcase)  # type: ignore[arg-type]


    def _mode3_spec(self, rec: VideoRecord, anchor_idx: int) -> WindowSpec: # Multi-complete window
        """Span the anchor plus k-1 successors, k drawn from `mode3_span_counts` (default: always 2), so >=2 sentences are fully inside; 
        degrade to Mode 1 if even the pair yields no 2 complete spans. The translation target is later chosen by `first_complete_span`. 
        Edges carry measured jitter like every other mode — an exact [anchor B, last end] window would train a boundary distribution
        (B at frame 0) the buffer never produces. The drawn k clamps to the sentences the video has left and to buffer cap: acceptance 
        only requires >=2 completes, so an over-ambitious k degrades to the widest window that fits, not to a retry storm."""
        anchor = rec.sentences[anchor_idx]
        k = int(self.rng.choice(self._mode3_counts, p=self._mode3_count_probs))
        end_idx = min(anchor_idx + k - 1, len(rec.sentences) - 1)
        # Respect the buffer cap AND skip quarantined successors (reliable=False): a quarantined span cannot be a
        # window end target — it has no usable boundary, and a 20s+ one would blow the window past the cap anyway.
        while end_idx > anchor_idx + 1 and (
            not getattr(rec.sentences[end_idx], "reliable", True) or rec.sentences[end_idx].end_s - anchor.start_s > self.buffer_cap_s
        ): end_idx -= 1
        end_anchor = rec.sentences[min(max(end_idx, anchor_idx + 1), len(rec.sentences) - 1)]
        eps = 1.0 / rec.pose.fps

        for _ in range(20):
            dh, dt = self.jitter.sample(self.rng)
            start_s, end_s = self._clip_window(rec, anchor.start_s + dh, end_anchor.end_s + dt)
            if count_complete_spans(rec.sentences, start_s, end_s, eps, min_span_s=self.min_span_frames / rec.pose.fps) >= 2:
                return WindowSpec(rec.video_id, start_s, end_s, "mode3", anchor_idx)

        start_s, end_s = self._clip_window(rec, anchor.start_s, end_anchor.end_s + eps)
        if count_complete_spans(
            rec.sentences, start_s, end_s, eps, min_span_s=self.min_span_frames / rec.pose.fps
        ) < 2: return self._mode1_spec(rec, anchor_idx)
        return WindowSpec(rec.video_id, start_s, end_s, "mode3", anchor_idx)


    def _mode4_spec(self, rec: VideoRecord) -> WindowSpec: # All-gap window
        # Sample inside an inter-sentence gap (≥0.5s of no signing); falls back to a Mode-2 window if the video has no usable gap.
        gaps: list[tuple[float, float]] = []
        prev = 0.0
        for span in rec.sentences:
            if span.start_s > prev: gaps.append((prev, span.start_s))
            prev = max(prev, span.end_s)

        if prev < rec.pose.duration_s: gaps.append((prev, rec.pose.duration_s))
        # Trusted gaps only: long uncaptioned stretches (> TRUSTED_GAP_S, mostly intros/outros) may contain uncaptioned signing — 
        # an "all-gap" window there could be all-signing, the exact opposite of what Mode 4 trains (stay quiet on non-signing input).
        gaps = [(s, e) for s, e in gaps if 0.5 <= e - s <= TRUSTED_GAP_S]
        if not gaps:
            # Reliable spans only: this is the one path that bypasses `self.anchors`, and a quarantined anchor would
            # send its multi-sentence text to the confidence-bound reference — the exact label quarantine exists to
            # withhold. `self.anchors` guarantees at least one reliable span in any record that reaches the sampler.
            cands = [i for i, sp in enumerate(rec.sentences) if getattr(sp, "reliable", True)]
            return self._mode2_spec(rec, cands[int(self.rng.integers(0, len(cands)))])

        gap = gaps[int(self.rng.integers(0, len(gaps)))]
        length = min(self.buffer_cap_s, max(0.5, gap[1] - gap[0]))
        start_s = float(self.rng.uniform(gap[0], max(gap[0], gap[1] - length)))
        end_s = min(gap[1], start_s + length)
        return WindowSpec(rec.video_id, start_s, end_s, "mode4")


    def spec_for(self, index: int) -> tuple[VideoRecord, WindowSpec]:
        """The WindowSpec for a global sample index, WITHOUT loading poses.

        Draws are seeded from the index, so this is a pure function: same index yields same window in main process and in any DataLoader worker. 
        That is what lets a length-bucketing batch sampler predict a window's frame count from a ~0.02 ms call instead of materialising it (see 
        data.loader.LengthBucketSampler), and it also makes an epoch's content reproducible across a resume, which per-worker rng state never was.
        """
        self.rng = np.random.default_rng(self.draw_seed + int(index))
        rec, anchor_idx = self._choose_anchor(index)
        mode = self._choose_mode()
        if mode == "mode1": spec = self._mode1_spec(rec, anchor_idx)
        elif mode == "mode2": spec = self._mode2_spec(rec, anchor_idx)
        elif mode == "mode3": spec = self._mode3_spec(rec, anchor_idx)
        elif mode == "mode4": spec = self._mode4_spec(rec)
        else: raise ValueError(f"Unknown mode {mode}")
        return rec, spec

    def spec_frames(self, index: int) -> int:
        # Frame count the window will have, for length BUCKETING — an ordering key, not a contract. It mirrors load_pose_window's floor/ceil framing 
        # (+1 for inclusive end) so it never UNDER-estimates; fps augmentation only ever down-samples, so realised window is never longer than this.
        rec, spec = self.spec_for(index)
        fps = float(rec.pose.fps)
        start_f = int(np.floor(max(0.0, spec.start_s) * fps))
        end_f = int(np.ceil(min(spec.end_s, rec.pose.duration_s) * fps))
        return max(1, end_f - start_f + 1)

    def sample(self, index: int) -> WindowSample:
        rec, spec = self.spec_for(index)   # index-seeded: the window a bucketing pre-pass predicted
        # Augment the sampled training window; the Mode-2a full-evidence view (materialized separately
        # with defaults) stays un-augmented so the no-grad self-target matches inference conditions.
        return self.materialize(rec, spec, fps_aug=self.fps_aug_enabled, augment=self.pose_augmentor is not None)


    def materialize(self, rec: VideoRecord, spec: WindowSpec, fps_aug: bool = False, augment: bool = False) -> WindowSample:
        """Realize a `WindowSpec` into tensors: load+normalize pose window, build per-frame BIO labels from GT boundaries, pick translation 
        target (Mode 1/3 first-complete-span), and for Mode-2a attach the Mode-1-equivalent `full_evidence_spec` the confidence-bound term 
        decodes under no_grad. `fps_aug=True` resamples the window to a random fps and rebuilds the BIO labels from resampled timestamps."""
        poses, timestamps = load_pose_window(
            rec.pose, spec.start_s, spec.end_s, normalize=True, augment=self.pose_augmentor if augment else None
        )
        if fps_aug and poses.shape[0] > 1: poses, timestamps, _ = apply_fps_aug(
            poses, source_fps=rec.pose.fps, min_fps=self.fps_aug_min, max_fps=self.fps_aug_max, 
            rng=self.rng, source_timestamps_s=timestamps,
        )
        frame_mask = np.ones((poses.shape[0],), dtype=bool)
        labels = make_bio_labels(
            timestamps, rec.sentences, spec.start_s, spec.end_s, frame_mask,
            video_duration_s=rec.pose.duration_s,  # long uncaptioned stretches -> UNK (see make_bio_labels)
        )
        anchor_span = rec.sentences[spec.anchor_index] if spec.anchor_index is not None else None
        target, full_evidence_spec = None, None

        # The mode names must describe what the window CONTAINS after jitter + buffer-cap clip, because supervision follows content: 
        # inference's first-complete-span rule selects whatever complete span is present, regardless of which mode drew the window. 
        # Relabelling keeps the logged per-mode losses / drift check honest AND closes 2 train/inference expectation gaps; 
        # rejection-resampling instead would distort the measured jitter CDF.
        eps = 1.0 / rec.pose.fps
        n_complete = count_complete_spans(rec.sentences, spec.start_s, spec.end_s, eps, min_span_s=self.min_span_frames / rec.pose.fps)
        if spec.mode == "mode1" and n_complete >= 2:
            spec = replace(spec, mode="mode3")  # jitter captured a complete neighbour → ≥2 complete spans
        elif spec.mode == "mode2" and n_complete >= 1:
            # The jittered edge swallowed a COMPLETE sentence (e.g. 'left' whose tail jitter pulled the successor fully inside). FSM 
            # would select & translate it — training must supervise it identically, not leave the window target-less as a nominal mode2.
            spec = replace(spec, mode="mode3" if n_complete >= 2 else "mode1", subcase=None)
        elif spec.mode in {"mode1", "mode3"} and n_complete == 0 and spec.anchor_index is not None:
            # The buffer-cap clip (anchor longer than buffer_cap_s) cut the anchor's terminator: no complete span was realized, so this 
            # IS a right-truncated window — mode2a semantics (BIO-only + the CB machinery), not a silently-unsupervised "mode1".
            spec = replace(spec, mode="mode2", subcase="right")
        elif spec.mode == "mode2" and spec.anchor_index is not None:
            # Subcase honesty after clipping: e.g. a 'left' window whose kept tail exceeded buffer_cap_s lost the anchor's terminator too 
            # → realized geometry is 'both'. Supervision is unchanged (2b/2c are both BIO-only).
            realized = classify_anchor_visibility(rec.sentences[spec.anchor_index], spec.start_s, spec.end_s)
            if realized in {"right", "left", "both"} and realized != spec.subcase: spec = replace(spec, subcase=realized)

        if spec.mode in {"mode1", "mode3"}: target = first_complete_span(
            rec.sentences, spec.start_s, spec.end_s, 1.0 / rec.pose.fps, min_span_s=self.min_span_frames / rec.pose.fps
        )
        elif spec.mode == "mode2" and spec.subcase == "right" and spec.anchor_index is not None:
            target = None
            # Attach the CB full-evidence view ONLY if the WHOLE anchor fits in a ≤buffer_cap_s window. An over-cap anchor (buffer-cap-clip 
            # relabel above) can't be seen complete even by the "full-evidence" view, so its y_full self-target would itself be truncated — 
            # keep it BIO-only rather than supervise CB against a truncated target.
            anchor = rec.sentences[spec.anchor_index]
            if anchor.end_s - anchor.start_s <= self.buffer_cap_s:
                full_evidence_spec = self._full_evidence_spec(rec, spec.anchor_index)

        # χ from sampler bookkeeping (membership gate §2.7): frames of a PREDECESSOR sentence straddling the window's left edge. 
        # In the streaming interpretation the window edge mimics the terminator−δ cut, so a predecessor's leftover tail is content 
        # the FSM already emitted — the gate floors it unconditionally (no cross-seam duplication). The ANCHOR itself straddling 
        # the edge (Mode 2b) is the MISSED-HEAD case, NOT a commit: the FSM's commit log would show nothing there (χ=0 at inference), 
        # so training mirrors it with χ=0 and leaves those frames to the trust-scaled wall/ramp — which is what lets translation vote
        # to relocate a start (§2.6). Inference-parity of χ is the invariant; the anchor test enforces it.
        commit_mask = np.zeros((poses.shape[0],), dtype=bool)
        for span in rec.sentences:
            straddles = span.start_s < spec.start_s < span.end_s
            is_predecessor = anchor_span is None or span.start_s < anchor_span.start_s
            if straddles and is_predecessor: commit_mask |= (timestamps >= span.start_s) & (timestamps < span.end_s)
        return WindowSample(
            spec=spec, poses=poses, timestamps_s=timestamps - spec.start_s, bio_labels=labels, 
            frame_mask=frame_mask, spans=rec.sentences, translation_target=target, anchor_span=anchor_span, 
            full_evidence_spec=full_evidence_spec, commit_mask=commit_mask,
        )

    @staticmethod
    def to_dict(sample: WindowSample) -> dict:
        return {
            "spec": asdict(sample.spec), "poses": sample.poses, "timestamps_s": sample.timestamps_s,
            "bio_labels": sample.bio_labels, "frame_mask": sample.frame_mask, "commit_mask": sample.commit_mask,
            "translation_target": asdict(sample.translation_target) if sample.translation_target else None,
            "anchor_span": asdict(sample.anchor_span) if sample.anchor_span else None,
            "full_evidence_spec": asdict(sample.full_evidence_spec) if sample.full_evidence_spec else None,
        }
