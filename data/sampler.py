"""Misalignment-aware window sampler over 4 real-timeline window modes, and its window-edge jitter."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace

import numpy as np
from data.loader import VideoRecord
from utils import lambda_min_frames
from data.windowing import (
    TRUSTED_GAP_S, WindowSample, WindowSpec, classify_anchor_visibility,
    count_complete_spans, first_complete_span, make_bio_labels,
)
from poses import load_pose_window, normalize_keypoints_unisign


@dataclass
class JitterSampler:
    # Each edge offset ~ Uniform(-context_s, +context_s). This spreads edges around the anchor;
    # the finite band and mode constraints still give information about the boundary location.
    context_s: float
    # Mode-2 cut depth ~ Uniform(cut_lo, cut_hi), a fraction of the anchor duration (WindowSampler._cut_time). 
    # The interior range avoids nearly empty windows and keeps mode 2 distinct from modes 1 and 4.
    cut_lo: float
    cut_hi: float

    @classmethod
    def from_config(cls, cfg: dict) -> "JitterSampler":
        cut_range = [float(x) for x in cfg["cut_range"]]
        if len(cut_range) != 2 or not 0.0 < cut_range[0] < cut_range[1] < 1.0: raise ValueError(
            f"jitter.cut_range must be [lo, hi] inside (0,1); got {cut_range}"
        )
        context = float(cfg["context_s"])
        if not np.isfinite(context) or context < 0: raise ValueError(f"jitter.context_s must be finite and >= 0, got {context}")
        return cls(context_s=context, cut_lo=cut_range[0], cut_hi=cut_range[1])

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        return float(rng.uniform(-self.context_s, self.context_s)), float(rng.uniform(-self.context_s, self.context_s))


def normalized_mode_ratios(raw: dict[str, float]) -> dict[str, float]:
    values = {k: max(float(v), 0.0) for k, v in raw.items()}
    total = sum(values.values())
    if total <= 0: raise ValueError("mode ratios need positive mass; set mode_ratios in the training config")
    return {k: v / total for k, v in values.items()}


class WindowSampler:
    """Emit one real-timeline training window per step.

    Each step picks a GT sentence anchor and a mode from the configured coverage mix,
    then cuts a window on the *real* video timeline — neighbour content and gaps inside the jittered range are the actual 
    adjacent frames, never concatenated clips (avoids seam artifacts). The 4 modes mirror the inference-time buffer states:

    - **Mode 1** — anchor fully inside (jittered head/tail). OPUT target = anchor.
    - **Mode 2** — anchor truncated: `right` (no terminator — the anchor's I-run reaches the window edge, → confidence-bound),
      `left` (no B, no translation loss), `both` (interior, rare).
    - **Mode 3** — ≥2 complete sentences (the spec spans the anchor and its successor); target = earliest complete span 
      (first-complete-span rule, identical at train and inference).
    - **Mode 4** — pure inter-sentence gap; BIO-only, trains the head to stay quiet.

    Truncation happens only here, by *window shaping* — never by relabelling text (premise P1). 
    BIO labels come from GT boundaries; padding is masked, never `O`.
    """

    def __init__(
        self, records: list[VideoRecord], jitter: JitterSampler, mode_ratios: dict[str, float], buffer_cap_s: float,
        mode2_subcase_weights: dict[str, float], min_span_frames: int = 0, delta_frames: int = 0, seed: int = 42
    ):
        self.records = records
        self.jitter = jitter
        self.mode_ratios = normalized_mode_ratios(mode_ratios)
        self.buffer_cap_s = float(buffer_cap_s)
        # Λ_min mirror of the commit gate's span_selection.min_span_frames: the training target/complete-span rules must skip sub-Λ_min sentences 
        # exactly like the deployed candidate_spans, or a window whose earliest complete span is sub-Λ_min supervises a sentence the gate never 
        # anchors (wrong-sentence Ω). Stage-2 windows stay on the native 24 fps grid (no fps augmentation), so this is a plain frame count.
        self.min_span_frames = int(min_span_frames)
        # δ of the FSM's post-commit cut (inference.yaml delta_enc_frames): after committing a unit, the buffer restarts 
        # at its terminator − δ. `materialize` gives a window the same left edge after a quarantined unit (see there).
        self.delta_frames = int(delta_frames)
        self.rng = np.random.default_rng(seed)
        # Base for index-seeded per-window draws (spec_for): every draw of a window comes from draw_seed + index, 
        # so an index maps to ONE window in the main process and in any DataLoader worker.
        self.draw_seed = int(seed)
        weights = dict(mode2_subcase_weights)
        # _mode2_spec dispatches by string compare with a bare else -> a typo'd key would silently sample as 'right'.
        unknown = set(weights) - {"right", "left", "both"}
        if unknown: raise ValueError(f"mode2_subcase_weights has unknown keys {sorted(unknown)}; valid: right/left/both")
        self._mode2_subcases = list(weights.keys())

        probs = np.asarray([float(weights[k]) for k in self._mode2_subcases], dtype=np.float64)
        self._mode2_subcase_probs = probs / probs.sum()
        # Quarantined spans (reliable=False) occupy time for gap/window purposes but are never anchors: no correct supervision exists for them.
        self.anchors = [(ri, si) for ri, rec in enumerate(records) for si, sp in enumerate(rec.sentences) if getattr(sp, "reliable", True)]
        if not self.anchors: raise ValueError("WindowSampler requires at least one sentence anchor.")


    @classmethod
    def from_slt_config(cls, records: list[VideoRecord], slt_cfg: dict, inference_cfg: dict) -> "WindowSampler":
        return cls( # Stage 2 only (S1 trains on whole-input chunks, data/chunks.py): windows use the deployed cap and floor.
            records=records, jitter=JitterSampler.from_config(slt_cfg["jitter"]), mode_ratios=slt_cfg["mode_ratios"],
            buffer_cap_s=float(inference_cfg["buffer_cap_s"]), min_span_frames=lambda_min_frames(inference_cfg),
            delta_frames=int(inference_cfg["boundary_stability"]["delta_enc_frames"]),
            mode2_subcase_weights=slt_cfg["mode2_subcase_weights"], seed=int(slt_cfg.get("seed", 42)),
        )

    def _choose_mode(self) -> str:
        keys = list(self.mode_ratios.keys())
        return str(self.rng.choice(keys, p=[self.mode_ratios[k] for k in keys]))

    def _choose_anchor(self, index: int) -> tuple[VideoRecord, int]:
        # Anchor is a DETERMINISTIC function of the global sample index: an epoch holds len(anchors) indices, so N consecutive indices 
        # map bijectively to anchors 0..N-1 and every GT sentence anchors exactly 1 window/epoch — with NO cross-call/cross-worker 
        # state. (A stateful cursor would give uneven coverage under num_workers>0: DataLoader dispatches whole BATCHES round-robin, 
        # so a worker's call count need not equal any fixed anchor shard.) Mode and jitter come from the index-seeded rng (spec_for).
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

    def _cut_time(self, anchor) -> float: # Absolute time of a spurious internal cut, uniform over jitter.cut_range of the anchor.
        return float(anchor.start_s + self.rng.uniform(self.jitter.cut_lo, self.jitter.cut_hi) * anchor.duration_s)


    def _mode1_spec(self, rec: VideoRecord, anchor_idx: int) -> WindowSpec: # Complete-anchor window
        # Jitter both edges, retry until anchor's B and terminator frame are both inside; fall back to a clean clip if jitter never fits.
        anchor = rec.sentences[anchor_idx]
        eps = 1.0 / rec.pose.fps
        for _ in range(20):
            dh, dt = self.jitter.sample(self.rng)
            start_s, end_s = self._clip_window(rec, anchor.start_s + dh, anchor.end_s + dt)
            # The end check mirrors first_complete_span(min_tail_s=1/fps) in materialize(); a looser check here
            # would classify the window complete yet yield no translation target (silently unsupervised Mode 1).
            if classify_anchor_visibility(anchor, start_s, end_s) == "complete" and anchor.end_s + eps <= end_s:
                return WindowSpec(rec.video_id, start_s, end_s, "mode1", anchor_idx)
        return WindowSpec(rec.video_id, *self._clip_window(rec, anchor.start_s, anchor.end_s + eps), "mode1", anchor_idx)


    def _full_evidence_spec(self, rec: VideoRecord, anchor_idx: int) -> WindowSpec:
        """Mode-1-equivalent window for full-evidence decode, with 1 extra constraint: anchor must be window's 1st complete span. Full-evidence 
        decode has no explicit target — the model (trained on first-complete-span rule) translates earliest complete sentence in its conditioning. 
        A plain `_mode1_spec` window whose head jitter pulls in a complete earlier neighbour would therefore yield y_full for the *neighbour*, not 
        the anchor the truncated view shows: verified gate (f==r) then never fires (dead CB batch). Falls back to the clean anchor clip, where the 
        anchor-first is guaranteed (earlier sentence can't have its B inside a window that starts at the anchor's start)."""
        anchor = rec.sentences[anchor_idx]
        eps = 1.0 / rec.pose.fps
        for _ in range(20):
            dh, dt = self.jitter.sample(self.rng)
            start_s, end_s = self._clip_window(rec, anchor.start_s + dh, anchor.end_s + dt)
            if (classify_anchor_visibility(anchor, start_s, end_s) == "complete" and anchor.end_s + eps <= end_s
                # (Λ_min - 1) frames: a unit that short can still hold Λ_min frames on the grid, where materialize counts frames.
                and first_complete_span(rec.sentences, start_s, end_s, eps, min_span_s=(self.min_span_frames - 1) / rec.pose.fps) is anchor):
                return WindowSpec(rec.video_id, start_s, end_s, "mode1", anchor_idx)
        return WindowSpec(rec.video_id, *self._clip_window(rec, anchor.start_s, anchor.end_s + eps), "mode1", anchor_idx)


    def _mode2_spec(self, rec: VideoRecord, anchor_idx: int) -> WindowSpec: # Truncated-anchor window
        """`right` keeps the start, cuts before the end (B, no terminator); `left` cuts after the start, keeps the end + its terminator 
        frame (no B); `both` is a strictly-interior slice (all I).

        The truncation depth — where the window cuts *inside* the anchor — is uniform over jitter.cut_range of the anchor duration (`_cut_time`). 
        The *surviving* outer edge carries the ordinary edge jitter (Δ_head/Δ_tail), so e.g. a right-truncated window's true start still wobbles 
        like a real Started-Pre/Post-Signing event."""
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
            # Keep true end, discard the head: window starts at the spurious cut. Tail jitter may only push the end OUTWARD (abs(dt), 
            # see below) so the terminator frame (O, or the next sentence's B) stays inside — otherwise the GT end leaves the window 
            # and labels no longer describe a left-truncation (P2: labels follow the window).
            cut = min(self._cut_time(anchor), anchor.end_s - eps)
            # End must sit strictly past the anchor end (classify_anchor_visibility uses end_s < window_end), so the terminator is 
            # inside; tail jitter only extends it further out — abs(dt), NOT max(dt, eps): with max(), half of the symmetric jitter 
            # draws would put the window edge EXACTLY on the GT terminator, and the head would learn "window edge = sentence edge", 
            # a shortcut that is false at deployment (FSM buffers start at terminator−δ; offline chunk edges fall at arbitrary times).
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
        """Span the anchor and its successor, so 2 sentences are fully inside; degrade to Mode 1 if the pair yields no 2 complete spans. 
        The translation target is later chosen by `first_complete_span`. Edges carry the designed jitter like every other mode — an exact 
        [anchor B, successor end] window would train a boundary distribution (B at frame 0) the buffer never produces."""
        anchor = rec.sentences[anchor_idx]
        end_anchor = rec.sentences[min(anchor_idx + 1, len(rec.sentences) - 1)]
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
        # The whole of one inter-sentence gap (≥0.5s of no signing); falls back to a Mode-2 window if the video has no usable gap.
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

        # A trusted gap is at most TRUSTED_GAP_S, far under buffer_cap_s, so the window is the whole gap.
        start_s, end_s = gaps[int(self.rng.integers(0, len(gaps)))]
        return WindowSpec(rec.video_id, start_s, end_s, "mode4")


    def spec_for(self, index: int) -> tuple[VideoRecord, WindowSpec]:
        """The WindowSpec for a global sample index, WITHOUT loading poses.

        Draws are seeded from the index, so this is a pure function: same index yields same window in main process and in any DataLoader worker. 
        That is what lets a length-bucketing batch sampler predict a window's frame count from a ~0.02 ms call instead of materialising it (see 
        data.loader.LengthBucketSampler), and it also makes an epoch's content reproducible across a resume and any worker layout.
        """
        self.rng = np.random.default_rng(self.draw_seed + int(index))
        rec, anchor_idx = self._choose_anchor(index)
        mode = self._choose_mode()
        if mode in ("mode1", "mode3") and rec.sentences[anchor_idx].duration_s + 1.0 / rec.pose.fps > self.buffer_cap_s:
            # No complete view of this unit exists at this capacity: every jittered mode-1/3 draw would be clamped by
            # _clip_window and fall back to the zero-jitter clean clip, pinning B at the window edge. Draw a truncation.
            mode = "mode2"
        if mode == "mode1": spec = self._mode1_spec(rec, anchor_idx)
        elif mode == "mode2": spec = self._mode2_spec(rec, anchor_idx)
        elif mode == "mode3": spec = self._mode3_spec(rec, anchor_idx)
        elif mode == "mode4": spec = self._mode4_spec(rec)
        else: raise ValueError(f"Unknown mode {mode}")
        return rec, spec

    def spec_frames(self, index: int) -> int:
        # Frame count the window will have, for length BUCKETING — an ordering key, not a contract. It mirrors 
        # load_pose_window's floor/ceil framing (+1 for inclusive end) so it never UNDER-estimates.
        rec, spec = self.spec_for(index)
        fps = float(rec.pose.fps)
        start_f = int(np.floor(max(0.0, spec.start_s) * fps))
        end_f = int(np.ceil(min(spec.end_s, rec.pose.duration_s) * fps))
        return max(1, end_f - start_f + 1)

    def sample(self, index: int) -> WindowSample:
        rec, spec = self.spec_for(index)   # index-seeded: the window a bucketing pre-pass predicted
        return self.materialize(rec, spec)


    def materialize(self, rec: VideoRecord, spec: WindowSpec) -> WindowSample:
        """Realize a `WindowSpec` into tensors: load+normalize pose window, build per-frame BIO labels from GT boundaries, pick translation 
        target (Mode 1/3 first-complete-span), and for Mode-2a attach the Mode-1-equivalent `full_evidence_spec` the confidence-bound term 
        decodes under no_grad. The window stays on the native pose grid, as at inference."""
        poses, timestamps = load_pose_window(rec.pose, spec.start_s, spec.end_s, normalize=False)
        # The loader floors the start frame, so it can return 1 frame BEFORE spec.start_s. Drop it: the labels, the completeness test and 
        # the commit frontier below all measure the window from spec.start_s, and a pre-start frame could show an onset that the labels 
        # call a cut (an illegal O->I). 1 ms absorbs float32 time noise. The drop comes before Uni-Sign normalization (below post-commit
        # shift), so the dropped frame cannot set the window's body box.
        keep = timestamps >= spec.start_s - 1e-3
        if keep.any() and not keep.all(): poses, timestamps = poses[keep], timestamps[keep]
        anchor_span = rec.sentences[spec.anchor_index] if spec.anchor_index is not None else None
        target, full_evidence_spec = None, None

        # Eligibility uses the actual sampled frame grid, also used by the gate's GT frame conversion.
        relative_times = (timestamps - spec.start_s).astype(np.float32)
        complete = []
        for span in sorted(rec.sentences, key=lambda x: (x.start_s, x.end_s)):
            if not span.reliable or span.start_s < spec.start_s: continue
            begin, end = np.searchsorted(relative_times, np.asarray([span.start_s - spec.start_s, span.end_s - spec.start_s], dtype=np.float32))
            if begin < end < len(relative_times): complete.append((span, int(end - begin)))
            
        # Λ_min in frames of the native grid, the same count the gate and the FSM apply.
        candidates = tuple(span for span, frames in complete if frames >= self.min_span_frames)
        if candidates:
            spec = replace(spec, mode="mode3" if len(candidates) >= 2 else "mode1", subcase=None)
            target = candidates[0]
        elif anchor_span is not None:
            represented = any(span is anchor_span for span, _ in complete)
            if represented: # A complete but too-short unit stays BIO-supervised; it is not a truncation target.
                spec = replace(spec, mode="mode3" if len(complete) >= 2 else "mode1", subcase=None)
            else:
                has_start = anchor_span.start_s >= spec.start_s and anchor_span.start_s < spec.end_s
                end_idx = int(np.searchsorted(relative_times, np.float32(anchor_span.end_s - spec.start_s)))
                has_end = anchor_span.end_s > spec.start_s and end_idx < len(relative_times)
                if has_start and not has_end: subcase = "right"
                elif not has_start and has_end: subcase = "left"
                else: subcase = "both"
                spec = replace(spec, mode="mode2", subcase=subcase)

        # POST-COMMIT SHIFT. The FSM cannot know a unit is quarantined: when 1 (U) ends inside the window before the text target starts (1st candidate, 
        # or a Mode-2a anchor, the confidence-bound reference), FSM commits U first and restarts its buffer at U's terminator − δ. The window does the 
        # same, and never moves left: U becomes the straddling committed predecessor that the commit mask below marks, and the target keeps its text. 
        # Materializing again drops the frames before the new start and relabels from it (P1/P2). The full view (a Mode-1 view of the anchor) takes 
        # the same shift in its own materialize, so the teacher and the student see the same committed predecessor.
        text_start = target.start_s if target is not None else anchor_span.start_s if spec.subcase == "right" else None
        if text_start is not None:
            ends = [span.end_s for span in rec.sentences if not span.reliable and spec.start_s < span.end_s <= text_start]
            cut = max(ends, default=-np.inf) - self.delta_frames / rec.pose.fps
            if cut > spec.start_s: return self.materialize(rec, replace(spec, start_s=cut))
        if poses.shape[1:] == (133, 3): poses = normalize_keypoints_unisign(poses).astype(np.float32, copy=False)
        labels = make_bio_labels(
            timestamps, rec.sentences, spec.start_s, spec.end_s,
            video_duration_s=rec.pose.duration_s,  # long uncaptioned stretches -> UNK (see make_bio_labels)
        )
        if spec.mode == "mode2" and spec.subcase == "right" and anchor_span is not None:
            # A full view needs the anchor and a represented terminator, within the same capacity.
            if anchor_span.duration_s + 1.0 / rec.pose.fps <= self.buffer_cap_s:
                full_evidence_spec = self._full_evidence_spec(rec, spec.anchor_index)

        # χ from sampler bookkeeping (docs/membership_gate.md §2.4 (committed prefix) / §2.5 (log eps floor)): frames of a PREDECESSOR sentence 
        # straddling the window's left edge. In the streaming interpretation the window edge mimics the terminator−δ cut, so a predecessor's 
        # leftover tail is content FSM already emitted — the gate floors it unconditionally (no cross-seam duplication). The ANCHOR itself 
        # straddling the edge (Mode 2b) is MISSED-HEAD case, NOT a commit: FSM's commit log would show nothing there (χ=0 at inference), 
        # so training mirrors it with χ=0 and leaves those frames to Ω (the posterior membership) — which is what lets translation vote to 
        # relocate a start. Inference-parity of χ is invariant; the anchor test enforces it.
        commit_mask = np.zeros((poses.shape[0],), dtype=bool)
        for span in rec.sentences:
            straddles = span.start_s < spec.start_s < span.end_s
            is_predecessor = anchor_span is None or span.start_s < anchor_span.start_s
            # Commit state is a time frontier, including any earlier context loaded by frame rounding.
            if straddles and is_predecessor: commit_mask |= timestamps < span.end_s
        return WindowSample(
            spec=spec, poses=poses, timestamps_s=timestamps - spec.start_s, bio_labels=labels, translation_target=target, 
            anchor_span=anchor_span, full_evidence_spec=full_evidence_spec, commit_mask=commit_mask, candidate_sentences=candidates
        )

    @staticmethod
    def to_dict(sample: WindowSample) -> dict:
        return {
            "spec": asdict(sample.spec), "poses": sample.poses, "timestamps_s": sample.timestamps_s,
            "bio_labels": sample.bio_labels, "commit_mask": sample.commit_mask,
            "translation_target": asdict(sample.translation_target) if sample.translation_target else None,
            "anchor_span": asdict(sample.anchor_span) if sample.anchor_span else None,
            "candidate_sentences": [
                {"start_s": float(sp.start_s - sample.spec.start_s), "end_s": float(sp.end_s - sample.spec.start_s), "text": sp.text} 
                for sp in sample.candidate_sentences
            ]
        }
