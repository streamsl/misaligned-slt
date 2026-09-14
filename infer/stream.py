from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from data.windowing import BIO
from poses import normalize_keypoints_unisign
from infer.duration_decode import DurationDecoder
from infer.commit_gate import CommitGate, open_span_start, select_target_span
from infer.stability import display_prefix
from utils import LAMBDA_MIN_FRAMES


def leading_i_run_end(bio_tags: torch.Tensor) -> int | None:
    # 1st O-or-B after a buffer-start I-run (the continuation of a force-cut sentence).
    # None if the buffer does not start with I, or the run reaches the buffer end.
    tags = bio_tags.tolist()
    if not tags or tags[0] != BIO["I"]: return None
    for idx, tag in enumerate(tags):
        if tag in (BIO["O"], BIO["B"]): return idx
    return None


@dataclass
class StrideHypothesis: # One stride's candidate decode for the span the FSM is currently considering (committed or not).
    commit_time_s: float      # wall-clock of the stride that produced it
    span_start_s: float       # absolute span bounds; used to group strides that concern the same sentence
    span_end_s: float
    token_ids: torch.Tensor
    token_confidence: torch.Tensor
    committed: bool


@dataclass
class StreamingEvent:
    start_s: float
    end_s: float
    token_ids: torch.Tensor
    token_confidence: torch.Tensor
    bio_start_index: int
    bio_end_index: int
    flagged_partial: bool = False
    commit_time_s: float = 0.0  # stride wall-time of the commit; for emission latency vs GT end
    # Cut landed on a certified BIO terminator (any complete span, forced or not); false only on the mid-sentence open-span cap path. Not 
    # `not flagged_partial`: also holds for forced COMPLETE spans & gating χ-restoration on it drops alternate sentences under cap pressure.
    terminator_commit: bool = False


class MoryossefRunnerAdapter(torch.nn.Module): # Run external segmenter on 1 normalized active buffer, with buffer-local velocity.
    def __init__(self, segmenter, velocity: bool = True):
        super().__init__()
        self.segmenter, self.velocity = segmenter, bool(velocity)
        self.front_end = SimpleNamespace(extract_bio_tap=lambda poses, mask, timestamps_s: (poses, mask, timestamps_s), prompt_length=lambda: 0)

    def bio_head(self, poses, frame_mask, timestamps_s):
        if poses.shape[0] != 1 or not bool(frame_mask.all()):
            raise ValueError("Online Moryossef requires one unpadded active buffer")
        if self.velocity:
            from moryossef26.dataset import append_velocity
            features = append_velocity(poses[0].detach().cpu().numpy(), timestamps_s[0].detach().cpu().numpy())
            poses = torch.as_tensor(features, device=poses.device).unsqueeze(0)
        return SimpleNamespace(logits=self.segmenter(poses, frame_mask=frame_mask, timestamps_s=timestamps_s)["phrase"])

    
class S1RunnerAdapter(torch.nn.Module):
    # Present a BioS1Model to StreamingSLTRunner: the FSM needs only the pose tap and the BIO head (gate off, no decoder).
    def __init__(self, s1):
        super().__init__()
        self.s1 = s1
        self.bio_head = s1.bio_head
        adapter = self
        class _FrontEnd:
            @staticmethod
            def extract_bio_tap(poses, frame_mask, timestamps_s=None):
                return adapter.s1.pose_encoder(poses, frame_mask), frame_mask, timestamps_s
            @staticmethod
            def prompt_length(): return 0
        self.front_end = _FrontEnd()


class StreamingSLTRunner:
    """Model-backed sawtooth inference runner.

    Stateless across strides: each stride recomputes features/BIO for the buffer and cold-starts translation for the
    selected target span (first complete span ≥ Λ_min).
    """
    def __init__(
        self, model, stride_s: float = 1.0, buffer_cap_s: float = 18.0, delta_enc_frames: int = 3, hysteresis_strides: int = 3,
        token_confidence_tau: float = 0.3, max_text_tokens: int = 128, diffusion_steps: int = 64, tau_dec: float = 0.75,
        spd_top_k: int = 1, spd_renormalize: bool = True, spd_revision: bool = True, temperature: float = 0.0,
        dcd_window_length: int | None = None, dcd_max_window_length: int | None = None, dcd_window_type: str = "sliding",
        dcd_decode_algo: str = "threshold", dcd_decode_param: int | float | None = None, dcd_sample_top_k: int | None = None,
        dcd_top_p: float | None = None, dcd_cache_type: str = "none", decode_conditioning: str = "window", 
        min_span_frames: int | None = None, forced_tail_policy: str = "skip", gate_enabled: bool = False, gate_eps: float = 1e-4, 
        record_trace: bool = False, translate: bool = True, commit_lag_s: float = 0.0, cascade_model=None
    ):
        # decode_conditioning: "window" (default) decodes under the FULL buffer — the training conditioning (Mode 1/3 feed
        # the whole jittered window; right-context is learned to be disregarded, not cropped). "span" crops to the BIO span,
        # unseen by training except at zero jitter — ablation. Emitted boundaries come from the BIO span either way.
        if decode_conditioning not in {"window", "span"}: raise ValueError(f"Unknown decode_conditioning: {decode_conditioning}")

        # forced_tail_policy — continuation of a sentence cut by a FORCED (cap) commit:
        #   "skip" (default): never selected; Mode-2b gives no supervision on that left-truncated conditioning, so the decode is undefined 
        #     behaviour. (Supervision covers states the FSM decodes with 1 known exception: anchors longer than buffer_cap_s train without 
        #     CB view (full-evidence self-target would itself be truncated) yet cap-forced commit decodes exactly that state, flagged PARTIAL.)
        #   "translate_partial": decode it flagged PARTIAL at its terminator — best-effort OOD recovery. Ablation.
        if forced_tail_policy not in {"skip", "translate_partial"}: 
            raise ValueError(f"Unknown forced_tail_policy: {forced_tail_policy}")

        self.decode_conditioning = str(decode_conditioning)
        self.forced_tail_policy = str(forced_tail_policy)
        self.model = model
        self.cascade_model = cascade_model
        if cascade_model is not None and gate_enabled: raise ValueError("The online cascade uses an ungated clean translator.")
        self.stride_s = float(stride_s)
        self.buffer_cap_s = float(buffer_cap_s)
        if not all(0.0 < value < float("inf") for value in (self.stride_s, self.buffer_cap_s)):
            raise ValueError("Stride and buffer capacity must be finite and positive.")
        self.max_text_tokens = int(max_text_tokens)
        self.diffusion_steps = int(diffusion_steps)
        self.tau_dec = float(tau_dec)

        # Λ_min (min_span_frames): shortest selectable span in encoder frames — a LABEL-domain floor that rejects flicker, set below the
        # shortest real unit so no annotated unit is unreachable. Not the re-emission guard (select_target_span(skip_term_before=χ) is).
        self.min_span_frames = int(min_span_frames) if min_span_frames is not None else LAMBDA_MIN_FRAMES
        if self.min_span_frames < 1: raise ValueError(f"min_span_frames ({self.min_span_frames}) must be >= 1")
        # translate=False: segmentation-only dry run — no decoder call, confidence 1.0 so the commit gate reduces to the boundary test. 
        # For tuning the FSM's decode on dev in minutes; events carry empty text.
        self.translate = bool(translate)
        self.commit_lag_s = float(commit_lag_s) # Delay B-terminated sentences until enough right context is observed. O terminators need no extra lag

        self.spd_top_k = int(spd_top_k)
        self.spd_renormalize = bool(spd_renormalize)
        self.spd_revision = bool(spd_revision)
        self.temperature = float(temperature)

        self.dcd_window_length = dcd_window_length
        self.dcd_max_window_length = dcd_max_window_length
        self.dcd_window_type = dcd_window_type
        self.dcd_decode_algo = dcd_decode_algo
        self.dcd_decode_param = dcd_decode_param
        self.dcd_sample_top_k = dcd_sample_top_k
        self.dcd_top_p = dcd_top_p

        self.dcd_cache_type = dcd_cache_type
        self.commit_gate = CommitGate(
            delta_enc_frames=delta_enc_frames, hysteresis_strides=hysteresis_strides, token_confidence_tau=token_confidence_tau
        )
        # Membership gate: per stride, Ω from the BIO posteriors (on-policy, no GT) + χ from commit log — decoder's training conditioning. 
        # 'window' only: 'span' crops the feature axis and changes the membership columns.
        self.gate_enabled = bool(gate_enabled) and self.decode_conditioning == "window"
        self.gate_eps = float(gate_eps)
        
        self._committed_until_s = 0.0 # χ frontier: abs end of last committed span. Earlier buffer frames already emitted (≤δ overlap kept).
        # Only consulted by forced_tail_policy="translate_partial".
        self._after_forced = False # True between a cap commit that cut an open span and that sentence's continuation resolving.
        # Did the LAST commit end at a real BIO terminator (O-or-B) rather than a cap cut? Only that certifies "a sentence
        # ENDED here", the χ-restoration's premise. Persistent: the seam is re-examined while the leftover lives.
        self._committed_is_terminator = False
        # Per-stride candidate hypotheses for offline stable-prefix analysis; None = disabled (the default).
        self.trace: list[StrideHypothesis] | None = [] if record_trace else None
        # Why-did-it-(not)-commit counters across run() calls. Low streaming recall with near-perfect frame BIO is usually
        # the gate; spans_seen vs boundary_ok vs translation_ok says which signal blocks (usually translation_ok).
        self.gate_stats: dict[str, int] = {}
        self._bio_timeline: torch.Tensor | None = None  # per-stream stitched decoded BIO tags; set by run()
        self._last_frame_s = -float("inf")  # Only a newly observed frame can supply a hysteresis vote.

    def _bump(self, key: str, n: int = 1) -> None:
        self.gate_stats[key] = self.gate_stats.get(key, 0) + int(n)

    def _decode_span(self, bio_tap: torch.Tensor, mask: torch.Tensor, span_slice: slice, omega_bias: torch.Tensor | None = None):
        if not self.translate:
            return torch.zeros((1, 0), dtype=torch.long, device=bio_tap.device), torch.ones((1, 1), dtype=torch.float32, device=bio_tap.device)
        decoder = self.cascade_model if self.cascade_model is not None else self.model
        if self.cascade_model is not None: # The clean translator encodes its own raw span, normalized like its training clips.
            raw = self._raw_buffer[span_slice]
            poses = torch.from_numpy(normalize_keypoints_unisign(raw)).unsqueeze(0).to(bio_tap.device)
            mask = torch.ones(poses.shape[:2], dtype=torch.bool, device=poses.device)
            bio_tap, mask, _ = decoder.front_end.extract_bio_tap(poses, mask)
            omega_bias = None
        elif self.decode_conditioning == "span": bio_tap, mask = bio_tap[:, span_slice], mask[:, span_slice]
        temporal = {} if self.cascade_model is not None else {
            "temporal_features": self._temporal_features if self.decode_conditioning == "window" or self._temporal_features is None \
                                                         else self._temporal_features[:, span_slice],
            "timestamps_s": self._buffer_timestamps if self.decode_conditioning == "window" else self._buffer_timestamps[:, span_slice]
        }
        tokens, confidence = decoder.generate_from_bio_tap(
            bio_tap, mask, max_text_tokens=self.max_text_tokens, diffusion_steps=self.diffusion_steps, tau_dec=self.tau_dec,
            spd_top_k=self.spd_top_k, spd_renormalize=self.spd_renormalize, spd_revision=self.spd_revision, temperature=self.temperature,
            dcd_window_length=self.dcd_window_length, dcd_max_window_length=self.dcd_max_window_length, dcd_window_type=self.dcd_window_type,
            dcd_decode_algo=self.dcd_decode_algo, dcd_decode_param=self.dcd_decode_param, dcd_sample_top_k=self.dcd_sample_top_k,
            dcd_top_p=self.dcd_top_p, dcd_cache_type=self.dcd_cache_type, omega_bias=omega_bias, **temporal
        )
        # DLM strips its synthetic BOS internally; the AR arm returns it raw (the training replay needs the start slot) — 
        # drop it here so the commit gate's confidence mean covers only produced tokens, like the DLM arm.
        if getattr(decoder, "decoder_type", "dlm") == "ar": tokens, confidence = tokens[:, 1:], confidence[:, 1:]
        return tokens, confidence

    def _stride_omega(self, bio_logits: torch.Tensor, mask: torch.Tensor, ts_b: torch.Tensor, start_s: float):
        # None when the gate is off or the model has no gate builder (test fakes / AR-only models).
        if not self.gate_enabled or not hasattr(self.model, "membership_gate"): return None
        chi = (ts_b + float(start_s)) < self._committed_until_s  # (1, T') bool, absolute timeline
        omega_bias, _ = self.model.membership_gate(
            bio_logits, mask, decoder=DurationDecoder(self.model.duration_model),
            memory_len=self.model.front_end.prompt_length() + int(bio_logits.shape[1]),
            commit_mask=chi, eps=self.gate_eps, min_span_frames=self.min_span_frames,
            # The certified frontier has the same start condition in hard and soft inference, even when overlap is 0.
            seam_is_terminator=self._committed_is_terminator, stream_start=self._committed_is_terminator, timestamps_s=ts_b,
        )
        return omega_bias

    @torch.no_grad()
    def step(
        self, poses: torch.Tensor, timestamps_s: torch.Tensor, end_s: float, last_commit_t: float = 0.0, 
        force: bool = False, emission_time_s: float | None = None,
    ) -> StreamingEvent | None:
        """One sawtooth stride over raw (T,133,3) poses in [last_commit_t, end_s).

        end_s is the exclusive evidence endpoint. emission_time_s defaults to it, but can advance during an EOF drain.
        Process at the capacity deadline before admitting more frames. force=True overrides unmet commit conditions at EOF.
        """
        if poses.ndim != 3 or poses.shape[1:] != (133, 3):
            raise ValueError("Streaming input must be raw (T,133,3) poses; normalization is per buffer.")
        start_s = float(last_commit_t)
        if end_s > start_s + self.buffer_cap_s + 1e-6:
            raise ValueError("Process the buffer at its capacity deadline before accepting more evidence.")
        emission_time_s = float(end_s if emission_time_s is None else emission_time_s)
        if not float(end_s) <= emission_time_s < float("inf"):
            raise ValueError("Emission time must be finite and cannot precede the evidence endpoint.")
        
        in_buffer = (timestamps_s >= start_s) & (timestamps_s < end_s)
        if not in_buffer.any(): return None
        buffer_full = force or end_s >= start_s + self.buffer_cap_s
        latest_frame_s = float(timestamps_s[in_buffer][-1].item())
        new_frames = latest_frame_s > self._last_frame_s
        self._last_frame_s = max(self._last_frame_s, latest_frame_s)
        if not new_frames and not buffer_full: return None

        # Normalize only the active buffer, as training does. Future and retired frames cannot set its body scale.
        device = next(self.model.parameters()).device
        raw_buffer = poses[in_buffer].detach().float().cpu().numpy()
        if self.cascade_model is not None: self._raw_buffer = raw_buffer
        poses_b = torch.from_numpy(normalize_keypoints_unisign(raw_buffer)).unsqueeze(0).to(device)
        ts_b = (timestamps_s[in_buffer] - start_s).unsqueeze(0).to(device)
        mask_b = torch.ones(poses_b.shape[:2], dtype=torch.bool, device=device)

        bio_tap, mask, ts = self.model.front_end.extract_bio_tap(poses_b, mask_b, ts_b)
        bio_output = self.model.bio_head(bio_tap, timestamps_s=ts, frame_mask=mask)
        bio_logits = bio_output.logits
        self._temporal_features = getattr(bio_output, "hidden_states", None)
        self._buffer_timestamps = ts
        chi = ts + float(start_s) < self._committed_until_s
        bio_tags = DurationDecoder(getattr(self.model, "duration_model", None)).decode(
            bio_logits, mask.long().sum(1), chi, known_start=self._committed_is_terminator, timestamps_s=ts
        )[0]
        # Stitch this stride's Viterbi tags into the whole-stream timeline
        # (latest estimate wins on revisits) so RQ2 can score the segmentation the FSM actually acted on.
        if self._bio_timeline is not None and bio_tags.numel() == int(in_buffer.sum().item()):
            self._bio_timeline[in_buffer.cpu()] = bio_tags.detach().cpu()
        omega_bias = self._stride_omega(bio_logits, mask, ts_b, start_s)

        # after_forced resolution (translate_partial only). Auto-reset if the buffer does not start with I: no continuation
        # is coming (flush landed in a gap), and waiting on a lead that never appears deadlocks the FSM.
        if self._after_forced and self.forced_tail_policy == "translate_partial":
            first = int(chi[0].sum().item())
            lead_term = leading_i_run_end(bio_tags[first:])
            if lead_term is not None: lead_term += first
            if first < bio_tags.numel() and int(bio_tags[first].item()) != BIO["I"]: self._after_forced = False
            elif lead_term is not None:
                self._after_forced = False
                if lead_term-first >= max(1, self.min_span_frames):
                    self._bump("committed"); self._bump("forced_tail_commit")
                    # Ungated: the stride's Ω is anchored where the buffer-start I-run can never open, so it would floor the
                    # very frames [0, lead_term) being decoded. Best-effort fragment decode anyway.
                    tokens, confidence = self._decode_span(bio_tap, mask, slice(first, lead_term), omega_bias=None)
                    self.commit_gate.reset()
                    return StreamingEvent(
                        start_s=float(start_s + ts_b[0, first].item()), end_s=float(start_s + ts_b[0, lead_term].item()),
                        token_ids=tokens[0].detach().cpu(), token_confidence=confidence[0].detach().cpu(),
                        bio_start_index=first, bio_end_index=lead_term, flagged_partial=True, commit_time_s=emission_time_s,
                        terminator_commit=True,  # lead_term IS the first O-or-B: a certified terminator
                    )
        elif self._after_forced: self._after_forced = False  # "skip": leftover I-run is never selected

        # χ filter: spans fully inside emitted content are never selectable (see select_target_span). 
        # Same absolute timeline as _stride_omega's commit_mask.
        chi_frames = int(((ts_b[0] + float(start_s)) < self._committed_until_s).sum().item())
        span = select_target_span(bio_tags, self.min_span_frames, skip_term_before=chi_frames)
        if span is None:
            self.commit_gate.update(None, token_confidence=None)
            if not buffer_full: return None

            # Forced commit at cap: decode the in-progress (right-truncated) span if any, else drop the gap/fragment buffer.
            b_idx = open_span_start(bio_tags)  # same open-span rule as the gate anchor
            self.commit_gate.reset()
            if b_idx is None: return None
            self._bump("committed"); self._bump("forced_commit")
            self._after_forced = True  # the cut lands mid-sentence; its continuation follows

            s_idx, last_idx = b_idx, int(bio_tags.numel()) - 1
            tokens, confidence = self._decode_span(bio_tap, mask, slice(s_idx, last_idx + 1), omega_bias=omega_bias)
            # The exclusive evidence endpoint also handles a cap between frame timestamps and a one-frame EOF tail.
            return StreamingEvent(
                start_s=float(start_s + ts_b[0, s_idx].item()), end_s=float(end_s),
                token_ids=tokens[0].detach().cpu(), token_confidence=confidence[0].detach().cpu(),
                bio_start_index=s_idx, bio_end_index=last_idx, flagged_partial=True, commit_time_s=emission_time_s,
                terminator_commit=False,  # cap cut mid-sentence: the seam is a continuation, never an onset
            )

        s_idx, term_idx = span
        # The same lag policy applies to every legal B terminator.
        lag_hold = (self.commit_lag_s > 0.0 and int(bio_tags[term_idx].item()) == BIO["B"]
                    and (float(end_s) - float(start_s + ts_b[0, term_idx].item())) < self.commit_lag_s)
        if lag_hold: self._bump("lag_hold")
        if lag_hold and not buffer_full and self.trace is None:
            self.commit_gate.update(span, token_confidence=None)
            self._bump("spans_seen")
            return None
        
        tokens, confidence = self._decode_span(bio_tap, mask, slice(s_idx, max(s_idx + 1, term_idx)), omega_bias=omega_bias)
        # A cap/drain pass can check confidence, but cannot supply a missing boundary vote on frozen evidence.
        # Such a pass always emits and resets the history below; preserve only stability already earned for this span.
        boundary_observed = new_frames or (self.commit_gate.history.latest() == span and self.commit_gate.history.stable())
        decision = self.commit_gate.update(span, token_confidence=confidence[0])
        decision.boundary_stable = decision.boundary_stable and boundary_observed
        eligible = bool(decision.should_commit) and not lag_hold
        forced = buffer_full and not eligible
        emits = eligible or forced

        # Stability trace: this candidate decode is recomputed every stride and DISCARDED whenever the gate defers, so
        # recording it costs nothing and captures how the hypothesis for one sentence evolves as frames arrive — the
        # input a stable-prefix policy replays offline (infer/stability.py). Off by default; never affects the FSM.
        if self.trace is not None:
            # Defensive trim only: generate_from_bio_tap already slices DLM output at 1st EOS (models/streaming_slt.py), and AR arm 
            # returns no trailing pad, so there is normally nothing here to cut. Kept so a future decode path that DID leak post-EOS pad 
            # (fabricated confidence 1.0, infer/decode.py) can't silently feed a reveal policy padding. It doesn't explain DLM over-reveal.
            tk = getattr(self.model, "tokenizer", None)
            tok_row, conf_row = display_prefix(
                tokens[0].detach().cpu(), confidence[0].detach().cpu(),
                eos_id=getattr(tk, "eos_token_id", None) if tk is not None else None,
                pad_id=getattr(tk, "pad_token_id", None) if tk is not None else None,
            )
            self.trace.append(StrideHypothesis(
                commit_time_s=emission_time_s,
                span_start_s=float(start_s + ts_b[0, s_idx].item()), span_end_s=float(start_s + ts_b[0, term_idx].item()),
                token_ids=tok_row.clone(), token_confidence=conf_row.clone(), committed=emits,
            ))
        self._bump("spans_seen")
        if decision.boundary_stable: self._bump("boundary_ok")
        if decision.translation_confident: self._bump("translation_ok")
        # A complete cap/drain event cuts at its terminator, retaining all later buffered sentences.
        if not emits: return None
        self._bump("committed")
        if forced: self._bump("forced_commit")
        self.commit_gate.reset()
        return StreamingEvent(
            start_s=float(start_s + ts_b[0, s_idx].item()), end_s=float(start_s + ts_b[0, term_idx].item()),
            token_ids=tokens[0].detach().cpu(), token_confidence=confidence[0].detach().cpu(),
            bio_start_index=s_idx, bio_end_index=term_idx, flagged_partial=forced, commit_time_s=emission_time_s,
            terminator_commit=True,  # complete span: the cut IS its terminator, forced or not
        )

    @torch.no_grad()
    def run(self, poses: torch.Tensor, fps: float) -> list[StreamingEvent]:
        if not 0.0 < float(fps) < float("inf"): raise ValueError("FPS must be finite and positive.")
        timestamps = torch.arange(poses.shape[0], device=poses.device, dtype=torch.float32) / float(fps)
        events: list[StreamingEvent] = []
        duration = poses.shape[0] / float(fps)  # exclusive endpoint, including the final frame
        # Overlap cut: keep the last δ frames at a committed terminator, so a terminator estimate late by up to δ still
        # leaves the next onset (its B) in the buffer — else buffer-start I never opens a span (Mode-2b silence) and that
        # sentence is dropped. The leftover is never re-emitted: select_target_span skips any span terminating at or before χ.
        overlap_s = float(self.commit_gate.history.delta_enc_frames) / float(fps)
        frame_s = 1.0 / float(fps)
        last_commit_t = 0.0
        end_s, next_stride_s = 0.0, self.stride_s

        self.commit_gate.reset()
        self._last_frame_s = -float("inf")
        self._after_forced = False
        self._committed_until_s = 0.0  # χ commit log resets per stream
        self._committed_is_terminator = False
        self._bio_timeline = torch.full((poses.shape[0],), int(BIO["UNK"]), dtype=torch.long)  # stitched tags
        self._fps = float(fps)

        def absorb(event) -> None:
            nonlocal last_commit_t
            events.append(event)
            # The ≤δ overlap the next buffer keeps must be attention-floored by membership gate (seam-duplication guard §2.7).
            self._committed_until_s = max(self._committed_until_s, float(event.end_s))
            # χ-restoration premise: `terminator_commit`, never `not flagged_partial` (see StreamingEvent).
            self._committed_is_terminator = bool(event.terminator_commit)
            # Advance without crossing the emitted endpoint, even if the cap falls between frame timestamps.
            last_commit_t = max(event.end_s - overlap_s, min(event.end_s, last_commit_t + frame_s))

        while end_s < duration:
            # Capacity is a processing deadline, not a crop. After a commit, the retained evidence is processed
            # before the next deadline; frames beyond this endpoint remain available for later passes.
            cap_end_s = last_commit_t + self.buffer_cap_s
            end_s = min(next_stride_s, cap_end_s, duration)
            event = self.step(poses, timestamps, end_s=end_s, last_commit_t=last_commit_t)
            if event is not None: absorb(event)
            elif end_s >= cap_end_s: # This gap/fragment buffer was processed at capacity with nothing to emit.
                last_commit_t = max(end_s - overlap_s, min(end_s, last_commit_t + frame_s))
                self._committed_is_terminator = False  # a capacity discard does not certify a new onset
            if end_s >= next_stride_s: next_stride_s += self.stride_s

        # EOF cannot supply missing boundary votes or right context. Flush the remaining spans explicitly;
        # any unmet stability, lag or confidence condition makes the event forced/partial.
        emission_time_s = next_stride_s
        # Loop the forced pass: 1 pass emits 1 span, else a frozen buffer with several pending sentences drops all but the first. Only the 
        # emission clock advances by 1 stride/pass; the evidence endpoint remains fixed, so clock ticks cannot satisfy right-context lag.
        while True:
            event = self.step(poses, timestamps, end_s=duration, last_commit_t=last_commit_t, force=True, emission_time_s=emission_time_s)
            if event is None: break
            absorb(event)
            emission_time_s += self.stride_s
        return events
