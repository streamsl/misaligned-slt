from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from data.windowing import BIO
from poses import normalize_keypoints_unisign
from infer.duration_decode import DurationDecoder
from infer.commit_gate import CommitGate, candidate_spans, open_span_start
from infer.stability import display_prefix
from utils import LAMBDA_MIN_FRAMES


@dataclass
class StrideHypothesis: # One stride's candidate decode for the span the FSM is currently considering (committed or not).
    commit_time_s: float      # evidence-clock time of the stride that produced it (stride end, not wall-clock)
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
    flagged_partial: bool = False
    commit_time_s: float = 0.0  # evidence-clock time of the commit (stride end); emission latency vs GT end
    # Cut landed on a certified BIO terminator (any complete span, forced or not); false only on the mid-sentence open-span cap path. Not 
    # `not flagged_partial`: also holds for forced COMPLETE spans & gating χ-restoration on it drops alternate sentences under cap pressure.
    terminator_commit: bool = False


class MoryossefRunnerAdapter(torch.nn.Module): # Run external segmenter on 1 normalized active buffer, with buffer-local velocity.
    def __init__(self, segmenter, velocity: bool = True):
        super().__init__()
        self.segmenter, self.velocity = segmenter, bool(velocity)
        # Width/height of the video now streaming (moryossef26.dataset.pose_aspect): the release normaliser needs 
        # isotropic coordinates and our poses are stored per axis. The caller sets it before each video's run.
        self.aspect: float | None = None
        self.front_end = SimpleNamespace(extract_bio_tap=lambda poses, mask, timestamps_s: (poses, mask, timestamps_s), prompt_length=lambda: 0)

    wants_raw_poses = True  # it applies the released model's own normalisation to the buffer (see step()).

    def segment(self, poses, frame_mask, timestamps_s):
        if poses.shape[0] != 1 or not bool(frame_mask.all()): raise ValueError("Online Moryossef requires one unpadded active buffer")
        from moryossef26.dataset import release_velocity, to_release_coords
        coords = to_release_coords(poses[0].detach().cpu().numpy(), getattr(self.segmenter, "release_stats", None), aspect=self.aspect)
        features = release_velocity(coords, timestamps_s[0].detach().cpu().numpy()) if self.velocity else coords
        poses = torch.as_tensor(features, device=poses.device).unsqueeze(0)
        return SimpleNamespace(logits=self.segmenter(poses, frame_mask=frame_mask, timestamps_s=timestamps_s)["phrase"])


class S1RunnerAdapter(torch.nn.Module): # Present BioS1Model to StreamingSLTRunner: FSM needs only its BIO head (gate off, no decoder)
    def __init__(self, s1):
        super().__init__()
        self.s1 = s1
        # No translator tap: the online cascade's clean translator encodes its own crop, so S1's encoder runs once per pass (in segment).
        self.front_end = SimpleNamespace(
            extract_bio_tap=lambda poses, mask, timestamps_s=None: (poses, mask, timestamps_s), prompt_length=lambda: 0
        )

    def segment(self, poses, frame_mask, timestamps_s=None): 
        return self.s1(poses, frame_mask, timestamps_s)


class StreamingSLTRunner:
    """Model-backed sawtooth inference runner.

    No decoder state across strides: each stride recomputes features/BIO for the buffer and cold-starts translation for 
    the selected target span (first complete span ≥ Λ_min); only the commit log and the per-span boundary votes persist.
    """
    def __init__(
        self, model, stride_s: float = 1.0, buffer_cap_s: float = 40.0, delta_enc_frames: int = 12, hysteresis_strides: int = 3, 
        max_text_tokens: int = 320, tau_dec: float = 0.5, spd_top_k: int = 1, spd_renormalize: bool = True, 
        min_span_frames: int | None = None, gate_enabled: bool = False, gate_eps: float = 1e-4, record_trace: bool = False, 
        translate: bool = True, cascade_model=None
    ):
        # The joint arm decodes the whole buffer under the Ω soft crop (its stage-2 training conditioning); emitted boundaries 
        # come from BIO span. The continuation of a sentence cut by a FORCED (cap) commit is never selected: it arrives as a 
        # buffer-start I-run, which never opens a span. Mode-2b gives no text supervision on that left-truncated state, so a 
        # decode of it is undefined. (Text supervision covers complete units only, P1: a premature or cap-forced commit decodes 
        # a truncated crop that no text loss trains; a cap-forced one is flagged PARTIAL.)
        self.model = model
        self.cascade_model = cascade_model
        if cascade_model is not None and gate_enabled: raise ValueError("The online cascade uses an ungated clean translator.")
        self.stride_s = float(stride_s)
        self.buffer_cap_s = float(buffer_cap_s)
        if not all(0.0 < value < float("inf") for value in (self.stride_s, self.buffer_cap_s)):
            raise ValueError("Stride and buffer capacity must be finite and positive.")
        self.max_text_tokens = int(max_text_tokens)
        self.tau_dec = float(tau_dec)

        # Λ_min (min_span_frames): shortest selectable span in encoder frames — a LABEL-domain floor that rejects flicker; 
        # genuine shorter units cannot be emitted. Not the re-emission guard (candidate_spans(skip_term_before=χ) is).
        self.min_span_frames = int(min_span_frames) if min_span_frames is not None else LAMBDA_MIN_FRAMES
        if self.min_span_frames < 1: raise ValueError(f"min_span_frames ({self.min_span_frames}) must be >= 1")
        # translate=False: segmentation-only dry run, no decoder call. Scores FSM's segmentation alone in minutes; events carry empty text.
        self.translate = bool(translate)

        self.spd_top_k = int(spd_top_k)
        self.spd_renormalize = bool(spd_renormalize)
        self.commit_gate = CommitGate(delta_enc_frames=delta_enc_frames, hysteresis_strides=hysteresis_strides)
        # Membership gate: per stride, Ω from the BIO posteriors (on-policy, no GT) + χ from commit log — decoder's training conditioning. 
        self.gate_enabled = bool(gate_enabled)
        self.gate_eps = float(gate_eps)
        self._committed_until_s = 0.0 # χ frontier: abs end of last committed span. Earlier buffer frames already emitted (≤δ overlap kept).
        # Did the LAST commit end at a real BIO terminator (O-or-B) rather than a cap cut? Only that certifies "a sentence
        # ENDED here", the χ-restoration's premise. Persistent: the seam is re-examined while the leftover lives.
        self._committed_is_terminator = False
        # Per-stride candidate hypotheses for offline stable-prefix analysis; None = disabled (the default).
        self.trace: list[StrideHypothesis] | None = [] if record_trace else None
        # Why-did-it-(not)-commit counters across run() calls. Low streaming recall with near-perfect frame BIO is usually the gate:
        # boundary_ok (of spans_seen) is how often target's terminator was stable; forced_commit (of committed) how often the cap or EOF cut.
        self.gate_stats: dict[str, int] = {}
        self._bio_timeline: torch.Tensor | None = None  # per-stream stitched decoded BIO tags; set by run()

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
        tokens, confidence = decoder.generate_from_bio_tap(
            bio_tap, mask, max_text_tokens=self.max_text_tokens, tau_dec=self.tau_dec, spd_top_k=self.spd_top_k,
            spd_renormalize=self.spd_renormalize, omega_bias=omega_bias,
        )
        # DLM strips its synthetic BOS internally; the AR arm returns it raw (the training replay needs the start slot) — 
        # drop it here so the event's confidences cover only produced tokens, like the DLM arm.
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
        force: bool = False, emission_time_s: float | None = None, vote: bool = True,
    ) -> StreamingEvent | None:
        """1 sawtooth pass over raw (T,133,3) poses in [last_commit_t, end_s).

        end_s is the exclusive evidence endpoint. emission_time_s defaults to it, but can advance during an EOF drain.
        Process at the capacity deadline before admitting more frames. force=True overrides unmet commit conditions at EOF.
        vote=False marks a pass that only reads votes: a cap deadline between strides, a post-commit re-check, the EOF drain.
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

        # Normalize only the active buffer, as training does. Future and retired frames cannot set its body scale.
        device = next(self.model.parameters()).device
        raw_buffer = poses[in_buffer].detach().float().cpu().numpy()
        if self.cascade_model is not None: self._raw_buffer = raw_buffer
        # The external segmenter normalises the buffer ITSELF, its own way (moryossef26.dataset.to_release_coords),
        # so it is handed the raw keypoints; every in-system path takes the Uni-Sign normalisation.
        buffer_np = raw_buffer if getattr(self.model, "wants_raw_poses", False) else normalize_keypoints_unisign(raw_buffer)
        poses_b = torch.from_numpy(buffer_np).unsqueeze(0).to(device)
        ts_b = (timestamps_s[in_buffer] - start_s).unsqueeze(0).to(device)
        mask_b = torch.ones(poses_b.shape[:2], dtype=torch.bool, device=device)

        # Segmentation branch and translator features read the same buffer (separate pose encoders in the joint model).
        bio_output = self.model.segment(poses_b, mask_b, ts_b)
        bio_tap, mask, ts = self.model.front_end.extract_bio_tap(poses_b, mask_b, ts_b)
        bio_logits = bio_output.logits
        chi = ts + float(start_s) < self._committed_until_s
        bio_tags = DurationDecoder(getattr(self.model, "duration_model", None)).decode(
            bio_logits, mask.long().sum(1), chi, known_start=self._committed_is_terminator, timestamps_s=ts
        )[0]
        # Stitch this stride's Viterbi tags into the whole-stream timeline
        # (latest estimate wins on revisits) so RQ2 can score the segmentation the FSM actually acted on.
        if self._bio_timeline is not None and bio_tags.numel() == int(in_buffer.sum().item()):
            self._bio_timeline[in_buffer.cpu()] = bio_tags.detach().cpu()
        omega_bias = self._stride_omega(bio_logits, mask, ts_b, start_s)

        # χ filter: spans fully inside emitted content are never selectable (see candidate_spans).
        # Same absolute timeline as _stride_omega's commit_mask.
        chi_frames = int(((ts_b[0] + float(start_s)) < self._committed_until_s).sum().item())
        spans = list(candidate_spans(bio_tags, self.min_span_frames, skip_term_before=chi_frames))
        frame_ids = in_buffer.nonzero().squeeze(1).cpu()  # buffer index -> absolute stream frame: votes survive the cut at a commit
        absolute = lambda span: (int(frame_ids[span[0]]), int(frame_ids[span[1]]))
        if vote: self.commit_gate.vote([absolute(span) for span in spans])  # 1 vote per candidate span per stride
        if not spans:
            if not buffer_full: return None

            # Forced commit at cap: decode the in-progress (right-truncated) span if any, else drop the gap/fragment buffer.
            b_idx = open_span_start(bio_tags)  # same open-span rule as the gate anchor
            self.commit_gate.reset()
            if b_idx is None: return None
            self._bump("committed"); self._bump("forced_commit")

            s_idx, last_idx = b_idx, int(bio_tags.numel()) - 1
            tokens, confidence = self._decode_span(bio_tap, mask, slice(s_idx, last_idx + 1), omega_bias=omega_bias)
            # The exclusive evidence endpoint also handles a cap between frame timestamps and a one-frame EOF tail.
            return StreamingEvent(
                start_s=float(start_s + ts_b[0, s_idx].item()), end_s=float(end_s),
                token_ids=tokens[0].detach().cpu(), token_confidence=confidence[0].detach().cpu(),
                flagged_partial=True, commit_time_s=emission_time_s, terminator_commit=False,  # cap cut mid-sentence.
            )

        s_idx, term_idx = spans[0]  # the target: first candidate span
        # Every terminator, O or B, commits only after K stable strides: the hysteresis is the one right-context requirement.
        # A cap or drain pass adds no vote: a target it finds unstable emits anyway, flagged PARTIAL.
        stable = self.commit_gate.stable(absolute(spans[0]))
        self._bump("spans_seen")
        if stable: self._bump("boundary_ok")
        forced = buffer_full and not stable
        # Decode only where the result is read: a commit, a cap-forced emission, or stability trace (which keeps deferred hypotheses).
        if not stable and not buffer_full and self.trace is None: return None
        # The crop holds the terminator frame when the buffer has it, as the offline cascade's [start, end + 1 frame] does.
        tokens, confidence = self._decode_span(bio_tap, mask, slice(s_idx, max(s_idx + 1, term_idx + 1)), omega_bias=omega_bias)
        self._bump("gated_decoded")

        # Stability trace: the hypothesis for one sentence as it evolves with arriving frames — the input a stable-prefix policy replays 
        # offline (infer/stability.py). It is the one consumer of strides the gate defers, so it also turns the skip above off and pays 
        # for their decodes. Off by default; never affects the FSM.
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
                token_ids=tok_row.clone(), token_confidence=conf_row.clone(), committed=stable or forced,
            ))
        # A complete cap/drain event cuts at its terminator, retaining all later buffered sentences and their votes.
        if not (stable or forced): return None
        self._bump("committed")
        if forced: self._bump("forced_commit")
        self.commit_gate.retire(absolute(spans[0]))
        return StreamingEvent(
            start_s=float(start_s + ts_b[0, s_idx].item()), end_s=float(start_s + ts_b[0, term_idx].item()),
            token_ids=tokens[0].detach().cpu(), token_confidence=confidence[0].detach().cpu(),
            flagged_partial=forced, commit_time_s=emission_time_s, terminator_commit=True,  # complete span: the cut IS its terminator.
        )

    @torch.no_grad()
    def run(self, poses: torch.Tensor, fps: float) -> list[StreamingEvent]:
        if not 0.0 < float(fps) < float("inf"): raise ValueError("FPS must be finite and positive.")
        timestamps = torch.arange(poses.shape[0], device=poses.device, dtype=torch.float32) / float(fps)
        events: list[StreamingEvent] = []
        duration = poses.shape[0] / float(fps)  # exclusive endpoint, including the final frame
        # Overlap cut: keep the last δ frames at a committed terminator, so a terminator estimate late by up to δ still
        # leaves the next onset (its B) in the buffer — else buffer-start I never opens a span (Mode-2b silence) and that
        # sentence is dropped. The leftover is never re-emitted: candidate_spans skips any span terminating at or before χ.
        overlap_s = float(self.commit_gate.delta_enc_frames) / float(fps)
        frame_s = 1.0 / float(fps)
        last_commit_t = 0.0
        end_s, next_stride_s = 0.0, self.stride_s

        self.commit_gate.reset()
        self._committed_until_s = 0.0  # χ commit log resets per stream
        self._committed_is_terminator = False
        self._bio_timeline = torch.full((poses.shape[0],), int(BIO["UNK"]), dtype=torch.long)  # stitched tags
        self._fps = float(fps)

        def absorb(event) -> None:
            nonlocal last_commit_t
            events.append(event)
            # The ≤δ overlap the next buffer keeps must be attention-floored by membership gate 
            # (seam-duplication guard, docs/membership_gate.md §2.4 (committed prefix) / §2.5 (log eps floor)).
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
            # Only the pass that ends a stride votes: a cap deadline between strides adds none, so a span still needs K strides.
            event = self.step(poses, timestamps, end_s=end_s, last_commit_t=last_commit_t, vote=end_s >= min(next_stride_s, duration))
            if event is None and end_s >= cap_end_s: # This gap/fragment buffer was processed at capacity with nothing to emit.
                last_commit_t = max(end_s - overlap_s, min(end_s, last_commit_t + frame_s))
                self._committed_is_terminator = False  # a capacity discard does not certify a new onset
            # After a commit, segment the post-commit buffer again in this stride (new normalization, χ, known_start, Ω). The re-check
            # adds no vote, so the next span commits now only if its own history is already stable; repeat until nothing commits.
            while event is not None:
                absorb(event)
                event = self.step(poses, timestamps, end_s=end_s, last_commit_t=last_commit_t, vote=False)
            if end_s >= next_stride_s: next_stride_s += self.stride_s

        # EOF cannot supply missing boundary votes or right context. Flush the remaining spans explicitly;
        # a span without an already stable history is forced/partial.
        emission_time_s = next_stride_s
        # Loop the forced pass: 1 pass emits 1 span, else a frozen buffer with several pending sentences drops all but the first. Only 
        # the emission clock advances by 1 stride/pass; the evidence endpoint remains fixed, so clock ticks cannot supply stability votes.
        while True:
            event = self.step(
                poses, timestamps, end_s=duration, last_commit_t=last_commit_t, force=True, emission_time_s=emission_time_s, vote=False
            )
            if event is None: break
            absorb(event)
            emission_time_s += self.stride_s
        return events
