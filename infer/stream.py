from __future__ import annotations
from dataclasses import dataclass

import torch
from data.windowing import BIO
from infer.commit_gate import CommitGate, open_span_start, select_target_span
from infer.duration_decode import duration_split_tags


def leading_i_run_end(bio_tags: torch.Tensor) -> int | None:
    # 1st O-or-B after a buffer-start I-run (the continuation of a force-cut sentence).
    # None if the buffer does not start with I, or the run reaches the buffer end.
    tags = bio_tags.tolist()
    if not tags or tags[0] != BIO["I"]: return None
    for idx, tag in enumerate(tags):
        if tag in (BIO["O"], BIO["B"]): return idx
    return None


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
    # Cut landed on a certified BIO terminator (any complete span, forced or not); false only on the mid-sentence
    # open-span cap path. Not `not flagged_partial`: that also holds for forced COMPLETE spans, and gating
    # χ-restoration on it drops alternate sentences under cap pressure.
    terminator_commit: bool = False


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
        dcd_top_p: float | None = None, dcd_cache_type: str = "none", 
        decode_conditioning: str = "window", min_span_frames: int | None = None, forced_tail_policy: str = "skip",
        gate_enabled: bool = False, gate_delta: int | None = None, gate_eps: float = 1e-4, duration_prior=None,
    ):
        # decode_conditioning: "window" (default) decodes under the FULL buffer — the training conditioning (Mode 1/3 feed
        # the whole jittered window; right-context is learned to be disregarded, not cropped). "span" crops to the BIO span,
        # unseen by training except at zero jitter — ablation. Emitted boundaries come from the BIO span either way.
        if decode_conditioning not in {"window", "span"}: raise ValueError(f"Unknown decode_conditioning: {decode_conditioning}")

        # forced_tail_policy — continuation of a sentence cut by a FORCED (cap) commit:
        #   "skip" (default): never selected; Mode-2b gives no supervision on that left-truncated conditioning, so the
        #     decode is undefined behaviour. (Supervision covers the states the FSM decodes with one known exception:
        #     anchors longer than buffer_cap_s train without a CB view — a full-evidence self-target would itself be
        #     truncated — yet the cap-forced commit decodes exactly that state, flagged PARTIAL. ~0.7% of asf steps.)
        #   "translate_partial": decode it flagged PARTIAL at its terminator — best-effort OOD recovery. Ablation.
        if forced_tail_policy not in {"skip", "translate_partial"}: 
            raise ValueError(f"Unknown forced_tail_policy: {forced_tail_policy}")

        self.decode_conditioning = str(decode_conditioning)
        self.forced_tail_policy = str(forced_tail_policy)
        self.model = model
        self.stride_s = float(stride_s)
        self.buffer_cap_s = float(buffer_cap_s)
        self.max_text_tokens = int(max_text_tokens)
        self.diffusion_steps = int(diffusion_steps)
        self.tau_dec = float(tau_dec)

        # Λ_min (min_span_frames): shortest selectable span in encoder frames — a duration noise floor; spans below δ are unresolvable 
        # from boundary evidence. Default δ+1, else max(δ+1, p1-p2 of dev sentence lengths); 0 re-admits 1-frame flicker. Not the 
        # re-emission guard (select_target_span(skip_term_before=χ) is), and not the old "Λ_min > 2δ" bound: 2δ < Λ_min < shortest real 
        # sentence is infeasible on short corpora, and the duration re-split only disfavors sub-second seam segments (~4-5 nats).
        self.min_span_frames = int(min_span_frames) if min_span_frames is not None else int(delta_enc_frames) + 1
        if self.min_span_frames < 1: raise ValueError(f"min_span_frames ({self.min_span_frames}) must be >= 1")
        # Optional semi-Markov duration decode (infer.duration_decode, inference.yaml duration_decode): injects interior B
        # splits so back-to-back sentences become separate selectable spans; the run touching buffer end uses the
        # right-censored "survival" rule. Onsets stay unmarked, leaving span-opening (O→signing) and forced commits untouched.
        self.duration_prior = duration_prior
        # build_gate_omega re-splits tags from the model's own duration_prior — sync unconditionally (None must clear too,
        # else a model left by an earlier duration-enabled runner gates on re-split tags while this FSM decodes raw argmax).
        # Either mismatch is train/infer divergence.
        if hasattr(self.model, "duration_prior"): self.model.duration_prior = duration_prior

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
            delta_enc_frames=delta_enc_frames,
            hysteresis_strides=hysteresis_strides,
            token_confidence_tau=token_confidence_tau,
        )
        # Membership gate (docs/membership_gate.md): per stride, Ω from the BIO posteriors (on-policy, no GT) + χ from the
        # commit log — the decoder's training conditioning. δ defaults to delta_enc_frames (1 constant, 3 roles).
        # 'window' only: 'span' crops the pose axis and misaligns Ω columns.
        self.gate_enabled = bool(gate_enabled) and self.decode_conditioning == "window"
        self.gate_delta = int(gate_delta if gate_delta is not None else delta_enc_frames)
        self.gate_eps = float(gate_eps)
        # χ frontier: absolute end of the last committed span. Earlier buffer frames are already emitted (≤δ overlap kept).
        self._committed_until_s = 0.0
        # True between a cap commit that cut an open span and that sentence's continuation resolving.
        # Only consulted by forced_tail_policy="translate_partial".
        self._after_forced = False
        # Did the LAST commit end at a real BIO terminator (O-or-B) rather than a cap cut? Only that certifies "a sentence
        # ENDED here", the χ-restoration's premise. Persistent: the seam is re-examined while the leftover lives.
        self._committed_is_terminator = False
        # Why-did-it-(not)-commit counters across run() calls. Low streaming recall with near-perfect frame BIO is usually
        # the gate; spans_seen vs boundary_ok vs translation_ok says which signal blocks (usually translation_ok).
        self.gate_stats: dict[str, int] = {}
        self._bio_timeline: torch.Tensor | None = None  # per-stream stitched BIO argmax; set by run()

    def _bump(self, key: str, n: int = 1) -> None:
        self.gate_stats[key] = self.gate_stats.get(key, 0) + int(n)

    def _decode_span(self, bio_tap: torch.Tensor, mask: torch.Tensor, span_slice: slice, omega_bias: torch.Tensor | None = None):
        if self.decode_conditioning == "span": bio_tap, mask = bio_tap[:, span_slice], mask[:, span_slice]
        tokens, confidence = self.model.generate_from_bio_tap(
            bio_tap, mask, max_text_tokens=self.max_text_tokens, diffusion_steps=self.diffusion_steps, tau_dec=self.tau_dec,
            spd_top_k=self.spd_top_k, spd_renormalize=self.spd_renormalize, spd_revision=self.spd_revision, temperature=self.temperature,
            dcd_window_length=self.dcd_window_length, dcd_max_window_length=self.dcd_max_window_length, dcd_window_type=self.dcd_window_type,
            dcd_decode_algo=self.dcd_decode_algo, dcd_decode_param=self.dcd_decode_param, dcd_sample_top_k=self.dcd_sample_top_k,
            dcd_top_p=self.dcd_top_p, dcd_cache_type=self.dcd_cache_type, omega_bias=omega_bias,
        )
        # DLM strips its synthetic BOS internally; the AR arm returns it raw (the training replay needs the start slot) — 
        # drop it here so the commit gate's confidence mean covers only produced tokens, like the DLM arm.
        if getattr(self.model, "decoder_type", "dlm") == "ar": tokens, confidence = tokens[:, 1:], confidence[:, 1:]
        return tokens, confidence

    def _stride_omega(self, bio_logits: torch.Tensor, mask: torch.Tensor, ts_b: torch.Tensor, start_s: float):
        # None when the gate is off or the model has no gate builder (test fakes / AR-only models).
        if not self.gate_enabled or not hasattr(self.model, "build_gate_omega"): return None
        chi = (ts_b + float(start_s)) < self._committed_until_s  # (1, T') bool, absolute timeline
        omega_bias, _ = self.model.build_gate_omega(
            bio_logits, None, mask, memory_len=self.model.front_end.prompt_length() + int(bio_logits.shape[1]),
            commit_mask=chi, delta=self.gate_delta, eps=self.gate_eps, min_span_frames=self.min_span_frames, timestamps_s=ts_b,
            # ONE seam rule with the FSM: the gate's χ-onset restoration fires only on a certified-terminator last commit,
            # else Ω anchors on a mid-sentence fragment the FSM's skip policy never decodes.
            seam_is_terminator=self._committed_is_terminator,
            # Same rule for the other mint: step() treats a leading I on the FIRST buffer as a real onset, so the gate
            # must too — otherwise it anchors Ω on "no span" and floors the frames the FSM is about to decode.
            stream_start=(start_s <= 0.0 and self._committed_until_s <= 0.0),
        )
        return omega_bias

    @torch.no_grad()
    def step(
        self, poses: torch.Tensor, timestamps_s: torch.Tensor, end_s: float, last_commit_t: float = 0.0, force: bool = False
    ) -> StreamingEvent | None:
        """One sawtooth stride over the buffer [last_commit_t, end_s).

        The buffer grows from the last commit cut, not a fixed trailing window, so a committed sentence is never re-emitted.
        BUFFER_CAP_S forces a PARTIAL commit to bound latency; `force=True` applies that policy at any buffer size — the
        end-of-stream drain (frozen evidence == cap reached).
        """
        start_s = float(last_commit_t)
        in_buffer = (timestamps_s >= start_s) & (timestamps_s < end_s)
        if not in_buffer.any(): return None

        poses_b = poses[in_buffer].unsqueeze(0)
        ts_b = (timestamps_s[in_buffer] - start_s).unsqueeze(0)
        mask_b = torch.ones(poses_b.shape[:2], dtype=torch.bool, device=poses.device)

        bio_tap, mask, ts = self.model.front_end.extract_bio_tap(poses_b, mask_b, ts_b)
        bio_logits = self.model.bio_head(bio_tap, timestamps_s=ts, frame_mask=mask).logits
        bio_tags = bio_logits.argmax(dim=-1)[0]
        # UNK closes like O (shared decode rule, close_on_unk=True). UNK is supervision-free (ignore_index) so argmax-UNK is
        # rare, but one UNK frame in a gap would leave a span terminator-less (commit deferred to the cap → spurious PARTIAL)
        # and O→UNK→I would never open (_span_opens needs prev==O exactly).
        bio_tags = torch.where(bio_tags == BIO["UNK"], torch.full_like(bio_tags, BIO["O"]), bio_tags)
        if self.duration_prior is not None and bio_tags.numel() > 2:
            dt = ts[0, 1:] - ts[0, :-1]
            # Clamp fps to [1,120]: a degenerate ~0 median dt gives fps ~1e6 -> lmax OOM in the DP.
            fps_b = min(max(1.0 / max(float(dt.median().item()), 1e-6), 1.0), 120.0) if dt.numel() else 24.0
            pB = torch.softmax(bio_logits[0].float(), dim=-1)[:, BIO["B"]].cpu().numpy()
            # split_open_tail="survival": the run touching buffer end is right-censored, so its LAST segment scores by
            # lognormal log-survival while the prefix still gets interior splits — else a b2b stretch longer than the buffer
            # stays unsplit forever. The censored tail stays open, so premature-commit protection is unchanged.
            bio_tags = torch.as_tensor(
                duration_split_tags(bio_tags.cpu().numpy(), pB, fps_b, self.duration_prior, mark_onsets=False, split_open_tail="survival"),
                device=bio_tags.device, dtype=bio_tags.dtype,
            )
            # χ-boundary onset restoration (alternate-sentence-drop guard). The re-split folds all B's to I and never splits at the ≤δ 
            # leftover seam → a back-to-back successor's onset lands in buffer-start I, which _span_opens cannot open → emits sentences 
            # k, k+2, k+4, ... χ certifies a sentence ENDED there, so the 1st mid-run signing frame at-or-after χ IS that onset: mark it B 
            # (no-op if the seam is in a gap). Only when `_committed_is_terminator`: a cap cut is mid-sentence, where a B fabricates a 
            # boundary, re-opens the fragment "skip" drops, and re-emits committed frames. At stream start frame 0 IS a real onset, else 
            # a mid-signing stream never opens a span.
            if start_s <= 0.0 and self._committed_until_s <= 0.0 and bio_tags.numel():
                if int(bio_tags[0].item()) == BIO["I"]: bio_tags[0] = BIO["B"]
            if self._committed_until_s > 0.0 and self._committed_is_terminator:
                abs_t = ts_b[0] + float(start_s)
                signing = (bio_tags == BIO["I"]) | (bio_tags == BIO["B"])
                after = ((abs_t >= self._committed_until_s) & signing).nonzero().flatten()
                if after.numel():
                    d = int(after[0].item())
                    if bio_tags[d] == BIO["I"] and (d == 0 or bool(signing[d - 1])): bio_tags[d] = BIO["B"]

        # Stitch this stride's argmax into the whole-stream timeline (latest estimate wins on revisits) so RQ2 can score
        # the deployed head's segmentation directly.
        if self._bio_timeline is not None and bio_tags.numel() == int(in_buffer.sum().item()):
            self._bio_timeline[in_buffer.cpu()] = bio_tags.detach().cpu()
        buffer_full = force or (end_s - start_s) >= self.buffer_cap_s
        omega_bias = self._stride_omega(bio_logits, mask, ts_b, start_s)

        # after_forced resolution (translate_partial only). Auto-reset if the buffer does not start with I: no continuation
        # is coming (flush landed in a gap), and waiting on a lead that never appears deadlocks the FSM.
        if self._after_forced and self.forced_tail_policy == "translate_partial":
            lead_term = leading_i_run_end(bio_tags)
            if bio_tags.numel() and int(bio_tags[0].item()) != BIO["I"]: self._after_forced = False
            elif lead_term is not None:
                self._after_forced = False
                if lead_term >= max(1, self.min_span_frames):
                    self._bump("committed"); self._bump("forced_tail_commit")
                    # Ungated: the stride's Ω is anchored where the buffer-start I-run can never open, so it would floor the
                    # very frames [0, lead_term) being decoded. Best-effort fragment decode anyway.
                    tokens, confidence = self._decode_span(bio_tap, mask, slice(0, lead_term), omega_bias=None)
                    self.commit_gate.reset()
                    return StreamingEvent(
                        start_s=float(start_s + ts_b[0, 0].item()), end_s=float(start_s + ts_b[0, lead_term].item()),
                        token_ids=tokens[0].detach().cpu(), token_confidence=confidence[0].detach().cpu(),
                        bio_start_index=0, bio_end_index=lead_term, flagged_partial=True, commit_time_s=float(end_s),
                        terminator_commit=True,  # lead_term IS the first O-or-B: a certified terminator
                    )
        elif self._after_forced: self._after_forced = False  # "skip": leftover I-run is never selected

        # χ filter: spans fully inside emitted content are never selectable (see select_target_span). Same absolute
        # timeline as _stride_omega's commit_mask.
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
            # END-TIME CONVENTION: end_s is EXCLUSIVE — χ is a strict `<` frontier here, in `_stride_omega`, and in
            # `deployed_decode_tags`. Complete spans get this free; this open-span path ends at the LAST INCLUDED frame, so
            # add one frame step, else that frame reads uncommitted and the drain re-emits it as zero-duration duplicates.
            step_s = float(ts_b[0, last_idx] - ts_b[0, last_idx - 1]) if last_idx > 0 else 1.0 / 24.0
            return StreamingEvent(
                start_s=float(start_s + ts_b[0, s_idx].item()), end_s=float(start_s + ts_b[0, last_idx].item() + step_s),
                token_ids=tokens[0].detach().cpu(), token_confidence=confidence[0].detach().cpu(),
                bio_start_index=s_idx, bio_end_index=last_idx, flagged_partial=True, commit_time_s=float(end_s),
                terminator_commit=False,  # cap cut mid-sentence: the seam is a continuation, never an onset
            )

        s_idx, term_idx = span
        tokens, confidence = self._decode_span(bio_tap, mask, slice(s_idx, max(s_idx + 1, term_idx)), omega_bias=omega_bias)
        decision = self.commit_gate.update(span, token_confidence=confidence[0])
        self._bump("spans_seen")
        if decision.boundary_stable: self._bump("boundary_ok")
        if decision.translation_confident: self._bump("translation_ok")
        # Deviates from the spec's flush-at-cap (cut at current_t − δ): a COMPLETE span forced here cuts at ITS terminator
        # − δ (event.end_s → run()'s overlap cut), not at buffer end. Flushing to the cap would drop later sentences already 
        # buffered; cutting at terminator keeps them and the buffer falls below the cap in a stride or 2 — same latency bound.
        forced = buffer_full and not decision.should_commit
        if not decision.should_commit and not forced: return None

        self._bump("committed")
        if forced: self._bump("forced_commit")
        self.commit_gate.reset()
        return StreamingEvent(
            start_s=float(start_s + ts_b[0, s_idx].item()), end_s=float(start_s + ts_b[0, term_idx].item()),
            token_ids=tokens[0].detach().cpu(), token_confidence=confidence[0].detach().cpu(),
            bio_start_index=s_idx, bio_end_index=term_idx, flagged_partial=forced, commit_time_s=float(end_s),
            terminator_commit=True,  # complete span: the cut IS its terminator, forced or not
        )

    @torch.no_grad()
    def run(self, poses: torch.Tensor, fps: float) -> list[StreamingEvent]:
        device = next(self.model.parameters()).device
        poses = poses.to(device)
        timestamps = torch.arange(poses.shape[0], device=device, dtype=torch.float32) / float(fps)
        events: list[StreamingEvent] = []
        duration = float(timestamps[-1].item()) if timestamps.numel() else 0.0
        # Overlap cut: keep the last δ frames at a committed terminator, so a terminator estimate late by up to δ still
        # leaves the next onset (its B) in the buffer — else buffer-start I never opens a span (Mode-2b silence) and that
        # sentence is dropped. The leftover is never re-emitted: select_target_span skips any span terminating at or before χ.
        overlap_s = float(self.commit_gate.history.delta_enc_frames) / float(fps)
        frame_s = 1.0 / float(fps)
        last_commit_t = 0.0
        end_s = self.stride_s
        self._after_forced = False
        self._committed_until_s = 0.0  # χ commit log resets per stream
        self._committed_is_terminator = False
        self._bio_timeline = torch.full((poses.shape[0],), int(BIO["UNK"]), dtype=torch.long)  # stitched tags

        def absorb(event: StreamingEvent) -> None:
            nonlocal last_commit_t
            events.append(event)
            # The ≤δ overlap the next buffer keeps must be attention-floored by membership gate (seam-duplication guard §2.7).
            self._committed_until_s = max(self._committed_until_s, float(event.end_s))
            # χ-restoration premise: `terminator_commit`, never `not flagged_partial` (see StreamingEvent).
            self._committed_is_terminator = bool(event.terminator_commit)
            # Advance ≥1 frame so a degenerate (≤δ) span cannot stall the stream.
            last_commit_t = max(event.end_s - overlap_s, last_commit_t + frame_s)

        while end_s <= duration + self.stride_s:
            event = self.step(poses, timestamps, end_s=end_s, last_commit_t=last_commit_t)
            if event is not None: absorb(event)
            elif (end_s - last_commit_t) >= self.buffer_cap_s: # gap/fragment buffer hit the cap with nothing to emit
                last_commit_t = max(end_s - overlap_s, last_commit_t + frame_s)
            end_s += self.stride_s

        # End-of-stream drain. Evidence is frozen at `duration`, but the K-stride hysteresis still needs votes: a terminator
        # first visible in the last K−1 strides, or a span awaiting its confidence gate, would be dropped — depressing
        # streaming recall. Re-votes on the identical frozen buffer make a real terminator stable by construction, so it
        # commits un-flagged. After K quiet strides, FORCED passes apply the buffer-cap policy (same frozen-evidence
        # condition), draining pending spans as PARTIAL.
        quiet = 0
        while quiet < int(self.commit_gate.history.hysteresis_strides):
            event = self.step(poses, timestamps, end_s=end_s, last_commit_t=last_commit_t)
            if event is not None: absorb(event); quiet = 0
            else: quiet += 1
            end_s += self.stride_s
        # Loop the forced pass: 1 pass emits 1 span, else a frozen buffer with several pending sentences drops all but the 
        # first. absorb advances last_commit_t by ≥ frame_s per pass, so this ends at a sub-Λ_min leftover. Advance end_s 
        # too — 1 stride tick/pass, else drained spans share a frozen commit_time_s & latency stats mix 2 clock conventions.
        while True:
            event = self.step(poses, timestamps, end_s=end_s, last_commit_t=last_commit_t, force=True)
            if event is None: break
            absorb(event)
            end_s += self.stride_s
        return events
