from __future__ import annotations
from dataclasses import dataclass

import torch
from data.windowing import BIO
from infer.commit_gate import CommitGate, open_span_start, select_target_span
from infer.duration_decode import duration_split_tags


def leading_i_run_end(bio_tags: torch.Tensor) -> int | None:
    # Terminator index of a buffer-start I-run (the continuation of a force-cut sentence): the first O-or-B
    # after the leading I frames. None if the buffer does not start with I, or the run reaches the buffer end.
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
    commit_time_s: float = 0.0  # stride wall-time the commit fired; for emission-latency vs GT end


class StreamingSLTRunner:
    """Model-backed sawtooth inference runner.

    Decoder state is not stored on the runner. Each stride recomputes visual features/BIO for the current buffer
    and cold-starts translation for the selected target span (first complete span ≥ Λ_min).
    """
    def __init__(
        self, model, stride_s: float = 1.0, buffer_cap_s: float = 18.0, delta_enc_frames: int = 3, hysteresis_strides: int = 3,
        token_confidence_tau: float = 0.75, max_text_tokens: int = 128, diffusion_steps: int = 64, tau_dec: float = 0.75,
        spd_top_k: int = 1, spd_renormalize: bool = True, spd_revision: bool = True, temperature: float = 0.0,
        dcd_window_length: int | None = None, dcd_max_window_length: int | None = None, dcd_window_type: str = "sliding",
        dcd_decode_algo: str = "threshold", dcd_decode_param: int | float | None = None, dcd_sample_top_k: int | None = None,
        dcd_top_p: float | None = None, dcd_cache_type: str = "none", dcd_refresh_count: int = 16,
        decode_conditioning: str = "window", min_span_frames: int | None = None, forced_tail_policy: str = "skip",
        gate_enabled: bool = False, gate_delta: int | None = None, gate_eps: float = 1e-4, duration_prior=None,
    ):
        # decode_conditioning: "window" (default) decodes under the FULL buffer's features — exactly the
        # training conditioning, where Mode 1/3 windows feed the whole jittered window to the encoder and
        # OPUT supervises the first-complete-span text under it (§5.1/§5.3: right-context is *learned to be
        # disregarded*, not cropped away). "span" crops conditioning to the predicted BIO span (the spec §7.1
        # pseudocode's `enc[span]`), which the trained conditional never sees except at the zero-jitter corner
        # of the CDF — kept as an ablation. The BIO span still defines the emitted boundaries either way.
        if decode_conditioning not in {"window", "span"}: 
            raise ValueError(f"Unknown decode_conditioning: {decode_conditioning}")

        # forced_tail_policy — what to do with the continuation of a sentence cut by a FORCED (cap) commit:
        #   "skip" (default): never translate it. The continuation is a left-truncated fragment; 
        #     Mode-2b training gives the decoder NO translation supervision on that conditioning, 
        #     so its decode is undefined behaviour. The leftover I-run at buffer start is simply never selected.
        #   "translate_partial": emit the fragment's decode flagged PARTIAL once its terminator appears —
        #     best-effort recovery of the sentence ending, at the cost of decoding out-of-training-distribution
        #     conditioning. Kept as an option/ablation, not the default.
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

        # Λ_min (min_span_frames): shortest selectable span, in encoder frames. Λ_min > 2δ, so ≤2δ post-commit 
        # overlap leftover can never be selected even if mislabelled B (commit_gate.select_target_span; the 
        # no-re-emission argument depends on it). None → the minimal safe value 2δ+1. There is deliberately NO 
        # disable value: the old "0 disables" escape hatch silently re-opened the duplicate-re-emission hole 
        # whenever a config lacked span_selection.
        self.min_span_frames = int(min_span_frames) if min_span_frames is not None else 2 * int(delta_enc_frames) + 1
        if self.min_span_frames <= 2 * int(delta_enc_frames): raise ValueError(
                f"min_span_frames ({self.min_span_frames}) must exceed 2*delta_enc_frames ({2 * int(delta_enc_frames)}); "
                "otherwise the post-commit overlap leftover is a selectable span."
            )
        # Optional buffer-level semi-Markov duration decode (infer.duration_decode, inference.yaml duration_decode).
        # Injects interior B splits into CLOSED signing runs of each stride's tags — back-to-back sentences inside
        # the buffer become separate selectable spans instead of one merged run. The run touching the buffer end
        # is left unsplit (right-truncated: its total duration is unknown mid-stream) and onsets are not re-marked
        # B (opening keys on the O→signing transition, so span-opening and the forced-commit path are untouched).
        # Opt-in until validated end-to-end in RQ2 (whole-video eval: asf dev tiou F1@0.5 0.19->0.51).
        self.duration_prior = duration_prior
        # The membership gate's anchor selection re-splits tags INSIDE build_gate_omega from the model's own
        # duration_prior — hand it the same prior, or _stride_omega would gate on raw-argmax merged runs while the
        # FSM above commits duration-split spans (the exact train/infer divergence the injection exists to prevent).
        if duration_prior is not None and hasattr(self.model, "duration_prior"):
            self.model.duration_prior = duration_prior

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
        self.dcd_refresh_count = int(dcd_refresh_count)
        self.commit_gate = CommitGate(
            delta_enc_frames=delta_enc_frames,
            hysteresis_strides=hysteresis_strides,
            token_confidence_tau=token_confidence_tau,
        )
        # Membership gate at streaming inference (docs/membership_gate.md): per stride, build Ω from the BIO posteriors (on-policy span, 
        # no GT) + χ from the runner's own commit log — the same conditioning the decoder trained under. δ defaults to the commit gate's 
        # delta_enc_frames (1 constant, 3 roles). Only under 'window' conditioning: 'span' crops the pose axis and would misalign Ω columns.
        self.gate_enabled = bool(gate_enabled) and self.decode_conditioning == "window"
        self.gate_delta = int(gate_delta if gate_delta is not None else delta_enc_frames)
        self.gate_eps = float(gate_eps)
        # Absolute end-time of the last committed span: buffer frames earlier than this are χ=1 (already emitted — 
        # the ≤δ overlap leftovers the cut deliberately keeps).
        self._committed_until_s = 0.0
        # True between a forced (cap) commit that cut through an open span and the resolution of that
        # sentence's continuation. Only consulted by forced_tail_policy="translate_partial".
        self._after_forced = False
        # Why-did-it-(not)-commit counters, accumulated across all run() calls. A low streaming recall with
        # near-perfect frame BIO is almost always the gate suppressing emission; comparing spans_seen vs
        # boundary_ok vs translation_ok pinpoints which signal blocks (usually translation_ok on a weak model).
        self.gate_stats: dict[str, int] = {}
        self._bio_timeline: torch.Tensor | None = None  # per-stream stitched BIO argmax; set by run()

    def _bump(self, key: str, n: int = 1) -> None:
        self.gate_stats[key] = self.gate_stats.get(key, 0) + int(n)

    def _decode_span(self, bio_tap: torch.Tensor, mask: torch.Tensor, span_slice: slice, omega_bias: torch.Tensor | None = None):
        if self.decode_conditioning == "span": bio_tap, mask = bio_tap[:, span_slice], mask[:, span_slice]
        return self.model.generate_from_bio_tap(
            bio_tap, mask, max_text_tokens=self.max_text_tokens, diffusion_steps=self.diffusion_steps, tau_dec=self.tau_dec,
            spd_top_k=self.spd_top_k, spd_renormalize=self.spd_renormalize, spd_revision=self.spd_revision, temperature=self.temperature,
            dcd_window_length=self.dcd_window_length, dcd_max_window_length=self.dcd_max_window_length, dcd_window_type=self.dcd_window_type,
            dcd_decode_algo=self.dcd_decode_algo, dcd_decode_param=self.dcd_decode_param, dcd_sample_top_k=self.dcd_sample_top_k,
            dcd_top_p=self.dcd_top_p, dcd_cache_type=self.dcd_cache_type, dcd_refresh_count=self.dcd_refresh_count, omega_bias=omega_bias,
        )

    def _stride_omega(self, bio_logits: torch.Tensor, mask: torch.Tensor, ts_b: torch.Tensor, start_s: float):
        # Per-stride membership gate: on-policy span (no GT at inference), χ = frames already emitted per the runner's own commit log 
        # (≤δ overlap leftovers the cut keeps). None when gate is off or the model has no gate builder (plain test fakes / AR-only models).
        if not self.gate_enabled or not hasattr(self.model, "build_gate_omega"): return None
        chi = (ts_b + float(start_s)) < self._committed_until_s  # (1, T') bool, absolute timeline
        omega_bias, _ = self.model.build_gate_omega(
            bio_logits, None, mask, memory_len=self.model.front_end.prompt_length() + int(bio_logits.shape[1]),
            commit_mask=chi, delta=self.gate_delta, eps=self.gate_eps, min_span_frames=self.min_span_frames, timestamps_s=ts_b,
        )
        return omega_bias

    @torch.no_grad()
    def step(
        self, poses: torch.Tensor, timestamps_s: torch.Tensor, 
        end_s: float, last_commit_t: float = 0.0, force: bool = False
    ) -> StreamingEvent | None:
        """One sawtooth stride over the buffer [last_commit_t, end_s).

        The buffer grows from the last commit cut, not a fixed trailing window, so a committed sentence is cut out and never re-emitted.
        A buffer that reaches BUFFER_CAP_S forces a PARTIAL commit to bound latency. `force=True` applies the same forced-commit policy 
        regardless of buffer size — the end-of-stream drain (evidence frozen == cap reached).
        """
        start_s = float(last_commit_t)
        in_buffer = (timestamps_s >= start_s) & (timestamps_s < end_s)
        if not in_buffer.any(): return None

        poses_b = poses[in_buffer].unsqueeze(0)
        ts_b = (timestamps_s[in_buffer] - start_s).unsqueeze(0)
        mask_b = torch.ones(poses_b.shape[:2], dtype=torch.bool, device=poses.device)

        bio_tap, mask, ts = self.model.front_end.extract_bio_tap(poses_b, mask_b, ts_b)
        bio_logits = self.model.bio_head(bio_tap, timestamps_s=ts).logits
        bio_tags = bio_logits.argmax(dim=-1)[0]
        # UNK closes like O (the rule every eval decoder applies via close_on_unk=True). UNK is supervision-free
        # (ignore_index) so argmax-UNK is rare, but without this remap a single UNK frame in a gap keeps a span
        # terminator-less (deferring the commit to the buffer cap → spurious PARTIAL) and an O→UNK→I sequence never
        # opens (_span_opens needs prev==O exactly) — the FSM would silently diverge from the shared decode rule.
        bio_tags = torch.where(bio_tags == BIO["UNK"], torch.full_like(bio_tags, BIO["O"]), bio_tags)
        if self.duration_prior is not None and bio_tags.numel() > 2:
            dt = ts[0, 1:] - ts[0, :-1]
            # Clamp to [1,120] fps: a degenerate ~0 median dt would otherwise give fps ~1e6 -> lmax OOM in the DP.
            fps_b = min(max(1.0 / max(float(dt.median().item()), 1e-6), 1.0), 120.0) if dt.numel() else 24.0
            pB = torch.softmax(bio_logits[0].float(), dim=-1)[:, BIO["B"]].cpu().numpy()
            bio_tags = torch.as_tensor(
                duration_split_tags(bio_tags.cpu().numpy(), pB, fps_b, self.duration_prior, mark_onsets=False, split_open_tail=False),
                device=bio_tags.device, dtype=bio_tags.dtype,
            )
        # FSM-internal BIO record: stitch this stride's per-frame argmax into the whole-stream timeline
        # (latest estimate wins on re-visited frames) so RQ2 can score the deployed head's segmentation directly.
        if self._bio_timeline is not None and bio_tags.numel() == int(in_buffer.sum().item()):
            self._bio_timeline[in_buffer.cpu()] = bio_tags.detach().cpu()
        buffer_full = force or (end_s - start_s) >= self.buffer_cap_s
        omega_bias = self._stride_omega(bio_logits, mask, ts_b, start_s)

        # after_forced resolution (translate_partial only). Auto-reset when the buffer does not start with I —
        # there is no continuation to wait for (e.g. the forced flush landed in a gap), so normal selection must
        # proceed immediately; blocking on a lead that never appears would deadlock the FSM into cap-only commits.
        if self._after_forced and self.forced_tail_policy == "translate_partial":
            lead_term = leading_i_run_end(bio_tags)
            if bio_tags.numel() and int(bio_tags[0].item()) != BIO["I"]: self._after_forced = False
            elif lead_term is not None:
                self._after_forced = False
                if lead_term >= max(1, self.min_span_frames):
                    self._bump("committed"); self._bump("forced_tail_commit")
                    # NO gate on this decode: the stride's Ω is anchored on a span selection in which the
                    # buffer-start I-run can never open, so it would floor exactly the frames [0, lead_term)
                    # being decoded (attention ×~1e-4 on the fragment's own evidence). The fragment decode is
                    # already declared best-effort/out-of-distribution (ablation only) — run it ungated.
                    tokens, confidence = self._decode_span(bio_tap, mask, slice(0, lead_term), omega_bias=None)
                    self.commit_gate.reset()
                    return StreamingEvent(
                        start_s=float(start_s + ts_b[0, 0].item()), end_s=float(start_s + ts_b[0, lead_term].item()),
                        token_ids=tokens[0].detach().cpu(), token_confidence=confidence[0].detach().cpu(),
                        bio_start_index=0, bio_end_index=lead_term, flagged_partial=True, commit_time_s=float(end_s),
                    )
        elif self._after_forced: self._after_forced = False  # "skip": leftover I-run is simply never selected

        span = select_target_span(bio_tags, self.min_span_frames)
        if span is None:
            self.commit_gate.update(None, token_confidence=None)
            if not buffer_full: return None

            # Forced commit at cap: decode the in-progress (right-truncated) span if one exists, else drop the (gap/fragment) buffer.
            b_idx = open_span_start(bio_tags)  # same open-span rule as the gate anchor (one decode rule, one place)
            self.commit_gate.reset()
            if b_idx is None: return None
            self._bump("committed"); self._bump("forced_commit")
            self._after_forced = True  # the cut lands mid-sentence; its continuation follows

            s_idx, last_idx = b_idx, int(bio_tags.numel()) - 1
            tokens, confidence = self._decode_span(bio_tap, mask, slice(s_idx, last_idx + 1), omega_bias=omega_bias)
            return StreamingEvent(
                start_s=float(start_s + ts_b[0, s_idx].item()), end_s=float(start_s + ts_b[0, last_idx].item()),
                token_ids=tokens[0].detach().cpu(), token_confidence=confidence[0].detach().cpu(),
                bio_start_index=s_idx, bio_end_index=last_idx, flagged_partial=True, commit_time_s=float(end_s),
            )

        s_idx, term_idx = span
        tokens, confidence = self._decode_span(bio_tap, mask, slice(s_idx, max(s_idx + 1, term_idx)), omega_bias=omega_bias)
        decision = self.commit_gate.update(span, token_confidence=confidence[0])
        self._bump("spans_seen")
        if decision.boundary_stable: self._bump("boundary_ok")
        if decision.translation_confident: self._bump("translation_ok")
        # Deliberate deviation from the spec pseudocode's flush-at-cap (cut at current_t − δ): a COMPLETE span
        # force-committed here still cuts at ITS terminator − δ (event.end_s → run()'s overlap cut), not at the
        # buffer end. Flushing to the cap would discard every later sentence already inside the buffer with no
        # emission; cutting at the terminator keeps them, and the buffer shrinks below the cap within a stride or
        # two as the remaining spans commit in turn — a strictly-no-lost-sentences resolution of the same latency bound.
        forced = buffer_full and not decision.should_commit
        if not decision.should_commit and not forced: return None

        self._bump("committed")
        if forced: self._bump("forced_commit")
        self.commit_gate.reset()
        return StreamingEvent(
            start_s=float(start_s + ts_b[0, s_idx].item()), end_s=float(start_s + ts_b[0, term_idx].item()),
            token_ids=tokens[0].detach().cpu(), token_confidence=confidence[0].detach().cpu(),
            bio_start_index=s_idx, bio_end_index=term_idx, flagged_partial=forced, commit_time_s=float(end_s),
        )

    @torch.no_grad()
    def run(self, poses: torch.Tensor, fps: float) -> list[StreamingEvent]:
        device = next(self.model.parameters()).device
        poses = poses.to(device)
        timestamps = torch.arange(poses.shape[0], device=device, dtype=torch.float32) / float(fps)
        events: list[StreamingEvent] = []
        duration = float(timestamps[-1].item()) if timestamps.numel() else 0.0
        # Overlap cut: keep last δ frames when cutting at a committed terminator. A terminator estimate late by up to δ 
        # then still leaves the next sentence's onset (its B) in the buffer — without the overlap, a late estimate eats 
        # the next sentence's B; buffer-start I never opens a span (Mode-2b silence), and that sentence is silently dropped. 
        # The ≤2δ leftover of the committed sentence is skipped by _span_opens / Λ_min, so it is never re-emitted.
        overlap_s = float(self.commit_gate.history.delta_enc_frames) / float(fps)
        frame_s = 1.0 / float(fps)
        last_commit_t = 0.0
        end_s = self.stride_s
        self._after_forced = False
        self._committed_until_s = 0.0  # χ commit log resets per stream
        self._bio_timeline = torch.full((poses.shape[0],), int(BIO["UNK"]), dtype=torch.long)  # stitched tags

        def absorb(event: StreamingEvent) -> None:
            nonlocal last_commit_t
            events.append(event)
            # χ: everything up to this commit's end is now emitted content — the ≤δ overlap the next
            # buffer keeps must be attention-floored by the membership gate (seam-duplication guard §2.7).
            self._committed_until_s = max(self._committed_until_s, float(event.end_s))
            # Advance by at least one frame so a degenerate (≤δ) span can never stall the stream.
            last_commit_t = max(event.end_s - overlap_s, last_commit_t + frame_s)

        while end_s <= duration + self.stride_s:
            event = self.step(poses, timestamps, end_s=end_s, last_commit_t=last_commit_t)
            if event is not None: absorb(event)
            elif (end_s - last_commit_t) >= self.buffer_cap_s:
                last_commit_t = max(end_s - overlap_s, last_commit_t + frame_s)  # gap/fragment buffer hit the cap with nothing to emit
            end_s += self.stride_s

        # End-of-stream drain. Evidence is frozen at `duration`, but the K-stride hysteresis still needs votes: a terminator that first 
        # became visible within the last K−1 strides — or a complete span awaiting its confidence gate — would otherwise be SILENTLY 
        # DROPPED (the loop is unbounded and says nothing about finite streams; dropping the final sentence of every clip systematically 
        # depresses streaming recall). Extra strides re-vote on identical frozen buffer, so a real terminator is stable by construction 
        # and commits normally (un-flagged). After K quiet strides, FORCED passes apply the buffer-cap policy — stream end IS the same 
        # evidence-frozen condition — draining EVERY still-pending span (each flagged PARTIAL).
        quiet = 0
        while quiet < int(self.commit_gate.history.hysteresis_strides):
            event = self.step(poses, timestamps, end_s=end_s, last_commit_t=last_commit_t)
            if event is not None: absorb(event); quiet = 0
            else: quiet += 1
            end_s += self.stride_s
        # Loop the forced pass: one pass emits one span, so a frozen buffer holding multiple pending sentences would drop all but 
        # the first. absorb advances last_commit_t by ≥ frame_s each iteration (buffer shrinks), so this terminates once only a 
        # sub-Λ_min leftover remains (step returns None).
        while True:
            event = self.step(poses, timestamps, end_s=end_s, last_commit_t=last_commit_t, force=True)
            if event is None: break
            absorb(event)
        return events
