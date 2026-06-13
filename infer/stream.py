from __future__ import annotations
from dataclasses import dataclass

import torch
from data.windowing import BIO
from infer.commit_gate import CommitGate, first_complete_bio_span, _span_opens


def active_span_start(bio_tags: torch.Tensor) -> int | None:
    # Start index of the in-progress span (opened, never closed). Opening matches the shared decode
    # (commit_gate._span_opens): B anywhere, or a mid-buffer O→I transition; buffer-start I without
    # B stays closed (left-truncated per Mode-2b training — never translate a headless fragment).
    start: int | None = None
    prev: int | None = None
    for idx, tag in enumerate(bio_tags.tolist()):
        if start is not None and tag == BIO["O"]: start = None
        if _span_opens(tag, prev, start is not None): start = idx
        prev = tag
    return start


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
    and cold-starts translation for the first complete predicted span.
    """
    def __init__(
        self, model, stride_s: float = 1.0, buffer_cap_s: float = 18.0, delta_enc_frames: int = 3, hysteresis_strides: int = 2,
        token_confidence_tau: float = 0.75, max_text_tokens: int = 128, diffusion_steps: int = 64, tau_dec: float = 0.75,
        spd_top_k: int = 1, spd_renormalize: bool = True, spd_revision: bool = True, temperature: float = 0.0,
        dcd_window_length: int | None = None, dcd_max_window_length: int | None = None, dcd_window_type: str = "sliding",
        dcd_decode_algo: str = "threshold", dcd_decode_param: int | float | None = None, dcd_sample_top_k: int | None = None,
        dcd_top_p: float | None = None, dcd_cache_type: str = "none", dcd_refresh_count: int = 16,
        decode_conditioning: str = "window",
    ):
        # decode_conditioning: "window" (default) decodes under the FULL buffer's features — exactly the
        # training conditioning, where Mode 1/3 windows feed the whole jittered window to the encoder and
        # OPUT supervises the first-complete-span text under it (§5.1/§5.3: right-context is *learned to be
        # disregarded*, not cropped away). "span" crops conditioning to the predicted BIO span (the spec §7.1
        # pseudocode's `enc[span]`), which the trained conditional never sees except at the zero-jitter corner
        # of the CDF — kept as an ablation. The BIO span still defines the emitted boundaries either way.
        if decode_conditioning not in {"window", "span"}: raise ValueError(f"Unknown decode_conditioning: {decode_conditioning}")
        self.decode_conditioning = str(decode_conditioning)
        self.model = model
        self.stride_s = float(stride_s)
        self.buffer_cap_s = float(buffer_cap_s)
        self.max_text_tokens = int(max_text_tokens)
        self.diffusion_steps = int(diffusion_steps)
        self.tau_dec = float(tau_dec)

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

    def _decode_span(self, post_vlp: torch.Tensor, mask: torch.Tensor, span_slice: slice):
        if self.decode_conditioning == "span": post_vlp, mask = post_vlp[:, span_slice], mask[:, span_slice]
        return self.model.generate_from_post_vlp(
            post_vlp, mask,
            max_text_tokens=self.max_text_tokens, diffusion_steps=self.diffusion_steps, tau_dec=self.tau_dec,
            spd_top_k=self.spd_top_k, spd_renormalize=self.spd_renormalize, spd_revision=self.spd_revision, temperature=self.temperature,
            dcd_window_length=self.dcd_window_length, dcd_max_window_length=self.dcd_max_window_length, dcd_window_type=self.dcd_window_type,
            dcd_decode_algo=self.dcd_decode_algo, dcd_decode_param=self.dcd_decode_param, dcd_sample_top_k=self.dcd_sample_top_k,
            dcd_top_p=self.dcd_top_p, dcd_cache_type=self.dcd_cache_type, dcd_refresh_count=self.dcd_refresh_count,
        )

    @torch.no_grad()
    def step(self, poses: torch.Tensor, timestamps_s: torch.Tensor, end_s: float, last_commit_t: float = 0.0) -> StreamingEvent | None:
        """One sawtooth stride over the buffer [last_commit_t, end_s).

        The buffer grows from the last commit point (spec §7.1), not a fixed trailing window, so a committed sentence is cut out and 
        never re-emitted. A buffer that reaches BUFFER_CAP_S forces a PARTIAL commit to bound latency.
        """
        start_s = float(last_commit_t)
        in_buffer = (timestamps_s >= start_s) & (timestamps_s < end_s)
        if not in_buffer.any(): return None

        poses_b = poses[in_buffer].unsqueeze(0)
        ts_b = (timestamps_s[in_buffer] - start_s).unsqueeze(0)
        mask_b = torch.ones(poses_b.shape[:2], dtype=torch.bool, device=poses.device)

        post_vlp, mask, ts = self.model.visual.extract_post_vlp(poses_b, mask_b, ts_b)
        bio_logits = self.model.bio_head(post_vlp, timestamps_s=ts).logits
        bio_tags = bio_logits.argmax(dim=-1)[0]
        buffer_full = (end_s - start_s) >= self.buffer_cap_s
        span = first_complete_bio_span(bio_tags)

        if span is None:
            self.commit_gate.update(bio_tags, token_confidence=None)
            if not buffer_full: return None

            # Forced commit at the cap: decode the in-progress (right-truncated)
            # span if one exists, else drop the (gap/fragment) buffer.
            b_idx = active_span_start(bio_tags)
            self.commit_gate.reset()
            if b_idx is None: return None

            s_idx, last_idx = b_idx, int(bio_tags.numel()) - 1
            tokens, confidence = self._decode_span(post_vlp, mask, slice(s_idx, last_idx + 1))
            return StreamingEvent(
                start_s=float(start_s + ts_b[0, s_idx].item()), end_s=float(start_s + ts_b[0, last_idx].item()),
                token_ids=tokens[0].detach().cpu(), token_confidence=confidence[0].detach().cpu(),
                bio_start_index=s_idx, bio_end_index=last_idx, flagged_partial=True, commit_time_s=float(end_s),
            )

        s_idx, closing_o_idx = span
        tokens, confidence = self._decode_span(post_vlp, mask, slice(s_idx, max(s_idx + 1, closing_o_idx)))
        decision = self.commit_gate.update(bio_tags, token_confidence=confidence[0])
        forced = buffer_full and not decision.should_commit
        if not decision.should_commit and not forced: return None

        self.commit_gate.reset()
        return StreamingEvent(
            start_s=float(start_s + ts_b[0, s_idx].item()), end_s=float(start_s + ts_b[0, closing_o_idx].item()),
            token_ids=tokens[0].detach().cpu(), token_confidence=confidence[0].detach().cpu(),
            bio_start_index=s_idx, bio_end_index=closing_o_idx, flagged_partial=forced, commit_time_s=float(end_s),
        )

    @torch.no_grad()
    def run(self, poses: torch.Tensor, fps: float) -> list[StreamingEvent]:
        device = next(self.model.parameters()).device
        poses = poses.to(device)
        timestamps = torch.arange(poses.shape[0], device=device, dtype=torch.float32) / float(fps)
        events: list[StreamingEvent] = []
        duration = float(timestamps[-1].item()) if timestamps.numel() else 0.0
        last_commit_t = 0.0
        end_s = self.stride_s
        while end_s <= duration + self.stride_s:
            event = self.step(poses, timestamps, end_s=end_s, last_commit_t=last_commit_t)
            if event is not None:
                events.append(event)
                last_commit_t = event.end_s  # cut the buffer at the commit point
            elif (end_s - last_commit_t) >= self.buffer_cap_s:
                last_commit_t = end_s  # gap/fragment buffer hit the cap with nothing to emit
            end_s += self.stride_s
        return events
