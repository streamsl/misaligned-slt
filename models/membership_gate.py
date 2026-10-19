# The selected first span as a hard crop (span_membership): 1 Ω biases the 
# translator encoder self-attention keys and the decoder cross-attention.
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import math, warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from infer.duration_decode import DurationDecoder
from infer.commit_gate import candidate_spans, open_span_start


def membership_bias(membership, frame_mask, commit_mask=None, eps=1e-4):
    """Ω = log(eps + (1-eps)·clamp(m, 0, 1)) on valid, uncommitted frames; log(eps) on committed and padded frames.

    `membership` is the visibility of each frame given the selected span (`span_membership`, or a supplied interval).
    There is no rescale by window length: a point mass gives Ω = 0 on the span and log(eps) elsewhere, a crop on a 
    window of any length (every gated read encodes the support again as a crop right after the prompt, 
    models.streaming_slt.support_crop, so the translator then matches a hard crop up to the eps leak).
    """
    if not 0 < eps < 1: raise ValueError("eps must lie between 0 and 1")
    valid = frame_mask.bool()
    if commit_mask is not None: valid = valid & ~commit_mask.bool()
    omega = torch.log(eps + (1.-eps)*membership.clamp(0., 1.))
    return omega.masked_fill(~valid, math.log(eps))


class CrossAttnOmegaInjector:
    """Add Ω(t) to a HF encoder-decoder's cross-attention via forward pre-hooks, so HF's `forward` / `generate` /
    beam / KV-cache run unchanged and the AR arm sees what the DLM arm injects in its manual decode loop.
    Verified against transformers 4.57.3 internals:

      T5 / mT5 : cross-attn folds its mask into `encoder_decoder_position_bias` at block 0 and reuses it across
                 blocks, so adding Ω to each block's cross `attention_mask` kwarg reaches every layer.

    Ω is (B,1,1,M), broadcasts over heads and queries, re-applied every decode step. Beam expands the batch to
    B·beams — the hook repeats Ω to match. Use via `with_omega(omega_bias)`, inert when no gate is active.
    """
    def __init__(self, lm_model: torch.nn.Module):
        self._omega: torch.Tensor | None = None
        self.handles = [
            m.register_forward_pre_hook(self._pre_hook, with_kwargs=True) 
            for m in self._cross_attn_modules(lm_model)
        ]

    @classmethod
    def attach(cls, lm_model: torch.nn.Module) -> "CrossAttnOmegaInjector":
        # ONE injector per decoder stack. The AR arm hooks the whole seq2seq model and the DLM decoder hooks the bare stack; 
        # both resolve to the same cross-attention modules, and two hook sets would add Ω twice.
        modules = cls._cross_attn_modules(lm_model)
        if not modules: raise ValueError(f"CrossAttnOmegaInjector found no cross-attention modules on {type(lm_model).__name__}")
        injector = getattr(modules[0], "_omega_injector", None)
        if injector is None:
            injector = cls(lm_model)
            modules[0]._omega_injector = injector
        return injector

    @staticmethod
    def _cross_attn_modules(lm_model: torch.nn.Module) -> list[torch.nn.Module]:
        # Accepts a full HF encoder-decoder OR a bare decoder stack (T5Stack) — the DLM's
        # prefix-KV-cache path holds only the stack, and Ω injection is identical either way.
        dec = getattr(lm_model, "decoder", None) or lm_model
        if hasattr(dec, "block"):   # T5 / mT5 stack: block[i].layer[1] == T5LayerCrossAttention
            return [blk.layer[1] for blk in dec.block]
        return []

    def _pre_hook(self, _module, args, kwargs):
        omega = self._omega
        if omega is None: return None  # gate inactive → identity hook
        mask = kwargs.get("attention_mask", None)
        ref = mask if isinstance(mask, torch.Tensor) else omega
        om = omega.to(dtype=ref.dtype, device=ref.device)
        # Beam search expands the batch to B·beams. Read the target batch from the mask when there is one, else
        # from the query states: keying expansion off the mask ALONE left Ω un-expanded on any backbone that
        # cross-attends without a mask, gating beam rows with another row's Ω.
        hidden = args[0] if args and isinstance(args[0], torch.Tensor) else kwargs.get("hidden_states")
        tgt = mask.shape[0] if isinstance(mask, torch.Tensor) else (
            hidden.shape[0] if isinstance(hidden, torch.Tensor) else om.shape[0])
        if tgt != om.shape[0] and om.shape[0] and tgt % om.shape[0] == 0:
            om = om.repeat_interleave(tgt // om.shape[0], dim=0)
        kwargs["attention_mask"] = om if not isinstance(mask, torch.Tensor) else mask + om
        return args, kwargs

    def with_omega(self, omega_bias: torch.Tensor | None):
        # Restores the value active on entry: the DLM's cached forwards wrap each pass in this context, so a decode
        # nested in an outer AR context must hand the outer Ω back instead of switching the gate off for its rest.
        injector = self
        class _Ctx:
            def __enter__(self): self.prev, injector._omega = injector._omega, omega_bias
            def __exit__(self, *exc): injector._omega = self.prev
        return _Ctx()


def omega_cross_bias(omega: torch.Tensor, memory_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Expand Ω (B, T) into a (B, 1, 1, M) additive key bias (encoder self-attention and decoder cross-attention).

    Left-pads `prompt_len = M − T` zeros: the prompt is never gated, Ω aligns to pose columns [prompt_len, M)
    and broadcasts over heads and queries (query-independent — doc §2.10).
    """
    B, T = omega.shape
    prompt_len = int(memory_len) - int(T)
    if prompt_len < 0:
        raise ValueError(f"memory_len ({memory_len}) < pose frames ({T}); Ω cannot be aligned to cross-attn columns")
    full = F.pad(omega, (prompt_len, 0), value=0.0).to(dtype)   # (B, M); prompt columns = 0
    return full.view(B, 1, 1, int(memory_len))


@dataclass
class SpanPosterior:
    closed: torch.Tensor          # P(t in the first closed span)
    open: torch.Tensor            # P(t in the open first span); the open span runs to the window end, so this is its start CDF
    closed_probability: torch.Tensor
    open_probability: torch.Tensor
    closed_start: torch.Tensor    # P(the first closed span starts at t)


def span_membership(starts, terms, has_term, lengths, t: int) -> torch.Tensor:
    """Visibility (B, T) of SELECTED span [s, e]: 1 on it (e = its terminator frame, which the crop holds, as cascade crop does)
    and 0 off it. An open span (no terminator) is visible to window end. Row with s < 0 (no span) is visible everywhere. Every
    gated read encodes this support again as 1 crop (models.streaming_slt.support_crop), so translator reads cascade's crop."""
    frames = torch.arange(int(t), device=starts.device)[None]
    s, e, n = starts[:, None], terms[:, None], lengths[:, None]
    inside = (frames >= s) & torch.where(has_term.bool()[:, None], frames <= e, frames < n)
    return torch.where(s < 0, torch.ones_like(inside), inside).float()


def _lse(x, dim=-1): # Unreachable states have 0 gradient, including an all-impossible log-sum.
    reachable = torch.isfinite(x).any(dim=dim, keepdim=True)
    value = torch.logsumexp(torch.where(reachable, x, torch.zeros_like(x)), dim=dim)
    return torch.where(reachable.squeeze(dim), value, torch.full_like(value, -torch.inf))


class MembershipGate(nn.Module): # First-eligible-span probabilities and the Ω crop of the selected span.
    @staticmethod
    @lru_cache(maxsize=2)
    def _recurrence_steps(device_type):
        cells = (MembershipGate._forward_step, MembershipGate._backward_step)
        if device_type != "cuda": return cells
        print("membership gate | compiling CUDA recurrence cells on first use", flush=True)
        # Outputs remain live until backward; CUDA graph buffer reuse is not safe here.
        return tuple(MembershipGate._compile_cell(fn) for fn in cells)

    @staticmethod
    def _compile_cell(fn):
        compiled = torch.compile(fn, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
        def run(*args):
            nonlocal compiled
            if compiled is None: return fn(*args)
            try: return compiled(*args)
            except Exception as exc:
                if not type(exc).__module__.startswith(("torch._dynamo", "torch._inductor")): raise
                warnings.warn(f"Membership CUDA compiler failed ({type(exc).__name__}); using the same eager recurrence.",
                              RuntimeWarning, stacklevel=2)
                compiled = None
                return fn(*args)
        return run

    @staticmethod
    def _forward_step(previous, emission, end, stay, residual_end, residual_stay, prefix_closures, valid):
        finished = (previous[:, :, 2:] + end[:, None]).masked_fill(prefix_closures, -torch.inf)
        boundary = _lse(torch.cat((previous[:, :, :1], previous[:, :, 1:2] + residual_end[:, None, None], finished), -1))
        grown = previous[:, :, 2:] + stay[:, None]
        continuation = torch.cat((grown[:, :, :-2], _lse(grown[:, :, -2:])[:, :, None]), -1) if end.shape[1] > 1 else grown[:, :, :0]
        advanced = torch.cat(((boundary + emission[:, None, 0])[:, :, None],
                              (previous[:, :, 1] + residual_stay[:, None] + emission[:, None, 2])[:, :, None],
                              (boundary + emission[:, None, 1])[:, :, None], continuation + emission[:, None, 2:3]), -1)
        scale = torch.where(valid, _lse(advanced[:, 0]), torch.zeros_like(emission[:, 0]))
        state = torch.where(valid[:, None, None], advanced - scale[:, None, None], previous)
        return state, scale

    @staticmethod
    def _members(active, close, opened, valid):
        values = torch.stack((_lse(active + close), _lse(active + opened), active[:, 0] + close[:, 0]), 1)
        return torch.where(valid[:, None], values, torch.full_like(values, -torch.inf)).exp()

    @staticmethod
    def _backward_step(beta, close, opened, emission, end, stay, eligible, next_age, scale, more, active, valid):
        boundary = _lse(torch.stack((emission[:, 0] + beta[:, 0], emission[:, 1] + beta[:, 1]), 1))
        closing = boundary[:, None] + end
        age_beta = _lse(torch.stack((closing, emission[:, 2:3] + stay + beta[:, 1:][:, next_age]), -1))
        future = torch.cat((boundary[:, None], age_beta), 1) - scale[:, None]
        close_next = _lse(torch.stack((
            closing.masked_fill(~eligible, -torch.inf), emission[:, 2:3] + stay + close[:, next_age]
        ), -1)) - scale[:, None]
        open_next = emission[:, 2:3] + stay + opened[:, next_age] - scale[:, None]
        beta = torch.where(more[:, None], future, torch.zeros_like(future))
        close = torch.where(more[:, None], close_next, torch.full_like(close_next, -torch.inf))
        opened = torch.where(more[:, None], open_next, torch.zeros_like(open_next))
        return beta, close, opened, MembershipGate._members(active, close, opened, valid)

    def posterior(self, logits, lengths, min_span_frames=1, commit_mask=None, known_start=False, *, decoder=None, timestamps_s=None):
        # Λ_min in frames: 1 int for every row, or 1 per row (each row's own frame rate).
        floors = torch.as_tensor(min_span_frames, dtype=torch.long).expand(len(logits))
        if len(floors) and int(floors.min()) < 1: raise ValueError("min_span_frames must be positive")
        scores = (decoder or DurationDecoder()).path_scores(
            logits, lengths, commit_mask, known_start, timestamps_s=timestamps_s, 
            min_age_states=int(floors.max()) if len(logits) else 1
        )
        e, factors, n = scores.emissions, scores.factors, scores.lengths
        b, padded_t, _ = e.shape
        t = int(n.max()) if b else 0
        if not t:
            empty = e[..., 0] * 0.; p = empty.sum(1)
            return SpanPosterior(empty, empty, p, p, empty)
        # Later padding and unreachable ages cannot affect a valid path; restore the original frame axes at the end.
        e, factors = e[:, :t], factors[..., :t]

        # Separate storage prevents compiler guards on changing residual-column offsets.
        end, stay, residual_end, residual_stay = (x.contiguous() for x in factors.unbind(1))
        ages = end.shape[1]
        eligible = torch.arange(1, ages + 1, device=e.device)[None] >= floors.to(e.device)[:, None]  # (B, ages)
        initial = scores.initial[:, :ages + 2]
        scale = _lse(initial); full = initial - scale[:, None]
        pref = full
        prefixes, scales = [pref], [scale]

        # Carry full-path and no-earlier-completion recurrences together.
        state = torch.stack((full, pref), 1)
        prefix_closures = torch.stack((torch.zeros_like(eligible), eligible), 1)
        forward_step, backward_step = self._recurrence_steps(e.device.type)

        # One unbind joins time-step gradients once; repeated slices scatter into the full sequence per step.
        emissions = e.unbind(1)
        for k in range(1, t):
            age = min(k - 1, ages - 1)
            step = self._forward_step if k == 1 else forward_step
            state, scale = step(state, emissions[k], end, stay, residual_end[:, age], residual_stay[:, age], prefix_closures, k < n)
            prefixes.append(state[:, 1]); scales.append(scale)

        # After a visible start, a suffix can enter O or another sentence, never the initial leading fragment.
        beta = e.new_zeros((b, ages+1))
        close = e.new_full((b, ages), -torch.inf)
        opened = e.new_zeros((b, ages))
        next_age = torch.arange(ages, device=e.device).add(1).clamp(max=ages-1)
        members_by_time = [self._members(prefixes[-1][:, 2:], close, opened, t - 1 < n)]
        for k in range(t - 2, -1, -1): # Initial suffixes and the first-frame view have different gradient/stride layouts.
            step = self._backward_step if k in (t - 2, 0) else backward_step
            beta, close, opened, members = step(
                beta, close, opened, emissions[k + 1], end, stay, eligible, next_age, 
                scales[k + 1], k + 1 < n, prefixes[k][:, 2:], k < n
            )
            members_by_time.append(members)

        closed, opened, starts = torch.stack(members_by_time[::-1], 1).unbind(-1)
        p_open = opened.gather(1, (n - 1).clamp(min=0)[:, None]).squeeze(1)
        return SpanPosterior(
            scores.restore(F.pad(closed, (0, padded_t - t))), scores.restore(F.pad(opened, (0, padded_t - t))),
            starts.sum(1), p_open, scores.restore(F.pad(starts, (0, padded_t - t)))
        )

    def forward(
        self, bio_logits, frame_mask, memory_len, *, commit_mask=None, eps=1e-4, min_span_frames=1,
        seam_is_terminator=True, stream_start=False, anchor_override=None, timestamps_s=None, decoder=None,
    ):
        # Ω for deployed path: predicted 1st span as a hard crop (`span_membership`), or a supplied interval on every row 
        # (`anchor_override`, (start, terminator) frames). Λ_min (`min_span_frames`) is a frame count..
        if anchor_override is not None:
            # A supplied interval replaces membership completely; there is no inferred posterior to compute or report.
            if not bool((anchor_override[:, 0] >= 0).all()): raise ValueError("a supplied interval needs a start on every row")
            if commit_mask is not None:
                if commit_mask.shape != frame_mask.shape: raise ValueError("Commit mask must match the frame axes")
                committed = commit_mask.bool() & frame_mask.bool()
                prefix = torch.arange(frame_mask.shape[1], device=frame_mask.device)[None] < committed.sum(1)[:, None]
                if not torch.equal(committed, prefix): raise ValueError("Committed frames must form a prefix")
            membership = self._provided_membership(anchor_override, bio_logits.shape[1])
            omega = membership_bias(membership, frame_mask, commit_mask, eps)
            return omega_cross_bias(omega, int(memory_len), bio_logits.dtype), {
                "skip": torch.zeros(len(bio_logits), dtype=torch.bool), "uncertain_rows": 0,
                "uncertain": torch.zeros(len(bio_logits), dtype=torch.bool),
            }
        selected = self.select(
            bio_logits, frame_mask, commit_mask=commit_mask, min_span_frames=min_span_frames, seam_is_terminator=seam_is_terminator,
            stream_start=stream_start, timestamps_s=timestamps_s, decoder=decoder,
        )
        starts, terms, has_term = selected["anchors"]
        membership = span_membership(starts, terms, has_term, selected["lengths"], bio_logits.shape[1])
        omega = membership_bias(membership, frame_mask, commit_mask, eps)
        bias = omega_cross_bias(omega, memory_len=int(memory_len), dtype=bio_logits.dtype)
        return bias, {
            "skip": starts.cpu() < 0,
            "uncertain_rows": int((starts < 0).sum()),
            "uncertain": (starts < 0).detach().cpu(),   # per row: no span and no open start, Ω falls back to the whole window
            "anchors": (starts, terms, has_term),
        }

    @staticmethod
    def _provided_membership(anchors, length):
        # [start, terminator]: the terminator frame stays visible, as in the training crop and the cascade crop. 
        # A terminator of -1 means the span runs to the window end.
        positions = torch.arange(length, device=anchors.device)[None]
        start, term = anchors[:, :1], anchors[:, 1:2]
        return ((positions >= start) & ((term < 0) | (positions <= term))).float()

    def select(
        self, bio_logits, frame_mask, *, commit_mask=None, min_span_frames=1, seam_is_terminator=True, 
        stream_start=False, timestamps_s=None, decoder=None,
    ) -> dict:
        # The first-span readout as inference makes it: Viterbi tags -> the first candidate span (start, terminator), or the 
        # open start when no span is complete; a row with no span and no open start has start -1 and reads the whole window.
        lengths = frame_mask.long().sum(1)
        known_start = torch.full_like(lengths, bool(stream_start), dtype=torch.bool)
        if commit_mask is not None and seam_is_terminator: known_start = known_start | commit_mask.any(dim=1)
        decoder = decoder or DurationDecoder()
        tags = decoder.decode(bio_logits, lengths, commit_mask, known_start, timestamps_s=timestamps_s)
        floors = torch.as_tensor(min_span_frames, dtype=torch.long).clamp(min=1).expand(len(lengths))
        spans = [next(candidate_spans(tags[b, :int(n)], int(floors[b])), None) for b, n in enumerate(lengths)]
        starts, terms, has_term = [], [], []

        for b, n in enumerate(lengths):
            span = spans[b]
            open_start = open_span_start(tags[b, :int(n)])
            starts.append(span[0] if span else (open_start if open_start is not None else -1))
            terms.append(span[1] if span else -1); has_term.append(span is not None)

        device = bio_logits.device
        return {"lengths": lengths, "anchors": (
            torch.tensor(starts, device=device), torch.tensor(terms, device=device), torch.tensor(has_term, device=device)
        )}
