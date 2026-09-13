# First-span membership as decoder cross-attention bias.
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache

import math, warnings
import torch
import torch.nn.functional as F
from infer.duration_decode import DurationDecoder
from infer.commit_gate import open_span_start, select_target_span


def membership_bias(membership, frame_mask, commit_mask=None, eps=1e-4):
    """Normalize the pose PRIOR weights to sum to the valid frame count.

    This preserves pairwise pose bias differences and makes a uniform prior neutral on valid frames.
    It does not fix the pose/prompt attention share: actual weights also depend on query-key scores.
    Padding and committed frames receive the finite floor; the encoder mask separately excludes padding.
    """
    if not 0 < eps < 1: raise ValueError("eps must lie between 0 and 1")
    valid = frame_mask.bool()
    if commit_mask is not None: valid = valid & ~commit_mask.bool()
    m = torch.where(valid, membership.clamp(0., 1.), torch.zeros_like(membership))
    omega = torch.log(eps + (1.-eps)*m).masked_fill(~valid, -torch.inf)
    # `_lse` rather than a plain logsumexp: a row with no attendable frame is all -inf, and logsumexp's backward would
    # form 0 * NaN there (zeroed downstream, but anomaly mode raises). _lse keeps that row's gradient exactly 0.
    total = _lse(omega, dim=-1).unsqueeze(-1)
    # A row with no attendable frame has no mass to redistribute; leave it at the floor rather than dividing by 0.
    scale = torch.where(total.isfinite(), valid.sum(-1, keepdim=True).clamp(min=1).log() - total, total.new_zeros(()))
    return (omega + scale).masked_fill(~valid, math.log(eps))


class CrossAttnOmegaInjector:
    """Add Ω(t) to a HF encoder-decoder's cross-attention via forward pre-hooks, so HF's `forward` / `generate` /
    beam / KV-cache run unchanged and the AR arm sees what the DLM arm injects in its manual decode loop.
    Verified against transformers 4.57.3 internals:

      T5 / mT5 : cross-attn folds its mask into `encoder_decoder_position_bias` at block 0 and reuses it across
                 blocks, so adding Ω to each block's cross `attention_mask` kwarg reaches every layer.
      mBART    : each `MBartDecoderLayer` adds `encoder_attention_mask` to the cross-attn scores directly
                 (eager `attn_weights += attention_mask`; SDPA `attn_mask`), so Ω gates every layer.

    Ω is (B,1,1,M), broadcasts over heads and queries, re-applied every decode step. Beam expands the batch to
    B·beams — the hook repeats Ω to match. Use via `with_omega(omega_bias)`, inert when no gate is active.
    """
    def __init__(self, lm_model: torch.nn.Module):
        self._omega: torch.Tensor | None = None
        self.handles = [
            m.register_forward_pre_hook(self._pre_hook, with_kwargs=True) 
            for m in self._cross_attn_modules(lm_model)
        ]
        if not self.handles:
            raise ValueError(f"CrossAttnOmegaInjector found no cross-attention modules on {type(lm_model).__name__}")

    @staticmethod
    def _cross_attn_modules(lm_model: torch.nn.Module) -> list[torch.nn.Module]:
        # Accepts a full HF encoder-decoder OR a bare decoder stack (T5Stack / MBartDecoder) — the DLM's
        # prefix-KV-cache path holds only the stack, and Ω injection is identical either way.
        dec = getattr(lm_model, "decoder", None) or getattr(getattr(lm_model, "model", None), "decoder", None) or lm_model
        if hasattr(dec, "block"):   # T5 / mT5 stack: block[i].layer[1] == T5LayerCrossAttention
            return [blk.layer[1] for blk in dec.block]
        if hasattr(dec, "layers"):  # mBART: layers[i].encoder_attn == MBartAttention (cross)
            return [layer.encoder_attn for layer in dec.layers]
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
        injector = self
        class _Ctx:
            def __enter__(self): injector._omega = omega_bias
            def __exit__(self, *exc): injector._omega = None
        return _Ctx()

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []


def omega_cross_bias(omega: torch.Tensor, memory_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Expand Ω (B, T) into a (B, 1, 1, M) additive bias for the existing cross-attention mask bias.

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
    closed: torch.Tensor
    open: torch.Tensor
    closed_probability: torch.Tensor
    open_probability: torch.Tensor

def _lse(x, dim=-1): # Unreachable states have 0 gradient, including an all-impossible log-sum.
    reachable = torch.isfinite(x).any(dim=dim, keepdim=True)
    value = torch.logsumexp(torch.where(reachable, x, torch.zeros_like(x)), dim=dim)
    return torch.where(reachable.squeeze(dim), value, torch.full_like(value, -torch.inf))


class MembershipGate(torch.nn.Module): # First-eligible-span probabilities and decoder cross-attention conditioning.
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
    def _forward_step(previous, emission, end, stay, bonus, residual_end, residual_stay, prefix_closures, valid):
        finished = (previous[:, :, 2:] + end[:, None] + bonus[:, None]).masked_fill(prefix_closures, -torch.inf)
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
    def _backward_step(beta, close, opened, emission, end, stay, bonus, eligible, next_age, scale, more, active, valid):
        boundary = _lse(torch.stack((emission[:, 0] + beta[:, 0], emission[:, 1] + beta[:, 1]), 1))
        closing = boundary[:, None] + end + bonus
        age_beta = _lse(torch.stack((closing, emission[:, 2:3] + stay + beta[:, 1:][:, next_age]), -1))
        future = torch.cat((boundary[:, None], age_beta), 1) - scale[:, None]
        close_next = _lse(torch.stack((
            closing.masked_fill(~eligible[None], -torch.inf), emission[:, 2:3] + stay + close[:, next_age]
        ), -1)) - scale[:, None]
        open_next = emission[:, 2:3] + stay + opened[:, next_age] - scale[:, None]
        beta = torch.where(more[:, None], future, torch.zeros_like(future))
        close = torch.where(more[:, None], close_next, torch.full_like(close_next, -torch.inf))
        opened = torch.where(more[:, None], open_next, torch.zeros_like(open_next))
        return beta, close, opened, MembershipGate._members(active, close, opened, valid)

    def posterior(self, logits, lengths, min_span_frames=1, commit_mask=None, known_start=False, *, decoder=None, timestamps_s=None):
        floor = int(min_span_frames)
        if floor < 1: raise ValueError("min_span_frames must be positive")
        scores = (decoder or DurationDecoder()).path_scores(
            logits, lengths, commit_mask, known_start, timestamps_s=timestamps_s, min_age_states=floor
        )
        e, factors, bonus, n = scores.emissions, scores.factors, scores.bonus, scores.lengths
        b, padded_t, _ = e.shape
        t = int(n.max()) if b else 0
        if not t:
            empty = e[..., 0] * 0.; p = empty.sum(1)
            return SpanPosterior(empty, empty, p, p)
        # Later padding and unreachable ages cannot affect a valid path; restore the original frame axes at the end.
        e, factors = e[:, :t], factors[..., :t]

        # Separate storage prevents compiler guards on changing residual-column offsets.
        end, stay, residual_end, residual_stay = (x.contiguous() for x in factors.unbind(1))
        ages = end.shape[1]
        eligible = torch.arange(1, ages + 1, device=e.device) >= floor
        initial = scores.initial[:, :ages + 2]
        scale = _lse(initial); full = initial - scale[:, None]
        pref = full
        prefixes, scales = [pref], [scale]

        # Carry full-path and no-earlier-completion recurrences together.
        state = torch.stack((full, pref), 1)
        prefix_closures = torch.stack((torch.zeros_like(eligible), eligible))[None]
        forward_step, backward_step = self._recurrence_steps(e.device.type)

        # One unbind joins time-step gradients once; repeated slices scatter into the full sequence per step.
        emissions = e.unbind(1)
        for k in range(1, t):
            age = min(k - 1, ages - 1)
            step = self._forward_step if k == 1 else forward_step
            state, scale = step(
                state, emissions[k], end, stay, bonus, residual_end[:, age], residual_stay[:, age], prefix_closures, k < n
            )
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
                beta, close, opened, emissions[k + 1], end, stay, bonus,
                eligible, next_age, scales[k + 1], k + 1 < n, prefixes[k][:, 2:], k < n
            )
            members_by_time.append(members)

        closed, opened, starts = torch.stack(members_by_time[::-1], 1).unbind(-1)
        p_open = opened.gather(1, (n - 1).clamp(min=0)[:, None]).squeeze(1)
        return SpanPosterior(scores.restore(F.pad(closed, (0, padded_t - t))),
                             scores.restore(F.pad(opened, (0, padded_t - t))), starts.sum(1), p_open)

    @staticmethod
    def _span_iou(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float: # tIoU of 2 [start, terminator) frame spans.
        if a is None or b is None: return 0.0
        lo = max(a[0], b[0]); hi = min(a[1], b[1])
        inter = max(0, hi - lo)
        union = (a[1] - a[0]) + (b[1] - b[0]) - inter
        return inter / union if union > 0 else 0.0

    def forward(
        self, bio_logits, frame_mask, memory_len, *, commit_mask=None, eps=1e-4, min_span_frames=1,
        seam_is_terminator=True, stream_start=False, anchor_override=None, timestamps_s=None, decoder=None,
    ):
        if anchor_override is not None and bool((anchor_override[:, 0] >= 0).all()):
            # A supplied interval replaces membership completely; there is no inferred posterior to compute or report.
            if commit_mask is not None:
                if commit_mask.shape != frame_mask.shape: raise ValueError("Commit mask must match the frame axes")
                committed = commit_mask.bool() & frame_mask.bool()
                prefix = torch.arange(frame_mask.shape[1], device=frame_mask.device)[None] < committed.sum(1)[:, None]
                if not torch.equal(committed, prefix): raise ValueError("Committed frames must form a prefix")
            membership = self._provided_membership(anchor_override, bio_logits.shape[1])
            omega = membership_bias(membership, frame_mask, commit_mask, eps)
            return omega_cross_bias(omega, int(memory_len), bio_logits.dtype), {"skip": torch.zeros(len(bio_logits), dtype=torch.bool)}
        return self._condition( # Inference uses predicted state or an explicitly supplied proposal; no supervision is accepted.
            bio_logits, frame_mask, memory_len, commit_mask=commit_mask, eps=eps, min_span_frames=min_span_frames,
            seam_is_terminator=seam_is_terminator, stream_start=stream_start, anchor_override=anchor_override,
            timestamps_s=timestamps_s, decoder=decoder,
        )

    @staticmethod
    def _provided_membership(anchors, length):
        positions = torch.arange(length, device=anchors.device)[None]
        start, end = anchors[:, :1], anchors[:, 1:2]
        return ((positions >= start) & ((end < 0) | (positions < end))).float()

    def for_supervision( # Train/dev loss only: GT selects closed/open state and diagnostics, never mask boundaries.
        self, bio_logits, bio_labels, frame_mask, memory_len, *, commit_mask=None, eps=1e-4, 
        min_span_frames=1, gt_spans=None, timestamps_s=None, decoder=None,
    ):
        if gt_spans is None:
            if bio_labels is None: raise ValueError("Supervised membership requires target spans or BIO labels")
            gt_spans = [
                select_target_span(row[:int(n)], max(1, min_span_frames))
                for row, n in zip(bio_labels, frame_mask.long().sum(1))
            ]
        if len(gt_spans) != len(bio_logits): raise ValueError("One supervised target state is required per row")
        return self._condition(
            bio_logits, frame_mask, memory_len, commit_mask=commit_mask, eps=eps, min_span_frames=min_span_frames,
            gt_spans=gt_spans, timestamps_s=timestamps_s, decoder=decoder,
        )

    def _condition(
        self, bio_logits, frame_mask, memory_len, *, commit_mask=None, eps=1e-4, min_span_frames=1,
        seam_is_terminator=True, stream_start=False, anchor_override=None, gt_spans=None, timestamps_s=None, decoder=None,
    ):
        lengths = frame_mask.long().sum(1)
        known_start = torch.full_like(lengths, bool(stream_start), dtype=torch.bool)
        if commit_mask is not None and seam_is_terminator: known_start = known_start | commit_mask.any(dim=1)
        decoder = decoder or DurationDecoder()
        tags = decoder.decode(bio_logits, lengths, commit_mask, known_start, timestamps_s=timestamps_s)
        spans = [select_target_span(tags[b, :int(n)], max(1, min_span_frames)) for b, n in enumerate(lengths)]
        starts, terms, has_term = [], [], []
        use_open, hits, targets = [], 0, 0

        for b, n in enumerate(lengths):
            span = spans[b]
            open_start = open_span_start(tags[b, :int(n)])
            starts.append(span[0] if span else (open_start if open_start is not None else -1))
            terms.append(span[1] if span else -1); has_term.append(span is not None)
            closed_state = span is not None
            if gt_spans is not None:
                gt = gt_spans[b]
                closed_state = gt is not None
                if gt is not None: targets += 1; hits += int(self._span_iou(span, gt) >= .5)
            use_open.append(not closed_state)

        posterior = self.posterior(
            bio_logits, lengths, max(1, min_span_frames), commit_mask, known_start, decoder=decoder, timestamps_s=timestamps_s
        )
        open_rows = torch.tensor(use_open, device=bio_logits.device)[:, None]
        membership = torch.where(open_rows, posterior.open, posterior.closed)

        if anchor_override is not None: # Given-span evaluation conditions on supplied boundaries: a point-mass interval posterior.
            fixed = self._provided_membership(anchor_override, bio_logits.shape[1])
            membership = torch.where(anchor_override[:, :1] >= 0, fixed, membership)

        omega = membership_bias(membership, frame_mask, commit_mask, eps)
        bias = omega_cross_bias(omega, memory_len=int(memory_len), dtype=bio_logits.dtype)
        return bias, {
            "skip": torch.tensor(starts) < 0,
            "anchor_hit_rate": hits/max(1, targets), "closed_probability": posterior.closed_probability.detach().mean(),
            "open_probability": posterior.open_probability.detach().mean(), "anchors": (
                torch.tensor(starts, device=bio_logits.device), torch.tensor(terms, device=bio_logits.device),
                torch.tensor(has_term, device=bio_logits.device)
            )
        }
