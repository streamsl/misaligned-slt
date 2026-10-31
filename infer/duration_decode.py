"""Shared semi-Markov BIO scores and exact Viterbi decoding.

A path scores the head's O/B/I log-probabilities + a duration prior fitted to the TRAIN unit durations; there is no tuned per-span 
bonus or B-evidence weight. The viterbi decode of a standalone segmenter (eval.py --decode viterbi, DurationDecoder(None)) and the
dev-chunk diagnostics use the same state grammar without duration factors. With no certified start, the leading fragment is exactly
the head's own leading I-run (`_leading_fragment`), and the first span may also open on O -> I after an all-O start. Caption
membership is computed from these scores in models.membership_gate. `argmax_tags` is the argmax decode of a standalone segmenter in the FSM (eval.py --online --decode argmax):
no path search.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from functools import lru_cache
import math
import numpy as np
import torch
from scipy.special import log_ndtr
from data.loader import cached_language_records, annotation_fingerprint
from data.windowing import BIO


@dataclass(frozen=True)
class DurationModel:
    mu_log_s: float
    sd_log_s: float
    tail_start_s: float
    annotation_signature: str | None = None

    def __post_init__(self):
        if self.annotation_signature is not None and not isinstance(self.annotation_signature, str):
            raise ValueError("annotation_signature must be a string or None")
        for key,value in asdict(self).items():
            if key == 'annotation_signature': continue
            if isinstance(value, bool): raise ValueError("Duration parameters must be numeric, not boolean")
            object.__setattr__(self, key, float(value))
        if not all(math.isfinite(x) for k,x in asdict(self).items() if k != 'annotation_signature'):
            raise ValueError("Duration parameters must be finite")
        if self.sd_log_s <= 0 or self.tail_start_s <= 0: raise ValueError("Duration spread and tail time must be positive")

    def to_dict(self): 
        return asdict(self)

    @classmethod
    def fit(cls, records): # Label-only: lognormal over reliable TRAIN unit durations, geometric tail from their p99.
        durations = np.asarray([
            s.duration_s for r in records for s in r.sentences
            if getattr(s, 'reliable', True) and math.isfinite(s.duration_s) and s.duration_s > 0
        ])
        if len(durations) < 10: raise ValueError("Duration fitting needs at least 10 reliable training sentences")
        values = np.log(durations)
        return cls(
            float(values.mean()), float(max(values.std(), .001)), float(np.quantile(durations, .99)), annotation_fingerprint(records)
        )

    @classmethod
    def for_language(cls, data_cfg, language):
        # Target language's prior, fitted on its train split: what stage 2 trains with and every cascade decodes with. 
        # The split is parsed once per process (eval's segmenter data guard reads the same records).
        return cls.fit(cached_language_records(data_cfg, language, "train"))

    @lru_cache(maxsize=128)
    def factors(self, fps: float):
        """Closing/continuation log ratios against a geometric reference with the same mean.

        Age a means a observed frames and D>=a. The final hazard repeats indefinitely.
        The leading fragment uses the residual life of this same discrete distribution.
        """
        if not math.isfinite(fps) or fps <= 0: raise ValueError("Frame rate must be finite and positive")
        k = max(2, math.ceil(self.tail_start_s * fps))
        times = np.arange(k + 1, dtype=np.float64) / fps
        with np.errstate(divide='ignore'): z = (np.log(times) - self.mu_log_s) / self.sd_log_s
        lc, ls = log_ndtr(z), log_ndtr(-z)

        def difference(a, b):
            with np.errstate(divide='ignore', invalid='ignore'): return a + np.log(-np.expm1(b - a))

        mass = np.where(z[1:] < 0, difference(lc[1:], lc[:-1]), difference(ls[:-1], ls[1:]))
        log_hazard = mass - ls[:-1]
        log_stay = ls[1:] - ls[:-1]
        tail_sum = ls[-2] - log_hazard[-1]
        remaining = np.logaddexp.accumulate(np.r_[ls[:-2], tail_sum][::-1])[::-1]
        log_mean = remaining[0]

        residual_hazard = ls[:-1] - remaining
        residual_stay = np.r_[remaining[1:] - remaining[:-1], log_stay[-1]]
        reference_end = -log_mean
        reference_stay = np.log1p(-np.exp(reference_end))
        result = (log_hazard - reference_end, log_stay - reference_stay, residual_hazard - reference_end, residual_stay - reference_stay)
        if not all(np.isfinite(x).all() for x in result): raise ValueError("Duration prior is numerically degenerate at this frame rate")
        return result


@dataclass
class BIOPathScores:
    emissions: torch.Tensor
    factors: torch.Tensor
    initial: torch.Tensor
    lengths: torch.Tensor
    prefix: torch.Tensor
    fragment: torch.Tensor   # (B,) frames 0..fragment-1 are the leading fragment and nothing else (`_leading_fragment`); 0: none
    known: torch.Tensor      # (B,) certified start; without one the first span may also open on I right after an all-O start

    def restore(self, values):
        t = values.shape[1]
        if not t: return values
        positions = torch.arange(t, device=values.device)[None]
        indices = (positions - self.prefix[:, None]).clamp(min=0)
        return values.gather(1, indices) * ((positions >= self.prefix[:, None]) & (positions < (self.prefix + self.lengths)[:, None]))


class DurationDecoder: # Max-product decoding and score preparation shared with sum-product membership.
    def __init__(self, model: DurationModel | list[DurationModel] | None = None):
        self.model = model

    def path_scores(self, logits, lengths, commit_mask=None, known_start=False, *, timestamps_s=None, min_age_states=1):
        e, n, prefix = _inputs(logits, lengths, commit_mask)
        b, t, _ = e.shape
        known = torch.as_tensor(known_start, device=e.device, dtype=torch.bool).expand(b)
        if not b or not t: return BIOPathScores(e, e.new_empty((b, 4, 0)), e.new_empty((b, 0)), n, prefix, torch.zeros_like(n), known)
        factors = _factors(n, prefix, self.model, timestamps_s, min_age_states, t).to(e)
        e, fragment = _leading_fragment(e, n, known)
        initial = torch.cat((_initial(e, known, fragment), e.new_full((b, factors.shape[-1] - 1), -torch.inf)), 1)
        return BIOPathScores(e, factors, initial, n, prefix, fragment, known)

    @torch.no_grad()
    def decode(self, logits, lengths, commit_mask=None, known_start=False, *, timestamps_s=None):
        scores = self.path_scores(logits, lengths, commit_mask, known_start, timestamps_s=timestamps_s)
        e, factors, n, prefix = scores.emissions, scores.factors, scores.lengths, scores.prefix
        b, t, _ = e.shape
        result = torch.full(logits.shape[:2], BIO['UNK'], dtype=torch.long, device=logits.device)
        if not b or not t: return result
        
        emissions = e.cpu().double().numpy()
        end, stay, rend, rstay = factors.cpu().double().numpy().transpose(1, 0, 2)
        ages = end.shape[1]
        state = np.full((b, ages + 2), -np.inf)
        state[:, :3] = scores.initial[:, :3].cpu().numpy()
        fragment = scores.fragment.cpu().numpy()
        # Without a certified start the first span may also open on I right after an all-O start (frame 0 onward), as the span rule
        # reads O -> I. `all_o` scores that start on the state's scale; `via_i` marks a first frame entered that way (tagged I).
        all_o = np.where(scores.known.cpu().numpy(), -np.inf, emissions[:, 0, 0])
        boundaries, tails, via_i = np.zeros((t, b), np.int32), np.zeros((t, b), bool), np.zeros((t, b), bool)
        sizes, offsets = n.cpu().numpy(), prefix.cpu().numpy()

        for k in range(1, t):
            ended = np.concatenate((state[:, :1],
                                    state[:, 1:2] + rend[:, min(k - 1, ages - 1):min(k - 1, ages - 1) + 1],
                                    state[:, 2:] + end), 1)
            boundaries[k] = ended.argmax(1)
            best = ended[np.arange(b), boundaries[k]]
            grown = state[:, 2:] + stay

            next_state = np.full_like(state, -np.inf)
            next_state[:, 0] = best + emissions[:, k, 0]
            # The leading fragment continues only on the head's own leading I-run (`_leading_fragment`).
            next_state[:, 1] = np.where(k < fragment, state[:, 1] + rstay[:, min(k - 1, ages - 1)] + emissions[:, k, 2], -np.inf)
            next_state[:, 2] = best + emissions[:, k, 1]
            via_i[k] = all_o + emissions[:, k, 2] > next_state[:, 2]   # B wins a tie
            next_state[:, 2] = np.maximum(next_state[:, 2], all_o + emissions[:, k, 2])

            if ages > 1:
                next_state[:, 3:] = grown[:, :-1] + emissions[:, k, 2:3]
                tails[k] = grown[:, -1] > grown[:, -2]
                next_state[:, -1] = np.maximum(grown[:, -1], grown[:, -2]) + emissions[:, k, 2]
            valid = k < sizes
            state = np.where(valid[:, None], next_state, state)
            shift = state.max(1)
            state -= shift[:, None]
            all_o = np.where(valid, all_o + emissions[:, k, 0], all_o) - shift

        for row, size in enumerate(sizes):
            if not size: continue
            q = int(state[row].argmax()); tags = []
            for k in range(int(size) - 1, -1, -1):
                tags.append(BIO['O'] if q == 0 else BIO['B'] if q == 2 and not via_i[k, row] else BIO['I'])
                if k:
                    if q == 2 and via_i[k, row]: tags += [BIO['O']] * k; break   # the all-O start of an O -> I start
                    if q in (0, 2): q = int(boundaries[k, row])
                    elif q > 2 and not (q == ages + 1 and tails[k, row]): q -= 1
            result[row, offsets[row]:offsets[row] + size] = torch.tensor(tags[::-1], device=result.device)
        return result


def argmax_tags(logits, lengths, commit_mask=None, known_start=False):
    """(B, T) tags of the argmax decode: each frame its most likely tag over every class of the head (`logits.argmax(dim=-1)`), the
    tags of the offline argmax row (moryossef26.infer.predict_phrase_segments). The FSM reads them with 2 rules of its buffer, the
    rules of `DurationDecoder.decode`: (1) the committed prefix (`commit_mask`, which must be a prefix) and the frames past each row's
    length are UNK, so no span opens again in content that the FSM committed; (2) with `known_start`, the first frame after the prefix
    is B where its argmax is I, because the committed terminator frame certifies a start there, as the legal decodes read it (an O or a
    B stays as it is). No other tag changes, so the tags need not be a legal BIO path: the span rule (infer.commit_gate._span_opens)
    opens a span on B or on O -> I."""
    _, n, prefix = _inputs(logits, lengths, commit_mask)
    position = torch.arange(logits.shape[1], device=logits.device)[None]
    tags = logits.argmax(dim=-1).masked_fill((position < prefix[:, None]) | (position >= (prefix + n)[:, None]), BIO['UNK'])
    if known_start: tags = tags.masked_fill((position == prefix[:, None]) & (tags == BIO['I']), BIO['B'])
    return tags


def _inputs(logits, lengths, commit_mask=None):
    if logits.ndim != 3 or logits.shape[-1] != 4: raise ValueError("Expected (batch, frames, UNK/O/B/I) logits")
    b, t, _ = logits.shape
    lengths = lengths.to(logits.device).long()
    if lengths.shape != (b,) or bool(((lengths < 0) | (lengths > t)).any()): raise ValueError("Invalid frame lengths")
    prefix = torch.zeros_like(lengths)

    if commit_mask is not None:
        cm = commit_mask.to(logits.device).bool()
        if cm.shape != (b, t): raise ValueError("Commit mask must match the frame axes")
        valid = torch.arange(t, device=logits.device)[None] < lengths[:, None]
        prefix = (cm & valid).sum(1)
        if bool(((cm & valid) != (torch.arange(t, device=logits.device)[None] < prefix[:, None])).any()):
            raise ValueError("Committed frames must form a prefix")

    raw = logits[..., [BIO['O'], BIO['B'], BIO['I']]].float()
    if t:
        indices = (torch.arange(t, device=logits.device)[None]+prefix[:, None]).clamp(max=t-1)
        raw = raw.gather(1, indices[..., None].expand(-1, -1, 3))

    valid = torch.arange(t, device=logits.device)[None] < (lengths - prefix)[:, None]
    raw = torch.where(valid[..., None], raw, torch.zeros_like(raw))
    if not bool(torch.isfinite(raw).all()): raise ValueError("Valid BIO logits must be finite")
    return raw.log_softmax(-1), lengths - prefix, prefix


def effective_fps(timestamps_s, lengths, prefix=None):
    if timestamps_s is None: raise ValueError("Duration decoding needs timestamps from the supplied frames")
    ts = timestamps_s.detach().cpu().double()
    if ts.ndim == 1: ts = ts[None].expand(len(lengths), -1)
    if ts.ndim != 2 or ts.shape[0] != len(lengths): raise ValueError("Timestamps must match the batch")
    prefix = torch.zeros_like(lengths) if prefix is None else prefix
    if ts.shape[1] < int((lengths + prefix).max()): raise ValueError("Timestamps do not cover the valid frames")

    rates = []
    for row, (count, start) in enumerate(zip(lengths.tolist(), prefix.tolist())):
        values = ts[row, start:start+count]
        # A singleton has no duration transition; its rate cannot affect a path score.
        if count < 2: rates.append(1.0); continue
        if not bool(torch.isfinite(values).all()) or not bool((values[1:] > values[:-1]).all()):
            raise ValueError("Valid timestamps must be finite and strictly increasing")
        rates.append((count-1)/float(values[-1]-values[0]))
    return rates


def _factors(n, prefix, duration, timestamps_s, floor, t):
    b = len(n)
    models = duration if isinstance(duration, (tuple, list)) else [duration]*b
    if len(models) != b: raise ValueError("1 duration model is required per batch row")
    sizes = n.detach().cpu().tolist()
    rates = effective_fps(timestamps_s, n, prefix) if any(m is not None and size >= 2 for m,size in zip(models,sizes)) else [1.] * b
    tables = [m.factors(f) if m is not None and size >= 2 else (np.zeros(2),) * 4 for m, f, size in zip(models, rates, sizes)]
    ages = min(max(t, 1), max(2, floor, max(len(x[0]) for x in tables)))
    return torch.as_tensor(np.stack([[x[np.arange(ages).clip(max=len(x)-1)] for x in row] for row in tables]))


def _leading_fragment(e, lengths, known):
    """With no certified start, the leading fragment (a unit that began before the buffer; the span rule never opens it, P1) is
    EXACTLY the head's own leading I-run: the frames from frame 0 whose most likely tag of O, B, I is I. O and B are closed on these
    frames, and the fragment ends where the run ends. So no path stretches it over a frame that the head reads O or B, which would
    take the unit after that frame into the fragment and leave it with no caption, and the prior cannot give it a unit that the head
    opens. A row with a certified start, or whose frame 0 the head does not read I, has no fragment. Returns (emissions, length)."""
    frames = torch.arange(e.shape[1], device=e.device)[None]
    run = ((e.argmax(-1) == 2) & (frames < lengths[:, None])).long().cumprod(1).sum(1)
    fragment = torch.where(known, torch.zeros_like(run), run)
    closed = (frames < fragment[:, None])[..., None]
    return torch.cat((e[..., :2].masked_fill(closed, -torch.inf), e[..., 2:]), -1), fragment


def _initial(e, known, fragment):
    # A certified frontier supplies start evidence. Without one a path starts in O, in B, or in the leading fragment, which exists
    # only on the head's own leading I-run (`_leading_fragment` closed O and B there).
    first_b = torch.where(known, torch.logaddexp(e[:, 0, 1], e[:, 0, 2]), e[:, 0, 1])
    leading = torch.where(fragment > 0, e[:, 0, 2], torch.full_like(first_b, -torch.inf))
    return torch.stack((e[:, 0, 0], leading, first_b), 1)
