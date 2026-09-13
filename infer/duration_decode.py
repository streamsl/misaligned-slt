"""Shared semi-Markov BIO scores and exact Viterbi decoding.

The legal-only S1 monitor uses the same state grammar without duration factors.
Caption membership is computed from these scores in models.membership_gate.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from functools import lru_cache
import math, hashlib, json
import numpy as np
import torch
from scipy.special import log_ndtr
from data.loader import annotation_fingerprint
from data.windowing import BIO


@dataclass(frozen=True)
class DurationModel:
    mu_log_s: float
    sd_log_s: float
    tail_start_s: float
    completion_bias: float = 0.0
    boundary_logit_weight: float = 1.0
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
        if self.sd_log_s <= 0 or self.tail_start_s <= 0 or self.boundary_logit_weight <= 0:
            raise ValueError("Duration spread, tail time and boundary weight must be positive")

    def to_dict(self): 
        return asdict(self)

    @property
    def signature(self): # Score parameters and the annotations used to fit them.
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def fit(cls, records, completion_bias=0.0, boundary_logit_weight=1.0):
        durations = np.asarray([
            s.duration_s for r in records for s in r.sentences
            if getattr(s, 'reliable', True) and math.isfinite(s.duration_s) and s.duration_s > 0
        ])
        if len(durations) < 10: raise ValueError("Duration fitting needs at least 10 reliable training sentences")
        values = np.log(durations)
        return cls(
            float(values.mean()), float(max(values.std(), .001)), float(np.quantile(durations, .99)),
            float(completion_bias), float(boundary_logit_weight), annotation_fingerprint(records)
        )

    def require_annotations(self, records):
        if self.annotation_signature != annotation_fingerprint(records):
            raise ValueError("Duration calibration uses different or unstamped annotations; retrain segmenter & rerun tune-decode on dev.")

    @classmethod
    def from_config(cls, cfg, language, arch='s1', required=True):
        entry = (cfg.get('duration_model', {}).get(arch, {}) or {}).get(str(language))
        if entry is None:
            if required: raise ValueError(f"Missing duration_model.{arch}.{language}; run analyze.py --stage tune-decode on dev first")
            return None
        return cls(**entry)

    def require_calibration(self, cfg, language):
        signatures = (cfg.get("boundary_stability", {}).get("duration_model_signature") or {})
        if signatures.get(str(language)) != self.signature:
            raise ValueError("Boundary tolerance belongs to a different decoder; run delta-enc --write-config after tune-decode")

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
    bonus: torch.Tensor
    initial: torch.Tensor
    lengths: torch.Tensor
    prefix: torch.Tensor

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
        if not b or not t: return BIOPathScores(e, e.new_empty((b, 4, 0)), e.new_zeros((b, 1)), e.new_empty((b, 0)), n, prefix)
        raw_initial = _initial(e, known_start)
        e, factors, bonus = _factors(e, n, prefix, self.model, timestamps_s, min_age_states)
        outside, leading, first_b = _initial(e, known_start)
        known = torch.as_tensor(known_start, device=e.device, dtype=torch.bool).expand(b)
        first_b = torch.where(known, raw_initial[2], first_b)
        initial = torch.cat((torch.stack((outside, leading, first_b), 1), e.new_full((b, factors.shape[-1] - 1), -torch.inf)), 1)
        return BIOPathScores(e, factors, bonus, initial, n, prefix)

    @torch.no_grad()
    def decode(self, logits, lengths, commit_mask=None, known_start=False, *, timestamps_s=None):
        scores = self.path_scores(logits, lengths, commit_mask, known_start, timestamps_s=timestamps_s)
        e, factors, bonus, n, prefix = scores.emissions, scores.factors, scores.bonus, scores.lengths, scores.prefix
        b, t, _ = e.shape
        result = torch.full(logits.shape[:2], BIO['UNK'], dtype=torch.long, device=logits.device)
        if not b or not t: return result
        
        emissions = e.cpu().double().numpy()
        end, stay, rend, rstay = factors.cpu().double().numpy().transpose(1, 0, 2)
        bonus = bonus.cpu().double().numpy()
        ages = end.shape[1]
        state = np.full((b, ages + 2), -np.inf)
        state[:, :3] = scores.initial[:, :3].cpu().numpy()
        boundaries, tails = np.zeros((t, b), np.int32), np.zeros((t, b), bool)
        sizes, offsets = n.cpu().numpy(), prefix.cpu().numpy()

        for k in range(1, t):
            ended = np.concatenate((state[:, :1],
                                    state[:, 1:2] + rend[:, min(k - 1, ages - 1):min(k - 1, ages - 1) + 1],
                                    state[:, 2:] + end + bonus), 1)
            boundaries[k] = ended.argmax(1)
            best = ended[np.arange(b), boundaries[k]]
            grown = state[:, 2:] + stay

            next_state = np.full_like(state, -np.inf)
            next_state[:, 0] = best + emissions[:, k, 0]
            next_state[:, 1] = state[:, 1] + rstay[:, min(k - 1, ages - 1)] + emissions[:, k, 2]
            next_state[:, 2] = best + emissions[:, k, 1]

            if ages > 1:
                next_state[:, 3:] = grown[:, :-1] + emissions[:, k, 2:3]
                tails[k] = grown[:, -1] > grown[:, -2]
                next_state[:, -1] = np.maximum(grown[:, -1], grown[:, -2]) + emissions[:, k, 2]
            state = np.where((k < sizes)[:, None], next_state, state)
            state -= state.max(1)[:, None]

        for row, size in enumerate(sizes):
            if not size: continue
            q = int(state[row].argmax()); tags = []
            for k in range(int(size) - 1, -1, -1):
                tags.append(BIO['O'] if q == 0 else BIO['B'] if q == 2 else BIO['I'])
                if k:
                    if q in (0, 2): q = int(boundaries[k, row])
                    elif q > 2 and not (q == ages + 1 and tails[k, row]): q -= 1
            result[row, offsets[row]:offsets[row] + size] = torch.tensor(tags[::-1], device=result.device)
        return result


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


def _factors(e, n, prefix, duration, timestamps_s, floor=1):
    b, t, _ = e.shape
    models = duration if isinstance(duration, (tuple, list)) else [duration]*b
    if len(models) != b: raise ValueError("1 duration model is required per batch row")
    sizes = n.detach().cpu().tolist()
    rates = effective_fps(timestamps_s, n, prefix) if any(m is not None and size >= 2 for m,size in zip(models,sizes)) else [1.] * b
    tables = [m.factors(f) if m is not None and size >= 2 else (np.zeros(2),) * 4 for m, f, size in zip(models, rates, sizes)]
    ages = min(max(t, 1), max(2, floor, max(len(x[0]) for x in tables)))
    arrays = np.stack([[x[np.arange(ages).clip(max=len(x)-1)] for x in row] for row in tables])
    weight = e.new_tensor([m.boundary_logit_weight if m else 1. for m in models])[:, None]
    bonus = e.new_tensor([m.completion_bias if m else 0. for m in models])[:, None]
    # B evidence is charged at every visible start, including after an O gap.
    emissions = torch.stack((e[..., 0], e[..., 2] + weight * (e[..., 1] - e[..., 2]), e[..., 2]), -1)
    return emissions, e.new_tensor(arrays), bonus


def _initial(e, known_start):
    b = len(e)
    known = torch.as_tensor(known_start, device=e.device, dtype=torch.bool).expand(b)
    # A certified frontier supplies start evidence; unknown initial I remains a leading fragment.
    first_b = torch.where(known, torch.logaddexp(e[:, 0, 1], e[:, 0, 2]), e[:, 0, 1])
    leading = torch.where(known, torch.full_like(first_b, -torch.inf), e[:, 0, 2])
    return e[:, 0, 0], leading, first_b
