"""Semi-Markov segment decode — this system's contribution, not part of the Moryossef protocol.

Argmax BIO under-segments caption-supervised corpora: adjacent captions are back-to-back (no `O` gap), argmax
rarely wins `B` there, and a merged run of k captions matches at most one under 1-to-1 tIoU. Fixed at decode
time — runs are reliable, only the split POSITIONS and COUNT are missing; the duration prior supplies the
length model and `boundary_logit_weight` lets the head's own boundary evidence set the count
(`duration_split_tags`), no model change.

One tuned block PER SEGMENTER in inference.yaml (`duration_decode_<arch>`, resolved by `duration_decode_params`):
two heads have differently calibrated posteriors, so one shared triple would decode the baseline with our head's
tuning. `duration_decode_s1` is the DEPLOYED one and drives `infer.stream.step`, the gate's anchor selection
(`build_gate_omega`, keeping the gate on-policy), whole-video eval, and `analyze --stage delta-enc` — which must
use the deployed decode or it reports argmax instability. `duration_decode_moryossef` drives calibration and
the external segmenter's own evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from data.loader import VideoRecord
from data.windowing import BIO
from metrics import _bio_runs


# Per-split log-score bonus — the decode's ONE hyperparameter, a Lagrange multiplier on segment count. It must roughly 
# offset the per-segment lognormal normalisation cost or the DP never/always splits; exact cancellation oversplits, so it 
# can't be auto-derived. F1(bias) is a sharp peak pinned to the corpus's duration prior: re-tune per corpus with 
# `analyze --stage tune-decode` and pin the pair in inference.yaml.
DURATION_SPLIT_BIAS = 4.0
# Snap radius for a split -> the model's P(B) peak. With a collapsed B head (the pre-class-weighting state) the emission 
# weight below tunes to 0 and snap is P(B)'s only entry point; with a live head the emission term subsumes it (tune-decode 
# selects snap 0.0) — kept for the w=0 operating point.
SNAP_RADIUS_S = 1.0
# Weight on boundary-emission term: DP split reward becomes split_bias + w*logit(P(B)) at candidate frame — a log-linear 
# interpolation of duration model & head's boundary posterior (ASR acoustic/LM-weight pattern). w=0 recovers duration-only 
# DP; w=1 is fully Bayesian combination, which over-trusts imperfectly calibrated head — tune per corpus and per segmenter.
BOUNDARY_LOGIT_WEIGHT = 0.0
DEPLOYED_SEGMENTER_ARCH = "s1"  # the in-system BIO head: what the FSM, the membership gate and RQ1/RQ2 run on
STREAM_DECODE_ARCH = "s1_stream"  # inference.yaml duration_decode_s1_stream: FSM's triple, chosen under streaming decode
STREAM_DECODE_KEY = "duration_decode_" + STREAM_DECODE_ARCH


@dataclass(frozen=True)
class DurationPrior:
    # Lognormal sentence-duration prior in LOG-SECONDS (fps-independent), closed-form moment fit — 
    # nothing trained. Carries 2 tuned hyperparameters so every consumer decodes with the same pair.
    mu_log_s: float
    sd_log_s: float
    cap_s: float
    split_bias: float = DURATION_SPLIT_BIAS
    snap_radius_s: float = SNAP_RADIUS_S
    boundary_logit_weight: float = BOUNDARY_LOGIT_WEIGHT


def _coerce_decode_entry(v) -> dict | None:
    # one entry -> None (off) | {} (module defaults) | {split_bias?, snap_radius_s?, boundary_logit_weight?} (tuned).
    if isinstance(v, dict):
        return {k: float(v[k]) for k in ("split_bias", "snap_radius_s", "boundary_logit_weight") if v.get(k) is not None}
    return {} if bool(v) else None


def decode_config_key(arch: str | None = None) -> str:
    """Config block holding the tuned decode for a segmenter: `duration_decode_<arch>`.

    The suffix is the `--segmenter-arch` value verbatim, so the block a run reads is derivable from its command
    line. Each segmenter needs its OWN triple: the decode is fit to that head's posterior calibration, so
    decoding the baseline with the triple tuned for ours would tune the baseline on our method's behalf — the
    unfair-baseline objection. Defaults to the deployed head.
    """
    return f"duration_decode_{arch or DEPLOYED_SEGMENTER_ARCH}"


def duration_decode_params(inference_cfg: dict, language: str | None = None, arch: str | None = None) -> dict | None:
    """Resolve a segmenter's tuned decode for a language -> None (off) | {} (on, defaults) | {tuned triple}.

    Reads `duration_decode_<arch>` (default arch: the deployed head). Falls back to a bare `duration_decode` block
    when the arch-specific one is absent, so a single-segmenter or not-yet-tuned corpus still runs.
    """
    cfg = inference_cfg or {}
    key = decode_config_key(arch)
    return _duration_decode_entry(cfg, key if key in cfg else "duration_decode", language)


def _duration_decode_entry(inference_cfg: dict, key: str, language: str | None = None) -> dict | None:
    """Resolve one decode block for a language.

    split_bias is corpus-specific, so one global value is wrong. Shapes:
      false / true                             -> off / on-with-defaults
      {split_bias: .., snap_radius_s: .., boundary_logit_weight: ..} -> one tuned triple (any subset allowed)
      {default: <entry>, <lang>: <entry>, ...} -> per-language (any other key); untuned languages fall to
                                                  `default` — keep it false until tune-decode pins a triple.
    """
    v = (inference_cfg or {}).get(key, False)
    if isinstance(v, dict) and not ({"split_bias", "snap_radius_s", "boundary_logit_weight"} & set(v)):
        entry = v.get(str(language), v.get("default", False)) if language is not None else v.get("default", False)
        return _coerce_decode_entry(entry)
    return _coerce_decode_entry(v)


def streaming_decode_params(inference_cfg: dict, language: str | None = None) -> tuple[dict | None, str]:
    """The triple the FSM (and the gate's anchor decode, which runs the same censored-tail decode) deploys.

    The streaming decode is a different estimator from the whole-video one: the survival tail changes the split economics,
    so `analyze --stage tune-stream` selects `duration_decode_s1_stream.<lang>` on dev under the FSM itself. Until that 
    row exists the whole-video triple is used and the caller says so. Returns (params, source_block)."""
    block = (inference_cfg or {}).get(STREAM_DECODE_KEY)
    if block is not None:
        params = _duration_decode_entry(inference_cfg, STREAM_DECODE_KEY, language)
        if params is not None: return params, STREAM_DECODE_KEY
        # An explicit per-language `false` is OFF (an arm trained without a prior deploys without one).
        if isinstance(block, dict) and language is not None and str(language) in block: return None, STREAM_DECODE_KEY
    return duration_decode_params(inference_cfg, language), decode_config_key(DEPLOYED_SEGMENTER_ARCH)


def fit_duration_prior(
    records: list[VideoRecord], split_bias: float | None = None, snap_radius_s: float | None = None,
    boundary_logit_weight: float | None = None,
) -> DurationPrior | None:
    # Fit the lognormal on a TRAIN split's caption durations (filter mirrors data.yaml bounds). `split_bias` /
    # `snap_radius_s`: per-language overrides from `duration_decode_params`; None -> module defaults.
    durs = np.asarray([
        s.duration_s for r in records for s in r.sentences if getattr(s, "reliable", True) and 0.1 < s.duration_s < 60.0
    ])
    if durs.size < 10: return None
    logd = np.log(durs)
    return DurationPrior(
        float(logd.mean()), float(max(logd.std(), 1e-3)), float(min(np.quantile(durs, 0.999), 45.0)),
        split_bias=DURATION_SPLIT_BIAS if split_bias is None else float(split_bias),
        snap_radius_s=SNAP_RADIUS_S if snap_radius_s is None else float(snap_radius_s),
        boundary_logit_weight=BOUNDARY_LOGIT_WEIGHT if boundary_logit_weight is None else float(boundary_logit_weight),
    )


def _boundary_logit(boundary_prob: np.ndarray) -> np.ndarray:
    # logit P(B) per frame, the DP's emission term (times w). The full semi-Markov emission over a partition is
    # sum log P(B at splits) + sum log(1-P(B) elsewhere); the second sum is partition-independent. Clipped so P(B)=0
    # (padding) disfavours a split instead of poisoning the argmax.
    bp = np.clip(np.asarray(boundary_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(bp) - np.log1p(-bp)


def _duration_tables(
    prior: DurationPrior, fps: float, n_frames: int, survival: bool
) -> tuple[int, np.ndarray, np.ndarray | None]:
    """(lmax, LP, LS): log-density and log-survival of the duration prior over integer FRAME lengths 1..lmax.

    LP is the seconds-lognormal at L/fps plus the -log(fps) Jacobian (a correct MAP over frame partitions; the count 
    penalty is therefore fps-dependent, one reason split_bias is per corpus). lmax is capped at n_frames so a degenerate 
    fps cannot blow the DP up. LS = log P(dur > L) for the right-censored tail, clamped where erfc underflows."""
    lmax = max(2, min(round(prior.cap_s * fps), int(n_frames)))
    L_s = np.arange(1, lmax + 1, dtype=float) / fps
    logp = -np.log(L_s * prior.sd_log_s * np.sqrt(2 * np.pi)) \
           - (np.log(L_s) - prior.mu_log_s) ** 2 / (2 * prior.sd_log_s ** 2) - np.log(fps)
    LP = np.concatenate([[-1e18], logp])  # LP[L_frames], L>=1
    LS = None
    if survival:
        z = (np.log(L_s) - prior.mu_log_s) / (prior.sd_log_s * np.sqrt(2.0))
        surv = 0.5 * torch.special.erfc(torch.as_tensor(z, dtype=torch.float64)).numpy()
        LS = np.concatenate([[0.0], np.log(np.maximum(surv, 1e-300))])  # LS[L_frames]; LS[0] unused
    return lmax, LP, LS


def _signing_runs(tags: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    # Fold the head's HARD B decisions to I (the DP re-derives every interior split from the P(B) evidence; only the
    # argmax is discarded) and return the signing runs whose interior the DP re-decides. O/UNK run boundaries are kept.
    base = np.array(tags, copy=True)
    base[base == BIO["B"]] = BIO["I"]
    return base, _bio_runs(base, split_on_b=False, open_on_i=True, close_on_unk=True)


def _backtrace(arg, j: int, a: int, tail_start: int) -> list[int]:
    # Global split frames of 1 run from DP argmax chain, ascending; `tail_start` (>0) is the censored tail's opening split.
    raw = [a + tail_start] if tail_start > 0 else []
    while j > 0:
        i = int(arg[j])
        if i > 0: raw.append(a + i)
        j = i
    raw.sort()
    return raw


def _snap_splits(raw: list[int], a: int, b: int, boundary_prob: np.ndarray, snap_r: int) -> list[int]:
    # Snap each split to the P(B) peak within snap_r, clamped between its neighbours so splits never collapse or reorder;
    # move only if the peak beats the DP position (flat evidence must not drift to the window edge).
    out, prev = [], a
    for idx, p in enumerate(raw):
        nxt = raw[idx + 1] if idx + 1 < len(raw) else b + 1
        lo, hi = max(prev + 1, p - snap_r), min(nxt, p + snap_r + 1)
        q = p
        if hi > lo:
            cand = lo + int(np.argmax(boundary_prob[lo:hi]))
            if boundary_prob[cand] > boundary_prob[p]: q = cand
        out.append(q); prev = q
    return out


def duration_split_tags(
    tags: np.ndarray, boundary_prob: np.ndarray, fps: float, prior: DurationPrior,
    split_bias: float | None = None, snap_radius_s: float | None = None, boundary_logit_weight: float | None = None,
    mark_onsets: bool = True, split_open_tail: bool | str = True, closed_tail_out: list[int] | None = None,
) -> np.ndarray:
    """Re-decode argmax `tags`: keep the signing/non-signing runs, re-place interior boundaries by segmental
    Viterbi under the duration prior, snap each split to the P(B) peak. New array; O/UNK untouched.

    Per run [a,b] maximise Σ_seg log LogNormal(L_seg) + Σ_splits (split_bias + w·logit P(B)). Output `B`s are exactly the
    DP's splits, plus the run onsets iff `mark_onsets` (False in streaming: opening keys on O->signing).
    `split_open_tail`, for the run touching the LAST frame (right-censored mid-stream):
      True       whole-video — nothing censored; split like any other run.
      "survival" streaming — score its LAST segment by log P(dur > L), commit the prefix's interior splits.
    `closed_tail_out` (survival mode): also receives the splits the CLOSED-tail reading of the censored run would place —
    the same DP table, a second backtrace from the run end — so the agreement rule needs no second decode.
    One (w, bias, radius) cell of `duration_split_grid` (which batches the DP over a whole grid).
    """
    fps = float(fps)
    split_bias = float(prior.split_bias if split_bias is None else split_bias)
    snap_r = max(0, round(float(prior.snap_radius_s if snap_radius_s is None else snap_radius_s) * fps))
    w_b = float(prior.boundary_logit_weight if boundary_logit_weight is None else boundary_logit_weight)
    lb = w_b * _boundary_logit(boundary_prob) if w_b != 0.0 else None
    lmax, LP, LS = _duration_tables(prior, fps, len(tags), survival=(split_open_tail == "survival"))
    out, runs = _signing_runs(tags)
    splits: list[int] = []
    for run in runs:
        a, b = int(run["start"]), int(run["end"])
        L = b - a + 1
        censored = split_open_tail == "survival" and b == len(out) - 1
        if L <= 2:
            if mark_onsets: splits.append(a)
            continue
        # Split reward at local frame i: split_bias plus the head's boundary evidence (bias[0]=0: i=0 opens the run).
        bias = np.full(L + 1, split_bias)
        if lb is not None: bias[:L] += lb[a:a + L]
        bias[0] = 0.0
        D = np.full(L + 1, -1e18); D[0] = 0.0
        arg = np.zeros(L + 1, np.int32)
        for j in range(1, L + 1):  # slices, not fancy indexing: this DP runs per stride inside the FSM and the gate
            i0 = max(0, j - lmax)
            cand = D[i0:j] + LP[j - i0:0:-1] + bias[i0:j]
            k = int(np.argmax(cand)); D[j] = cand[k]; arg[j] = i0 + k
        if censored:
            # Censored tail start = best fully-segmented prefix + survival of the open tail; i*=0 
            # (whole run one open sentence) is reachable only while L <= lmax, so over-long runs must split.
            i_rng = np.arange(max(0, L - lmax), L)
            j = int(i_rng[int(np.argmax(D[i_rng] + bias[i_rng] + LS[L - i_rng]))])
            tail_start = j
            if closed_tail_out is not None:
                closed_tail_out.extend(_snap_splits(_backtrace(arg, L, a, 0), a, b, boundary_prob, snap_r))
        else: j, tail_start = L, 0
        splits.extend(_snap_splits(_backtrace(arg, j, a, tail_start), a, b, boundary_prob, snap_r))
        if mark_onsets: splits.append(a)
    for p in sorted(set(splits)): out[p] = BIO["B"]
    return out


def duration_split_grid(
    tags: np.ndarray, boundary_prob: np.ndarray, fps: float, prior: DurationPrior, split_biases: list[float], 
    boundary_logit_weights: list[float], snap_radii_s: list[float], device: torch.device | str | None = None,
) -> dict[tuple[float, float, float], np.ndarray]:
    """`duration_split_tags` over a whole (w, split_bias, snap_radius) grid in ONE pass — the tune-decode sweep.

    Same recurrence as the scalar path, batched over the (w, bias) cells as 1 max-plus step per frame (float64, first-max
    tie-break, so every cell equals its scalar call: tests/test_duration_decode_grid.py); backtrace and snap run per cell.
    Whole-video semantics only (`mark_onsets=True`, `split_open_tail=True`). Return {(w, split_bias, snap_radius_s): tags}.
    """
    fps = float(fps)
    if device is None: device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cells = [(float(w), float(b)) for w in boundary_logit_weights for b in split_biases]
    C = len(cells)
    logit = _boundary_logit(boundary_prob)
    lmax, LP, _ = _duration_tables(prior, fps, len(tags), survival=False)
    LPrev = torch.as_tensor(LP[::-1].copy(), dtype=torch.float64, device=device)  # LP[m] == LPrev[lmax - m]
    w_col = np.array([w for w, _ in cells], dtype=float)[:, None]
    b_col = np.array([b for _, b in cells], dtype=float)[:, None]
    base, runs = _signing_runs(tags)
    per_cell_splits: dict[tuple[float, float, float], list[int]] = {
        (w, b, float(r)): [] for w, b in cells for r in snap_radii_s
    }
    for run in runs:
        a, b_end = int(run["start"]), int(run["end"])
        L = b_end - a + 1
        if L <= 2:
            for key in per_cell_splits: per_cell_splits[key].append(a)
            continue
        bias = np.repeat(b_col, L + 1, axis=1)   # split_bias + w*logit, as the scalar path (w=0 adds exact zeros)
        bias[:, :L] += w_col * logit[a:a + L][None, :]
        bias[:, 0] = 0.0
        bias_t = torch.as_tensor(bias, dtype=torch.float64, device=device)
        D = torch.full((C, L + 1), -1e18, dtype=torch.float64, device=device); D[:, 0] = 0.0
        arg = torch.zeros((C, L + 1), dtype=torch.long, device=device)
        
        for j in range(1, L + 1):
            i0 = max(0, j - lmax)
            cand = D[:, i0:j] + LPrev[lmax - (j - i0):lmax].unsqueeze(0) + bias_t[:, i0:j]
            k = cand.argmax(dim=1)
            D[:, j] = cand.gather(1, k.view(C, 1)).squeeze(1)
            arg[:, j] = i0 + k
        arg_np = arg.cpu().numpy()

        for c, (w, b) in enumerate(cells):
            raw = _backtrace(arg_np[c], L, a, 0)
            for r in snap_radii_s:
                splits = per_cell_splits[(w, b, float(r))]
                splits.extend(_snap_splits(raw, a, b_end, boundary_prob, max(0, round(float(r) * fps))))
                splits.append(a)

    out: dict[tuple[float, float, float], np.ndarray] = {}
    for key, splits in per_cell_splits.items():
        t = base.copy()
        for p in sorted(set(splits)): t[p] = BIO["B"]
        out[key] = t
    return out


def streaming_split_tags(
    tags: np.ndarray, boundary_prob: np.ndarray, fps: float, prior: DurationPrior, delta_frames: int
) -> np.ndarray:
    """Re-split ONE buffer the way the FSM reads it: the survival decode for every interior split, plus the agreement
    rule for the split that OPENS the right-censored tail (the sentence still in progress at the buffer edge).

    A censored tail costs only log P(dur > L), which is ~0 while the tail is short, so the survival DP alone opens a new
    sentence just before the buffer edge on weak evidence (far lower P(B) than the same split needs in whole-video decode) 
    and the sentence before it is committed in pieces. Opening split thus counts as terminator only when the closed-tail 
    decode (Tail scored as finished sentence of its current length) also places a split within `delta_frames`; otherwise 
    it is folded back to I, the sentence stays open and the FSM waits for more of the tail. A terminator is then the MAP 
    boundary under both readings of the unknown tail length. Interior splits of the prefix are the survival decode's.
    """
    closed: list[int] = []
    surv = duration_split_tags(
        tags, boundary_prob, fps, prior, mark_onsets=False, split_open_tail="survival", closed_tail_out=closed
    )
    base, runs = _signing_runs(tags)
    if not runs or int(runs[-1]["end"]) != len(base) - 1: return surv   # no run touches the edge: nothing is censored
    a, b = int(runs[-1]["start"]), int(runs[-1]["end"])
    tail_splits = [i for i in range(a + 1, b + 1) if surv[i] == BIO["B"]]
    if not tail_splits: return surv
    p = max(tail_splits)   # the split that opens the censored tail (the survival DP's tail_start)
    lo, hi = max(a + 1, p - int(delta_frames)), min(b, p + int(delta_frames))
    if not any(lo <= q <= hi for q in closed): surv[p] = BIO["I"]
    return surv


def duration_decode_tags(logits: torch.Tensor, fps: float, prior: DurationPrior) -> torch.Tensor:
    # argmax tags -> semi-Markov re-split, with the model's own softmax P(B) as snap evidence.
    pB = torch.softmax(logits[0].float(), dim=-1)[:, BIO["B"]].cpu().numpy()
    return torch.as_tensor(duration_split_tags(logits.argmax(dim=-1)[0].cpu().numpy(), pB, fps, prior))


def window_fps(timestamps_s: torch.Tensor | None, lengths: torch.Tensor, default: float = 24.0) -> torch.Tensor:
    """Per-row frame rate (B,) float64 from the total timestamp span; `default` when timestamps are missing/degenerate.

    Span-based, NOT median-of-diffs: fps-augmented windows are sub-sampled onto the native 24fps lattice, so consecutive
    gaps are integer multiples of 1/24 and the median quantises a true 15-22fps window to 12 or 24 — mis-scaling the
    duration prior on ~half of training. The total span recovers the true rate and gives 24 exactly on native buffers.
    """
    B = int(lengths.shape[0])
    fps = torch.full((B,), float(default), dtype=torch.float64)
    if timestamps_s is None: return fps
    for b in range(B):
        n = int(lengths[b].item())
        if n > 1:
            span = float(timestamps_s[b, n - 1] - timestamps_s[b, 0])
            if span > 1e-6: fps[b] = min(max((n - 1) / span, 1.0), 120.0)
    return fps


@torch.no_grad()
def deployed_decode_tags(
    bio_logits: torch.Tensor, lengths: torch.Tensor, duration_prior=None, timestamps_s: torch.Tensor | None = None,
    commit_mask: torch.Tensor | None = None, seam_is_terminator: bool = True, stream_start: bool = False, *, delta: int
) -> torch.Tensor:
    """THE deployed tag decode, shared with the FSM: argmax → UNK→O remap → duration re-split → χ-onset restoration.
    (B, T) long, no gradient.

    UNK→O: unremapped UNK neither opens nor closes a span → diverges from the FSM's rule.
    Re-split: the same `streaming_split_tags` call as `infer/stream.py step()` (survival decode for interior splits, the
    agreement rule for the split that opens the censored tail, `delta` = its tolerance).
    χ restoration: the DP never opens at the ≤δ committed-leftover seam, so a b2b successor's onset lands in unopenable
    leading I and is dropped (alternate-sentence drop); the commit record proves a sentence ENDED there → mark the first
    mid-run signing frame at/after the seam B. Only under `seam_is_terminator` (= FSM `_committed_is_terminator`), else
    B at an open-span buffer-cap cut fabricates a boundary. Default True fits TRAINING (χ = sampler GT commit_mask);
    streaming's `_stride_omega` passes the runner's live flag.
    """
    device = bio_logits.device
    tags = bio_logits.detach().argmax(dim=-1)
    tags = torch.where(tags == BIO["UNK"], torch.full_like(tags, BIO["O"]), tags)
    pB = torch.softmax(bio_logits.detach().float(), dim=-1)[..., BIO["B"]].cpu().numpy()
    tags_np = tags.cpu().numpy()
    fps = window_fps(timestamps_s, lengths)

    for b in range(tags_np.shape[0]):
        n = int(lengths[b].item())
        if n <= 2: continue
        if duration_prior is not None:
            tags_np[b, :n] = streaming_split_tags(
                tags_np[b, :n], pB[b, :n], float(fps[b]), duration_prior, delta_frames=int(delta)
            )
        # Both restorations below are FSM COMMIT-LOG facts, independent of how tags were decoded, so they run whether or 
        # not the re-split did. Under plain argmax they are needed at least as much: `B` rarely wins argmax at back-to-back
        # seam, which is the premise the whole decode exists for. Gating them on `duration_prior` left every untuned corpus 
        # (duration_decode_<arch>.default:false) emitting nothing on mid-signing stream start & dropping every b2b successor.
        # Stream-start onset, mirroring the FSM (stream.py step — same position, after the re-split and before the χ mint): 
        # on 1st buffer of a stream a leading I IS a real onset, else a mid-signing stream never opens a span. Without it 
        # the gate saw a headless I-run where the FSM sees a span, and anchored Ω on "no span", flooring the very frames 
        # the FSM was decoding. Flag, not a timestamp test: window timestamps are window-relative (train/sampler.py), so 
        # "starts at 0" is true of every training window. Training passes False: every training window carries a commit 
        # mask, so a leading I is minted below through the χ seam (seam 0 when nothing is committed), exactly as after a 
        # terminator commit; only buffers without a commit log (the FSM's first buffer, cascade windows) need this flag.
        if stream_start and tags_np[b, 0] == BIO["I"]: tags_np[b, 0] = BIO["B"]
        if commit_mask is not None and seam_is_terminator:
            cm = commit_mask[b, :n].detach().cpu().numpy().astype(bool)
            # Guard matches the FSM's (stream.py step): `seam_is_terminator` is only ever True after a terminator
            # commit, so the seam is live even when the delta-frame overlap rounds to ZERO committed frames in this
            # buffer (cm all-False -> seam = frame 0). Requiring cm.any() here made the gate skip the mint on
            # exactly those strides while the FSM applied it — the successor sentence then had no Omega anchor.
            if not cm.all():
                seam = int(np.argmax(~cm))  # first non-committed frame (0 when the overlap is empty)
                sig = (tags_np[b, :n] == BIO["I"]) | (tags_np[b, :n] == BIO["B"])
                cand = np.flatnonzero(sig[seam:])
                if cand.size:
                    d = seam + int(cand[0])
                    if tags_np[b, d] == BIO["I"] and (d == 0 or sig[d - 1]): tags_np[b, d] = BIO["B"]
    return torch.as_tensor(tags_np, device=device, dtype=tags.dtype)
