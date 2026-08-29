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


# Per-split log-score bonus — the decode's ONE hyperparameter, a Lagrange multiplier on segment count. It must
# roughly offset the per-segment lognormal normalisation cost or the DP never/always splits; exact cancellation
# oversplits, so it can't be auto-derived. F1(bias) is a sharp peak pinned to the corpus's duration prior:
# re-tune per corpus with `analyze --stage tune-decode` and pin the pair in inference.yaml.
DURATION_SPLIT_BIAS = 4.0
# Snap radius for a split -> the model's P(B) peak. With a collapsed B head (the pre-class-weighting state) the
# emission weight below tunes to zero and snap is P(B)'s only entry point; with a live head the emission term
# subsumes it (tune-decode selects snap 0.0) — kept for the w=0 operating point.
SNAP_RADIUS_S = 1.0
# Weight on the boundary-emission term: the DP split reward becomes split_bias + w*logit(P(B)) at the candidate
# frame — a log-linear interpolation of the duration model and the head's boundary posterior (the ASR
# acoustic/LM-weight pattern). w=0 recovers the duration-only DP; w=1 is the fully Bayesian combination, which
# over-trusts an imperfectly calibrated head — tune per corpus and per segmenter.
BOUNDARY_LOGIT_WEIGHT = 0.0
DEPLOYED_SEGMENTER_ARCH = "s1"  # the in-system BIO head: what the FSM, the membership gate and RQ1/RQ2 run on


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


def duration_split_tags(
    tags: np.ndarray, boundary_prob: np.ndarray, fps: float, prior: DurationPrior,
    split_bias: float | None = None, snap_radius_s: float | None = None, boundary_logit_weight: float | None = None,
    mark_onsets: bool = True, split_open_tail: bool | str = True,
) -> np.ndarray:
    """Re-decode argmax `tags`: keep the signing/non-signing runs, re-place interior boundaries by segmental
    Viterbi under the duration prior, snap each split to the P(B) peak. New array; O/UNK untouched.

    Per run [a,b] maximise Σ_seg log LogNormal(L_seg) + (#splits)·split_bias. Argmax `B`s fold to `I` first (~1%
    prevalence, noise; the DP owns splitting), so output `B`s are exactly the DP's splits + onsets iff `mark_onsets`.

    `mark_onsets=False` (streaming) leaves onsets `I`: inert for span opening, which keys on O->signing.
    `split_open_tail`, for the run touching the LAST frame (right-censored mid-stream):
      True       whole-video — nothing censored; split like any other run.
      "survival" streaming — score its LAST segment by log P(dur > L) (a probability, no Jacobian), commit the
                 prefix's interior splits. Without it a stretch longer than the buffer never gets split.
      False      legacy — tail unsplit; starves long runs. Prefer "survival".
    """
    fps = float(fps)
    split_bias = float(prior.split_bias if split_bias is None else split_bias)
    snap_radius_s = float(prior.snap_radius_s if snap_radius_s is None else snap_radius_s)
    w_b = float(prior.boundary_logit_weight if boundary_logit_weight is None else boundary_logit_weight)
    # Boundary emission per frame: w*logit(P(B)). The full semi-Markov emission over a partition is
    # sum log P(B at splits) + sum log(1-P(B) elsewhere); the second sum is partition-independent, so it reduces
    # to this per-split logit. Clipped: P(B)=0 exactly (padding) must disfavour, not poison, the argmax.
    lb = None
    if w_b != 0.0:
        bp = np.clip(np.asarray(boundary_prob, dtype=float), 1e-6, 1.0 - 1e-6)
        lb = w_b * (np.log(bp) - np.log1p(-bp))
    # Capping at len(tags) bounds the DP to O(T) even if a degenerate timestamp stream inflates fps.
    lmax = max(2, min(round(prior.cap_s * fps), len(tags)))
    # Density over integer FRAME lengths: seconds-lognormal at L/fps plus the -log(fps) Jacobian, for a correct
    # MAP over frame-partitions. -log(fps) scales with segment count too, so the count penalty is fps-dependent —
    # another reason split_bias is per-corpus.
    L_s = np.arange(1, lmax + 1, dtype=float) / fps
    logp = -np.log(L_s * prior.sd_log_s * np.sqrt(2 * np.pi)) \
           - (np.log(L_s) - prior.mu_log_s) ** 2 / (2 * prior.sd_log_s ** 2) - np.log(fps)
    LP = np.concatenate([[-1e18], logp])  # LP[L_frames], L>=1
    LS = None
    if split_open_tail == "survival":
        # log-survival log P(dur > L) = log(0.5 * erfc((ln L - mu)/(sd*sqrt2))). Clamped: erfc underflows to 0 in
        # the far tail and log(0) would poison the argmax instead of disfavouring the length.
        z = (np.log(L_s) - prior.mu_log_s) / (prior.sd_log_s * np.sqrt(2.0))
        surv = 0.5 * torch.special.erfc(torch.as_tensor(z, dtype=torch.float64)).numpy()
        LS = np.concatenate([[0.0], np.log(np.maximum(surv, 1e-300))])  # LS[L_frames]; LS[0] unused

    out = np.array(tags, copy=True)
    # Drop the head's HARD B decisions and re-derive every interior split below. Only the argmax is discarded,
    # not the evidence: P(B) re-enters as the DP's emission term. Deliberate — at B's frame prevalence the argmax
    # is a poor estimator (it collapses outright under an unweighted loss) while the posterior still ranks
    # boundaries usefully, so scoring the soft value beats trusting the hard one. Run boundaries (O/UNK) are the
    # head's to keep; only splits WITHIN a signing run are re-decided.
    out[out == BIO["B"]] = BIO["I"]
    snap_r = max(0, round(snap_radius_s * fps))
    splits: list[int] = []
    for run in _bio_runs(out, split_on_b=False, open_on_i=True, close_on_unk=True):
        a, b = int(run["start"]), int(run["end"])
        L = b - a + 1
        # Explicit: only "survival" builds LS, so any other non-True value must take the legacy unsplit path.
        censored = split_open_tail in (False, "survival") and b == len(out) - 1
        if split_open_tail is False and censored:
            if mark_onsets: splits.append(a)
            continue
        if L <= 2:
            if mark_onsets: splits.append(a)
            continue
        D = np.full(L + 1, -1e18); D[0] = 0.0
        arg = np.zeros(L + 1, np.int32)
        # Slices, not arange+fancy-index: same numbers, no per-j temporaries. This DP runs per row per step
        # through the membership gate, so the constant factor is not free.
        # Split reward at local frame i (global a+i): split_bias plus the head's boundary evidence when tuned
        # (bias[0]=0: i=0 opens the run, it is not a split).
        bias = np.full(L + 1, float(split_bias))
        if lb is not None: bias[:L] += lb[a:a + L]
        bias[0] = 0.0
        for j in range(1, L + 1):
            i0 = max(0, j - lmax)
            cand = D[i0:j] + LP[j - i0:0:-1] + bias[i0:j]
            k = int(np.argmax(cand)); D[j] = cand[k]; arg[j] = i0 + k

        if censored:
            # Censored segment start = best fully-segmented prefix + survival of the open tail. i*=0 (whole run
            # one open sentence) is reachable only while L <= lmax, so over-long runs must split.
            i_rng = np.arange(max(0, L - lmax), L)
            cand = D[i_rng] + bias[i_rng] + LS[L - i_rng]
            j = int(i_rng[int(np.argmax(cand))])
            tail_start = j  # backtrace covers the prefix [0, j); j itself is the censored segment's opening split
        else: j = L; tail_start = 0

        raw = []  # DP split frames, snapped left-to-right below
        if tail_start > 0: raw.append(a + tail_start)
        while j > 0:
            i = int(arg[j])
            if i > 0: raw.append(a + i)
            j = i
        raw.sort()
        # Clamp each snap between its neighbours: two splits can never collapse onto one peak (dedup would drop a
        # segment) or reorder, so they stay strictly increasing, one per DP segment.
        prev = a
        for idx, p in enumerate(raw):
            nxt = raw[idx + 1] if idx + 1 < len(raw) else b + 1
            lo, hi = max(prev + 1, p - snap_r), min(nxt, p + snap_r + 1)
            q = p
            if hi > lo:
                cand = lo + int(np.argmax(boundary_prob[lo:hi]))
                # Move only if the peak beats the DP position — flat evidence must not drift to the window edge.
                if boundary_prob[cand] > boundary_prob[p]: q = cand
            splits.append(q); prev = q
        if mark_onsets: splits.append(a)  # run onset stays a segment start (first signing frame after a gap)
    for p in sorted(set(splits)): out[p] = BIO["B"]
    return out


def duration_decode_tags(logits: torch.Tensor, fps: float, prior: DurationPrior) -> torch.Tensor:
    # argmax tags -> semi-Markov re-split, with the model's own softmax P(B) as snap evidence.
    pB = torch.softmax(logits[0].float(), dim=-1)[:, BIO["B"]].cpu().numpy()
    return torch.as_tensor(duration_split_tags(logits.argmax(dim=-1)[0].cpu().numpy(), pB, fps, prior))


def deployed_decode_tags(
    bio_logits: torch.Tensor, lengths: torch.Tensor, duration_prior=None, timestamps_s: torch.Tensor | None = None,
    commit_mask: torch.Tensor | None = None, seam_is_terminator: bool = True, stream_start: bool = False,
) -> torch.Tensor:
    """THE deployed tag decode, shared with the FSM: argmax → UNK→O remap → duration re-split → χ-onset restoration.
    (B, T) long, no gradient.

    UNK→O: unremapped UNK neither opens nor closes a span → diverges from the FSM's rule.
    Re-split: same `duration_split_tags` call as `infer/stream.py step()`; `mark_onsets=False` (opening keys on
    O→signing, onset B's inert), `split_open_tail="survival"` (right-censored tail).
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

    for b in range(tags_np.shape[0]):
        n = int(lengths[b].item())
        if n <= 2: continue
        if duration_prior is not None:
            fps_b = 24.0 # Default fallback if timestamps are missing or degenerate.
            if timestamps_s is not None and n > 1:
                # Span-based rate, NOT median-of-diffs: fps-augmented windows are sub-sampled onto the native
                # 24fps lattice, so consecutive gaps are integer multiples of 1/24 and the median quantises a
                # true 15-22fps window to 12 or 24 — mis-scaling the duration prior on ~half of training. The
                # total-span estimate recovers the true rate and still gives 24 exactly on native uniform buffers.
                span = float(timestamps_s[b, n - 1] - timestamps_s[b, 0])
                if span > 1e-6: fps_b = min(max((n - 1) / span, 1.0), 120.0)

            tags_np[b, :n] = duration_split_tags(
                tags_np[b, :n], pB[b, :n], fps_b, duration_prior, mark_onsets=False, split_open_tail="survival",
            )
        # Both restorations below are FSM COMMIT-LOG facts, independent of how tags were decoded, so they run whether or 
        # not the re-split did. Under plain argmax they are needed at least as much: `B` rarely wins argmax at back-to-back
        # seam, which is the premise the whole decode exists for. Gating them on `duration_prior` left every untuned corpus 
        # (duration_decode_<arch>.default:false) emitting nothing on mid-signing stream start & dropping every b2b successor.
        # Stream-start onset, mirroring the FSM (stream.py step — same position, after the re-split and before the χ mint): 
        # on 1st buffer of a stream a leading I IS a real onset, else a mid-signing stream never opens a span. Without it 
        # the gate saw a headless I-run where the FSM sees a span, and anchored Ω on "no span", flooring the very frames 
        # the FSM was decoding. Flag, not a timestamp test: window timestamps are window-relative (train/sampler.py), so 
        # "starts at 0" is true of every training window. Training passes False — a sampled window simulates a buffer at 
        # an arbitrary stream position, and a left-truncated one is Mode 2b.
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
