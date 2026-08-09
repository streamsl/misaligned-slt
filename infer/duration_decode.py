"""Semi-Markov (HSMM) duration decode — this system's contribution, not part of the Moryossef protocol.

Argmax BIO under-segments caption-supervised corpora: adjacent captions are back-to-back (no `O` gap), argmax
never wins `B` there, and a merged run of k captions matches at most one under 1-to-1 tIoU. Fix at inference,
not learning — runs are reliable, only sentence LENGTH is missing; a lognormal duration prior supplies it
(`duration_split_tags`), no model change.

Must be semi-Markov: a linear-chain CRF/HMM implies a GEOMETRIC duration law (memoryless, monotone-decreasing)
whose Viterbi never splits under flat `B` evidence; an interior-mode law needs segment LENGTHS as hypotheses —
the O(T·L_max) DP below.

Consumers, all keyed on the one `inference.yaml duration_decode` switch (`duration_decode_params`): whole-video
eval + Analysis A/B (`moryossef26.infer`); `infer.stream.step` and the gate's anchor selection
(`build_gate_omega`) with streaming flags — this keeps the gate on-policy; `analyze --stage delta-enc`, which
must use the deployed decode or it reports argmax instability.
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
# Snap radius for a split -> the model's P(B) peak: duration decides HOW MANY, P(B) only WHERE. Weak evidence
# perturbs positions, never counts (folding P(B) into the DP objective tunes to zero weight).
SNAP_RADIUS_S = 1.0


@dataclass(frozen=True)
class DurationPrior:
    # Lognormal sentence-duration prior in LOG-SECONDS (fps-independent), closed-form moment fit — 
    # nothing trained. Carries 2 tuned hyperparameters so every consumer decodes with the same pair.
    mu_log_s: float
    sd_log_s: float
    cap_s: float
    split_bias: float = DURATION_SPLIT_BIAS
    snap_radius_s: float = SNAP_RADIUS_S


def _coerce_decode_entry(v) -> dict | None:
    # one entry -> None (off) | {} (module defaults) | {split_bias?, snap_radius_s?} (tuned).
    if isinstance(v, dict): return {k: float(v[k]) for k in ("split_bias", "snap_radius_s") if v.get(k) is not None}
    return {} if bool(v) else None


def duration_decode_params(inference_cfg: dict, language: str | None = None) -> dict | None:
    """Resolve `duration_decode` for a language -> None (off) | {} (on, defaults) | {split_bias, snap_radius_s}.

    split_bias is corpus-specific, so one global value is wrong. Shapes:
      false / true                             -> off / on-with-defaults
      {split_bias: .., snap_radius_s: ..}      -> one tuned pair
      {default: <entry>, <lang>: <entry>, ...} -> per-language (any other key); untuned languages fall to
                                                  `default` — keep it false until tune-decode pins a pair.
    """
    v = (inference_cfg or {}).get("duration_decode", False)
    if isinstance(v, dict) and not ({"split_bias", "snap_radius_s"} & set(v)):
        entry = v.get(str(language), v.get("default", False)) if language is not None else v.get("default", False)
        return _coerce_decode_entry(entry)
    return _coerce_decode_entry(v)


def fit_duration_prior(
    records: list[VideoRecord], split_bias: float | None = None, snap_radius_s: float | None = None,
) -> DurationPrior | None:
    # Fit the lognormal on a TRAIN split's caption durations (filter mirrors data.yaml bounds). `split_bias` /
    # `snap_radius_s`: per-language overrides from `duration_decode_params`; None -> module defaults.
    durs = np.asarray([s.duration_s for r in records for s in r.sentences if 0.1 < s.duration_s < 60.0])
    if durs.size < 10: return None
    logd = np.log(durs)
    return DurationPrior(
        float(logd.mean()), float(max(logd.std(), 1e-3)), float(min(np.quantile(durs, 0.999), 45.0)),
        split_bias=DURATION_SPLIT_BIAS if split_bias is None else float(split_bias),
        snap_radius_s=SNAP_RADIUS_S if snap_radius_s is None else float(snap_radius_s),
    )


def duration_split_tags(
    tags: np.ndarray, boundary_prob: np.ndarray, fps: float, prior: DurationPrior,
    split_bias: float | None = None, snap_radius_s: float | None = None,
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
        for j in range(1, L + 1):
            i0 = max(0, j - lmax)
            rng = np.arange(i0, j)
            cand = D[rng] + LP[j - rng] + (rng > 0) * split_bias
            k = int(np.argmax(cand)); D[j] = cand[k]; arg[j] = i0 + k

        if censored:
            # Censored segment start = best fully-segmented prefix + survival of the open tail. i*=0 (whole run
            # one open sentence) is reachable only while L <= lmax, so over-long runs must split.
            i_rng = np.arange(max(0, L - lmax), L)
            cand = D[i_rng] + (i_rng > 0) * split_bias + LS[L - i_rng]
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
    commit_mask: torch.Tensor | None = None, seam_is_terminator: bool = True,
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
    if duration_prior is None: return tags
    pB = torch.softmax(bio_logits.detach().float(), dim=-1)[..., BIO["B"]].cpu().numpy()
    tags_np = tags.cpu().numpy()

    for b in range(tags_np.shape[0]):
        n = int(lengths[b].item())
        if n <= 2: continue
        
        fps_b = 24.0 # Default fallback if timestamps are missing or degenerate.
        if timestamps_s is not None and n > 1:
            dt = timestamps_s[b, 1:n] - timestamps_s[b, : n - 1]
            # Clamp: duplicate-timestamp median ~0 → fps ~1e6, so lmax = cap_s*fps OOMs the LP array / O(T*lmax) DP.
            if dt.numel(): fps_b = min(max(1.0 / max(float(dt.median().item()), 1e-6), 1.0), 120.0)

        tags_np[b, :n] = duration_split_tags(
            tags_np[b, :n], pB[b, :n], fps_b, duration_prior, mark_onsets=False, split_open_tail="survival",
        )
        if commit_mask is not None and seam_is_terminator:
            cm = commit_mask[b, :n].detach().cpu().numpy().astype(bool)
            if cm.any() and not cm.all():
                seam = int(np.argmax(~cm))  # first non-committed frame
                sig = (tags_np[b, :n] == BIO["I"]) | (tags_np[b, :n] == BIO["B"])
                cand = np.flatnonzero(sig[seam:])
                if cand.size:
                    d = seam + int(cand[0])
                    if tags_np[b, d] == BIO["I"] and (d == 0 or sig[d - 1]): tags_np[b, d] = BIO["B"]
    return torch.as_tensor(tags_np, device=device, dtype=tags.dtype)
