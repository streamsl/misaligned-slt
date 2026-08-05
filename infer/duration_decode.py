"""Semi-Markov (HSMM-style) duration decode — THIS SYSTEM'S contribution, not part of the Moryossef protocol.

Whole-video argmax BIO decoding under-segments catastrophically on caption-supervised YouTube corpora: a large
fraction of adjacent captions are back-to-back (no O gap), so argmax never wins `B` there, and under one-to-one
tIoU matching a merged run spanning k captions matches at most one. The fix is inference, not learning: the
signing/non-signing runs are reliable, and the information missing from the frames — how long a sentence
plausibly lasts — is exactly what a two-moment lognormal duration prior carries. MAP decoding under that prior
(exact segmental Viterbi per signing run, then snapping each split to the segmenter's own P(B) peak) recovers
the merged boundaries with NO model change, for the in-system head and the external segmenter alike.

Why this must be semi-Markov: a linear-chain CRF/HMM encodes duration implicitly through self-transitions, which
is a GEOMETRIC (monotone-decreasing, memoryless) duration law — with flat `B` evidence its Viterbi path never
splits, and per-frame threshold rules (Moryossef 2023 Algorithm 1) fail the same way. An interior-mode duration
law requires hypothesising segment LENGTHS — the O(T·L_max) segmental DP below.

Consumers (all keyed on ONE inference.yaml `duration_decode` switch, resolved per-language by
`duration_decode_params`): whole-video eval + Analysis A/B via `moryossef26.infer` (`--segmenter-decode plain`
keeps the faithful argmax protocol for the external baseline row); the streaming FSM's tag stream
(`infer.stream.step`) and the membership gate's anchor selection (`build_gate_omega`) — both with the
streaming flags (`mark_onsets=False, split_open_tail="survival"`), which is what keeps the gate on-policy
with the deployed decode; and `analyze --stage delta-enc`, which must measure the terminator statistic under
the SAME deployed decode or it reports the raw-argmax instability instead of the head's noise floor.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from data.loader import VideoRecord
from data.windowing import BIO
from metrics import _bio_runs


# Per-split log-score bonus, the decode's ONE real hyperparameter. It must roughly offset the per-segment
# lognormal normalisation cost or the DP never/always splits — F1(bias) is a sharp peak, not a plateau, and
# exact normalisation cancellation oversplits, so the value cannot be auto-derived. The peak's location follows
# the corpus's duration prior: for a NEW corpus, run `analyze.py --stage tune-decode` on its dev
# (the same two-fold protocol; decode-only sweep on cached predictions) and pin the pair in inference.yaml's
# `duration_decode` mapping rather than trusting this constant.
DURATION_SPLIT_BIAS = 4.0
# Splits are snapped to the segmenter's own P(B) argmax within this radius (only when the peak actually beats
# the DP position): duration decides HOW MANY, the model's boundary posterior decides WHERE. Weak boundary
# evidence must only perturb positions, never counts — folding P(B) into the DP objective instead tunes to zero
# weight. Tuned jointly with DURATION_SPLIT_BIAS by `analyze --stage tune-decode`; re-tune per corpus.
SNAP_RADIUS_S = 1.0


@dataclass(frozen=True)
class DurationPrior:
    """Log-normal sentence-duration prior, fitted on TRAIN caption spans in LOG-SECONDS (fps-independent).

    Two moments + a tail cap — nothing trained, no EM: `fit_duration_prior` is a closed-form moment fit. The
    emission model of the implied HSMM is whatever segmenter produced the tags; this class only supplies the
    duration law that argmax decoding lacks. It also CARRIES the two tuned decode hyperparameters (per-language,
    `analyze --stage tune-decode`) so every consumer — whole-video eval, the streaming FSM, the membership gate —
    decodes with the same pair without threading extra arguments.
    """
    mu_log_s: float
    sd_log_s: float
    cap_s: float
    split_bias: float = DURATION_SPLIT_BIAS
    snap_radius_s: float = SNAP_RADIUS_S


def _coerce_decode_entry(v) -> dict | None:
    # one duration_decode value -> None (off) | {} (on, module defaults) | {split_bias?, snap_radius_s?} (tuned).
    if isinstance(v, dict): return {k: float(v[k]) for k in ("split_bias", "snap_radius_s") if v.get(k) is not None}
    return {} if bool(v) else None


def duration_decode_params(inference_cfg: dict, language: str | None = None) -> dict | None:
    """Resolve inference.yaml `duration_decode` for a language. Returns None (decode OFF), {} (ON, module
    defaults), or {split_bias, snap_radius_s} (ON, tuned).

    split_bias is corpus-specific (a Lagrange multiplier pinned to the duration prior's normalisation scale —
    F1(bias) is a knife edge), so ONE global value is wrong across corpora. Three accepted shapes:
      false / true                                -> global off / global on-with-defaults
      {split_bias: .., snap_radius_s: ..}         -> global tuned pair (single-corpus configs)
      {default: <entry>, <lang>: <entry>, ...}    -> PER-LANGUAGE; each entry is any of the scalars/maps above
    A per-language map is detected by any key other than the two param names; an untuned language falls to
    `default` (recommend false — leave the decode OFF until `analyze --stage tune-decode` pins its pair)."""
    v = (inference_cfg or {}).get("duration_decode", False)
    if isinstance(v, dict) and not ({"split_bias", "snap_radius_s"} & set(v)):
        entry = v.get(str(language), v.get("default", False)) if language is not None else v.get("default", False)
        return _coerce_decode_entry(entry)
    return _coerce_decode_entry(v)


def fit_duration_prior(
    records: list[VideoRecord], split_bias: float | None = None, snap_radius_s: float | None = None,
) -> DurationPrior | None:
    """Fit the two-moment lognormal on a TRAIN split's caption durations (span filter mirrors data.yaml bounds).
    `split_bias`/`snap_radius_s`: tuned per-language overrides (duration_decode_params); None -> module defaults."""
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
    """Semi-Markov MAP re-decode of argmax `tags`: keep the (reliable) signing/non-signing runs, re-place ALL
    interior sentence boundaries by segmental Viterbi under the duration prior, then snap each split to the
    P(B) peak within ±`snap_radius_s`. Returns a new tag array (same BIO id space); O/UNK frames untouched.

    Per signing run [a,b] maximise  Σ_seg log LogNormal(L_seg) + (#splits)·split_bias  over split sets — an
    O(L·L_max) DP (exact segmental Viterbi). ALL argmax `B`s are folded to `I` first (`out[out==B]=I`): at ~1%
    prevalence argmax-B is noise and the DP owns splitting (probes: keeping them changes nothing). The output B's
    are therefore EXACTLY the DP's split points (plus each run onset iff `mark_onsets`).

    `mark_onsets=True` (whole-video) re-marks every run onset as `B`, so the tags are canonical BIO (a B at every
    segment start); it does NOT change the segmentation any downstream decoder produces, since the shared span
    decode opens on the O→signing transition, not on `B` specifically. `mark_onsets=False` (streaming-buffer,
    `infer.stream` + the gate's `build_gate_omega`) omits those onset B's: run onsets stay `I`, which is INERT
    for span opening (same O→signing rule) — so span-opening semantics are untouched even though an argmax onset
    B, if any, is not preserved verbatim. 
    
    `split_open_tail` handles the run touching the LAST frame — the one run whose total length is RIGHT-CENSORED 
    mid-stream (the sentence may continue past the buffer):
      True        (whole-video): the input ended, nothing is censored — split it like any other run.
      "survival"  (streaming): score the run's LAST segment by the lognormal log-survival log P(dur > L) — the
                  correct likelihood for a still-open segment (a probability, so no Jacobian) — and every earlier
                  segment as usual. Interior splits of the observed prefix commit; only the censored tail stays
                  open. Without this a signing stretch longer than the buffer never closes inside any buffer and
                  never gets split; vanishing survival mass beyond the prior's tail also forces splits on
                  implausibly long observed runs.
      False       (legacy): leave the tail run entirely unsplit — starves long runs of splits; prefer "survival".
    """
    fps = float(fps)
    split_bias = float(prior.split_bias if split_bias is None else split_bias)
    snap_radius_s = float(prior.snap_radius_s if snap_radius_s is None else snap_radius_s)
    # Cap lmax at the sequence length: no segment can exceed the whole input, and this bounds the LP array +
    # per-run DP to O(T) even if a degenerate (near-constant) timestamp stream drives fps → a huge value at the
    # caller's `1/median(dt)` estimate (the estimate can blow up; the DP cost must not).
    lmax = max(2, min(round(prior.cap_s * fps), len(tags)))
    # Lognormal is over DURATION IN SECONDS; evaluate at L/fps and add the −log(fps) change-of-variables Jacobian
    # (d(seconds)/d(frames) = 1/fps) so LP is a proper density over integer FRAME lengths — required for the 
    # segmental DP to be a correct MAP over frame-partitions. The count-dependent part of a run's score is exactly
    # K·(split_bias − log fps), K = #segments: split_bias and the Jacobian's −log fps BOTH act on segment count, so
    # the effective count-penalty is fps-dependent. This is NOT a transfer device — split_bias is corpus-specific
    # (different μ_ln/σ_ln AND different fps both move the F1 peak) and is re-tuned per language by
    # `analyze --stage tune-decode`; do NOT reuse one corpus's value.
    # log N(L_s | μ_ln, σ_ln) = −log(L_s·σ_ln·√(2π)) − (log(L_s) − μ_ln)² / (2·σ_ln²)
    L_s = np.arange(1, lmax + 1, dtype=float) / fps
    logp = -np.log(L_s * prior.sd_log_s * np.sqrt(2 * np.pi)) \
           - (np.log(L_s) - prior.mu_log_s) ** 2 / (2 * prior.sd_log_s ** 2) - np.log(fps)
    LP = np.concatenate([[-1e18], logp])  # LP[L_frames], L>=1
    LS = None
    if split_open_tail == "survival":
        # Lognormal log-survival log P(dur > L): S(x) = ½·erfc((ln x − μ)/(σ√2)). A probability over the
        # censored observation, so no −log(fps) Jacobian. Clamped: erfc underflows to exactly 0 in the far
        # tail, and log(0) would poison the DP argmax instead of merely disfavouring the length.
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
        censored = split_open_tail != True and b == len(out) - 1  # noqa: E712 — "survival" is also != True
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
            # Right-censored tail: pick the censored segment's start i* maximising fully-segmented-prefix score
            # + survival of the open tail. i* = 0 (whole run still one open sentence) is only reachable while
            # L <= lmax — an observed run beyond the prior's cap has S ~ 0 mass left, so splits are forced.
            i_rng = np.arange(max(0, L - lmax), L)
            cand = D[i_rng] + (i_rng > 0) * split_bias + LS[L - i_rng]
            j = int(i_rng[int(np.argmax(cand))])
            tail_start = j  # backtrace covers the prefix [0, j); j itself is the censored segment's opening split
        else: j = L; tail_start = 0

        raw = []  # DP split frames for this run, collected then snapped LEFT-TO-RIGHT
        if tail_start > 0: raw.append(a + tail_start)
        while j > 0:
            i = int(arg[j])
            if i > 0: raw.append(a + i)
            j = i
        raw.sort()
        # Snap each split to its P(B) peak, but CLAMP the search window between the neighbouring splits (prev
        # snapped position on the left, next raw DP split on the right) so two adjacent splits can never both
        # collapse onto one peak (set()-dedup would silently drop a segment) nor reorder past each other — the
        # snapped splits stay strictly increasing, one per DP-chosen segment.
        prev = a
        for idx, p in enumerate(raw):
            nxt = raw[idx + 1] if idx + 1 < len(raw) else b + 1
            lo, hi = max(prev + 1, p - snap_r), min(nxt, p + snap_r + 1)
            q = p
            if hi > lo:
                cand = lo + int(np.argmax(boundary_prob[lo:hi]))
                # Move only when the peak genuinely beats the DP position — flat evidence keeps the
                # duration-optimal split instead of drifting to the window edge (argmax-of-constant).
                if boundary_prob[cand] > boundary_prob[p]: q = cand
            splits.append(q); prev = q
        if mark_onsets: splits.append(a)  # run onset stays a segment start (first signing frame after a gap)
    for p in sorted(set(splits)): out[p] = BIO["B"]
    return out


def duration_decode_tags(logits: torch.Tensor, fps: float, prior: DurationPrior) -> torch.Tensor:
    # argmax tags -> semi-Markov re-split, with the segmenter's own softmax P(B) as the snap evidence.
    pB = torch.softmax(logits[0].float(), dim=-1)[:, BIO["B"]].cpu().numpy()
    return torch.as_tensor(duration_split_tags(logits.argmax(dim=-1)[0].cpu().numpy(), pB, fps, prior))
