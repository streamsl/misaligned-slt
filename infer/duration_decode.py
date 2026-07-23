"""Semi-Markov (HSMM-style) duration decode — THIS SYSTEM'S contribution, not part of the Moryossef protocol.

Whole-video argmax BIO decoding under-segments catastrophically on caption-supervised YouTube corpora: ~half of
adjacent captions are back-to-back (no O gap), so argmax never wins `B` there, and under one-to-one tIoU matching
a merged run spanning k captions matches at most one (asf dev: 7.9 predicted vs 33.8 gold segments/video). The
fix is inference, not learning: the signing/non-signing runs are excellent (binary frame F1 ~0.96), and the
information that is missing from the frames — "sentences last ~4 s, not 40 s" — is exactly what a two-moment
lognormal duration prior carries. MAP decoding under that prior (exact segmental Viterbi per signing run, then
snapping each split to the segmenter's own P(B) peak) lifts asf dev whole-video tiou F1@0.5 0.19→0.51 (S1 head)
and 0.25→0.46 (external Moryossef segmenter) with NO model change.

Why this must be semi-Markov: a linear-chain CRF/HMM encodes duration implicitly through self-transitions, which
is a GEOMETRIC (monotone-decreasing, memoryless) duration law — with flat `B` evidence its Viterbi path never
splits, and per-frame threshold rules (Moryossef 2023 Algorithm 1) fail the same way. An interior-mode duration
law requires hypothesising segment LENGTHS — the O(T·L_max) segmental DP below.

Used by `eval --segmenter-eval` and `analyze --stage segmenter-infer` (Analysis A/B + RQ2 cascade upstream) via
`moryossef26.infer`, where `--segmenter-decode plain` keeps faithful argmax protocol. Streaming FSM/commit-gate 
do NOT use this yet — buffer-level duration decoding is a coherent system-wide change deferred to gate refactor.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from data.loader import VideoRecord
from data.windowing import BIO
from metrics import _bio_runs


# Per-split log-score bonus, the decode's ONE real hyperparameter. It must roughly offset the per-segment
# lognormal normalisation cost (-max LP ~= 4.8 for asf@24fps) or the DP never/always splits — the F1(mu) curve
# is a sharp peak, NOT a plateau: mu<=3 -> zero splits, mu>=6 -> oversplit collapse. mu=4.0 was grid-selected
# and replicated in all four (dev-fold x arch) cells; exact normalisation cancellation (mu = -max LP) oversplits
# by ~0.15 F1, so it cannot be auto-derived — for a NEW corpus, run `analyze.py --stage tune-decode` on its dev
# (the same two-fold protocol; decode-only sweep on cached predictions) and pin the pair in inference.yaml's
# `duration_decode` mapping rather than trusting this constant.
DURATION_SPLIT_BIAS = 4.0
# Splits are snapped to the segmenter's own P(B) argmax within this radius (only when the peak actually beats
# the DP position): duration decides HOW MANY, the model's boundary posterior decides WHERE. Swept 0..1.5s on
# both folds x both arches: monotone gains to 1.0s at BOTH tIoU 0.5 and 0.7 (s1 0.503->0.529/0.515->0.524),
# fold-inconsistent beyond -> capped at 1.0s. Using P(B) INSIDE the DP objective instead was tuned to zero
# weight — weak evidence may perturb positions, never counts.
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


def duration_decode_params(inference_cfg: dict) -> dict | None:
    """Parse inference.yaml `duration_decode`: false/absent -> None (decode off); true -> {} (module defaults);
    mapping -> {split_bias, snap_radius_s} per-language tuned overrides (from `analyze --stage tune-decode`)."""
    v = (inference_cfg or {}).get("duration_decode", False)
    if isinstance(v, dict): return {k: float(v[k]) for k in ("split_bias", "snap_radius_s") if v.get(k) is not None}
    return {} if bool(v) else None


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
    mark_onsets: bool = True, split_open_tail: bool = True,
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
    B, if any, is not preserved verbatim. `split_open_tail=False` leaves a run that touches the LAST frame
    unsplit — a right-truncated run's total length is unknown mid-stream, so duration evidence about where to cut
    it is not yet valid (the buffer-cap forced commit already bounds the pathological case).
    """
    fps = float(fps)
    split_bias = float(prior.split_bias if split_bias is None else split_bias)
    snap_radius_s = float(prior.snap_radius_s if snap_radius_s is None else snap_radius_s)
    lmax = max(2, round(prior.cap_s * fps))
    # Lognormal over seconds evaluated at L/fps, with the −log(fps) change-of-variables term so the density is
    # over FRAMES — this constant per segment is what makes `split_bias` transfer across corpora with other fps.
    L_s = np.arange(1, lmax + 1, dtype=float) / fps
    logp = -np.log(L_s * prior.sd_log_s * np.sqrt(2 * np.pi)) - (np.log(L_s) - prior.mu_log_s) ** 2 / (2 * prior.sd_log_s ** 2) - np.log(fps)
    LP = np.concatenate([[-1e18], logp])  # LP[L_frames], L>=1

    out = np.array(tags, copy=True)
    out[out == BIO["B"]] = BIO["I"]
    snap_r = max(0, round(snap_radius_s * fps))
    splits: list[int] = []
    for run in _bio_runs(out, split_on_b=False, open_on_i=True, close_on_unk=True):
        a, b = int(run["start"]), int(run["end"])
        L = b - a + 1
        if not split_open_tail and b == len(out) - 1:
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
        j = L
        while j > 0:
            i = int(arg[j])
            if i > 0:
                p = a + i
                lo, hi = max(a + 1, p - snap_r), min(b, p + snap_r + 1)
                if hi > lo:
                    q = lo + int(np.argmax(boundary_prob[lo:hi]))
                    # Move only when the peak genuinely beats the DP position — flat evidence keeps the
                    # duration-optimal split instead of drifting to the window edge (argmax-of-constant).
                    if boundary_prob[q] > boundary_prob[p]: p = q
                splits.append(p)
            j = i
        if mark_onsets: splits.append(a)  # run onset stays a segment start (first signing frame after a gap)
    for p in sorted(set(splits)): out[p] = BIO["B"]
    return out


def duration_decode_tags(logits: torch.Tensor, fps: float, prior: DurationPrior) -> torch.Tensor:
    # argmax tags -> semi-Markov re-split, with the segmenter's own softmax P(B) as the snap evidence.
    pB = torch.softmax(logits[0].float(), dim=-1)[:, BIO["B"]].cpu().numpy()
    return torch.as_tensor(duration_split_tags(logits.argmax(dim=-1)[0].cpu().numpy(), pB, fps, prior))
