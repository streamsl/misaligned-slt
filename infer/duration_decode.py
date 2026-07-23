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
        raw = []  # DP interior split frames for this run, collected then snapped LEFT-TO-RIGHT
        j = L
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
