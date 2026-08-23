"""Stable-prefix policies for the streaming display, scored offline from recorded stride hypotheses.

The FSM re-decodes its candidate span every stride and shows nothing until the commit gate fires, so today the displayed 
text never changes — normalised erasure is 0 by construction and the open question is LATENCY, not flicker. A stable-prefix 
policy trades that: show a prefix earlier, at the risk of revealing a token later evidence would have changed. Every policy 
is monotone in prefix LENGTH — the revealed prefix only ever grows. It is NOT monotone in CONTENT: these traces come from 
unforced re-translation, so a later stride may return a different token at an already-revealed position, and the display 
would visibly rewrite. `frozen_prefix_error` is exactly that rate, and it is the measurement that says whether prefix-FORCED 
decoding (constraining the decoder to continue from what was shown) is worth building. Forcing cannot be simulated from 
these traces: it changes the decode, so it needs a re-run, not a replay.

Policies replay `StreamingSLTRunner(record_trace=True).trace`, so they cost no extra decoding and every policy sees exactly 
the same hypotheses. 2 signal families, each parameterised by how many strides of evidence it demands:

  reveal_on_agreement(n)             the token at each position must be IDENTICAL across the last n strides
                                     (Local Agreement). Purely a string test; no training counterpart.
  reveal_on_confidence(n, tau)       the token's own confidence must be >= tau for n consecutive strides, with a
                                     changed token resetting its run. n=1 reads only the current stride — the
                                     single-measurement policy the confidence-bound term is meant to license.
  reveal_on_agreement_and_confidence both conditions (conservative).
  reveal_at_commit                   what ships today: nothing until the gate commits. The latency ceiling.

THRESHOLD CONVENTION: a token is retained when `confidence >= tau` — at-or-above, not strictly above. Exactly-tau retains.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
import numpy as np

# Strides of evidence each family is swept over. Every registry name is generated from these, so there is never a
# policy whose name implies a sibling that does not exist.
AGREEMENT_STRIDES = (2, 3)
CONFIDENCE_STRIDES = (1, 2)
# Per-token display thresholds. Spans the range the decoder actually produces: commit_gate documents function words
# sitting at 0.2-0.5 with an AR-arm mean near 0.4, so a 0.75 point alone would reveal almost nothing and read as
# "confidence loses to Local Agreement" when it was only mis-scaled.
TAU_GRID = (0.3, 0.5, 0.7, 0.9)


@dataclass
class Track: # Successive hypotheses for ONE sentence, in stride order.
    span_start_s: float
    hyps: list = field(default_factory=list)   # list[StrideHypothesis]

    @property
    def final_tokens(self) -> list[int]:
        return [int(t) for t in self.hyps[-1].token_ids.tolist()] if self.hyps else []

    @property
    def commit_time_s(self) -> float:
        return float(self.hyps[-1].commit_time_s) if self.hyps else float("nan")


def group_tracks(trace, delta_s: float) -> list[Track]:
    """Group stride hypotheses by the sentence they concern.

    Same rule the commit gate uses for target identity: a span whose start moves by more than delta is a DIFFERENT target, 
    not a revision of current one. Using the gate's own criterion keeps the harness consistent with the FSM it measures.
    """
    tracks: list[Track] = []
    for h in trace:
        if tracks and abs(h.span_start_s - tracks[-1].span_start_s) <= delta_s: tracks[-1].hyps.append(h)
        else: tracks.append(Track(span_start_s=float(h.span_start_s), hyps=[h]))
    return tracks


def _lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

def _agreed_prefix_len(track: Track, i: int, n: int) -> int:
    # Positions identical across strides [i-n+1, i]. n=1 means the whole current hypothesis.
    window = [[int(t) for t in track.hyps[j].token_ids.tolist()] for j in range(i - n + 1, i + 1)]
    return len(window[0]) if n == 1 else min(_lcp(window[0], w) for w in window[1:])

def _prefix_len_at_or_above(conf: list[float], tau: float, limit: int | None = None) -> int:
    k = 0 # Longest prefix with confidence >= tau, stopping at the first token below it.
    for c in conf if limit is None else conf[:limit]:
        if float(c) < tau: break
        k += 1
    return k

def _finish(track: Track, reveals: list[tuple[float, int]], shown: int) -> list[tuple[float, int]]:
    # The commit still reveals whatever the policy held back, so every policy ends on the same complete sentence and
    # only the TIMING (and any prematurely frozen prefix) differs.
    final = len(track.final_tokens)
    if shown < final: reveals.append((track.commit_time_s, final))
    return reveals


def reveal_at_commit(track: Track, **_) -> list[tuple[float, int]]:
    # Reveal the whole hypothesis at the committing stride (or the last one, if the run ended uncommitted).
    for h in track.hyps:
        if h.committed: return [(float(h.commit_time_s), int(h.token_ids.numel()))]
    return [(track.commit_time_s, len(track.final_tokens))]


def reveal_on_agreement(track: Track, n: int = 2, **_) -> list[tuple[float, int]]:
    reveals, shown = [], 0
    for i in range(len(track.hyps)):
        if i + 1 < n: continue
        k = _agreed_prefix_len(track, i, n)
        if k > shown:
            shown = k
            reveals.append((float(track.hyps[i].commit_time_s), shown))
    return _finish(track, reveals, shown)


def reveal_on_confidence(track: Track, n: int = 1, tau: float = 0.75, **_) -> list[tuple[float, int]]:
    """Reveal a prefix once each of its tokens has held confidence >= tau for `n` consecutive strides.

    A position's run resets when its token changes, so for n > 1 this implicitly demands token stability too — the
    confidence-side analogue of Local Agreement. n=1 is the pure single-measurement policy: read the current stride's
    confidence column top-down and stop at the first token below tau.
    """
    reveals, shown = [], 0
    runs: dict[int, int] = {}      # position -> consecutive strides at-or-above tau with an unchanged token
    prev: list[int] = []
    for h in track.hyps:
        toks = [int(t) for t in h.token_ids.tolist()]
        conf = [float(c) for c in h.token_confidence.tolist()]
        for j, (tok, c) in enumerate(zip(toks, conf)):
            if c < tau: runs[j] = 0
            elif j < len(prev) and prev[j] == tok: runs[j] = runs.get(j, 0) + 1
            else: runs[j] = 1      # first stride at-or-above tau, or the token just changed

        prev, k = toks, 0
        while runs.get(k, 0) >= n: k += 1
        if k > shown:
            shown = k
            reveals.append((float(h.commit_time_s), shown))
    return _finish(track, reveals, shown)


def reveal_on_agreement_and_confidence(track: Track, n: int = 2, tau: float = 0.75, **_) -> list[tuple[float, int]]:
    # Agreement over the last `n` strides AND current-stride confidence >= tau.
    reveals, shown = [], 0
    for i, h in enumerate(track.hyps):
        if i + 1 < n: continue
        agreed = _agreed_prefix_len(track, i, n)
        k = _prefix_len_at_or_above([float(c) for c in h.token_confidence.tolist()], tau, limit=agreed)
        if k > shown:
            shown = k
            reveals.append((float(h.commit_time_s), shown))
    return _finish(track, reveals, shown)


def build_policies(
    taus=TAU_GRID, agreement_strides=AGREEMENT_STRIDES, confidence_strides=CONFIDENCE_STRIDES,
) -> dict[str, object]:
    """Named policies for the comparison table, generated from the parameter grids.

    Names carry every parameter that varies (`agreement_n2`, `confidence_n1_tau50`, ...), so no name implies a sibling 
    that was never built. Read the table as 2 curves, not as a winner per row.

    `tau` is SWEPT, never inherited from a single config value. It is a different quantity from both thresholds already in 
    configs — `commit_confidence_tau` (0.3) is a MEAN over a whole span, `tau_cb` (0.75) scores remasked training logits — 
    so neither transfers to a per-token display rule. And a single tau produces 1 point, which cannot be compared against 
    Local Agreement's own curve over n; the comparison only means something at matched operating points. .
    """
    policies: dict[str, object] = {"commit_only": reveal_at_commit}
    for n in agreement_strides:
        policies[f"agreement_n{n}"] = partial(reveal_on_agreement, n=n)
    for tau in taus:
        t = int(round(100 * tau))
        for n in confidence_strides:
            policies[f"confidence_n{n}_tau{t}"] = partial(reveal_on_confidence, n=n, tau=tau)
        for n in agreement_strides:
            policies[f"agreement_n{n}_confidence_tau{t}"] = partial(reveal_on_agreement_and_confidence, n=n, tau=tau)
    return policies


def score_policy(tracks: list[Track], policy, **kw) -> dict:
    """Latency and prefix-correctness for one policy over all tracks.

    first_token_latency_s    when the FIRST token becomes visible, relative to the commit the FSM would have made:
                             negative = the policy showed text EARLIER than today's behaviour (the point of the exercise).
    frozen_prefix_error      fraction of revealed tokens that disagree with the sentence finally committed — the price
                             of revealing early. 0 for commit_only by construction. This is the FIRST-ORDER cost only:
                             the traces come from unfrozen re-translation, so it cannot capture a frozen prefix
                             degrading the continuation. That needs prefix-forced decoding.
    revealed_fraction        how much of the sentence was visible before the commit.
    contradicted_track_rate  fraction of sentences whose revealed prefix was contradicted at least once — the
                             user-visible rewrite rate, and the decision number for prefix forcing.
    """
    lat, err, revealed, contradicted = [], [], [], []
    for tr in tracks:
        if not tr.hyps: continue
        reveals = policy(tr, **kw)
        if not reveals: continue
        final = tr.final_tokens
        commit_t = tr.commit_time_s
        lat.append(float(reveals[0][0]) - commit_t)

        pre = [n for t, n in reveals if t < commit_t]
        k = max(pre) if pre else 0
        revealed.append(k / max(1, len(final)))
        wrong = 0
        
        for t, n in reveals:
            if t >= commit_t: break
            hyp = next((h for h in tr.hyps if h.commit_time_s == t), None)
            if hyp is None: continue
            toks = [int(x) for x in hyp.token_ids.tolist()][:n]
            wrong = max(wrong, sum(1 for a, b in zip(toks, final[:n]) if a != b))

        err.append(wrong / max(1, k) if k else 0.0)
        contradicted.append(1.0 if wrong else 0.0)
    return {
        "n_tracks": len(lat),
        "first_token_latency_s": float(np.mean(lat)) if lat else 0.0,
        "revealed_fraction": float(np.mean(revealed)) if revealed else 0.0,
        "frozen_prefix_error": float(np.mean(err)) if err else 0.0,
        # Fraction of SENTENCES whose display would visibly rewrite at least once. This is the number that decides
        # whether prefix-forced decoding is worth building: near 0 means unforced re-translation is already stable.
        "contradicted_track_rate": float(np.mean(contradicted)) if contradicted else 0.0,
    }
