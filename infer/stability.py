"""Stable-prefix policies for the streaming display, scored offline from recorded stride hypotheses.

FSM decodes its target span only on the stride it commits it, and shows nothing until then, so the displayed text never changes 
— normalised erasure is 0 by construction and the open question is LATENCY, not flicker. A stable-prefix policy trades that: 
show a prefix earlier, at risk of revealing a token later evidence would have changed. Every policy except mask-k only grows its
revealed LENGTH; mask-k shows each stride's hypothesis less its last k tokens, so its length can shrink, as in re-translation
systems. Under every policy the screen still shrinks when a stride's hypothesis is shorter than revealed length (NE counts it). 
No policy is monotone in CONTENT: these traces are from unforced re-translation, so a later stride may return a different token 
at an already-shown position, and the display visibly rewrites. Every screen before the commit is checked against the committed 
sentence (`frozen_prefix_error`, `contradicted_track_rate`); that is the measurement that says whether prefix-FORCED decoding 
(constraining the decoder to continue from what was shown) is worth building. Forcing cannot be simulated from these traces: 
it changes the decode, so it needs a re-run, not a replay.

Policies replay `StreamingSLTRunner(record_trace=True).trace`: recording turns FSM's decode skip off, so the analysis pays for
deferred strides' decodes once, and every policy then replays exactly same hypotheses. Recording also decodes the final OPEN span
on every voting stride (trace only, never emitted): the unit before its terminator appears, the incomplete input a re-translation
system such as ReSCUE re-reads every step, so a policy can show text while the person still signs; `group_tracks` joins it to the
unit's track. 2 signal families, each parameterised by how many strides of evidence it demands:

  reveal_on_agreement(n)             the token at each position must be IDENTICAL across the last n strides: the agreement test
                                     of LocalAgreement, re-checked on every stride, so the shown text is NOT frozen (it shows
                                     the current hypothesis cut at the longest length agreed so far). A string test only.
  reveal_on_frozen_agreement(n)      the same test, but the agreed text is FROZEN on screen and only grows: a later stride that
                                     disagrees changes nothing shown, and the commit replaces the screen with the committed text 
                                     (the display of Whisper-Streaming; no forced decoding, so every decode stays fresh).
  reveal_on_confidence(n, tau)       the token's own confidence must be >= tau for n consecutive strides, with a changed token
                                     resetting its run. A position the current hypothesis does not reach has no run, so nothing 
                                     beyond the current hypothesis is revealed. n=1 reads only the current stride.
  reveal_on_agreement_and_confidence both conditions (conservative).
  reveal_mask_k(k)                   each stride's hypothesis without its last k tokens, recomputed every stride, so the shown
                                     prefix can shrink (mask-k, the display rule of re-translation systems). Fixed-k baseline 
                                     curve, never tuned and never a commit rule. Its input premise (incomplete input) holds on 
                                     open-span hypotheses, not on closed span's. Its error premise (only last tokens are uncertain) 
                                     relies on prefix-to-prefix training. That translator trains on complete units only (P1), 
                                     so on an open span it can complete sentence anywhere. The replay measures that cost.
  reveal_at_commit                   the deployed behaviour: nothing until the gate commits. The latency ceiling.

Policies, tracks and screens are here. The scores of a policy over tracks (latency, erasure, rewrites, LAAL, boundary flips) are 
reported numbers and live in `evaluation.scoring` (`score_policy`). Unit separator is not a displayed word: `score_policy` drops it 
from every hypothesis before scoring, and counts a stride whose separators changed from the previous stride's of same sentence as a 
boundary flip: 1 added or removed, or 1 moved (unit before it and unit after it both changed; an edit inside 1 unit is no flip).

A policy returns its reveals as (stride time, shown) pairs: `shown` is a length, and the screen at each stride shows that stride's
hypothesis cut at the LATEST shown length at or before it; a frozen policy gives the shown tokens themselves (`displays`).

THRESHOLD CONVENTION: a token is retained when `confidence >= tau` — at-or-above, not strictly above. Exactly-tau retains.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial

# Strides of evidence each family is swept over. Every registry name is generated from these, 
# so there is never a policy whose name implies a sibling that does not exist.
AGREEMENT_STRIDES = (2, 3)
CONFIDENCE_STRIDES = (1, 2)
MASK_K = (0, 1, 2, 3)            # tokens held back by mask-k; 0 shows each stride's whole hypothesis
TAU_GRID = (0.3, 0.5, 0.7, 0.9)  # Per-token display thresholds

@dataclass
class Track: # Successive hypotheses for ONE sentence, in stride order.
    span_start_s: float
    hyps: list = field(default_factory=list)   # list[StrideHypothesis]

    @property
    def committed(self) -> bool:   # the FSM emitted this sentence; a track closes at its committed hypothesis
        return any(bool(h.committed) for h in self.hyps)

    @property
    def final_tokens(self) -> list[int]:
        return [int(t) for t in self.hyps[-1].token_ids.tolist()] if self.hyps else []

    @property
    def commit_time_s(self) -> float:
        return float(self.hyps[-1].commit_time_s) if self.hyps else float("nan")


def display_prefix(token_ids, token_confidence, eos_id: int | None = None, pad_id: int | None = None):
    """Cut a decoded canvas down to the tokens a viewer could actually be shown.

    The DLM writes a FABRICATED confidence of 1.0 into every slot it back-fills with pad after a committed EOS
    (infer/decode.py). generate_from_bio_tap already slices at 1st EOS on live path, so this is defence in depth 
    for any caller that records raw decodes. A stride can legitimately decode MORE real tokens than final commit 
    (a no-EOS canvas — hallucinated-continuation mode); those are charged via vanished_track_rate, never trimmed.
    """
    stop = {int(i) for i in (eos_id, pad_id) if i is not None}
    keep = int(token_ids.numel())
    if stop:
        for i, t in enumerate(token_ids.tolist()):
            if int(t) in stop: keep = i; break
    return token_ids[:keep], token_confidence[:keep]


def group_tracks(trace) -> list[Track]:
    """Group stride hypotheses by the sentence they concern: commit gate's own span identity, which the runner records on every 
    hypothesis (`track_id`, serial of the history that CommitGate matches to the target span; infer/commit_gate.py). A history 
    lives until the gate drops it (unseen for K voting strides, retired by a commit, or cleared at the cap), and a span seen 
    after that gets a new serial, so tracks are exactly gate's histories; a committed hypothesis closes its track. A hypothesis 
    the gate holds no history for (target first seen on a pass that adds no vote: cap deadline between strides, or the drain) is
    a track of its own. Nothing is re-derived from times, so no frame rounding, vote count or stride clock disagree with the gate.

    Open-span hypothesis (`open_span`: the runner's decode of the final span before its terminator appears) has no gate history.
    It joins the track whose FIRST start is within δ frames of its own start and that has a hypothesis after it (the same unit, 
    whose terminator appears later), the nearest start first, but never the track of closed target its own pass records (a unit
    before it); so the open hypotheses of a unit join its later commit; a unit that the cap drops, or that the stream end reaches 
    open, gets no commit, and one that the cap keeps can commit in the next buffer, in same track. Without such a track it joins, 
    or opens, a track of open hypotheses only (again by first start within δ frames): a unit that vanished before its terminator 
    appeared, which never commits and is scored like any never-committed track."""
    tracks: list[Track] = []
    members: list[list[int]] = []   # trace indices of each track, first its closed hypotheses
    open_by_id: dict[int, int] = {}

    for i, h in enumerate(trace):
        if h.open_span: continue
        k = None if h.track_id is None else open_by_id.get(int(h.track_id))
        if k is None:
            k = len(tracks)
            tracks.append(Track(span_start_s=float(h.span_start_s))); members.append([])
            if h.track_id is not None: open_by_id[int(h.track_id)] = k
        members[k].append(i)
        if h.committed and h.track_id is not None: open_by_id.pop(int(h.track_id), None)

    closed, last = len(tracks), [m[-1] for m in members]
    owner = {j: k for k, m in enumerate(members) for j in m}
    for i, h in enumerate(trace):
        if not h.open_span: continue
        gap = lambda k: abs(int(trace[members[k][0]].span_start_frame) - int(h.span_start_frame))
        nearest = lambda ks: min((k for k in ks if gap(k) <= int(h.delta_frames)), key=gap, default=None)
        # The closed target the same pass records right after it (same emission time) is another unit: the open span starts 
        # at or after that target's terminator, so a short target within δ frames must not take the successor's text.
        same_pass = owner.get(i + 1)
        if same_pass is not None and trace[i + 1].commit_time_s != h.commit_time_s: same_pass = None
        k = nearest(k for k in range(closed) if last[k] > i and k != same_pass)
        if k is None: k = nearest(range(closed, len(tracks)))
        if k is None:
            k = len(tracks)
            tracks.append(Track(span_start_s=float(h.span_start_s))); members.append([])
        members[k].append(i)

    order = sorted(range(len(tracks)), key=lambda k: min(members[k]))   # stride order of each track's first hypothesis
    for k in order: tracks[k].hyps = [trace[i] for i in sorted(members[k])]
    return [tracks[k] for k in order]


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
    # only the TIMING (and any prematurely frozen prefix) differs. A track that never commits gets no final reveal.
    final = len(track.final_tokens)
    if track.committed and shown < final: reveals.append((track.commit_time_s, final))
    return reveals

def reveal_at_commit(track: Track, **_) -> list[tuple[float, int]]:
    # Reveal the whole hypothesis at the committing stride; a track that never commits shows nothing, as the FSM does.
    for h in track.hyps:
        if h.committed: return [(float(h.commit_time_s), int(h.token_ids.numel()))]
    return []

def reveal_on_agreement(track: Track, n: int = 2, **_) -> list[tuple[float, int]]:
    reveals, shown = [], 0
    for i in range(len(track.hyps)):
        if i + 1 < n: continue
        k = _agreed_prefix_len(track, i, n)
        if k > shown:
            shown = k
            reveals.append((float(track.hyps[i].commit_time_s), shown))
    return _finish(track, reveals, shown)

def reveal_on_frozen_agreement(track: Track, n: int = 2, **_) -> list[tuple[float, object]]:
    """Show the prefix that the last `n` strides agree on and never take it back: the screen holds == frozen tokens, and only an
    agreed prefix that continues them extends it. A stride that disagrees with a frozen token changes nothing on screen. The commit
    shows the committed sentence, so a frozen text that is not its prefix is rewritten there (erasure, `frozen_prefix_error`)."""
    reveals, frozen = [], []
    for i, h in enumerate(track.hyps):
        if i + 1 < n: continue
        agreed = [int(t) for t in h.token_ids.tolist()][:_agreed_prefix_len(track, i, n)]
        if len(agreed) > len(frozen) and agreed[:len(frozen)] == frozen:
            frozen = agreed
            reveals.append((float(h.commit_time_s), list(frozen)))
    if track.committed: reveals.append((track.commit_time_s, len(track.final_tokens)))
    return reveals

def reveal_on_confidence(track: Track, n: int = 1, tau: float = 0.75, **_) -> list[tuple[float, int]]:
    """Reveal a prefix once each of its tokens has held confidence >= tau for `n` consecutive strides.

    A position's run resets when its token changes, so for n > 1 this implicitly demands token stability too — the
    confidence-side analogue of Local Agreement. n=1 is the pure single-measurement policy: read the current stride's
    confidence column top-down and stop at the first token below tau.
    """
    reveals, shown = [], 0
    runs: list[int] = []   # per position of the CURRENT hypothesis: consecutive strides at-or-above tau, token unchanged
    prev: list[int] = []
    for h in track.hyps:
        toks = [int(t) for t in h.token_ids.tolist()]
        conf = [float(c) for c in h.token_confidence.tolist()]
        # Rebuilt every stride, so a position the current hypothesis does not reach keeps no run from an older, longer one.
        runs = [
            0 if c < tau else (runs[j] + 1 if j < len(prev) and prev[j] == tok else 1)  # 1: first stride, or a changed token
            for j, (tok, c) in enumerate(zip(toks, conf))
        ]
        prev, k = toks, 0
        while k < len(runs) and runs[k] >= n: k += 1
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


def reveal_mask_k(track: Track, k: int = 1, **_) -> list[tuple[float, int]]:
    # Each stride shows its own hypothesis without its last k tokens (ReSCUE's max(0, length - k))
    reveals, shown = [], 0
    for h in track.hyps:
        n = max(0, int(h.token_ids.numel()) - int(k))
        if n != shown:
            shown = n
            reveals.append((float(h.commit_time_s), shown))
    return _finish(track, reveals, shown)


def build_policies(
    taus=TAU_GRID, agreement_strides=AGREEMENT_STRIDES, confidence_strides=CONFIDENCE_STRIDES, mask_k=MASK_K,
) -> dict[str, object]:
    """Named policies for the comparison table, generated from the parameter grids.

    Names carry every parameter that varies (`agreement_n2`, `confidence_n1_tau50`, ...), so no name implies a sibling 
    that was never built. Read the table as 2 curves, not as a winner per row.

    `tau` is SWEPT, never inherited from a config value: no training threshold transfers to a per-token display rule. 
    And a single tau produces 1 point, which cannot be compared against Local Agreement's own curve over n; 
    the comparison only means something at matched operating points. .
    """
    policies: dict[str, object] = {"commit_only": reveal_at_commit}
    for n in agreement_strides:
        policies[f"agreement_n{n}"] = partial(reveal_on_agreement, n=n)
        policies[f"frozen_agreement_n{n}"] = partial(reveal_on_frozen_agreement, n=n)
    for tau in taus:
        t = int(round(100 * tau))
        for n in confidence_strides:
            policies[f"confidence_n{n}_tau{t}"] = partial(reveal_on_confidence, n=n, tau=tau)
        for n in agreement_strides:
            policies[f"agreement_n{n}_confidence_tau{t}"] = partial(reveal_on_agreement_and_confidence, n=n, tau=tau)
    for k in mask_k: policies[f"mask_k{k}"] = partial(reveal_mask_k, k=k)
    return policies


def displays(track: Track, reveals: list[tuple[float, object]]) -> list[tuple[float, list[int]]]:
    """What the screen shows after each stride of a track under a policy: stride's hypothesis cut at latest length revealed by then, 
    or latest revealed tokens themselves (frozen policy). Tokens of already-shown position can change (unforced re-translation)."""
    out, shown, r = [], 0, sorted(reveals, key=lambda x: x[0])  # stable: reveals at 1 time keep their order, the last wins
    for h in track.hyps:
        while r and r[0][0] <= float(h.commit_time_s): shown = r.pop(0)[1]
        out.append((
            float(h.commit_time_s), 
            list(shown) if isinstance(shown, list) else [int(x) for x in h.token_ids.tolist()][:int(shown)]
        ))
    return out
