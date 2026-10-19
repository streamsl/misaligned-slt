"""Cut rules for a committed span that the translator decoded as several units ("A <sep> B ...").

The method rule is the word share (`infer.stream.split_unit_events`): each part's duration is in proportion to its words. The rules
here cut inner boundaries of saved split spans again from model evidence (`eval.py --resplit`). Each is a candidate under the split
rule in docs/implementation_notes.md ("Merged spans"), not the method rule. Likelihood and attention cuts read ONE common scorer for
every arm, the language's trained clean AR floor (AR and DLM rows alike: DLM has no exact left-to-right likelihood). The own-attention
cut reads the arm that wrote the events. Texts are the arm's own parts. Only the order of the units is assumed, never the word order 
inside a unit. A span keeps its own start, end, commit time, part texts and forced flag. Only its inner cuts move.

- Likelihood cut (SegFree-style; Iranzo-Sanchez et al. 2024, TACL). The k-1 cuts maximize, over a δ grid and then a finer pass within
  ±δ, the sum over parts of log p_AR(text_i | cascade crop of part i) plus log N(duration_i; a + b * words_i, sigma). (a, b, sigma)
  come from least squares on the language's train reliable units. Declared differences from SegFree: a forward model scores each
  part's text given its frames (there is no pose generator, so no reverse model); the two features have equal weights (SegFree fits
  its weights); the length regression has an intercept.
- Attention cut (Whisper's word timestamps; Radford et al. 2023, timing.py). Scorer reads the span's cascade crop, normalized on its
  own frames as the floor trains, and is teacher-forced in ONE decoder pass on the parts' texts (each part tokenized alone, the parts
  joined as by spaces, then EOS). The cross-attention of the configured decoder layers, averaged over heads and layers, is kept on the
  pose columns (the task-prompt memory columns dropped) and the text-token rows (row k predicts token k; the last row, which predicts
  EOS, dropped); it is normalized over tokens per frame and median-filtered along frames, as Whisper does, and a monotone DTW on its
  negative aligns tokens to frames. The cut before part i+1 is the first frame of its first token on the path; then every part keeps
  at least Λ_min frames. The layers are chosen on train (`eval.py --resplit attention --fit-layer`, the lowest mean cut error), never
  on dev. Declared differences from Whisper: the heads (and layers) are averaged before the normalization (Whisper normalizes each
  head, then averages); the attention is not renormalized over the pose columns after the prompt columns are dropped; the row that
  predicts EOS is not in the normalization or the DTW (Whisper keeps it, to end its last word).
- Own-attention cut. The same readout on the cross-attention of the gated AR arm that wrote the events, read as its `--segments` row 
  reads a span (evaluation.rq2.run_cascade: crop normalized with the commit-buffer body box, Ω anchored on the span, the support crop) 
  and teacher-forced on its parts joined by UNIT_SEPARATOR, as it writes and trains them; the separator rows are dropped before DTW. 
  The same layers and filter as the attention cut. AR arms only: the DLM decodes no left-to-right teacher-forced pass.

The separator commit (error analysis only, not deployed) is a text-side unit commit on the open-span decodes of a stream: it commits 
the first unit of a still-open span when the text before its first UNIT_SEPARATOR is the same in K decodes of K consecutive strides.
`eval.py --rq 2 --online --shadow-open-spans` saves the decodes; `eval.py --separator-replay` replays and scores the commits.
"""
from __future__ import annotations
from dataclasses import replace
import math, time
import numpy as np

import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter

from infer.stream import split_unit_text
from poses import load_pose_window, normalize_keypoints_unisign
from utils import FRAME_SLACK, UNIT_SEPARATOR, seconds_to_frames


def span_groups(evs) -> list[list]: # Events of 1 video as decoded spans: consecutive parts 0..parts-1 of 1 split span form 1 group.
    groups: list[list] = []
    for e in evs:
        if e.part == 0 or not groups or len(groups[-1]) >= groups[-1][0].parts: groups.append([e])
        else: groups[-1].append(e)
    return groups

def _span_crop(group, records, vid):
    """(record, start s, end s) of a split span's cascade crop in its record's time: [start, end + 1 frame], 
    cut to the stream end and to the frames held at the commit; None when no record holds the span."""
    first, last = group[0], group[-1]
    rec = next((r for r in records if r.pose.video_id == vid
                and r.pose.offset_s - 1e-6 <= first.start_s < r.pose.offset_s + r.pose.duration_s), None)
    if rec is None: return None
    off, fps = float(rec.pose.offset_s), float(rec.pose.fps)
    end = min(float(rec.pose.duration_s), last.end_s - off + 1.0 / fps)   # [start, end + 1 frame], as the cascade crop
    if first.commit_time_s is not None: end = min(end, first.commit_time_s - off - 1e-6)   # frames held at the commit
    return rec, first.start_s - off, end

# ---------------------------------------------------------------- likelihood cut

def fit_duration_prior(records) -> tuple[float, float, float]:
    # (a, b, sigma) of duration = a + b * words + noise, least squares over the reliable units of `records` (the train split).
    units = [u for r in records for u in r.sentences if getattr(u, "reliable", True)]
    d = np.array([u.end_s - u.start_s for u in units], float)
    w = np.array([max(1, len(u.text.split())) for u in units], float)
    b, a = np.polyfit(w, d, 1)
    return float(a), float(b), float(np.std(d - (a + b * w), ddof=2))


def dp_cuts(score, n: int, k: int, grid: int, min_frames: int, refine_step: int, prefetch=lambda triples: None) -> list[int] | None:
    """k-1 cut frames in [0, n] for k parts: part i reads frames [c_{i-1}, c_i] (the cascade crop holds the cut frame), 1st part 
    starts at 0 and the last ends at n, every part spans at least `min_frames`. An exact monotone DP over a `grid`-frame lattice 
    maximizes sum_i score(i, a, b); then each cut, left to right, moves in `refine_step` frames within ±grid. `prefetch` receives 
    every (i, a, b) a pass will read, so a model scorer can batch them. None when no legal cut exists. Buffer cap bounds a span, 
    so the lattice (about (n / grid)^2 / 2 distinct crops for any k) stays small enough for the exact DP on every span."""
    lattice = sorted({0, n, *range(grid, n, grid)})
    pos = {q: j for j, q in enumerate(lattice)}
    segs = [
        (i, a, b) for i in range(k) for a in lattice for b in lattice   # only parts that can lie on a full path
        if b - a >= min_frames and (i > 0 or a == 0) and (i < k - 1 or b == n)
        and a >= i * min_frames and n - b >= (k - 1 - i) * min_frames
    ]
    prefetch(segs)
    best = np.full((k, len(lattice)), -np.inf); back = np.zeros((k, len(lattice)), int)
    for i, a, b in sorted(segs):   # part by part: best[i - 1] is complete before part i reads it
        prev = 0.0 if i == 0 else best[i - 1, pos[a]]
        if not np.isfinite(prev): continue
        value = prev + score(i, a, b)
        if value > best[i, pos[b]]: best[i, pos[b]], back[i, pos[b]] = value, pos[a]

    if not np.isfinite(best[k - 1, pos[n]]): return None
    cuts, j = [], pos[n]
    for i in range(k - 1, 0, -1): j = back[i, j]; cuts.append(lattice[j])
    cuts = cuts[::-1]
    for i in range(k - 1):
        lo, hi = (0 if i == 0 else cuts[i - 1]), (n if i == k - 2 else cuts[i + 1])
        reach = grid - grid % refine_step   # a whole number of steps, so the DP cut itself is a candidate on any grid
        cands = [c for c in range(cuts[i] - reach, cuts[i] + reach + 1, refine_step) 
                    if c - lo >= min_frames and hi - c >= min_frames]
        if not cands: continue
        prefetch([(i, lo, c) for c in cands] + [(i + 1, c, hi) for c in cands])
        cuts[i] = max(cands, key=lambda c: score(i, lo, c) + score(i + 1, c, hi))
    return cuts


class CropScorer:
    """log p_AR(text_i | the cascade crop of frames [a, b]) for one span, teacher-forced through the clean AR floor. Each crop is
    normalized on its own frames, as the cascade normalizes its clips; scores are cached, and `prefetch` batches every missing crop
    (frame_budget padded frames per encoder call) and its texts (token_budget label tokens per decoder call)."""

    def __init__(self, model, tokenizer, raw_poses: np.ndarray, texts: list[str], device, frame_budget: int, token_budget: int):
        self.model, self.raw, self.device = model, raw_poses, device
        self.ids = [tokenizer(t)["input_ids"] for t in texts]     # with EOS
        self.frame_budget, self.token_budget = int(frame_budget), int(token_budget)
        self.cache: dict[tuple[int, int, int], float] = {}
        self.encodes, self.seconds = 0, 0.0

    @torch.no_grad()
    def prefetch(self, triples) -> None:
        todo: dict[tuple[int, int], set[int]] = {}
        for i, a, b in triples:
            if (i, a, b) not in self.cache: todo.setdefault((a, b), set()).add(i)
        if not todo: return
        started = time.perf_counter()
        fe, lm = self.model.front_end, self.model.front_end.lm_model
        crops = sorted(todo, key=lambda ab: ab[1] - ab[0])
        s = 0
        while s < len(crops):
            n = 1
            while s + n < len(crops) and (n + 1) * (crops[s + n][1] - crops[s + n][0] + 2) <= self.frame_budget: n += 1
            batch, s = crops[s:s + n], s + n
            xs = [normalize_keypoints_unisign(self.raw[a:b + 1]) for a, b in batch]
            x = torch.zeros(len(xs), max(len(c) for c in xs), *xs[0].shape[1:], device=self.device)
            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=self.device)

            for r, c in enumerate(xs): x[r, :len(c)] = torch.as_tensor(c, device=self.device); mask[r, :len(c)] = True
            tap, m, _ = fe.extract_bio_tap(x, mask, None)
            enc, enc_mask = fe.encode_memory(tap, m)
            self.encodes += len(batch)
            rows = [(r, i) for r, ab in enumerate(batch) for i in sorted(todo[ab])]
            q = 0
            while q < len(rows):
                chunk, ntok = [], 0
                while q < len(rows) and (not chunk or ntok + len(self.ids[rows[q][1]]) <= self.token_budget):
                    chunk.append(rows[q]); ntok += len(self.ids[rows[q][1]]); q += 1

                labels = torch.full((len(chunk), max(len(self.ids[i]) for _, i in chunk)), -100, dtype=torch.long, device=self.device)
                for j, (_, i) in enumerate(chunk): labels[j, :len(self.ids[i])] = torch.tensor(self.ids[i], device=self.device)
                idx = torch.tensor([r for r, _ in chunk], device=self.device)
                hidden = lm.decoder(input_ids=lm._shift_right(labels), encoder_hidden_states=enc[idx],
                                    encoder_attention_mask=enc_mask[idx]).last_hidden_state
                if lm.config.tie_word_embeddings: hidden = hidden * (lm.model_dim ** -0.5)
                valid = labels != -100
                logp = F.log_softmax(lm.lm_head(hidden[valid]).float(), -1).gather(-1, labels[valid][:, None])[:, 0]
                sums = torch.zeros(len(chunk), dtype=torch.float64).index_add_(0, valid.nonzero()[:, 0].cpu(), logp.cpu().double())
                for (r, i), v in zip(chunk, sums.tolist()): self.cache[(i, *batch[r])] = float(v)
        self.seconds += time.perf_counter() - started


def likelihood_recut(
    events: dict, records, model, tokenizer, prior: tuple[float, float, float], device, 
    delta_s: float, min_span_s: float, cfg: dict
) -> tuple[dict, dict]:
    """(events with every split span cut again by the likelihood rule, cost). `events` = an RQ2 events file in source time 
    (parts 0..parts-1 of each split span). A split span that no record holds stays as it is (counted as unmatched)."""
    a0, b0, sigma = prior
    out, cost = {}, {"spans": 0, "unmatched": 0, "encodes": [], "seconds": []}
    for vid, evs in events.items():
        new = []
        for group in span_groups(evs):
            if len(group) < 2: new.extend(group); continue
            crop = _span_crop(group, records, vid)
            if crop is None: cost["unmatched"] += 1; new.extend(group); continue

            rec, start, end = crop
            first, last = group[0], group[-1]
            off, fps = float(rec.pose.offset_s), float(rec.pose.fps)
            raw, ts = load_pose_window(rec.pose, start, end, normalize=False)

            n = min(len(ts) - 1, int(round((last.end_s - off) * fps)) - int(np.floor(max(0.0, start) * fps + FRAME_SLACK)))
            grid, floor = seconds_to_frames(delta_s, fps), seconds_to_frames(min_span_s, fps)
            texts = [e.text or "" for e in group]
            words = [max(1, len(t.split())) for t in texts]
            
            scorer = CropScorer(model, tokenizer, raw, texts, device, cfg["frame_budget"], cfg["token_budget"])
            prior_lp = lambda i, a, b: -0.5 * (((b - a) / fps - (a0 + b0 * words[i])) / sigma) ** 2 - \
                                        math.log(sigma * math.sqrt(2 * math.pi))
            cost["spans"] += 1
            cuts = dp_cuts(
                lambda i, a, b: scorer.cache[(i, a, b)] + prior_lp(i, a, b), n, len(group), grid, floor,
                seconds_to_frames(float(cfg["refine_step_s"]), fps), prefetch=scorer.prefetch
            )
            if cuts is None: new.extend(group); continue   # no legal cut: the parts cannot all hold Λ_min
            times = [float(t) + off for t in ts]
            bounds = [first.start_s, *(times[c] for c in cuts), last.end_s]
            new.extend(replace(e, start_s=bounds[j], end_s=bounds[j + 1]) for j, e in enumerate(group))
            cost["encodes"].append(scorer.encodes); cost["seconds"].append(scorer.seconds)
        out[vid] = new
    return out, cost

# ---------------------------------------------------------------- attention cut

def pose_attention(cross_attentions, prompt_len: int, n_frames: int, n_tokens: int) -> np.ndarray:
    """(layers, tokens, frames): batch row 0 of each layer's cross-attention, heads averaged, with the task-prompt memory columns
    and the rows after the text tokens (the row that predicts EOS) dropped. The memory is prompt_len + one column per frame
    (UniSignFrontEndBase.encode_memory)."""
    a = torch.stack([layer[0].float().mean(0) for layer in cross_attentions])
    assert a.shape[-1] == prompt_len + n_frames, f"memory of {a.shape[-1]} columns, expected {prompt_len} prompt + {n_frames} frames"
    return a[:, :n_tokens, prompt_len:].cpu().numpy()


@torch.no_grad()
def span_attention(model, tokenizer, poses: np.ndarray, texts: list[str], device) -> tuple[np.ndarray, list[int]]:
    # (pose_attention of ONE teacher-forced decoder pass over the normalized crop `poses`, the token count of each part).
    fe, lm = model.front_end, model.front_end.lm_model
    ids = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
    labels = torch.tensor([sum(ids, []) + [tokenizer.eos_token_id]], device=device)
    tap, mask, _ = fe.extract_bio_tap(torch.as_tensor(poses, device=device)[None], None)
    enc, enc_mask = fe.encode_memory(tap, mask)
    out = lm.decoder(
        input_ids=lm._shift_right(labels), encoder_hidden_states=enc, encoder_attention_mask=enc_mask,
        output_attentions=True, use_cache=False, return_dict=True
    )
    return pose_attention(out.cross_attentions, fe.prompt_length(), len(poses), labels.shape[1] - 1), [len(i) for i in ids]


@torch.no_grad()
def own_span_attention(
    model, tokenizer, poses: np.ndarray, timestamps_s: np.ndarray, texts: list[str], device, read_kwargs: dict
) -> tuple[np.ndarray, list[int]]:
    """(pose_attention of ONE teacher-forced decoder pass of a gated AR arm over its own read of a span, the token count of each part).
    The read is the arm's decode read (MisalignedSLTModel.gated_read under `read_kwargs`: the gate with its anchor on the span, then the
    support crop), on `poses` normalized and `timestamps_s` window-relative as the caller's row reads them; the decoder pass reads Ω as
    the arm's decode does (front_end.ar_generate). The parts are teacher-forced joined by UNIT_SEPARATOR, as the arm writes and trains
    them (data.windowing.joined_target), and the separator rows are dropped, so each kept row is a token of one part."""
    fe, lm = model.front_end, model.front_end.lm_model
    p = torch.as_tensor(poses, dtype=torch.float32, device=device)[None]
    ts = torch.as_tensor(timestamps_s, dtype=torch.float32, device=device)[None]
    _, tap, mask, omega, _ = model.gated_read(p, torch.ones(p.shape[:2], dtype=torch.bool, device=device), ts, **read_kwargs)
    sep = tokenizer(UNIT_SEPARATOR, add_special_tokens=False)["input_ids"]
    assert len(sep) == 1, f"the unit separator must be one token, found {sep}"

    ids = tokenizer(f" {UNIT_SEPARATOR} ".join(texts), add_special_tokens=False)["input_ids"]
    labels = torch.tensor([ids + [tokenizer.eos_token_id]], device=device)
    enc, enc_mask = model.encode_memory(tap, mask, omega_bias=omega)
    with fe.ar_omega_context(omega):
        out = lm.decoder(
            input_ids=lm._shift_right(labels), encoder_hidden_states=enc, encoder_attention_mask=enc_mask, 
            output_attentions=True, use_cache=False, return_dict=True
        )
    a = pose_attention(out.cross_attentions, fe.prompt_length(), int(tap.shape[1]), len(ids))
    seps = [j for j, t in enumerate(ids) if t == sep[0]]
    return np.delete(a, seps, axis=1), [int(n) for n in np.diff([-1, *seps, len(ids)]) - 1]


def dtw_path(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cheapest path via `cost` (tokens x frames) from 1st cell to last, each step to next token, next frame or both (steps of Whisper's 
    timing.dtw). On an exact cost tie it takes the diagonal, then the next token. Whisper's strict comparison can take a costlier step 
    there. The token and frame index of every cell on it, in order."""
    n, t = cost.shape
    c, d = cost.tolist(), [[math.inf] * (t + 1) for _ in range(n + 1)]
    d[0][0] = 0.0
    for i in range(1, n + 1):
        up, row, ci = d[i - 1], d[i], c[i - 1]
        for j in range(1, t + 1): row[j] = ci[j - 1] + min(up[j - 1], up[j], row[j - 1])

    path, i, j = [], n, t
    while i > 0 and j > 0:   # d is infinite on its first row and column except at (0, 0), so the walk ends at 1st cell
        path.append((i - 1, j - 1))
        diag, up, left = d[i - 1][j - 1], d[i - 1][j], d[i][j - 1]
        if diag <= up and diag <= left: i, j = i - 1, j - 1
        elif up <= left: i -= 1
        else: j -= 1
    tok, frame = np.array(path[::-1]).T
    return tok, frame


def attention_cuts(a: np.ndarray, ntok: list[int], width: int) -> list[int]:
    """Whisper's readout on one span's attention `a` (tokens x frames): z-scores over tokens per frame, a `width`-frame median filter
    along frames (mirror padding, as Whisper's), a monotone DTW on the negative. The cut before part i+1 is the first frame of its 1st
    token on the path, so the cuts never decrease."""
    sd = a.std(0, keepdims=True)
    z = median_filter((a - a.mean(0, keepdims=True)) / np.where(sd > 0, sd, 1.0), size=(1, width), mode="mirror")
    tok, frame = dtw_path(-z)
    return [int(frame[tok == b].min()) for b in np.cumsum(ntok)[:-1]]

def floor_cuts(cuts: list[int], n: int, floor: int) -> list[int]:
    # The cuts clamped so every part [c_{i-1}, c_i] of the frames [0, n] holds at least `floor` frames (needs n >= parts x floor).
    out, lo, k = [], 0, len(cuts) + 1
    for i, c in enumerate(cuts):
        lo = min(max(c, lo + floor), n - (k - 1 - i) * floor)
        out.append(lo)
    return out

def attention_recut(
    events: dict, records, model, tokenizer, device, min_span_s: float, 
    layer_sets: list[list[int]], half_width_s: float, attend=None
) -> tuple[list[dict], dict]:
    """([per layer set: events with every split span cut again by attention rule], cost). `events` = an RQ2 events file in source time 
    (parts 0..parts-1 of each split span). 1 decoder pass per span serves every layer set. A split span that no record holds, or whose 
    parts can't all hold Λ_min, keeps its word-share cuts (counted as unmatched, too_short). `attend(record, start s, end s, group)` -> 
    (attention, part token counts) reads a span on frames of its crop. None reads it as clean floor reads a crop."""
    outs = [{} for _ in layer_sets]
    cost = {"spans": 0, "unmatched": 0, "too_short": 0, "floored": [0] * len(layer_sets), "seconds": []}
    for vid, evs in events.items():
        news = [[] for _ in layer_sets]
        for group in span_groups(evs):
            crop = _span_crop(group, records, vid) if len(group) > 1 else None
            if crop is None:
                cost["unmatched"] += len(group) > 1
                for new in news: new.extend(group)
                continue

            rec, start, end = crop
            first, last = group[0], group[-1]
            off, fps = float(rec.pose.offset_s), float(rec.pose.fps)
            poses, ts = load_pose_window(rec.pose, start, end)   # normalized on the crop's own frames, as the clean floor trains
            n = min(len(ts) - 1, int(round((last.end_s - off) * fps)) - int(np.floor(max(0.0, start) * fps + FRAME_SLACK)))
            floor = seconds_to_frames(min_span_s, fps)

            if n < len(group) * floor:   # the parts cannot all hold Λ_min
                cost["too_short"] += 1
                for new in news: new.extend(group)
                continue

            started = time.perf_counter()
            a, ntok = span_attention(model, tokenizer, poses, [e.text or "" for e in group], device) \
                        if attend is None else attend(rec, start, end, group)
            width = 2 * seconds_to_frames(half_width_s, fps) + 1

            for s, layers in enumerate(layer_sets):
                path_cuts = attention_cuts(a[[l - 1 for l in layers]].mean(0), ntok, width)
                cuts = floor_cuts(path_cuts, n, floor)
                cost["floored"][s] += cuts != path_cuts
                bounds = [first.start_s, *(float(ts[c]) + off for c in cuts), last.end_s]
                news[s].extend(replace(e, start_s=bounds[j], end_s=bounds[j + 1]) for j, e in enumerate(group))
            cost["spans"] += 1; cost["seconds"].append(time.perf_counter() - started)
        for out, new in zip(outs, news): out[vid] = new
    return outs, cost


def separator_commits(hypotheses: dict, k: int, delta_s: float, stride_s: float, fps: dict) -> dict:
    """Replayed commits on the open-span decodes {video: [{emission_time_s, span_start_s, buffer_end_s, text}]} (source time). Per 
    video the decodes form tracks as the commit gate forms histories: a start within delta FRAMES (at video fps) of the track's last 
    start, and a track not seen for more than k strides ends. A track commits its first unit at 1st stride where the text before 1st 
    UNIT_SEPARATOR is the same and non-empty in k decodes of k consecutive strides (a vote per unit prefix, not on whole text: the 
    hallucinated rest may change). The unit runs from the track start to its word-count share of the open span [start, buffer end] 
    at that stride (rule of `infer.stream.split_unit_events`). As FSM's commit mask does, a commit that would start before the end 
    of the video's previous replayed commit minus delta is not made. Returns {tracks, tracks_with_separator, commits: [{video_id, 
    start_s, end_s, commit_time_s, text, buffer_end_s}], rule}."""
    def prefix(text):
        if UNIT_SEPARATOR not in (text or ""): return None
        return " ".join(text.split(UNIT_SEPARATOR)[0].split()) or None
    
    commits, n_tracks, n_sep = [], 0, 0
    within = (k - 1) * stride_s + 0.5 * stride_s   # k decodes of k consecutive strides span at most k - 1 strides
    for vid, hyps in hypotheses.items():
        rate = float(fps[vid])
        d, frame = seconds_to_frames(delta_s, rate), (lambda t, rate=rate: int(round(t * rate)))
        tracks: list[list[dict]] = []

        for h in sorted(hyps, key=lambda h: h["emission_time_s"]):
            last = tracks[-1][-1] if tracks else None
            if last is not None and abs(frame(h["span_start_s"]) - frame(last["span_start_s"])) <= d \
                and h["emission_time_s"] - last["emission_time_s"] <= k * stride_s + 0.5 * stride_s: tracks[-1].append(h)
            else: tracks.append([h])

        committed_until = float("-inf")
        for track in tracks:
            n_tracks += 1
            pre = [prefix(h["text"]) for h in track]
            n_sep += any(pre)
            i = next((i for i in range(k - 1, len(track)) if pre[i] and len(set(pre[i - k + 1:i + 1])) == 1
                      and track[i]["emission_time_s"] - track[i - k + 1]["emission_time_s"] <= within), None)
            if i is None: continue
            h = track[i]
            parts = split_unit_text(h["text"])   # the live split's parts and measure
            share = max(1, len(parts[0].split())) / sum(max(1, len(part.split())) for part in parts)
            span = (h["span_start_s"], h["span_start_s"] + share * (h["buffer_end_s"] - h["span_start_s"]))
            if span[0] < committed_until - delta_s: continue   # its frames are already committed
            committed_until = span[1]
            commits.append({
                "video_id": vid, "start_s": span[0], "end_s": span[1], "commit_time_s": h["emission_time_s"],
                "text": pre[i], "buffer_end_s": h["buffer_end_s"]
            })
    return {"tracks": n_tracks, "tracks_with_separator": n_sep, "commits": commits,
            "rule": {"k": int(k), "delta_s": float(delta_s), "stride_s": float(stride_s)}}
