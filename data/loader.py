from __future__ import annotations
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass
from typing import Any, Iterable
from pathlib import Path
from tqdm import tqdm

import re, random, csv, html, json, unicodedata, zlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, get_worker_info
from train import distributed as dist

from data.windowing import SentenceSpan
from poses import PoseIndex, build_pose_index
from poses.pose_io import META_FILENAME, load_video_meta

TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")
WORD_TIMING_RE = re.compile(r"<\d{1,2}:\d{2}:\d{2}[.,]\d{3}>")
SPEAKER_PREFIX_RE = re.compile(r"^[A-Z][A-Z\s_-]{1,30}:\s*")
# Non-verbal annotations / stylistic markers: (laughter), [music], *flush*. 
# Newline-free and length-bounded so an unclosed bracket cannot swallow the rest of the cue.
#
# Square brackets and asterisks are subtitle annotation conventions and are stripped outright. ROUND parentheses are NOT: in ordinary prose 
# they carry lexical content, and captions here are the translation targets, so deleting them corrupts the reference and penalises a correct 
# translation. Measured on asf, blanket removal cost drug brand names ("Comirnaty (Pfizer)") and destroyed whole sentences ("(e.g., fever, 
# cough, sore throat)." -> "."). Parentheses are therefore removed only when their content is purely non-verbal (`is_noise_caption`).
BRACKET_ANNOTATION_RE = re.compile(r"\[[^\[\]\n]{0,80}\]|\*[^*\n]{0,80}\*")
PAREN_GROUP_RE = re.compile(r"\([^()\n]{0,80}\)")
LEADING_SYMBOL_RE = re.compile(r"^[\s♪♫•·\-–—>»]+")
TRAILING_SYMBOL_RE = re.compile(r"[\s♪♫•·]+$")  # a cue often closes with the note it opened with
# Speaker identifier: ONE word (optionally two, e.g. "MRS SMITH:") then a colon, at the cue start. Bounded to a single token so a genuine 
# clause like "One thing: ..." keeps its text — that costs recall on rare speaker labels but never deletes signed content.
SPEAKER_ID_RE = re.compile(r"^[A-Za-z][\w'\-]{0,20}(?:\s+[A-Z][\w'\-]{0,20})?:\s+")
NOISE_WORD_RE = re.compile(r"[a-z]+")
NOISE_CAPTION_WORDS = {
    "applause", "background", "foreign", "gentle", "inaudible", "laugh", "laughs",
    "laughter", "music", "piano", "silence", "silent",
}
# Unicode punctuation -> ASCII, so a curly apostrophe in a reference ("Sydney\u2019s") and a straight one from the model ("Sydney's") are the 
# SAME BLEU token. Applied after html.unescape (so &#8217; is folded too) and before the leading-symbol strip (so a leading en/em dash still 
# counts as a speaker dash). nbsp/thin spaces -> space, later collapsed by the \\s+ pass.
_PUNCT_NORMALISE = {ord(k): v for k, v in {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2026": "...",
    "\u00a0": " ", "\u2009": " ", "\u200a": " ", "\u202f": " ", "\u200b": "",
    # Zero-width formatting: invisible, but each fuses to its neighbour and yields a BLEU token distinct from
    # the identical-looking model output ("text\u2060 hints" != "text hints"). Deleted, not spaced.
    "\u2060": "", "\ufeff": "", "\u200c": "", "\u200d": "", "\u00ad": "",
}.items()}
_PUNKT = None
_SENTENCE_FINAL_RE = re.compile(r'[.!?][\"\')\]]*\s*$')  # a unit ending here is a Punkt-confirmed sentence
_LANG_RECORDS_CACHE: dict[tuple, list[VideoRecord]] = {}


@dataclass(frozen=True)
class VideoRecord:
    language: str
    video_id: str
    pose: PoseIndex
    subtitle_path: Path
    sentences: tuple[SentenceSpan, ...]


def timestamp_to_seconds(value: str) -> float:
    value = value.replace(",", ".")
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def clean_caption_text(lines: Iterable[str]) -> str:
    """Normalise one cue's text. Stage order follows Lost-in-Translation (arXiv 2512.08040) §A.7.

    Entities are decoded BEFORE tags are stripped: an entity-encoded tag (`&lt;font&gt;`) is invisible to the tag
    regex until it is decoded, so the reverse order leaks markup into the reference text.
    """
    raw = " ".join(line.strip() for line in lines if line.strip())
    raw = WORD_TIMING_RE.sub(" ", raw)          # <00:00:01.234> karaoke timings
    raw = html.unescape(raw)                    # &amp; -> &, &lt;font&gt; -> <font>
    raw = raw.translate(_PUNCT_NORMALISE)       # curly quotes/dashes/nbsp -> ASCII (BLEU-token parity)
    raw = TAG_RE.sub(" ", raw)                  # ...then any real or decoded markup
    raw = BRACKET_ANNOTATION_RE.sub(" ", raw)   # [music] / *flush* — annotation-only conventions
    # (laughter) yes; (Pfizer) no — see PAREN_GROUP_RE. `is_noise_caption` is the single definition of
    # "purely non-verbal", so the inline test and the whole-cue test can never disagree.
    raw = PAREN_GROUP_RE.sub(lambda m: " " if is_noise_caption(m.group(0)) else m.group(0), raw)
    raw = LEADING_SYMBOL_RE.sub("", raw.strip())  # leading music notes, bullets, speaker dashes
    raw = TRAILING_SYMBOL_RE.sub("", raw)
    raw = SPEAKER_ID_RE.sub("", raw)            # "John:" / "NARRATOR:" at the start of a cue
    raw = re.sub(r"\s+([,.!?;:])", r"\1", raw)  # no space before punctuation
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw.strip("\"' ").strip()


def is_noise_caption(text: str) -> bool:
    """True for captions that are only non-signed stage directions.

    Keep real sentences that merely contain words like "music" or "Facebook"; drop only whole-cue
    annotations such as "AUDIENCE: (APPLAUSE)" or "(gentle piano music)".
    """
    text = SPEAKER_PREFIX_RE.sub("", text.strip())
    # Sound glyphs are annotation marks, not words: strip them so the WORD test decides. Catches "(\u266a\u266a\u266a)" and 
    # "(\u266amusic\u266a)". Deliberately does NOT catch "(\u266arock music\u266a)": "rock" is not a noise word, and widening 
    # the set is unsafe — measured over asf/bfi/ase, whole-cue parentheticals that survive are ~0.2% of cues and their frequent 
    # words are "the", "to", "you", "breathe", "slowly", i.e. real signed content in parentheses. Deleting those would corrupt 
    # references, so the filter stops here.
    text = re.sub(r"[\u266a\u266b\u266c\u2669\u25ba\u25c4]+", " ", text)
    stripped = text.strip("[](){} \t\r\n").casefold()
    if not stripped: return True
    words = NOISE_WORD_RE.findall(stripped)
    return bool(words) and all(word in NOISE_CAPTION_WORDS for word in words)


def parse_vtt(path: str | Path, drop_noise: bool = False) -> list[tuple[float, float, str]]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    captions: list[tuple[float, float, str]] = []
    i = 0
    while i < len(lines):
        match = TIMESTAMP_RE.search(lines[i])
        if match is None:
            i += 1
            continue
        start_s = timestamp_to_seconds(match.group("start"))
        end_s = timestamp_to_seconds(match.group("end"))
        i += 1
        text_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            if "-->" not in lines[i]: text_lines.append(lines[i])
            i += 1
        text = clean_caption_text(text_lines)
        if drop_noise and is_noise_caption(text): text = ""
        if text and end_s > start_s: captions.append((start_s, end_s, text))
        i += 1
    return captions


def _punkt_tokenizer():
    # Pretrained English Punkt: segments the caption stream into sentences, joining fragments by the ABSENCE of a boundary and 
    # splitting multi-sentence cues, with an abbreviation model so "Dr." / "U.S." never split. Loaded once; None if unavailable, 
    # so the caller falls back to the raw cues rather than crashing.
    global _PUNKT
    if _PUNKT is None:
        try:
            import nltk
            try: _PUNKT = nltk.data.load("tokenizers/punkt/english.pickle")
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                _PUNKT = nltk.data.load("tokenizers/punkt/english.pickle")
        except Exception:
            print("[loader] WARNING: Punkt unavailable; caption cues are used as-is (no sentence reconstruction).", flush=True)
            _PUNKT = False
    return _PUNKT or None


def non_latin_ratio(texts: Iterable[str]) -> float:
    """Share of a video's ALPHABETIC caption characters that are not Latin script.

    Script, not language ID: a target-English corpus written in Latin script needs no model to spot a video whose
    captions are Japanese or Chinese, and a ratio is robust to the odd quoted term in a way a presence test is not.
    """
    total = non_latin = 0
    for text in texts:
        for ch in text:
            if not ch.isalpha(): continue
            total += 1
            try: name = unicodedata.name(ch)
            except ValueError:
                non_latin += 1
                continue
            if not name.startswith("LATIN"): non_latin += 1
    return non_latin / total if total else 0.0


def _clamp_overlaps(captions: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Force a strictly non-overlapping, time-ordered span sequence.

    Applied to the FINAL caption stream (after rolling-duplicate merging and sentence reconstruction), so it is the
    1 place that guarantees the invariant every downstream consumer assumes. A span left fully inside its predecessor 
    after clamping is dropped: it carries no exclusive frames, so it can never be selected or scored.
    """
    out: list = []
    prev_end = float("-inf")
    # Tuples may carry a 4th `reliable` field (quarantined chains) — pass any extra fields through untouched.
    for c in sorted(captions, key=lambda x: (x[0], x[1])):
        s, e = max(c[0], prev_end), c[1]
        if e <= s: continue  # wholly swallowed by the previous span
        out.append((s, e, *c[2:])); prev_end = e
    return out


def _quarantine_end_straddlers(captions: list[tuple], duration_s: float, slack_s: float = 1.0) -> list[tuple]:
    """Captions that START inside pose stream but END well past it become QUARANTINED spans over visible frames, instead of being dropped.

    Streams end before the caption timeline (duration = frames/24 underestimates the video), so this straddle is systematic, not an edge 
    case. Dropping caption relabels its visible frames as uncaptioned — and a leftover tail gap of trusted_gap_s or less is then supervised 
    as trusted `O` over frames the caption says are signing. Clipping to `duration_s` as a RELIABLE span would instead mint an end boundary 
    no cue marks, with text for signing partly outside the poses. Quarantine is the one honest option (same known-wrong-labels-are-excluded 
    rule reconstruct_sentences applies): frames UNK, never an anchor, reference, or Mode-4 gap. Ends within `slack_s` of the stream end are 
    left alone — the existing span filter tolerates them, and their label error is below the timestamp noise floor.
    """
    return [
        (c[0], float(duration_s), c[2], False) if (c[0] < duration_s and c[1] > duration_s + float(slack_s)) else c
        for c in captions
    ]


def reconstruct_sentences(captions: list[tuple[float, float, str]], max_tokens: int = 60) -> list[tuple]:
    """Rebuild sentence units from display-wrapped caption cues; QUARANTINE what cannot be labelled correctly.

    Punkt segments concatenated cue text; a junction that a Punkt sentence straddles is sentence-interior, and junction-joined cues form a 
    chain. Each candidate unit (chain/single cue) is then judged by Punkt segmentation OF ITS OWN TEXT — k sentences — under 1 precondition:

    PUNCTUATION-RELIABILITY GATE (`max_tokens`): if any of a unit's own "sentences" exceeds it, the channel omits terminal punctuation,
    junction evidence is vacuous, and the source cues are kept as-is. Measured single-cue Punkt-sentence lengths over asf+bfi+ase are
    p50 7 / p99 21 / max 40 tokens, so the default sits well above any real sentence and fires only on fusion. It is NOT redundant with
    the k>1 quarantine and quarantine cannot replace it: on an unpunctuated channel a single stray period makes Punkt read the whole run
    as ONE sentence, so k==1 and the unit would be emitted RELIABLE (measured: 12 unpunctuated cues fuse into one 83-token 36s span with
    the gate off, 12 preserved cues with it on). Quarantine catches unlocatable boundaries; this gate catches absent evidence.

    For reliable segmentations:
      single cue, k <= 1           -> kept as-is (the caption author's own sentence hypothesis).
      chain,      k == 1, punct.   -> merged unit: ONE sentence, outer bounds = real cue timestamps.
      chain,      k == 1, no punct -> source cues kept (no completion evidence).
      k > 1                        -> QUARANTINED `(start, end, text, False)`: every interior boundary in a chain is mid-cue by construction 
                                      (a boundary AT a junction would have broken the chain there), and a multi-sentence single cue likewise. 
                                      No cue timestamp marks them, and interpolating from character position is invalid under P1 (sign order 
                                      is not text order; measured ~35% BLEU deficit on interpolated-boundary spans). Known-wrong labels are 
                                      excluded, not approximated: the region becomes `SentenceSpan(reliable=False)` — frames UNK, never an 
                                      anchor or reference, never a Mode-4 "gap" (the span still occupies its timeline).

    Cues are never SPLIT: separating back-to-back sentences in TIME is the semi-Markov duration decode's job at inference, not label builder's. 
    Every emitted timestamp is a source-cue timestamp.
    """
    tok = _punkt_tokenizer()
    if tok is None or not captions: return list(captions)
    parts: list[str] = []; char_cue: list[int] = []
    for ci, (cs, ce, t) in enumerate(captions):
        if parts: parts.append(" "); char_cue.append(-1)      # junction char between cue ci-1 and ci
        parts.append(t); char_cue.extend([ci] * len(t))

    text = "".join(parts)
    crossed: set[int] = set()                                 # junction k = boundary between cue k and k+1
    for a, b in tok.span_tokenize(text):
        ids = sorted({c for c in char_cue[a:b] if c != -1})
        for x, y in zip(ids, ids[1:]):
            if y == x + 1: crossed.add(x)
    out: list[tuple] = []

    def emit(run: list[int]) -> None:
        s0, e1 = captions[run[0]][0], captions[run[-1]][1]
        joined = " ".join(captions[c][2] for c in run).strip()
        sents = [joined[a:b] for a, b in tok.span_tokenize(joined)]
        if any(len(x.split()) > int(max_tokens) for x in sents):   # punctuation-unreliable: evidence is vacuous
            out.extend(captions[c] for c in run)
        elif len(sents) > 1: out.append((s0, e1, joined, False))   # interior mid-cue boundaries: quarantine
        elif len(run) == 1: out.append(captions[run[0]])
        elif _SENTENCE_FINAL_RE.search(joined): out.append((s0, e1, joined))
        else: out.extend(captions[c] for c in run)                 # unpunctuated remnant: keep the author's cues
        
    run = [0]
    for k in range(len(captions) - 1):
        if k in crossed: run.append(k + 1)
        else: emit(run); run = [k + 1]
    emit(run)
    return out


def merge_rolling_captions(captions: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Sort cues and merge rolling-caption duplicates (YouTube auto/scroll subs re-display the same text across overlapping cues). 
    2 overlapping cues whose texts duplicate or contain 1 another are 1 sentence shown twice, not 2 sentences — left unmerged they produce 
    overlapping SentenceSpans, which corrupt BIO labels (a neighbour's `I` overwrites the closing `O`) and make the first-complete-span 
    rule ill-defined. Genuine overlaps with distinct text are kept as-is (GT boundaries are treated as clean; this is caption-format 
    cleanup, not boundary editing). Downstream also assumes time-ordered spans (`_mode3_spec` uses `anchor_idx + 1` as successor).
    
    For example, the same text may be shown across 2 cues with a rolling update:
    0:00:01.000 --> 0:00:05.000
    HELLO WORLD
    0:00:04.000 --> 0:00:08.000
    HELLO WORLD
    becomes a single span 0:00:01.000 --> 0:00:08.000 HELLO WORLD, instead of 2 overlapping spans with identical text.
    """
    if not captions: return captions
    merged: list[tuple[float, float, str]] = []
    for start_s, end_s, text in sorted(captions, key=lambda c: (c[0], c[1])):
        if merged:
            prev_start, prev_end, prev_text = merged[-1]
            overlap = start_s < prev_end
            duplicate = text == prev_text or text in prev_text or prev_text in text
            if overlap and duplicate:
                merged[-1] = (prev_start, max(prev_end, end_s), text if len(text) >= len(prev_text) else prev_text)
                continue
        merged.append((start_s, end_s, text))
    return merged


def _subtitle_score(path: Path, preferred_suffixes: list[str], reject_suffixes: list[str]) -> tuple[int, int, str]:
    name = path.name
    for rejected in reject_suffixes:
        if name.endswith(rejected): return (10_000, 0, name)
    for rank, suffix in enumerate(preferred_suffixes):
        if name.endswith(suffix): return (rank, 0, name)
    return (len(preferred_suffixes) + 100, 0, name)


def looks_flattened_transcript(
    captions: list[tuple[float, float, str]], max_cues: int = 2, min_chars: int = 500, max_chars_per_second: float = 120.0,
) -> bool:
    """Reject YouTube VTT variants that put the whole transcript in one short cue.

    ASF commonly ships paired files where `.en-GB.vtt` has normal cue timing but `.en-en-GB.vtt` contains thousands of characters in first 
    few seconds and empty cues afterwards. Such files are unusable for pose-text alignment and should lose to any non-flattened candidate.
    """
    if not captions or len(captions) > int(max_cues): return False
    total_chars = sum(len(text) for _, _, text in captions)
    if total_chars < int(min_chars): return False
    max_cps = 0.0
    for start_s, end_s, text in captions:
        dur = max(float(end_s - start_s), 1e-3)
        max_cps = max(max_cps, len(text) / dur)
    return max_cps >= float(max_chars_per_second)


def find_best_subtitle(
    subtitle_root: str | Path, video_id: str,
    preferred_suffixes: list[str], reject_suffixes: list[str],
    min_caption_chars: int = 2, reject_flattened_transcripts: bool = True,
    flattened_max_cues: int = 2, flattened_min_chars: int = 500, flattened_max_chars_per_second: float = 120.0, 
    drop_noise: bool = False, lang_prefix: str | None = None,
) -> Path | None:
    subtitle_root = Path(subtitle_root)
    # `lang_prefix` restricts to `<vid>.<prefix>*.vtt` (e.g. "en" → .en-GB/.en; "de" → .de*). The preferred/reject
    # suffix lists are English-oriented and do NOT encode language, so harvesting shard tracks for a non-English
    # target needs this to avoid picking an English track and labelling it the target language.
    pattern = f"{video_id}.{lang_prefix}*.vtt" if lang_prefix else f"{video_id}*.vtt"
    candidates = sorted(subtitle_root.glob(pattern))
    scored: list[tuple[tuple[int, int, str], Path]] = []
    for path in candidates:
        try: parsed = parse_vtt(path, drop_noise=drop_noise)
        except OSError: continue

        char_count = sum(len(text) for _, _, text in parsed)
        if char_count < min_caption_chars: continue
        if reject_flattened_transcripts and looks_flattened_transcript(
            parsed, max_cues=flattened_max_cues, min_chars=flattened_min_chars, 
            max_chars_per_second=flattened_max_chars_per_second,
        ): continue
        score = _subtitle_score(path, preferred_suffixes, reject_suffixes)
        scored.append(((score[0], -char_count, score[2]), path))
    return min(scored)[1] if scored else None


def best_subtitle(subtitle_root: str | Path, video_id: str, subtitle_cfg: dict, lang_prefix: str | None = None) -> Path | None:
    """`find_best_subtitle` driven by the `subtitles:` config block — the ONE selection rule, shared by the loader
    (lang_prefix=None; only the canonical `<vid>.<target>.vtt` exists) and prepare_data (shard tracks, lang_prefix=target)."""
    return find_best_subtitle(
        subtitle_root, video_id,
        preferred_suffixes=list(subtitle_cfg.get("preferred_suffixes", [".en.vtt"])),
        reject_suffixes=list(subtitle_cfg.get("reject_suffixes", [".en-orig.vtt"])),
        min_caption_chars=int(subtitle_cfg.get("min_caption_chars", 2)),
        reject_flattened_transcripts=bool(subtitle_cfg.get("reject_flattened_transcripts", True)),
        flattened_max_cues=int(subtitle_cfg.get("flattened_max_cues", 2)),
        flattened_min_chars=int(subtitle_cfg.get("flattened_min_chars", 500)),
        flattened_max_chars_per_second=float(subtitle_cfg.get("flattened_max_chars_per_second", 120.0)),
        drop_noise=bool(subtitle_cfg.get("drop_noise_captions", True)), lang_prefix=lang_prefix,
    )


def _load_signverse_splits(path: Path) -> dict[str, str]:
    # str(Path("")) is ".", which exists and is a directory — guard before opening.
    if not path.name or not path.is_file(): return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows: return {}
    id_cols = ["video_id", "youtube_id", "id", "video", "source_video"]
    split_cols = ["split", "subset", "partition"]
    id_col = next((c for c in id_cols if c in rows[0]), None)
    split_col = next((c for c in split_cols if c in rows[0]), None)
    if id_col is None or split_col is None: return {}

    result: dict[str, str] = {}
    for row in rows:
        video_id = (row.get(id_col) or "").strip()
        split = (row.get(split_col) or "").strip().lower()
        if split == "val": split = "dev"
        if video_id and split in {"train", "dev", "test"}: result[video_id] = split
    return result


def build_splits(video_ids: list[str], split_cfg: dict) -> dict[str, list[str]]:
    csv_path = str(split_cfg.get("signverse_csv", "") or "")
    signverse = _load_signverse_splits(Path(csv_path))
    # Fail loud rather than fall through to the random split: signverse_csv is CWD-relative, so an entry point started
    # from another directory would silently re-partition every video, making a checkpoint trained under one split and
    # evaluated under the other train-on-test. The fallback below is only for configs that declare no CSV at all.
    if csv_path and not signverse: raise FileNotFoundError(
        f"splits.signverse_csv={csv_path!r} is configured but unusable from cwd {Path.cwd()} (missing, empty, or "
        f"no recognised id/split columns). Refusing the random fallback split."
    )
    if signverse:
        splits = {"train": [], "dev": [], "test": []}
        unmatched = 0
        for video_id in video_ids:
            split = signverse.get(video_id)
            if split in splits: splits[split].append(video_id)
            else: unmatched += 1
        # A CSV that parses but shares no ids with the pose index (wrong language, stale export, ids vs stems) is the
        # same train-on-test hazard as a missing one — without this the loop falls through to the rng below.
        if not any(splits.values()): raise ValueError(
            f"splits.signverse_csv={csv_path!r} parsed {len(signverse)} rows but matched NONE of the {len(video_ids)} "
            f"pose videos (e.g. {sorted(video_ids)[:3]} vs {sorted(signverse)[:3]}). Refusing the random fallback split."
        )
        if unmatched: print(
            f"[loader] {unmatched}/{len(video_ids)} pose videos absent from the split CSV (dropped from all splits). "
            f"Add them to {csv_path} to include them.", flush=True
        )
        return {k: sorted(v) for k, v in splits.items()}

    rng = random.Random(int(split_cfg.get("fallback_seed", 42)))
    ids = sorted(video_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * float(split_cfg.get("fallback_train", 0.8))))
    n_dev = int(round(n * float(split_cfg.get("fallback_dev", 0.1))))
    return {
        "train": sorted(ids[:n_train]),
        "dev": sorted(ids[n_train:n_train + n_dev]),
        "test": sorted(ids[n_train + n_dev:]),
    }


def _split_caption_sets(root: Path, video_ids, subtitle_cfg: dict, drop_noise: bool) -> dict[str, set[str]]:
    # {video_id: {normalised caption}} for overlap testing. Parses subtitles only (no poses).
    out: dict[str, set[str]] = {}
    for vid in tqdm(video_ids, desc="[loader] Splitting caption sets", unit="vid", leave=False, dynamic_ncols=True):
        path = best_subtitle(root / "subs", vid, subtitle_cfg)
        if path is None: continue
        caps = {
            " ".join(str(t).lower().split()) 
            for _, _, t in merge_rolling_captions(parse_vtt(path, drop_noise=drop_noise)) if t
        }
        if caps: out[vid] = caps
    return out


def _duplicate_pairs(caps: dict[str, set[str]], cfg: dict) -> list[tuple[str, str, float]]:
    """Near-duplicate video pairs — the same talk re-uploaded under a different id.

    A caption identifies content only if it is RARE (document frequency <= df_cap; above that it is series
    boilerplate such as a scripted contact-info outro, which otherwise chains unrelated videos into one cluster)
    and LONG enough (>= min_words; single-sign vocabulary clips share a word with everything). A pair is flagged
    when its shared identifying captions cover more than `cover` of the smaller video's identifying set.
    Videos below min_captions are skipped: the cover statistic is quantised to 1/n and is meaningless there.
    """
    df_cap, min_words = int(cfg.get("df_cap", 20)), int(cfg.get("min_words", 4))
    min_caps, cover = int(cfg.get("min_captions", 10)), float(cfg.get("cover", 0.5))
    ident = {v: k for v, cs in caps.items() if len(k := {c for c in cs if len(c.split()) >= min_words}) >= min_caps}
    inv: dict[str, set[str]] = defaultdict(set)
    for v, cs in ident.items():
        for c in cs: inv[c].add(v)

    shared: dict[tuple[str, str], int] = defaultdict(int)
    for c, vs in inv.items():
        if 2 <= len(vs) <= df_cap:
            for a, b in combinations(sorted(vs), 2): shared[(a, b)] += 1

    pairs = [(a, b, n / min(len(ident[a]), len(ident[b]))) for (a, b), n in shared.items()]
    return sorted([p for p in pairs if p[2] > cover], key=lambda p: -p[2])


def assert_pool_safe(cfg: dict) -> None:
    """A pooled run may not read a MEASURED, language-keyed artifact.

    Measured calibration (jitter, mode ratios) describes one corpus decoded by one segmenter. Pooling the data while keeping 1 language's 
    artifact trains the pool under that language's error distribution — a silent mismatch, since the path resolves and the run looks healthy. 
    A pool has no such artifact, so it must train on DESIGNED fallbacks (`source: null`).

    Deployment constants are unaffected: delta/Lambda/buffer_cap/the decode triple are measured per TARGET language after pretraining, with 
    the deployed head, and the head is always evaluated at the context it trained under (`rope_eval_chunk_s` pinned in the checkpoint).
    """
    if not cfg.get("pretrain_languages"): return
    bad = [k for k in ("jitter", "mode_ratios") if (cfg.get(k) or {}).get("source")]
    if bad: raise SystemExit(
        f"pretrain_languages={list(cfg['pretrain_languages'])} but {', '.join(f'{k}.source' for k in bad)} "
        f"is set to a per-language artifact ({', '.join(str(cfg[k]['source']) for k in bad)}). A pool has no single "
        f"target language: set those to null to train on the designed fallbacks, or drop pretrain_languages."
    )


def resolve_pretrain_records(
    cfg: dict, data_cfg: dict, language: str, split: str, requested: str | None = None, epoch: int = 0,
) -> tuple[list[VideoRecord], dict[str, int] | None]:
    """Records for a SEGMENTATION trainer: the target language alone, or the multilingual pretraining pool.

    `pretrain_languages` in trainer config switches it on. Both segmentation trainers (S1 and the Moryossef baseline) route through here, 
    so 2 arms can never end up trained on different pools by accident — which is what would make a cascade comparison unfair.
    """
    langs = cfg.get("pretrain_languages") or None
    if not langs:
        recs, _ = load_language_records(data_cfg, language, split=split)
        return recs, None
    
    if requested: raise SystemExit(
        f"--language {requested!r} is meaningless with pretrain_languages={list(langs)}. "
        f"Drop --language to pretrain, or set pretrain_languages: null for a monolingual run on that language."
    )
    langs = [str(x) for x in langs]
    recs, realised = load_multilingual_records(
        data_cfg, langs, split=split, temperature=float(cfg.get("pretrain_temperature", 0.5)),
        seed=int(cfg.get("seed", 42)), epoch=int(epoch),
    )
    return recs, realised


def _cached_language_records(data_cfg: dict, language: str, split: str) -> list[VideoRecord]:
    """Per-language record lists for the pool, parsed ONCE per process.

    The per-epoch rotation re-resolves the pool every epoch, but only the sub-sample changes — the underlying corpus does not. Without this 
    cache each epoch re-parses every subtitle file, re-runs Punkt sentence reconstruction and the dedup pair scan for all pooled languages. 
    Keyed by the config content that shapes the records, so an edited data config is not served a stale corpus; records are never mutated 
    downstream (the pool copies before shuffling), so sharing one list across epochs is safe.
    """
    lang_fingerprint = json.dumps(
        {"lang": (data_cfg.get("languages") or {}).get(language), "subtitles": data_cfg.get("subtitles"), "splits": data_cfg.get("splits")},
        sort_keys=True, default=str,
    )
    # The active loader function is part of the key: tests monkeypatch `load_language_records`, and a key of
    # (language, split, fingerprint) alone would serve one test's fake records to the next.
    key = (language, str(split), lang_fingerprint, load_language_records)
    if key not in _LANG_RECORDS_CACHE: _LANG_RECORDS_CACHE[key] = load_language_records(data_cfg, language, split=split)[0]
    return _LANG_RECORDS_CACHE[key]


def load_multilingual_records(
    data_cfg: dict, languages: list[str], split: str, temperature: float = 0.5, seed: int = 42, epoch: int = 0,
) -> tuple[list[VideoRecord], dict[str, int]]:
    """Records pooled across languages for language-agnostic SEGMENTATION pretraining.

    Sentence boundaries are marked by prosody (pauses, holds, movement dynamics) that is shared across signed languages, so a BIO 
    head can be pretrained on several corpora at once. Translation cannot: it is language-specific and stays monolingual (stage 2).

    Corpora differ in size by several-fold, so proportional pooling would make the largest one the de-facto training set. Sampling 
    weights are TEMPERATURE-FLATTENED, the standard multilingual-NMT recipe: `p_l propto n_l ** temperature` — 1.0 is proportional, 
    0.0 uniform, 0.5 the usual compromise.

    Balance is reached by SUB-sampling the over-represented languages, never by replicating the under-represented ones. Replicating
    up to the largest corpus makes the pool as big as the most-scaled-up language demands: on ase/asf/bfi that was a 16x epoch with
    each asf video repeated ~5 times WITHIN one epoch, so the model saw many epochs' worth of data (and heavy repetition) before the
    first checkpoint. Sub-sampling keeps 1 epoch a bounded, comparable unit of compute.

    `epoch` rotates WHICH subset each language contributes, so no video is permanently discarded: coverage of the large corpora is
    spread across epochs instead of forced into one. Deterministic in (seed, epoch), so a resumed run replays its epoch exactly.

    Returns the pooled records and the realised per-language video counts, which belong in the paper: the sampling rates are part 
    of the experimental setup, not an implementation detail.
    """
    if not languages: raise ValueError("load_multilingual_records needs at least one language")
    per_lang: dict[str, list[VideoRecord]] = {}
    for lang in languages:
        recs = _cached_language_records(data_cfg, lang, split)
        if recs: per_lang[lang] = recs

    if not per_lang: raise ValueError(f"no records for any of {languages} on split {split!r}")
    if len(per_lang) == 1 or split == "test": # TEST is pooled AS-IS: it is a REPORTING set.
        pooled = [r for recs in per_lang.values() for r in recs]
        return pooled, {k: len(v) for k, v in per_lang.items()}

    counts = {k: len(v) for k, v in per_lang.items()}
    weights = {k: n ** float(temperature) for k, n in counts.items()}
    total_w = sum(weights.values())
    # SUB-sample to the target shares: pick the pool size that the most over-represented language can support
    # WITHOUT replication, i.e. the largest total for which every target <= that language's real video count.
    # Upsampling instead (scaling up to the biggest corpus) repeats the small corpora several times inside ONE
    # epoch, so the model sees many epochs' worth of a language before the first checkpoint and overfits during
    # epoch 1 — the failure this bound exists to prevent. Temperature then only sets the SHARES, never the
    # repetition, so lowering it rebalances instead of inflating the epoch.
    scale = min(counts[k] * total_w / weights[k] for k in counts)
    rng = random.Random(int(seed))
    pooled: list[VideoRecord] = []
    realised: dict[str, int] = {}

    for lang, recs in per_lang.items():
        target = max(1, min(len(recs), int(round(scale * weights[lang] / total_w))))
        # ROTATE the slice per epoch so a sub-sampled corpus is never permanently truncated: a language reduced to
        # `target` of `n` videos covers all of them every ceil(n / target) epochs. Order is shuffled once per
        # language (seeded, so it is stable across epochs and resumes) and the window then advances by `target`.
        order = list(recs)
        # crc32, NOT hash(): str.__hash__ is PYTHONHASHSEED-salted, so hash() gave a DIFFERENT sub-sample on
        # every process launch — train and dev alike. Best-checkpoint selection then compared scores measured
        # on different dev sets, and no run was reproducible or resumable. crc32 is stable across processes.
        random.Random(int(seed) ^ zlib.crc32(lang.encode())).shuffle(order)
        offset = (int(epoch) * target) % len(order)
        pooled.extend((order + order)[offset:offset + target])
        realised[lang] = target

    rng.shuffle(pooled)
    shares = {k: round(v / sum(realised.values()), 3) for k, v in realised.items()}
    print(f"[loader] multilingual {split}: videos {counts} -> sampled {realised} (tau={temperature}, shares {shares})", flush=True)
    return pooled, realised


def load_language_records(data_cfg: dict, language: str, split: str | None = None) -> tuple[list[VideoRecord], dict[str, list[str]]]:
    lang_cfg = data_cfg["languages"][language]
    root = Path(lang_cfg["root"])
    # Per-video fps from the video_meta.csv sidecar (our extractions vary per video; SignVerse is fixed 24 fps).
    # config pose_fps is fallback-only: without the sidecar timestamps drift ~2x and ~44% of captions get dropped.
    video_meta = load_video_meta(root / META_FILENAME)
    pose_cfg = lang_cfg.get("pose", {}) or {}
    fps_fallback = float(pose_cfg.get("fps", 25.0))
    pose_index = build_pose_index(
        root / "poses", fps=fps_fallback,
        width=int(pose_cfg["width"]) if pose_cfg.get("width") is not None else None,
        height=int(pose_cfg["height"]) if pose_cfg.get("height") is not None else None,
        video_meta=video_meta,
    )
    if not pose_index: raise FileNotFoundError( # empty/missing poses/ → 0 records everywhere; fail loud
        f"[loader] no pose .npy files under {root / 'poses'} for language '{language}'. "
        f"For SignVerse-2M languages (asf/bfi) run `python prepare_data.py --stage all --languages {language}` "
        f"(docs/run_real_data.md §2a); for own extractions, place per-video (T,133,3) .npy there."
    )
    missing_meta = [vid for vid in pose_index if vid not in video_meta]
    if missing_meta: print(
        f"[loader] WARNING: {len(missing_meta)}/{len(pose_index)} {language} videos missing from "
        f"{root / META_FILENAME}; falling back to pose.fps={fps_fallback} "
        f"for them — run `python -m poses {root}` (yt-dlp metadata fetch, no video download)."
    )
    subtitle_cfg = data_cfg.get("subtitles", {})
    splits = build_splits(sorted(pose_index.keys()), data_cfg.get("splits", {}))
    selected_ids = splits.get(split, []) if split else sorted(pose_index.keys())
    drop_noise = bool(subtitle_cfg.get("drop_noise_captions", True))

    # Human-caption-only splits (default: test). NLLB machine-translated references are noisy BLEU targets — scoring 
    # against them measures "agreement with NLLB", not translation quality — so drop MT-captioned videos on those 
    # splits. Provenance is video_meta.csv `caption_source` (written by `prepare_data.py --stage subs`); the excluded
    # sources default to just "mt" (raw-shard captions are kept — usually human uploads). Absent → nothing excluded.
    human_only = set(subtitle_cfg.get("human_only_splits", ["test"]) or [])
    exclude_sources = set(subtitle_cfg.get("human_only_exclude_sources", ["mt"]) or [])
    if split in human_only:
        drop_ids = {vid for vid, m in video_meta.items() if (m.get("caption_source") in exclude_sources)}
        before = len(selected_ids)
        selected_ids = [v for v in selected_ids if v not in drop_ids]
        if before != len(selected_ids): print(
            f"[loader] {language}/{split}: excluded {before - len(selected_ids)} video(s) with "
            f"{'/'.join(sorted(exclude_sources))} captions (human references only; subtitles.human_only_splits).", flush=True
        )
    records: list[VideoRecord] = []
    dropped_no_caption = 0
    # 1.0 disables the filter (no video can exceed a full share). The key is GLOBAL under `subtitles:`, which is
    # correct while every corpus targets English; a non-Latin-target corpus would need a per-language override, not
    # a global 1.0, which would switch the filter off for the English corpora too.
    max_non_latin = float(subtitle_cfg.get("max_non_latin_ratio", 1.0))
    dropped_non_latin: list[tuple[str, float]] = []
    for video_id in tqdm(selected_ids, desc=f"[loader] {language}/{split or 'all'}", unit="vid", leave=False, dynamic_ncols=True):
        subtitle_path = best_subtitle(root / "subs", video_id, subtitle_cfg)
        if subtitle_path is None:
            dropped_no_caption += 1
            continue
        captions = merge_rolling_captions(parse_vtt(subtitle_path, drop_noise=drop_noise))
        if subtitle_cfg.get("merge_sentences"):  # rebuild sentence units from display-wrapped cues (Punkt over the caption stream)
            captions = reconstruct_sentences(captions)
        min_dur = float(subtitle_cfg.get("min_duration_s", 0.2))
        max_dur = float(subtitle_cfg.get("max_duration_s", 60.0))
        # `s < duration`: sentence ONSET must land inside extracted poses, else no visible signing to anchor on. SignVerse streams 
        # end before their caption timeline (duration = pose_frames/24 underestimates the video), so late captions start past the poses; 
        # `e <= duration + 1.0` bounds only the END. Without it, the sampler builds start_s > end_s windows → load_pose_frames raises.
        dur = pose_index[video_id].duration_s
        # Some source VTTs ship genuinely OVERLAPPING cues with distinct text. merge_rolling_captions only fuses overlapping DUPLICATES, 
        # so these survive, and overlapping SentenceSpans corrupt BIO labels — a neighbour's `I` overwrites the closing `O`, and 
        # first_complete_span becomes ill-defined. Clamp each start to the previous end: it trims the disputed frames from the LATER
        # sentence (whose onset is the less certain of the two) and never invents a boundary.
        captions = _quarantine_end_straddlers(_clamp_overlaps(captions), dur)  # Dropped straddlers become trusted-O over signing
        # WRONG-LANGUAGE videos. `.en` track is not a guarantee: ASE pool carries Japanese Sign Language and Chinese-teaching content whose 
        # captions are largely Japanese/Chinese. As TRANSLATION TARGETS those are unusable — the model is asked to emit non-English from ASL 
        # — and they also corrupt any batch they land in. Judged per VIDEO on script share, never per cue: an English sentence quoting a 
        # foreign term ("and 阿曼达 in Chinese!") is real data and must survive. 2 populations separate cleanly (bilingual videos 0.39-0.48, 
        # English-with-quotes below 0.05), so the threshold sits in the empty band between.
        if max_non_latin < 1.0 and captions:
            ratio = non_latin_ratio(c[2] for c in captions)
            if ratio > max_non_latin:
                dropped_non_latin.append((video_id, ratio))
                continue
        spans = tuple(
            SentenceSpan(video_id=video_id, start_s=c[0], end_s=c[1], text=c[2], reliable=bool(c[3]) if len(c) > 3 else True)
            for c in captions if min_dur <= (c[1] - c[0]) and (len(c) > 3 or (c[1] - c[0]) <= max_dur) 
            and c[0] < dur and c[1] <= dur + 1.0 and any(ch.isalpha() for ch in c[2])
        )
        # Require >=1 RELIABLE span: an all-quarantined record contributes no anchor, target, or gold event, so
        # keeping it only loads poses nothing uses. Invariant: a record that reaches training/eval is usable.
        if any(sp.reliable for sp in spans): records.append(VideoRecord(language, video_id, pose_index[video_id], subtitle_path, spans))
    if dropped_non_latin: print(
        f"[loader] {language}/{split or 'all'}: {len(dropped_non_latin)} video(s) dropped as WRONG-LANGUAGE (>{max_non_latin:.0%} non-Latin "
        f"caption characters; e.g. " + ", ".join(f"{v} {r:.0%}" for v, r in sorted(dropped_non_latin, key=lambda x: -x[1])[:3])
        + "); subtitles.max_non_latin_ratio.", flush=True)
    if dropped_no_caption: print(
        f"[loader] {language}/{split or 'all'}: {dropped_no_caption}/{len(selected_ids)} videos dropped "
        f"(no usable caption in {root / 'subs'}).", flush=True
    )
    # Cross-split de-duplication. YouTube corpora contain RE-UPLOADS: the same talk under different video ids, so an id-based
    # split puts it in BOTH train and dev/test and the model scores by memorisation. Removal is TRAIN-side, following the
    # decontamination convention (keep the benchmark intact, purge the training copy) — deleting the eval twin instead would
    # shrink an already small eval set and bias what remains toward content unlike training.
    dedup_cfg = subtitle_cfg.get("dedup", {}) or {}
    if dedup_cfg.get("enabled") and split == "train" and records:
        eval_ids = {v for s in ("dev", "test") for v in splits.get(s, [])}
        caps = _split_caption_sets(root, [r.video_id for r in records] + sorted(eval_ids), subtitle_cfg, drop_noise)
        pairs = _duplicate_pairs(caps, dedup_cfg)
        drop: dict[str, str] = {}
        for a, b, frac in pairs:
            if (a in eval_ids) != (b in eval_ids):
                drop.setdefault(b if a in eval_ids else a, f"{frac:.0%} of {b if a in eval_ids else a}")
        if drop:
            records = [r for r in records if r.video_id not in drop]
            print(f"[loader] {language}/train: de-duplicated {len(drop)} train video(s) whose content also appears in dev/test "
                  f"({', '.join(sorted(drop)[:5])}{'...' if len(drop) > 5 else ''}); subtitles.dedup.", flush=True)
    return records, splits


class StreamingWindowDataset(Dataset):
    """On-the-fly Stage 2 window dataset. `__getitem__` samples from the training distribution rather than indexing
    a fixed window table, keeping the stochastic sampler inside PyTorch/HF Trainer's map-style interface."""
    def __init__(
        self, records: list[VideoRecord], slt_cfg: dict[str, Any], inference_cfg: dict[str, Any], steps_per_epoch: int | None = None, 
        include_full_evidence: bool = True, deterministic: bool = False, pose_augment_cfg: dict | None = None, records_for_epoch=None,
    ):
        # Optional `epoch -> records` provider: a multilingual pool re-draws its balanced sub-sample each epoch so
        # coverage of the sub-sampled corpora rotates (see set_epoch). None = a fixed record list, as before.
        self._records_for_epoch = records_for_epoch
        self._slt_cfg, self._inference_cfg, self._pose_augment_cfg = slt_cfg, inference_cfg, pose_augment_cfg
        self.records = records
        self.records_by_id = {record.video_id: record for record in records}
        # Lazy, NOT module-level: train.sampler imports VideoRecord from this module, so a top-level import here is a data.loader <-> 
        # train.sampler cycle that breaks every entry point at import time. WindowSampler is only ever used inside methods.
        from train.sampler import WindowSampler
        self.sampler = WindowSampler.from_slt_config(records, slt_cfg, inference_cfg, pose_augment_cfg=pose_augment_cfg)
        self.steps_per_epoch = int(steps_per_epoch or max(len(self.sampler.anchors), 1))
        # Epoch cursor for CAPPED epochs (steps_per_epoch < anchor count — e.g. multilingual pool, where a full pass is several hours): 
        # successive epochs walk successive anchor slices, so every anchor is still visited every ceil(anchors/steps) epochs. Without 
        # offset, DataLoader indices restart at 0 each epoch and anchors[index % N] would revisit the SAME first slice forever — a silent 
        # fixed-prefix training set. No-op when uncapped: (index + e*N) % N == index % N. Dev loaders are deterministic and never offset.
        self._epoch = 0
        self.include_full_evidence = bool(include_full_evidence)
        # Eval loaders set deterministic=True: an index then always yields the SAME anchor under a per-index rng, so early-stopping 
        # monitor scores a fixed dev set each epoch instead of a fresh draw (else "best epoch" is partly a lottery).
        self.deterministic = bool(deterministic)
        self.seed = int(slt_cfg.get("seed", 42))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        if self.deterministic: return
        self._epoch = int(epoch)
        # Multilingual pool: the balanced sub-sample ROTATES so a large corpus reduced to k of n videos is covered
        # in full every ceil(n/k) epochs. Without this the run trains on one fixed epoch-0 slice forever and the
        # rest of the dominant corpus is never loaded at all. Rebuilding is cheap (records hold no pose data).
        if getattr(self, "_records_for_epoch", None) is None: return
        records = self._records_for_epoch(self._epoch)
        if not records: return
        self.records = records
        self.records_by_id = {r.video_id: r for r in records}
        from train.sampler import WindowSampler   # lazy: see __init__ (import cycle)
        self.sampler = WindowSampler.from_slt_config(records, self._slt_cfg, self._inference_cfg, pose_augment_cfg=self._pose_augment_cfg,)
        # steps_per_epoch is deliberately NOT recomputed: it is the epoch's fixed compute BUDGET. Rotation changes the anchor COUNT by a few 
        # percent (different videos hold different numbers of sentences), and `effective_index` shifts the cursor by epoch, so the slice each 
        # epoch walks moves and nothing is systematically skipped. Exact once-per-epoch coverage holds only when the anchor count is stable.

    def effective_index(self, index: int) -> int:
        # The index actually handed to the sampler. Uncapped epochs shift by epoch so a capped run does not replay
        # the same slice forever; the length pre-pass must apply the SAME shift or it predicts the wrong window.
        return index if self.deterministic else index + self._epoch * self.steps_per_epoch

    def _sample_item(self, index: int) -> dict:
        index = self.effective_index(index)
        sample = self.sampler.sample(index)   # anchor = anchors[index % N]
        item = self.sampler.to_dict(sample)
        if self.include_full_evidence and sample.full_evidence_spec is not None:
            rec = self.records_by_id[sample.full_evidence_spec.video_id]
            full = self.sampler.materialize(rec, sample.full_evidence_spec)
            item["full_evidence"] = self.sampler.to_dict(full)
        else: item["full_evidence"] = None
        return item

    def __getitem__(self, index: int) -> dict:
        if not self.deterministic: return self._sample_item(index)
        # Per-index rng makes mode/jitter reproducible. fps_aug off — a TRAIN augmentation (Moryossef 2026 gates it
        # on split==TRAIN, evaluates at native fps); leaving it on scored the monitor on 15–30fps resampled windows
        # the head never deploys under.
        rng = np.random.default_rng(self.seed * 100_003 + int(index))
        saved = (self.sampler.rng, self.sampler.fps_aug_enabled)
        self.sampler.rng = rng
        self.sampler.fps_aug_enabled = False
        try: return self._sample_item(index)
        finally: (self.sampler.rng, self.sampler.fps_aug_enabled) = saved


def _streaming_worker_init(worker_id: int) -> None:
    """Reseed each worker's WindowSampler so parallel workers don't replay identical mode/jitter streams.

    Forked/spawned workers inherit the sampler's Generator state IDENTICALLY (PyTorch's per-worker seeding never
    touches a Generator stored on the dataset); `info.seed` is unique per worker (base_seed + worker_id). Anchors
    stay index-driven, so coverage is untouched — only mode/jitter/fps/pose-aug draws are decorrelated.
    """
    info = get_worker_info()
    if info is None: return
    sampler = getattr(info.dataset, "sampler", None)
    if sampler is not None and hasattr(sampler, "configure_worker"):
        sampler.configure_worker(int(info.seed) % (2**32))


class LengthBucketSampler(torch.utils.data.Sampler):
    """Batch indices so each batch holds windows of SIMILAR length.

    Step cost is linear in the total frames a batch computes, and a batch is padded to its longest window — so with random batching the 
    median window (~146 frames) is computed at the batch maximum (~433).

    Coverage is untouched: this is a PERMUTATION of the same index set, and the anchor is `anchors[index % N]`, so every anchor is still 
    realised exactly once per epoch. Lengths come from `WindowSampler.spec_frames`, pose-free index-seeded pre-pass costing ~0.02ms/index.

    Shuffling happens WITHIN a pool: indices are permuted, then sorted by length inside each `pool` slice. A global sort would fix batch 
    composition for every epoch and correlate each batch with one length regime; pooling keeps batches fresh while still grouping.
    """
    def __init__(self, dataset, batch_size: int, pool_batches: int = 32, seed: int = 0, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.pool = max(1, int(pool_batches)) * self.batch_size
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)      # re-permute each epoch; mirrors DistributedSampler's contract

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        order = np.random.default_rng(self.seed + self.epoch).permutation(n)
        sampler = self.dataset.sampler
        batches: list[list[int]] = []
        for i in range(0, n, self.pool):
            chunk = order[i:i + self.pool]
            eff = getattr(self.dataset, "effective_index", lambda x: x)
            lens = np.fromiter((sampler.spec_frames(eff(int(j))) for j in chunk), dtype=np.int64, count=len(chunk))
            chunk = chunk[np.argsort(lens, kind="stable")]
            for b in range(0, len(chunk), self.batch_size):
                batch = [int(x) for x in chunk[b:b + self.batch_size]]
                if len(batch) == self.batch_size or not self.drop_last: batches.append(batch)
        # Batches themselves are shuffled: consecutive batches would otherwise march monotonically through the
        # length range, correlating batch order with window length within every epoch.
        np.random.default_rng(self.seed + 7919 + self.epoch).shuffle(batches)
        return iter(batches)


def streaming_loader(
    dataset: StreamingWindowDataset, batch_size: int, collate_fn, num_workers: int = 0,
    bucket_by_length: bool = False, bucket_seed: int = 0
) -> DataLoader:
    """The ONE DataLoader constructor for StreamingWindowDataset (both trainers route through here).

    num_workers is a plain throughput knob: the ANCHOR is a deterministic function of the global sample index
    (WindowSampler.sample → anchors[index % N]), so every anchor is realized exactly once per epoch however indices
    are partitioned — no duplication, no lost coverage. `_streaming_worker_init` decorrelates the per-window random
    stream forked workers would otherwise share; dev datasets seed from the index and need neither.
    """
    # A dataset whose __getitem__ depends on the epoch cannot use persistent workers.
    epoch_stateful = hasattr(dataset, "set_epoch") and not bool(getattr(dataset, "deterministic", False))
    persistent = num_workers > 0 and not epoch_stateful
    sampler = None
    if dist.is_distributed(): sampler = DistributedSampler(dataset, num_replicas=dist.world_size(), rank=dist.rank(), shuffle=False)
    # Length bucketing: same index set, grouped so a batch is not padded to a much longer neighbour. Incompatible
    # with a DistributedSampler (both decide the index order), so it is single-process only; multi-GPU already
    # splits the batch and gets its speed there.
    if bucket_by_length and sampler is None and hasattr(dataset, "sampler"): return DataLoader(
        dataset, batch_sampler=LengthBucketSampler(dataset, int(batch_size), seed=int(bucket_seed)), num_workers=int(num_workers), 
        persistent_workers=persistent, collate_fn=collate_fn, worker_init_fn=_streaming_worker_init if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(), prefetch_factor=4 if num_workers > 0 else None,
    )
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, sampler=sampler, num_workers=int(num_workers),
        persistent_workers=persistent, collate_fn=collate_fn, worker_init_fn=_streaming_worker_init if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(), prefetch_factor=4 if num_workers > 0 else None,
    )
