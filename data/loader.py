from __future__ import annotations
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, replace
from typing import Any, Iterable
from pathlib import Path
from tqdm import tqdm

import re, os, sys, io, random, csv, html, json, unicodedata, zlib, hashlib, inspect, bisect, itertools, pickle, contextlib
import importlib.metadata
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler, SequentialSampler
from data.windowing import SentenceSpan, ANNOTATION_PROTOCOL, TRUSTED_GAP_S, untrusted_o_intervals

from poses import PoseIndex, build_pose_index
from poses.pose_io import META_FILENAME, base_video_id, grid_runs, load_video_meta
from train import distributed as dist
from utils import cfg_get, lambda_min_s, load_yaml, seconds_to_frames

TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{3})"
)
WORD_TIMING_RE = re.compile(r"<\d{1,2}:\d{2}:\d{2}[.,]\d{3}>")
_ASR_WORD_TIMING_RE = re.compile(r"<\d\d:\d\d:\d\d\.\d\d\d>")
_SENTENCE_FINAL_RE = re.compile(r'[.!?][\"\')\]]*\s*$')  # terminal punctuation, allowing closing quotes and brackets
WORD_RE = re.compile(r"[A-Za-z][\w']*")

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
TAG_RE = re.compile(r"<[^>]+>")
# Speaker identifier: ONE capitalised word (optionally two, e.g. "MRS SMITH:") then a colon, at the cue start. Bounded to a single token so
# a genuine clause like "One thing: ..." keeps its text, and capitalised so a lowercase word before a colon ("today: ...", "one: ...", in
# lowercase captions too) keeps its text — that costs recall on rare speaker labels but never deletes signed content.
SPEAKER_PREFIX_RE = re.compile(r"^[A-Z][A-Z\s_-]{1,30}:\s*")
SPEAKER_ID_RE = re.compile(r"^[A-Z][\w'\-]{0,20}(?:\s+[A-Z][\w'\-]{0,20})?:\s+")
NOISE_WORD_RE = re.compile(r"[a-z]+")
NOISE_CAPTION_WORDS = {
    "applause", "background", "foreign", "gentle", "inaudible", "laugh", "laughs",
    "laughter", "music", "piano", "silence", "silent",
}
# Most frequent closed-class English words. Closed class since it can't grow with the topic: a caption about
# any subject still needs these to be English prose, and no other Latin-script language shares more than a few.
ENGLISH_FUNCTION_WORDS = frozenset("""
the a an and or but if then than that this these those of to in on at for with from by as is are was were be been being am do does did have 
has had will would can could should may might must not no nor so such it its he she they them his her their we us our you your i me my there 
here what when where who whom which how why all any both each few more most other some only own same too very
""".split())
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
_LANG_RECORDS_CACHE: dict[tuple, list[VideoRecord]] = {}
RUN_COLUMNS = ("empty_runs",)   # video_meta.csv frame runs the loader cuts out (split_clean_segments)
# The quarantine kinds (quarantine_units): per kind, the video_meta.csv frame runs that it reads, each with its data.yaml poses 
# share key. A unit is quarantined when, for every column of the kind, at least that share of its frames lies inside the column's runs.
QUARANTINE_RULES = {
    "handless": {"handless_runs": "handless_unit_share"},
    "switch_doubt": {"switch_runs": "switch_unit_share", "default_signing_runs": "switch_doubt_share"},
}
QUARANTINE_COLUMNS = tuple(key for rule in QUARANTINE_RULES.values() for key in rule)

# Measured on asf/bfi/ase train cues: merged prose signs at 0.70 w/s (asf p1) to 2.2 w/s (p50) with 6-11 words per display cue and 4-22% 
# capitalised tokens; fingerspelling and vocabulary lists sign at ~0.1 w/s with 1.3-2.2 words per cue and 66-78% capitalised tokens. 
PROSE_MIN_WORDS_PER_S = 0.3
GLOSS_MAX_WORDS_PER_CUE = 2.5
GLOSS_MIN_CAPITALISED_SHARE = 0.6

@dataclass(frozen=True)
class VideoRecord:
    language: str
    video_id: str
    pose: PoseIndex
    sentences: tuple[SentenceSpan, ...]
    # UNK stretches of a clean segment in segment time: the source video's uncaptioned gaps longer than TRUSTED_GAP_S, clipped.
    # None on an uncut video, whose labeller finds them itself, unless an unreliable span (mark_unreliable) lies in it: then the gaps of
    # its kept units. Neither a cut nor a mark may shorten a long source gap into a trusted O gap.
    untrusted_s: tuple[tuple[float, float], ...] | None = None

def _is_pronoun_i(word: str) -> bool:
    return word == "I" or word.startswith("I'")

def _fold_rule_id() -> str: # The counting rule's own identity, so an edited rule invalidates every lexicon built under the old one.
    return hashlib.sha256(inspect.getsource(fold_lexicon).encode()).hexdigest()

def _lexicon_key(sources: list[tuple[Path, list[str], dict]]) -> str:
    # Identity of a built lexicon: the caption BYTES it reads, each corpus's selection rules, the counting rule itself, the Punkt
    # model that splits the cues into sentences, and the loader code (RECORDS_CODE) that cleans the cue text it counts (a changed
    # speaker-label or noise rule changes the words), so a train load builds it again after any change of that code.
    digest = hashlib.sha256(json.dumps([ANNOTATION_PROTOCOL, _fold_rule_id(), _punkt_identity()], sort_keys=True, default=str).encode())
    repo = Path(__file__).resolve().parent.parent
    for path in RECORDS_CODE: digest.update((repo / path).read_bytes())
    for subs, ids, subtitle_cfg in sources:
        digest.update(json.dumps(subtitle_cfg, sort_keys=True, default=str).encode())
        want = set(ids)
        # 1 directory glob, not 1 per video: `<vid>.*.vtt` per id is ~2850x slower over a 10k-video corpus.
        # `<vid>.<target>.vtt` -> vid: `Path.stem` strips 1 suffix only, and video ids never contain a dot.
        for path in sorted(Path(subs).glob("*.vtt")):
            if base_video_id(path.name.split(".", 1)[0]) in want: 
                digest.update(f"{path.name}\t".encode() + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()

def _train_captions(data_cfg: dict, language: str) -> tuple[Path, list[str]]:
    """(subs root, train ids that HAVE a caption file) without loading records — the id universe `build_pose_index` keys on.

    Filtered by caption presence because the lexicon reads captions: a corpus whose poses are unpacked but whose subs are not would otherwise 
    pass the presence check in `case_lexicon` and contribute nothing, silently changing the rendering it was required for.
    """
    root = Path(data_cfg["languages"][language]["root"])
    ids = sorted({base_video_id(p) for p in (root / "poses").glob("*.npy")})
    if not ids: return root / "subs", []
    captioned = {base_video_id(p.name.split(".", 1)[0]) for p in (root / "subs").glob("*.vtt")}   # `<vid>.<target>.vtt` -> vid
    return root / "subs", [v for v in build_splits(ids, corpus_config(data_cfg, language).get("splits", {})).get("train", []) if v in captioned]

def timestamp_to_seconds(value: str) -> float:
    value = value.replace(",", ".")
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

def fold_lexicon(sources: list[tuple[Path, list[str], dict]]) -> frozenset[str]:
    """Words safe to lowercase at a comma-joined sentence start, from the TRAIN splits' raw cues.

    Capital at sentence start is positional; after the period becomes comma it is noise decoder must learn ("..., Consequently, this ..."). 
    But a capital can also be lexical (a name, "Deaf" as identity, "BSL"), and only the corpus can tell: a word is foldable when it occurs 
    in lowercase more often than it occurs capitalised INSIDE a sentence (Punkt over each raw cue, so sentence-initial capitals are not 
    counted as name evidence). Built on train only, applied to every split, so dev/test references never shape the rule and an unattested
    word simply keeps its case. `sources` (subs folder, train ids, that corpus's subtitles block) are the corpora of one data.yaml casing
    pool — see `case_lexicon`, which owns the pool and the released artifact; this builds and nothing else.
    """
    tok = _punkt_tokenizer()
    lower: dict[str, int] = {}; cap_inside: dict[str, int] = {}

    for subs, ids, subtitle_cfg in sources:
        drop_noise = bool(subtitle_cfg.get("drop_noise_captions", True))
        for vid in ids:
            path = best_subtitle(subs, vid, subtitle_cfg)
            if path is None: continue

            for _, _, text in merge_rolling_captions(parse_vtt(path, drop_noise=drop_noise)):
                sentences = [text[a:b] for a, b in tok.span_tokenize(text)] if tok is not None else [text]
                for sentence in sentences:
                    for i, word in enumerate(WORD_RE.findall(sentence)):
                        if word[0].islower(): lower[word.lower()] = lower.get(word.lower(), 0) + 1
                        elif i > 0: cap_inside[word.lower()] = cap_inside.get(word.lower(), 0) + 1

    return frozenset(w for w, n in lower.items() if n > cap_inside.get(w, 0))


def case_lexicon_entry(data_cfg: dict, language: str) -> tuple[Path, list[str]] | None:
    # The casing file of corpus `language` (`subtitles.case_lexicon` of its view, which data.yaml names per dataset) and the corpora
    # whose views name the same file: its pool, whose train captions build it. None for a corpus that names no file or that data.yaml
    # does not name (only a load with subtitles.merge_sentences needs one: case_lexicon refuses it there).
    langs = data_cfg["languages"]
    if language not in langs: return None
    named = {corpus: corpus_config(data_cfg, corpus)["subtitles"].get("case_lexicon") for corpus in langs}
    if not named[language]: return None
    artifact = Path(named[language])
    pool = sorted(corpus for corpus, file in named.items() if file and Path(file) == artifact)
    target = lambda corpus: str(langs[corpus].get("target_lang", "en_XX"))
    if mixed := [corpus for corpus in pool if target(corpus) != target(language)]: raise SystemExit(
        f"[loader] the casing file {artifact} is named by corpora of another target language than {language}'s ({', '.join(mixed)}): "
        f"1 file holds the word list of 1 target language, so name another file for them."
    )
    return artifact, pool


def case_lexicon(data_cfg: dict, language: str, split: str | None = None) -> frozenset[str]:
    """The fold lexicon of `language`: the word list of its casing file (case_lexicon_entry), built from the train captions of every
    corpus that names the same file.

    data.yaml names 1 file per dataset: 1 caption source and 1 target language. Inside a source, pooling renders better where the corpora
    disagree (asf's train captions never lowercase "consequently", ase's do), and it keeps the small corpora's identity capitals ("Deaf",
    "God", "Bible"), which a corpus-sized rule folds away. Across sources it does not: a word's ordinary case depends on the source as well
    as on the language ("Bush" in US news is "bush" in British broadcast subtitles, which also write "OK" where YouTube writes "ok"), and
    the largest source decides the shared list. The pool is the corpora that name the file, not "every corpus of this config", so a
    config that adds or leaves out a corpus changes no other source's text. A train load (`prepare_yt25.py --stage dedup` makes one) builds the file again when the pooled
    train captions or the loader code (`_lexicon_key`) changed; a dev or test load reads only the file, because building it reads train captions; a machine that lacks a
    pooled corpus reads the stored file.
    """
    langs = data_cfg["languages"]
    if (entry := case_lexicon_entry(data_cfg, language)) is None: raise SystemExit(
        f"[loader] data.yaml names no casing file for the corpus {language} (subtitles.case_lexicon, named per dataset): name the file of "
        f"its caption source, whose word list folds its sentence-initial capitals; the corpora that name 1 file pool their train captions."
    )
    artifact, pool = entry
    try: stored = json.loads(artifact.read_text(encoding="utf-8")) if artifact.exists() else {}
    except ValueError: stored = {}
    if split in ("dev", "test"):
        if stored.get("pool") == pool and stored.get("rule") == _fold_rule_id(): return frozenset(stored["words"])
        raise SystemExit(
            f"[loader] {artifact} is missing, or it was built for another pool or counting rule; the {split} loader only reads it, "
            f"because building it reads the train captions. Run `python prepare_yt25.py --stage dedup --languages {language}` (its "
            f"train load builds it), or copy the file from the machine that built it."
        )

    sources, absent = [], []
    for other in pool:
        subs, ids = _train_captions(data_cfg, other) if other in langs else (None, [])
        if ids: sources.append((subs, ids, corpus_config(data_cfg, other).get("subtitles", {})))
        else: absent.append(other)

    if absent:
        if stored.get("pool") == pool and stored.get("rule") == _fold_rule_id(): return frozenset(stored["words"])
        raise FileNotFoundError(
            f"[loader] the casing file {artifact} pools the train captions of {', '.join(pool)} (the corpora that name it); this "
            f"machine has none for {', '.join(absent)}. Copy {artifact} from the machine that holds the whole pool, or fetch the corpus."
        )
    key = _lexicon_key(sources)
    if stored.get("key") == key: return frozenset(stored["words"])
    lexicon = fold_lexicon(sources)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    
    # Atomic: every rank of a distributed run builds this, and a half-written artifact is a rebuild for everyone after.
    tmp = artifact.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"key": key, "pool": pool, "rule": _fold_rule_id(), "words": sorted(lexicon)}) + "\n", encoding="utf-8")
    os.replace(tmp, artifact)
    return lexicon


def _is_prose(cues: list[tuple], joined: str, duration_s: float) -> bool:
    # Is a multi-cue chain wrapped prose, so that Punkt finding no boundary across its junctions is evidence?
    words = WORD_RE.findall(joined)
    if not words or duration_s <= 0: return False
    if len(words) / duration_s < PROSE_MIN_WORDS_PER_S: return False
    words_per_cue = sum(len(WORD_RE.findall(text)) for _, _, text in cues) / len(cues)
    capitalised = sum(1 for w in words if w[0].isupper()) / len(words)
    return not (words_per_cue <= GLOSS_MAX_WORDS_PER_CUE and capitalised >= GLOSS_MIN_CAPITALISED_SHARE)


def _fold_sentence_start(part: str, lexicon: frozenset[str] | None) -> str:
    # Lowercase 1st word of an interior sentence when the corpus says its capital was positional.
    if not lexicon: return part
    m = WORD_RE.search(part)
    if m is None: return part
    word = m.group(0)
    # An ALL-CAPS token carries no positional capital to remove: it is a gloss or emphasis, and folding only its first
    # letter renders "BYE" as "bYE". Also skip I / I'm and fingerspelled single letters.
    if not word[0].isupper() or len(word) == 1 or word.isupper() or _is_pronoun_i(word): return part
    nxt = WORD_RE.search(part, m.end())
    if nxt and nxt.group(0)[0].isupper() and not _is_pronoun_i(nxt.group(0)): return part   # "New Zealand", "Deaf Youth"
    if word.lower() not in lexicon: return part
    return part[:m.start()] + word[0].lower() + part[m.start() + 1:]


def annotation_fingerprint(records) -> str: # Identify the actual unit texts, times, reliability and frame geometry used by a dataset.
    digest = hashlib.sha256(json.dumps([ANNOTATION_PROTOCOL, TRUSTED_GAP_S]).encode())
    for r in sorted(records, key=lambda r: (r.language, r.video_id)):
        row = [
            r.language, r.video_id, float(r.pose.fps), float(r.pose.duration_s), 
            [(s.start_s, s.end_s, s.text, s.reliable) for s in r.sentences]
        ]
        digest.update(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode())
    return ANNOTATION_PROTOCOL + ':' + digest.hexdigest()


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
    # A quote after final punctuation closes a quotation; an apostrophe inside a word is lexical.
    # Preserve both leading elisions ('cause) and trailing possessives (students').
    if len(raw) > 1 and raw[0] == raw[-1] and (raw[0] == '"' or (raw[0] == "'" and raw[-2] in '.!?')): raw = raw[1:-1].strip()
    return re.sub(r"(?<=[.!?])[\"']+$", "", raw)


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
    # Pretrained English Punkt: segments caption stream into sentences, joining fragments by the ABSENCE of a boundary and 
    # splitting multi-sentence cues, with an abbreviation model. Loaded once; grouping refuses to proceed if unavailable.
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
            print("[loader] WARNING: Punkt unavailable; caption-unit grouping requires its tokenizer data.", flush=True)
            _PUNKT = False
    return _PUNKT or None


def _punkt_identity() -> str | None:
    # The Punkt model groups the units and splits the casing counts, and an `nltk.download` can replace it with no change of the NLTK
    # version: its parameters, sorted (the order of a set depends on the hash seed of the process). None without Punkt.
    tok = _punkt_tokenizer()
    if tok is None: return None
    p = tok._params
    return hashlib.sha256(json.dumps([sorted(p.abbrev_types), sorted(map(list, p.collocations)), sorted(p.sent_starters),
                                      sorted(p.ortho_context.items())]).encode()).hexdigest()


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

def english_function_word_ratio(texts: Iterable[str]) -> tuple[int, float]:
    """(word count, share of words that are closed-class English) for 1 video's captions.

    `non_latin_ratio` tests SCRIPT, so Spanish and French captions filed as `.en.vtt` pass it untouched. Content
    words cannot separate those from English, because a name or a loanword looks the same in both; function words
    can, because every language builds its own from a closed set that does not transfer.

    Return the word count too: the caller must not judge a vocabulary list this way. A gloss track carries content
    words alone ("RUBBISH / MESS / CLEAN"), so it scores near zero in any language.
    """
    words = WORD_RE.findall(" ".join(texts).lower())
    if not words: return 0, 0.0
    return len(words), sum(word in ENGLISH_FUNCTION_WORDS for word in words) / len(words)


def _clamp_overlaps(captions: list[tuple[float, float, str]], max_overlap_s: float) -> tuple[list[tuple], int, int, int, int]:
    """Force a strictly non-overlapping, time-ordered span sequence. Returns (spans, clamped, swallowed, containers, disputed).

    Applied to the FINAL caption stream (after rolling-duplicate merging and sentence reconstruction), so it is the
    1 place that guarantees the invariant every downstream consumer assumes. Each start clamps to the previous end. When the
    overlap is longer than `max_overlap_s` (δ, inference.yaml boundary_stability.delta_s), no boundary between the 2 units is
    known to δ, so the clamped unit and every kept unit that it overlapped are quarantined (reliable=False): their frames
    become UNK and neither is a target or a reference; a shorter overlap only moves the later start, by at most δ. A span
    left fully inside its predecessors after clamping is dropped: it carries no exclusive frames, so it can never be selected
    or scored. Its TEXT leaves the reference with it, so every kept span that holds its frames is quarantined: that span's
    text misses words that are signed inside it, so it is neither a correct target nor a correct reference.

    The first 2 counts are per UNIT and disjoint — 1 unit is either moved or deleted — so neither can go negative and their
    sum is the number of units an overlap touched; `containers` counts the kept units quarantined for a deleted one, and
    `disputed` the units quarantined for an overlap above δ. A count derived instead from ADJACENT-PAIR overlaps is not on
    this basis: 1 malformed cue containing 5 short ones is a single adjacent pair and five deletions.
    """
    out: list = []
    clamped = swallowed = containers = disputed = 0
    prev_end = float("-inf")
    reliable = lambda span: len(span) < 4 or bool(span[3])
    quarantine = lambda span: (*span[:3], False, *span[4:])

    def quarantine_overlapped(start: float, end: float) -> int:
        # Quarantine the kept spans that overlap (start, end) (out is ordered and disjoint); the number newly quarantined.
        n = 0
        for i in range(len(out) - 1, -1, -1):
            if out[i][1] <= start: break
            if out[i][0] < end and reliable(out[i]): out[i] = quarantine(out[i]); n += 1
        return n

    # Tuples may carry a 4th `reliable` field (unsupported coverage) — pass any extra fields through untouched.
    for c in sorted(captions, key=lambda x: (x[0], x[1])):
        s, e = max(c[0], prev_end), c[1]
        if e <= s:
            swallowed += 1  # wholly swallowed by the previous spans; its TEXT leaves the reference with it
            containers += quarantine_overlapped(c[0], e)
            continue
        clamped += s > c[0]
        span = (s, e, *c[2:])
        if c[0] < prev_end - float(max_overlap_s):   # an overlap above δ: no boundary is known to δ
            disputed += quarantine_overlapped(c[0], prev_end) + reliable(span)
            span = quarantine(span)
        out.append(span); prev_end = e
    return out, clamped, swallowed, containers, disputed


def _quarantine_end_straddlers(captions: list[tuple], duration_s: float, slack_s: float = 1.0) -> list[tuple]:
    """Captions that START inside pose stream but END well past it become QUARANTINED spans over visible frames, instead of being dropped.

    Streams end before the caption timeline (duration = frames/fps underestimates the video), so this straddle is systematic, not an edge 
    case. Dropping caption relabels its visible frames as uncaptioned — and a leftover tail gap of trusted_gap_s or less is then supervised 
    as trusted `O` over frames the caption says are signing. Clipping to `duration_s` as a RELIABLE span would instead mint an end boundary 
    no cue marks, with text for signing partly outside the poses. Marking the coverage unsupported (reliable=False) is the one honest option: 
    frames UNK, never an anchor, reference, or Mode-4 gap. Ends within `slack_s` of the stream end are left alone — the existing span filter 
    tolerates them, and their label error is below the timestamp noise floor.
    """
    return [(c[0], float(duration_s), c[2], False) if (c[0] < duration_s and c[1] > duration_s + float(slack_s)) else c for c in captions]


def split_clean_segments(rec: VideoRecord, runs, min_span_s: float) -> list[VideoRecord]:
    """The clean segments of `rec` between pose-invalid frame `runs` (half-open .npy frames [a, b), any order, may overlap).

    Runs closer than Λ_min (`min_span_s`) merge, and a run within Λ_min of a video end reaches it, so no segment is shorter than
    Λ_min (in frames at the record's fps). No run: `[rec]`, unchanged. Otherwise segment [a, b) is the record `f"{vid}@{a}"` whose pose is the view `start_frame=a, 
    num_frames=b - a` and whose units are in segment time (source time - a/fps). A unit wholly inside a run is gone. A unit that crosses a 
    segment edge stays in each segment it touches, QUARANTINED, with its true times (so start < 0 or end > the segment duration). A unit 
    may end past the LAST segment by the loader's pose-end tolerance.
    """
    n, fps = rec.pose.total_frames, float(rec.pose.fps)
    floor = seconds_to_frames(min_span_s, fps)
    cuts: list[list[int]] = []
    for a, b in sorted(runs):
        if cuts and a - cuts[-1][1] < floor: cuts[-1][1] = max(cuts[-1][1], b)
        else: cuts.append([a, b])

    if not cuts: return [rec]
    if cuts[0][0] < floor: cuts[0][0] = 0
    if n - cuts[-1][1] < floor: cuts[-1][1] = n
    edges = [0, *(x for cut in cuts for x in cut), n]
    untrusted = rec.untrusted_s if rec.untrusted_s is not None else untrusted_o_intervals(rec.sentences, rec.pose.duration_s, TRUSTED_GAP_S)
    out = []

    for a, b in zip(edges[::2], edges[1::2]):
        if b <= a: continue
        lo, hi, sid = a / fps, b / fps, f"{rec.video_id}@{a}"
        spans = tuple(replace(
            s, video_id=sid, start_s=s.start_s - lo, end_s=s.end_s - lo, 
            reliable=s.reliable and lo <= s.start_s and (b == n or s.end_s <= hi)
        ) for s in rec.sentences if s.end_s > lo and s.start_s < hi)
        gaps = tuple((max(x, lo) - lo, min(y, hi) - lo) for x, y in untrusted if x < hi and y > lo)
        out.append(replace(rec, video_id=sid, pose=replace(rec.pose, start_frame=a, num_frames=b - a), sentences=spans, untrusted_s=gaps))
    return out


def quarantine_units(rec: VideoRecord, rule) -> VideoRecord:
    """`rec` with every reliable unit quarantined (reliable=False) that has, for each (runs, share) pair of `rule`, at least `share` of
    its frames inside `runs`.

    `rule` holds the columns of one QUARANTINE_RULES kind of video_meta.csv (half-open .npy frames [a, b) on the timeline of `rec.pose`),
    each with its share: handless_runs, where the kept body shows no hand; or switch_runs, where the kept body is the signer switch's
    body and not the converter's default body, with default_signing_runs, where that default body signs. A unit covers frames
    [floor(start * fps), ceil(end * fps)), clipped to the pose stream. No unit time moves and no frame is cut, so the signing transitions
    into and out of the unit stay in the stream.
    """
    n, fps = rec.pose.total_frames, float(rec.pose.fps)
    marked = []
    for runs, share in rule:
        frames = np.zeros(n, dtype=bool)
        for a, b in runs: frames[a:b] = True
        marked.append((frames, share))

    def inside(s: SentenceSpan) -> bool:
        a, b = max(0, int(np.floor(s.start_s * fps))), min(n, int(np.ceil(s.end_s * fps)))
        return b > a and all(float(frames[a:b].mean()) >= share for frames, share in marked)

    return replace(rec, sentences=tuple(replace(s, reliable=False) if s.reliable and inside(s) else s for s in rec.sentences))


def mark_unreliable(spans: tuple[SentenceSpan, ...], unreliable, video_id: str, duration_s: float) -> tuple[tuple[SentenceSpan, ...], int]:
    """`spans` with the corpus's unreliable spans (video_meta.csv unreliable_spans, [start_s, end_s] of the source video), and the number
    of reliable units that they quarantine. Every unit that overlaps a span is quarantined, and each span becomes a quarantined unit
    with no text, so its frames are UNK (not O), it is never a target, an anchor or gold, and a prediction mostly inside it is not
    scored. Units that no span touches keep their times and reliability. None or [] (no annotation, or none found) changes nothing."""
    marks = [(max(0.0, float(a)), min(float(duration_s), float(b))) for a, b in unreliable or []]
    marks = [(a, b) for a, b in marks if b > a]
    if not marks: return spans, 0
    # VTT and sidecar times are whole milliseconds, so an overlap under half of one is float noise, not an overlap: "00:04:55.160"
    # parses to 295.15999999999997, below the sidecar's 295.16, and a unit that only touches a span would be quarantined.
    hit = [s.reliable and any(s.start_s < b - 5e-4 and s.end_s > a + 5e-4 for a, b in marks) for s in spans]
    out = [replace(s, reliable=False) if h else s for s, h in zip(spans, hit)]
    out += [SentenceSpan(video_id=video_id, start_s=a, end_s=b, text="", reliable=False) for a, b in marks]
    return tuple(sorted(out, key=lambda s: (s.start_s, s.end_s))), sum(hit)


_STOP_END_RE = re.compile(r'(?<!\.)[.!?]["\')\]]*$')   # a cue that ends with 1 sentence-final mark (closing quotes and brackets allowed)
_ELLIPSIS_END_RE = re.compile(r'\.{2,}["\')\]]*$')     # a cue that ends with an ellipsis ("..", "...")
UNRELIABLE_MARK = "unreliable_span"   # 5th field of a caption tuple that is a mark of video_meta.csv unreliable_spans, not a cue
# A title that stands before a name ends no sentence ("my colleague, Dr." | "David Evans", "a teacher in St." | "Patrick's"). Only
# titles: the other abbreviations of Punkt's list also end sentences ("U.S.", "Inc.", "OK.", "p.m."), and there the cue end is right.
_TITLE_END_RE = re.compile(r"(?<![A-Za-z])(?:dr|mr|mrs|ms|st|prof|rev|sen|gen|gov|lt|col|capt|sgt|mt)\.$", re.IGNORECASE)


def _may_cross(cue: str, nxt: str, cased: bool) -> bool:
    """May 1 linguistic sentence run from `cue` into the next cue `nxt`? Punkt reads the capital of the next word, so it does not
    split before a lowercase word and it reads "OK." and an ellipsis as no sentence end; a cue track adds what Punkt cannot see,
    the cue end. After a cue that ends with . ! or ? a sentence continues only into a lowercase word, and only in a `cased` video
    (in a video whose captions carry no capitals, a lowercase word says nothing: Auslan-Daily), except after a title (Dr., St.).
    After an ellipsis it continues into a lowercase word or a cue that starts with "..". Any other junction is Punkt's to decide."""
    cue, nxt = cue.rstrip(), nxt.lstrip()
    letter = re.search(r"[^\W\d_]", nxt)
    lower = bool(letter) and letter.group(0).islower()
    if _TITLE_END_RE.search(cue): return True
    if _ELLIPSIS_END_RE.search(cue): return lower or nxt.startswith("..")
    if _STOP_END_RE.search(cue): return cased and lower
    return True


def _continues_after_pause(nxt: str, cased: bool) -> bool:
    # May a unit run on into the cue `nxt` after a long pause? Only when `nxt` shows it: a word that starts lowercase, or the pronoun "I"
    # (always a capital), in a video with both cases. In a video with one case only (Auslan-Daily's lowercase captions, an ALL-CAPS track)
    # the case says nothing, so Punkt decides as before.
    if not cased: return True
    letter, word = re.search(r"[^\W\d_]", nxt), WORD_RE.search(nxt)
    return bool(letter) and letter.group(0).islower() or bool(word) and _is_pronoun_i(word.group(0))


def reconstruct_sentences(
    captions: list[tuple[float, float, str]], max_tokens: int = 60, fold: frozenset[str] | None = None, pause_s: float | None = None,
) -> list[tuple]:
    """Group whole display cues into timestamp-supported caption units.

    Punkt reads the original text to find cue junctions inside linguistic sentences, and the cue ends limit it (_may_cross: a cue that
    ends a sentence ends its unit unless a lowercase word continues it). Connected cues form 1 unit; a unit can contain several
    linguistic sentences. Its BIO labels describe unit membership, not each linguistic sentence. Only after grouping, internal sentence-final
    periods become commas, and the sentence that follows a converted period loses its positional capital when the train lexicon `fold` says
    the word is ordinarily lowercase (see `fold_lexicon`; names, "I", initialisms and proper-noun compounds keep theirs). Questions and
    exclamations are retained, and the sentence after them keeps its capital.

    A pause longer than `pause_s` (the loader passes δ, inference.yaml boundary_stability.delta_s: the training targets join units and
    the scorer's chunks break at that pause too) ends the unit unless the next cue shows that the sentence runs on (_continues_after_pause):
    Punkt finds no sentence end where a caption has no full stop, and after a pause a capital most often starts a new sentence ("...studying
    Yoga Therapy" | 3.2 s | "They explained..."), while "...because of pride" | 1.8 s | "or vanity" is 1 sentence. None: no pause limit.

    Missing punctuation or a sentence above max_tokens leaves the source cues separate. The limit guards unreliable grouping evidence; it does
    not clip text or invent timestamps. Cues are never split.
    """
    if not captions: return []
    tok = _punkt_tokenizer()
    if tok is None: raise RuntimeError("Caption-unit grouping requires NLTK Punkt; install punkt_tab before loading annotations.")
    parts: list[str] = []; char_cue: list[int] = []
    for ci, (cs, ce, t) in enumerate(captions):
        if parts: parts.append(" "); char_cue.append(-1)      # junction char between cue ci-1 and ci
        parts.append(t); char_cue.extend([ci] * len(t))

    text = "".join(parts)
    # A video whose captions carry both cases: in one with no capitals (Auslan-Daily) or only capitals, case says nothing.
    cased = any(ch.isupper() for ch in text) and any(ch.islower() for ch in text)
    crossed: set[int] = set()                                 # junction k = boundary between cue k and k+1
    for a, b in tok.span_tokenize(text):
        ids = sorted({c for c in char_cue[a:b] if c != -1})
        for x, y in zip(ids, ids[1:]):
            # Text evidence alone is not enough to assert membership: the unit is timestamp-SUPPORTED, so a junction
            # the labeller distrusts cannot be crossed. `untrusted_o_intervals` calls an uncaptioned stretch longer
            # than TRUSTED_GAP_S unlocatable and marks it UNK; merging across the same stretch would relabel it I.
            gap = captions[y][0] - captions[x][1]
            if y == x + 1 and gap <= TRUSTED_GAP_S and _may_cross(captions[x][2], captions[y][2], cased) and (
                pause_s is None or gap <= float(pause_s) or _continues_after_pause(captions[y][2], cased)
            ): crossed.add(x)
    out: list[tuple] = []

    def emit(run: list[int]) -> None:
        s0, e1 = captions[run[0]][0], max(captions[c][1] for c in run)   # an earlier cue of the run can end after the last one
        joined = " ".join(captions[c][2] for c in run).strip()
        sents = [joined[a:b] for a, b in tok.span_tokenize(joined)]
        # A multi-cue chain is merged only when Punkt's failure to split it is INFORMATIVE, i.e. the chain is wrapped prose. A gloss list 
        # ("HOST DOG GARDEN ...") has no boundaries to find, so every junction reads as crossed and the whole list would fuse into one unit 
        # of minutes. 2 measured signatures of non-prose, either one refusing the merge: language rate under PROSE_MIN_WORDS_PER_S (asf prose 
        # p1 is 0.70 w/s; lists sign at ~0.1), or display cues that are glosses (<= GLOSS_MAX_WORDS_PER_CUE words each, mostly capitalised).
        prose = len(run) == 1 or _is_prose([captions[c] for c in run], joined, e1 - s0)
        if any(len(x.split()) > int(max_tokens) for x in sents): out.extend(captions[c] for c in run) # punctuation-unreliable
        elif len(sents) > 1 and _SENTENCE_FINAL_RE.search(joined) and prose:   # same completion evidence the k==1 chain needs
            # Boundaries were obtained from the unchanged text. Commas and case only change the target's rendering.
            rendered, after_period = [], False
            for i, part in enumerate(sents):
                if after_period: part = _fold_sentence_start(part, fold)
                after_period = False
                if i < len(sents) - 1 and not re.search(r"(?:[!?]|\.{3})[\"')\]]*$", part):
                    # A period becomes a comma; an abbreviation ("U.S.") keeps its period and gains one. ?, ! and an ellipsis stay.
                    joined_part = part + ',' if re.search(r"\b(?:[A-Za-z]\.){2,}$", part) else re.sub(r"\.(?=[\"')\]]*$)", ",", part)
                    after_period = joined_part != part
                    part = joined_part
                rendered.append(part)
            out.append((s0, e1, " ".join(rendered)))
        elif len(run) == 1: out.append(captions[run[0]])
        elif _SENTENCE_FINAL_RE.search(joined) and prose: out.append((s0, e1, joined))
        else: out.extend(captions[c] for c in run) # unpunctuated remnant or gloss list: keep the author's cues
        
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


def looks_asr_transcript(path: Path) -> bool:
    """Heuristic for word-timed tracks; this does not prove automatic-caption provenance.

    Plain WebVTT class tags are formatting and are not a reason to reject a track. Manual karaoke tracks can
    also contain word timestamps, so this existing exclusion must be reported as a heuristic.
    """
    try:
        with path.open(encoding="utf-8", errors="ignore") as fh: head = fh.read(20_000)
    except OSError: return False
    return bool(_ASR_WORD_TIMING_RE.search(head))


def is_scrolling_display(cues) -> bool:
    """A caption track that scrolls: a new line appears while the previous one is still on screen.

    Overlapping cue times are the signature. On such a track a cue boundary is a DISPLAY event, not an utterance event, so it lands wherever 
    the line filled up — `hello my name is nikki stratton and my` followed by `company`. Nothing downstream can recover an utterance boundary 
    from that, because the annotation never encoded one. Measured on the raw cues, before rolling-duplicate merging removes the repeats.
    """
    cues = list(cues)
    if len(cues) < 5: return False
    overlapping = sum(1 for a, b in zip(cues, cues[1:]) if b[0] < a[1] - 1e-6)
    return overlapping / (len(cues) - 1) > 0.5


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
    drop_noise: bool = False, lang_prefix: str | None = None, names: list[str] | None = None, exclude_rejected: bool = False,
) -> Path | None:
    subtitle_root = Path(subtitle_root)
    # `lang_prefix` restricts to `<vid>.<prefix>*.vtt` (e.g. "en" → .en-GB/.en; "de" → .de*). The preferred/reject
    # suffix lists are English-oriented and do NOT encode language, so harvesting shard tracks for a non-English
    # target needs this to avoid picking an English track and labelling it the target language.
    prefix = f"{video_id}.{lang_prefix}" if lang_prefix else video_id
    # `names`: the folder's sorted file names (`sorted_names`), from a caller that selects for many videos; a glob per video lists the
    # whole folder again (ase train: 10,761 listings of about 11,000 files). The names that `<prefix>*.vtt` matches are one block of
    # the sorted list when the prefix is literal (video ids hold no glob character; one that does takes the glob).
    if names is None or any(c in prefix for c in "*?["): candidates = sorted(subtitle_root.glob(f"{prefix}*.vtt"))
    else: candidates = [subtitle_root / n for n in itertools.takewhile(
        lambda n: n.startswith(prefix), itertools.islice(names, bisect.bisect_left(names, prefix), None)
    ) if n.endswith(".vtt") and len(n) >= len(prefix) + len(".vtt")]
    scored: list[tuple[tuple[int, int, str], Path]] = []
    for path in candidates:
        if exclude_rejected and any(path.name.endswith(rejected) for rejected in reject_suffixes): continue
        if looks_asr_transcript(path): continue
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


def sorted_names(folder: str | Path) -> list[str]: # 1 listing of a caption folder for many best_subtitle calls (find_best_subtitle `names`)
    return sorted(os.listdir(folder)) if Path(folder).is_dir() else []

def best_subtitle(
    subtitle_root: str | Path, video_id: str, subtitle_cfg: dict, lang_prefix: str | None = None, names: list[str] | None = None,
) -> Path | None:
    """`find_best_subtitle` driven by the `subtitles:` config block — the ONE selection rule, shared by the loader
    (lang_prefix=None; only the canonical `<vid>.<target>.vtt` exists) and prepare_yt25 (shard tracks, lang_prefix=target).
    Among raw shard tracks a rejected suffix only loses to every other track; in a prepared caption folder it is never read, because
    there it is another corpus's track (BOBSL's 2 corpora share subs/), so a video without its own track has no caption."""
    return find_best_subtitle(
        subtitle_root, video_id,
        preferred_suffixes=list(subtitle_cfg.get("preferred_suffixes", [".en.vtt"])),
        reject_suffixes=list(subtitle_cfg.get("reject_suffixes", [".en-orig.vtt"])),
        min_caption_chars=int(subtitle_cfg.get("min_caption_chars", 2)),
        reject_flattened_transcripts=bool(subtitle_cfg.get("reject_flattened_transcripts", True)),
        flattened_max_cues=int(subtitle_cfg.get("flattened_max_cues", 2)),
        flattened_min_chars=int(subtitle_cfg.get("flattened_min_chars", 500)),
        flattened_max_chars_per_second=float(subtitle_cfg.get("flattened_max_chars_per_second", 120.0)),
        drop_noise=bool(subtitle_cfg.get("drop_noise_captions", True)), lang_prefix=lang_prefix, names=names,
        exclude_rejected=lang_prefix is None,
    )


CONFIG_LEVEL_BLOCKS = ("splits", "subtitles", "pose")   # the data.yaml blocks that a dataset and a corpus refine (corpus_config)


def dataset_config(data_cfg: dict, dataset: str | None, *corpus: dict) -> dict:
    """`data_cfg` as the corpora of `dataset` read it: each block of CONFIG_LEVEL_BLOCKS is the global block (the rules of every corpus),
    then the block of data.yaml `datasets.<dataset>` (what the corpora of 1 source of videos and captions share: split file, track
    names, casing file, frame size), then the block of each `corpus` entry given; each level replaces only the keys it names."""
    datasets = data_cfg.get("datasets") or {}
    if dataset is not None and dataset not in datasets: raise SystemExit(
        f"[loader] data.yaml names the dataset {dataset!r}, which data.yaml datasets does not define ({', '.join(sorted(datasets))})."
    )
    # The global blocks travel with a view, so the view of another corpus taken from a view never inherits this corpus's keys.
    base = data_cfg.get("_global_blocks") or {block: data_cfg.get(block) or {} for block in CONFIG_LEVEL_BLOCKS}
    view = {**data_cfg, "_global_blocks": base}
    levels = [datasets[dataset]] if dataset is not None else []
    for block in CONFIG_LEVEL_BLOCKS:
        view[block] = {**base[block], **{k: v for level in (*levels, *corpus) for k, v in (level.get(block) or {}).items()}}
    return view


def corpus_config(data_cfg: dict, language: str) -> dict:
    """`data_cfg` as corpus `language` reads it (dataset_config with its data.yaml entry): BOBSL's 2 corpora keep their own split file,
    subtitle track and sidecar on 1 dataset, while every other rule stays shared. Every reader of one corpus's split file, subtitle
    rules or fallback frame size reads this view."""
    lang_cfg = data_cfg["languages"][language]
    return dataset_config(data_cfg, lang_cfg.get("dataset"), lang_cfg)


def video_meta_path(data_cfg: dict, language: str) -> Path:
    # The corpus's sidecar: data.yaml languages.<language>.video_meta where 2 corpora share a root (BOBSL), else <root>/video_meta.csv.
    lang_cfg = data_cfg["languages"][language]
    return Path(lang_cfg.get("video_meta") or Path(lang_cfg["root"]) / META_FILENAME)


def _load_split_file(path: Path) -> dict[str, str]:
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
    csv_path = str(split_cfg.get("file", "") or "")
    signverse = _load_split_file(Path(csv_path))
    # Fail loud rather than fall through to the random split: splits.file is CWD-relative, so an entry point started
    # from another directory would silently re-partition every video, making a checkpoint trained under one split and
    # evaluated under the other train-on-test. The fallback below is only for configs that declare no CSV at all.
    if csv_path and not signverse: raise FileNotFoundError(
        f"splits.file={csv_path!r} is configured but unusable from cwd {Path.cwd()} (missing, empty, or "
        f"no recognised id/split columns). Refusing the random fallback split."
    )
    if signverse:
        splits = {"train": [], "dev": [], "test": []}
        for video_id in video_ids:
            if (split := signverse.get(video_id)) in splits: splits[split].append(video_id)
        # A CSV that parses but shares no ids with the pose index (wrong language, stale export, ids vs stems) is the
        # same train-on-test hazard as a missing one — without this the loop falls through to the rng below.
        if not any(splits.values()): raise ValueError(
            f"splits.file={csv_path!r} parsed {len(signverse)} rows but matched NONE of the {len(video_ids)} "
            f"pose videos (e.g. {sorted(video_ids)[:3]} vs {sorted(signverse)[:3]}). Refusing the random fallback split."
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


def _split_captions(root: Path, video_ids, subtitle_cfg: dict, drop_noise: bool) -> dict[str, list[str]]:
    # {video_id: [each display cue (merge_rolling_captions) lowercased, in time order]}: the text the copy check compares. Subtitles only.
    out: dict[str, list[str]] = {}
    names = sorted_names(root / "subs")
    for vid in tqdm(video_ids, desc="[loader] Splitting caption sets", unit="vid", leave=False, dynamic_ncols=True):
        path = best_subtitle(root / "subs", vid, subtitle_cfg, names=names)
        if path is None: continue
        caps = [" ".join(str(t).lower().split()) for _, _, t in merge_rolling_captions(parse_vtt(path, drop_noise=drop_noise)) if t]
        if caps: out[vid] = caps
    return out


def _shingles(words: list[str], n: int) -> set[str]:
    # The n-word runs of `words`, as strings. Storing hash() ints instead would cut the loader's peak memory on ase.
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _shared_counts(sets: dict[str, set[str]], df_cap: int) -> dict[tuple[str, str], int]:
    # {(video a, video b): number of shared items that are RARE (in at most df_cap videos)}.
    inv: dict[str, set[str]] = defaultdict(set)
    for v, items in sets.items():
        for x in items: inv[x].add(v)
    shared: dict[tuple[str, str], int] = defaultdict(int)
    for vs in inv.values():
        if 2 <= len(vs) <= df_cap:
            for a, b in combinations(sorted(vs), 2): shared[(a, b)] += 1
    return shared


def _duplicate_pairs(caps: dict[str, list[str]], cfg: dict) -> list[tuple[str, str, float]]:
    """Video pairs that share content: a talk re-uploaded under a different id, or another version of the same script.

    2 tests score a pair, and the higher score counts; a pair is flagged above `cover`. Only RARE items count (in at most df_cap videos;
    above that an item is series boilerplate such as a scripted contact-info outro, which otherwise chains unrelated videos into one cluster).
    Cue test: a cue identifies content only if it is also LONG enough (>= min_words; single-sign vocabulary clips share a word with
    everything). The score is the shared identifying cues over the smaller video's identifying cues, and a video needs min_captions of
    them. Exact cue-set copies of min_captions x min_words words or more score 1.
    Transcript test: the same words broken into other cue lines. The cues are joined in time order, and the score is the shared word
    n-grams (shingle_words words) over the smaller video's n-grams. A video needs min_captions x min_words words, the evidence budget of
    the exact-copy rule.
    """
    df_cap, min_words, min_caps = int(cfg["df_cap"]), int(cfg["min_words"]), int(cfg["min_captions"])
    cover, n = float(cfg["cover"]), int(cfg["shingle_words"])
    cue_sets = {v: set(cs) for v, cs in caps.items()}
    ident = {v: k for v, cs in cue_sets.items() if len(k := {c for c in cs if len(c.split()) >= min_words}) >= min_caps}
    scores = {(a, b): k / min(len(ident[a]), len(ident[b])) for (a, b), k in _shared_counts(ident, df_cap).items()}
    exact: dict[frozenset[str], list[str]] = defaultdict(list)
    for vid, cs in cue_sets.items():
        if sum(len(c.split()) for c in cs) >= min_caps * min_words: exact[frozenset(cs)].append(vid)
    for vids in exact.values():
        if 2 <= len(vids) <= df_cap:
            for pair in combinations(sorted(vids), 2): scores[pair] = 1.0
    grams = {v: _shingles(w, n) for v, cs in caps.items() if len(w := WORD_RE.findall(" ".join(cs))) >= min_caps * min_words}
    for (a, b), k in _shared_counts(grams, df_cap).items():
        scores[(a, b)] = max(scores.get((a, b), 0.0), k / min(len(grams[a]), len(grams[b])))
    return sorted([(a, b, score) for (a, b), score in scores.items() if score > cover], key=lambda p: -p[2])


def train_copies(train: list[VideoRecord], splits: dict[str, list[str]], root: Path, subtitle_cfg: dict) -> dict[str, list]:
    """{train video: copy_of}: the video-level copy check (subtitles.dedup). `prepare_yt25.py --stage dedup` runs it once and writes the
    result to the video_meta.csv column copy_of; it is the only code that reads the captions of more than 1 split.

    `train` are the train records before the copy check (`load_language_records(..., before_dedup=True)`); they are compared with every dev
    and test video of `splits` that has a caption. A train video that shares content with a dev/test video (`_duplicate_pairs`: a re-upload,
    or another version of the same script) is a copy: [that video, score] (the highest score, then the first ids), and the train loader
    drops it whole. Every other train video is [] (not a copy). The eval sets stay as the split CSV defines them."""
    cfg = subtitle_cfg["dedup"]
    eval_ids = {v for s in ("dev", "test") for v in splits.get(s, [])}
    caps = _split_captions(root, [r.video_id for r in train] + sorted(eval_ids), subtitle_cfg, bool(subtitle_cfg.get("drop_noise_captions", True)))
    copy_of: dict[str, list] = {}
    for a, b, frac in sorted(_duplicate_pairs(caps, cfg), key=lambda p: (-p[2], p[0], p[1])):
        if (a in eval_ids) != (b in eval_ids): copy_of.setdefault(b if a in eval_ids else a, [a if a in eval_ids else b, round(frac, 4)])
    return {r.video_id: copy_of.get(r.video_id, []) for r in train}


class PooledEpochRecords:
    """`epoch -> train records` for the pooled rotation, as an importable object rather than a closure.

    A DataLoader worker pickles whole dataset and Python 3.14 starts workers with `forkserver` on POSIX instead of `fork`, so a lambda held 
    on the dataset stops the run with `Can't pickle local object`. Only the main process ever calls this (set_epoch runs before the workers 
    are rebuilt), but it has to survive the trip.
    """
    def __init__(self, cfg: dict, data_cfg: dict, language: str | None):
        self.cfg, self.data_cfg, self.language = cfg, data_cfg, language

    def __call__(self, epoch: int) -> list["VideoRecord"]:
        return resolve_pretrain_records(self.cfg, self.data_cfg, self.language, "train", epoch=epoch)[0]

    def cycle_epochs(self) -> int:
        # Consecutive epochs until the rotation has drawn every train record once: the slowest language's ceil(hours / hours per epoch).
        hours = {str(l): sum(r.pose.duration_s for r in cached_language_records(self.data_cfg, str(l), "train")) / 3600.0
                 for l in self.cfg["pretrain"]["languages"]}
        targets = _pool_targets(hours, float(self.cfg["pretrain"]["temperature"]), self.cfg["pretrain"].get("weights"))
        # 1e-9: the corpus that sets the epoch size divides its own hours exactly, up to float rounding
        return max(int(np.ceil(h / targets[l] - 1e-9)) for l, h in hours.items())


def resolve_pretrain_records(
    cfg: dict, data_cfg: dict, language: str, split: str, requested: str | None = None, epoch: int = 0,
) -> tuple[list[VideoRecord], dict[str, int] | None]:
    """Records for a SEGMENTATION trainer: the target language alone, or the multilingual pretraining pool.

    `pretrain.languages` in trainer config switches it on. Both segmentation trainers (S1 and the Moryossef baseline) route through here, 
    so 2 arms can never end up trained on different pools by accident — which is what would make a cascade comparison unfair. The dev
    split reads `pretrain.dev_languages` (default: the pool): the corpora whose dev labels choose the checkpoint, so a corpus can train
    without its dev set joining the check (BOBSL's automatic labels, and its official val holds the manual dev episodes).
    """
    langs = cfg_get(cfg, "pretrain", "languages")
    if langs and split != "train": langs = cfg_get(cfg, "pretrain", "dev_languages") or langs
    if not langs:
        recs, _ = load_language_records(data_cfg, language, split=split)
        return recs, None
    
    if requested: raise SystemExit(
        f"--language {requested!r} is meaningless with pretrain.languages={list(langs)}. "
        f"Drop --language to pretrain, or set pretrain.languages: null for a monolingual run on that language."
    )
    langs = [str(x) for x in langs]
    recs, realised = load_multilingual_records(
        data_cfg, langs, split=split, temperature=float(cfg["pretrain"]["temperature"]), seed=int(cfg.get("seed", 42)), epoch=int(epoch),
        weights=cfg["pretrain"].get("weights"),
    )
    return recs, realised


def segmenter_language_fingerprints(cfg: dict, data_cfg: dict, language: str, train_records: list[VideoRecord]) -> dict[str, str]:
    """Annotation fingerprint of the FULL train split of every language a segmenter trains on: the pool, or the target alone.

    A pooled run trains on a rotating sub-sample, so its epoch-0 records cannot be compared with a stage-2 or eval run, 
    which reads one whole language. These per-language stamps can (train.bio_pretrain.require_segmenter_data)."""
    langs = cfg_get(cfg, "pretrain", "languages")
    if not langs: return {str(language): annotation_fingerprint(train_records)}
    return {str(l): annotation_fingerprint(cached_language_records(data_cfg, str(l), "train")) for l in langs}


def stale_language_fingerprints(stamped: dict | None, data_cfg: dict, known: dict[str, str] | None = None) -> list[str]:
    """The corpora of per-language stamps (segmenter_language_fingerprints) whose whole train split is labelled otherwise now. A pooled
    segmenter trained on every corpus of its pool, so each stamped corpus is compared with its own current labels, whatever corpus a
    run reads (a benchmark target is in no pool). `known`: current fingerprints that the caller already has. A corpus that data.yaml
    does not name, or whose pose files this machine does not hold, is not compared."""
    langs, stale = data_cfg.get("languages") or {}, []
    for corpus, stamp in (stamped or {}).items():
        if corpus in (known or {}): now = known[corpus]
        elif corpus in langs:
            try: now = annotation_fingerprint(cached_language_records(data_cfg, corpus, "train"))
            except FileNotFoundError: continue
        else: continue
        if now != stamp: stale.append(corpus)
    return stale


def cached_language_records(data_cfg: dict, language: str, split: str) -> list[VideoRecord]:
    """Per-language record lists for the pool, parsed ONCE per process.

    The per-epoch rotation re-resolves the pool every epoch, but only the sub-sample changes — the underlying corpus does not. Without this 
    cache each epoch re-parses every subtitle file and re-runs Punkt sentence reconstruction for all pooled languages.
    Keyed by the config content that shapes the records, so an edited data config is not served a stale corpus; records are never mutated 
    downstream (the pool copies before shuffling), so sharing one list across epochs is safe.
    """
    lang_cfg = (data_cfg.get("languages") or {}).get(language) or {}
    corpus = dataset_config(data_cfg, lang_cfg.get("dataset"), lang_cfg)   # its corpus_config view
    lang_fingerprint = json.dumps(
        {"lang": lang_cfg, **{block: corpus[block] for block in CONFIG_LEVEL_BLOCKS}, "poses": data_cfg.get("poses")},
        sort_keys=True, default=str,
    )
    # The active loader function is part of the key: tests monkeypatch `load_language_records`, and a key of
    # (language, split, fingerprint) alone would serve one test's fake records to the next.
    key = (language, str(split), lang_fingerprint, load_language_records)
    if key not in _LANG_RECORDS_CACHE: _LANG_RECORDS_CACHE[key] = load_language_records(data_cfg, language, split=split)[0]
    return _LANG_RECORDS_CACHE[key]


def load_multilingual_records(
    data_cfg: dict, languages: list[str], split: str, temperature: float, seed: int = 42, epoch: int = 0,
    weights: dict[str, float] | None = None,
) -> tuple[list[VideoRecord], dict[str, int]]:
    """Records pooled across languages for language-agnostic SEGMENTATION pretraining.

    Sentence boundaries are marked by prosody (pauses, holds, movement dynamics) that is shared across signed languages, so a BIO 
    head can be pretrained on several corpora at once. Translation cannot: it is language-specific and stays monolingual (stage 2).

    Corpora differ in size by several-fold, so proportional pooling would make the largest one the de-facto training set. Sampling 
    weights are TEMPERATURE-FLATTENED, the standard multilingual-NMT recipe: `p_l propto w_l * h_l ** temperature` over each corpus's
    hours h_l (record durations: the loss sees time, and a BOBSL record lasts about 5 times a YouTube one), with an optional weight w_l
    per corpus (`weights`, pretrain.weights; default 1) — 1.0 is proportional, 0.0 uniform, 0.5 the usual compromise.

    Balance is reached by SUB-sampling the over-represented languages (train only), never by replicating the under-represented ones. Replicating
    up to the largest corpus would make the pool as big as the most-scaled-up language demands: on ase/asf/bfi a 16x epoch with
    each asf video repeated ~5 times WITHIN 1 epoch, so the model would see many epochs' worth of data (and heavy repetition) 
    before the first checkpoint. Sub-sampling keeps 1 epoch a bounded, comparable unit of compute.

    `epoch` rotates WHICH subset each language contributes, so no video is permanently discarded: coverage of the large corpora is
    spread across epochs instead of forced into one. Deterministic in (seed, epoch), so a resumed run replays its epoch exactly.

    Returns the pooled records and the realised per-language record counts (records are clean segments), which belong in the paper: 
    the sampling rates are part of the experimental setup, not an implementation detail (the S1 checkpoint also stamps the hours).
    """
    if not languages: raise ValueError("load_multilingual_records needs at least one language")
    per_lang: dict[str, list[VideoRecord]] = {}
    for lang in languages: per_lang[lang] = cached_language_records(data_cfg, lang, split)

    # A configured language that contributes nothing means this machine lacks that corpus. Skipping it would train a
    # DIFFERENT model that still stamps the configured pool key, and nothing downstream compares the realised mix.
    empty = sorted(l for l, recs in per_lang.items() if not recs)
    if empty: raise FileNotFoundError(
        f"pretrain.languages lists {list(languages)}, but {', '.join(empty)} has no {split} record on this machine. "
        f"Fetch the corpus (prepare_yt25.py --languages {' '.join(empty)}), or drop it from pretrain.languages."
    )
    # Dev and test are pooled AS-IS, every record of each language: the segmenter monitor scores each language apart and averages the
    # languages (evaluation.segmenter_eval.segmenter_dev_scores), so no language outweighs another and no dev video is left out.
    if len(per_lang) == 1 or split != "train":
        pooled = [r for recs in per_lang.values() for r in recs]
        return pooled, {k: len(v) for k, v in per_lang.items()}

    hours = {k: sum(r.pose.duration_s for r in v) / 3600.0 for k, v in per_lang.items()}
    targets = _pool_targets(hours, temperature, weights)
    rng = random.Random(int(seed))
    pooled: list[VideoRecord] = []
    realised: dict[str, int] = {}

    for lang, recs in per_lang.items():
        # ROTATE the slice per epoch so a sub-sampled corpus is never permanently truncated: the records lie end to end on a circle of
        # the corpus's hours, in an order shuffled once per language (seeded, so it is stable across epochs and resumes), and epoch e
        # takes the records that start in the window [e * target, (e + 1) * target) of it. A corpus reduced to `target` of `h` hours
        # covers all of them every ceil(h / target) epochs; the corpus that sets the epoch size gives every record every epoch.
        order = list(recs)
        # crc32, NOT hash(): str.__hash__ is PYTHONHASHSEED-salted, so hash() would give a DIFFERENT train slice on
        # every process launch, and a resumed run would not replay its epoch. crc32 is stable across processes, so a
        # run is reproducible and resumable.
        random.Random(int(seed) ^ zlib.crc32(lang.encode())).shuffle(order)
        length = np.array([r.pose.duration_s for r in order]) / 3600.0
        starts, total = np.concatenate([[0.0], np.cumsum(length)[:-1]]), float(length.sum())
        lo = (int(epoch) * targets[lang]) % total if total > 0 else 0.0
        taken = [r for r, s in zip(order, starts) if lo <= s < lo + targets[lang] or lo <= s + total < lo + targets[lang]]
        pooled.extend(taken)
        realised[lang] = len(taken)

    rng.shuffle(pooled)
    shares = {k: round(targets[k] / sum(targets.values()), 3) for k in targets}
    print(f"[loader] multilingual {split}: hours {({k: round(v, 1) for k, v in hours.items()})} -> {({k: round(v, 1) for k, v in targets.items()})} "
          f"per epoch, records {realised} (tau={temperature}, weights {weights or {}}, hour shares {shares})", flush=True)
    return pooled, realised


def _pool_targets(amounts: dict[str, float], temperature: float, weights: dict[str, float] | None = None) -> dict[str, float]:
    """Hours each language contributes to one pooled epoch, for per-language hours `amounts` (the durations of its train records).

    SUB-sample to target shares: pick the epoch size that the most over-represented language can support WITHOUT replication, i.e. the
    largest total for which every target <= that language's own hours. Upsampling instead (scaling up to the biggest corpus) repeats the
    small corpora several times inside ONE epoch, so the model sees many epochs' worth of a language before the first checkpoint and
    overfits during epoch 1 — the failure this bound exists to prevent. Temperature and the weights (pretrain.weights: 0.5 each for 2
    corpora of 1 sign language gives that language one share) then only set the SHARES, never the repetition.
    """
    shares = {k: float((weights or {}).get(k, 1.0)) * float(h) ** float(temperature) for k, h in amounts.items()}
    if any(w <= 0 for w in shares.values()): raise ValueError(f"pool weights must be positive: {shares}")
    total_w = sum(shares.values())
    scale = min(amounts[k] * total_w / shares[k] for k in amounts)
    return {k: min(float(h), scale * shares[k] / total_w) for k, h in amounts.items()}


# The code that decides a load's records, so a change to it misses the records cache (records_cache_key).
RECORDS_CODE = ("data/loader.py", "data/windowing.py", "poses/pose_io.py", "utils.py")

def records_cache_key(data_cfg: dict, language: str, split: str | None) -> str:
    """Digest of everything that decides `load_language_records(data_cfg, language, split)`: the loader code (RECORDS_CODE), data.yaml
    (less the translation warm starts, which no record reads), configs/inference.yaml (Λ_min), the Python and NLTK versions, the Punkt
    model (_punkt_identity), and the
    name, size and change time of every file that a load reads: the casing file, and the split CSV, the sidecar and every pose and
    caption file of this corpus and of each corpus in its casing pool (a train load folds case by the train captions of the pool)."""
    cfg = json.loads(json.dumps(data_cfg, default=str))
    for lang_cfg in cfg["languages"].values(): lang_cfg.pop("pretrained_slt", None)
    try: nltk_version = importlib.metadata.version("nltk")
    except importlib.metadata.PackageNotFoundError: nltk_version = None
    h = hashlib.sha256(json.dumps([ANNOTATION_PROTOCOL, language, split, list(sys.version_info[:2]), nltk_version, _punkt_identity(), cfg],
                                  sort_keys=True).encode())
    repo = Path(__file__).resolve().parent.parent
    for path in (*(repo / p for p in RECORDS_CODE), Path("configs/inference.yaml")): h.update(path.read_bytes())
    langs = cfg["languages"]
    # The casing file and its pool decide the texts only when the load folds case (subtitles.merge_sentences).
    entry = case_lexicon_entry(cfg, language) if corpus_config(cfg, language).get("subtitles", {}).get("merge_sentences") else None
    lexicon_file, pool = entry or (None, [])
    files = [lexicon_file] if lexicon_file else []
    for other in sorted({language, *(l for l in pool if l in langs)}):
        root = Path(langs[other]["root"])
        # A pool corpus's split CSV decides which of its captions are train captions, and so the casing file.
        if split_file := str(corpus_config(cfg, other)["splits"].get("file", "") or ""): files.append(Path(split_file))
        files.append(video_meta_path(cfg, other))
        for folder in (root / "poses", root / "subs"):
            if folder.is_dir(): files += sorted(Path(entry.path) for entry in os.scandir(folder))
    for path in files:
        try: stat = path.stat(); h.update(f"{path}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
        except OSError: h.update(f"{path}\tmissing\n".encode())
    return h.hexdigest()

class _Tee(io.TextIOBase): # Prints through to `out` and keeps a copy: the lines that a load from the records cache prints again.
    def __init__(self, out, copy: io.StringIO): self.out, self.copy = out, copy
    def write(self, text: str) -> int: self.copy.write(text); return self.out.write(text)
    def flush(self) -> None: self.out.flush()

def load_language_records(
    data_cfg: dict, language: str, split: str | None = None, report: dict | None = None, before_dedup: bool = False,
) -> tuple[list[VideoRecord], dict[str, list[str]]]:
    """`_build_language_records`, cached on disk: `<root>/.cache/records_<language>_<split>.pkl` holds the records of the last load with its
    `records_cache_key`, and a load with the same key reads them (seconds, not minutes) and prints that load's messages again. Any
    change of the key loads again and rewrites the file; deleting the file does too. A load that fills `report` or asks for the
    records `before_dedup` is not cached."""
    if report is not None or before_dedup: return _build_language_records(data_cfg, language, split, report, before_dedup)
    # Named by the corpus, because 2 corpora may share a root (BOBSL's automatic and manual labels).
    path = Path(data_cfg["languages"][language]["root"]) / ".cache" / f"records_{language}_{split or 'all'}.pkl"
    key = records_cache_key(data_cfg, language, split)
    try:
        with path.open("rb") as f:
            if pickle.load(f) == key:
                records, splits, printed = pickle.load(f)
                print(printed, end="", flush=True)
                print(f"[loader] {language}/{split or 'all'}: {len(records)} records from {path}, written by a load of the same inputs "
                      f"(the loader code, data.yaml, and every pose, caption and sidecar file); delete it to load again.", flush=True)
                return records, splits
    except Exception: pass   # no file, a half-written one, or one that this Python cannot read: load again
    printed = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, printed)): records, splits = _build_language_records(data_cfg, language, split)
    # Atomic: the ranks of a torchrun job load at once, and a half-written file only makes the next start load again.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as f:
            pickle.dump(key, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((records, splits, printed.getvalue()), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        print(f"[loader] WARNING: cannot write the records cache {path} ({exc}); every start loads again.", flush=True)
    return records, splits

def _build_language_records(
    data_cfg: dict, language: str, split: str | None = None, report: dict | None = None, before_dedup: bool = False,
) -> tuple[list[VideoRecord], dict[str, list[str]]]:
    """Records of the clean segments of every kept video (`split_clean_segments`): the id is the source video id, or
    `f"{vid}@{a}"` for a video cut at an empty run, and `rec.pose.video_id` is always the source id.

    `report`, when given, is filled with the per-rule tallies of this load (videos and cues kept or dropped by each
    pipeline rule) — the corpus audit `report.py data` prints. Every consumer of the records is unaffected.
    `before_dedup` returns the source records before the copy check, the quarantines and the cuts: the input of `train_copies`."""
    lang_cfg = data_cfg["languages"][language]
    root = Path(lang_cfg["root"])
    corpus = corpus_config(data_cfg, language)   # this corpus's split file, subtitle rules and fallback frame size
    meta_path = video_meta_path(data_cfg, language)
    # Per-video fps from the video_meta.csv sidecar (our extractions vary per video; SignVerse is fixed 24 fps).
    # config pose_fps is only the fallback for a video without a sidecar row: on it timestamps drift ~2x and ~44% of captions get dropped.
    video_meta = load_video_meta(meta_path)
    # A sidecar without the run columns predates the visual cuts: every stream would stay uncut, so refuse it. A blank
    # value in an existing column is one unknown video and only warns (below). A column that the loader does not read
    # (masked_runs of an earlier converter) is ignored, and unreliable_spans may be absent (a corpus with no such annotation).
    header: list[str] = []
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8", newline="") as f: header = next(csv.reader(f), [])
    if missing := [key for key in (*RUN_COLUMNS, *QUARANTINE_COLUMNS) if key not in header]: raise SystemExit(
        f"[loader] {meta_path} has no {'/'.join(missing)} column (stale or absent sidecar). For SignVerse poses run "
        f"`python prepare_yt25.py --stage all --overwrite --languages {language}`: a sidecar without these columns comes from a converter "
        f"that chose the kept body without the signer switch, so its pose files are converted again (`--stage person-counts`, "
        f"`--stage subs` and `--stage convert` without `--overwrite` refuse such a file); for poses from slp-pose, write its metadata "
        f"again with the current slp-pose; for other poses run `python prepare_yt25.py --stage meta --languages {language}` (it fills "
        f"handless_runs from the pose files; its blank empty_runs/switch_runs/default_signing_runs then warn as unknown)."
    )
    pose_cfg = corpus["pose"]   # the fallback frame size (and rate) of its dataset, then its own
    fps_fallback = pose_cfg.get("fps")   # optional: every video's rate is its frames / video_meta.csv duration_s
    all_ids = sorted({base_video_id(p) for p in (root / "poses").glob("*.npy")})   # the file names only: the split lists need no header
    if not all_ids: raise FileNotFoundError( # empty/missing poses/ → 0 records everywhere; fail loud
        f"[loader] no pose .npy files under {root / 'poses'} for language '{language}'. "
        f"For SignVerse-2M languages (asf/bfi) run `python prepare_yt25.py --stage all --languages {language}` "
        f"(docs/run_real_data.md §2a); for own extractions, place per-video (T,133,3) .npy there."
    )
    subtitle_cfg = corpus.get("subtitles", {})
    splits = build_splits(all_ids, corpus.get("splits", {}))
    # Not an error: a corpus that shares its pose folder with another (BOBSL's manual alignment) puts only its own videos in train, dev
    # or test. Only the corpus's own load prints this, never the casing file's scan of its pool.
    if (loaded := sum(map(len, splits.values()))) < len(all_ids): print(
        f"[loader] {language}: {corpus.get('splits', {}).get('file')} puts {loaded} of the {len(all_ids)} videos in {root / 'poses'} in "
        f"train, dev or test; this corpus loads only those.", flush=True)
    selected_ids = splits.get(split, []) if split else all_ids
    drop_noise = bool(subtitle_cfg.get("drop_noise_captions", True))

    # Human-caption-only splits (default: test). NLLB machine-translated references are noisy BLEU targets — scoring against them 
    # measures "agreement with NLLB", not translation quality — so drop MT-captioned videos on those splits. Provenance is video_meta.csv 
    # `caption_source` (written by `prepare_yt25.py --stage subs`); excluded sources default to just "mt" (raw-shard captions are kept 
    # — usually human uploads). Absent → nothing excluded.
    human_only = set(subtitle_cfg.get("human_only_splits", ["test"]) or [])
    exclude_sources = set(subtitle_cfg.get("human_only_exclude_sources", ["mt"]) or [])
    dropped_mt_caption = 0
    if split in human_only:
        drop_ids = {vid for vid, m in video_meta.items() if (m.get("caption_source") in exclude_sources)}
        before = len(selected_ids)
        selected_ids = [v for v in selected_ids if v not in drop_ids]
        dropped_mt_caption = before - len(selected_ids)
        if before != len(selected_ids): print(
            f"[loader] {language}/{split}: excluded {before - len(selected_ids)} video(s) with "
            f"{'/'.join(sorted(exclude_sources))} captions (human references only; subtitles.human_only_splits).", flush=True
        )
    # The .npy headers (frame counts) of this load's own videos only, and the whole file of a video with repeated frames: a loader reads
    # no pose file of another split.
    pose_index = build_pose_index(
        root / "poses", fps=fps_fallback, video_meta=video_meta, video_ids=selected_ids,
        width=int(pose_cfg["width"]) if pose_cfg.get("width") is not None else None,
        height=int(pose_cfg["height"]) if pose_cfg.get("height") is not None else None,
        model_fps=float(data_cfg["poses"]["model_fps"]),
        pulldown=(tuple(data_cfg["poses"]["pulldown_content_fps"]), float(data_cfg["poses"]["pulldown_line_ratio"]),
                  float(data_cfg["poses"]["pulldown_repeat_motion"])),
    )
    # The run columns hold stored frames; every rule below reads the grid that the reader gives a video (poses.pose_io.grid_runs).
    for vid, index in pose_index.items():
        if vid in video_meta: video_meta[vid] = {
            **video_meta[vid], **{key: grid_runs(index, video_meta[vid].get(key)) for key in (*RUN_COLUMNS, *QUARANTINE_COLUMNS)}
        }
    missing_meta = [vid for vid in pose_index if vid not in video_meta]
    if missing_meta and fps_fallback is not None: print(
        f"[loader] WARNING: {len(missing_meta)}/{len(pose_index)} {language} videos missing from {meta_path}; falling back "
        f"to pose.fps={fps_fallback} for them — run `python prepare_yt25.py --stage meta --root {root.parent} --languages {root.name}` "
        f"(yt-dlp metadata fetch, no video download)."
    )
    min_covered = float((data_cfg.get("poses", {}) or {}).get("min_cue_coverage", 0.0))
    dropped_uncovered: list[tuple[str, float]] = []
    records: list[VideoRecord] = []
    dropped_no_caption, dropped_all_quarantined = 0, 0
    per_video: dict[str, dict[str, int]] = {}
    fold = case_lexicon(data_cfg, language, split) if subtitle_cfg.get("merge_sentences") else None

    # 1.0 disables the filter (no video can exceed a full share). The key is GLOBAL under `subtitles:`, which is correct while every 
    # corpus targets English; non-Latin-target corpus need a per-language override which would switch the filter off for English corpora.
    max_non_latin = float(subtitle_cfg.get("max_non_latin_ratio", 1.0))
    min_english = float(subtitle_cfg.get("min_english_function_word_ratio", 0.0))
    language_check_min_words = int(subtitle_cfg.get("language_check_min_words", 100))
    dropped_non_latin: list[tuple[str, float]] = []
    dropped_wrong_language: list[tuple[str, float]] = []
    drop_scrolling = bool(subtitle_cfg.get("drop_scrolling_tracks", True))
    dropped_scrolling: list[tuple[str, float]] = []
    # δ (inference.yaml boundary_stability.delta_s): the longest overlap of 2 units that _clamp_overlaps repairs, and the pause after
    # which a unit runs on only into a cue that continues its sentence (reconstruct_sentences); the loader reads configs/inference.yaml
    # itself, as for Λ_min, so every stage builds the same units.
    delta_s = float(load_yaml("configs/inference.yaml")["boundary_stability"]["delta_s"])
    names = sorted_names(root / "subs")
    for video_id in tqdm(selected_ids, desc=f"[loader] {language}/{split or 'all'}", unit="vid", leave=False, dynamic_ncols=True):
        subtitle_path = best_subtitle(root / "subs", video_id, subtitle_cfg, names=names)
        if subtitle_path is None:
            dropped_no_caption += 1
            continue
        source_cues = parse_vtt(subtitle_path, drop_noise=drop_noise)
        # A SCROLLING display (more than half the cues start before previous one ends: YouTube's rolling two-line auto-captions) encodes 
        # DISPLAY times at both ends of every cue: a cue leaves the screen when line 2 later arrives, so its end runs a median 1.9-2.6 s 
        # (p90 3.0-3.9 s) past the next cue's start — 3.8-7.8x the 0.5 s δ. Punctuation can fix the grouping of such cues into units; 
        # it cannot move their timestamps, so the whole track is dropped, not repaired (bfi/0cw4rELLAtc, ase/T9C4QbZ7qOs).
        if drop_scrolling and is_scrolling_display(source_cues):
            n = len(source_cues)
            dropped_scrolling.append((video_id, sum(1 for a, b in zip(source_cues, source_cues[1:]) if b[0] < a[1] - 1e-6) / max(1, n - 1)))
            continue
        raw_cues = merge_rolling_captions(source_cues)
        captions = raw_cues
        if subtitle_cfg.get("merge_sentences"):  # group display-wrapped cues into caption units (Punkt over the caption stream)
            captions = reconstruct_sentences(captions, fold=fold, pause_s=delta_s)
        min_dur = float(subtitle_cfg.get("min_duration_s", 0.2))
        # Caption times refer to the source video. The supplied pose timeline can be shorter; a caption onset
        # outside it has no pose evidence. This comparison does not identify why the source poses stop.
        dur = pose_index[video_id].duration_s
        # Some source VTTs ship genuinely OVERLAPPING cues with distinct text. merge_rolling_captions only fuses overlapping DUPLICATES, 
        # so these survive, and overlapping SentenceSpans corrupt BIO labels — a neighbour's `I` overwrites the closing `O`, and the 
        # first-complete-span target becomes ill-defined. Clamp each start to the previous end: it trims the disputed frames from the 
        # LATER sentence (whose onset is the less certain of the two) and never invents a boundary; an overlap above δ quarantines both.
        ordered = sorted(captions, key=lambda c: (c[0], c[1]))
        # Measure temporal coverage on the cues as the track gives them, before the grouping, any text removal and the overlap repair:
        # the share then depends on the 2 timelines alone, so no caption rule moves a video's decision (on units, a split of the units
        # after the pose end could drop a whole video), and a short or textless cue still counts as covered. Use the same one-second
        # end tolerance as the span filter.
        covered = sum(0 <= c[0] < dur and c[1] <= dur + 1.0 for c in raw_cues) / max(1, len(raw_cues))
        # Quarantine unsupported ends before overlap repair so they remain untrusted supervision.
        captions = _quarantine_end_straddlers(captions, dur)
        straddlers = sum(1 for c in captions if len(c) > 3 and not c[3])
        # The corpus's own marks of where its track does not hold the signing (video_meta.csv unreliable_spans) enter the overlap repair as
        # text-less quarantined cues, so 1 rule holds for every pair of cues: an overlap of at most δ is trimmed, a longer one quarantines
        # both, and a unit that holds a mark is quarantined. They become unlabelled spans with the unusable units below. The overlap counts
        # are those between the captions; `marked` is what the marks cost: the reliable units with text that they remove or quarantine.
        marks = [(max(0.0, float(a)), min(float(dur), float(b)), "", False, UNRELIABLE_MARK)
                 for a, b in (video_meta.get(video_id) or {}).get("unreliable_spans") or []]
        marks = [m for m in marks if m[1] > m[0]]
        repaired, clamped, swallowed, containers, disputed = _clamp_overlaps(captions, delta_s)
        marked = 0
        if marks:
            reliable_text = lambda cues: sum(1 for c in cues if c[2] and (len(c) < 4 or c[3]))
            with_marks = _clamp_overlaps([*captions, *marks], delta_s)[0]
            marked, repaired = reliable_text(repaired) - reliable_text(with_marks), with_marks
        captions = repaired
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
        if min_english > 0.0 and captions:
            words, english = english_function_word_ratio(c[2] for c in captions)
            if words >= language_check_min_words and english < min_english:
                dropped_wrong_language.append((video_id, english))
                continue
        # 3 independent reasons a unit cannot be supervision, attributed to the FIRST that fires so the rows sum to the drop: shorter than 
        # minimum unit, outside the pose stream, or carrying no letters at all (a cue of digits or punctuation is not a translation target).
        # A too-short or letterless unit still marks signing ("Good." in 0.1 s, "999."): it becomes an unlabelled span (UNK), as an
        # unreliable span does below, never O. A too-short unit's times collapsed, so its signing lies somewhere in the uncaptioned gap
        # around it: that whole gap, from the cue before it to the cue after it (the captions are in order and disjoint here), becomes its
        # span (BOBSL's collapsed SAT alignments, 0.16 s, are such units). A unit outside the pose stream has no frames to label.
        kept, too_short, outside_poses, no_text, unusable = [], 0, 0, 0, []
        for i, c in enumerate(captions):
            if len(c) > 4 and c[4] == UNRELIABLE_MARK: unusable.append(c[:2]); continue   # a mark of the corpus (above)
            if not min_dur <= (c[1] - c[0]):
                too_short += 1
                unusable.append((min(c[0], captions[i - 1][1]) if i else c[0], max(c[1], captions[i + 1][0]) if i + 1 < len(captions) else c[1]))
                continue
            if not (c[0] < dur and c[1] <= dur + 1.0): outside_poses += 1; continue
            if not any(ch.isalpha() for ch in c[2]): no_text += 1; unusable.append(c[:2]); continue
            kept.append(SentenceSpan(video_id=video_id, start_s=c[0], end_s=c[1], text=c[2], reliable=bool(c[3]) if len(c) > 3 else True))
        # The marks and the unusable units become unlabelled spans (after the overlap repair, no kept unit overlaps one). A span only adds
        # UNK: the long uncaptioned stretches stay those of the kept units, so a span inside one does not make its sides short (trusted O).
        spans, _ = mark_unreliable(tuple(kept), unusable, video_id, dur)
        untrusted = tuple(untrusted_o_intervals(kept, dur, TRUSTED_GAP_S)) if len(spans) > len(kept) else None
        # Exclude poor temporal coverage as a whole-video data-quality rule. Do not stretch the pose timestamps
        # to the caption duration: neither an offset nor a different frame rate is established by this mismatch.
        if covered < min_covered:
            dropped_uncovered.append((video_id, covered))
            continue
        # Require >=1 RELIABLE span: an all-quarantined record contributes no anchor, target, or gold event, so
        # keeping it only loads poses nothing uses. Invariant: a record that reaches training/eval is usable.
        if not any(sp.reliable for sp in spans):
            dropped_all_quarantined += 1
            continue
        records.append(VideoRecord(language, video_id, pose_index[video_id], spans, untrusted))
        per_video[video_id] = {
            "source_cues": len(source_cues), "rolling_duplicates_merged": len(source_cues) - len(raw_cues),
            "units_after_grouping": len(ordered), "overlaps_swallowed": swallowed, "overlaps_clamped": clamped,
            "end_straddlers_quarantined": straddlers, "overlaps_containers_quarantined": containers, "overlaps_disputed_quarantined": disputed, "spans_kept": len(kept), 
            "spans_dropped_too_short": too_short, "spans_dropped_outside_poses": outside_poses, "spans_dropped_no_text": no_text,
            "units_quarantined_unreliable": marked,
        }
    if dropped_uncovered: print(
        f"[loader] {language}/{split or 'all'}: {len(dropped_uncovered)} video(s) dropped as POSE-TRUNCATED "
        f"(under {min_covered:.0%} of their caption cues fall inside the supplied pose timeline; e.g. "
        + ", ".join(f"{v} {c:.0%}" for v, c in sorted(dropped_uncovered, key=lambda x: x[1])[:3]) + "); poses.min_cue_coverage.", flush=True
    )
    if dropped_scrolling: print(
        f"[loader] {language}/{split or 'all'}: {len(dropped_scrolling)} video(s) dropped as SCROLLING "
        f"(over half the cues start before the previous one ends, so every cue time is a display event; e.g. "
        + ", ".join(f"{v} {r:.0%}" for v, r in sorted(dropped_scrolling, key=lambda x: -x[1])[:3]) 
        + "); subtitles.drop_scrolling_tracks.", flush=True
    )
    if dropped_non_latin: print(
        f"[loader] {language}/{split or 'all'}: {len(dropped_non_latin)} video(s) dropped as WRONG-LANGUAGE (>{max_non_latin:.0%} non-Latin "
        f"caption characters; e.g. " + ", ".join(f"{v} {r:.0%}" for v, r in sorted(dropped_non_latin, key=lambda x: -x[1])[:3])
        + "); subtitles.max_non_latin_ratio.", flush=True
    )
    if dropped_wrong_language: print(
        f"[loader] {language}/{split or 'all'}: {len(dropped_wrong_language)} video(s) dropped as WRONG-LANGUAGE "
        f"(under {min_english:.0%} English function words over {language_check_min_words}+ words; e.g. "
        + ", ".join(f"{v} {r:.1%}" for v, r in sorted(dropped_wrong_language, key=lambda x: x[1])[:3])
        + "); subtitles.min_english_function_word_ratio.", flush=True
    )
    if dropped_no_caption: print(
        f"[loader] {language}/{split or 'all'}: {dropped_no_caption}/{len(selected_ids)} videos dropped "
        f"(no usable caption in {root / 'subs'}).", flush=True
    )
    # Cross-split copy check. YouTube corpora contain RE-UPLOADS: the same talk under different video ids, so an id-based split puts it
    # in BOTH train and dev/test and the model scores by memorisation. Removal is TRAIN-side, following the decontamination convention
    # (keep the benchmark intact, purge the training copy): a train copy of a dev/test video is dropped whole. Single sentences that a
    # train video shares with dev/test stay, as in machine translation; report.py results discloses that overlap (train_text_videos).
    # The decisions are made ONCE, in data preparation (`prepare_yt25.py --stage dedup`, `train_copies`), and stored in the video_meta.csv
    # column copy_of. Only the train loader reads it; dev and test change nothing.
    dedup_cfg = subtitle_cfg["dedup"]
    deduplicated_train = 0
    if before_dedup: return records, splits
    if dedup_cfg["enabled"] and split == "train":
        if unknown := sorted(r.video_id for r in records if (video_meta.get(r.video_id) or {}).get("copy_of") is None): raise SystemExit(
            f"[loader] {meta_path} has no copy_of value for {len(unknown)} train video(s) (e.g. {', '.join(unknown[:3])}), "
            f"so the train copies of dev/test videos are unknown (a stage that writes captions blanks the column). Run "
            f"`python prepare_yt25.py --stage dedup --languages {language}`: it reads the train, dev and test captions once and writes it."
        )
        # Only checked train videos hold a value, so a dev or test video with one moved there from train after the check.
        if moved := sorted(v for s in ("dev", "test") for v in splits.get(s, []) if (video_meta.get(v) or {}).get("copy_of") is not None):
            raise SystemExit(
                f"[loader] {meta_path} holds a copy_of value for {len(moved)} dev/test video(s) (e.g. {', '.join(moved[:3])}): "
                f"the split changed after the copy check. Run `python prepare_yt25.py --stage dedup --languages {language}`."
            )
        if drop := {r.video_id for r in records if video_meta[r.video_id]["copy_of"]}:
            records = [r for r in records if r.video_id not in drop]
            print(f"[loader] {language}/train: de-duplicated {len(drop)} train video(s) whose content also appears in dev/test "
                  f"({', '.join(sorted(drop)[:5])}{'...' if len(drop) > 5 else ''}); video_meta.csv copy_of, subtitles.dedup.", flush=True)
            deduplicated_train = len(drop)

    # NO VISIBLE HANDS, every split: a reliable unit with at least poses.handless_unit_share of its frames in handless_runs (the
    # kept body shows no hand: credits, a URL, an end card, lyrics over a person whose hands are out of shot) is quarantined
    # in source time, BEFORE the cuts. No frame is cut and no unit time moves.
    # SWITCH DOUBT, every split, the same way: a reliable unit with at least poses.switch_unit_share of its frames in switch_runs (the
    # converter keeps the signer switch's body, not its default body) and at least poses.switch_doubt_share of its frames in
    # default_signing_runs (that default body signs: two people sign, so whom the caption follows is in doubt).
    # VISUAL CUTS, every split: frames where no real body is present (empty_runs) leave every stream, and each clean segment
    # between them is a record (split_clean_segments). A second person in the frame cuts nothing: the signer switch keeps the
    # person who signs, and the switch doubt above quarantines a unit where the switch keeps another body while the default body
    # signs. Split AFTER the copy check above, which reads copy_of by source id. A blank column is unknown: warn, and cut or
    # quarantine nothing for that kind.
    rules = {kind: [(key, float(data_cfg["poses"][share])) for key, share in rule.items()] for kind, rule in QUARANTINE_RULES.items()}
    for key in (*RUN_COLUMNS, *QUARANTINE_COLUMNS):
        if unknown := sum(1 for r in records if (video_meta.get(r.video_id) or {}).get(key) is None): print(
            f"[loader] {language}/{split or 'all'}: WARNING {unknown}/{len(records)} videos have no {key} in video_meta.csv; " + (
                "for SignVerse poses, the current converter did not write their pose files (an older converter, or a stopped `--overwrite` "
                "run): run `prepare_yt25.py --stage all` to convert them again." if key in QUARANTINE_RULES["switch_doubt"] else
                "run `prepare_yt25.py --stage person-counts` to measure them."
            ) + f" Nothing is {'cut' if key in RUN_COLUMNS else 'quarantined'} for them by that column.", flush=True
        )
    segments: list[VideoRecord] = []
    # Λ_min (inference.yaml span_selection.min_span_s): no clean segment is shorter than the shortest span the FSM may commit.
    min_span_s = lambda_min_s(load_yaml("configs/inference.yaml"))
    for source in records:
        meta = video_meta.get(source.video_id) or {}
        cut_runs = [run for key in RUN_COLUMNS for run in meta.get(key) or []]
        # The cut is attributed first, then each quarantine kind in QUARANTINE_RULES order: a unit that the cut alone removes or
        # quarantines is a cut unit, also when it is handless (an empty run is always a handless run). A reliable unit lies in
        # exactly 1 segment, so these sums count units.
        parts = split_clean_segments(source, cut_runs, min_span_s)
        reliable = sum(sp.reliable for part in parts for sp in part.sentences)
        counts = {"units_cut": sum(sp.reliable for sp in source.sentences) - reliable}
        quarantined = source
        for kind, rule in rules.items():
            quarantined = quarantine_units(quarantined, [(meta.get(key) or [], share) for key, share in rule])
            parts = split_clean_segments(quarantined, cut_runs, min_span_s)
            before, reliable = reliable, sum(sp.reliable for part in parts for sp in part.sentences)
            counts[f"units_quarantined_{kind}"] = before - reliable
        if not reliable:  # the all-quarantined rule, after the cuts
            dropped_all_quarantined += 1
            continue
        segments.extend(parts)
        per_video[source.video_id].update({
            **counts,
            "frames_cut": source.pose.total_frames - sum(part.pose.total_frames for part in parts),
            **{f"frames_{key}": sum(b - a for a, b in meta.get(key) or []) for key in RUN_COLUMNS},
        })
    records = segments
    if report is not None:
        kept = list(dict.fromkeys(r.pose.video_id for r in records))   # source videos, in record order
        kept_meta = [video_meta.get(v) or {} for v in kept]
        known = [(m["undetected_ratio"], pose_index[v].total_frames) for v, m in zip(kept, kept_meta) if m.get("undetected_ratio") is not None]
        report.update({
            "videos_in_split": len(splits.get(split, [])) if split else len(all_ids), "videos_kept": len(kept),
            "segments_kept": len(records), "videos_cut": len(kept) - sum(r.video_id == r.pose.video_id for r in records),
            "dropped_mt_caption": dropped_mt_caption, "dropped_no_caption": dropped_no_caption,
            "dropped_wrong_language": len(dropped_non_latin) + len(dropped_wrong_language), "dropped_scrolling": len(dropped_scrolling),
            "dropped_pose_coverage": len(dropped_uncovered), "pose_coverage_rule_enabled": min_covered > 0.0,
            "deduplicated_train": deduplicated_train, "dropped_all_quarantined": dropped_all_quarantined,
            "frames_kept": sum(r.pose.total_frames for r in records),
            "frames_undetected": int(round(sum(u * f for u, f in known))), "undetected_ratio_known_videos": len(known),
            **{f"{key}_unknown": sum(1 for m in kept_meta if m.get(key) is None) for key in (*RUN_COLUMNS, *QUARANTINE_COLUMNS)},
            # Summed over the SURVIVORS only (see `per_video` above).
            **{k: sum(per_video[v][k] for v in kept) for k in (per_video[kept[0]] if kept else {})},
        })
    return records, splits


class StreamingWindowDataset(Dataset):
    """On-the-fly Stage 2 window dataset. `__getitem__` samples from the training distribution rather than indexing
    a fixed window table, keeping the stochastic sampler inside PyTorch/HF Trainer's map-style interface. 1 epoch
    holds 1 index per anchor, so every reliable sentence anchors exactly 1 row/epoch. A train set turns each index into 1 of its
    `inputs` (dlm.yaml train_inputs, data/sampler.py spec_for); `edge_reach` (the reach of an arm's BIO head modules) masks the edges
    of its windows for the BIO loss. Dev sets draw windows only."""
    def __init__(
        self, records: list[VideoRecord], slt_cfg: dict[str, Any], inference_cfg: dict[str, Any], deterministic: bool = False,
        edge_reach: dict | None = None, inputs=("windows",),
    ):
        if deterministic and (edge_reach is not None or tuple(inputs) != ("windows",)): raise ValueError(
            "dev sets draw windows only, with no edge mask; edge_reach and other inputs are for a train set"
        )
        self.records = records
        self.records_by_id = {record.video_id: record for record in records}
        # Lazy, NOT module-level: data.sampler imports VideoRecord from this module, so a top-level import here is a data.loader
        # <-> data.sampler cycle that breaks every entry point at import time. WindowSampler is only ever used inside methods.
        from data.sampler import WindowSampler
        # Train rows get the shared pose augmentation (dlm.yaml `augmentation`); dev windows never do.
        self.sampler = WindowSampler.from_slt_config(
            records, slt_cfg, inference_cfg, augmentation=None if deterministic else slt_cfg["augmentation"], edge_reach=edge_reach, inputs=inputs,
        )
        self._epoch = 0
        # Eval loaders set deterministic=True: an index then always yields the SAME window (the sampler seeds every draw from the
        # index and no epoch shift applies, and no augmentation), so the early-stopping monitor scores a fixed dev set each epoch.
        self.deterministic = bool(deterministic)

    def __len__(self) -> int:
        return len(self.sampler.anchors)

    def set_epoch(self, epoch: int) -> None:
        if self.deterministic: return
        self._epoch = int(epoch)

    def effective_index(self, index: int) -> int:
        # The index actually handed to the sampler. Training shifts it by whole epochs: the anchor (anchors[index % N]) stays the 
        # same, but the sampler seeds its draws from the index, so each epoch draws new modes and edges. The length pre-pass must 
        # apply the SAME shift or it predicts the wrong window.
        return index if self.deterministic else index + self._epoch * len(self)

    def __getitem__(self, index: int) -> dict:
        index = self.effective_index(index)
        sample = self.sampler.sample(index)   # anchor = anchors[index % N]
        return self.sampler.to_dict(sample)


class LengthBucketSampler(torch.utils.data.Sampler):
    """Batch indices so each batch holds windows of SIMILAR length.

    Step cost is linear in the total frames a batch computes, and a batch is padded to its longest window — so with random batching the 
    median window is computed at the batch maximum.

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


class RankSlice(torch.utils.data.Sampler):
    """This rank's part of every GLOBAL batch that `batches` (a batch sampler at the global batch size) yields: the global batch split
    into world-size parts in order, sizes within 1. So N ranks (torchrun) train on the global batches of 1 process. A last global
    batch with fewer items than ranks is dropped (every rank needs an item); the batch samplers here can make only the last one short.
    A dev set (`keep_short`) keeps that batch: every rank receives it whole, and train.slt.evaluate_slt computes it on rank 0 alone."""
    def __init__(self, batches, num_items: int, batch_size: int, keep_short: bool = False):
        if batch_size < dist.world_size(): raise SystemExit(
            f"batch {batch_size} is a GLOBAL batch and gives no item to some of the {dist.world_size()} ranks: use a batch_size (or "
            f"checkpoint.dev_batch_size) of at least the world size, or fewer ranks.")
        self.batches, self.short, self.keep_short = batches, 0 < num_items % batch_size < dist.world_size(), bool(keep_short)

    def set_epoch(self, epoch: int) -> None:   # the trainer's per-epoch reshuffle reaches the global batch sampler or its sampler
        for sampler in (self.batches, getattr(self.batches, "sampler", None)):
            if hasattr(sampler, "set_epoch"): sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.batches) - int(self.short and not self.keep_short)

    def __iter__(self):
        for batch in self.batches:
            if len(batch) >= dist.world_size(): yield np.array_split(np.asarray(batch), dist.world_size())[dist.rank()].tolist()
            elif self.keep_short: yield list(batch)


def streaming_loader(
    dataset: StreamingWindowDataset, batch_size: int, collate_fn, num_workers: int = 0,
    bucket_by_length: bool = False, bucket_seed: int = 0, split_dev: bool = False,
) -> DataLoader:
    """The ONE DataLoader constructor for the window and chunk datasets (stage 2, S1 and the Moryossef segmenter).

    num_workers is a plain throughput knob: the ANCHOR is a deterministic function of the global sample index (WindowSampler.sample → 
    anchors[index % N]). Every anchor is realized exactly once per epoch however indices are partitioned — no duplication or lost coverage. 
    Every per-window draw is seeded from the index (WindowSampler.spec_for, ChunkDataset.__getitem__), so forked workers need no reseed.

    `batch_size` is the GLOBAL batch. Under torchrun each rank of a training set loads its part of each global batch (RankSlice), so
    N ranks draw the global batches of 1 process. A deterministic (dev) set keeps its order. With `split_dev` (the dev windows of
    train-slt) each rank loads its part of each global dev batch too, and a last batch with fewer items than ranks whole; train.slt.
    evaluate_slt merges the parts in the 1-process order. Otherwise (S1's and the Moryossef segmenter's dev chunks) every rank reads
    all of it at `batch_size`, as 1 process does.
    """
    deterministic = bool(getattr(dataset, "deterministic", False))
    # A dataset whose __getitem__ depends on the epoch cannot use persistent workers.
    epoch_stateful = hasattr(dataset, "set_epoch") and not deterministic
    persistent = num_workers > 0 and not epoch_stateful
    # Length bucketing: same index set, grouped so a batch is not padded to a much longer neighbour.
    bucketed = bucket_by_length and hasattr(dataset, "sampler")
    sampler = None
    # Any other training set is shuffled per epoch by a DistributedSampler with world size 1 (the trainer's set_epoch reseeds it): a 
    # batch is not a run of consecutive anchors from 1 or 2 videos. A deterministic (dev) set keeps its order.
    if not (deterministic or bucketed): sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=int(bucket_seed))
    batches = LengthBucketSampler(dataset, int(batch_size), seed=int(bucket_seed)) if bucketed else None
    if dist.world_size() > 1 and (split_dev or not deterministic):
        batches = RankSlice(batches or BatchSampler(sampler or SequentialSampler(dataset), int(batch_size), drop_last=False),
                            len(dataset), int(batch_size), keep_short=deterministic)
    if batches is not None: return DataLoader(
        dataset, batch_sampler=batches, num_workers=int(num_workers), persistent_workers=persistent, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(), prefetch_factor=4 if num_workers > 0 else None,
    )
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, sampler=sampler, 
        num_workers=int(num_workers), persistent_workers=persistent, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(), prefetch_factor=4 if num_workers > 0 else None,
    )
