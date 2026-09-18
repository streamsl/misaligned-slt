from __future__ import annotations
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass
from typing import Any, Iterable
from pathlib import Path
from tqdm import tqdm

import re, os, random, csv, html, json, unicodedata, zlib, hashlib, inspect
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, get_worker_info
from train import distributed as dist

from data.windowing import SentenceSpan, ANNOTATION_PROTOCOL, TRUSTED_GAP_S
from poses import PoseIndex, build_pose_index
from poses.pose_io import META_FILENAME, base_video_id, load_video_meta

TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{3})"
)
WORD_TIMING_RE = re.compile(r"<\d{1,2}:\d{2}:\d{2}[.,]\d{3}>")
_ASR_WORD_TIMING_RE = re.compile(r"<\d\d:\d\d:\d\d\.\d\d\d>")
_SENTENCE_FINAL_RE = re.compile(r'[.!?][\"\')\]]*\s*$')  # terminal punctuation, allowing closing quotes and brackets
_WORD_RE = re.compile(r"[A-Za-z][\w']*")

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
# Speaker identifier: ONE word (optionally two, e.g. "MRS SMITH:") then a colon, at the cue start. Bounded to a single token so a genuine 
# clause like "One thing: ..." keeps its text — that costs recall on rare speaker labels but never deletes signed content.
SPEAKER_PREFIX_RE = re.compile(r"^[A-Z][A-Z\s_-]{1,30}:\s*")
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
_LANG_RECORDS_CACHE: dict[tuple, list[VideoRecord]] = {}
_FOLD_LEXICON_CACHE: dict[tuple, frozenset[str]] = {}

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
    subtitle_path: Path
    sentences: tuple[SentenceSpan, ...]

def _is_pronoun_i(word: str) -> bool:
    return word == "I" or word.startswith("I'")

def _caption_video_id(path: Path) -> str: # `<vid>.<target>.vtt` -> vid. `Path.stem` strips 1 suffix only, and video ids never contain a dot.
    return base_video_id(path.name.split(".", 1)[0])

def _fold_rule_id() -> str: # The counting rule's own identity, so an edited rule invalidates every lexicon built under the old one.
    return hashlib.sha256(inspect.getsource(fold_lexicon).encode()).hexdigest()

def _lexicon_key(sources: list[tuple[Path, list[str]]], subtitle_cfg: dict) -> str:
    # Identity of a built lexicon: the caption BYTES it reads, the selection rules, and the counting rule itself.
    digest = hashlib.sha256(json.dumps([ANNOTATION_PROTOCOL, subtitle_cfg, _fold_rule_id()], sort_keys=True, default=str).encode())
    for subs, ids in sources:
        want = set(ids)
        # 1 directory glob, not 1 per video: `<vid>.*.vtt` per id is ~2850x slower over a 10k-video corpus.
        for path in sorted(Path(subs).glob("*.vtt")):
            if _caption_video_id(path) in want: digest.update(f"{path.name}\t".encode() + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()

def _train_captions(data_cfg: dict, language: str) -> tuple[Path, list[str]]:
    """(subs root, train ids that HAVE a caption file) without loading records — the id universe `build_pose_index` keys on.

    Filtered by caption presence because the lexicon reads captions: a corpus whose poses are unpacked but whose subs are not would otherwise 
    pass the presence check in `case_lexicon` and contribute nothing, silently changing the rendering it was required for.
    """
    root = Path(data_cfg["languages"][language]["root"])
    ids = sorted({base_video_id(p) for p in (root / "poses").glob("*.npy")})
    if not ids: return root / "subs", []
    captioned = {_caption_video_id(p) for p in (root / "subs").glob("*.vtt")}
    return root / "subs", [v for v in build_splits(ids, data_cfg.get("splits", {})).get("train", []) if v in captioned]

def timestamp_to_seconds(value: str) -> float:
    value = value.replace(",", ".")
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

def fold_lexicon(sources: list[tuple[Path, list[str]]], subtitle_cfg: dict) -> frozenset[str]:
    """Words safe to lowercase at a comma-joined sentence start, from the TRAIN splits' raw cues.

    Capital at sentence start is positional; after the period becomes comma it is noise decoder must learn ("..., Consequently, this ..."). 
    But a capital can also be lexical (a name, "Deaf" as identity, "BSL"), and only the corpus can tell: a word is foldable when it occurs 
    in lowercase more often than it occurs capitalised INSIDE a sentence (Punkt over each raw cue, so sentence-initial capitals are not 
    counted as name evidence). Built on train only, applied to every split, so dev/test references never shape the rule and an unattested 
    word simply keeps its case. `sources` pools the corpora that share a target language — see `case_lexicon`, which owns the pool and the
    released artifact; this builds and nothing else.
    """
    key = tuple((str(subs), tuple(sorted(ids))) for subs, ids in sources)
    if key in _FOLD_LEXICON_CACHE: return _FOLD_LEXICON_CACHE[key]
    tok = _punkt_tokenizer()
    lower: dict[str, int] = {}; cap_inside: dict[str, int] = {}
    drop_noise = bool(subtitle_cfg.get("drop_noise_captions", True))

    for subs, ids in sources:
        for vid in ids:
            path = best_subtitle(subs, vid, subtitle_cfg)
            if path is None: continue

            for _, _, text in merge_rolling_captions(parse_vtt(path, drop_noise=drop_noise)):
                sentences = [text[a:b] for a, b in tok.span_tokenize(text)] if tok is not None else [text]
                for sentence in sentences:
                    for i, word in enumerate(_WORD_RE.findall(sentence)):
                        if word[0].islower(): lower[word.lower()] = lower.get(word.lower(), 0) + 1
                        elif i > 0: cap_inside[word.lower()] = cap_inside.get(word.lower(), 0) + 1

    lexicon = frozenset(w for w, n in lower.items() if n > cap_inside.get(w, 0))
    _FOLD_LEXICON_CACHE[key] = lexicon
    return lexicon


def case_lexicon(data_cfg: dict, language: str) -> frozenset[str]:
    """The fold lexicon for `language`'s TARGET language, pooled over every configured corpus with that target.

    Capitalisation convention belongs to the target language, not to the sign corpus: asf's train captions never lowercase "consequently", 
    ase's do. Pooling measurably renders better where the corpora disagree, and it is what keeps the small corpora's identity capitals
    ("Deaf", "God", "Bible"), which a corpus-sized rule folds away.
    """
    langs = data_cfg["languages"]
    target = str(langs[language].get("target_lang", "en_XX"))
    pool = sorted(l for l, c in langs.items() if str(c.get("target_lang", "en_XX")) == target)
    artifact = Path(langs[language]["root"]).parent / f"case_lexicon.{target}.json"
    try: stored = json.loads(artifact.read_text(encoding="utf-8")) if artifact.exists() else {}
    except ValueError: stored = {}

    sources, absent = [], []
    for other in pool:
        subs, ids = _train_captions(data_cfg, other)
        if ids: sources.append((subs, ids))
        else: absent.append(other)

    if absent:
        if stored.get("pool") == pool and stored.get("rule") == _fold_rule_id(): return frozenset(stored["words"])
        raise FileNotFoundError(
            f"[loader] the case lexicon for target {target} pools the train captions of {', '.join(pool)}; this machine has none for "
            f"{', '.join(absent)}. Copy {artifact} from the machine that holds the whole pool, or fetch the corpus "
            f"(prepare_data.py --languages {' '.join(absent)}), or drop it from data.yaml languages and re-render everything."
        )
    key = _lexicon_key(sources, data_cfg.get("subtitles", {}))
    if stored.get("key") == key: return frozenset(stored["words"])
    lexicon = fold_lexicon(sources, data_cfg.get("subtitles", {}))
    artifact.parent.mkdir(parents=True, exist_ok=True)
    
    # Atomic: every rank of a distributed run builds this, and a half-written artifact is a rebuild for everyone after.
    tmp = artifact.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"key": key, "pool": pool, "rule": _fold_rule_id(), "words": sorted(lexicon)}) + "\n", encoding="utf-8")
    os.replace(tmp, artifact)
    return lexicon


def _is_prose(cues: list[tuple], joined: str, duration_s: float) -> bool:
    # Is a multi-cue chain wrapped prose, so that Punkt finding no boundary across its junctions is evidence?
    words = _WORD_RE.findall(joined)
    if not words or duration_s <= 0: return False
    if len(words) / duration_s < PROSE_MIN_WORDS_PER_S: return False
    words_per_cue = sum(len(_WORD_RE.findall(text)) for _, _, text in cues) / len(cues)
    capitalised = sum(1 for w in words if w[0].isupper()) / len(words)
    return not (words_per_cue <= GLOSS_MAX_WORDS_PER_CUE and capitalised >= GLOSS_MIN_CAPITALISED_SHARE)


def _fold_sentence_start(part: str, lexicon: frozenset[str] | None) -> str:
    # Lowercase 1st word of an interior sentence when the corpus says its capital was positional.
    if not lexicon: return part
    m = _WORD_RE.search(part)
    if m is None: return part
    word = m.group(0)
    # An ALL-CAPS token carries no positional capital to remove: it is a gloss or emphasis, and folding only its first
    # letter renders "BYE" as "bYE". Also skip I / I'm and fingerspelled single letters.
    if not word[0].isupper() or len(word) == 1 or word.isupper() or _is_pronoun_i(word): return part
    nxt = _WORD_RE.search(part, m.end())
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


def _clamp_overlaps(captions: list[tuple[float, float, str]]) -> tuple[list[tuple[float, float, str]], int, int]:
    """Force a strictly non-overlapping, time-ordered span sequence. Returns (spans, clamped, swallowed).

    Applied to the FINAL caption stream (after rolling-duplicate merging and sentence reconstruction), so it is the
    1 place that guarantees the invariant every downstream consumer assumes. A span left fully inside its predecessor 
    after clamping is dropped: it carries no exclusive frames, so it can never be selected or scored.

    Both counts are per UNIT and disjoint — 1 unit is either moved or deleted — so neither can go negative and their
    sum is the number of units an overlap touched. A count derived instead from ADJACENT-PAIR overlaps is not on this
    basis: 1 malformed cue containing 5 short ones is a single adjacent pair and five deletions.
    """
    out: list = []
    clamped = swallowed = 0
    prev_end = float("-inf")
    # Tuples may carry a 4th `reliable` field (unsupported coverage) — pass any extra fields through untouched.
    for c in sorted(captions, key=lambda x: (x[0], x[1])):
        s, e = max(c[0], prev_end), c[1]
        if e <= s:
            swallowed += 1  # wholly swallowed by the previous span; its TEXT leaves the reference with it
            continue
        clamped += s > c[0]
        out.append((s, e, *c[2:])); prev_end = e
    return out, clamped, swallowed


def _quarantine_end_straddlers(captions: list[tuple], duration_s: float, slack_s: float = 1.0) -> list[tuple]:
    """Captions that START inside pose stream but END well past it become QUARANTINED spans over visible frames, instead of being dropped.

    Streams end before the caption timeline (duration = frames/24 underestimates the video), so this straddle is systematic, not an edge 
    case. Dropping caption relabels its visible frames as uncaptioned — and a leftover tail gap of trusted_gap_s or less is then supervised 
    as trusted `O` over frames the caption says are signing. Clipping to `duration_s` as a RELIABLE span would instead mint an end boundary 
    no cue marks, with text for signing partly outside the poses. Marking the coverage unsupported (reliable=False) is the one honest option: 
    frames UNK, never an anchor, reference, or Mode-4 gap. Ends within `slack_s` of the stream end are left alone — the existing span filter 
    tolerates them, and their label error is below the timestamp noise floor.
    """
    return [(c[0], float(duration_s), c[2], False) if (c[0] < duration_s and c[1] > duration_s + float(slack_s)) else c for c in captions]


def reconstruct_sentences(captions: list[tuple[float, float, str]], max_tokens: int = 60, fold: frozenset[str] | None = None) -> list[tuple]:
    """Group whole display cues into timestamp-supported caption units.

    Punkt reads the original text to find cue junctions inside linguistic sentences. Connected cues form 1 unit; a unit can contain several 
    linguistic sentences. Its BIO labels describe unit membership, not each linguistic sentence. Only after grouping, internal sentence-final 
    periods become commas, and the sentence that follows a converted period loses its positional capital when the train lexicon `fold` says 
    the word is ordinarily lowercase (see `fold_lexicon`; names, "I", initialisms and proper-noun compounds keep theirs). Questions and 
    exclamations are retained, and the sentence after them keeps its capital.

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
    crossed: set[int] = set()                                 # junction k = boundary between cue k and k+1
    for a, b in tok.span_tokenize(text):
        ids = sorted({c for c in char_cue[a:b] if c != -1})
        for x, y in zip(ids, ids[1:]):
            # Text evidence alone is not enough to assert membership: the unit is timestamp-SUPPORTED, so a junction
            # the labeller distrusts cannot be crossed. `untrusted_o_intervals` calls an uncaptioned stretch longer
            # than TRUSTED_GAP_S unlocatable and marks it UNK; merging across the same stretch would relabel it I.
            if y == x + 1 and captions[y][0] - captions[x][1] <= TRUSTED_GAP_S: crossed.add(x)
    out: list[tuple] = []

    def emit(run: list[int]) -> None:
        s0, e1 = captions[run[0]][0], captions[run[-1]][1]
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


def marked_boundary_ratio(units, fold: frozenset[str] | None = None) -> float:
    """Share of a video's UNIT boundaries the caption author marked, read AFTER grouping.

    A boundary is marked when the unit ends in sentence punctuation, or the next unit opens with a POSITIONAL capital. Terminal punctuation alone 
    is a punctuation-STYLE test: over the three corpora 37 % (asf), 73 % (ase) and 69 % (bfi) of boundaries carrying no period are followed by a 
    capital, so a period-only rule rejects song lyrics and capital-marked prose whose boundaries are real.

    A capital is POSITIONAL only when the word is ordinarily lowercase, which is what the train `fold` lexicon records (`fold_lexicon`, same test 
    `_fold_sentence_start` applies). "NDIS", "Auslan" and "David" carry a LEXICAL capital and say nothing about a boundary: a wrap that lands before 
    a proper noun would otherwise read as a sentence start. Without a lexicon only the punctuation half is available, and an all-caps unit is never
    evidence, because in a gloss list every line opens with a capital.

    Read after grouping since that is where gold boundaries live. On raw cues a continuous unpunctuated narration scores near 0 although grouping 
    fuses it into 1 unit without internal boundary at all. What this leaves is the channel whose line breaks fall mid-phrase: the annotation marks 
    no boundary of either kind.
    """
    units = list(units)
    if len(units) < 2: return 1.0

    def positional_capital(text: str) -> bool:
        word = text.strip().split(" ")[0] if text.strip() else ""
        if not word[:1].isupper() or word.isupper(): return False
        return fold is not None and word.strip(".,;:!?\"')]").lower() in fold

    marked = sum(1 for a, b in zip(units, units[1:]) if _SENTENCE_FINAL_RE.search(a[2]) or positional_capital(b[2]))
    return marked / (len(units) - 1)


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

    A caption identifies content only if it is RARE (document frequency <= df_cap; above that it is series boilerplate such as a scripted 
    contact-info outro, which otherwise chains unrelated videos into one cluster) and LONG enough (>= min_words; single-sign vocabulary 
    clips share a word with everything). A pair is flagged when its shared identifying captions cover more than `cover` of the smaller 
    video's identifying set. Near-match coverage needs min_captions. Exact caption-set copies can instead meet same evidence budget in words.
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

    scores = {(a, b): n / min(len(ident[a]), len(ident[b])) for (a, b), n in shared.items()}
    exact: dict[frozenset[str], list[str]] = defaultdict(list)
    for vid, cs in caps.items():
        if sum(len(c.split()) for c in cs) >= min_caps * min_words: exact[frozenset(cs)].append(vid)
    for vids in exact.values():
        if 2 <= len(vids) <= df_cap:
            for pair in combinations(sorted(vids), 2): scores[pair] = 1.0
    return sorted([(a, b, score) for (a, b), score in scores.items() if score > cover], key=lambda p: -p[2])


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
        {"lang": (data_cfg.get("languages") or {}).get(language), "subtitles": data_cfg.get("subtitles"),
         "splits": data_cfg.get("splits"), "poses": data_cfg.get("poses")},
        sort_keys=True, default=str,
    )
    # The active loader function is part of the key: tests monkeypatch `load_language_records`, and a key of
    # (language, split, fingerprint) alone would serve one test's fake records to the next.
    key = (language, str(split), lang_fingerprint, load_language_records)
    if key not in _LANG_RECORDS_CACHE: _LANG_RECORDS_CACHE[key] = load_language_records(data_cfg, language, split=split)[0]
    return _LANG_RECORDS_CACHE[key]


def sentence_p99_s(data_cfg: dict, languages: list[str], split: str = "train") -> dict[str, float]:
    # Per-language p99 of reliable sentence durations: the label-only cap statistic. Shares the pool's record cache.
    out: dict[str, float] = {}
    for lang in languages:
        d = [sp.duration_s for r in _cached_language_records(data_cfg, lang, split) for sp in r.sentences if getattr(sp, "reliable", True)]
        out[lang] = float(np.percentile(d, 99)) if d else 0.0
    return out


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
    for lang in languages: per_lang[lang] = _cached_language_records(data_cfg, lang, split)

    # A configured language that contributes nothing means this machine lacks that corpus. Skipping it would train a
    # DIFFERENT model that still stamps the configured pool key, and nothing downstream compares the realised mix.
    empty = sorted(l for l, recs in per_lang.items() if not recs)
    if empty: raise FileNotFoundError(
        f"pretrain_languages lists {list(languages)}, but {', '.join(empty)} has no {split} record on this machine. "
        f"Fetch the corpus (prepare_data.py --languages {' '.join(empty)}), or drop it from pretrain_languages."
    )
    if len(per_lang) == 1 or split == "test": # TEST is pooled AS-IS: it is a REPORTING set.
        pooled = [r for recs in per_lang.values() for r in recs]
        return pooled, {k: len(v) for k, v in per_lang.items()}

    counts = {k: len(v) for k, v in per_lang.items()}
    weights = {k: n ** float(temperature) for k, n in counts.items()}
    total_w = sum(weights.values())
    # SUB-sample to target shares: pick the pool size that the most over-represented language can support WITHOUT replication, i.e. largest 
    # total for which every target <= that language's real video count. Upsampling instead (scaling up to biggest corpus) repeats the small 
    # corpora several times inside ONE epoch, so the model sees many epochs' worth of a language before the first checkpoint and overfits 
    # during epoch 1 — the failure this bound exists to prevent. Temperature then only sets the SHARES, never the repetition, so lowering 
    # it rebalances instead of inflating the epoch.
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


def load_language_records(
    data_cfg: dict, language: str, split: str | None = None, report: dict | None = None,
) -> tuple[list[VideoRecord], dict[str, list[str]]]:
    """`report`, when given, is filled with the per-rule tallies of this load (videos and cues kept or dropped by
    each pipeline rule) — the corpus audit `report.py data` prints. Every consumer of the records is unaffected."""
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
    # MULTI-PERSON videos: 2+ people on screen AT SAME TIME. Converter keeps person_000 = largest body per frame, so the caption may follow 
    # a person the stored pose is not, and 2nd body on screen is exactly what the detector can lose track between. Signers appearing 1 AFTER 
    # the other are not this. BOTH conditions must hold. Low measured shape variation exempts those extra detections.
    max_multi = float((data_cfg.get("poses", {}) or {}).get("max_multi_person_ratio", 1.0))
    min_motion = float((data_cfg.get("poses", {}) or {}).get("min_extra_person_motion", 0.0))
    # A high undetected share means little pose evidence is available for the captions. It can reflect a screen
    # recording, or a detector missing a real signer; the filter does not distinguish these causes.
    max_undetected = float((data_cfg.get("poses", {}) or {}).get("max_undetected_ratio", 1.0))
    min_covered = float((data_cfg.get("poses", {}) or {}).get("min_unit_coverage", 0.0))
    dropped_multi_person: list[tuple[str, float]] = []
    dropped_undetected: list[tuple[str, float]] = []
    dropped_uncovered: list[tuple[str, float]] = []
    if max_multi < 1.0:
        considered = len(selected_ids)
        unknown = [v for v in selected_ids if (video_meta.get(v) or {}).get("multi_person_ratio") is None]
        over = {v for v in selected_ids if ((video_meta.get(v) or {}).get("multi_person_ratio") or 0.0) > max_multi
                                        and ((video_meta.get(v) or {}).get("extra_person_motion") is None
                                        or video_meta[v]["extra_person_motion"] > min_motion)}
        unmeasured = sum(video_meta[v].get("extra_person_motion") is None for v in over)
        dropped_multi_person.extend((v, float(video_meta[v]["multi_person_ratio"])) for v in selected_ids if v in over)
        selected_ids = [v for v in selected_ids if v not in over]
        if unmeasured: print(
            f"[loader] {language}/{split or 'all'}: {unmeasured} multi-person exclusions have no measurable extra-person "
            f"shape; the count rule applies. Run `prepare_data.py --stage person-counts` if that column is missing.", flush=True
        )
        if unknown: print(
            f"[loader] {language}/{split or 'all'}: WARNING the multi-person rule is on but {len(unknown)}/{considered} "
            f"videos lack multi_person_ratio in video_meta.csv; run `prepare_data.py --stage person-counts` "
            f"to measure them. Those videos are NOT filtered.", flush=True
        )
    for key, limit, dropped in (("undetected_ratio", max_undetected, dropped_undetected),):
        if limit >= 1.0: continue
        considered = len(selected_ids)   # the denominator: an unknown video stays in selected_ids, so adding 2 double-counts it
        unknown = [v for v in selected_ids if (video_meta.get(v) or {}).get(key) is None]
        over = {v for v in selected_ids if ((video_meta.get(v) or {}).get(key) or 0.0) > limit}
        dropped.extend((v, float(video_meta[v][key])) for v in selected_ids if v in over)
        selected_ids = [v for v in selected_ids if v not in over]
        if unknown: print(
            f"[loader] {language}/{split or 'all'}: WARNING poses.max_{key.replace('_ratio', '')}_ratio is set but "
            f"{len(unknown)}/{considered} videos have no {key} in video_meta.csv; run `prepare_data.py --stage person-counts` "
            f"to measure it. Those videos are NOT filtered.", flush=True
        )
    records: list[VideoRecord] = []
    dropped_no_caption, dropped_all_quarantined = 0, 0
    per_video: dict[str, dict[str, int]] = {}
    fold = case_lexicon(data_cfg, language) if subtitle_cfg.get("merge_sentences") else None

    # 1.0 disables the filter (no video can exceed a full share). The key is GLOBAL under `subtitles:`, which is correct while every 
    # corpus targets English; non-Latin-target corpus need a per-language override which would switch the filter off for English corpora.
    max_non_latin = float(subtitle_cfg.get("max_non_latin_ratio", 1.0))
    dropped_non_latin: list[tuple[str, float]] = []
    drop_scrolling = bool(subtitle_cfg.get("drop_scrolling_tracks", True))
    dropped_scrolling: list[tuple[str, float]] = []
    for video_id in tqdm(selected_ids, desc=f"[loader] {language}/{split or 'all'}", unit="vid", leave=False, dynamic_ncols=True):
        subtitle_path = best_subtitle(root / "subs", video_id, subtitle_cfg)
        if subtitle_path is None:
            dropped_no_caption += 1
            continue
        source_cues = parse_vtt(subtitle_path, drop_noise=drop_noise)
        # A SCROLLING display (more than half the cues start before previous one ends: YouTube's rolling two-line auto-captions) encodes 
        # DISPLAY times at both ends of every cue: a cue leaves the screen when line 2 later arrives, so its end runs a median 1.9-2.6 s 
        # (p90 3.0-3.9 s) past the next cue's start — 5-10x delta_enc. Punctuation can fix the grouping of such cues into units; it cannot 
        # move their timestamps, so the whole track is dropped, not repaired (bfi/0cw4rELLAtc, ase/T9C4QbZ7qOs).
        if drop_scrolling and is_scrolling_display(source_cues):
            n = len(source_cues)
            dropped_scrolling.append((video_id, sum(1 for a, b in zip(source_cues, source_cues[1:]) if b[0] < a[1] - 1e-6) / max(1, n - 1)))
            continue
        raw_cues = merge_rolling_captions(source_cues)
        captions = raw_cues
        if subtitle_cfg.get("merge_sentences"):  # group display-wrapped cues into caption units (Punkt over the caption stream)
            captions = reconstruct_sentences(captions, fold=fold)
        min_dur = float(subtitle_cfg.get("min_duration_s", 0.2))
        # Caption times refer to the source video. The supplied pose timeline can be shorter; a caption onset
        # outside it has no pose evidence. This comparison does not identify why the source poses stop.
        dur = pose_index[video_id].duration_s
        # Some source VTTs ship genuinely OVERLAPPING cues with distinct text. merge_rolling_captions only fuses overlapping DUPLICATES, 
        # so these survive, and overlapping SentenceSpans corrupt BIO labels — a neighbour's `I` overwrites the closing `O`, and 
        # first_complete_span becomes ill-defined. Clamp each start to the previous end: it trims the disputed frames from the LATER
        # sentence (whose onset is the less certain of the two) and never invents a boundary.
        ordered = sorted(captions, key=lambda c: (c[0], c[1]))
        # Measure temporal coverage before any text removal or overlap repair. Otherwise a short or textless
        # unit would count as missing pose data. Use the same one-second end tolerance as the span filter.
        covered = sum(0 <= c[0] < dur and c[1] <= dur + 1.0 for c in ordered) / max(1, len(ordered))
        # Quarantine unsupported ends before overlap repair so they remain untrusted supervision.
        captions = _quarantine_end_straddlers(captions, dur)
        captions, clamped, swallowed = _clamp_overlaps(captions)
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
        # 3 independent reasons a unit cannot be supervision, attributed to the FIRST that fires so the rows sum to the drop: shorter than 
        # minimum unit, outside the pose stream, or carrying no letters at all (a cue of digits or punctuation is not a translation target).
        kept, too_short, outside_poses, no_text = [], 0, 0, 0
        for c in captions:
            if not min_dur <= (c[1] - c[0]): too_short += 1; continue
            if not (c[0] < dur and c[1] <= dur + 1.0): outside_poses += 1; continue
            if not any(ch.isalpha() for ch in c[2]): no_text += 1; continue
            kept.append(SentenceSpan(video_id=video_id, start_s=c[0], end_s=c[1], text=c[2], reliable=bool(c[3]) if len(c) > 3 else True))
        spans = tuple(kept)
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
        records.append(VideoRecord(language, video_id, pose_index[video_id], subtitle_path, spans))
        per_video[video_id] = {
            "source_cues": len(source_cues), "rolling_duplicates_merged": len(source_cues) - len(raw_cues),
            "units_after_grouping": len(ordered), "overlaps_swallowed": swallowed, "overlaps_clamped": clamped,
            "end_straddlers_quarantined": sum(1 for c in captions if len(c) > 3 and not c[3]), "spans_kept": len(spans), 
            "spans_dropped_too_short": too_short, "spans_dropped_outside_poses": outside_poses, "spans_dropped_no_text": no_text,
        }
    if dropped_uncovered: print(
        f"[loader] {language}/{split or 'all'}: {len(dropped_uncovered)} video(s) dropped as POSE-TRUNCATED "
        f"(under {min_covered:.0%} of their caption units fall inside the supplied pose timeline; e.g. "
        + ", ".join(f"{v} {c:.0%}" for v, c in sorted(dropped_uncovered, key=lambda x: x[1])[:3]) + "); poses.min_unit_coverage.", flush=True
    )
    if dropped_multi_person: print(
        f"[loader] {language}/{split or 'all'}: {len(dropped_multi_person)} video(s) dropped by MULTI-PERSON rule (2+ detected bodies in >"
        f"{max_multi:.0%} of frames, extra-slot shape variation >{min_motion:g} or unmeasured; e.g. "
        + ", ".join(f"{v} {r:.0%}" for v, r in sorted(dropped_multi_person, key=lambda x: -x[1])[:3])
        + "); poses.max_multi_person_ratio / min_extra_person_motion.", flush=True
    )
    if dropped_undetected: print(
        f"[loader] {language}/{split or 'all'}: {len(dropped_undetected)} video(s) dropped as LOW-DETECTION (no pose detected in >"
        f"{max_undetected:.0%} of frames; e.g. " + ", ".join(f"{v} {r:.0%}" for v, r in sorted(dropped_undetected, key=lambda x: -x[1])[:3])
        + "); poses.max_undetected_ratio.", flush=True
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
    deduplicated_train = 0
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
            deduplicated_train = len(drop)
    if report is not None:
        kept_meta = [video_meta.get(r.video_id) or {} for r in records]
        frames = [r.pose.total_frames for r in records]
        known = [(m["undetected_ratio"], f) for m, f in zip(kept_meta, frames) if m.get("undetected_ratio") is not None]
        report.update({
            "videos_in_split": len(splits.get(split, [])) if split else len(pose_index), "videos_kept": len(records),
            "dropped_mt_caption": dropped_mt_caption, "dropped_no_caption": dropped_no_caption,
            "dropped_wrong_language": len(dropped_non_latin), "dropped_scrolling": len(dropped_scrolling),
            "dropped_multi_person": len(dropped_multi_person), "dropped_undetected": len(dropped_undetected),
            "dropped_pose_coverage": len(dropped_uncovered), "pose_coverage_rule_enabled": min_covered > 0.0,
            "deduplicated_train": deduplicated_train, "dropped_all_quarantined": dropped_all_quarantined,
            "multi_person_ratio_known": sum(1 for m in kept_meta if m.get("multi_person_ratio") is not None),
            "multi_person_rule_enabled": max_multi < 1.0,
            # Raw detector counts among kept videos; these do not include the shape-variation condition.
            **_multi_person_band([m.get("multi_person_ratio") for m in kept_meta]),
            "frames_kept": int(sum(frames)), "frames_undetected": int(round(sum(r * f for r, f in known))),
            "undetected_ratio_known_videos": len(known), "undetected_rule_enabled": max_undetected < 1.0,
            # Summed over the SURVIVORS only (see `per_video` above).
            **{k: sum(per_video[r.video_id][k] for r in records) for k in next(iter(per_video.values()), {})},
        })
    return records, splits


def _multi_person_band(ratios: list) -> dict: # Raw detected-body share over kept videos; `videos_above` doesn't apply the shape test.
    known = sorted(r for r in ratios if r is not None)
    if not known: return {}
    pct = lambda q: float(known[min(len(known) - 1, int(q * (len(known) - 1)))])
    return {
        "multi_person_p50": pct(0.5), "multi_person_p90": pct(0.9), "multi_person_p99": pct(0.99), "multi_person_max": float(known[-1]),
        "multi_person_videos_above": {f"{t:g}": sum(1 for r in known if r > t) for t in (0.0, 0.01, 0.05, 0.25, 0.5)},
    }


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
            # CB must compare 2 views of the same complete anchor, after actual frame materialization.
            item["full_evidence"] = self.sampler.to_dict(full) if full.translation_target == sample.anchor_span else None
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
