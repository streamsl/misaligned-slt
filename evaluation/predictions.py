"""Predicted and gold events and their files.

An event is 1 timed unit with its text, on source-video timeline. This module maps a stream's clock to its source video, builds the reliable 
gold and other inputs of the scores of a row (gold chunks of edge chunk score, UNK stretches of long-form score, scoreable events of segmenter spans), 
applies the quarantine rule of scoring, turns runner and translator outputs into events, writes events files, reads events files and 
span files with their annotation stamps, and makes the artifact names of the rows. Every metric calculation is in `metrics.py`.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
import json

from data.loader import (WORD_RE, annotation_fingerprint, VideoRecord, best_subtitle, corpus_config,
                         merge_rolling_captions, parse_vtt, sorted_names)
from data.windowing import TRUSTED_GAP_S, untrusted_o_intervals
from infer.stream import join_unit_text
from metrics import Segment

CHUNK_CLASSES = ("over_cap", "video_start", "cut_start", "after_quarantine", "end_blocked", "free")
BOUNDARY_CLASSES = ("abutting", "pause", "quarantine", "stream")   # the boundary at a gold unit edge (gold_edges)

@dataclass(frozen=True)
class PredictionEvent:
    video_id: str
    start_s: float
    end_s: float
    text: str | None = None
    flagged_partial: bool = False
    commit_time_s: float | None = None  # stride time the commit fired (streaming only)
    # The engine's cap cut this span open at the cap (infer.stream.StreamingEvent.cap_cut; online rows, and a cascade of their spans). 
    # Last field, so a positional construction of the fields above keeps its meaning.
    cap_cut: bool = False

    @property
    def segment(self) -> Segment:
        return Segment(self.start_s, self.end_s)


def translated_events(video_id: str, spans, results) -> list[PredictionEvent]:
    # Translated spans -> 1 event per span (never cut). A text with UNIT_SEPARATOR keeps its unit texts, joined with spaces
    # (infer.stream.join_unit_text).
    return [PredictionEvent(video_id=video_id, start_s=span.start_s, end_s=span.end_s, text=join_unit_text(text))
            for span, (text, _, _) in zip(spans, results)]

def to_source(events: dict[str, list[PredictionEvent]], records: list[VideoRecord]) -> dict[str, list[PredictionEvent]]:
    """Record-keyed events on each stream's local clock -> the source video id and source time. A cut video is 1 stream per clean segment 
    (data.loader.split_clean_segments). Every artifact and every score is per source video in absolute time. Records come in time order 
    inside a video, so the concatenation keeps each video's events in time order."""
    out: dict[str, list[PredictionEvent]] = {}
    for r in records:
        off = r.pose.offset_s
        out.setdefault(r.pose.video_id, []).extend(replace(
            ev, video_id=r.pose.video_id, start_s=ev.start_s + off, end_s=ev.end_s + off,
            commit_time_s=None if ev.commit_time_s is None else ev.commit_time_s + off,
        ) for ev in events.get(r.video_id, []))
    return out

def drop_quarantined_predictions(predicted: dict[str, list], records: list[VideoRecord]) -> dict[str, list]:
    """Ignore-region protocol, prediction side, on the source timeline. Gold already excludes quarantined spans (reliable=False). 
    A prediction majority-inside 1 must not count as unmatched either — the region is unscoreable, not empty. Majority-overlap, not 
    any-overlap: a span merely grazing a quarantine edge is still a real, scoreable event. A unit that crosses a cut is in every 
    segment it touches, so each copy is clipped to its own segment first; unclipped, the copies would count the same time twice."""
    zones: dict[str, list[tuple[float, float]]] = {}
    for r in records:
        off, dur = r.pose.offset_s, r.pose.duration_s
        zones.setdefault(r.pose.video_id, []).extend(
            (max(0.0, s.start_s) + off, min(dur, s.end_s) + off) for s in r.sentences if not getattr(s, "reliable", True)
        )
    out: dict[str, list] = {}
    for vid, events in predicted.items():
        keep = []
        for ev in events:
            dur = max(1e-9, float(ev.end_s) - float(ev.start_s))
            inside = sum(max(0.0, min(ev.end_s, b) - max(ev.start_s, a)) for a, b in zones.get(vid, []))
            if inside / dur <= 0.5: keep.append(ev)
        out[vid] = keep
    return out

def scoreable_predictions(
    predicted: dict[str, list[PredictionEvent]], records: list[VideoRecord], tag: str = "row",
) -> dict[str, list[PredictionEvent]]:
    # Model duration policy belongs to event generation. Evaluation keeps short true and false predictions alike.
    n0 = sum(len(v) for v in predicted.values())
    kept = drop_quarantined_predictions(predicted, records)
    n1 = sum(len(v) for v in kept.values())
    print(f"[{tag}] scoreable events: {n1} of {n0} (dropped {n0 - n1} majority-inside quarantine)", flush=True)
    return kept

def gold_events(records: list[VideoRecord]) -> dict[str, list[PredictionEvent]]:
    # Reliable units per SOURCE video in source time; every source video of the split has a key.
    return to_source({record.video_id: [PredictionEvent(
        video_id=record.video_id, start_s=float(span.start_s), end_s=float(span.end_s), text=span.text,
    ) for span in record.sentences if getattr(span, "reliable", True)] for record in records}, records)

def gold_chunks(records: list[VideoRecord], gap_s: float) -> dict[str, list[list[PredictionEvent]]]:
    """{source video: its chunks, each a list of reliable gold units in time order}, in source time. A chunk is the reliable units
    of one stream that follow each other with caption gaps of at most `gap_s` (eval.yaml scoring.chunk_gap_s): a quarantined unit ends
    it, and no chunk crosses a stream edge. Every reliable unit of `gold_events` is in exactly one chunk."""
    out: dict[str, list[list[PredictionEvent]]] = {}
    for rec in records:
        off, chunks, chunk = rec.pose.offset_s, out.setdefault(rec.pose.video_id, []), []
        for u in sorted(rec.sentences, key=lambda u: u.start_s):
            reliable = getattr(u, "reliable", True)
            if chunk and (not reliable or u.start_s + off - chunk[-1].end_s > gap_s): chunks.append(chunk); chunk = []
            if reliable: 
                chunk.append(PredictionEvent(video_id=rec.pose.video_id, start_s=u.start_s + off, end_s=u.end_s + off, text=u.text))
        if chunk: chunks.append(chunk)
    return out

def gold_edges(records: list[VideoRecord], gap_s: float) -> dict[str, list[tuple[str, float, str]]]:
    """{source video: [(side, time, class)]}: the start and the end of every reliable unit of `gold_events`, in source time, each with
    the class of its boundary (BOUNDARY_CLASSES), read from the nearest unit on that side in the same stream, reliable or quarantined:
    before a start, the unit that ends last of those that start earlier; after an end, the next unit to start. The gap is the later
    start minus the earlier end (an overlap is a negative gap). abutting: that unit is reliable and the gap is at most `gap_s`
    (eval.yaml scoring.chunk_gap_s), so the edge is inside a chunk of `gold_chunks`; quarantine: it is quarantined and the gap is at
    most `gap_s`; pause: a longer gap, or no unit on that side and the stream edge more than `gap_s` away; stream: no unit on that side
    and the stream edge within `gap_s`. Input of `metrics.edge_boundary_hits`."""
    kind = lambda w, gap: "pause" if gap > gap_s else "stream" if w is None else "abutting" if getattr(w, "reliable", True) else "quarantine"
    out: dict[str, list[tuple[str, float, str]]] = {}
    for rec in records:
        off, edges, last = rec.pose.offset_s, out.setdefault(rec.pose.video_id, []), None   # last: the unit that ends last so far
        units = sorted(rec.sentences, key=lambda u: u.start_s)
        for k, u in enumerate(units):
            if getattr(u, "reliable", True):
                after = units[k + 1] if k + 1 < len(units) else None
                # a gap to a unit from the source-time floats that gold_chunks compares (a stream-time gap of exactly gap_s can round
                # to the other side of it); a gap to the stream edge in stream time
                edges += [("start", u.start_s + off, kind(last, u.start_s + off - (last.end_s + off) if last else u.start_s)),
                          ("end", u.end_s + off, kind(after, after.start_s + off - (u.end_s + off) if after else rec.pose.duration_s - u.end_s))]
            if last is None or u.end_s > last.end_s: last = u
    return out

def chunk_classes(records: list[VideoRecord], gap_s: float, max_span_s: float) -> dict[str, list[str]]:
    """{source video: 1 class per chunk of `gold_chunks(records, gap_s)`, in its order}: 1st rule of CHUNK_CLASSES that fits. over_cap: 
    the chunk is longer than `max_span_s` (inference.yaml buffer_cap_s - boundary_stability.delta_s, the longest span that the engine 
    commits whole, with a decoded start and terminator; online, a longer span can commit only as a cap-cut closed part, which ends at the 
    cap-cut frame, not at a unit edge; the offline rows keep every span whole). video_start: it starts within `gap_s` of a stream start that is frame 0 of the source video. cut_start: it starts within 
    `gap_s` of a stream start made by a cut (an empty run), or within `gap_s` after a quarantined unit that crosses the stream 
    start. after_quarantine: it starts within `gap_s` after another quarantined unit. end_blocked: it ends within `gap_s` of the stream 
    end or before a quarantined unit. free: a pause at both edges. Input of `metrics.edge_chunk_classes`."""
    out: dict[str, list[str]] = {}
    for rec in records:
        off, dur = rec.pose.offset_s, rec.pose.duration_s
        quarantined = [u for u in rec.sentences if not getattr(u, "reliable", True)]
        for chunk in gold_chunks([rec], gap_s)[rec.pose.video_id]:
            a, b = chunk[0].start_s - off, chunk[-1].end_s - off   # stream time
            after = lambda crossing: any((u.start_s < 0) == crossing and 0 <= a - u.end_s <= gap_s for u in quarantined)
            out.setdefault(rec.pose.video_id, []).append(
                "over_cap" if b - a > max_span_s else
                "video_start" if rec.pose.start_frame == 0 and a <= gap_s else
                "cut_start" if (rec.pose.start_frame > 0 and a <= gap_s) or after(True) else
                "after_quarantine" if after(False) else
                "end_blocked" if dur - b <= gap_s or any(0 <= u.start_s - b <= gap_s for u in quarantined) else "free"
            )
    return out

def cue_breaks(records: list[VideoRecord], data_cfg: dict, language: str) -> dict[str, list[tuple[float, float]]]:
    """{source video: [(end of display cue k, start of cue k + 1)]}: every junction between 2 consecutive display cues of the video's
    caption track, in source time (a VTT time is a source time). The cues are the loader's own (`best_subtitle`, `parse_vtt` with
    subtitles.drop_noise_captions, `merge_rolling_captions`), which `reconstruct_sentences` groups into caption units, so a unit of
    several cues holds the junctions between them. Input of `metrics.cue_edge_share`."""
    subtitle_cfg = corpus_config(data_cfg, language)["subtitles"]   # the corpus's own subtitle track
    root = Path(data_cfg["languages"][language]["root"]) / "subs"
    out: dict[str, list[tuple[float, float]]] = {}
    names = sorted_names(root)
    for vid in dict.fromkeys(r.pose.video_id for r in records):
        path = best_subtitle(root, vid, subtitle_cfg, names=names)
        cues = merge_rolling_captions(parse_vtt(path, drop_noise=bool(subtitle_cfg["drop_noise_captions"]))) if path else []
        out[vid] = [(a[1], b[0]) for a, b in zip(cues, cues[1:])]
    return out

def train_text_videos(records: list[VideoRecord], train_records: list[VideoRecord], min_words: int, df_cap: int) -> dict[str, int]:
    """{source video of `records`: its reliable units whose text is the text of a reliable train unit (a translation target)}, 0 for a
    clean video: the train/eval sentence overlap that the copy check leaves, as MT papers report it. Texts compare as the copy check's
    transcript test reads them (the words of data.loader `WORD_RE`, lowercased), and only a unit of at least `min_words` words counts
    (subtitles.dedup.min_words, the copy check's floor for an identifying cue), so 1 such unit makes a train-text video. As in the copy
    check, a train text of more than `df_cap` train videos (subtitles.dedup.df_cap) is series boilerplate, not evidence: a news
    programme opens every broadcast with 1 sentence, which would make every eval video of the corpus a train-text video."""
    words = lambda text: WORD_RE.findall(text.lower())
    train_videos: dict[str, set[str]] = {}
    for r in train_records:
        for u in r.sentences:
            if getattr(u, "reliable", True): train_videos.setdefault(" ".join(words(u.text)), set()).add(r.pose.video_id)
    train = {text for text, videos in train_videos.items() if len(videos) <= df_cap}
    out: dict[str, int] = {}
    for rec in records:
        out[rec.pose.video_id] = out.get(rec.pose.video_id, 0) + sum(
            1 for u in rec.sentences if getattr(u, "reliable", True) and len(w := words(u.text)) >= min_words and " ".join(w) in train
        )
    return out

def unk_stretches(records) -> dict:
    """ {source video: [(start, end)]} in source time where long-form BLEU-4 has no reference: 
    uncaptioned gaps (UNK) and quarantined units of every stream."""
    unk: dict = {}
    for rec in records:
        off = rec.pose.offset_s
        stretches = rec.untrusted_s if rec.untrusted_s is not None \
                                    else untrusted_o_intervals(rec.sentences, rec.pose.duration_s, TRUSTED_GAP_S)
        unk.setdefault(rec.pose.video_id, []).extend(
            [(a + off, b + off) for a, b in stretches] +
            [(u.start_s + off, u.end_s + off) for u in rec.sentences if not getattr(u, "reliable", True)]
        )
    return unk

def segment_events(segments_by_video: dict, records: list[VideoRecord], tag: str) -> dict[str, list[PredictionEvent]]:
    """Segmenter spans on each stream's clock (keyed by record id) -> the scoreable events of the scores of a row: source time, 
    the quarantine drop, as a row applies them (evaluation.rows). The segmenter dev monitor reads it."""
    return scoreable_predictions(to_source({vid: [
        PredictionEvent(video_id=vid, start_s=float(s.start_s), end_s=float(s.end_s)) for s in segs
    ] for vid, segs in segments_by_video.items()}, records), records, tag=tag)

def prediction_events(video_id: str, events, tokenizer) -> list[PredictionEvent]:
    # The runner's committed events on the stream's clock, 1 event per committed span (never cut); tokenizer None = segmentation 
    # only (no text). A text with UNIT_SEPARATOR keeps its unit texts, joined with spaces (infer.stream.join_unit_text).
    return [PredictionEvent(
        video_id=video_id, start_s=float(ev.start_s), end_s=float(ev.end_s), flagged_partial=bool(ev.flagged_partial), 
        cap_cut=bool(ev.cap_cut), commit_time_s=float(ev.commit_time_s),
        text=None if tokenizer is None else join_unit_text(tokenizer.decode(ev.token_ids.tolist(), skip_special_tokens=True)),
    ) for ev in events]

def _event_rows(path) -> dict:
    # The rows {vid: [event]} of an events file (write_events_json, under `events`), the one file of spans that a row writes; a span
    # file is the events file of a no-text row. Every row writes one event per committed span (never cut). A row with `parts` > 1 is
    # one word-share part of a span: read as a caption, each part would score a split under the never-cut name.
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("events"), dict): raise ValueError(
        f"{path} is not the events file of a row (no `events`): give the events file that an eval.py row wrote (for spans alone, the "
        f"one of a --text none row)"
    )
    if any(int(row.get("parts", 1)) > 1 for rows in raw["events"].values() for row in rows if isinstance(row, dict)): raise ValueError(
        f"{path}: rows with parts > 1 are word-share parts of one span, not never-cut events; run the row that wrote it again"
    )
    return raw["events"]

def load_prediction_file(path: str | Path) -> dict[str, list[Segment]]:
    # The spans of an events file, as Segments.
    return {str(vid): [Segment(float(row["start_s"]), float(row["end_s"])) for row in rows] for vid, rows in _event_rows(path).items()}

def stamped_annotation(path: str | Path) -> str | None:
    # Annotation fingerprint an artifact was made under: flat (a boundary sweep) or under `provenance`.
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict): return None
    return raw.get("annotation_fingerprint") or (raw.get("provenance") or {}).get("annotation_fingerprint")

def warn_annotation_mismatch(path: str | Path, records: list[VideoRecord], what: str) -> bool:
    # Warn when an artifact was made under other annotations than the loaded records; the caller still scores it. True on a match.
    # Every writer stamps `annotation_fingerprint` (caption-unit protocol + a hash of every span it produced)
    stamped, live = stamped_annotation(path), annotation_fingerprint(records)
    if stamped != live: print(
        f"WARNING: {what} {path} was made under annotations {stamped or 'UNSTAMPED'}, but the loaded split is {live}. "
        f"Its spans or scores can disagree with the current labels: regenerate it before you report it.", flush=True
    )
    return stamped == live

def _segment_from_any(value: Any, video_id: str | None = None) -> PredictionEvent:
    if isinstance(value, dict):
        vid = str(value.get("video_id", video_id or ""))
        text = value.get("text")
        commit = value.get("commit_time_s")
        return PredictionEvent(
            video_id=vid, start_s=float(value["start_s"]), end_s=float(value["end_s"]), text=str(text) if text is not None else None,
            flagged_partial=bool(value.get("flagged_partial", False)), cap_cut=bool(value.get("cap_cut", False)),
            commit_time_s=float(commit) if commit is not None else None,
        )
    raise TypeError(f"Unsupported prediction segment row: {value!r}")

def load_event_predictions(path: str | Path) -> dict[str, list[PredictionEvent]]:
    # An events file: what eval.py --predictions scores again and what eval.py --spans <file> reads.
    return {str(video_id): [_segment_from_any(row, video_id=str(video_id)) for row in segments]
            for video_id, segments in _event_rows(path).items()}

def corpus_token(language: str, source_language: str | None = None) -> str:
    # The corpus field of an artifact name: the evaluated corpus, or `<source>-to-<language>` for a zero-shot run (models trained on
    # eval.py --source-language), so a zero-shot artifact never takes the name of an in-corpus one.
    return language if source_language in (None, language) else f"{source_language}-to-{language}"

def output_stem(when: str, spans: str, decode: str, text: str, corpus: str, split: str) -> str:
    # Name shape for every row: `{when}_{spans}_{decode}_{text}_{corpus}_{split}`, in outputs/{when}/: each field follows a flag of
    # eval.py (--offline/--online, --spans, --decode, --text, --language with --source-language, --split); `corpus` is corpus_token's.
    for field, value in (("when", when), ("spans", spans), ("decode", decode), ("text", text)):
        if not value or "_" in str(value): raise ValueError(f"row output {field}={value!r}: one non-empty token, no underscore")
    return f"{when}_{spans}_{decode}_{text}_{corpus}_{split}"

def write_events_json(predicted: dict[str, list[PredictionEvent]], path: str | Path, provenance: dict | None = None,) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"events": {vid: [asdict(ev) for ev in evs] for vid, evs in predicted.items()}}
    if provenance: payload["provenance"] = dict(provenance)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
