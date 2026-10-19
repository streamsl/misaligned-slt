"""Predicted and gold events and their files.

An event is 1 timed unit with its text, on the source-video timeline. This module maps a stream's clock to its source video, builds the
reliable gold, applies the quarantine rule of scoring, turns runner and translator outputs into events, and reads and writes span files,
RQ2 events files, gold segments, their annotation stamps and the RQ2 artifact names.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
import json

from data.loader import annotation_fingerprint, ANNOTATION_PROTOCOL, VideoRecord
from infer.stream import split_unit_events, split_unit_text
from infer.unit_split import span_groups
from metrics import Segment


@dataclass(frozen=True)
class PredictionEvent:
    video_id: str
    start_s: float
    end_s: float
    text: str | None = None
    flagged_partial: bool = False
    commit_time_s: float | None = None  # stride time the commit fired (streaming only)
    # Unit split (infer.stream.split_unit_events): this event is part `part` of `parts` consecutive events that 1 decoded span became. 
    # A span file reader joins them back into that span (`load_prediction_file`), so a same-span control re-translates the span the 
    # arm decoded, not its word-count pieces.
    part: int = 0
    parts: int = 1

    @property
    def segment(self) -> Segment:
        return Segment(self.start_s, self.end_s)


def _translated_events(video_id: str, spans, results, src_rows=None, split: bool = True) -> list[PredictionEvent]:
    # Translated spans -> events carrying the commit time and forced flag of the span's source row when known. With `split`, a text with 
    # UNIT_SEPARATOR becomes 1 event per unit (infer.stream.split_unit_events; only the last can carry the flag). Without it, each span 
    # stays 1 event and its parts are joined with spaces (a fixed-span row moves no boundary).
    out = []
    for i, (span, (text, _, _)) in enumerate(zip(spans, results)):
        row = (src_rows[i] if src_rows and i < len(src_rows) else None) or {}
        ct = row.get("commit_time_s")
        parts = split_unit_events(span.start_s, span.end_s, text) \
                 if split else [(span.start_s, span.end_s, " ".join(split_unit_text(text)))]
        out.extend(PredictionEvent(
            video_id=video_id, start_s=s0, end_s=s1, text=part, commit_time_s=None if ct is None else float(ct),
            flagged_partial=bool(row.get("flagged_partial", False)) and k == len(parts) - 1, part=k, parts=len(parts),
        ) for k, (s0, s1, part) in enumerate(parts))
    return out


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


def _drop_quarantined_predictions(predicted: dict[str, list], records: list[VideoRecord]) -> dict[str, list]:
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
    predicted: dict[str, list[PredictionEvent]], records: list[VideoRecord], tag: str = "rq2",
) -> dict[str, list[PredictionEvent]]:
    # Model duration policy belongs to event generation. Evaluation keeps short true and false predictions alike.
    n0 = sum(len(v) for v in predicted.values())
    kept = _drop_quarantined_predictions(predicted, records)
    n1 = sum(len(v) for v in kept.values())
    print(f"[{tag}] scoreable events: {n1} of {n0} (dropped {n0 - n1} majority-inside quarantine)", flush=True)
    return kept


def _gold_events(records: list[VideoRecord]) -> dict[str, list[PredictionEvent]]:
    # Reliable units per SOURCE video in source time; every source video of the split has a key.
    return to_source({record.video_id: [PredictionEvent(
        video_id=record.video_id, start_s=float(span.start_s), end_s=float(span.end_s), text=span.text,
    ) for span in record.sentences if getattr(span, "reliable", True)] for record in records}, records)


def _prediction_events(video_id: str, events, tokenizer) -> list[PredictionEvent]:
    # The runner's committed events on the stream's clock; tokenizer None = segmentation only (no text). A span whose 
    # text holds UNIT_SEPARATOR becomes 1 event per unit (infer.stream.split_unit_events), all at the span's commit time.
    out = []
    for ev in events:
        if tokenizer is None:
            out.append(PredictionEvent(
                video_id=video_id, start_s=float(ev.start_s), end_s=float(ev.end_s),
                flagged_partial=bool(ev.flagged_partial), commit_time_s=float(ev.commit_time_s)
            ))
            continue
        parts = split_unit_events(ev.start_s, ev.end_s, tokenizer.decode(ev.token_ids.tolist(), skip_special_tokens=True).strip())
        out.extend(PredictionEvent(
            video_id=video_id, start_s=s0, end_s=s1, text=part, commit_time_s=float(ev.commit_time_s),
            flagged_partial=bool(ev.flagged_partial) and k == len(parts) - 1, part=k, parts=len(parts)
        ) for k, (s0, s1, part) in enumerate(parts))
    return out


def joined_events(events: dict) -> dict:
    """Events with every split span joined back into the one decoded span it came from (`infer.stream.split_unit_events` wrote 
    parts 0..parts-1): [first part's start, last part's end], parts' texts joined with spaces, last part's forced flag. Scoring 
    both versions of 1 events file measures what unit split changes, with the model, spans, their commit times and words fixed."""
    return {vid: [replace(g[0], end_s=g[-1].end_s, text=" ".join(x.text for x in g if x.text) or g[0].text,
            flagged_partial=g[-1].flagged_partial, part=0, parts=1) for g in span_groups(evs)] for vid, evs in events.items()}


def save_prediction_file(predictions: dict[str, list[Segment]], path: str | Path, provenance: dict) -> Path:
    # Predicted spans + WHO produced them. Without the stamp a predictions file is just spans, and the
    # arch/decode that made it is unrecoverable — a cascade row could silently inherit the wrong decode.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {
        video_id: [{"start_s": float(s.start_s), "end_s": float(s.end_s)} for s in segments]
        for video_id, segments in sorted(predictions.items())
    }
    payload = {"provenance": dict(provenance), "predictions": rows}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _joined_parts(rows: list[dict]) -> list[list[dict]]:
    # Consecutive rows of one split span (part 0..parts-1) as one group; a row with no split fields is its own group.
    groups: list[list[dict]] = []
    for row in rows:
        if int(row.get("part", 0)) == 0 or not groups: groups.append([row])
        else: groups[-1].append(row)
    return groups


def load_prediction_file(path: str | Path) -> dict[str, list[Segment]]:
    # Predicted-segments JSON: a stamped dict {vid: [{start_s, end_s}]} under `predictions`, `segments` or `events`. 
    # The parts of a split span join back into that span: [first part's start, last part's end].
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "predictions" in raw: raw = raw["predictions"]  # stamped form
    if isinstance(raw, dict) and "segments" in raw: raw = raw["segments"]  # stamped gold-segments file (write_gold_segments)
    if isinstance(raw, dict) and "events" in raw: raw = raw["events"]  # RQ2 events file: its spans feed the same-span control.
    if not isinstance(raw, dict): raise TypeError(f"{path}: a prediction file is a stamped JSON dict, not {type(raw).__name__}")
    return {str(vid): [Segment(float(g[0]["start_s"]), float(g[-1]["end_s"])) for g in _joined_parts(rows)] for vid, rows in raw.items()}


def stamped_annotation(path: str | Path) -> str | None:
    # Annotation fingerprint an artifact was made under: flat (RQ1, segmenter-eval payloads) or under `provenance`.
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


def load_prediction_rows(path: str | Path) -> dict[str, list[dict]]:
    # Rows of an RQ2 events file, aligned with load_prediction_file's spans (1 per joined span); empty for a plain segments file. Carries 
    # what a Segment cannot: commit time and forced-partial flag (of the span's last part), so a re-translated row keeps emitter's record.
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not (isinstance(raw, dict) and "events" in raw): return {}
    return {str(vid): [
        {**g[0], "flagged_partial": bool(g[-1].get("flagged_partial", False))} for g in _joined_parts(rows)
    ] for vid, rows in raw["events"].items()}


def write_gold_segments(records: list[VideoRecord], path: str | Path) -> Path:
    # GT sentence spans in the `--segments` schema (load_prediction_file dict form).
    # Feeds the RQ2 oracle-input (ceiling) rows: `--rq 2 --segments <this>` is RQ1 @ delta=0, framed for RQ2.
    rows = {vid: [{"start_s": ev.start_s, "end_s": ev.end_s, "text": ev.text} for ev in evs] for vid, evs in _gold_events(records).items()}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": {"annotation_protocol": ANNOTATION_PROTOCOL, "annotation_fingerprint": annotation_fingerprint(records),
                       "segmenter_arch": "gold", "decode": "none"},
        "segments": rows
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _segment_from_any(value: Any, video_id: str | None = None) -> PredictionEvent:
    if isinstance(value, dict):
        vid = str(value.get("video_id", video_id or ""))
        text = value.get("text")
        commit = value.get("commit_time_s")
        return PredictionEvent(
            video_id=vid, start_s=float(value["start_s"]), end_s=float(value["end_s"]), text=str(text) if text is not None else None,
            flagged_partial=bool(value.get("flagged_partial", False)), commit_time_s=float(commit) if commit is not None else None,
            part=int(value.get("part", 0)), parts=int(value.get("parts", 1)),
        )
    raise TypeError(f"Unsupported prediction segment row: {value!r}")


def load_event_predictions(path: str | Path) -> dict[str, list[PredictionEvent]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "events" in data: data = data["events"]  # stamped form (see _write_events_json)
    if not isinstance(data, dict): raise TypeError(f"{path}: an events file is a stamped JSON dict, not {type(data).__name__}")
    return {str(video_id): [_segment_from_any(row, video_id=str(video_id)) for row in segments] for video_id, segments in data.items()}


def rq2_output_stem(when: str, spans: str, decode: str, translator: str, language: str, split: str) -> str:
    # Name shape for every RQ2 row: `rq2_{when}_{spans}_{decode}_{translator}_{lang}_{split}`.
    for field, value in (("when", when), ("spans", spans), ("decode", decode), ("translator", translator)):
        if not value or "_" in str(value): raise ValueError(f"rq2 output {field}={value!r}: one non-empty token, no underscore")
    return f"rq2_{when}_{spans}_{decode}_{translator}_{language}_{split}"


def _write_events_json(predicted: dict[str, list[PredictionEvent]], path: str | Path, provenance: dict | None = None,) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"events": {vid: [asdict(ev) for ev in evs] for vid, evs in predicted.items()}}
    if provenance: payload["provenance"] = dict(provenance)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
