"""Corpus audit: what every preprocessing rule removed or repaired, per language and split.

One table per language, one row per rule, counted by the loader itself (`load_language_records(report=...)`) so the
numbers are the ones training and evaluation actually saw. Video-level rules count videos; cue-level rules count
cues or units; the visual cuts count segments, units and frames; the quarantines (the corpus's unreliable marks, no hand, switch doubt) count units. Person measurements and visual runs come from
`video_meta.csv`, populated by `prepare_yt25.py --stage convert` / `person-counts`. Missing values are reported as unknown.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

from data.loader import QUARANTINE_RULES, corpus_config, load_language_records

SPLITS = ("train", "dev", "test")
ROWS = (  # (report key, label, unit)
    ("videos_listed", "videos listed by the corpus", "videos"),
    ("videos_no_pose_upload", "listed but no pose data published upstream", "videos"),
    ("videos_in_split", "videos in the split", "videos"),
    ("dropped_mt_caption", "machine-translated captions (human-only splits)", "videos"),
    ("dropped_no_caption", "no usable caption track", "videos"),
    ("dropped_wrong_language", "wrong-language captions (max_non_latin_ratio, min_english_function_word_ratio)", "videos"),
    ("dropped_scrolling", "scrolling caption track (drop_scrolling_tracks)", "videos"),
    ("dropped_pose_coverage", "caption cues outside the available pose timeline (min_cue_coverage)", "videos"),
    ("deduplicated_train", "train copy of a dev/test video (dedup)", "videos"),
    ("dropped_all_quarantined", "every unit quarantined or cut (no reliable span)", "videos"),
    ("videos_kept", "videos kept", "videos"),
    ("videos_cut", "kept videos cut at an empty run", "videos"),
    ("segments_kept", "clean segments kept (one stream each)", "segments"),
    ("source_cues", "source cues (kept videos)", "cues"),
    ("rolling_duplicates_merged", "rolling-display duplicates merged", "cues"),
    ("units_after_grouping", "caption units after grouping", "units"),
    ("overlaps_clamped", "overlapping units REPAIRED (start moved to the previous end)", "units"),
    ("overlaps_swallowed", "units REMOVED, contained in their predecessor (text lost)", "units"),
    ("overlaps_containers_quarantined", "units quarantined: they hold a removed unit, so their text misses its words", "units"),
    ("overlaps_disputed_quarantined", "units quarantined: they overlap a neighbour by more than δ, so no boundary is known to δ", "units"),
    ("end_straddlers_quarantined", "units straddling the pose stream end (quarantined)", "units"),
    ("spans_dropped_too_short", "units shorter than min_duration_s (kept as unlabelled spans: frames UNK)", "units"),
    ("spans_dropped_outside_poses", "units outside the pose stream", "units"),
    ("spans_dropped_no_text", "units with no alphabetic text (kept as unlabelled spans: frames UNK)", "units"),
    ("spans_kept", "units kept, before the visual cuts", "units"),
    ("units_quarantined_unreliable", "units removed or quarantined by the corpus's unreliable spans (video_meta.csv unreliable_spans: an overlap above δ, or a unit that holds one or lies inside one)", "units"),
    ("units_quarantined_handless", "units quarantined: no visible hands (handless_unit_share; frames kept)", "units"),
    ("units_quarantined_switch_doubt", "units quarantined: the switch keeps another body and the converter's default body signs (switch_unit_share, switch_doubt_share; frames kept)", "units"),
    ("units_cut", "reliable units lost to a visual cut (inside a run, or crossing a segment edge: quarantined)", "units"),
    ("frames_empty_runs", "frames in empty runs (no real body)", "frames"),
    ("frames_cut", "frames cut (the empty runs, short gaps and video ends absorbed)", "frames"),
    ("frames_kept", "pose frames in clean segments", "frames"),
    ("frames_undetected", "frames with no detected body, before the cuts", "frames"),
)
CUT_ROWS = {"videos_cut", "units_cut", "frames_empty_runs", "frames_cut"}   # read the empty_runs column


def _listed_per_split(data_cfg: dict, language: str) -> dict[str, int]:
    """Videos the corpus listing assigns to each split, whether or not poses were ever published for them.

    The loader can only see videos that HAVE a pose file — it builds its splits from the pose index — so its
    `videos_in_split` silently excludes every video the upstream release never uploaded poses for. On ase that is
    28.6 % of the corpus, which is larger than every preprocessing rule in this table combined and is not a choice
    this project made. Counted straight from the same CSV the loader splits on (the corpus's `splits.file`, data.loader
    corpus_config). A corpus with its own split file lists only its own videos, so a file with no sign_language column is all of it.
    """
    path = Path(corpus_config(data_cfg, language)["splits"].get("file", ""))
    if not path.is_file(): return {}
    counts = dict.fromkeys(SPLITS, 0)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("sign_language", language) != language: continue
            split = row.get("split")
            split = "dev" if split == "val" else split          # the loader's own rename
            if split in counts: counts[split] += 1
    return counts


def audit_language(data_cfg: dict, language: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    listed = _listed_per_split(data_cfg, language)
    for split in SPLITS:
        report: dict = {}
        load_language_records(data_cfg, language, split=split, report=report)
        if listed:
            report["videos_listed"] = listed.get(split, 0)
            report["videos_no_pose_upload"] = max(0, listed.get(split, 0) - report.get("videos_in_split", 0))
        out[split] = report
    return out


def markdown(language: str, per_split: dict[str, dict]) -> str:
    lines = [f"### {language}", "", "| rule | " + " | ".join(SPLITS) + " | unit |", "|---|" + "---:|" * len(SPLITS) + "---|"]
    for key, label, unit in ROWS:
        cells = []
        for split in SPLITS:
            r = per_split[split]
            absent = r.get("empty_runs_unknown", 0)
            if key == "frames_undetected" and r.get("undetected_ratio_known_videos", 0) < r.get("videos_kept", 0):
                cells.append(f"{r.get(key, 0):,} (of {r.get('undetected_ratio_known_videos', 0)} videos with the column)")
            elif key == "dropped_pose_coverage" and not r.get("pose_coverage_rule_enabled"):
                cells.append("rule disabled")
            elif absent and key in CUT_ROWS: cells.append(f"{r.get(key, 0):,} (empty_runs is absent for {absent} kept videos)")
            elif unknown := [column for column in QUARANTINE_RULES.get(key.removeprefix("units_quarantined_"), ()) if r.get(f"{column}_unknown")]:
                cells.append(f"{r.get(key, 0):,} (" + "; ".join(f"{column} is absent for {r[f'{column}_unknown']} kept videos" for column in unknown) + ")")
            else: cells.append(f"{r.get(key, 0):,}")
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {unit} |")
    return "\n".join(lines) + "\n"


def run(args) -> dict:
    from utils import load_yaml, output_dir
    data_cfg = load_yaml(args.data_config)
    out_dir = Path(args.out) if args.out else output_dir("report") / "data"; out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for language in args.language:
        per_split = audit_language(data_cfg, language)
        result[language] = per_split
        (out_dir / f"data_audit_{language}.json").write_text(json.dumps(per_split, indent=2) + "\n", encoding="utf-8")
        md = markdown(language, per_split)
        (out_dir / f"data_audit_{language}.md").write_text(md, encoding="utf-8")
        print(md, flush=True)
    return result
