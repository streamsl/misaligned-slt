from __future__ import annotations
import re, random, csv, html
from dataclasses import dataclass, asdict
from typing import Any, Iterable
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset
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
NOISE_WORD_RE = re.compile(r"[a-z]+")
NOISE_CAPTION_WORDS = {
    "applause", "background", "foreign", "gentle", "inaudible", "laugh", "laughs",
    "laughter", "music", "piano", "silence", "silent",
}


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
    raw = " ".join(line.strip() for line in lines if line.strip())
    raw = WORD_TIMING_RE.sub(" ", raw)
    raw = TAG_RE.sub(" ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def is_noise_caption(text: str) -> bool:
    """True for captions that are only non-signed stage directions.

    Keep real sentences that merely contain words like "music" or "Facebook"; drop only whole-cue
    annotations such as "AUDIENCE: (APPLAUSE)" or "(gentle piano music)".
    """
    text = SPEAKER_PREFIX_RE.sub("", text.strip())
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
    captions: list[tuple[float, float, str]],
    max_cues: int = 2,
    min_chars: int = 500,
    max_chars_per_second: float = 120.0,
) -> bool:
    """Reject YouTube VTT variants that put the whole transcript in one short cue.

    ASF commonly ships paired files where `.en-GB.vtt` has normal cue timing but `.en-en-GB.vtt`
    contains thousands of characters in the first few seconds and empty cues afterwards. Such files
    are unusable for pose-text alignment and should lose to any non-flattened candidate.
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
    min_caption_chars: int = 2,
    reject_flattened_transcripts: bool = True,
    flattened_max_cues: int = 2,
    flattened_min_chars: int = 500,
    flattened_max_chars_per_second: float = 120.0,
    drop_noise: bool = False,
) -> Path | None:
    subtitle_root = Path(subtitle_root)
    candidates = sorted(subtitle_root.glob(f"{video_id}*.vtt"))
    scored: list[tuple[tuple[int, int, str], Path]] = []
    for path in candidates:
        try: parsed = parse_vtt(path, drop_noise=drop_noise)
        except OSError: continue

        char_count = sum(len(text) for _, _, text in parsed)
        if char_count < min_caption_chars: continue
        if reject_flattened_transcripts and looks_flattened_transcript(
            parsed, max_cues=flattened_max_cues,
            min_chars=flattened_min_chars, max_chars_per_second=flattened_max_chars_per_second,
        ): continue
        score = _subtitle_score(path, preferred_suffixes, reject_suffixes)
        scored.append(((score[0], -char_count, score[2]), path))
    return min(scored, default=(None, None))[1] if scored else None


def _load_signverse_splits(path: Path) -> dict[str, str]:
    if not path.exists(): return {}
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
    signverse = _load_signverse_splits(Path(split_cfg.get("signverse_csv", "")))
    if signverse:
        splits = {"train": [], "dev": [], "test": []}
        for video_id in video_ids:
            split = signverse.get(video_id)
            if split in splits: splits[split].append(video_id)
        if any(splits.values()): return {k: sorted(v) for k, v in splits.items()}

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


def load_language_records(data_cfg: dict, language: str, split: str | None = None) -> tuple[list[VideoRecord], dict[str, list[str]]]:
    lang_cfg = data_cfg["languages"][language]
    # Synthetic streaming corpora (data_synth: PHOENIX / CSL-Daily / How2Sign / ...) share 1 self-contained layout 
    # (per-stream npy + vtt, explicit splits, exact fps) — delegate to their shared loader. The YouTube path below 
    # assumes poses/+subs/, per-video fps calibration, and SignVerse splits.
    if str(lang_cfg.get("source", "")).lower() in {"streams", "synth_streams", "phoenix"}:
        from data.synth_streams import load_stream_records
        return load_stream_records(data_cfg, language, split=split)

    # Official pre-trimmed benchmark releases: one single-sentence record per clip, real context frames included — 
    # the clean literature-comparison corpus for RQ1 (--severity-grid-s 0.0 = the paper-comparable point).
    if str(lang_cfg.get("source", "")).lower() == "pretrimmed":
        from data.pretrimmed import load_pretrimmed_records
        return load_pretrimmed_records(data_cfg, language, split=split)

    root = Path(lang_cfg["root"])
    # Per-video fps from the video_meta.json sidecar (poses were extracted at native/2 fps, which
    # VARIES per YouTube video: 12.0-15.0 measured). config pose_fps is fallback-only; with no
    # sidecar every timestamp drifts ~2x and ~44% of captions get dropped by the duration filter.
    video_meta = load_video_meta(root / META_FILENAME)
    pose_cfg = lang_cfg.get("pose", {}) or {}
    fps_fallback = float(pose_cfg.get("fps", 25.0))
    pose_index = build_pose_index(
        root / "poses", fps=fps_fallback,
        width=int(pose_cfg["width"]) if pose_cfg.get("width") is not None else None,
        height=int(pose_cfg["height"]) if pose_cfg.get("height") is not None else None,
        video_meta=video_meta,
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

    records: list[VideoRecord] = []
    for video_id in selected_ids:
        subtitle_path = find_best_subtitle(
            root / "subs", video_id,
            preferred_suffixes=list(subtitle_cfg.get("preferred_suffixes", [".en.vtt"])),
            reject_suffixes=list(subtitle_cfg.get("reject_suffixes", [".en-orig.vtt"])),
            min_caption_chars=int(subtitle_cfg.get("min_caption_chars", 2)),
            reject_flattened_transcripts=bool(subtitle_cfg.get("reject_flattened_transcripts", True)),
            flattened_max_cues=int(subtitle_cfg.get("flattened_max_cues", 2)),
            flattened_min_chars=int(subtitle_cfg.get("flattened_min_chars", 500)),
            flattened_max_chars_per_second=float(subtitle_cfg.get("flattened_max_chars_per_second", 120.0)),
            drop_noise=drop_noise,
        )
        if subtitle_path is None: continue
        captions = merge_rolling_captions(parse_vtt(subtitle_path, drop_noise=drop_noise))
        min_dur = float(subtitle_cfg.get("min_duration_s", 0.2))
        max_dur = float(subtitle_cfg.get("max_duration_s", 60.0))
        spans = tuple(
            SentenceSpan(video_id=video_id, start_s=s, end_s=e, text=t)
            for s, e, t in captions
            if min_dur <= (e - s) <= max_dur and e <= pose_index[video_id].duration_s + 1.0
        )
        if spans: records.append(VideoRecord(language, video_id, pose_index[video_id], subtitle_path, spans))
    return records, splits


class StreamingWindowDataset(Dataset):
    """On-the-fly Stage 2 window dataset.

    `__getitem__` samples from the training distribution rather than indexing a fixed window table. This mirrors the intended 
    stochastic window sampler while still satisfying PyTorch/HF Trainer's map-style dataset interface.
    """
    def __init__(
        self, records: list[VideoRecord], slt_cfg: dict[str, Any], inference_cfg: dict[str, Any],
        steps_per_epoch: int | None = None, include_full_evidence: bool = True, 
        deterministic: bool = False, pose_augment_cfg: dict | None = None,
    ):
        # `WindowSampler` is imported lazily inside StreamingWindowDataset.__init__ to break the
        # data.loader <-> train.sampler import cycle (train.sampler needs VideoRecord, defined below, for type hints).
        from train.sampler import WindowSampler
        self.records = records
        self.records_by_id = {record.video_id: record for record in records}
        self.sampler = WindowSampler.from_slt_config(
            records, slt_cfg, inference_cfg, pose_augment_cfg=pose_augment_cfg
        )
        self.steps_per_epoch = int(steps_per_epoch or max(len(self.sampler.anchors), 1))
        self.include_full_evidence = bool(include_full_evidence)
        # Eval loaders set deterministic=True: window `index` then always yields the SAME anchor
        # under a per-index rng, so the early-stopping monitor scores a fixed dev set every epoch
        # instead of a fresh random draw (which makes "best epoch" partly a lottery).
        self.deterministic = bool(deterministic)
        self.seed = int(slt_cfg.get("seed", 42))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _sample_item(self) -> dict:
        sample = self.sampler.sample()
        item = self.sampler.to_dict(sample)
        if self.include_full_evidence and sample.full_evidence_spec is not None:
            rec = self.records_by_id[sample.full_evidence_spec.video_id]
            full = self.sampler.materialize(rec, sample.full_evidence_spec)
            item["full_evidence"] = self.sampler.to_dict(full)
        else: item["full_evidence"] = None
        return item

    def __getitem__(self, index: int) -> dict:
        if not self.deterministic: return self._sample_item()
        rng = np.random.default_rng(self.seed * 100_003 + int(index))
        saved = (self.sampler.rng, self.sampler._anchor_order, self.sampler._anchor_cursor, self.sampler.fps_aug_enabled)
        self.sampler.rng = rng
        # Validation must enumerate GT sentence anchors. A random permutation per index picks only
        # the first element of many independent permutations, so anchors can duplicate while others
        # never appear. Force the chosen anchor and let the per-index rng still draw mode/jitter.
        anchor_idx = int(index) % len(self.sampler.anchors)
        self.sampler._anchor_order = np.asarray([anchor_idx], dtype=np.int64)
        self.sampler._anchor_cursor = 0
        # fps_aug is a TRAIN augmentation (Moryossef 2026 gates it on split==TRAIN; eval runs native fps).
        # Leaving it on here scored the monitor on 15–30fps resampled windows the head never deploys under.
        self.sampler.fps_aug_enabled = False
        try: return self._sample_item()  # incl. full-evidence materialization, under the per-index rng
        finally: (self.sampler.rng, self.sampler._anchor_order,
                  self.sampler._anchor_cursor, self.sampler.fps_aug_enabled) = saved


def streaming_loader(dataset: StreamingWindowDataset, batch_size: int, collate_fn, num_workers: int = 0) -> DataLoader:
    """The ONE DataLoader constructor for StreamingWindowDataset (both trainers route through here).

    The non-deterministic train path draws from a SINGLE stateful WindowSampler rng, and forked/spawned workers
    inherit IDENTICAL Generator state (PyTorch per-worker seeding never touches a Generator instance stored on the
    dataset) — num_workers=4 makes batches 0..3, 4..7, ... byte-identical: 4x duplicate gradients and ~1/4 unique
    anchors per epoch (sampler.py documents the num_workers=0 requirement). So workers are CLAMPED to 0 there.
    Deterministic (dev) datasets derive their rng from the sample index, so workers are safe and kept.
    """
    if num_workers and not dataset.deterministic:
        print(f"loader | num_workers {num_workers} -> 0: the stateful WindowSampler emits identical streams in "
              f"every worker (duplicate batches; see train/sampler.py)", flush=True)
        num_workers = 0
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers),
        persistent_workers=num_workers > 0, collate_fn=collate_fn,
    )


def build_streaming_window_dataset(
    records: list[VideoRecord], slt_cfg: dict[str, Any], inference_cfg: dict[str, Any],
    steps_per_epoch: int | None = None, pose_augment_cfg: dict | None = None,
) -> StreamingWindowDataset:
    return StreamingWindowDataset(
        records=records, slt_cfg=slt_cfg, inference_cfg=inference_cfg,
        steps_per_epoch=steps_per_epoch, pose_augment_cfg=pose_augment_cfg,
    )
