from __future__ import annotations
import re, random, csv, html
from dataclasses import dataclass, asdict
from typing import Any, Iterable
from pathlib import Path

from torch.utils.data import Dataset
from poses import PoseIndex, build_pose_index
from train.sampler import WindowSampler
from .windowing import SentenceSpan


TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")
WORD_TIMING_RE = re.compile(r"<\d{1,2}:\d{2}:\d{2}[.,]\d{3}>")


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


def parse_vtt(path: str | Path) -> list[tuple[float, float, str]]:
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
        if text and end_s > start_s: captions.append((start_s, end_s, text))
        i += 1
    return captions


def _subtitle_score(path: Path, preferred_suffixes: list[str], reject_suffixes: list[str]) -> tuple[int, int, str]:
    name = path.name
    for rejected in reject_suffixes:
        if name.endswith(rejected):
            return (10_000, 0, name)
    for rank, suffix in enumerate(preferred_suffixes):
        if name.endswith(suffix):
            return (rank, 0, name)
    return (len(preferred_suffixes) + 100, 0, name)


def find_best_subtitle(
    subtitle_root: str | Path, video_id: str,
    preferred_suffixes: list[str], reject_suffixes: list[str],
    min_caption_chars: int = 2,
) -> Path | None:
    subtitle_root = Path(subtitle_root)
    candidates = sorted(subtitle_root.glob(f"{video_id}*.vtt"))
    scored: list[tuple[tuple[int, int, str], Path]] = []
    for path in candidates:
        try: parsed = parse_vtt(path)
        except OSError: continue

        char_count = sum(len(text) for _, _, text in parsed)
        if char_count < min_caption_chars: continue
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
    root = Path(lang_cfg["root"])
    pose_index = build_pose_index(
        root / "poses",
        fps=float(lang_cfg.get("pose_fps", 25.0)),
        width=int(lang_cfg["width"]) if lang_cfg.get("width") is not None else None,
        height=int(lang_cfg["height"]) if lang_cfg.get("height") is not None else None,
    )
    subtitle_cfg = data_cfg.get("subtitles", {})
    splits = build_splits(sorted(pose_index.keys()), data_cfg.get("splits", {}))
    selected_ids = splits.get(split, []) if split else sorted(pose_index.keys())

    records: list[VideoRecord] = []
    for video_id in selected_ids:
        subtitle_path = find_best_subtitle(
            root / "subs", video_id,
            preferred_suffixes=list(subtitle_cfg.get("preferred_suffixes", [".en.vtt"])),
            reject_suffixes=list(subtitle_cfg.get("reject_suffixes", [".en-orig.vtt"])),
            min_caption_chars=int(subtitle_cfg.get("min_caption_chars", 2)),
        )
        if subtitle_path is None: continue
        captions = parse_vtt(subtitle_path)
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

    `__getitem__` samples from the training distribution rather than indexing a
    fixed window table. This mirrors the intended stochastic window sampler while
    still satisfying PyTorch/HF Trainer's map-style dataset interface.
    """
    def __init__(
        self, records: list[VideoRecord],
        stage2_cfg: dict[str, Any], inference_cfg: dict[str, Any],
        steps_per_epoch: int | None = None, include_full_evidence: bool = True,
    ):
        self.records = records
        self.records_by_id = {record.video_id: record for record in records}
        self.sampler = WindowSampler.from_stage2_config(records, stage2_cfg, inference_cfg)
        self.steps_per_epoch = int(steps_per_epoch or max(len(self.sampler.anchors), 1))
        self.include_full_evidence = bool(include_full_evidence)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, index: int) -> dict:
        del index
        sample = self.sampler.sample()
        item = self.sampler.to_dict(sample)
        if self.include_full_evidence and sample.full_evidence_spec is not None:
            rec = self.records_by_id[sample.full_evidence_spec.video_id]
            full = self.sampler.materialize(rec, sample.full_evidence_spec)
            item["full_evidence"] = self.sampler.to_dict(full)
        else: item["full_evidence"] = None
        return item


def build_streaming_window_dataset(
    records: list[VideoRecord],
    stage2_cfg: dict[str, Any],
    inference_cfg: dict[str, Any],
    steps_per_epoch: int | None = None,
) -> StreamingWindowDataset:
    return StreamingWindowDataset(
        records=records,
        stage2_cfg=stage2_cfg,
        inference_cfg=inference_cfg,
        steps_per_epoch=steps_per_epoch,
    )
