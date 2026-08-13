from __future__ import annotations
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass
from typing import Any, Iterable
from pathlib import Path

import re, random, csv, html
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
    max_cues: int = 2, min_chars: int = 500, max_chars_per_second: float = 120.0,
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
    for vid in video_ids:
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
    for video_id in selected_ids:
        subtitle_path = best_subtitle(root / "subs", video_id, subtitle_cfg)
        if subtitle_path is None:
            dropped_no_caption += 1
            continue
        captions = merge_rolling_captions(parse_vtt(subtitle_path, drop_noise=drop_noise))
        min_dur = float(subtitle_cfg.get("min_duration_s", 0.2))
        max_dur = float(subtitle_cfg.get("max_duration_s", 60.0))
        # `s < duration`: the sentence ONSET must land inside the extracted poses, else no visible signing to anchor
        # on. SignVerse streams end before their caption timeline (duration = pose_frames/24 underestimates the
        # video), so late captions start past the poses; `e <= duration + 1.0` bounds only the END. Without it the
        # sampler builds start_s > end_s windows → load_pose_frames raises. No-op for stream corpora.
        dur = pose_index[video_id].duration_s
        spans = tuple(
            SentenceSpan(video_id=video_id, start_s=s, end_s=e, text=t)
            for s, e, t in captions
            if min_dur <= (e - s) <= max_dur and s < dur and e <= dur + 1.0
        )
        if spans: records.append(VideoRecord(language, video_id, pose_index[video_id], subtitle_path, spans))
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
        self, records: list[VideoRecord], slt_cfg: dict[str, Any], inference_cfg: dict[str, Any],
        steps_per_epoch: int | None = None, include_full_evidence: bool = True, 
        deterministic: bool = False, pose_augment_cfg: dict | None = None,
    ):
        # Lazy import breaks the data.loader <-> train.sampler cycle (train.sampler needs VideoRecord for type hints).
        from train.sampler import WindowSampler
        self.records = records
        self.records_by_id = {record.video_id: record for record in records}
        self.sampler = WindowSampler.from_slt_config(
            records, slt_cfg, inference_cfg, pose_augment_cfg=pose_augment_cfg
        )
        self.steps_per_epoch = int(steps_per_epoch or max(len(self.sampler.anchors), 1))
        self.include_full_evidence = bool(include_full_evidence)
        # Eval loaders set deterministic=True: an index then always yields the SAME anchor under a per-index rng, so early-stopping 
        # monitor scores a fixed dev set each epoch instead of a fresh draw (else "best epoch" is partly a lottery).
        self.deterministic = bool(deterministic)
        self.seed = int(slt_cfg.get("seed", 42))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _sample_item(self, index: int) -> dict:
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


def streaming_loader(dataset: StreamingWindowDataset, batch_size: int, collate_fn, num_workers: int = 0) -> DataLoader:
    """The ONE DataLoader constructor for StreamingWindowDataset (both trainers route through here).

    num_workers is a plain throughput knob: the ANCHOR is a deterministic function of the global sample index
    (WindowSampler.sample → anchors[index % N]), so every anchor is realized exactly once per epoch however indices
    are partitioned — no duplication, no lost coverage. `_streaming_worker_init` decorrelates the per-window random
    stream forked workers would otherwise share; dev datasets seed from the index and need neither.
    """
    sampler = None
    if dist.is_distributed(): sampler = DistributedSampler(dataset, num_replicas=dist.world_size(), rank=dist.rank(), shuffle=False)
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, sampler=sampler, num_workers=int(num_workers),
        persistent_workers=num_workers > 0, collate_fn=collate_fn,
        worker_init_fn=_streaming_worker_init if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(), prefetch_factor=4 if num_workers > 0 else None,
    )
