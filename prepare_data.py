"""SignVerse-2M downloader + converter → the repo's YouTube-SL-25 language layout.

Fetches exactly the shards containing the requested languages' videos from HuggingFace SignerX/SignVerse-2M
(shards are mixed-language; asf+bfi span ~206 of 723 shards, ~80 GB vs ~1.3 TB full corpus), converts each
wanted video's DWPose-128 npz (either packaging scheme) to the canonical (T,133,3) raw-pixel .npy via
poses.signverse, writes ONE caption per video, and video_meta.csv (duration from the npz frame count at the
corpus's unified 24 fps — no yt-dlp needed). After this the language behaves exactly like the repo's own
extractions: train.py / analyze.py / eval.py run unchanged with --language asf|bfi.

    python prepare_data.py --stage plan [--size]              # shard/video counts (+ HEAD size estimate); no download
    python prepare_data.py --stage all --languages asf bfi    # download + convert + subs gap-fill (resumable)
    python prepare_data.py --stage subs --languages asf       # gap-fill captions from the subtitles tar only
    python prepare_data.py --stage convert --delete-tars      # convert already-downloaded shards, free disk

ON-DISK LAYOUT (root = data/youtube-sl-25):
    <root>/SignVerse-2M-metadata_split.csv        # splits (shipped)         ── dataset metadata (siblings)
    <root>/archive_upload_progress.json           # video→shard index (fetched once)
    <root>/signverse_subtitles_with_english.tar   # captions tar (fetched only if gaps)
    <root>/signverse_shards/*.tar                 # shard-tar CACHE ONLY (transient; --delete-tars frees it)
    <root>/{asf,bfi}/poses/<vid>.npy , subs/<vid>.<target>.vtt , video_meta.csv

CAPTIONS — one file per video, `<vid>.<target>.vtt`, from ONE selection rule (data.loader.best_subtitle), both paths:
  - `--stage convert` harvests the single best shard-bundled track → caption_source=shard (no extra download).
  - `--stage subs` GAP-FILLS only the videos convert left caption-less, from the curated subtitles tar, choosing the
    dataset target language (`configs/data.yaml` target_lang): the HUMAN track (original.<target>*.manual, or
    english.en.native for target=en) over NLLB MACHINE-translated english.en.nllb (target=en only). If convert left
    no gaps, the 700 MB tar is never fetched. Provenance → video_meta.csv `caption_source` (human|mt|shard|none) so
    the loader holds the TEST split to human references (`subtitles.human_only_splits`). Language-general.

The authoritative video→shard index is runtime_state/archive_upload_progress.json::uploaded_folders (the other
manifests in the repo are stale snapshots — verified). A handful of asf/bfi videos are not yet uploaded upstream
(no failure markers; past the upload frontier), and a few are listed in the index but absent from their shard
(upstream packaging gap) — both are reported and reconciled, not treated as errors.
"""
from __future__ import annotations
import argparse, csv, json, shutil, sys, tarfile, urllib.request
import numpy as np

from pathlib import Path
from data.loader import best_subtitle
from poses.pose_io import META_FILENAME, load_video_meta, save_video_meta
from poses.signverse import SIGNVERSE_DEFAULT_FPS, convert_video
from utils import load_yaml

HF_BASE = "https://huggingface.co/datasets/SignerX/SignVerse-2M/resolve/main"
PROGRESS_JSON = "runtime_state/archive_upload_progress.json"
SUBTITLES_TAR = "signverse_subtitles_with_english.tar"
DEFAULT_SPLIT_CSV = "data/youtube-sl-25/SignVerse-2M-metadata_split.csv"
DEFAULT_CACHE = "data/youtube-sl-25/signverse_shards"
DEFAULT_ROOT = "data/youtube-sl-25"

# signverse_subtitles_with_english.tar holds, per video: english.en.native.vtt (HUMAN English — source was already English), 
# english.en.nllb.vtt (English MACHINE-translated via Meta's NLLB — noisy), and original.<lang>.manual.vtt (the raw HUMAN 
# upload in video's own language; `manual` = human). The caption we want is in DATASET's target language (data.yaml target_lang), 
# preferring human over machine-translated — NOT hardcoded to English, so this generalises to future non-English-target corpora.
_ENGLISH_TRACKS = ("english.en.native.vtt", "english.en.nllb.vtt")  # SignVerse only auto-normalizes to English

def _target_code(target_lang: str) -> str:
    return str(target_lang or "en_XX").split("_")[0].lower()  # en_XX->en, de_DE->de, zh_CN->zh

def _lang_targets(data_cfg: dict, languages) -> dict[str, str]:
    langs = data_cfg.get("languages", {})
    return {lang: _target_code(langs.get(lang, {}).get("target_lang", "en_XX")) for lang in set(languages)}

def _is_caption_file(fname: str) -> bool:
    return fname in _ENGLISH_TRACKS or (fname.startswith("original.") and fname.endswith(".manual.vtt"))


def _pick_caption(files: dict[str, bytes], target_code: str) -> tuple[str, bytes] | None:
    """Best caption in the TARGET language → (provenance 'human'|'mt', vtt bytes), or None.

    Priority: (1) human caption in target language — english.en.native for target=en, else human original.<target>*.manual upload; 
    (2) NLLB machine-English, but ONLY for target=en (SignVerse produces no MT for other targets). A non-English target with no 
    human original.<target> caption is genuinely caption-less (the English tracks are the wrong language and are not used)."""
    if target_code == "en" and "english.en.native.vtt" in files: return "human", files["english.en.native.vtt"]
    for name, content in sorted(files.items()):  # human original upload IN the target language
        if name.startswith(f"original.{target_code}") and name.endswith(".manual.vtt"): return "human", content
    if target_code == "en" and "english.en.nllb.vtt" in files: return "mt", files["english.en.nllb.vtt"]
    return None


def _download(url: str, dest: Path, resume: bool = True) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {}
    start = tmp.stat().st_size if resume and tmp.exists() else 0
    if start: headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            # If we asked to resume but the server ignored Range (200, not 206), it is sending the WHOLE file —
            # appending it onto the partial would corrupt the tar. Restart from byte 0 in that case.
            append = start > 0 and r.status == 206
            with open(tmp, "ab" if append else "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
    except urllib.error.HTTPError as e:
        if e.code == 416 and tmp.exists(): pass  # already fully downloaded (range beyond EOF)
        else: raise
    tmp.rename(dest)
    return dest


def load_plan(split_csv: Path, root: Path, languages: list[str]) -> dict:
    # video→shard plan for the requested languages, from the split CSV + the authoritative upload index.
    # The index is DATASET metadata (like the split CSV), so it lives in root/ — NOT in the shard-tar cache.
    progress = root / "archive_upload_progress.json"
    if not progress.exists():
        print(f"prepare | fetching upload index → {progress}", flush=True)
        _download(f"{HF_BASE}/{PROGRESS_JSON}", progress)
    uploaded = json.loads(progress.read_text())["uploaded_folders"]

    with split_csv.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("sign_language") in set(languages)]
    plan = {"videos": {}, "missing": [], "shards": {}}
    for r in rows:
        vid, lang = r["video_id"], r["sign_language"]
        shard = uploaded.get(vid)
        if shard is None:
            plan["missing"].append({"video_id": vid, "language": lang})
            continue
        plan["videos"][vid] = {"language": lang, "shard": shard}
        plan["shards"].setdefault(shard, []).append(vid)
    return plan


def stage_plan(args, plan: dict) -> None:
    langs: dict[str, int] = {}
    for v in plan["videos"].values(): langs[v["language"]] = langs.get(v["language"], 0) + 1
    shards = sorted(plan["shards"])
    done = [s for s in shards if (Path(args.cache) / s).exists()]
    print(f"plan | videos: {len(plan['videos'])} ({langs}) | shards: {len(shards)} "
          f"| not yet uploaded upstream: {len(plan['missing'])}")
    print(f"plan | shards already downloaded: {len(done)}/{len(shards)} in {args.cache}")
    if args.size:  # opt-in HEAD probe (one request/undownloaded shard) — off by default (206 round-trips)
        total = sum((Path(args.cache) / s).stat().st_size for s in done)
        for s in (x for x in shards if x not in set(done)):
            try:
                req = urllib.request.Request(f"{HF_BASE}/dataset/{s}", method="HEAD")
                with urllib.request.urlopen(req) as r: total += int(r.headers.get("content-length", 0))
            except OSError: pass
        print(f"plan | estimated total download size: {total / 1e9:.1f} GB")
    print("plan | first shards: " + ", ".join(shards[:8]) + (" ..." if len(shards) > 8 else ""))


def stage_download(args, plan: dict) -> None:
    cache = Path(args.cache)
    shards = sorted(plan["shards"])
    if args.limit: shards = shards[: args.limit]
    for i, shard in enumerate(shards, 1):
        dest = cache / shard
        if dest.exists():
            print(f"download | [{i}/{len(shards)}] {shard} already present", flush=True)
            continue
        print(f"download | [{i}/{len(shards)}] {shard} ...", flush=True)
        _download(f"{HF_BASE}/dataset/{shard}", dest)


def _convert_one(tar: tarfile.TarFile, vid: str, lang_root: Path, tmp_dir: Path, tcode: str, subtitle_cfg: dict) -> dict | None:
    members = [m for m in tar.getmembers() if m.name.startswith(f"{vid}/")]
    if not any(m.name.startswith(f"{vid}/npz/") for m in members): return None
    tar.extractall(tmp_dir, members=members, filter="data")
    stats = convert_video(tmp_dir / vid / "npz", lang_root / "poses" / f"{vid}.npy")

    # Harvest the SINGLE best shard-bundled caption → one canonical `<vid>.<tcode>.vtt` (same file/name `--stage subs` writes, 
    # so both paths agree on exactly one caption per video). `best_subtitle` is the loader's selection rule; lang_prefix=tcode 
    # restricts it to TARGET-language shard tracks (so a non-en target never harvests an English track mislabelled as its own). 
    # No target-language track → caption_source "none" → `--stage subs` gap-fills the human original.<target> from subtitles tar.
    best = best_subtitle(tmp_dir / vid / "captions", vid, subtitle_cfg, lang_prefix=tcode) if (tmp_dir / vid / "captions").exists() else None
    if best is not None:
        subs_dir = lang_root / "subs"
        subs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best, subs_dir / f"{vid}.{tcode}.vtt")
        stats["caption_source"] = "shard"
    else: stats["caption_source"] = "none"
    shutil.rmtree(tmp_dir / vid, ignore_errors=True)
    return stats


def _backfill_meta(vid: str, lang_root: Path, meta: dict[str, dict]) -> None:
    # A prior run that crashed before save leaves a .npy with no meta row. Recompute duration from the .npy frame
    # count at the corpus's fixed 24 fps (cheap: mmap header only) so video_meta stays complete across resumes.
    if vid in meta: return
    npy = lang_root / "poses" / f"{vid}.npy"
    if not npy.exists(): return
    frames = int(np.load(npy, mmap_mode="r").shape[0])
    meta[vid] = {"video_id": vid, "duration_s": f"{frames / SIGNVERSE_DEFAULT_FPS:.3f}", "width": "", "height": ""}


def stage_convert(args, plan: dict) -> None:
    cache, root = Path(args.cache), Path(args.root)
    data_cfg = load_yaml(args.data_config)
    subtitle_cfg = data_cfg.get("subtitles", {}) or {}
    targets = _lang_targets(data_cfg, {v["language"] for v in plan["videos"].values()})
    per_lang_meta: dict[str, dict[str, dict]] = {}
    report = {"converted": 0, "skipped_existing": 0, "empty_heavy": [], "npz_missing": [], "not_downloaded": 0, "no_caption": 0}

    def _meta_for(lang: str) -> dict:
        return per_lang_meta.setdefault(lang, load_video_meta(root / lang / META_FILENAME))

    for shard in sorted(plan["shards"]):
        tar_path = cache / shard
        if not tar_path.exists():
            # Shard tar not present — a --limit smoke run, an interrupted download, or `--delete-tars` already freed it. 
            # Videos whose pose IS on disk were converted from that (now-deleted) tar: count them as already present 
            # (and backfill any meta a crash lost), NOT as "not downloaded" — else reconciliation advises a needless 
            # re-download for work already done. Only pose-less videos are genuinely not downloaded.
            touched = set()
            for vid in plan["shards"][shard]:
                lang = plan["videos"][vid]["language"]
                if (root / lang / "poses" / f"{vid}.npy").exists():
                    _backfill_meta(vid, root / lang, _meta_for(lang)); touched.add(lang)
                    report["skipped_existing"] += 1
                else: report["not_downloaded"] += 1
            for lang in touched: save_video_meta(root / lang / META_FILENAME, per_lang_meta[lang])
            continue

        in_shard = plan["shards"][shard]
        wanted = [v for v in in_shard if args.overwrite or not (root / plan["videos"][v]["language"] / "poses" / f"{v}.npy").exists()]
        # Even for fully-converted shards, backfill any meta rows a prior crash lost, then honor --delete-tars
        # (the "re-run convert --delete-tars to free disk" flow must delete these too).
        touched_langs = set()
        for vid in in_shard:
            if vid not in wanted:
                lang = plan["videos"][vid]["language"]
                _backfill_meta(vid, root / lang, _meta_for(lang))
                touched_langs.add(lang)
        report["skipped_existing"] += len(in_shard) - len(wanted)

        if wanted:
            print(f"convert | {shard}: {len(wanted)} video(s)", flush=True)
            tmp_dir = cache / "_extract"
            with tarfile.open(tar_path) as tar:
                for vid in wanted:
                    lang = plan["videos"][vid]["language"]
                    lang_root = root / lang
                    stats = _convert_one(tar, vid, lang_root, tmp_dir, targets.get(lang, "en"), subtitle_cfg)
                    if stats is None:
                        # Upstream inconsistency: the upload index lists this video in this shard, but the tar has no npz for it. 
                        # Nothing to retry — record it so the final tally reconciles.
                        report["npz_missing"].append(vid)
                        print(f"convert |   {vid}: npz absent from {shard} (upstream index/packaging gap) — skipped", flush=True)
                        continue
                    _meta_for(lang)[vid] = {
                        "video_id": vid, "duration_s": f"{stats['duration_s']:.3f}",
                        "width": str(stats["width"] or ""), "height": str(stats["height"] or ""), 
                        "caption_source": stats["caption_source"]
                    }
                    touched_langs.add(lang)
                    report["converted"] += 1
                    if stats["caption_source"] == "none": report["no_caption"] += 1
                    if stats["frames"] and stats["empty_frames"] / stats["frames"] > 0.5:
                        report["empty_heavy"].append((vid, round(stats["empty_frames"] / stats["frames"], 2)))

        # Persist meta AFTER EACH SHARD so a crash costs at most one shard, not the whole run.
        for lang in touched_langs: save_video_meta(root / lang / META_FILENAME, per_lang_meta[lang])
        if args.delete_tars: tar_path.unlink()

    shutil.rmtree(cache / "_extract", ignore_errors=True)  # remove the transient extraction scratch dir
    for lang, meta in per_lang_meta.items(): print(f"convert | {root / lang / META_FILENAME}: {len(meta)} rows", flush=True)

    # Full reconciliation so the counts add up without cross-referencing: every split video is accounted for as
    # converted, already-present, not-yet-uploaded-upstream, absent-from-its-shard, or in a not-downloaded shard.
    n_split = len(plan["videos"]) + len(plan["missing"])
    accounted = report["converted"] + report["skipped_existing"] + len(plan["missing"]) + len(report["npz_missing"]) + report["not_downloaded"]
    print(f"convert | reconciliation ({n_split} split videos = {accounted} accounted): converted {report['converted']} + "
          f"already present {report['skipped_existing']} + not uploaded upstream {len(plan['missing'])} + npz absent "
          f"from shard {len(report['npz_missing'])} + shard not downloaded {report['not_downloaded']}")
    if report["not_downloaded"]: print(f"convert | {report['not_downloaded']} video(s) are in shards not present in {cache} — "
                                       f"run `--stage download` (or `--stage all` without --limit) to fetch them.")
    print(f"convert | {report['converted'] - report['no_caption']} video(s) captioned from their shard track (caption_source=shard); "
          f"{report['no_caption']} without a usable shard caption → `--stage subs` gap-fills from subtitles tar (else the loader drops them)")
    if report["empty_heavy"]: print(f"convert | {len(report['empty_heavy'])} converted video(s) have >50% undetected-signer frames.")
    if report["npz_missing"]: print("convert | npz absent from shard (upstream gap, unrecoverable here): " + ", ".join(report["npz_missing"]))


def stage_subs(args, plan: dict) -> None:
    """GAP-FILL captions that `--stage convert` could not harvest from shard tracks, using the curated subtitles tar. A video already carrying 
    a `<vid>.<target>.vtt` in subs/ is left untouched (convert wrote the same canonical name from its shard track, caption_source=shard). 
    Only GAPS (no subs file) get a `<vid>.<target>.vtt` from the tar, choosing the caption in the dataset's target language (human original
    preferred; NLLB machine-English only for target=en). If there are no gaps, the 700 MB tar is never fetched. Provenance (human | mt) → 
    video_meta.csv `caption_source` so the loader can hold the test split to human refs."""
    root = Path(args.root)
    data_cfg = load_yaml(args.data_config)
    lang_target = _lang_targets(data_cfg, {info["language"] for info in plan["videos"].values()})

    # A GAP = a video whose POSE is present (converted) but which has no `<vid>.<target>.vtt` caption. Gating on the .npy avoids writing orphan 
    # captions for videos not yet downloaded/converted, and makes the "no gaps → skip the 700 MB tar" fast path fire on partial runs too. 
    # Filesystem-based, so deleting subs/ & re-running correctly re-fills from the tar even when video_meta still records caption_source=shard.
    gaps: dict[str, str] = {}  # vid -> language
    for vid, info in plan["videos"].items():
        lang = info["language"]
        if not (root / lang / "poses" / f"{vid}.npy").exists(): continue        # no pose → caption not useful yet
        if not (root / lang / "subs" / f"{vid}.{lang_target[lang]}.vtt").exists():
            gaps[vid] = lang
    if not gaps:
        print("subs | every converted video already has a `<vid>.<target>.vtt` (convert harvested the shard tracks) "
              "— no gaps, subtitles tar not needed.", flush=True)
        return

    # Reuse an already-downloaded tar (explicit --subs-tar, else root/) before fetching 700 MB into root/.
    candidates = [Path(args.subs_tar)] if getattr(args, "subs_tar", None) else [root / SUBTITLES_TAR]
    tar_path = next((p for p in candidates if p.exists()), None)
    if tar_path is None:
        tar_path = root / SUBTITLES_TAR
        print(f"subs | {len(gaps)} caption gap(s); fetching {SUBTITLES_TAR} (~700 MB, once) → {tar_path}", flush=True)
        _download(f"{HF_BASE}/{SUBTITLES_TAR}", tar_path)
    else: print(f"subs | {len(gaps)} caption gap(s); using existing {tar_path}", flush=True)

    buf: dict[str, dict[str, bytes]] = {}  # vid -> {caption filename: bytes}, only for gap videos
    with tarfile.open(tar_path) as tar:
        for m in tar:  # stream: subtitles/<vid>/<file>
            if not m.isfile(): continue
            parts = m.name.split("/")
            if len(parts) != 3 or parts[0] != "subtitles" or parts[1] not in gaps: continue
            if not _is_caption_file(parts[2]): continue  # english tracks + ALL original.<lang>.manual (target picks)
            f = tar.extractfile(m)
            if f is not None: buf.setdefault(parts[1], {})[parts[2]] = f.read()

    source_by_lang: dict[str, dict[str, str]] = {}  # lang -> {vid: human|mt|none}
    for vid, lang in gaps.items():
        tcode = lang_target[lang]
        picked = _pick_caption(buf.get(vid, {}), tcode)
        if picked is not None:
            source, content = picked  # human | mt
            subs_dir = root / lang / "subs"
            subs_dir.mkdir(parents=True, exist_ok=True)
            (subs_dir / f"{vid}.{tcode}.vtt").write_bytes(content)
        else: source = "none"  # no target-language caption anywhere → the loader drops this video
        source_by_lang.setdefault(lang, {})[vid] = source

    total_mt = 0
    for lang, by_vid in source_by_lang.items():
        meta = load_video_meta(root / lang / META_FILENAME)
        for vid, source in by_vid.items():
            # Ensure the row carries duration_s (gaps are pose-gated, so the .npy exists): a bare
            # {video_id, caption_source} row has no duration and load_video_meta drops rows without one — which
            # would silently erase this video's provenance and let an `mt` caption slip past human_only_splits.
            _backfill_meta(vid, root / lang, meta)
            meta.setdefault(vid, {"video_id": vid})["caption_source"] = source
        save_video_meta(root / lang / META_FILENAME, meta)
        c = {s: sum(v == s for v in by_vid.values()) for s in ("human", "mt", "none")}
        total_mt += c["mt"]
        print(f"subs | {lang}: filled {c['human']} human + {c['mt']} NLLB-MT; {c['none']} still caption-less "
              f"(dropped by loader). Provenance → {root / lang / META_FILENAME} caption_source.", flush=True)
    if total_mt: print("subs | NLLB-machine-translated captions are noisy SLT targets — the loader scores the TEST split "
                       "against human references only (subtitles.human_only_splits + caption_source in video_meta.csv).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignVerse-2M → repo language layout (poses/ + subs/ + video_meta.csv)")
    parser.add_argument("--stage", default="all", choices=["plan", "download", "convert", "subs", "all"])
    parser.add_argument("--languages", nargs="+", default=["asf", "bfi"])
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--data-config", default="configs/data.yaml", help="reads languages[lang].target_lang for the caption language")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="shard tar cache dir")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="language roots parent (root/<lang>/poses etc.)")
    parser.add_argument("--limit", type=int, default=0, help="stop after N shards (smoke runs)")
    parser.add_argument("--size", action="store_true", help="plan: HEAD-probe undownloaded shards for a size estimate")
    parser.add_argument("--subs-tar", default=None, help="path to an already-downloaded signverse_subtitles_with_english.tar")
    parser.add_argument("--delete-tars", action="store_true",
                        help="delete each shard tar after conversion. Shards are MIXED-language, so include every "
                             "language you will ever want in ONE run (e.g. --languages asf bfi) — a later run for a "
                             "language whose videos sat in an already-deleted shard would have to re-download it.")
    parser.add_argument("--overwrite", action="store_true", help="re-convert videos whose .npy already exists")
    args = parser.parse_args()

    split_csv = Path(args.split_csv)
    if not split_csv.exists(): sys.exit(f"split CSV not found: {split_csv}")
    plan = load_plan(split_csv, Path(args.root), list(args.languages))
    if plan["missing"]: print(f"prepare | {len(plan['missing'])} video(s) not yet uploaded upstream (no failure markers) — skipped")
    if args.stage in ("plan",): stage_plan(args, plan)
    if args.stage in ("download", "all"): stage_download(args, plan)
    if args.stage in ("convert", "all"): stage_convert(args, plan)
    if args.stage in ("subs", "all"): stage_subs(args, plan)
