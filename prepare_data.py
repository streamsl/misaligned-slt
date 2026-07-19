"""SignVerse-2M downloader + converter → the repo's YouTube-SL-25 language layout.

Fetches exactly the shards containing the requested languages' videos from HuggingFace SignerX/SignVerse-2M
(shards are mixed-language; asf+bfi span ~206 of 723 shards, ~80 GB vs ~1.3 TB full corpus), converts each
wanted video's DWPose-128 npz (either packaging scheme) to the canonical (T,133,3) raw-pixel .npy via
poses.signverse, bundles per-video captions into subs/, and writes video_meta.csv (duration from the npz frame
count at the corpus's unified 24 fps — no yt-dlp needed). After this, the language behaves exactly like the
repo's own extractions: train.py / analyze.py / eval.py run unchanged with --language asf|bfi.

    python prepare_data.py --stage plan [--size]              # shard/video counts (+ HEAD size estimate); no download
    python prepare_data.py --stage all --languages asf bfi    # download + convert (resumable; skips done work)
    python prepare_data.py --stage convert --delete-tars      # convert already-downloaded shards, free disk

The authoritative video→shard index is runtime_state/archive_upload_progress.json::uploaded_folders (the other
manifests in the repo are stale snapshots — verified). ~19 asf/bfi videos are not yet uploaded upstream (no
failure markers; simply past the upload frontier) and are reported, not treated as errors.
"""
from __future__ import annotations
import argparse, csv, json, shutil, sys, tarfile, urllib.request
import numpy as np

from pathlib import Path
from poses.pose_io import META_FILENAME, load_video_meta, save_video_meta
from poses.signverse import SIGNVERSE_DEFAULT_FPS, convert_video

HF_BASE = "https://huggingface.co/datasets/SignerX/SignVerse-2M/resolve/main"
PROGRESS_JSON = "runtime_state/archive_upload_progress.json"
DEFAULT_SPLIT_CSV = "data/youtube-sl-25/SignVerse-2M-metadata_split.csv"
DEFAULT_CACHE = "data/youtube-sl-25/signverse_shards"
DEFAULT_ROOT = "data/youtube-sl-25"


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


def load_plan(split_csv: Path, cache: Path, languages: list[str]) -> dict:
    # video→shard plan for the requested languages, from the split CSV + the authoritative upload index.
    progress = cache / "archive_upload_progress.json"
    if not progress.exists():
        print(f"prepare | fetching upload index → {progress}", flush=True)
        _download(f"{HF_BASE}/{PROGRESS_JSON}", progress)
    uploaded = json.loads(progress.read_text())["uploaded_folders"]

    with split_csv.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("sign_language") in set(languages)]
    plan = {"videos": {}, "missing": [], "shards": {}}
    for r in rows:
        vid, lang, split = r["video_id"], r["sign_language"], r["split"]
        shard = uploaded.get(vid)
        if shard is None:
            plan["missing"].append({"video_id": vid, "language": lang, "split": split})
            continue
        plan["videos"][vid] = {"language": lang, "split": split, "shard": shard}
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


def _convert_one(tar: tarfile.TarFile, vid: str, lang_root: Path, tmp_dir: Path) -> dict | None:
    members = [m for m in tar.getmembers() if m.name.startswith(f"{vid}/")]
    if not any(m.name.startswith(f"{vid}/npz/") for m in members): return None
    tar.extractall(tmp_dir, members=members, filter="data")
    stats = convert_video(tmp_dir / vid / "npz", lang_root / "poses" / f"{vid}.npy")

    subs_dir = lang_root / "subs"
    for vtt in sorted((tmp_dir / vid / "captions").glob("*.vtt")) if (tmp_dir / vid / "captions").exists() else []:
        subs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vtt, subs_dir / vtt.name)
        stats["caption"] = vtt.name
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
    per_lang_meta: dict[str, dict[str, dict]] = {}
    report = {"converted": 0, "skipped_existing": 0, "no_caption": [], "empty_heavy": []}

    def _meta_for(lang: str) -> dict:
        return per_lang_meta.setdefault(lang, load_video_meta(root / lang / META_FILENAME))

    for shard in sorted(plan["shards"]):
        tar_path = cache / shard
        if not tar_path.exists(): continue
        in_shard = plan["shards"][shard]
        wanted = [v for v in in_shard
                  if args.overwrite or not (root / plan["videos"][v]["language"] / "poses" / f"{v}.npy").exists()]
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
                    stats = _convert_one(tar, vid, lang_root, tmp_dir)
                    if stats is None:
                        print(f"convert |   {vid}: npz missing in shard — skipped", flush=True)
                        continue
                    _meta_for(lang)[vid] = {"video_id": vid, "duration_s": f"{stats['duration_s']:.3f}",
                                            "width": str(stats["width"] or ""), "height": str(stats["height"] or "")}
                    touched_langs.add(lang)
                    report["converted"] += 1
                    if "caption" not in stats: report["no_caption"].append(vid)
                    if stats["frames"] and stats["empty_frames"] / stats["frames"] > 0.5:
                        report["empty_heavy"].append((vid, round(stats["empty_frames"] / stats["frames"], 2)))

        # Persist meta AFTER EACH SHARD so a crash costs at most one shard, not the whole run.
        for lang in touched_langs: save_video_meta(root / lang / META_FILENAME, per_lang_meta[lang])
        if args.delete_tars: tar_path.unlink()

    for lang, meta in per_lang_meta.items():
        print(f"convert | {root / lang / META_FILENAME}: {len(meta)} rows", flush=True)
    print(f"convert | converted {report['converted']} | already present {report['skipped_existing']} "
          f"| without bundled caption {len(report['no_caption'])} | >50% empty frames {len(report['empty_heavy'])}")
    if report["no_caption"]:
        print("convert | caption-less videos (fetch signverse_subtitles_with_english.tar or use your own subs/): "
              + ", ".join(report["no_caption"][:20]) + (" ..." if len(report["no_caption"]) > 20 else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignVerse-2M → repo language layout (poses/ + subs/ + video_meta.csv)")
    parser.add_argument("--stage", default="all", choices=["plan", "download", "convert", "all"])
    parser.add_argument("--languages", nargs="+", default=["asf", "bfi"])
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="shard tar cache dir")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="language roots parent (root/<lang>/poses etc.)")
    parser.add_argument("--limit", type=int, default=0, help="stop after N shards (smoke runs)")
    parser.add_argument("--size", action="store_true", help="plan: HEAD-probe undownloaded shards for a size estimate")
    parser.add_argument("--delete-tars", action="store_true", help="delete each shard tar after conversion")
    parser.add_argument("--overwrite", action="store_true", help="re-convert videos whose .npy already exists")
    args = parser.parse_args()

    split_csv = Path(args.split_csv)
    if not split_csv.exists(): sys.exit(f"split CSV not found: {split_csv}")
    plan = load_plan(split_csv, Path(args.cache), list(args.languages))
    if plan["missing"]:
        print(f"prepare | {len(plan['missing'])} video(s) not yet uploaded upstream (no failure markers) — skipped")

    if args.stage in ("plan",): stage_plan(args, plan)
    if args.stage in ("download", "all"): stage_download(args, plan)
    if args.stage in ("convert", "all"): stage_convert(args, plan)
