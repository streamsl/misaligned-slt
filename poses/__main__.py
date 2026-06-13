# CLI: python -m poses <lang_root> [--format SEL]   (e.g. python -m poses data/youtube-sl-25/asf)
# Builds <lang_root>/video_meta.csv from YouTube metadata (yt-dlp, no video download) — see
# poses.pose_io.build_video_meta. Pass --format the SAME selector the videos were downloaded
# with so the width/height columns describe the downloaded stream.
# Lives in __main__.py (not pose_io) because `python -m poses.pose_io` raises RuntimeWarning:
# the poses package __init__ imports pose_io before runpy executes it as __main__.
import argparse
from poses.pose_io import YTDLP_FORMAT, build_video_meta

parser = argparse.ArgumentParser(prog="python -m poses", description="Build <lang_root>/video_meta.csv")
parser.add_argument("lang_root", help="language data root, e.g. data/youtube-sl-25/asf")
parser.add_argument("--format", default=YTDLP_FORMAT,
                    help="yt-dlp format selector; use the one the videos were downloaded with")
args = parser.parse_args()
build_video_meta(args.lang_root, ytdlp_format=args.format)
