from __future__ import annotations
import argparse, json

from IPython.display import display

import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)         # show all columns
pd.set_option("display.max_colwidth", None)        # don't truncate long text in cells
pd.set_option("display.width", None)               # auto-detect terminal width
pd.set_option("display.expand_frame_repr", False)  # don't wrap columns into blocks

from data.loader import load_language_records
from evaluation.predictions import write_gold_segments
from evaluation.replays import run_resplit, run_separator_replay
from evaluation.rq1 import run_rq1
from evaluation.rq2 import run_rq2
from evaluation.segmenter_eval import run_segmenter_eval
from utils import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Misaligned-SLT evaluation entry points")
    parser.add_argument("--rq", choices=["1", "2"], default=None)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--method-config", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--method", default="dlm", choices=["baseline", "ar", "dlm"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--severity-grid-s", default=None,
                        help="Comma-separated signed RQ1 grid (fractions of duration in relative mode, else seconds)")
    # Fall back to --severity-grid-s / eval.yaml. One shared grid's cross-product cannot express a truncation-only sweep.
    parser.add_argument("--severity-grid-head", default=None,
                        help="RQ1 head-axis grid override (use --severity-grid-head=-0.3,0 for negative-leading values)")
    parser.add_argument("--severity-grid-tail", default=None,
                        help="RQ1 tail-axis grid override (use --severity-grid-tail=-0.3,0 for negative-leading values)")
    parser.add_argument("--severity-mode", default=None, choices=["relative", "absolute"],  # None -> eval.yaml rq1.severity_mode
                        help="RQ1 perturbation: relative (fraction of sentence duration, default) or absolute seconds")
    parser.add_argument("--emit-gold-segments", default=None, 
                        help="Write GT spans JSON (for --rq 2 --segments oracle-input rows) and exit")
    parser.add_argument("--segmenter-eval", action="store_true",
                        help="Standalone whole-video segmentation eval (Moryossef protocol) for --segmenter-arch, then exit")
    parser.add_argument("--resplit", default=None, choices=["likelihood", "attention", "own-attention"],
                        help="cut every split span of --events again by the clean floor's likelihood or cross-attention, or by the "
                             "cross-attention of the AR arm that wrote the events (own-attention, --checkpoint = that arm; infer.unit_split)")
    parser.add_argument("--events", default=None, help="--resplit: a rows-6/10 RQ2 events file; --separator-replay: the same run's RQ2 "
                        "events file (its deployed commit times)")
    parser.add_argument("--fit-layer", action="store_true", help="--resplit attention, train only: the mean cut error of every "
                        "candidate layer set (the rule that sets eval.yaml attention_cut.layers)")
    parser.add_argument("--separator-replay", action="store_true",
                        help="Error analysis, then exit: replay a text-side unit commit (infer/unit_split.py) on the open-span decodes "
                             "of --shadow-file and score it against gold and the same run's --events. Changes no deployed event")
    parser.add_argument("--shadow-file", default=None, help="--separator-replay: outputs/shadow/shadow_*.json from --rq 2 --online --shadow-open-spans")
    parser.add_argument("--num-beams", type=int, default=None, help="AR search override for this run; main comparisons default to greedy")
    parser.add_argument("--segmenter-decode", choices=["plain", "duration"], default=None,
                        help="BIO decoder (default duration: legal BIO + train-fitted duration prior, both segmenters). "
                             "With --online it selects the online cascade's decoder (plain = legal paths only, no duration scores); "
                             "a joint arm always uses its trained duration model")
    parser.add_argument("--segmenter-arch", default=None, choices=["moryossef", "s1"],
                        help="Standalone segmenter (default moryossef). With --online, select an online cascade using --method baseline; "
                             "omit for a joint AR/DLM head")
    parser.add_argument("--moryossef-config", default="configs/moryossef26.yaml", help="Moryossef segmenter config")
    parser.add_argument("--bio-config", default="configs/bio_pretrain.yaml", help="S1 (in-system head) config for --segmenter-arch s1")
    parser.add_argument("--segmenter-init", default="finetuned", choices=["finetuned", "released", "random"],
                        help="moryossef only: 'released' evaluates the untrained released weights under their own input contract "
                             "(moryossef26/dataset.py), 'random' same architecture untrained at --seed. Diagnostics read as a PAIR, "
                             " not a baseline: quote released against the spread of several random seeds.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="--segmenter-init random only: init seed, stamped in the payload and artifact name (run several to get the range)")
    parser.add_argument("--num-sentences", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16, help="windows per translate/forward batch in loop-decode paths")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--predictions", default=None,
                        help="RQ2: re-score a saved events JSON (outputs/rq2_offline/ or outputs/rq2_online/) without re-running the model")
    parser.add_argument("--segments", default=None, help="Segmenter spans JSON (analyze --stage segmenter-infer)")
    parser.add_argument("--offline", action="store_true",
                        help="Final misaligned model self-segments each whole video offline (its own BIO head) and translates each span")
    parser.add_argument("--online", action="store_true", help="The online row: the streaming FSM over each stream produces the events")
    parser.add_argument("--no-translate", action="store_true", 
                        help="--online only: segmentation-only dry run (no decoder calls; text metrics are 0).")
    parser.add_argument("--stability", action="store_true",
                        help="With --online: also score stable-prefix display policies (LA-n vs confidence) from per-stride candidate decodes. "
                             "Recording turns the FSM's decode skip off, so it decodes every stride that has a candidate span. "
                             "Writes outputs/stability/stability_*.json (policy scores) and provisional_*.json (every decode as text).")
    parser.add_argument("--shadow-open-spans", action="store_true",
                        help="With --online (translating): separator-commit replay, error analysis only. Every stride with no closed span "
                             "also decodes its open span (never emitted) and writes outputs/shadow/shadow_*.json for `eval.py --separator-replay`; "
                             "events, scores and stride times are unchanged.")
    parser.add_argument("--tiou-thresholds", default=None, help="Comma-separated RQ2 tIoU thresholds")
    parser.add_argument("--no-densevid", action="store_true", help="RQ2: skip the densevid_eval headline columns (SODA rows only)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-test", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    data_cfg = load_yaml(args.data_config)
    if args.language is None: args.language = str(data_cfg.get("active_languages", ["asf"])[0])
    if args.emit_gold_segments:
        records, _ = load_language_records(data_cfg, args.language, split=args.split)
        path = write_gold_segments(records, args.emit_gold_segments)
        result = {
            "emit_gold_segments": str(path), "videos": len({r.pose.video_id for r in records}), 
            "segments": sum(1 for r in records for sp in r.sentences if getattr(sp, "reliable", True))
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0)
    if args.segmenter_eval:
        print(json.dumps(run_segmenter_eval(args), indent=2, sort_keys=True))
        raise SystemExit(0)
    if args.resplit:
        run_resplit(args)
        raise SystemExit(0)
    if args.separator_replay:
        run_separator_replay(args)
        raise SystemExit(0)

    if args.rq == "1": result = run_rq1(args)
    elif args.rq == "2": result = run_rq2(args)
    else: raise SystemExit(
        "eval.py needs a mode: --rq 1 (controlled misalignment sweep), --rq 2 (end-to-end ladder), "
        "--segmenter-eval (whole-video segmentation), --resplit (cut split spans again), "
        "--separator-replay (replay a separator commit), or --emit-gold-segments (write GT spans)."
    )
    display(result)
