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
from evaluation.rows import run_row
from evaluation.segmenter_eval import run_segmenter_eval
from utils import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Misaligned-SLT evaluation entry points")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--method-config", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--source-language", default=None,
                        help="Zero-shot rows and --segmenter-eval: the corpus the models trained on (default --language). Their method "
                             "config, stamp checks, warm start (data.yaml pretrained_slt), train-label warnings and duration prior are this "
                             "corpus's; the data, gold, test guard and text scoring stay --language's, and the artifacts are named "
                             "<source>-to-<language>. --predictions and --emit-gold-segments build no model and refuse it.")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--method", default="dlm", choices=["baseline", "ar", "dlm"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--emit-gold-segments", default=None, 
                        help="Write GT spans JSON (for the --segments rows on true cuts) and exit")
    parser.add_argument("--segmenter-eval", action="store_true",
                        help="Standalone whole-video segmentation eval (Moryossef protocol) for --segmenter-arch, then exit")
    parser.add_argument("--num-beams", type=int, default=None, help="AR search override for this run; main comparisons default to greedy")
    parser.add_argument("--segmenter-decode", choices=["plain", "duration"], default=None,
                        help="BIO decoder (default duration: legal BIO + train-fitted duration prior, both segmenters). "
                             "With --online it selects the online cascade's decoder (plain = legal paths only, no duration scores); "
                             "a joint arm always uses its trained duration model")
    parser.add_argument("--segmenter-arch", default=None, choices=["moryossef", "s1"],
                        help="Standalone segmenter (default moryossef). With --online, "
                             "select an online cascade using --method baseline; omit for a joint AR/DLM head")
    parser.add_argument("--moryossef-config", default="configs/moryossef26.yaml", help="Moryossef segmenter config")
    parser.add_argument("--bio-config", default="configs/bio_pretrain.yaml", help="S1 (in-system head) config for --segmenter-arch s1")
    parser.add_argument("--segmenter-init", default="finetuned", choices=["finetuned", "released", "random"],
                        help="moryossef only: 'released' evaluates the untrained released weights under their own input contract "
                             "(moryossef26/dataset.py), 'random' same architecture untrained at --seed. Diagnostics read as a PAIR, "
                             " not a baseline: quote released against the spread of several random seeds.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="--segmenter-init random only: init seed, stamped in the payload and artifact name (run several to get the range)")
    parser.add_argument("--batch-size", type=int, default=16, help="windows per translate/forward batch in loop-decode paths")
    parser.add_argument("--predictions", default=None,
                        help="Re-score a saved events JSON (outputs/offline/ or outputs/online/) without re-running the model, or score "
                             "a span file (analyze.py --stage segmenter-infer or chance-spans) as a segmentation-only row")
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
    parser.add_argument("--tiou-thresholds", default=None, help="Comma-separated tIoU thresholds of the tIoU scores")
    parser.add_argument("--no-densevid", action="store_true", 
                        help="Skip the tIoU DVC columns of densevid_eval (the main edge chunk scores always run)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-test", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    data_cfg = load_yaml(args.data_config)
    if args.language is None: args.language = str(data_cfg["active_languages"][0])
    if args.source_language not in (None, *data_cfg["languages"]): raise SystemExit(
        f"--source-language {args.source_language!r} is not a corpus of {args.data_config} (languages: {', '.join(data_cfg['languages'])})"
    )
    if args.source_language is not None and (args.emit_gold_segments or args.predictions): raise SystemExit(
        "--source-language names the corpus of the models that a run builds; --emit-gold-segments and --predictions build none"
    )
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

    if not (args.online or args.offline or args.segments or args.predictions): raise SystemExit(
        "eval.py needs a mode: a row (--online, --offline, --segments or --predictions), --segmenter-eval (whole-video "
        "segmentation) or --emit-gold-segments (write GT spans). The boundary sweep is analyze.py --stage boundary-sweep."
    )
    display(run_row(args))
