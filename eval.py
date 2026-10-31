from __future__ import annotations
import argparse

from IPython.display import display

import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)         # show all columns
pd.set_option("display.max_colwidth", None)        # don't truncate long text in cells
pd.set_option("display.width", None)               # auto-detect terminal width
pd.set_option("display.expand_frame_repr", False)  # don't wrap columns into blocks

from evaluation.rows import run_row
from utils import load_yaml


def build_parser() -> argparse.ArgumentParser:
    # No abbreviated flags: a prefix never selects another flag.
    parser = argparse.ArgumentParser(
        description="Misaligned-SLT evaluation: one row per command. The flags are the fields of the row's name, "
                    "{when}_{spans}_{decode}_{text}_{corpus}_{split}: python eval.py --offline|--online --spans SPANS --text TEXT "
                    "[--decode DECODE] --language L --split S", allow_abbrev=False)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--method-config", default=None,
                        help="The method config of the row's translator (--text), or with --text none of the arm of --spans; default "
                             "configs/baseline_eval.yaml (pretrimmed), ar.yaml or dlm.yaml. Its stem goes into the row's name")
    parser.add_argument("--moryossef-config", default="configs/moryossef26.yaml", help="The Moryossef segmenter's config (--spans moryossef)")
    parser.add_argument("--bio-config", default="configs/bio_pretrain.yaml", help="S1's config (--spans s1)")
    parser.add_argument("--language", default=None)
    parser.add_argument("--source-language", default=None,
                        help="Zero-shot rows: the corpus the models trained on (default --language). Their method config, stamp checks, "
                             "warm start (data.yaml pretrained_slt), train-label warnings and duration prior are this corpus's; the data, "
                             "gold, test guard and text scoring stay --language's, and the artifacts are named <source>-to-<language>. "
                             "A row that builds no model refuses it")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    when = parser.add_mutually_exclusive_group()
    when.add_argument("--offline", action="store_true", help="Whole videos: each stream is cut once, offline")
    when.add_argument("--online", action="store_true",
                      help="The streaming FSM decides from the past alone (the spans of --spans moryossef, s1, ar or dlm)")
    parser.add_argument("--spans", default=None,
                        help="Where the spans come from: gold (the reliable gold units; offline), moryossef or s1 (a standalone "
                             "segmenter), ar or dlm (the head of that arm), chance (spans of the median train unit length, made by no "
                             "model; offline), or the path of a span or events file (its spans, offline or online as its provenance says)")
    parser.add_argument("--text", default=None, choices=["none", "pretrimmed", "ar", "dlm"],
                        help="The translator: none (the spans alone, no decoder call), pretrimmed (the pretrimmed translator, "
                             "trained on caption clips cut at the true edges: baseline_eval.yaml decoder, from data.yaml "
                             "pretrained_slt), ar or dlm (that arm's translator)")
    parser.add_argument("--decode", default=None, choices=["argmax", "viterbi", "duration"],
                        help="The decode of a standalone segmenter (--spans moryossef or s1): argmax (each frame its most likely tag, "
                             "offline and online), viterbi (the best legal BIO path) or duration (the best legal BIO path with the "
                             "train-fitted duration prior, the semi-Markov decode). Default: the decode that selected the segmenter "
                             "checkpoint (its monitor_decode stamp: argmax, offline and online, for an argmax stamp, else duration). "
                             "An arm's head always decodes with its trained duration model")
    parser.add_argument("--segmenter-checkpoint", default=None,
                        help="The segmenter of --spans moryossef or s1 (default: its pooled folder); for an untrained control, the "
                             "fine-tuned Moryossef file whose input table it reads")
    parser.add_argument("--arm-checkpoint", default=None, help="The arm of --spans or --text ar/dlm (default: the arm's checkpoint.dir)")
    parser.add_argument("--pretrimmed-checkpoint", default=None,
                        help="The pretrimmed translator of --text pretrimmed (default: data.yaml pretrained_slt of baseline_eval.yaml "
                             "decoder); a pretrimmed DLM translator names the row pretrimmed-dlm")
    parser.add_argument("--segmenter-init", default="finetuned", choices=["finetuned", "released", "random"],
                        help="--spans moryossef --text none only: 'released' evaluates the untrained released weights under their own "
                             "input contract (moryossef26/dataset.py), 'random' same architecture untrained at --seed. The spans field of "
                             "the row's name is moryossef-released or moryossef-random<seed>. Diagnostics read as a PAIR, not a baseline: "
                             "quote released against the spread of several random seeds.")
    parser.add_argument("--seed", type=int, default=42,
                        help="--segmenter-init random only: init seed, stamped in the provenance and the artifact name (run several to get the range)")
    parser.add_argument("--num-beams", type=int, default=None, help="AR search override for this run; main comparisons default to greedy")
    parser.add_argument("--batch-size", type=int, default=16, help="windows per translate/forward batch in loop-decode paths")
    parser.add_argument("--predictions", default=None,
                        help="Score a saved events file again (outputs/offline/ or outputs/online/), with no model run and no new file: "
                             "the main score's grid only, as every row prints it")
    parser.add_argument("--stability", action="store_true",
                        help="An online row whose FSM translates: also score stable-prefix display policies (LA-n vs confidence) from "
                             "per-stride candidate decodes. Recording turns the FSM's decode skip off, so it decodes every stride that has "
                             "a candidate span. Writes outputs/stability/stability_*.json (policy scores) and provisional_*.json (every decode as text).")
    parser.add_argument("--tiou-thresholds", default=None,
                        help="Comma-separated tIoU thresholds (default eval.yaml scoring.tiou_thresholds): the tIoU columns of the "
                             "main score's grid (key main.edge_chunk of the _scores.json; the main score is its F1 at edge tau avg and "
                             "tIoU avg), the thresholds of the densevid scores (key densevid) and the run scores of a no-text row's "
                             "frame block (key frame). Other thresholds give another main score, so a row on them needs --output")
    parser.add_argument("--no-densevid", action="store_true",
                        help="Skip the DVC columns of densevid_eval (key densevid; the main scores always run)")
    parser.add_argument("--output", default=None,
                        help="The events file of the row (its _scores.json goes beside it); default outputs/<when>/<the row's name>.json")
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
    table = run_row(args)   # the main score; its notes go below it
    display(table)
    print("\n".join(table.attrs["notes"]), flush=True)
