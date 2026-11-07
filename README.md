# Misaligned SLT (SayWhen)

This repository trains and evaluates SayWhen. SayWhen reads body poses of a sign language video and writes timed English captions while the video streams in. It uses no glosses, and it stays useful when its own sentence cuts are wrong.

This README is a runbook. It gives the commands that fill the two main tables of the paper, in the order to run them:

- **Table 1**: the YouTube-SL-25 test sets (`ase`, `asf`, `bfi`).
- **Table 2**: the How2Sign, Auslan-Daily (News) and BOBSL (manual Sent-Test) test sets, in 2 blocks: "Trained on the benchmark" and "Transfer from YouTube-SL-25".

Each step gives the commands, what the step writes, and how to check it. The rules and the reasons are in [`streaming_slt_prompt.md`](streaming_slt_prompt.md) (the design spec) and in `docs/`.

**Main path:** [1. Setup](#1-setup) · [2. Rules for every run](#2-rules-for-every-run) · [3. Data checks](#3-data-checks-a0-a0b) · [4. Segmenters](#4-segmenters-a1-a2) · [5. Models of each corpus](#5-models-of-each-corpus-b2a-b2b-b3) · [6. Decode choice on dev](#6-choose-the-decode-of-each-system-on-dev) · [7. Table 1](#7-table-1-youtube-sl-25-test) · [8. Table 2](#8-table-2-benchmark-test) · [9. Where each number is](#9-where-each-number-is)

**Other:** [10. Other runs](#10-other-runs) · [11. After a change](#11-after-a-change-what-to-run-again) · [12. Reference](#12-reference)

## Run order

| Step | What | Runs | Machine |
| --- | --- | --- | --- |
| §3 (A0, A0b) | Prepare and check the data | once | any, no GPU |
| §4 (A1, A2) | Train the Moryossef segmenter and SayWhen-Seg on the pool | 1 run each | CUDA GPU |
| §5.1 (B2a) | Train Uni-Sign on `ase`, `asf`, `bfi`, `how2sign`, `auslan_daily`, `bobsl_auto` | 6 runs | CUDA GPU |
| §5.2 (B2b) | Point `configs/data.yaml` at the 6 Uni-Sign files | 1 edit | every machine |
| §5.3 (B3) | Train SayWhen on `ase`, `asf`, `bfi`, `how2sign`, `auslan_daily`, `bobsl_auto` | 6 runs | CUDA GPU with much memory |
| §6 | Choose the decode of each system on dev | 12 no-text rows per corpus | GPU or Mac |
| §7 | Table 1: the test rows of `ase`, `asf`, `bfi` | 7 commands and 1 report per corpus | GPU or Mac |
| §8 | Table 2: the test rows of the benchmarks, then the transfer rows | 7 + 6 commands and 2 reports per benchmark | GPU or Mac |
| §9 | Copy the numbers into the tables | — | any |

The labels mark the steps: A0, A0b, A1 and A2 run once, and B2a to B3 run for each corpus. Configs and code messages use them. §4 and §5.1 do not depend on each other, so run them at the same time.

## Names

| Name in the paper | What it is | Train command | Checkpoint | `eval.py` flag | In file names |
| --- | --- | --- | --- | --- | --- |
| Moryossef | The external segmenter of Moryossef et al., trained again on the pool with the input rules and decode of its release | `train.py --stage train-moryossef` | `checkpoints/moryossef/multi_ase-asf-bfi-bobsl_auto/model.pt` | `--spans moryossef` | `moryossef` |
| SayWhen-Seg | Our segmenter, pretrained on the pool (S1 in the code) | `train.py --stage train-bio` | `checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto/model.pt` | `--spans s1` | `s1` |
| Uni-Sign | The pre-trimmed translator: Uni-Sign's OpenASL release, fine-tuned on the gold sentences of one corpus | `train.py --stage train-slt --slt-config configs/baseline_train.yaml` | `checkpoints/baseline_train_ar/<corpus>/model.pt` | `--text pretrimmed` | `pretrimmed` |
| SayWhen | Segmentation and translation trained together: two pose streams, AR decoder (stage 2) | `train.py --stage train-slt --slt-config configs/ar.yaml` | `checkpoints/ar/<corpus>/model.pt` | `--spans ar --text ar` | `joint-ar` (spans), `ar` (text) |

| Corpus key | Data | Use |
| --- | --- | --- |
| `ase`, `asf`, `bfi` | YouTube-SL-25 ASL, Auslan and BSL, with the SignVerse-2M poses | Table 1, the pool, and the sources of the transfer rows |
| `how2sign` | How2Sign (ASL) | Table 2 |
| `auslan_daily` | Auslan-Daily News, re-split (Auslan) | Table 2 |
| `bobsl_auto` | BOBSL under the LiTFiC protocol: it trains on the automatic captions of the official train split, and its dev and test are the hand-aligned Sent-Val and Sent-Test | the pool, and Table 2 |
| `bobsl_manual` | The hand-aligned BOBSL captions: the dev and test labels of `bobsl_auto`. No command of this guide names it, except the copy check | none |

Words used below:

- **pool**: `ase`, `asf`, `bfi` and `bobsl_auto`. SayWhen-Seg and the Moryossef segmenter each train on it once.
- **row**: one `eval.py` command. It captions whole videos of one split and writes 2 files: the **events file** `<name>.json` (the timed captions) and the **scores file** `<name>_scores.json`.
- **access**: `--offline` reads each whole video at once. `--online` runs the streaming engine, which decides from the past alone.
- **decode**: how a segmenter turns its frame scores into sentence spans: `argmax`, `viterbi` or `duration` (the semi-Markov decode).
- **no-text row**: a row with `--text none`. It gives the spans alone, with no translation. It is not a row of a no-text arm (§10.2).

## 1. Setup

**Needs.** Python 3.9 to 3.13. A CUDA GPU to train. A Mac (MPS) or CPU machine is enough to evaluate. The code selects `cuda`, then `mps`, then `cpu`. `--device` changes this.

Install the packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the released files into `checkpoints/`. The code does not download them.

```bash
mkdir -p checkpoints
curl -L -o checkpoints/openasl_pose_only_slt.pth \
    https://huggingface.co/ZechengLi19/Uni-Sign/resolve/eab251b7fe7e8521afc0e67be98add670ea40a0d/openasl_pose_only_slt.pth
curl -L -o checkpoints/moryossef26.safetensors \
    https://raw.githubusercontent.com/sign-language-processing/segmentation/a4596048fadf22864e2fdd6a2ef3ca1004835737/sign_language_segmentation/dist/2026/model.safetensors
```

| File | What it is | Used by |
| --- | --- | --- |
| `checkpoints/openasl_pose_only_slt.pth` | Uni-Sign's OpenASL pose-only translator ([Uni-Sign](https://github.com/ZechengLi19/Uni-Sign)) | SayWhen-Seg (A2), and Uni-Sign of every corpus (B2a) |
| `checkpoints/moryossef26.safetensors` | The released weights of the Moryossef segmenter ([sign-language-processing/segmentation](https://github.com/sign-language-processing/segmentation)) | The Moryossef segmenter (A1) |

The first run also downloads mT5 (`google/mt5-base`), the NLTK `punkt` and `punkt_tab` data, and the METEOR metric. For a machine with no internet, run these commands on a machine with internet, then copy `~/.cache/huggingface` and `~/nltk_data` to the other machine:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python -c "import evaluate; evaluate.load('meteor')"
python -c "from transformers import T5Tokenizer; T5Tokenizer.from_pretrained('google/mt5-base', legacy=False)"
python -c "from transformers import MT5ForConditionalGeneration; MT5ForConditionalGeneration.from_pretrained('google/mt5-base')"
```

**BLEURT.** The B-RT column needs it on every machine that evaluates. Without it, BLEURT reads 0.0 and the run prints one WARNING. The code reads the checkpoint at `/tmp/BLEURT-20`. Many systems clear `/tmp` at restart, so make sure that the folder is there before each evaluation.

```bash
pip install git+https://github.com/google-research/bleurt.git
curl -L -o /tmp/BLEURT-20.zip https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip /tmp/BLEURT-20.zip -d /tmp
```

On a CUDA machine, BLEURT also needs 2 files of the CUDA compiler. Without them it stops with `JIT compilation failed` and `libdevice not found`, and the column reads 0.0. Install the package into the site-packages folder of TensorFlow (`python -m pip show tensorflow` shows it; add `--user` when it is in `~/.local`). Use the version of your torch wheel (12.8 for `+cu128`). A machine with a full CUDA toolkit can set `XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda` instead.

```bash
python -m pip install "nvidia-cuda-nvcc-cu12==12.8.*"
```

**Check before you continue.**

- This command prints a score near 1.0 and no WARNING:

```bash
python -c "import metrics; print(metrics.pair_text_scores(['the cat sat on the mat'], ['the cat sat on the mat'], False, keys=('bleurt',)))"
```

- `python -c "import transformers; print(transformers.__version__)"` prints a 4.x version. Version 5.x loads the Uni-Sign checkpoint wrong with no error.
- `python -c "import sacrebleu, evaluate, rouge, nltk, pycocoevalcap"` prints no error. When a package is missing, its score column reads 0.0 and the run prints a WARNING.

## 2. Rules for every run

Set the corpus once in each shell. Every per-corpus command reads `$LANG`:

```bash
LANG=asf    # ase | asf | bfi | how2sign | auslan_daily | bobsl_auto
```

`LANG` is also the locale variable of the shell, so some tools (for example `perl`) can print locale warnings in this shell.

- `train.py --dev-split test` chooses the epoch on test: its test scores are not held-out scores. The run writes to the normal checkpoint folder, and no file or row name shows the choice, so its files replace those of a dev-chosen run of the same config.
- Read test only after all training and all choices on dev are final. Every command that reads model outputs on test needs `--allow-test`. No training stage reads test.
- Every epoch writes `latest.pt`. To continue a stopped run, run the same command again with `--resume`. `--resume` refuses a run whose recipe changed: start a new run then.
- Without `--resume`, a run refuses to start while `latest.pt` is in its folder. To start again from the beginning, move `latest.pt`, `model.pt` and `best.json` out of the folder.
- Every epoch prints one line. Its `ckpt` column shows `saved` when the run writes a new best `model.pt`. The same lines go to `history.csv`.
- The loader keeps its records in `<data root>/.cache/` and loads again by itself after a change. To load again by force, delete that folder.
- Each output goes to `outputs/<kind>/` (§12.2). A command writes over a file with the same name, so keep a copy of your result files before you run a command again.
- Some runs need a config edit (§10). While an edit is in place, start no other run from the same `configs/` folder. Use a second copy of the repository to run them at the same time.
- A WARNING that a model or a file comes from other train labels does not stop the run. Train again, or make the file again, before you report it.
- Several GPUs: put `torchrun --standalone --nproc-per-node=N` before `train.py` and change nothing else. `batch_size` and `checkpoint.dev_batch_size` are global batches, so each must be at least N. Evaluation runs in one process.

```bash
torchrun --standalone --nproc-per-node=4 train.py --stage train-slt --language "$LANG" --slt-config configs/ar.yaml
```

**Two machines.** Train on a CUDA machine and evaluate on another one, if you want. Copy these files:

| To | Copy |
| --- | --- |
| each training machine | The repository and the 2 released files of §1. Of `data/youtube-sl-25/`: the split CSV, `case_lexicon.en_XX.json`, and in each `<lang>/` folder `poses/`, `subs/` and `video_meta.csv` (not `signverse_shards/`). Of each benchmark folder: `poses/`, `subs/`, the split files, the sidecars (`video_meta*.csv`) and `case_lexicon.en_XX.json` (not `persons/` and not `.cache/`). |
| the machine of B3 | The `model.pt` of SayWhen-Seg (A2) and of Uni-Sign of the corpus (B2a), each in its folder. |
| the evaluation machine | The same data. The `model.pt` of every trained model, each in its folder (see "Names"). |
| a machine that continues a run with `--resume` | The `latest.pt` of that run too. |

Use the same `configs/` (with the B2b edit) and the same prepared data on every machine. After a data step changes files on one machine, copy the changed files to the others (usually only the sidecars `video_meta*.csv` and the casing files).

## 3. Data checks (A0, A0b)

### 3.1 YouTube-SL-25 (A0)

This step downloads the SignVerse-2M pose shards of `ase`, `asf` and `bfi`, converts them to one pose file per video, and writes the captions and one sidecar `video_meta.csv` per language. Its last stage, the copy check (`dedup`), marks each train video that copies a dev or test video; the train loader drops it. It needs no GPU, but much free disk space for the shard cache. The data rules are in `docs/data_pipeline.md`.

```bash
mkdir -p data/youtube-sl-25
curl -L -o data/youtube-sl-25/SignVerse-2M-metadata_split.csv \
    https://huggingface.co/datasets/SignerX/SignVerse-2M/resolve/0c7e956a8471acad3a9cebaeb66bb23bf5deaafa/SignVerse-2M-metadata_split.csv
# Optional: shard and video counts, with a size estimate.
python prepare_yt25.py --stage plan --size
# All stages in order: download, convert, person-counts, subs and the copy check.
python prepare_yt25.py --stage all --languages ase asf bfi --workers 12
# The corpus audit.
python report.py data --language ase --language asf --language bfi
```

Then check the sampler on each language (`LANG=ase`, then `asf`, then `bfi`):

```bash
python train.py --stage smoke-data --language "$LANG" --split train --num-samples 64
```

**Writes.** `data/youtube-sl-25/<lang>/poses/`, `subs/` and `video_meta.csv`; the casing file `data/youtube-sl-25/case_lexicon.en_XX.json` (the dev and test loaders refuse to start without it, so build it on a machine with the train captions of all 3 languages and copy it to the others); `outputs/report/data/data_audit_<lang>.md` and `.json`.

**Check before you continue.**

- Every `person-counts` line shows `empty_runs 0/N`, `handless_runs 0/N`, `switch_runs 0/N` and `default_signing_runs 0/N`. The first number counts the rows that are still unknown.
- Each `dedup | <lang>` line gives the number of train videos that copy a dev or test video.
- In the `convert | reconciliation` line, `shard not downloaded` is 0. If it is not 0, run `--stage all` again.
- `smoke-data` shows `records` and `translation_targets` above 0 and `padding_label` `UNK`. Its `batch_pose_shape` ends in `[69, 3]`, and `modes` holds more than one mode.

**Repair.** `--stage all` already runs every stage. Run one stage alone only to repair the data:

```bash
python prepare_yt25.py --stage person-counts --languages ase asf bfi --workers 12
python prepare_yt25.py --stage subs --languages ase asf bfi
python prepare_yt25.py --stage dedup --languages ase asf bfi
```

- `person-counts` fills the blank values of the sidecar. `subs` adds a caption to a converted video that has none. Neither changes a pose file or an existing caption.
- `convert` and `subs` blank the `copy_of` column, and the train loader refuses a blank value. So run `dedup` again after them.
- If the loader refuses a sidecar and names `--stage all --overwrite`, convert the pose files again: `python prepare_yt25.py --stage all --overwrite --languages ase asf bfi --workers 12` (it needs the shard tars).
- `--workers N` converts N shards at the same time. Keep N below the number of CPU cores and below the free memory in GB, with about 9 GB more free for the longest videos. A stopped run loses at most the shards in progress: run the same command again without `--overwrite`.
- `--delete-tars` frees the shard cache in `data/youtube-sl-25/signverse_shards/`. A later conversion needs the tars again.

### 3.2 The benchmarks (A0b)

The poses, captions, split files and sidecars of the 3 benchmarks come ready from the extractor slp-pose, in `data/how2sign`, `data/auslan-daily-abc` and `data/bobsl`. This step runs only the copy check on them. Its first train load also builds the casing file of each benchmark.

```bash
python prepare_yt25.py --stage dedup --languages how2sign auslan_daily bobsl_auto bobsl_manual
python report.py data --language how2sign --language auslan_daily --language bobsl_auto
```

Then run `smoke-data` (§3.1) with `LANG=how2sign`, `auslan_daily` and `bobsl_auto`.

**Writes.** The `copy_of` column of each benchmark sidecar, and `case_lexicon.en_XX.json` in each benchmark folder.

**Check before you continue.**

- Each `dedup | <corpus>` line shows 0 train videos that copy a dev or test video.
- Each load of `bobsl_auto` prints how many episodes it uses: the automatic captions of the official train split, and 4 dev (Sent-Val) and 35 test (Sent-Test) hand-aligned episodes.
- `smoke-data` passes the checks of §3.1 on each benchmark key.

## 4. Segmenters (A1, A2)

This step trains the 2 segmenters, each once, on the pool. Both choose their best epoch on the dev sets of `ase`, `asf` and `bfi`. The Moryossef segmenter (A1) gives the spans of the "Moryossef + Uni-Sign" rows. It is a baseline, so it keeps the method rules of its release, except these: its train chunks and their labels are SayWhen-Seg's, not the release's, and its Dice term sits on its sentence head, because our corpora have no sign head (the head of `configs/moryossef26.yaml` lists what differs, and why). SayWhen-Seg (A2) gives the spans of the "SayWhen-Seg + Uni-Sign" rows, and it is the start of the segmentation stream of SayWhen. Run them on a CUDA GPU, at the same time as B2a (§5.1) if you have 2 GPUs.

Run A1 once. It takes no `--language`: a pool has no target corpus. A run refuses to start while a `latest.pt` of an older run is in its folder: move that folder first.

```bash
python train.py --stage train-moryossef
```

Run A2 once, in the same way.

```bash
python train.py --stage train-bio
```

**Writes.** `checkpoints/moryossef/multi_ase-asf-bfi-bobsl_auto/` and `checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto/`, each with `model.pt`, `best.json`, `latest.pt` and `history.csv`.

**Check before you continue.**

- Each run prints its best-epoch floor. No epoch before the floor shows `ckpt saved`.
- Train both segmenters from one code state. This command prints the pool mix of each segmenter and the decode that chose its epoch (it must be `argmax`). For each corpus of the pool, the 2 label fingerprints must be the same:

```bash
python - <<'EOF'
from models.checkpointing import load_checkpoint_meta
s1 = load_checkpoint_meta("checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto/model.pt")
mory = load_checkpoint_meta("checkpoints/moryossef/multi_ase-asf-bfi-bobsl_auto/model.pt")
print("bio_s1", s1.get("pretrain_mix"), s1.get("monitor_decode"))
print("moryossef", mory.get("pretrain_mix"), mory.get("monitor_decode"))
for corpus in ("ase", "asf", "bfi", "bobsl_auto"):
    print(corpus, (mory.get("language_fingerprints") or {}).get(corpus), (s1.get("language_fingerprints") or {}).get(corpus))
EOF
```

- If a segmenter runs out of GPU memory, lower `batch_size` in its config (`configs/bio_pretrain.yaml` or `configs/moryossef26.yaml`) and raise `grad_accum_steps` by the same factor.

## 5. Models of each corpus (B2a, B2b, B3)

### 5.1 Uni-Sign: the pre-trimmed translator (B2a)

This step fine-tunes Uni-Sign's OpenASL release on the gold sentences of one corpus, each clip cut at the true edges of its sentence. Every corpus starts from the same OpenASL release. Each run chooses its epoch on the dev BLEU-4 of its corpus. `configs/baseline_eval.yaml` must hold `decoder: ar`.

Run it on a CUDA GPU once for each of `ase`, `asf`, `bfi`, `how2sign`, `auslan_daily` and `bobsl_auto`:

```bash
python train.py --stage train-slt --language "$LANG" --slt-config configs/baseline_train.yaml
```

- BOBSL follows the LiTFiC protocol in every stage: `--language bobsl_auto` trains on the automatic captions, and its dev and test are the hand-aligned Sent-Val and Sent-Test.
- `bobsl_auto` is the largest train set, so its run is the longest.

**Writes.** `checkpoints/baseline_train_ar/$LANG/` (`model.pt`, `best.json`, `latest.pt`, `history.csv`).

**Check before you continue.**

- `val_translation_bleu4` in `best.json` is the dev BLEU-4 of the selected epoch. It must be well above the dev BLEU-4 of the OpenASL release with no fine-tuning. This command measures the release on the same dev clips; read `severity[0].text_metrics.translation_bleu4` in its output file.

```bash
python analyze.py --stage boundary-sweep --text pretrimmed --pretrimmed-checkpoint checkpoints/openasl_pose_only_slt.pth --language "$LANG" --split dev \
    --severity-grid-head=0 --severity-grid-tail=0 --output outputs/analysis/boundary_sweep_released_${LANG}_dev.json
```

- `val_translation_len_ratio` in `best.json` is near 1. A value far from 1 shows outputs that are too short or too long.

### 5.2 Point the configs at Uni-Sign (B2b)

SayWhen and every `--text pretrimmed` row read Uni-Sign from the `pretrained_slt` entry of the corpus in `configs/data.yaml`. Until you edit it, that entry names the OpenASL release. A row then reads the release with no error, and its files still carry the name `pretrimmed`. So do this edit on every machine that trains SayWhen or evaluates.

In `configs/data.yaml`, find the entry of each target corpus. Replace its active `ar` line with the comment line under it. For example, for `asf`:

```yaml
languages:
    asf:
        pretrained_slt:
            ar: checkpoints/baseline_train_ar/asf/model.pt
            dlm: checkpoints/openasl_pose_only_slt.pth
```

Do this for `ase`, `asf`, `bfi`, `how2sign`, `auslan_daily` and `bobsl_auto`. Leave the `dlm` lines as they are (§10.1).

**Check before you continue.** This command prints 6 trained files and no OpenASL file:

```bash
python -c "
from utils import load_yaml
cfg = load_yaml('configs/data.yaml')
for c in ('ase', 'asf', 'bfi', 'how2sign', 'auslan_daily', 'bobsl_auto'): print(c, cfg['languages'][c]['pretrained_slt']['ar'])"
```

### 5.3 SayWhen (B3)

This step trains segmentation and translation together (stage 2). The segmentation stream starts from SayWhen-Seg (A2), and the translation stream starts from Uni-Sign of the corpus (B2b). Each run chooses its epoch on its dev set. It needs a CUDA GPU with much memory.

Run it once for each of `ase`, `asf`, `bfi`, `how2sign`, `auslan_daily` and `bobsl_auto`:

```bash
python train.py --stage train-slt --language "$LANG" --slt-config configs/ar.yaml
```

**Writes.** `checkpoints/ar/$LANG/` (`model.pt`, `best.json`, `latest.pt`, `history.csv`).

**Check before you continue.**

- Before the first full run, watch the GPU memory with `nvidia-smi` during the first steps. Near the limit, lower `batch_size` and raise `grad_accum_steps` by the same factor. If a dev evaluation runs out of memory, lower `checkpoint.dev_batch_size`; no score depends on it.
- The dev losses `val_bio_loss` and `val_translation_loss` fall over the first epochs.
- The run prints the counts of the input types that it draws when it builds the data (each epoch draws again).
- `text_grad_rate` is above 0: the text trains the end of the predicted span.
- `text_accepted_rate` (the share of rows that read the model's own span) does not fall on rows with one sentence. A correct split of a joined target also lowers it, so read it beside `text_pair_rate` (the share of joined targets).
- Each dev evaluation logs the monitor `val_offline_score` = `val_offline_f1` × `val_offline_bleu4`. Both factors rise over the run.
- `best.json` gives the selected epoch.

## 6. Choose the decode of each system on dev

SayWhen-Seg and the head of SayWhen turn their frame scores into sentence spans with one of 3 decodes: `argmax` (the best label of each frame), `viterbi` (the best legal B-I-O path) or `duration` (the best legal path with a sentence-length prior, the semi-Markov decode). Choose one decode for each corpus, system and access on dev, with one fixed rule, before you read test. The Moryossef segmenter is a baseline, so it stays on `argmax`, the decode of its release.

**The rule.** Take the decode with the highest dev main score (`main.edge_chunk.no_edge_limit.avg.f1`, rounded to 4 decimals). On a tie, keep `argmax`, the decode under which each model chose its epoch. The main score reads only the spans, so the no-text row gives the same score as the row with text, at a much lower cost.

**Step 1: the 12 no-text dev rows of one corpus.** Run them for each of `ase`, `asf`, `bfi`, `how2sign`, `auslan_daily` and `bobsl_auto`. They need §4 and §5.3 (the arm), but no translator.

```bash
for D in argmax viterbi duration; do
    python eval.py --offline --spans s1 --text none --decode "$D" --language "$LANG" --split dev
    python eval.py --online --spans s1 --text none --decode "$D" --language "$LANG" --split dev
    python eval.py --offline --spans ar --text none --decode "$D" --language "$LANG" --split dev
    python eval.py --online --spans ar --text none --decode "$D" --language "$LANG" --split dev
done
```

**Step 2: apply the rule.** This command reads the 12 scores files of the corpus in its first argument. It prints the 3 dev scores and the choice of each system and access, and it sets 6 shell variables: `D_MORY_OFF` and `D_MORY_ON`, always `argmax`, and `D_S1_OFF`, `D_S1_ON`, `D_AR_OFF` and `D_AR_ON`. The test commands of §7 and §8 read them.

```bash
eval "$(python - "$LANG" <<'EOF'
import json, sys
lang, out = sys.argv[1], []
print("unset D_MORY_OFF D_MORY_ON D_S1_OFF D_S1_ON D_AR_OFF D_AR_ON", flush=True)
for when, tag in (("offline", "OFF"), ("online", "ON")):
    out.append(f"export D_MORY_{tag}=argmax")
    for spans, name in (("s1", "S1"), ("joint-ar", "AR")):
        f1 = {d: json.load(open(f"outputs/{when}/{when}_{spans}_{d}_none_{lang}_dev_scores.json"))
              ["main"]["edge_chunk"]["no_edge_limit"]["avg"]["f1"] for d in ("argmax", "viterbi", "duration")}
        best = max(f1, key=lambda d: (round(f1[d], 4), d == "argmax"))
        print(f"{lang} dev, {when} {spans}: " + ", ".join(f"{d} {v:.4f}" for d, v in f1.items()) + f" -> {best}", file=sys.stderr)
        out.append(f"export D_{name}_{tag}={best}")
print("\n".join(out))
EOF
)"
```

**Step 3: write the choice down.** Save the 4 printed lines of each corpus in your notes. You can run step 2 again at any time: it only reads the dev files. Before the test rows of a corpus, run step 2 for that corpus. The transfer rows (§8.2) use the choice of their source corpus (`ase` for `how2sign`, `asf` for `auslan_daily`, `bfi` for `bobsl_auto`), because they read no label of the benchmark.

**Writes.** `outputs/offline/offline_<spans>_<decode>_none_<lang>_dev.json` and `outputs/online/online_<spans>_<decode>_none_<lang>_dev.json`, each with its scores file. `<spans>` is `s1` or `joint-ar`.

**Check before you continue.** Step 2 prints 4 lines with 3 scores each, and `echo $D_AR_ON` prints a decode. When a file is missing, step 2 stops with an error and sets no variable.

## 7. Table 1: YouTube-SL-25 test

Do §7.1 to §7.4 for `LANG=ase`, then `asf`, then `bfi`. First run step 2 of §6 for the corpus: it sets the 6 decode variables that the commands read.

### 7.1 The system rows

One command per row of the table. Each online row also records the display trace that gives NE (`--stability`); it does not change the captions. Its decodes run in batches of `--batch-size` rows, in each process, so lower `--batch-size` if a process runs out of memory; `--batch-size 1` runs each decode at its stride. Each online row also counts its compute (`--count-flops`), which gives GFLOP per second of video in the paper: it reads the shapes of the row's own model calls, so it adds little time (a row with a DLM translator runs its model passes once more, untimed), and its captions, scores and timings stay those of the plain pass. It counts only the work that the committed captions need: not the `--stability` trace, not the confidence pass after each AR decode, and not the padding of the encoder cache. The timings include the last 2. Keep `--count-flops` in these commands, and leave it out of dev rows, which need no compute numbers. Add `--workers 2` to an online row to run its streams in 2 processes (each loads its own models): the captions, scores and files stay the same. Time a row with `--workers 1` when you report its real-time numbers.

```bash
# Moryossef + Uni-Sign, Offline and Online.
python eval.py --offline --spans moryossef --text pretrimmed --decode "$D_MORY_OFF" --language "$LANG" --split test --allow-test
python eval.py --online --spans moryossef --text pretrimmed --decode "$D_MORY_ON" --stability --stability-hours 6 --count-flops --language "$LANG" --split test --allow-test
# SayWhen-Seg + Uni-Sign, Offline and Online.
python eval.py --offline --spans s1 --text pretrimmed --decode "$D_S1_OFF" --language "$LANG" --split test --allow-test
python eval.py --online --spans s1 --text pretrimmed --decode "$D_S1_ON" --stability --stability-hours 6 --count-flops --language "$LANG" --split test --allow-test
# SayWhen, Offline and Online.
python eval.py --offline --spans ar --text ar --decode "$D_AR_OFF" --language "$LANG" --split test --allow-test
python eval.py --online --spans ar --text ar --decode "$D_AR_ON" --stability --stability-hours 6 --count-flops --language "$LANG" --split test --allow-test
```

The online rows take much longer than the offline rows.

**Writes.** For each row, its events file and scores file in `outputs/offline/` or `outputs/online/`, named `{access}_{spans}_{decode}_{text}_{corpus}_test` (§12.3). Each online row also writes `outputs/stability/stability_<system>_<corpus>_test.json` and `provisional_<system>_<corpus>_test.json`. `<system>` is `moryossef`, `s1` or `ar`, with `-<decode>` added when the decode is not the one under which the model chose its epoch (`argmax`), for example `stability_s1-viterbi_asf_test.json`. `--stability-hours 6` adds `-6h` after that, for example `stability_s1-6h_asf_test.json`.

**Check before you continue.**

- Each row prints a title line with its main score and one table. The F1 line `no limit`, column `avg`, is the main score.
- The translator of the cascade rows is Uni-Sign of the corpus, not the OpenASL release. This command prints `checkpoints/baseline_train_ar/<corpus>/model.pt`:

```bash
python -c "import json, sys; print(json.load(open(sys.argv[1]))['provenance']['checkpoint'])" \
    outputs/offline/offline_moryossef_${D_MORY_OFF}_pretrimmed_${LANG}_test_scores.json
```

- Each online row with `--stability` prints a table of display policies with the column `NE`. The line `frozen_agreement_n2` gives the NE of the paper.
- `--stability-hours 6` reads NE on a fixed sample of streams with at least 6 hours of pose, the same streams for every system of a corpus (the whole split when the split is shorter). The captions, the GFLOP count and all other scores of the row still read every stream. The row prints the streams and hours of the sample, and `provenance.compute` of the scores file holds them (`trace_streams`, `trace_pose_s`): give them per corpus beside NE in the paper. Compare this NE only between systems of one corpus: a few long streams weigh much in it, so it is not the NE of the whole split. Leave the flag out to read every stream. Use one value in all rows of a corpus: another value reads other streams and gives other file names, and a lower value lets a few long streams decide NE.
- Each online row prints a `[row] compute` line under its table with its GFLOP per second of video. If the line says `not counted`, the command had no `--count-flops`: run the row again with it.

### 7.2 The gray row: Uni-Sign on gold sentences

This command translates each gold test sentence, cut at its true edges, with Uni-Sign of the corpus, and scores all of them as one corpus.

```bash
python analyze.py --stage boundary-sweep --text pretrimmed --language "$LANG" --split test --allow-test \
    --severity-grid-head=0 --severity-grid-tail=0 --output outputs/analysis/boundary_sweep_gold_${LANG}_test.json
```

**Writes.** `outputs/analysis/boundary_sweep_gold_<corpus>_test.json`. Its one cell `severity[0]` holds `text_metrics` with `translation_bleu4`, `translation_bleurt` and `translation_rougeL`.

**Check before you continue.** In the output file, `provenance.checkpoint` names Uni-Sign of the corpus (as in §7.1), `severity[0].grid_head` and `grid_tail` are 0, and `severity[0].dropped_fraction` is 0.

### 7.3 The window rows (StreamSLST and Multi-Stage)

These systems train and decode outside this repository. To get F1, B-4, B-RT, R-L and Lat. under the same protocol, score their test captions here. Count their GFLOPs per second of video in the other code, with the same counter (`evaluation.systems.count_flops`: `torch.utils.flop_counter`, 2 FLOPs per multiply-accumulate) and the same rule. Count every model call that the captions of the events file need: each call at its real input frames (no padding of a batch or a kernel), and each decode at the number of tokens it produced (not a fixed maximum). Do not count a pass whose output no caption reads (here: the padding of the encoder cache, the confidence pass after each AR decode, and the `--stability` trace decodes). `--count-flops` itself reads only rows of the streaming engine.

1. In the other code, write the test captions of each system and corpus as one events file. Use the video ids of the corpus split file, and times in seconds of the source video. `commit_time_s` is the video time at which the system shows the caption; Lat. reads it.

```json
{"events": {"<video_id>": [{"start_s": 12.0, "end_s": 15.5, "text": "the caption", "commit_time_s": 18.0}]}}
```

2. Keep these files outside `outputs/`, for example in a folder `$EXT`. Check each file: this command prints the main score table of the file and writes nothing.

```bash
python eval.py --predictions "$EXT/multistage_${LANG}_test.json" --language "$LANG" --split test --allow-test
```

3. Add each file to the command of §7.4 with one more `--events`, for example `--events "Multi-Stage Window"="$EXT/multistage_${LANG}_test.json"`.

**Check before you continue.** The command of step 2 prints no WARNING that video ids are absent from the split. One WARNING that the file has no annotation stamp is expected: the file comes from outside this repository.

### 7.4 Collect the numbers of one corpus

`report.py results` scores all rows of one corpus again under one protocol. It writes one table with the main scores of every row, the compute table (GFLOPs) and the latency table (Lat.).

```bash
python report.py results --language "$LANG" --split test \
    --events "Moryossef + Uni-Sign Offline"=outputs/offline/offline_moryossef_${D_MORY_OFF}_pretrimmed_${LANG}_test.json \
    --events "Moryossef + Uni-Sign Online"=outputs/online/online_moryossef_${D_MORY_ON}_pretrimmed_${LANG}_test.json \
    --events "SayWhen-Seg + Uni-Sign Offline"=outputs/offline/offline_s1_${D_S1_OFF}_pretrimmed_${LANG}_test.json \
    --events "SayWhen-Seg + Uni-Sign Online"=outputs/online/online_s1_${D_S1_ON}_pretrimmed_${LANG}_test.json \
    --events "SayWhen Offline"=outputs/offline/offline_joint-ar_${D_AR_OFF}_ar_${LANG}_test.json \
    --events "SayWhen Online"=outputs/online/online_joint-ar_${D_AR_ON}_ar_${LANG}_test.json
```

**Writes.** `outputs/report/results_<corpus>_test/`, with `results.md`, `fig/` and `tables/`. The tables of the paper read `tables/edge_chunk.csv` (F1, B-4, B-RT, R-L) and `tables/compute.csv` (GFLOPs). §9 gives the column of each cell. `tables/compute.csv` also gives the parameters, the device and the real-time factor of each online row. `tables/held_unit_latency.csv` gives Lat., which is no column of the tables.

**Check before you continue.** In `tables/edge_chunk.csv`, the lines with `tau_s` = `no limit` and `tiou` = `avg` give the same `f1` as the title line of each row in §7.1. `tables/held_unit_latency.csv` has one line for each online row and for each window row with commit times, and none for the offline rows.

## 8. Table 2: benchmark test

### 8.1 Trained on the benchmark

Do §7.1, §7.2, §7.3 and §7.4 again for `LANG=how2sign`, then `auslan_daily`, then `bobsl_auto`. First run step 2 of §6 for the corpus. SayWhen-Seg and the Moryossef segmenter are the pooled ones of §4. Uni-Sign and SayWhen are those of the benchmark (§5).

- Run §7.2 for How2Sign too: its gray cells hold our scores, not the published ones, because the public Uni-Sign code selects its How2Sign checkpoint on the test set.
- `auslan_daily` scores every text in lowercase, because its captions are all lowercase. The commands do not change.

### 8.2 Transfer from YouTube-SL-25

These rows run the models of a YouTube-SL-25 language on the benchmark of the same sign language. They train nothing, and they read no label of the benchmark: the models and the duration prior come from the source corpus (`--source-language`), and the decodes come from the dev choice of the source corpus (§6). The data, the gold sentences and the scores come from `--language`. The file names carry `<source>-to-<target>`.

For BOBSL, the pool of the 2 segmenters holds the automatic alignment of the BOBSL train episodes, but no model read a manual label.

Set the pair. Then run step 2 of §6 with `"$SRC"` in place of `"$LANG"`, so the decode variables hold the choice of the source corpus.

```bash
SRC=ase; LANG=how2sign    # then SRC=asf LANG=auslan_daily, then SRC=bfi LANG=bobsl_auto
```

```bash
# Moryossef + Uni-Sign, Offline and Online.
python eval.py --offline --spans moryossef --text pretrimmed --decode "$D_MORY_OFF" --language "$LANG" --source-language "$SRC" --split test --allow-test
python eval.py --online --spans moryossef --text pretrimmed --decode "$D_MORY_ON" --stability --stability-hours 6 --count-flops --language "$LANG" --source-language "$SRC" --split test --allow-test
# SayWhen-Seg + Uni-Sign, Offline and Online.
python eval.py --offline --spans s1 --text pretrimmed --decode "$D_S1_OFF" --language "$LANG" --source-language "$SRC" --split test --allow-test
python eval.py --online --spans s1 --text pretrimmed --decode "$D_S1_ON" --stability --stability-hours 6 --count-flops --language "$LANG" --source-language "$SRC" --split test --allow-test
# SayWhen, Offline and Online.
python eval.py --offline --spans ar --text ar --decode "$D_AR_OFF" --language "$LANG" --source-language "$SRC" --split test --allow-test
python eval.py --online --spans ar --text ar --decode "$D_AR_ON" --stability --stability-hours 6 --count-flops --language "$LANG" --source-language "$SRC" --split test --allow-test
```

Collect them as in §7.4. The report goes to `outputs/report/results_<source>-to-<target>_test/`.

```bash
C="${SRC}-to-${LANG}"
python report.py results --language "$LANG" --split test \
    --events "Moryossef + Uni-Sign Offline"=outputs/offline/offline_moryossef_${D_MORY_OFF}_pretrimmed_${C}_test.json \
    --events "Moryossef + Uni-Sign Online"=outputs/online/online_moryossef_${D_MORY_ON}_pretrimmed_${C}_test.json \
    --events "SayWhen-Seg + Uni-Sign Offline"=outputs/offline/offline_s1_${D_S1_OFF}_pretrimmed_${C}_test.json \
    --events "SayWhen-Seg + Uni-Sign Online"=outputs/online/online_s1_${D_S1_ON}_pretrimmed_${C}_test.json \
    --events "SayWhen Offline"=outputs/offline/offline_joint-ar_${D_AR_OFF}_ar_${C}_test.json \
    --events "SayWhen Online"=outputs/online/online_joint-ar_${D_AR_ON}_ar_${C}_test.json
```

**Writes.** The row files of §7.1 with `<source>-to-<target>` as the corpus field, for example `outputs/online/online_joint-ar_argmax_ar_asf-to-auslan_daily_test.json`, and `outputs/report/results_<source>-to-<target>_test/`.

**Check before you continue.** The provenance of each row holds `"source_language": "<source>"`. The check command of §7.1, with `${C}` in place of `${LANG}`, prints the Uni-Sign file of the source corpus.

## 9. Where each number is

`<corpus>` is the corpus key (Table 1, and the block "Trained on the benchmark"), or `<source>-to-<target>` (the block "Transfer from YouTube-SL-25"). `<D>` is the decode that §6 chose for that system and access. B-4, B-RT and R-L of the system rows are the DVC (dense video captioning) text scores of the same pairs as F1. The paper shows F1, B-RT and R-L × 100; B-4 is already on a 0 to 100 scale.

**The cells and their keys.**

| Column | Rows | File | Key or column |
| --- | --- | --- | --- |
| F1 | all except the gray row | scores file of the row | `main.edge_chunk.no_edge_limit.avg.f1` |
| B-4 | all except the gray row | scores file of the row | `main.edge_chunk.no_edge_limit.avg.dvc.bleu4` |
| B-RT | all except the gray row | scores file of the row | `main.edge_chunk.no_edge_limit.avg.dvc.bleurt` |
| R-L | all except the gray row | scores file of the row | `main.edge_chunk.no_edge_limit.avg.dvc.rougeL` |
| F1, B-4, B-RT, R-L | the same rows, also window rows | `outputs/report/results_<corpus>_test/tables/edge_chunk.csv` | the line with `system` = the label, `tau_s` = `no limit`, `tiou` = `avg`; columns `f1`, `dvc_bleu4`, `dvc_bleurt`, `dvc_rougeL` |
| GFLOPs | Online rows | `outputs/report/results_<corpus>_test/tables/compute.csv` | column `gflop_per_video_s` of the line with `system` = the label, or `provenance.compute.gflop_per_video_s` in the scores file of the row |
| Lat. (no column of the tables) | Online rows and window rows | `outputs/report/results_<corpus>_test/tables/held_unit_latency.csv` | column `median_s` of the line with `system` = the label (seconds) |
| NE | Online rows | `outputs/stability/stability_<system>-6h_<corpus>_test.json` (`<system>-<D>-6h` when `<D>` is not `argmax`, §7.1) | `frozen_agreement_n2.normalized_erasure` |
| B-4, B-RT, R-L | gray row | `outputs/analysis/boundary_sweep_gold_<corpus>_test.json` | `severity[0].text_metrics.translation_bleu4`, `translation_bleurt`, `translation_rougeL` |

The offline rows have no GFLOPs and no NE (`--` in the tables): `--count-flops` reads only online rows, and offline captions have no commit time. The window rows have no NE, and their GFLOPs come from the other code (§7.3). The gray row has no F1, GFLOPs or NE.

**The files of each row.**

| Row | Access | Scores file | Label in §7.4 | `<system>` of the stability file |
| --- | --- | --- | --- | --- |
| StreamSLST (mBART), StreamSLST (LSTM), Multi-Stage | Window | none: the events file of §7.3 | your label, for example `Multi-Stage Window` | none |
| Moryossef + Uni-Sign | Offline | `outputs/offline/offline_moryossef_<D>_pretrimmed_<corpus>_test_scores.json` | `Moryossef + Uni-Sign Offline` | none |
| Moryossef + Uni-Sign | Online | `outputs/online/online_moryossef_<D>_pretrimmed_<corpus>_test_scores.json` | `Moryossef + Uni-Sign Online` | `moryossef` |
| SayWhen-Seg + Uni-Sign | Offline | `outputs/offline/offline_s1_<D>_pretrimmed_<corpus>_test_scores.json` | `SayWhen-Seg + Uni-Sign Offline` | none |
| SayWhen-Seg + Uni-Sign | Online | `outputs/online/online_s1_<D>_pretrimmed_<corpus>_test_scores.json` | `SayWhen-Seg + Uni-Sign Online` | `s1` |
| SayWhen | Offline | `outputs/offline/offline_joint-ar_<D>_ar_<corpus>_test_scores.json` | `SayWhen Offline` | none |
| SayWhen | Online | `outputs/online/online_joint-ar_<D>_ar_<corpus>_test_scores.json` | `SayWhen Online` | `ar` |
| Uni-Sign (gray) | Gold | `outputs/analysis/boundary_sweep_gold_<corpus>_test.json` | none | none |

For the window rows, F1 and the text columns come only from `edge_chunk.csv` (the window rows have no scores file).

## 10. Other runs

These runs are not in the two main tables. Each one keeps the rules of §2.

### 10.1 The DLM decoder

The DLM decoder writes a block of tokens at a time. Its runs repeat the AR runs with these changes:

1. **Pre-trimmed DLM translator (B2a).** Set `decoder: dlm` in `configs/baseline_eval.yaml`, run the B2a command of §5.1, then set `decoder: ar` again. It writes `checkpoints/baseline_train_dlm/$LANG/`. While `decoder: dlm` is set, `--text pretrimmed` with no `--pretrimmed-checkpoint` reads the DLM translator. Its `val_translation_loss` is the OPUT loss: it is noisy and does not compare with the cross-entropy of Uni-Sign. Only `val_translation_bleu4` selects the epoch.
2. **B2b.** Point the `dlm` line of `pretrained_slt` at `checkpoints/baseline_train_dlm/<corpus>/model.pt`.
3. **DLM arm (B3).** `python train.py --stage train-slt --language "$LANG" --slt-config configs/dlm.yaml`. It writes `checkpoints/dlm/$LANG/`.
4. **Rows.** Use `--spans dlm` and `--text dlm` for the arm. Compare the DLM arm only with rows of the pre-trimmed DLM translator: give them `--pretrimmed-checkpoint checkpoints/baseline_train_dlm/${LANG}/model.pt`. Their files carry `pretrimmed-dlm` in place of `pretrimmed`, and their stability files `<segmenter>-dlm`.

The DLM decoder has no beam search: `--num-beams` above 1 is refused for it.

### 10.2 Other arms: the no-text arms and the one-stream arms

Each of these arms is the run of §5.3 with a config edit. In a **no-text arm**, the text loss does not reach the segmentation part; it is the control of "text improves the segmentation". The **one-stream arm** has 1 pose encoder, the translator's, and the segmentation head reads it too through a small stitch network that the run fits at its start.

| Arm | `shared_pose_encoder` | `bio.adapter` | `membership_gate.detach_omega` | `checkpoint.dir` | `wandb.name` |
| --- | --- | --- | --- | --- | --- |
| two-stream no-text arm | `false` (no change) | `false` | `true` | `checkpoints/ar_notext/${language}` | `ar_notext_${language}` |
| one-stream arm | `true` | `false` | `false` (no change) | `checkpoints/ar_one_stream/${language}` | `ar_one_stream_${language}` |
| one-stream no-text arm | `true` | `false` | `true` | `checkpoints/ar_one_stream_notext/${language}` | `ar_one_stream_notext_${language}` |

1. Set the keys of one line in `configs/ar.yaml` (for a DLM arm: `configs/dlm.yaml`, with `dlm` in place of `ar` in the folder and the name). For example, the one-stream no-text AR arm:

```yaml
shared_pose_encoder: true
bio:
    adapter: false
membership_gate:
    detach_omega: true
checkpoint:
    dir: checkpoints/ar_one_stream_notext/${language}
wandb:
    name: ar_one_stream_notext_${language}
```

2. Train with the command of §5.3. Keep the edit until the run is complete: `--resume` refuses a changed config.
3. Undo the edit before you start another run or any evaluation. While the edit is in place, `--spans ar` or `--text ar` with no `--arm-checkpoint` reads this arm, and its files take the names of SayWhen.
4. Evaluate it with the commands of §7 or §10.3, each with `--arm-checkpoint <its model.pt>`. Its files carry the name tokens of its edit:

| AR arm | `--arm-checkpoint` | Spans field | Text field | Text field of its no-text row |
| --- | --- | --- | --- | --- |
| SayWhen (two-stream arm) | none | `joint-ar` | `ar` | `none` |
| two-stream no-text arm | `checkpoints/ar_notext/$LANG/model.pt` | `joint-ar-detach-noadapter` | `ar-detach-noadapter` | `none-detach-noadapter` |
| one-stream arm | `checkpoints/ar_one_stream/$LANG/model.pt` | `joint-ar-onestream` | `ar-onestream` | `none-onestream` |
| one-stream no-text arm | `checkpoints/ar_one_stream_notext/$LANG/model.pt` | `joint-ar-onestream-detach` | `ar-onestream-detach` | `none-onestream-detach` |

**Check.** A one-stream arm prints `slt | one stream: ...` before its first epoch. The line gives the train rows of the stitch fit and its held-out dev windows (whole dev videos that the fit never reads), the frames that it read (1 of every 13 real frames of a row), the R^2 on the dev windows and its epoch, and the minutes of the extraction, the gather and the fit. A `--resume` run prints that the stitch comes from `latest.pt` and fits nothing. Before it extracts features, the run prints the memory bound of the kept frames and the memory of the machine, and it stops when the bound does not fit. Under torchrun on 1 machine, the share of each rank stays in CPU memory until the gather ends, and rank 0 fits alone while the other ranks wait. `text_grad_rate` is 0 only in a no-text arm.

### 10.3 More rows

These are the numbered rows of the full result table that the design spec reads (`streaming_slt_prompt.md` §9.2, "How to read the table"). The main tables hold rows 3, 5 and 6 (Offline) and 8, 9 and 10 (Online); the gray row is the translator of row 1, scored as one corpus (§7.2). Every row below also takes `--language "$LANG" --split test --allow-test`, and with `--source-language` it is a transfer row (§8.2). A row with no `--decode` uses the decode under which its model chose its epoch (`argmax`).

| # | Spans from | Translator | Command |
| --- | --- | --- | --- |
| 1 | the gold sentences | Uni-Sign | `eval.py --offline --spans gold --text pretrimmed` |
| 2 | the gold sentences | SayWhen's translator | `eval.py --offline --spans gold --text ar` |
| 3 | Moryossef | Uni-Sign | `eval.py --offline --spans moryossef --text pretrimmed` |
| 4 | Moryossef | SayWhen's translator | `eval.py --offline --spans moryossef --text ar` |
| 5 | SayWhen-Seg | Uni-Sign | `eval.py --offline --spans s1 --text pretrimmed` |
| 6 | SayWhen's head, on the whole video | SayWhen | `eval.py --offline --spans ar --text ar` |
| 7 | the spans of row 6 (the same-span control) | Uni-Sign | `eval.py --offline --spans ar --text pretrimmed` |
| 8 | Moryossef under the streaming engine | Uni-Sign | `eval.py --online --spans moryossef --text pretrimmed` |
| 9 | SayWhen-Seg under the streaming engine | Uni-Sign | `eval.py --online --spans s1 --text pretrimmed` |
| 10 | SayWhen's head under the streaming engine | SayWhen | `eval.py --online --spans ar --text ar --stability --stability-hours 6` |
| 11 | the committed spans of row 10 (the same-span control) | Uni-Sign | `eval.py --online --spans ar --text pretrimmed` |
| 10n | the head of the no-text arm under the streaming engine | the no-text arm | `eval.py --online --spans ar --text ar --arm-checkpoint checkpoints/ar_notext/$LANG/model.pt` |
| 11n | the committed spans of row 10n (the fair cascade) | Uni-Sign | `eval.py --online --spans ar --text pretrimmed --arm-checkpoint checkpoints/ar_notext/$LANG/model.pt` |

The arm rows (2, 4, 6, 7, 10 and 11) run for each arm of §10.2 with its `--arm-checkpoint`. The same-span controls read the spans that the arm commits, so they change only the translator: an online engine with no text commits the same spans as the engine with text. A cascade on the spans of an arm keeps the spans field of the arm and takes the text field of its translator, for example `offline_joint-ar_argmax_pretrimmed_<corpus>_test.json`.

| Other row | Command |
| --- | --- |
| A segmenter or an arm alone (its spans, no text), offline / online | `eval.py --offline --spans s1 --text none` / `eval.py --online --spans s1 --text none` (or `moryossef`, `ar`) |
| Spans of the median train sentence length, made by no model (chance) | `eval.py --offline --spans chance --text none` |
| The spans of a saved row, read by a translator | `eval.py --spans <events file> --text pretrimmed` |
| The scores of a saved row, again (no model run, no new file) | `eval.py --predictions <events file>` |

The chance row writes `offline_chance_none_none_<corpus>_test.json`. It shows how much of the main score a corpus gives with no model, so compare systems only inside one corpus.

### 10.4 Segmenter scores and error types

Each no-text row of a segmenter saves, beside its main scores, the frame scores of the protocol of Moryossef et al. (key `frame`). The error analysis reads the spans of one no-text row and writes the share of each error type.

```bash
# The 6 offline no-text rows of the 2 segmenters (the 3 decodes each).
for D in argmax viterbi duration; do
    python eval.py --offline --spans moryossef --text none --decode "$D" --language "$LANG" --split test --allow-test
    python eval.py --offline --spans s1 --text none --decode "$D" --language "$LANG" --split test --allow-test
done
# The error analysis of one no-text row (give the events file of the decode you want).
python analyze.py --stage segmenter-errors --language "$LANG" --split test --allow-test \
    --predictions outputs/offline/offline_moryossef_argmax_none_${LANG}_test.json
```

Use `--online` for the online twins. The motivation statistics run the same commands on `--split dev` with no `--allow-test`, beside the chance row. Run them for each corpus as soon as A1 exists, and report each corpus apart:

```bash
python eval.py --offline --spans moryossef --text none --language "$LANG" --split dev
python analyze.py --stage segmenter-errors --language "$LANG" --split dev \
    --predictions outputs/offline/offline_moryossef_argmax_none_${LANG}_dev.json
python eval.py --offline --spans chance --text none --language "$LANG" --split dev
python report.py results --language "$LANG" --split dev \
    --events moryossef=outputs/offline/offline_moryossef_argmax_none_${LANG}_dev.json \
    --events chance=outputs/offline/offline_chance_none_none_${LANG}_dev.json --out outputs/report/motivation_${LANG}_dev
```

The untrained controls of the Moryossef architecture: the released weights with no training on this data, against random weights. Read them as a pair, never as a baseline. Their files carry `moryossef-released` and `moryossef-random<seed>`.

```bash
python eval.py --offline --spans moryossef --text none --segmenter-init released --language "$LANG" --split test --allow-test
for SEED in 0 1 2; do
    python eval.py --offline --spans moryossef --text none --segmenter-init random --seed "$SEED" --language "$LANG" --split test --allow-test
done
```

**Writes.** The row files in `outputs/offline/` (or `outputs/online/`), and `outputs/analysis/error_types_<segmenter>_<decode>_<lang>_<split>.json`, `jitter_*` and `mode_ratios_*`.

### 10.5 Boundary sweep

It moves the start and the end of each gold sentence on a grid and translates each moved clip. `boundary_sweep.severity_grid_rel` in `configs/eval.yaml` sets the grid. `--text` names the translator, as in `eval.py`.

```bash
for TR in pretrimmed ar; do
    python analyze.py --stage boundary-sweep --text "$TR" --language "$LANG" --split test --allow-test \
        --output outputs/analysis/boundary_sweep_${TR}_${LANG}.json
done
# Context after the sentence only, in seconds. Write a grid that starts with a minus sign with "=", for example --severity-grid-head=-0.3,0.
python analyze.py --stage boundary-sweep --text pretrimmed --language "$LANG" --split test --allow-test --severity-mode absolute \
    --severity-grid-head=0 --severity-grid-tail=0,0.5,1,1.5,2,3,4,6 --output outputs/analysis/boundary_sweep_pretrimmed_tail_${LANG}.json
```

For another translator file, give `--pretrimmed-checkpoint` or `--arm-checkpoint`. The sweep then refuses an `--output` whose name lacks the name tokens of that model (for example `dlm`, `onestream`, `seed1`), so it cannot write over another sweep. `report.py progress` reads `outputs/analysis/boundary_sweep_pretrimmed_<lang>.json`.

**Check.** A positive head value starts the clip late, and a negative tail value ends it early. For an arm, read `gate_skip_rate` and `text_metrics_decoded_only` beside `text_metrics`, and never plot the decoded-only scores alone. Read BLEU against the full caption only in the cells with head ≤ 0 and tail ≥ 0. Compare each curve with its own (0, 0) cell.

### 10.6 Segmenter pretraining study

It measures what the pool gives each segmenter on one target corpus `T`, in up to 3 conditions. Run it only if the paper makes a claim about the pool.

| Condition | Train data | Start weights | SayWhen-Seg (S1) | Moryossef segmenter |
| --- | --- | --- | --- | --- |
| 1. Target only | `T` | the released weights | new run | new run |
| 2. Pooled | the pool | the released weights | A2 (no new run) | A1 (no new run) |
| 3. Pooled, then target | `T` | the checkpoint of condition 2 | new run | not supported |

- **2 against 1** (both segmenters): the value of the pool. For `ase`, `asf` and `bfi`, the pool holds `T`.
- **3 against 2** (S1): the value of more training on `T`.
- **3 against 1** (S1): if condition 3 does not beat condition 1, do not claim that the pool helps.

**Fix the budget first.** Write down the maximum epochs, the patience and the batch size before the first run, and use them for every condition and every `T`. To claim a gain at equal budget, also match the optimizer updates of conditions 1 and 3.

**Set a condition.** A condition is a few keys in the config file of the segmenter. With `pretrain.languages: null`, the run trains and chooses its epoch on `T` alone.

SayWhen-Seg, in `configs/bio_pretrain.yaml`:

| Key | Condition 1 | Condition 2 (as shipped) | Condition 3 |
| --- | --- | --- | --- |
| `pretrain.languages` | `null` | `[ase, asf, bfi, bobsl_auto]` | `null` |
| `checkpoint.from_pretrained` | `checkpoints/openasl_pose_only_slt.pth` | `checkpoints/openasl_pose_only_slt.pth` | `checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto/model.pt` |
| `checkpoint.dir` | `checkpoints/bio_s1_target/${corpus}` | `checkpoints/bio_s1/${corpus}` | `checkpoints/bio_s1_finetune/${corpus}` |

Moryossef segmenter, in `configs/moryossef26.yaml`:

| Key | Condition 1 | Condition 2 (as shipped) |
| --- | --- | --- |
| `pretrain.languages` | `null` | `[ase, asf, bfi, bobsl_auto]` |
| `pretrain.dev_languages` | `null` | `[ase, asf, bfi]` |
| `checkpoint.from_pretrained` | `checkpoints/moryossef26.safetensors` | `checkpoints/moryossef26.safetensors` |
| `checkpoint.dir` | `checkpoints/moryossef_target/${corpus}` | `checkpoints/moryossef/${corpus}` |

**Rules.** A segmenter checkpoint loads only while its config file holds the pool that it trained with, so score each checkpoint while its condition is set. Run no `train-slt`, no row of §7 and no `report.py` while a condition is set. After each condition, put the shipped values back: `git diff configs/bio_pretrain.yaml configs/moryossef26.yaml` shows nothing. Choose on dev; read test once at the end.

```bash
T=asf          # asf | bfi | ase | how2sign | auslan_daily | bobsl_auto
E=100          # the maximum epochs, fixed before the first run
# The shipped keys, no training: condition 2 of SayWhen-Seg (A2) and of the Moryossef segmenter (A1).
python eval.py --offline --spans s1 --text none --decode argmax --language "$T" --split dev \
    --segmenter-checkpoint checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto/model.pt --output outputs/offline/study_case2_s1_${T}_dev.json
python eval.py --offline --spans moryossef --text none --decode argmax --language "$T" --split dev \
    --segmenter-checkpoint checkpoints/moryossef/multi_ase-asf-bfi-bobsl_auto/model.pt --output outputs/offline/study_case2_moryossef_${T}_dev.json
# Condition 1 of SayWhen-Seg: set its keys in configs/bio_pretrain.yaml, then train and score while the keys are set.
python train.py --stage train-bio --language "$T" --epochs "$E"
python eval.py --offline --spans s1 --text none --decode argmax --language "$T" --split dev \
    --segmenter-checkpoint checkpoints/bio_s1_target/${T}/model.pt --output outputs/offline/study_case1_s1_${T}_dev.json
# Condition 1 of the Moryossef segmenter: set its keys in configs/moryossef26.yaml, then train and score while the keys are set.
python train.py --stage train-moryossef --language "$T" --epochs "$E"
python eval.py --offline --spans moryossef --text none --decode argmax --language "$T" --split dev \
    --segmenter-checkpoint checkpoints/moryossef_target/${T}/model.pt --output outputs/offline/study_case1_moryossef_${T}_dev.json
# Condition 3 (S1 only): set its keys in configs/bio_pretrain.yaml. Start without --resume; after a stop, use --resume.
python train.py --stage train-bio --language "$T" --epochs "$E"
python eval.py --offline --spans s1 --text none --decode argmax --language "$T" --split dev \
    --segmenter-checkpoint checkpoints/bio_s1_finetune/${T}/model.pt --output outputs/offline/study_case3_s1_${T}_dev.json
```

The error types of a condition, while its keys are set (`S` is `s1` or `moryossef`, `K` the condition):

```bash
python analyze.py --stage segmenter-errors --language "$T" --split dev \
    --predictions outputs/offline/study_case${K}_${S}_${T}_dev.json --output-dir outputs/analysis/study_case${K}_${S}
```

**Test, once, at the end.** For each condition, set its keys again and run its scoring command with `--split test --allow-test` and `test` in place of `dev` in the output name.

**Report** for each condition: the main score, the selected epoch (`best.json`), the optimizer updates and the training time. Also say these differences: condition 2 chose its epoch on the dev sets of `ase`, `asf` and `bfi`, conditions 1 and 3 on the dev set of `T`; each SayWhen-Seg run measures its class weights on its own train records; only condition 2 has a best-epoch floor; the Moryossef condition 1 fits its input normalization on `T`. Not supported: the Moryossef condition 3, and stage 2 or a cascade row with a study segmenter.

### 10.7 SayWhen-Seg variants

**Input types.** `pretrain.inputs` in `configs/bio_pretrain.yaml` lists the input types of the train set (default: `[chunks, windows, restart_chunks]`). To compare another list, for example `[chunks]`:

1. Train one choice. Then move its folder, for example to `checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto_chunks`.
2. Train the next choice. Train the choice that SayWhen uses last, into the default folder: stage 2 reads SayWhen-Seg there.
3. Score a moved run with `--segmenter-checkpoint <folder>/model.pt` and its own `--output`. Before an online row with `--stability` of a moved run, move the `stability_s1*` files aside.

**Head from the Moryossef release.** Set `checkpoint.head_from_pretrained: checkpoints/moryossef26.safetensors` and `checkpoint.dir: checkpoints/bio_s1_moryhead/${corpus}` in `configs/bio_pretrain.yaml`, and train with the A2 command. The log prints `head initialized from`. Put the shipped values back, and score it as above with `--segmenter-checkpoint checkpoints/bio_s1_moryhead/multi_ase-asf-bfi-bobsl_auto/model.pt` and its own `--output`. The two segmenters then share most of the start of their heads, so a comparison with the Moryossef segmenter must say so.

### 10.8 Beam size and seed repeat

**Beam size.** Every main row decodes greedy. An offline AR translator can search with more beams: `--num-beams 2`, `3` or `4` (Uni-Sign with `--text pretrimmed`, and SayWhen with `--spans ar --text ar`). The text field of the file takes `-beam<k>`. Online rows and DLM decoders refuse it.

```bash
python eval.py --offline --spans ar --text ar --language "$LANG" --split dev --num-beams 4
```

**Seed repeat.** In `configs/ar.yaml`, set `seed: <n>`, `checkpoint.dir: checkpoints/ar_seed<n>/${language}` and a new `wandb.name`. Train with §5.3, undo the edit, and evaluate with `--arm-checkpoint checkpoints/ar_seed<n>/$LANG/model.pt`. Its files carry `seed<n>`. Compare two arms only when their difference is larger than the change between seeds.

### 10.9 The 4 files of an arm

An arm writes 4 weight files in its folder. Give one with `--arm-checkpoint`.

| File | Epoch | Rows carry |
| --- | --- | --- |
| `model.pt` | best `val_offline_score` (F1 × BLEU-4): segmentation and translation together | no token |
| `model_seg.pt` | best `val_offline_f1` | `-bestseg` |
| `model_text.pt` | best `val_offline_bleu4` | `-besttext` |
| `model_last.pt` | the last epoch, written when the run ends | `-last` |

The tokens keep their rows apart from the rows of `model.pt`. `model_seg.json` and `model_text.json` name the epoch of each file, as `best.json` does for `model.pt`. `val_offline_bleu4` counts only captions that match a unit, so it also moves with the F1. The tables read `model.pt`.

### 10.10 Reports

`report.py results` takes any events files as systems (`--events LABEL=PATH`). More inputs: `--history LABEL=PATH` (the `history.csv` of an arm: training curves and the selected epoch), `--segmentation-scores LABEL=PATH` (the scores file of a no-text row), `--stability-file PATH` (a stability file: the display-policy figure) and `--boundary-sweep-file PATH`. One report reads the systems of one source corpus. `report.py progress --language "$LANG"` makes one report over all finished rows. For single videos, use `report.py poses` and `report.py predict` (see their `--help`).

The tables of `outputs/report/results_<corpus>_<split>/tables/`:

| Table | What it holds |
| --- | --- |
| `edge_chunk.csv` | The main scores of each system (`no limit` lines), then the stricter grid with an edge limit τ. Precision, recall, F1, the DVC and SODA-style text scores, the counts. |
| `densevid.csv` | The scores of densevid_eval's one-caption-per-sentence pairing, with `located_corpus_bleu4`: the corpus BLEU-4 of the tIoU-matched pairs (on the gold row every sentence matches, so it is the pooled corpus BLEU-4 of all sentences). |
| `held_unit_latency.csv` | Per online system: the gold sentences, the share that a caption holds, and the median, mean and 90th percentile of the latency (s). |
| `error_types.csv` | Per system and cell (τ, tIoU): the share of each error type of the captions and the sentences. |
| `outcomes_<system>.csv` | One line per sentence and cell: its error type and the caption that counts with it. |
| `differences.csv` | The difference of each system against the reference system (values only). |
| `uncaptioned.csv` | The reliable sentences that no caption overlaps. |
| `cap_cut.csv` | The captions that the engine cut at the buffer cap, and the main scores without them. |
| `boundary.csv`, `edge_boundaries.csv`, `edge_chunk_gaps.csv`, `chunk_classes.csv`, `caption_shape.csv`, `cue_breaks.csv`, `train_text_videos.csv`, `edge_chunk_by_train_text.csv` | Breakdowns of the edges, the chunks, the caption shapes and the videos with train text. |

The code computes no confidence interval or p-value. Run resampling statistics outside the repository.

Also report, beside the scores: for the online rows, the share of forced commits, `main.uncaptioned`, the `gate_stats` counters `open_span_kept`, `open_span_cut` and `open_span_dropped_at_cap` (the events are `committed` + `open_span_cut`; the counters count cap passes, so one sentence can count more than once), the cap-cut captions apart (`tables/cap_cut.csv`, and the `cap_cut` cases in `tables/cases.json`), and the compute time per stride (`provenance.compute`); for the arms, both factors of the monitor (`val_offline_f1`, `val_offline_bleu4`); from the data audit (§3), the frames cut out of each video, the sentences quarantined for no visible hand (`units_quarantined_handless`) or for a second signer (`units_quarantined_switch_doubt`), and the train copies of dev and test videos (`deduplicated_train`).

## 11. After a change: what to run again

Find your change in the left column. Run again everything in the right column, in the order of the main path. Before a new training run, move its `latest.pt` (§2). The guards that refuse a stale checkpoint, and the checks that only warn, are in `docs/run_real_data.md` §8. Its section "No guard" lists the code changes that have no guard.

| Change | Run again |
| --- | --- |
| Arm architecture, loss, sampler, or how the translator reads the span | B3 (every arm, as new runs), then every row of the arms |
| SayWhen-Seg recipe, pool or input types (`pretrain.*` in `configs/bio_pretrain.yaml`, or a new slowest train frame rate); with windows in `pretrain.inputs`, also the window sampler (`data/sampler.py`, and `mode_ratios`, `mode2_subcase_weights` or `jitter` in `dlm.yaml`) | A2, its rows, then B3 and every row of the arms |
| `segmenter_decode` (`configs/dlm.yaml` for SayWhen-Seg and the arms, `configs/moryossef26.yaml` for A1) | A2 and B3, or A1, as new runs, then their rows |
| Moryossef checkpoint or its whole-video inference | The Moryossef rows and its error analysis |
| The input rules of `moryossef26/dataset.py`, the `release` branch of `data/chunks.py`, or the loss, augmentation or training keys of `moryossef26.yaml` | A1 as new runs, then the Moryossef rows and its error analysis |
| Uni-Sign of a corpus (or the pre-trimmed DLM translator) | The B2b entry of that decoder, the B3 arms of that decoder, and every `--text pretrimmed` row that reads it |
| `stitch.*` (`configs/dlm.yaml`) | The one-stream arms and their no-text arms, as new runs, then their rows |
| Learning rate of a stage | That stage and every stage that loads it |
| `augmentation` in `dlm.yaml`, its `enabled` key too | A2, B2a and the B2b edits, B3, then their rows. `augmentation: {enabled: false}` in `ar.yaml` turns it off for the AR arm alone: then only B3 of that arm and its rows |
| Pose converter (`prepare_yt25.py`), `poses.switch_signing_speed` or `poses.switch_raised_offset` | `python prepare_yt25.py --stage all --overwrite --languages ase asf bfi --workers 12` (it needs the shard tars), `report.py data`, then every training stage and every row |
| `poses.handless_unit_share`, `poses.switch_unit_share` or `poses.switch_doubt_share` | No conversion. Every training stage and every row |
| `poses.model_fps`, or the grid or body rule of the pose reader | No conversion. Every pose model (A1, A2, B2a and the B2b edits, B3), then every row |
| `poses.pulldown_*`, or the test of the pose reader for repeated frames | No conversion. B2a and B3 of `how2sign` and `auslan_daily`, then every row of those corpora. Nothing warns |
| The frame geometry of the pose reader | No conversion. A2, B2a and the B2b edits, B3, then every row |
| The `width`/`height` of a video in `video_meta.csv` | No conversion. Every pose model, A1 too, then every row. Nothing warns |
| A dataset's `subtitles.case_lexicon`, or the train captions of a corpus of its pool | The next train load builds that casing file again (copy it to the other machines). Then every translator (B2a, B3) and every text row of those corpora |
| New or changed data in a benchmark folder | `python prepare_yt25.py --stage dedup --languages <the benchmark keys>`, then every training stage and every row of those corpora |
| `languages.<corpus>.lowercase_references` | Every text row and report of that corpus, and its B2a and B3 runs |
| Annotation rules in `configs/data.yaml`, or any other change of the labels | `python prepare_yt25.py --stage dedup` for the corpora whose labels changed, then every training stage and every row |
| A caption file, a split file or `subtitles.dedup` | `python prepare_yt25.py --stage dedup` for those corpora, then every training stage and every row (after a change of `subtitles.dedup` alone: only when a `copy_of` value changed) |
| δ (`boundary_stability.delta_s`) or Λ_min (`span_selection.min_span_s`) in `configs/inference.yaml` | Every training stage (A1, A2, B2a, B3), then every row |
| K or stride | A2 and B3, then every row |
| Buffer cap | A2, B2a and B3, then every row |
| A larger `max_text_tokens` in `dlm.yaml` | Nothing, while every target fits the value that the models trained with (else `train-slt` stops at its start and names the value it needs). Run again a row whose captions reach about that length |
| A smaller `max_text_tokens`, or one that changes the batches | B2a and the B2b entries, B3, then every row and every boundary sweep that you keep |
| `spd.tau_dec` | The pre-trimmed DLM translator and its B2b entry, the DLM arm, then their rows |
| A checkpoint monitor | Every model that the monitor selects, then every row that reads these models |
| The GFLOP rule (`evaluation.systems.FLOP_RULE`, `FlopLedger`, `count_flops`) | Every online row with `--count-flops`, then `report.py results`. A row counted under another rule gets an empty GFLOP cell and a warning |
| Scoring (`metrics.py`, `scoring` in `configs/eval.yaml`) | Every row, the error types and every `report.py` output. `chunk_gap_s` and `tiou_thresholds` also change both monitors (the line above); `edge_taus_s` changes only the stricter grid (null or [] turns it off: `report.py` then skips its readouts, and `analyze.py --stage segmenter-errors` needs it); `chunk_gap_rows_s` changes only `report.py results` |
| The rule of the main score, or the shape of the scores file | Every saved row, because `report.py` refuses a scores file of another shape. Then the error types and every `report.py` output. `--resume` refuses a run whose `latest.pt` stamps another monitor; a finished `model.pt` still loads |
| FSM rules (`infer/stream.py`) | Every online row and the stability files. An arm that selects with `val_online_score` can select another epoch |
| Offline span rule (`infer/commit_gate.py`) | Every offline row of a segmenter or an arm, and the error types. `val_offline_score` and `val_offline_f1` can select another epoch |
| Legal decodes (`infer/duration_decode.py`: `viterbi` and `duration`) | Every row with `viterbi` or `duration` and their error types; the rows and boundary sweeps of an arm whose stamp `segmentation_decode` is `semi_markov_viterbi`; and a new B3 run, because its end term reads the semi-Markov posterior. The `argmax` rows and the `--text pretrimmed` sweep do not change |

## 12. Reference

### 12.1 Configuration

| Config | Drives | Inheritance |
| --- | --- | --- |
| `dlm.yaml` | The DLM arm; the single source of the sampler, augmentation, BIO head, gate and optimizer keys | — |
| `ar.yaml` | SayWhen (the AR arm) | `extends: dlm.yaml` (decoder and output folder only) |
| `baseline_eval.yaml` | Uni-Sign at evaluation, with no gate. Its `decoder` picks the translator that B2a trains and the `pretrained_slt` entry that `--text pretrimmed` reads | `extends: dlm.yaml` |
| `baseline_train.yaml` | B2a into `checkpoints/baseline_train_<decoder>/<lang>` | `extends: baseline_eval.yaml` |
| `bio_pretrain.yaml` | SayWhen-Seg (`train-bio`): its rates, its pool and its input types | `extends: dlm.yaml` |
| `moryossef26.yaml` | The Moryossef segmenter (`train-moryossef`) | standalone |
| `inference.yaml` | The fixed constants of the streaming engine (stride, K, δ, cap, Λ_min), in seconds. No step writes them; do not tune them | standalone |
| `data.yaml` | The corpora, their roots, split files, caption rules, casing files and frame sizes; the translator starts (`pretrained_slt`); the pose reader (`poses`) | — |
| `eval.yaml` | The scoring of the rows (`scoring`) and the boundary-sweep grid (`boundary_sweep`) | — |

`--language` points every templated path (`${language}`) at the corpus, so do not edit a config for one corpus.

### 12.2 Output folders

| Folder | Holds | Written by |
| --- | --- | --- |
| `outputs/offline/`, `outputs/online/` | The events files of the rows and their scores files, by the access of the row | `eval.py` |
| `outputs/analysis/` | The error analysis of the segmenters and the boundary sweeps | `analyze.py` |
| `outputs/stability/` | The display-policy scores (`stability_*`) and every recorded decode (`provisional_*`) | `eval.py --online --stability` |
| `outputs/report/` | One folder per report: `results_<corpus>_<split>/`, `progress/`, `data/`, `poses/`, `predict/` | `report.py` |

### 12.3 Row file names

Every row writes 2 files. The fields of the name are the flags of the row:

```
{access}_{spans}_{decode}_{text}_{corpus}_{split}.json          the events file
{access}_{spans}_{decode}_{text}_{corpus}_{split}_scores.json   the scores file
```

| Field | Values |
| --- | --- |
| `access` | `offline`, `online` (`--offline`, `--online`). A row on the spans of a file takes the access of the file. |
| `spans` | `gold`, `moryossef`, `s1`, `joint-ar`, `joint-dlm`, `chance` (`--spans`). An arm other than SayWhen adds its name tokens (§10.2). The untrained controls are `moryossef-released` and `moryossef-random<seed>`. |
| `decode` | `argmax`, `viterbi`, `duration` (`--decode`, else the decode that chose the epoch of the model), or `none` for the gold and chance spans. |
| `text` | `pretrimmed`, `pretrimmed-dlm`, `ar`, `dlm`, `none` (`--text`), with the name tokens of an arm and `-beam<k>` for a beam search. |
| `corpus` | The corpus key, or `<source>-to-<target>` for a transfer row (`--source-language`). |

For example, `offline_moryossef_argmax_pretrimmed_asf_test.json` holds the spans of the Moryossef segmenter with the argmax decode, read by Uni-Sign, offline, on the `asf` test set.

The scores file holds, in this order: `provenance` (what made the row), `main` (`edge_chunk`: the main scores and the stricter grid; `uncaptioned`), `densevid` (the scores of densevid_eval's pairing) and, for a no-text row, `frame` (the frame scores of the protocol of Moryossef et al.). The definitions are in `streaming_slt_prompt.md` §9.2.

### 12.4 Repository map

```
backbones/       the Uni-Sign pose encoder (ST-GCN over 69 keypoints)
poses/           pose files: Uni-Sign normalization, pose loading and video_meta.csv, augmentation
data/            records to samples: loader, BIO labels and text targets, stage-2 sampler, segmenter chunks, batches
models/          model parts: BIO head, gate, Uni-Sign front end with mT5, DLM decoder, the full model, checkpoint checks
train/           training loops: arms and Uni-Sign (slt), SayWhen-Seg (bio_pretrain), losses, epoch loop, several GPUs
moryossef26/     the Moryossef segmenter: model, input, training, whole-video inference
infer/           inference: the decodes, span and commit rules, DLM decode, the streaming engine, display stability
evaluation/      what eval.py runs: events and gold, model builds, rows, offline and online rows, the boundary sweep
reporting/       what report.py runs: breakdowns, outcome tables, figures, audits, debug views
metrics.py       every metric calculation
utils.py         config loading, checkpoint and output folders, device choice
train.py         --stage {smoke-data, train-bio, train-moryossef, train-slt}
eval.py          one row per command; --predictions scores a saved events file again
analyze.py       --stage {dataset-summary, segmenter-errors, boundary-sweep}
report.py        {results, progress, data, poses, predict}
prepare_yt25.py  YouTube-SL-25 through SignVerse-2M: shards, pose converter, captions, video_meta.csv, copy check
docs/            data_pipeline.md, membership_gate.md, segmentation_decoding.md, run_real_data.md, implementation_notes.md, literature_notes.md
```
