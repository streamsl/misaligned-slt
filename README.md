# Misaligned SLT

Misaligned SLT translates sign language into English text while the video streams in. It reads only body poses (keypoints) and uses no glosses (word-by-word sign labels). It must stay correct when the sentence segmentation is wrong: a predicted span can start late, end early, or join two captions.

| Corpus key | Sign language | Data | Use |
| --- | --- | --- | --- |
| `ase`, `asf`, `bfi` | ASL, Auslan, BSL | YouTube-SL-25, with the SignVerse-2M DWPose keypoints | target corpora, and the segmenter pool |
| `how2sign` | ASL | How2Sign | target corpus |
| `auslan_daily` | Auslan | Auslan-Daily News (V1 and V2) | target corpus |
| `bobsl_manual` | BSL | BOBSL, manual alignment | target corpus |
| `bobsl_auto` | BSL | BOBSL, automatic alignment | the segmenter pool only |

The data of the 3 benchmarks (How2Sign, Auslan-Daily, BOBSL) is already in place in `data/how2sign`, `data/auslan-daily-abc` and `data/bobsl`. Their keypoints come from the extractor slp-pose. The pose reader gives every video at 24 fps. All targets are English.

Two rules control all training data:

- **P1:** a window that cuts a caption unit gets no text label for that unit. For example, a window that ends 1 s before its only unit ends trains only the segmentation head.
- **P2:** a segmenter error changes the window, never the label. For example, if the predicted span starts 0.3 s late, the translator still learns the full caption text of that unit.

More detail:

- [`streaming_slt_prompt.md`](streaming_slt_prompt.md): the design spec, with its hard rules in §1.4. This README cites it as "spec §x".
- [`docs/membership_gate.md`](docs/membership_gate.md): how segmentation conditions translation.
- [`docs/segmentation_decoding.md`](docs/segmentation_decoding.md): the segmentation decoder.
- [`docs/data_pipeline.md`](docs/data_pipeline.md): the captions, labels and poses. Read it before you change anything that touches the labels.
- [`docs/run_real_data.md`](docs/run_real_data.md): the checks on real data and the code guards.

**Contents:** [Words](#words-used-in-this-readme) · [How the system works](#how-the-system-works) · [The models](#the-models)

**Steps:** [1. Setup](#1-setup) · [2. Run order and rules](#2-run-order-and-rules) · [3. Prepare the data](#3-prepare-the-data) · [4. Stage A](#4-stage-a-the-segmenters-a1-a2) · [5. Clean translators](#5-stage-b-the-clean-translators-b2a-b2b) · [6. Arms](#6-stage-b-the-arms-b3) · [7. Evaluate](#7-evaluate) · [8. Read the results](#8-read-the-results) · [9. Optional studies](#9-optional-studies) · [10. After a change](#10-after-a-change-what-to-run-again) · [11. Reference](#11-reference)

## Words used in this README

Each table gives the words in the order of their first use.

**Data and labels**

| Word | Meaning | Example |
| --- | --- | --- |
| caption unit | One or more caption cues joined into one timed unit. The system finds and translates units. A unit can hold several sentences. | Four cues of video `_YKe8JJzju4` form one unit from 48.041 s to 70.790 s. |
| BIO | The frame labels of segmentation: B is the first frame of a unit, I its other frames, and O a frame outside every unit. | Two pause frames, then a unit of four frames: `O O B I I I`. |
| span | A run of frames that the segmentation marks as one unit: a B frame and its I frames. | |
| terminator | The first O or B frame after a span. It ends the span. | When two units are back to back, the B of the second unit ends the first. |
| complete span, open span | A complete span has its terminator in the frames that the model reads. An open span has no terminator yet. | |
| window, buffer | The pose frames that a model reads at one time. The sampler makes the training windows. The streaming loop holds a buffer of at most 40 s. | |
| gold | The caption units of the annotation and their times. | `outputs/gold/gold_asf_test.json` |
| UNK | A label that every loss and every score ignores: padding, and the frames of quarantined units. | |
| quarantined unit | A unit that the data cannot support. It is UNK on every frame, and it is never a training target or gold. | A unit with no visible hands, or a unit that crosses a stream edge. |
| stream | One clean part of a source video. The loader cuts out the frames with no person, and each remaining part is a stream. | `abc123@1440` starts at frame 1440 of video `abc123`. |
| window mode | The 4 kinds of stage-2 training window. Mode 1 holds one complete unit. Mode 2 cuts a unit, so it has no text label (P1). Mode 3 holds several complete units. Mode 4 holds no unit. | A window from 1 s before a unit to after its end is a Mode-1 window. |
| joined target, `<sep>` | Complete units with gaps of at most δ give one text target, "A `<sep>` B". `<sep>` is a separator token. A committed span stays one caption with its whole text, and `<sep>` reads as a space. | "Hello." and "My name is Sam." with a 0.2 s gap give "Hello. `<sep>` My name is Sam." |

**Steps and models**

| Word | Meaning | Example |
| --- | --- | --- |
| Stage A, Stage B | Stage A trains the 2 segmenters once, on the pool. Stage B runs once per target corpus: it trains the translators and runs every evaluation. | |
| pool | The corpora that Stage A trains on together: `ase`, `asf`, `bfi` and `bobsl_auto`. | Its folders are named `multi_ase-asf-bfi-bobsl_auto`. |
| S1 | The pretrained segmentation part: a pose encoder and a BIO head, trained on the pool. | `checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto/model.pt` |
| Moryossef segmenter | The external segmenter baseline of Moryossef et al., trained again on the pool, with its own 50-landmark input. | `checkpoints/moryossef/multi_ase-asf-bfi-bobsl_auto/model.pt` |
| clean translator | The released Uni-Sign translator, fine-tuned on the caption crops of one corpus. It has no segmentation part. There is one for each decoder: the clean AR translator and the clean DLM translator. | `checkpoints/baseline_train_ar/asf/model.pt` |
| arm, stage 2 | A model that segments and translates. Stage 2 (step B3) trains it from S1 and the clean translator of its decoder. The main arms are the AR arm and the DLM arm. | `checkpoints/dlm/asf/model.pt` |
| no-text arm | An arm whose text loss does not reach its segmentation part. It shows what stage 2 gives without the text. | `checkpoints/dlm_notext/asf/model.pt` |
| AR, DLM | The 2 decoder families. AR writes one token at a time. DLM (a diffusion language model) fills a block of tokens and can change them before the block is final. | |
| Ω (the gate) | A hard crop of the selected span. The translator of an arm reads only these frames. | For the span [100, 136), the translator reads frames 100 to 136. Frame 136 is the terminator. |
| monitor | The dev score that selects the best epoch of a training run. The best epoch goes to `model.pt`. | `val_video_f1` for the segmenters |
| best-epoch floor | The first epoch that can become the best epoch of a segmenter run: one full rotation of the pool. The run prints it. | |
| annotation fingerprint | A hash of the record ids, the unit times and the quarantine flags. Checkpoints and output files stamp it, so a later run sees when the labels changed. | |

**Streaming**

| Word | Meaning | Example |
| --- | --- | --- |
| δ, K, stride, cap, Λ_min | The fixed constants of `configs/inference.yaml`: the boundary tolerance δ = 0.5 s, K = 3 strides, the stride of 1 s, the cap of 40 s (the longest buffer), and the shortest unit Λ_min = 0.5 s. | At 24 fps, δ and Λ_min are 12 frames each. |
| FSM | The streaming loop (a finite-state machine). On each stride it segments the buffer again. It commits a span when the span's terminator stays within δ for K strides. | |
| commit | The time when the FSM emits the text of a unit. The text never changes after its commit. | |
| forced commit (PARTIAL) | At the cap or at the stream end, the FSM commits each complete span at once and flags it PARTIAL. | |
| cap cut | At the cap, an open span that cannot wait for the next buffer commits its closed part once. The FSM cuts it where the head is least sure that the unit goes on. It is the one exception to P1 at inference (spec §1.4 rule 18). The offline rows have no cap. | A unit starts at 0.6 s and has no end at 40.5 s: the FSM emits one caption for 0.6–31.2 s, and the next buffer starts at 30.7 s. |
| reveal policy | A display rule that shows text before its commit. Only the stability analysis uses it. The deployed output shows text only at its commit. | `agreement_n2` |

**Evaluation**

| Word | Meaning | Example |
| --- | --- | --- |
| row, result table | A row captions whole test videos end to end, from segmentation to text. The result table holds the rows (§8.1). | Row 10 is the deployed system. |
| span file, events file | A span file holds predicted spans with no text. An events file holds the timed captions of one row. | |
| cascade | A row in which a separate segmenter makes the spans and a translator reads them. | Row 3: Moryossef spans, clean translator. |
| same-span control | A row in which the clean translator reads the spans of an arm's row, so only the translator changes. | Row 7 reads the spans of row 6. |
| edge chunk F1, τ | The main localization score. A caption matches a run of units when both of its edges are within τ of the edges of the run (§8.3). | |
| error types | Per τ, each caption is matched, joined, cut, bridged, offset or phantom, and each unit is matched, joined, cut, bridged, offset or missed (§8.3). | A caption that holds units on both sides of a pause is bridged. |
| tIoU | Temporal intersection over union: the overlap of two time intervals, divided by their union. | [0, 4] s and [1, 5] s give 3 / 5 = 0.6. |
| DVC | Dense video captioning. Here it names the text scores of all captions of a video, added up as the densevid_eval toolkit does. | |
| SODA | A DVC text score (Fujita et al.) that also charges captions with no match and missed units. | |
| boundary sweep | An analysis that moves the edges of each gold unit on a grid and translates each moved crop (§7.2). It is not part of the paper's evidence. | |
| provenance | The block of an output file that records what made it: the checkpoint, the configs and the decode. | |
| `$LANG` | The target corpus of a Stage B command. | `LANG=asf` |

## How the system works

```
pose frames → Uni-Sign normalization ─┬→ segmentation pose encoder (from S1) → BIO head → BIO scores
                                      │     → duration decode (train-fitted duration prior) → spans
                                      │           ├→ first complete span → Ω (a hard crop)
                                      │           └→ the FSM commits the spans while the video streams in
                                      └→ translator pose encoder (from the clean translator)
                                            → task prompt + mT5 encoder   ← Ω on the self-attention keys
                                            → AR or DLM decoder           ← Ω on the cross-attention
```

- Two pose encoders with separate weights read the same pose frames. The segmentation encoder starts from S1, and the translator encoder starts from the clean translator (spec §4.7).
- The BIO head is a transformer over time. Each frame attends only to nearby frames, so the head reads only a short time ahead (§11.1). The duration decode scores whole spans with a prior on the unit duration, fitted on the train labels ([`docs/segmentation_decoding.md`](docs/segmentation_decoding.md)).
- The gate Ω selects the first complete span, and the translator encodes only that crop. In training, the text moves the segmentation only at the end of a target that a pause follows, so it cannot reward a merge ([`docs/membership_gate.md`](docs/membership_gate.md) "Current design").
- The translator is mT5. Its encoder reads a fixed task prompt (the instruction text of Uni-Sign), then the pose features. The AR arm decodes token by token, and the DLM arm decodes in blocks (spec §4.5).
- The FSM commits a span when its terminator stays still for K strides. At the cap or at the stream end it forces the commits, and at the cap it cuts an open span that it cannot keep (spec §7).

## The models

This README uses one name for each model. The table gives the name of the same model in commands, configs and output files.

| Model | Step | Training command | Config | Checkpoint | Evaluation flag | Name in output files |
| --- | --- | --- | --- | --- | --- | --- |
| Moryossef segmenter | A1 | `train.py --stage train-moryossef` | `moryossef26.yaml` | `checkpoints/moryossef/multi_ase-asf-bfi-bobsl_auto/model.pt` | `--segmenter-arch moryossef` | spans `moryossef` |
| S1 | A2 | `train.py --stage train-bio` | `bio_pretrain.yaml` | `checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto/model.pt` | `--segmenter-arch s1` | spans `s1` |
| clean AR translator | B2a | `train.py --stage train-slt --slt-config configs/baseline_train.yaml` | `baseline_train.yaml` to train, `baseline_eval.yaml` to evaluate (`decoder: ar`) | `checkpoints/baseline_train_ar/$LANG/model.pt` | `--method baseline` | translator `clean` |
| clean DLM translator | B2a | the same command, with `decoder: dlm` in `baseline_eval.yaml` for the run (§5) | the same | `checkpoints/baseline_train_dlm/$LANG/model.pt` | `--method baseline --checkpoint <that file>` | translator `clean-dlm` |
| AR arm | B3 | `train.py --stage train-slt --slt-config configs/ar.yaml` | `ar.yaml` | `checkpoints/ar/$LANG/model.pt` | `--method ar` | spans `joint-ar`, translator `ar` |
| DLM arm (the main system) | B3 | `train.py --stage train-slt --slt-config configs/dlm.yaml` | `dlm.yaml` | `checkpoints/dlm/$LANG/model.pt` | `--method dlm` | spans `joint-dlm`, translator `dlm` |

## 1. Setup

**Needs.** Python 3.9 to 3.13. A CUDA GPU to train: the mT5 backward pass does not fit in CPU memory. An Apple MPS or CPU machine is enough to evaluate. The code selects `cuda`, then `mps`, then `cpu`; `--device` overrides this order.

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
curl -L -o checkpoints/how2sign_pose_only_slt.pth \
    https://huggingface.co/ZechengLi19/Uni-Sign/resolve/eab251b7fe7e8521afc0e67be98add670ea40a0d/how2sign_pose_only_slt.pth
curl -L -o checkpoints/moryossef26.safetensors \
    https://raw.githubusercontent.com/sign-language-processing/segmentation/a4596048fadf22864e2fdd6a2ef3ca1004835737/sign_language_segmentation/dist/2026/model.safetensors
```

| File | What it is | Used by |
| --- | --- | --- |
| `checkpoints/openasl_pose_only_slt.pth` | The released Uni-Sign OpenASL pose-only translator ([Uni-Sign](https://github.com/ZechengLi19/Uni-Sign)) | S1 (A2), and the clean translators of every corpus except How2Sign (B2a) |
| `checkpoints/how2sign_pose_only_slt.pth` | The released Uni-Sign How2Sign pose-only translator | The clean translators of How2Sign (B2a) |
| `checkpoints/moryossef26.safetensors` | The released weights of the Moryossef segmenter ([sign-language-processing/segmentation](https://github.com/sign-language-processing/segmentation)) | The Moryossef segmenter (A1) |

The first run also downloads mT5 (`google/mt5-base`) from HuggingFace, the NLTK `punkt` and `punkt_tab` data, and the METEOR metric with its NLTK data. For a machine with no internet, run these commands on a machine with internet. Then copy the HuggingFace cache (`~/.cache/huggingface`) and the NLTK data folder (`~/nltk_data`) to the other machine.

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python -c "import evaluate; evaluate.load('meteor')"
python -c "from transformers import T5Tokenizer; T5Tokenizer.from_pretrained('google/mt5-base', legacy=False)"
python -c "from transformers import MT5ForConditionalGeneration; MT5ForConditionalGeneration.from_pretrained('google/mt5-base')"
```

**BLEURT (optional).** Without it, the BLEURT column reads 0.0, and one WARNING says so. The code reads the BLEURT-20 checkpoint at `/tmp/BLEURT-20`. Many systems clear `/tmp` when the machine starts again, so make sure that the folder is there before an evaluation.

```bash
pip install git+https://github.com/google-research/bleurt.git
curl -L -o /tmp/BLEURT-20.zip https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip /tmp/BLEURT-20.zip -d /tmp
```

On a CUDA machine, BLEURT also needs 2 files of the CUDA compiler that the torch wheels do not include. Without them, BLEURT stops with `JIT compilation failed` and `libdevice not found`, and the column reads 0.0. Install the compiler package into the site-packages folder of TensorFlow (`python -m pip show tensorflow` shows it; add `--user` when it is in `~/.local`). Version 12.8 agrees with torch `+cu128`. A machine with a full CUDA toolkit can set `XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda` instead.

```bash
python -m pip install "nvidia-cuda-nvcc-cu12==12.8.*"
```

Check BLEURT before a long evaluation. For two equal sentences, this command prints a score near 1.0 and no WARNING:

```bash
python -c "import metrics; print(metrics.pair_text_scores(['the cat sat on the mat'], ['the cat sat on the mat'], False, keys=('bleurt',)))"
```

**Check before you continue.**

- `python -c "import transformers; print(transformers.__version__)"` prints a 4.x version. Version 5.x loads the Uni-Sign checkpoint wrong with no error, and every translation is then wrong.
- `python -c "import sacrebleu, evaluate, rouge, nltk, pycocoevalcap"` prints no error. Training and evaluation do not start without `sacrebleu`. When one of the other packages is missing, its score column reads 0.0, and the run prints a WARNING.

## 2. Run order and rules

### 2.1 Run order

Run the steps in this order. The step labels are fixed names that configs and code messages also use (for example "README B2b"). They are names, not a count, so there is no step B1.

| Step | What | Section | Runs on | Needs | Writes |
| --- | --- | --- | --- | --- | --- |
| A0 | Prepare YouTube-SL-25 | §3.1 | any machine with internet and much disk space | the split CSV | `data/youtube-sl-25/` |
| A0b | Check the 3 benchmarks | §3.2 | any machine | the benchmark folders | `copy_of` in the benchmark sidecars, and their casing files |
| A1 | Train the Moryossef segmenter | §4 | CUDA GPU | A0, A0b | `checkpoints/moryossef/multi_ase-asf-bfi-bobsl_auto/` |
| A2 | Train S1 | §4 | CUDA GPU | A0, A0b | `checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto/` |
| B2a | Train the clean AR translator and the clean DLM translator | §5 | CUDA GPU | A0 (A0b for a benchmark) | `checkpoints/baseline_train_ar/$LANG/`, `checkpoints/baseline_train_dlm/$LANG/` |
| B2b | Point `configs/data.yaml` at the clean translators | §5 | every machine that runs B3 or an evaluation | B2a | an edit of `configs/data.yaml` |
| B3 | Train the AR arm, the DLM arm, and the no-text arm of each | §6 | CUDA GPU with much memory | A2, B2b | `checkpoints/{ar,dlm,ar_notext,dlm_notext}/$LANG/` |
| B4 | Segmenter scores, span files, error types | §7.1 | GPU or Mac | A1, A2 | `outputs/segmenter/`, `outputs/analysis/` |
| B5 | Boundary sweep (optional, not in the paper) | §7.2 | GPU or Mac | B2b, B3 | `outputs/analysis/` |
| B6 | The rows of the result table | §7.3, §7.4, §7.6 | GPU or Mac | B2b, B3, B4 | `outputs/gold/`, `outputs/offline/`, `outputs/online/`, `outputs/stability/` |
| B7 | Reports | §7.5 | any machine | B4, B6 | `outputs/report/` |

- A1, A2 and B2a do not depend on each other, so run them at the same time. Run the 2 B2a runs one after the other, or from 2 copies of the repository: the clean DLM translator needs a config edit (§5).
- The target corpora are `ase`, `asf`, `bfi`, `how2sign`, `auslan_daily` and `bobsl_manual`. `bobsl_auto` only joins the segmenter pool.
- The main path trains 38 models: the 2 segmenters once, and for each target corpus the 2 clean translators, the 2 arms and the 2 no-text arms. The optional studies of §9 do not change the main results.
- Stage A commands refuse `--language`, because a pool has no target. Every Stage B command needs `--language "$LANG"`. Without it, the code uses the first entry of `active_languages` in `configs/data.yaml` (`asf`).
- The arms are the largest runs, and they need a GPU with much memory. The online rows (§7.4) take much longer than the offline rows (§7.3).

### 2.2 Two machines

Training needs a CUDA GPU, and evaluation can run on a Mac. When you use more than one machine, copy these files between them:

| To | Copy |
| --- | --- |
| each training machine | The repository, and the 3 released files of §1 in `checkpoints/`. Of `data/youtube-sl-25/`: the split CSV, `case_lexicon.en_XX.json`, and in each `<lang>/` folder `poses/`, `subs/` and `video_meta.csv` (not `signverse_shards/`). Of each benchmark folder: `poses/`, `subs/`, the split files, the sidecars (`video_meta*.csv`) and `case_lexicon.en_XX.json` (not `persons/` and not `.cache/`). |
| the machine of B3 | The `model.pt` of S1 (A2) and of the clean translator of each decoder (B2a), each in its folder. |
| the evaluation machine | The data, with the casing file of each corpus (the dev and test loaders refuse to start without it). The `model.pt` of every trained model, each in its folder ("The models" gives the folders). The `history.csv` of the arms, for the training curves of the reports (§7.5). |
| a machine that continues a run with `--resume` | The `latest.pt` of that run too. |

Use the same `configs/` (with the B2b edit) and the same prepared data on every machine: data from another preparation gives fingerprint WARNINGs. After a data step changes files on one machine, copy the changed files to the others. A rule change or a copy check (`--stage dedup`) usually changes only the sidecars (`video_meta*.csv`) and the casing files.

### 2.3 Rules for every run

Set the target corpus once in each shell:

```bash
LANG=ase    # ase (ASL) | asf (Auslan) | bfi (BSL) | how2sign (ASL) | auslan_daily (Auslan) | bobsl_manual (BSL)
```

`LANG` is also the locale variable of the shell, so some tools (for example `perl`) can print locale warnings in this shell.

- Run on test only when training and checkpoint selection are final. Commands that read model outputs refuse `--split test` without `--allow-test`. No training stage reads test.
- Every epoch writes `latest.pt`. To continue an interrupted run, run the same command again with `--resume`. `--resume` refuses a run whose recipe changed (rates, augmentation, monitor or architecture): start a new run then. [`docs/run_real_data.md`](docs/run_real_data.md) §8 lists these guards.
- Without `--resume`, a run refuses to start while `latest.pt` exists. To start a run again from the beginning, move `latest.pt`, `model.pt` and `best.json` out of the checkpoint folder.
- Every epoch prints one row. Its `ckpt` column shows `saved` when the run writes a new best `model.pt`. The same rows go to `history.csv` in the checkpoint folder.
- The loader keeps the records of each corpus and split in `<root>/.cache/`. It loads them again by itself when the loader code, a config or a data file changes. To load again by force, delete the `.cache/` folder.
- Every output file goes to `outputs/<kind>/` (§11.2). A run writes over a file with the same name, so archive your result files before you run a command again.
- Some runs need a config edit while they train: the clean DLM translator (§5), the no-text arms (§6) and the ablation arms (§9.1). While such an edit is in place, start no other run from the same `configs/` folder, because it reads the edit too. To run them at the same time, use a second copy of the repository.
- A WARNING that a model or a file comes from other train labels (another annotation fingerprint) does not stop the run. Train again, or make the files again, before you report that run.

### 2.4 Several GPUs

Put `torchrun --standalone --nproc-per-node=N` before `train.py`, and change nothing else:

```bash
torchrun --standalone --nproc-per-node=4 train.py --stage train-slt --language "$LANG" --slt-config configs/dlm.yaml
```

- `batch_size` and `checkpoint.dev_batch_size` are global batches, and each must be at least N. N GPUs train as 1 GPU on the same global batches, up to float rounding.
- fp16 works on several GPUs. `mixed_precision: auto` uses bf16 when the GPU has it, and fp16 when it does not (for example a T4).
- Evaluation runs in one process. Details: [`docs/run_real_data.md`](docs/run_real_data.md) §5.

## 3. Prepare the data

### 3.1 YouTube-SL-25 (A0)

**What it does.** It downloads the SignVerse-2M pose shards of `ase`, `asf` and `bfi`, and converts them to one pose file per video that keeps the person who signs. It writes the captions, the casing file, and one sidecar `video_meta.csv` per language with the facts about each video that the loader reads. The last step, the copy check, marks each train video that copies a dev or test video (a re-upload), and the train loader drops it. The data rules are in [`docs/data_pipeline.md`](docs/data_pipeline.md).

**Runs on.** Any machine with internet and much free disk space. It needs no GPU.

**Needs.** The split CSV `data/youtube-sl-25/SignVerse-2M-metadata_split.csv` (the first command below downloads it) and free disk space for the shard cache (`--stage plan --size` estimates it). Fix the annotation rules in `configs/data.yaml` (caption cleanup, cue grouping, case folding) before Stage A, and keep them fixed while you compare systems.

Download the split CSV:

```bash
mkdir -p data/youtube-sl-25
curl -L -o data/youtube-sl-25/SignVerse-2M-metadata_split.csv \
    https://huggingface.co/datasets/SignerX/SignVerse-2M/resolve/0c7e956a8471acad3a9cebaeb66bb23bf5deaafa/SignVerse-2M-metadata_split.csv
```

Prepare the three languages, then write the corpus audit:

```bash
# Optional: shard and video counts, with a size estimate.
python prepare_yt25.py --stage plan --size
# All steps in order: download, convert, person-counts, subs and the copy check (dedup).
python prepare_yt25.py --stage all --languages ase asf bfi --workers 12
# The corpus audit.
python report.py data --language ase --language asf --language bfi
```

Check the sampler on each language:

```bash
python train.py --stage smoke-data --language "$LANG" --split train --num-samples 64
```

**Repair.** `--stage all` already runs every step. Run a step alone only to repair the data:

```bash
python prepare_yt25.py --stage person-counts --languages ase asf bfi --workers 12
python prepare_yt25.py --stage subs --languages ase asf bfi
python prepare_yt25.py --stage dedup --languages ase asf bfi
```

- `person-counts` fills the blank values of the sidecar, and `subs` adds a caption to a converted video that has none. Neither changes a pose file or an existing caption.
- `convert` and `subs` blank the `copy_of` column of their languages, and the train loader refuses a blank value. So run `dedup` again after them.
- If the loader refuses a sidecar and names `--stage all --overwrite`, its pose files need a new conversion. Run `python prepare_yt25.py --stage all --overwrite --languages ase asf bfi --workers 12` (it needs the shard tars).
- `--workers N` converts N shards at the same time. Keep N below the number of CPU cores and below the free memory in GB, with about 9 GB more free for the longest videos. A stopped run loses at most the shards in progress: run the same command again without `--overwrite`.
- `--delete-tars` frees the shard cache in `data/youtube-sl-25/signverse_shards/`, but a later conversion needs the tars again.

**Writes.**

- `data/youtube-sl-25/<lang>/poses/<vid>.npy`, `data/youtube-sl-25/<lang>/subs/<vid>.<track>.vtt` and `data/youtube-sl-25/<lang>/video_meta.csv`.
- `data/youtube-sl-25/case_lexicon.en_XX.json`: the casing file of the three languages. The dev and test loaders refuse to start without it. So build it on a machine that holds the train captions of all three languages, and copy it to the other machines.
- `outputs/report/data/data_audit_<lang>.md` and `.json`: what each data rule removed or repaired, per split.

**Check before you continue.**

- Every `person-counts` line shows `empty_runs 0/N`, `handless_runs 0/N`, `switch_runs 0/N` and `default_signing_runs 0/N`. The first number counts the rows that are still unknown, so 0 is correct.
- Each `dedup | <lang>` line gives the number of train videos that copy a dev or test video.
- In the `convert | reconciliation` line, `shard not downloaded` is 0. If it is not 0, run `--stage all` again.
- `smoke-data` shows `records` and `translation_targets` above 0, and `padding_label` `UNK`. Its `batch_pose_shape` ends in `[69, 3]`, and `modes` holds more than one mode ([`docs/run_real_data.md`](docs/run_real_data.md) §4).

### 3.2 The benchmarks (A0b)

**What it does.** The 3 benchmarks come ready in `data/how2sign`, `data/auslan-daily-abc` and `data/bobsl`: poses, captions, split files and sidecars. This step runs only the copy check on them, and it builds the casing file of each benchmark. The pose reader gives their videos the same input form as YouTube-SL-25.

| Corpus key | Root | Split file | Subtitle track | Sidecar |
| --- | --- | --- | --- | --- |
| `how2sign` | `data/how2sign` | `splits.csv` | `.en.vtt` | `video_meta.csv` |
| `auslan_daily` | `data/auslan-daily-abc` | `splits.csv` | `.en.vtt` (all lowercase) | `video_meta.csv` |
| `bobsl_manual` | `data/bobsl` | `splits_manual.csv` | `.en-manual.vtt` | `video_meta_manual.csv` |
| `bobsl_auto` | `data/bobsl` | `splits_automatic.csv` | `.en.vtt` | `video_meta.csv` |

- `bobsl_auto` only joins the segmenter pool (§4). Its test split is never read, and its dev split does not choose a checkpoint.
- The 2 BOBSL corpora share one folder of episodes. Each split file puts only the episodes of its corpus in train, dev or test, and each load prints how many episodes it uses.
- No segmenter sees a dev or test episode of `bobsl_manual`: they are in the official val and test splits, which the pool does not train on.
- In `bobsl_manual`, the cues that the BOBSL annotators tagged as not matching the signing are unlabelled.

**Runs on.** Any machine with the benchmark folders. It needs no GPU.

**Needs.** In each benchmark folder: `poses/`, `subs/`, the split files and the sidecars.

```bash
# The copy check of each benchmark. Its first train load also builds the casing file of the benchmark.
python prepare_yt25.py --stage dedup --languages how2sign auslan_daily bobsl_auto bobsl_manual
python report.py data --language how2sign --language auslan_daily --language bobsl_manual
```

**Writes.** The `copy_of` column of each benchmark sidecar, and `case_lexicon.en_XX.json` in each benchmark folder.

**Check before you continue.**

- Each `dedup | <corpus>` line gives the number of train videos that copy a dev or test video (expected: 0 for every benchmark).
- `smoke-data` runs on each benchmark key (`--language how2sign` and so on), as in §3.1.

## 4. Stage A: the segmenters (A1, A2)

**What it does.** It trains the 2 segmenters once, on the pool. Each sign language fills one third of the hours of every epoch: ASL (`ase`), Auslan (`asf`), and BSL (`bfi` with `bobsl_auto`). The dev sets of `ase`, `asf` and `bfi` choose the checkpoint (`val_video_f1`). The Moryossef segmenter (A1) is the external baseline: it makes the spans of the cascade rows and of the error analysis. S1 (A2) is the start of the segmentation part of every arm.

**Runs on.** A CUDA GPU.

**Needs.** The data of §3, with the BOBSL official-train episodes. `checkpoints/moryossef26.safetensors` for A1, and `checkpoints/openasl_pose_only_slt.pth` for A2. The recipe and its reasons are in spec §4.4b and §4.6.

```bash
# A1: the Moryossef segmenter.
python train.py --stage train-moryossef
# A2: S1, the pose encoder and the BIO head.
python train.py --stage train-bio
```

**Writes.** `checkpoints/moryossef/multi_ase-asf-bfi-bobsl_auto/` and `checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto/`, each with `model.pt`, `best.json`, `latest.pt` and `history.csv`.

`pretrain.inputs` in `configs/bio_pretrain.yaml` sets the input types of S1: `chunks`, `windows` and `restart_chunks`, all three by default (§11.1, spec §4.4b). §9.3 compares other choices.

**Check before you continue.**

- Each run prints its best-epoch floor, and no epoch before the floor shows `ckpt saved`. `train.py` refuses a run with fewer epochs than the floor.
- Train both segmenters from one code state. Before you quote a cascade result, make sure that the 2 checkpoints have the same `pretrain_mix` and the same `annotation_fingerprint`. This command prints both values for each checkpoint:

```bash
python - <<'EOF'
from models.checkpointing import load_checkpoint_meta
for kind in ("moryossef", "bio_s1"):
    meta = load_checkpoint_meta(f"checkpoints/{kind}/multi_ase-asf-bfi-bobsl_auto/model.pt")
    print(kind, meta.get("pretrain_mix"), meta.get("annotation_fingerprint"))
EOF
```

- If S1 runs out of GPU memory, lower `batch_size` in `configs/bio_pretrain.yaml`, and raise `grad_accum_steps` by the same factor.
- Quote the segmenter scores of §7.1, not the monitor.

## 5. Stage B: the clean translators (B2a, B2b)

**What it does.** B2a fine-tunes a released Uni-Sign translator on the caption crops of one corpus. It does this once for each decoder: the clean AR translator (cross-entropy) and the clean DLM translator (the OPUT loss). Each corpus starts from the OpenASL release, except How2Sign, which starts from Uni-Sign's How2Sign release (`released_slt` in `configs/data.yaml`). B2b points `configs/data.yaml` at the 2 clean translators, so the arm and the cascade start from the same translator. B2 does not need Stage A, so run it at the same time.

**Runs on.** A CUDA GPU for B2a. B2b is a config edit.

**Needs.** The data of §3, and the released files of §1.

```bash
# B2a: train the clean AR translator (baseline_eval.yaml decoder: ar).
python train.py --stage train-slt --language "$LANG" --slt-config configs/baseline_train.yaml
# B2a: train the clean DLM translator. First set `decoder: dlm` in configs/baseline_eval.yaml.
# Each decoder writes to its own folder, so the two runs never share a model.pt or a latest.pt.
python train.py --stage train-slt --language "$LANG" --slt-config configs/baseline_train.yaml
```

Set `decoder: ar` again after the DLM run. While `baseline_eval.yaml` holds `decoder: dlm`, `--method baseline` with no `--checkpoint` reads the clean DLM translator.

**B2b.** In `configs/data.yaml`, find the entry of the corpus. Replace the active `ar` and `dlm` lines of its `pretrained_slt` (they name the released file) with the clean translators. The comment lines under them hold the new values. For example, for `asf`:

```yaml
languages:
    asf:
        pretrained_slt:
            ar: checkpoints/baseline_train_ar/asf/model.pt
            dlm: checkpoints/baseline_train_dlm/asf/model.pt
```

B2a always starts from a released file, so a later B2a run needs no change of these lines. `train-slt` refuses an arm whose entry names a released file, or the clean translator of another corpus or of the other decoder.

**Writes.** `checkpoints/baseline_train_ar/$LANG/` and `checkpoints/baseline_train_dlm/$LANG/` (`model.pt`, `best.json`, `latest.pt`, `history.csv`).

**Check before you continue.**

- `val_translation_bleu4` in the `best.json` of each clean translator is the dev BLEU-4 of the selected epoch. It must be well above the dev BLEU-4 of the released file with no fine-tuning (spec §10.3). This command measures the released file on the same dev crops; read `text_metrics.translation_bleu4` of the one cell in its output file:

```bash
python analyze.py --stage boundary-sweep --method baseline --checkpoint checkpoints/openasl_pose_only_slt.pth --language "$LANG" --split dev \
    --severity-grid-head=0 --severity-grid-tail=0 --output outputs/analysis/boundary_sweep_released_${LANG}_dev.json
```

- `val_translation_len_ratio` stays near 1. A value far from 1 shows outputs that are too short or too long.
- The `val_translation_loss` of the clean DLM translator is the OPUT loss. It is noisy, and it does not compare with the cross-entropy of the clean AR translator. Only `val_translation_bleu4` selects the epoch.
- Run no separate clean test: the clean number that you report is row 1 of the result table (§7.3). `report.py results` also gives the pooled corpus BLEU-4 of row 1 (`located_corpus_bleu4`), as a paper on pre-trimmed clips computes it. Do not compare it with a published number: the references are our caption units, and BOBSL and Auslan-Daily use re-splits.

## 6. Stage B: the arms (B3)

**What it does.** It trains segmentation and translation together under the gate (stage 2). Train the AR arm first, then the DLM arm, the main system. Both arms go through every row of §7 (spec §9.3). For each decoder, also train a no-text arm (below).

**Runs on.** A CUDA GPU with much memory.

**Needs.** S1 (A2) and the B2b entry of the arm's decoder. Train the arms and run every evaluation under one `configs/inference.yaml`. Design: [`docs/membership_gate.md`](docs/membership_gate.md) "Current design", spec §5 and §6.

```bash
# The AR arm.
python train.py --stage train-slt --language "$LANG" --slt-config configs/ar.yaml
# The DLM arm (the main system).
python train.py --stage train-slt --language "$LANG" --slt-config configs/dlm.yaml
```

**Writes.** `checkpoints/ar/$LANG/` and `checkpoints/dlm/$LANG/`.

`train_inputs` in `configs/dlm.yaml` lists the input types of the train rows, in equal shares (by default the 3 types of S1). The dev rows are windows. The run prints the counts of the first epoch.

**The no-text arms.** Train one no-text arm for each decoder and corpus: the same run, with the text path into the segmentation cut. It is the control of "text improves the segmentation" (claim 1). The clean translator on its committed spans is the fair cascade of claim 2 (§8.2).

1. In `configs/ar.yaml` (or `configs/dlm.yaml`), set `membership_gate.detach_omega: true`, `bio.adapter: false`, `checkpoint.dir: checkpoints/ar_notext/${language}` (or `dlm_notext`) and a new `wandb.name`.
2. Train with the `train-slt` line above. Keep the edit until the run is complete: `--resume` refuses a changed config.
3. Undo the edit before any evaluation. While the edit is in place, `--method ar` with no `--checkpoint` loads the no-text arm, and its files take the names of the main arm. Evaluate the no-text arm with `--checkpoint "checkpoints/ar_notext/${LANG}/model.pt"`; its files carry the tokens `detach` and `noadapter`.

If GPU time is short, train the DLM no-text arm first: its spans can also give the cascade of the AR arm (say so where you report it).

**Check before you continue.**

- Before the first full run, watch the GPU memory with `nvidia-smi` during the first steps. If the run is near the memory limit, lower `batch_size`, and raise `grad_accum_steps` by the same factor.
- The dev losses `val_bio_loss` and `val_translation_loss` fall over epochs 1 to 3.
- `text_accepted_rate` (the share of supervised rows that read the model's own span) must not fall on one-unit targets. A correct split of a joined target also lowers it, so read it beside `text_pair_rate` (the share of joined targets).
- `text_grad_rate` (the share of rows whose text trains the segmentation) is above 0. It is 0 only in the no-text arm.
- Each dev evaluation logs the monitor `val_offline_score` = `val_offline_f1` × `val_offline_bleu4`, and `val_offline_events_per_gold`. Report both factors, not only their product.
- `model.pt` stamps the selected epoch as `best_epoch`. `report.py results --history` shows it (§7.5).
- If a dev evaluation runs out of memory, lower `checkpoint.dev_batch_size`. No score depends on it.
- `checkpoint.monitor: val_online_score` selects on the streaming FSM instead. It is the deployed path, but it is much slower.

**Evaluate the last epoch.** `model.pt` holds the best epoch. To evaluate the last epoch, copy its weights out of `latest.pt` with the command below, because eval cannot load that file. Then give `--checkpoint "checkpoints/ar/${LANG}/last.pt"` to the commands of §7 (for the DLM arm, write `dlm` in place of `ar`). Its row files take the names of the files of `model.pt`, so move one set aside before you run the other.

```bash
python -c "import torch; s = torch.load('checkpoints/ar/$LANG/latest.pt', map_location='cpu', weights_only=False); torch.save({'model': s['model'], 'meta': s['meta']}, 'checkpoints/ar/$LANG/last.pt')"
```

## 7. Evaluate

Run this section only when training and checkpoint selection are final. All commands read test (`--split test --allow-test`). Run the subsections in order, on a GPU or on a Mac.

### 7.1 Segmenter scores and error types (B4)

**What it does.** It scores each segmenter alone, with no translator, under 2 decodes: `plain` (the most likely label of each frame) and `duration` (the duration decode). It writes the span files that the cascade rows read. The error analysis then shows how a real segmenter fails, with the error types of §8.3. This analysis motivates the window modes of stage 2, but no training stage reads it.

**Needs.** The 2 segmenters of §4.

Score the four cells (two segmenters, two decodes):

```bash
for S in moryossef s1; do
    for D in plain duration; do
        python eval.py --segmenter-eval --segmenter-arch "$S" --segmenter-decode "$D" --language "$LANG" --split test --allow-test \
            --output outputs/segmenter/segmenter_eval_${S}_${D}_${LANG}_test.json
    done
done
```

Optional: the untrained controls of the Moryossef architecture, the released weights with no training on this data against random weights of the same model. Read them as a pair, never as a baseline. Give no `--output`, so that the file name keeps the init token.

```bash
python eval.py --segmenter-eval --segmenter-arch moryossef --segmenter-init released --language "$LANG" --split test --allow-test
for SEED in 0 1 2; do
    python eval.py --segmenter-eval --segmenter-arch moryossef --segmenter-init random --seed "$SEED" --language "$LANG" --split test --allow-test
done
```

Write the span files, one for each segmenter:

```bash
python analyze.py --stage segmenter-infer --segmenter-arch moryossef --segmenter-decode duration --language "$LANG" --split test --allow-test \
    --output outputs/segmenter/segmenter_predictions_moryossef_duration_${LANG}_test.json
python analyze.py --stage segmenter-infer --segmenter-arch s1 --segmenter-decode duration --language "$LANG" --split test --allow-test \
    --output outputs/segmenter/segmenter_predictions_s1_duration_${LANG}_test.json
```

Run the error analysis on the span files of both segmenters:

```bash
for S in moryossef s1; do
    python analyze.py --stage segmenter-errors --segmenter-arch "$S" --language "$LANG" --split test --allow-test \
        --predictions outputs/segmenter/segmenter_predictions_${S}_duration_${LANG}_test.json
done
```

**Writes.**

- `outputs/segmenter/segmenter_eval_<arch>_<decode>_<lang>_test.json`: the four cells. The controls write `segmenter_eval_moryossef_released_<lang>_test.json` and `segmenter_eval_moryossef_random<seed>_<lang>_test.json`.
- The span files `outputs/segmenter/segmenter_predictions_{moryossef,s1}_duration_<lang>_test.json`.
- `outputs/analysis/error_types_*` (the share and the count of each error type, per τ and their mean), `jitter_*` (the edge offsets and the cut positions) and `mode_ratios_*` (the window mode that each error type matches).

**Check before you continue.**

- Each `--segmenter-eval` run prints `MAIN edge chunk F1 avg`.
- Quote the `edge_chunk` block (the main score) and the `tiou` block of the JSON. For the same checkpoint and decode, they equal the localization scores of rows 3 and 5. The `metrics` block (the protocol of Moryossef et al.) is only for comparison with other papers.

### 7.2 Boundary sweep (B5, optional)

**What it does.** It moves the start and the end of each gold unit on a grid (fractions of the unit duration) and translates each moved crop. It is not part of the paper's evidence: the method never translates a cut unit (P1), so the rows measure the real segmenter errors instead (§8.2). Each run translates every unit once for each cell of the grid.

**Needs.** The clean translators after the B2b edit, and the arms.

`boundary_sweep.severity_grid_rel` in `configs/eval.yaml` sets the grid of the head axis and of the tail axis. The sweep is the full product of the two axes.

```bash
for M in baseline ar dlm; do
    python analyze.py --stage boundary-sweep --method "$M" --language "$LANG" --split test --allow-test \
        --output outputs/analysis/boundary_sweep_${M}_${LANG}.json
done
```

The tail-context sweep in seconds (spec §9.1) adds context after the unit only, and it writes its own files. Write a grid list that starts with a minus sign with `=`, for example `--severity-grid-head=-0.3,0`.

```bash
for M in baseline ar dlm; do
    python analyze.py --stage boundary-sweep --method "$M" --language "$LANG" --split test --allow-test --severity-mode absolute \
        --severity-grid-head=0 --severity-grid-tail=0,0.5,1,1.5,2,3,4,6 --output outputs/analysis/boundary_sweep_${M}_tail_${LANG}.json
done
```

For the clean DLM translator, give its file with `--checkpoint`. The sweep refuses an `--output` without `dlm` in its name for it, so it cannot write over the sweep of the clean AR translator.

```bash
python analyze.py --stage boundary-sweep --method baseline --checkpoint checkpoints/baseline_train_dlm/${LANG}/model.pt --language "$LANG" \
    --split test --allow-test --output outputs/analysis/boundary_sweep_baseline-dlm_${LANG}.json
```

**Writes.** The `--output` file. `report.py progress` reads `outputs/analysis/boundary_sweep_baseline_<lang>.json` (the clean AR translator) when it exists.

**Check before you continue.**

- A positive head value starts the window late. A negative tail value ends the window early.
- For the arms, read `gate_skip_rate` and `text_metrics_decoded_only` beside `text_metrics`. A window with a late start has no B, so the FSM would skip it; `gate_skip_rate` counts these windows. Never plot the decoded-only scores alone: they read a smaller and easier subset.
- Read BLEU against the full caption only in the cells with head ≤ 0 and tail ≥ 0. Only these windows hold all the signing of the unit.
- Compare each curve with its own (0,0) cell, and report that cell separately.

### 7.3 Offline rows: rows 1-7 (B6, block A)

**What it does.** The rows caption whole test videos end to end: they find the units, then translate them. In the offline rows, the whole video is available.

**Needs.** The span files of §7.1, the clean translators after the B2b edit, and the arms.

After the B2b edit, every `--method baseline` command reads the clean AR translator from `configs/data.yaml`, so it needs no `--checkpoint`. The DLM arm is compared only with rows of the clean DLM translator (one decoder family, §8.2). These commands give `--checkpoint "$DLM_FLOOR"`, and their files carry `clean-dlm` in place of `clean`. Set the paths once in each shell:

```bash
GOLD=outputs/gold/gold_${LANG}_test.json
MORY=outputs/segmenter/segmenter_predictions_moryossef_duration_${LANG}_test.json
S1SPANS=outputs/segmenter/segmenter_predictions_s1_duration_${LANG}_test.json
ROW6_AR=outputs/offline/offline_joint-ar_duration_ar_${LANG}_test.json
ROW6_DLM=outputs/offline/offline_joint-dlm_duration_dlm_${LANG}_test.json
DLM_FLOOR=checkpoints/baseline_train_dlm/${LANG}/model.pt
```

**Rows 1 and 2: gold spans.** The first command writes the gold span file. Row 1 reads it with the clean translator, and row 2 with each arm.

```bash
python eval.py --emit-gold-segments "$GOLD" --language "$LANG" --split test
# Row 1.
python eval.py --segments "$GOLD" --method baseline --language "$LANG" --split test --allow-test
# Row 2, AR arm and DLM arm.
python eval.py --segments "$GOLD" --method ar --language "$LANG" --split test --allow-test
python eval.py --segments "$GOLD" --method dlm --language "$LANG" --split test --allow-test
```

**Rows 3 and 4: the Moryossef cascades.** They read the Moryossef span file of §7.1: row 3 with the clean translator, row 4 with each arm.

```bash
# Row 3.
python eval.py --segments "$MORY" --method baseline --language "$LANG" --split test --allow-test
# Row 4, AR arm and DLM arm.
python eval.py --segments "$MORY" --method ar --language "$LANG" --split test --allow-test
python eval.py --segments "$MORY" --method dlm --language "$LANG" --split test --allow-test
```

**Row 5: the S1 cascade.** It reads the S1 span file of §7.1.

```bash
python eval.py --segments "$S1SPANS" --method baseline --language "$LANG" --split test --allow-test
```

**Row 6: the arm cuts the video itself.** Offline there is no 40 s buffer: each complete span of at least Λ_min is one event, at any length. Only the online rows have the cap cut.

```bash
python eval.py --offline --method ar --language "$LANG" --split test --allow-test
python eval.py --offline --method dlm --language "$LANG" --split test --allow-test
```

**Row 7: the offline same-span control.** The clean translator of each arm's decoder reads the spans of that arm's row 6.

```bash
python eval.py --segments "$ROW6_AR" --method baseline --language "$LANG" --split test --allow-test
python eval.py --segments "$ROW6_DLM" --method baseline --checkpoint "$DLM_FLOOR" --language "$LANG" --split test --allow-test
```

**Rows 1, 3 and 5 for the DLM arm.** The same spans, read by the clean DLM translator:

```bash
for SPANS in "$GOLD" "$MORY" "$S1SPANS"; do
    python eval.py --segments "$SPANS" --method baseline --checkpoint "$DLM_FLOOR" --language "$LANG" --split test --allow-test
done
```

**Writes.** `outputs/gold/gold_<lang>_test.json`. For each row, the events file `outputs/offline/offline_<spans>_<decode>_<translator>_<lang>_test.json` and its `_scores.json` beside it (§11.3).

**Check before you continue.** Each row prints its scores, and the provenance of each row names the translator checkpoint that it loaded. To print the scores of a saved events file again, with no model run and no new `_scores.json`:

```bash
python eval.py --predictions "$ROW6_DLM" --language "$LANG" --split test --allow-test
```

### 7.4 Online rows: rows 8-11 (B6, block B)

**What it does.** It makes the same comparison when the FSM must decide from the past alone. These rows take much longer than the offline rows.

**Needs.** Arms trained under the same `configs/inference.yaml` that these runs read, and the paths of §7.3 in the shell. As in §7.3, the rows of the DLM arm read the clean DLM translator (`--checkpoint "$DLM_FLOOR"`).

**Rows 8 and 9: the online cascades.** The Moryossef segmenter (row 8) or S1 (row 9) runs under the same FSM, and the clean translator reads its commits.

```bash
# Row 8.
python eval.py --online --segmenter-arch moryossef --method baseline --language "$LANG" --split test --allow-test
# Row 9.
python eval.py --online --segmenter-arch s1 --method baseline --language "$LANG" --split test --allow-test
# Rows 8 and 9 for the DLM arm.
python eval.py --online --segmenter-arch moryossef --method baseline --checkpoint "$DLM_FLOOR" --language "$LANG" --split test --allow-test
python eval.py --online --segmenter-arch s1 --method baseline --checkpoint "$DLM_FLOOR" --language "$LANG" --split test --allow-test
```

**Row 10: the deployed system.** The arm runs under the FSM. `--stability` also records the display trace of §8.4; it does not change the events.

```bash
python eval.py --online --stability --method ar --language "$LANG" --split test --allow-test
python eval.py --online --stability --method dlm --language "$LANG" --split test --allow-test
```

**Row 11: the online same-span control.** The clean translator reads the committed spans of each arm's row 10.

```bash
ROW10_AR=outputs/online/online_joint-ar_duration_ar_${LANG}_test.json
ROW10_DLM=outputs/online/online_joint-dlm_duration_dlm_${LANG}_test.json
python eval.py --segments "$ROW10_AR" --method baseline --language "$LANG" --split test --allow-test
python eval.py --segments "$ROW10_DLM" --method baseline --checkpoint "$DLM_FLOOR" --language "$LANG" --split test --allow-test
```

**Rows 10n and 11n: the no-text arms.** The no-text arm (§6) runs as row 10, and the clean translator reads its committed spans as row 11. Row 11n is the fair cascade of claim 2: its segmenter had the same target training as the arm, without text.

```bash
NOTEXT_DLM=checkpoints/dlm_notext/${LANG}/model.pt
python eval.py --online --method dlm --checkpoint "$NOTEXT_DLM" --language "$LANG" --split test --allow-test
python eval.py --segments outputs/online/online_joint-dlm-detach-noadapter_duration_dlm-detach-noadapter_${LANG}_test.json \
    --method baseline --checkpoint "$DLM_FLOOR" --language "$LANG" --split test --allow-test
python eval.py --online --no-translate --method dlm --checkpoint "$NOTEXT_DLM" --language "$LANG" --split test --allow-test
```

Take the exact file name of row 10n from its printed output (the name tokens of the no-text arm). Do the same with `ar` for the AR arm, with no `--checkpoint` in its row 11n (the clean AR translator).

**Optional: segmentation with no text.** An FSM commit needs only a stable terminator. So these runs give the segmentation of rows 8-10 with no translation, and they compare the segmenters under one FSM. Their translator token is `none`, so they do not write over the rows with text. Do not give `--checkpoint` to a cascade run with `--no-translate`: there it names the segmenter checkpoint, not the translator.

```bash
# Rows 8 and 9, no text.
python eval.py --online --no-translate --segmenter-arch moryossef --method baseline --language "$LANG" --split test --allow-test
python eval.py --online --no-translate --segmenter-arch s1 --method baseline --language "$LANG" --split test --allow-test
# Row 10, no text.
python eval.py --online --no-translate --method ar --language "$LANG" --split test --allow-test
python eval.py --online --no-translate --method dlm --language "$LANG" --split test --allow-test
```

**Writes.** For each row, `outputs/online/online_<spans>_<decode>_<translator>_<lang>_test.json` and its `_scores.json`. Row 10 with `--stability` also writes `outputs/stability/stability_<translator>_<lang>_test.json` (the scores of the reveal policies) and `outputs/stability/provisional_<translator>_<lang>_test.json` (every recorded decode as text).

**Check before you continue.** Each row prints its scores and its `gate_stats` counters. §8.5 lists what to report from them.

### 7.5 Reports (B7)

**What it does.** `report.py results` scores again the systems that you give it, under one protocol. It gives the main and tIoU scores, the error types per caption and per unit (§8.3), the differences against a reference system, case studies and figures. Any events file of a row can be a system. `report.py progress` makes one report over the finished rows of blocks A and B for the arms of the live config. Report the no-text rows, the zero-shot rows, the ablations and the dev rows with `report.py results --events`.

**Needs.** The events files of §7.3 and §7.4, and the stability files of row 10.

```bash
python report.py results --language "$LANG" --split test --reference online \
    --events "true cuts"=outputs/offline/offline_gold_none_clean_${LANG}_test.json \
    --events online=outputs/online/online_joint-dlm_duration_dlm_${LANG}_test.json \
    --events offline=outputs/offline/offline_joint-dlm_duration_dlm_${LANG}_test.json \
    --events cascade=outputs/offline/offline_moryossef_duration_clean_${LANG}_test.json \
    --stability-file outputs/stability/stability_dlm_${LANG}_test.json \
    --stability-file outputs/stability/stability_ar_${LANG}_test.json
python report.py progress --language "$LANG"
```

Other inputs of `report.py results`: `--boundary-sweep-file` (§7.2), `--history LABEL=PATH` (the `history.csv` of an arm: the training curves and the selected epoch) and `--segmenter-eval LABEL=PATH` (a segmenter score file). One report reads the systems of one source corpus.

**Writes.** `outputs/report/results_<lang>_<split>/` (`results_<S>-to-<T>_<split>/` for zero-shot systems), with `results.md`, `fig/*.png` and `tables/*.csv`, and `outputs/report/progress/`. The main tables:

| Table | What it holds |
| --- | --- |
| `scores.csv` | The scores of each system, with the pooled corpus BLEU-4 `located_corpus_bleu4` (§5). |
| `error_types.csv` | Per system and τ, the share of the units and of the captions of each error type. |
| `outcomes_<system>.csv` | One line per unit and τ: its error type, and the caption that the edge match pairs with it. |
| `differences.csv` | The difference of each system against the reference system: values only, no confidence interval. |
| `boundary.csv` | The start and end errors of the captions that hold exactly one unit. |
| `edge_chunk_gaps.csv` | The main localization at each chunk gap of `scoring.chunk_gap_rows_s` in `configs/eval.yaml`. |
| `chunk_classes.csv` | The units of each kind of chunk, for example the chunks longer than the cap. |
| `caption_shape.csv` | The caption durations, and the units that each caption holds. |
| `cue_breaks.csv` | How often a caption edge inside a unit falls on a junction of 2 display cues, against chance. |
| `train_text_videos.csv`, `edge_chunk_by_train_text.csv` | The eval videos with a unit whose words are those of a train unit, and every main score with and without them. |
| `cap_cut.csv` | The cap-cut captions apart, and the main scores without them. |

**Check before you continue.**

- A system whose model, translator start or S1 trained on other train labels gets one WARNING line. The report still scores it.
- The code computes no confidence interval or p-value. Run resampling statistics outside the repository.
- For debug views of single videos, use `python report.py poses` and `python report.py predict` (see their `--help`).

### 7.6 Zero-shot rows on the benchmarks (B6, block C)

**What it does.** It scores the models of a YouTube-SL-25 language on the benchmark of the same sign language. It runs the `asf` models on `auslan_daily`, the `bfi` models on `bobsl_manual`, and the `ase` models on `how2sign`. It trains nothing, and no label of the benchmark's own annotation goes into the training or the checkpoint choice of these models. `--source-language S` takes every model and the duration prior from corpus S, and the data, the gold and the scoring from `--language T`. File names carry `S-to-T`, so a zero-shot row never writes over a row of the benchmark's own models.

On `bobsl_manual`, the claim is "no manual labels", not "unseen domain". S1 and the Moryossef segmenter pretrained on other BOBSL episodes (the official train split, with the automatic labels), and no dev or test episode of `bobsl_manual` is among them.

**Needs.** The models of the source corpus (A1, A2, B2, B3 and the no-text arms) and the data of §3.2.

```bash
SRC=asf; TGT=auslan_daily     # or SRC=bfi TGT=bobsl_manual, SRC=ase TGT=how2sign
python eval.py --emit-gold-segments outputs/gold/gold_${TGT}_test.json --language "$TGT" --split test
python eval.py --segments outputs/gold/gold_${TGT}_test.json --method baseline --language "$TGT" --source-language "$SRC" --split test --allow-test
python analyze.py --stage segmenter-infer --segmenter-arch moryossef --language "$TGT" --source-language "$SRC" --split test --allow-test
python eval.py --online --segmenter-arch moryossef --method baseline --language "$TGT" --source-language "$SRC" --split test --allow-test
python eval.py --online --segmenter-arch s1 --method baseline --language "$TGT" --source-language "$SRC" --split test --allow-test
python eval.py --offline --method dlm --language "$TGT" --source-language "$SRC" --split test --allow-test
python eval.py --online --stability --method dlm --language "$TGT" --source-language "$SRC" --split test --allow-test
```

Every other row of §7.3 and §7.4 takes `--source-language` the same way. With `--segments`, give the span file of the zero-shot segmenter run: a `--segments` row refuses a span file of another source.

**Writes.** The row files of §7.3 and §7.4, with `<S>-to-<T>` in the language field (§11.3).

**Lowercase scoring.** `auslan_daily` sets `lowercase_references: true`. Its captions are all lowercase, so every text score of its rows compares lowercase text, for every system alike. The training targets do not change.

### 7.7 Chance rows (appendix)

**What it does.** It makes a model-free span file per corpus: spans of the median length of the corpus's train units, from each stream start to its end. These rows show how much edge chunk F1 a corpus gives with no model, so compare systems only inside one corpus.

```bash
python analyze.py --stage chance-spans --language "$LANG" --split test --allow-test
python eval.py --predictions outputs/segmenter/segmenter_predictions_chance_${LANG}_test.json --language "$LANG" --split test --allow-test
```

**Writes.** The chance span file in `outputs/segmenter/`. `--predictions` prints the main scores and writes no file. `report.py results --events chance=<file>` puts the row in a table.

## 8. Read the results

### 8.1 The result table

Block A decides offline, with the whole video available. Block B decides online, from the past alone. Block C repeats the rows on a benchmark with the models of the same sign language from YouTube-SL-25 (§7.6). Inside each block, the rows go from gold spans to the cascades, then to the arm and its same-span control. Compare rows across blocks A and B only through row 10 against row 6.

| # | Spans from | Translator | Command | Artifact |
| --- | --- | --- | --- | --- |
| | **A. Offline: the whole video is available** | | | |
| 1 | gold spans (the annotation) | clean translator | `--segments "$GOLD"` `--method baseline` | `offline_gold_none_clean_…` |
| 2 | gold spans (the annotation) | AR arm, DLM arm | `--segments "$GOLD"` `--method ar/dlm` | `offline_gold_none_{ar,dlm}_…` |
| 3 | Moryossef segmenter, duration decode | clean translator | `--segments "$MORY"` `--method baseline` | `offline_moryossef_duration_clean_…` |
| 4 | Moryossef segmenter, duration decode | AR arm, DLM arm | `--segments "$MORY"` `--method ar/dlm` | `offline_moryossef_duration_{ar,dlm}_…` |
| 5 | S1, duration decode | clean translator | `--segments "$S1SPANS"` `--method baseline` | `offline_s1_duration_clean_…` |
| 6 | the BIO head of the arm, on the whole stream; Ω = a hard crop of each span | AR arm, DLM arm | `--offline --method ar/dlm` | `offline_joint-ar_duration_ar_…` / `…joint-dlm_duration_dlm_…` |
| 7 | the spans of row 6 | clean translator | `--segments "$ROW6_AR"` (or `"$ROW6_DLM"`) `--method baseline` | `offline_joint-{ar,dlm}_duration_clean_…` |
| | **B. Online: the FSM decides from the past alone** | | | |
| 8 | Moryossef segmenter under the FSM | clean translator | `--online --segmenter-arch moryossef --method baseline` | `online_moryossef_duration_clean_…` |
| 9 | S1 under the FSM | clean translator | `--online --segmenter-arch s1 --method baseline` | `online_s1_duration_clean_…` |
| 10 | the BIO head of the arm under the FSM | AR arm, DLM arm | `--online --stability --method ar/dlm` | `online_joint-ar_duration_ar_…` / `…joint-dlm_duration_dlm_…` |
| 11 | the committed spans of row 10 | clean translator | `--segments "$ROW10_AR"` (or `"$ROW10_DLM"`) `--method baseline` | `online_joint-{ar,dlm}_duration_clean_…` |
| 10n | the BIO head of the no-text arm under the FSM | no-text AR arm, no-text DLM arm | `--online --method ar/dlm --checkpoint "$NOTEXT"` | `online_joint-{ar,dlm}-detach-noadapter_duration_{ar,dlm}-detach-noadapter_…` |
| 11n | the committed spans of row 10n (the fair cascade) | clean translator | `--segments <row 10n file>` `--method baseline` | `online_joint-{ar,dlm}-detach-noadapter_duration_clean_…` |
| | **C. Zero-shot: models of the same sign language, a benchmark's data** | | | |
| 1-11, 10n, 11n | as above, with the source models | as above | each command with `--source-language S` (§7.6) | the name field `S-to-T`, for example `…_asf-to-auslan_daily_test.json` |

A row of the clean DLM translator writes `clean-dlm` in place of `clean`, for example `offline_s1_duration_clean-dlm_…` and `online_joint-dlm_duration_clean-dlm_…`. The `--stability` files of its online cascade are `stability_<segmenter>-dlm_…`.

### 8.2 Which rows to compare

Compare rows of one target corpus and one decoder family. The full reasons are in spec §9.2 "How to read the table".

The rows follow the story of the paper. A real segmenter makes errors (§7.1), and these errors cost a translator that trained on clean crops (1 → 3 and 1 → 5 offline; 8 and 9 online). The window modes give each error state a label rule, so the arm learns from wrong segments, and one model learns both tasks. The arm then runs offline (row 6) and under the FSM (row 10). The window modes are a label rule, not a claim that the segmenter finds better edges. A segmentation gain of the arm must come from the text, and only the no-text arm isolates the text.

- **Joint training against the cascades: 10 against 11n, 8 and 9 online; 6 against 3 and 5 offline.** This is the deployed claim. 11n is the fair cascade: its segmenter had the same target training as the arm, without text. The gain 10 − 11n has 2 parts that add up exactly: 10 − 11n = (10n − 11n) + (10 − 10n). 10n − 11n changes only the translator on the same committed spans, and 10 − 10n is the share of the text path into the segmentation. Report both parts.
- **Text improves the segmentation: the arm against its no-text arm** (10 against 10n with `--no-translate`, and 11 against 11n); also 7 against 5 offline and 11 against 9 online. From S1 to the arm, the data and the training also change, so only the no-text arm shows that the text is the cause.
- **Zero-shot (block C).** The same pairs on a benchmark, with no label of the benchmark's own annotation in any model: does joint minus cascade keep its sign when the domain changes?
- **Span quality with one translator: 1 → 3 → 5 → 7.** Only the span source changes.
- **Translator at fixed spans: 2 − 1, 4 − 3 and 7 − 6 offline; 10 − 11 online.** These pairs measure the whole training package of the arm. (4 − 3) against (2 − 1) shows if the arm loses less than the clean translator on the real errors of one segmenter.
- **Access: 10 − 6.** The same arm, online against offline.
- **Online split: 9 → 11 → 10.** 9 → 11 keeps the clean translator and changes the segmentation. 11 → 10 keeps the committed spans and changes the translator.
- **Which errors change.** Compare the error types (§8.3) of the 2 rows of a pair to see which errors a change removes.
- Compare systems only inside one corpus: the chance rows (§7.7) give each corpus its own edge F1 with no model.

### 8.3 What the scores mean

Each row's `_scores.json` holds `provenance`, `main` and `tiou`. The full definitions are in spec §9.2.

| Key | What it measures |
| --- | --- |
| `main.edge_chunk.f1_avg` | The main localization score: the edge chunk F1, averaged over the τ values of `scoring.edge_taus_s` in `configs/eval.yaml`. |
| `main.edge_chunk.dvc_avg` | The main text scores (BLEU-4, BLEURT, ROUGE-L, METEOR, CIDEr-D) on the edge chunk pairs, with the DVC aggregation of densevid_eval. A caption with no match is scored against a random garbage text. |
| `main.edge_chunk.soda_avg` | A SODA-style text F1 on the same pairs. It charges captions with no match and missed units in one number. It has no CIDEr-D column. |
| `main.uncaptioned` | The reliable units that no caption overlaps. |
| `tiou.threshold_average` | The tIoU scores, for comparison with other papers: the sentence tIoU F1 (`segmentation.f1`) and the DVC columns of densevid_eval (`text_metrics.densevid_*`), averaged over `scoring.tiou_thresholds`. |

An example of the edge chunk match: units A [10.0, 14.0] s and B [14.2, 18.0] s have a gap of 0.2 s, so they are one chunk (`scoring.chunk_gap_s`). A caption at [10.3, 17.6] s matches the run A B at τ = 0.5 s, and the scorer compares its text with the joined text "A B". A caption matches only when both of its edges are within τ of the edges of one run. So a caption that runs more than τ past a pause, or that ends more than τ inside a unit, matches nothing, and a cut within τ of a unit edge passes.

Quote the edge chunk precision and recall beside the edge chunk DVC: the DVC charges a caption with no match, but a missed unit adds no pair. Quote `segmentation.recall` beside a tIoU DVC column.

**Error types.** They read the same edge match as the main score. A caption holds a unit when it covers at least half of the unit. An edge cuts a unit when it lies inside the unit, more than τ from both of its edges. Per τ, each caption and each unit takes the first type whose rule fits. With the units of the example above and a third unit C [20.0, 23.0] s after a 2 s pause, at τ = 0.5 s:

| Type | Rule for a caption | Example caption | Rule for a unit |
| --- | --- | --- | --- |
| matched | it matches a run of one unit | [10.2, 13.8] for A | it is that unit |
| joined | it matches a run of two or more units of one chunk (correct) | [10.3, 17.6] for A B | it is in that run |
| cut | an edge cuts a unit | [12.5, 17.9]: its start is 2.5 s inside A | a caption edge cuts it |
| bridged | it holds units of two or more chunks (it crosses a pause) | [14.4, 22.8] holds B and C | a caption holds it and a unit of another chunk |
| offset | it holds units of one chunk, but an edge is more than τ from the run edges, or another caption took the match | [19.0, 23.1]: it starts 1 s early, in the pause (matched at τ = 1 s) | a caption holds it |
| phantom (captions) / missed (units) | it holds no unit and cuts none | [18.4, 19.6], in the pause | no caption holds it or cuts it (every unit of `main.uncaptioned` is missed) |

At each τ, the share of matched and joined captions is the edge chunk precision. The share of matched and joined units is its recall. Both are per video, then the mean over the videos. The other types split the misses. The tIoU scores pair captions in another way. For example, [20.2, 25.5] has tIoU 0.51 with C, so tIoU 0.5 pairs it. But its end is 2.5 s late, so it is offset at every τ.

Each table computes BLEU-4 in its own way. The clean number is row 1 (§5). The exact rules are in spec §9.2 "BLEU rule".

| Column | Where | What it is |
| --- | --- | --- |
| `translation_bleu4` | `val_translation_bleu4` in training; the `text_metrics` of each boundary-sweep cell | Corpus BLEU-4 over all captions of the set |
| `located_corpus_bleu4` | `report.py results` `tables/scores.csv` | Corpus BLEU-4 over the tIoU-matched pairs; on row 1 every unit matches, so it is the pooled corpus BLEU-4 of all units |
| `main.edge_chunk.dvc_avg.bleu4` | `_scores.json` | Corpus BLEU-4 on the edge chunk pairs of each video, then the mean over videos and τ |
| `main.edge_chunk.soda_avg.bleu4` | `_scores.json` | Sentence BLEU-4 of each edge chunk pair, inside the SODA-style F1 |
| `text_metrics.densevid_bleu4` | `_scores.json`, under `tiou` | Corpus BLEU-4 on the tIoU pairs of each video, then the mean over videos |

### 8.4 Display stability

Row 10 with `--stability` writes the stability files (§7.4); no other run is necessary. The replay applies reveal policies (for example `agreement_nK` and `frozen_agreement_nK`) to the decodes of row 10, also the decodes of the unit that is still open. The deployed output stays commit-only. The policies and their scores are in spec §9.2 "Display stability", and `report.py results --stability-file` puts them in the report (§7.5).

### 8.5 What to report beside the scores

- For the online rows: the share of forced commits, `main.uncaptioned`, and the `gate_stats` counters `open_span_kept`, `open_span_cut` and `open_span_dropped_at_cap`. The events are `committed` + `open_span_cut`, because a cap cut is an event that `committed` does not count. Show the cap-cut events apart (`tables/cap_cut.csv`, and the `cap_cut` cases in `tables/cases.json`). The counters count cap passes, so one unit can count more than once.
- For the online rows: the compute time per stride, in `provenance.compute`.
- For the monitor of the arms: both factors, `val_offline_f1` and `val_offline_bleu4`.
- From the data audit of §3: the frames that the loader cuts out of each video, the units quarantined for no visible hand (`units_quarantined_handless`) or for a second signer (`units_quarantined_switch_doubt`), and the train copies of dev and test videos (`deduplicated_train`).
- For the untrained controls of §7.1: the training data, initialization, normalization and trained context beside each model (spec §11 item 6).

## 9. Optional studies

### 9.1 Ablation arms

**What it does.** It trains each ablation as its own arm, never as a switch at evaluation time. The clean translator already is the ablation with no misalignment training.

**Needs.** The same as B3 (§6).

| Ablation | Keys to set | `checkpoint.dir` | Name token |
| --- | --- | --- | --- |
| One pose encoder (the one-stream arm; spec §11 item 5) | `shared_pose_encoder: true`, `bio.adapter: false` | `checkpoints/ar_one_stream/${language}` | `onestream` |
| Seed repeat | `seed: <n>` | `checkpoints/ar_seed<n>/${language}` | `seed<n>` |

1. In `configs/ar.yaml` (or `configs/dlm.yaml`), set the keys of the table, a new `checkpoint.dir` and a new `wandb.name`.
2. Train with the `train-slt` line of §6. Keep the edit until the run is complete: `--resume` refuses a run whose config changed.
3. Undo the edit before any evaluation. While the config holds the edit, `--method ar` with no `--checkpoint` loads the ablation arm, and its files can take the names of the main arm and write over them.
4. Evaluate the ablation with the commands of §7. Give `--checkpoint` at the `model.pt` of its folder in each command with `--method ar` or `--method dlm`.

A boundary-sweep `--output` path (§7.2) must hold each name token of the arm. For example, for the one-stream arm:

```bash
python analyze.py --stage boundary-sweep --method ar --checkpoint "checkpoints/ar_one_stream/${LANG}/model.pt" --language "$LANG" --split test \
    --allow-test --output outputs/analysis/boundary_sweep_ar_onestream_${LANG}.json
```

**Writes.** The checkpoint folder of each ablation, and row files that carry its name token (§11.3).

**Check before you continue.**

- Read the one-stream arm against the main arm and against the no-text arm.
- Spec §11 item 4 gives the rule that decides whether to keep the end-side text term and the adapter (`bio.adapter`, which gives translator features to the BIO head). The rule needs a confidence interval of the dev difference, which the code does not compute: compute it outside the repository.
- A seed repeat shows how much the score of one arm changes from one run to the next. Compare two arms only when their difference is larger than that change.

### 9.2 S1 continued-training study (each target corpus)

**What it does.** It tests the value of pooled pretraining, with one S1 architecture and the released pose initialization in all three conditions. Condition 2 is "pooled evaluation without target fine-tuning", not zero-shot, because the target language is in the pool.

| Condition | Training schedule | Purpose |
| --- | --- | --- |
| 1. Target only | S1 training on the target corpus | Reference for the value of multilingual training |
| 2. Pooled, no target fine-tune | Train on the pool of §4; evaluate that checkpoint on each target | Multilingual model before specialization |
| 3. Pooled → target fine-tune | Start from the exact checkpoint of condition 2; continue on each target separately | Effect of target specialization after multilingual training |

**Runs on.** A CUDA GPU.

**Needs.** The data of §3 and `checkpoints/openasl_pose_only_slt.pth`. It does not need Stage B.

Every step edits `configs/bio_pretrain.yaml`. In each state of the file, `checkpoint.dir` names a new folder, so that A2 stays unchanged. Choose the epoch budget of each phase before you run any condition, and write it down. The values below are examples, not tuned values.

```bash
TARGET=asf               # asf | bfi | how2sign | auslan_daily | bobsl_manual
POOLED_EPOCHS=50         # example value
TARGET_EPOCHS=20         # example value
TARGET_ONLY_EPOCHS=20    # example value
POOLED=checkpoints/bio_s1_pooled/multi_ase-asf-bfi-bobsl_auto/model.pt
```

1. Condition 2, the pooled phase. A2 already is a pooled S1. If A2 trained with the budget of the study, set `POOLED=checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto/model.pt` and go to step 2. Otherwise, set the pooled state below and train condition 2 into its own folder. Do not give `--language`: a pooled run refuses it.

```yaml
checkpoint:
    dir: checkpoints/bio_s1_pooled/${corpus}
    from_pretrained: checkpoints/openasl_pose_only_slt.pth
pretrain:
    languages: [ase, asf, bfi, bobsl_auto]
```

```bash
python train.py --stage train-bio --epochs "$POOLED_EPOCHS"
```

2. Evaluate condition 2 on each target while the file holds the pooled state. Every S1 loader refuses a checkpoint whose pool differs from the config. This checkpoint is condition 2 and the start of condition 3.

```bash
python eval.py --segmenter-eval --segmenter-arch s1 --language "$TARGET" --checkpoint "$POOLED" --split test --allow-test \
    --output outputs/segmenter/s1_study_case2_${TARGET}_test.json
```

3. Condition 1, target only. Set the target-only state, and train:

```yaml
checkpoint:
    dir: checkpoints/bio_s1_target/${corpus}
    from_pretrained: checkpoints/openasl_pose_only_slt.pth
pretrain:
    languages: null
```

```bash
python train.py --stage train-bio --language "$TARGET" --epochs "$TARGET_ONLY_EPOCHS"
```

4. Condition 3, pooled then target. Set the fine-tune state, with `from_pretrained` at the checkpoint of condition 2 (the value of `POOLED`), and train. Do not use `--resume`: condition 3 starts a new optimizer and schedule, and `--resume` cannot change the pool.

```yaml
checkpoint:
    dir: checkpoints/bio_s1_finetune/${corpus}
    from_pretrained: checkpoints/bio_s1_pooled/multi_ase-asf-bfi-bobsl_auto/model.pt
pretrain:
    languages: null
```

```bash
python train.py --stage train-bio --language "$TARGET" --epochs "$TARGET_EPOCHS"
```

5. Evaluate conditions 1 and 3 on the same target test data, while the file holds a target state (`pretrain.languages: null`):

```bash
python eval.py --segmenter-eval --segmenter-arch s1 --language "$TARGET" --checkpoint "checkpoints/bio_s1_target/${TARGET}/model.pt" \
    --split test --allow-test --output outputs/segmenter/s1_study_case1_${TARGET}_test.json
python eval.py --segmenter-eval --segmenter-arch s1 --language "$TARGET" --checkpoint "checkpoints/bio_s1_finetune/${TARGET}/model.pt" \
    --split test --allow-test --output outputs/segmenter/s1_study_case3_${TARGET}_test.json
```

6. Restore `configs/bio_pretrain.yaml` to the state of A2 (`checkpoint.dir: checkpoints/bio_s1/${corpus}`, the released `from_pretrained` and the pool `[ase, asf, bfi, bobsl_auto]`). Do this before you train or evaluate an arm: `checkpoint.bio_head_init: auto` in `configs/dlm.yaml` finds S1 through the `checkpoint.dir` of this file.

**Writes.** The folders `checkpoints/bio_s1_pooled/`, `checkpoints/bio_s1_target/` and `checkpoints/bio_s1_finetune/`, and the score files `s1_study_case*` in `outputs/segmenter/`.

**Rules of the study.**

- Keep the chunk lengths, `pretrain.inputs`, the augmentation, the BIO loss, the attention band, the initialization family and the evaluation fixed in the three conditions.
- Each condition fits its duration prior on the train labels of the target corpus, so the decoder is the same in the three conditions.
- Select checkpoints on dev, and fix the selection rules before test. Each epoch budget is an upper limit, so report the selected epochs.
- Record the optimizer updates, the target-language exposure and the training time: pooled and target epochs hold different numbers of batches.
- Read 3 against 1 as the test of the recipe, and 3 against 2 as the effect of more target training. How to interpret the results: [`docs/membership_gate.md`](docs/membership_gate.md) §4.

### 9.3 S1 input types

**What it does.** It compares choices of `pretrain.inputs` (§4): for example `[chunks]` (the data of the Moryossef segmenter) against the default `[chunks, windows, restart_chunks]`. Dev chunks are plain chunks for every choice, and the checkpoint stamps the list (`pretrain_inputs`).

**Needs.** The same as A2 (§4).

Every choice trains into the same folder, and a run refuses to start while `latest.pt` is in it. So:

1. Train one choice. Then move its folder by hand, for example to `checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto_chunks`.
2. Train the next choice. Train the choice that the arms use last, into the default folder: the arms find S1 there.
3. Score a moved run with `--checkpoint <folder>/model.pt` and its own `--output` (`eval.py --segmenter-eval`, `analyze.py --stage segmenter-infer`). Their files save the list as `segmenter_inputs`.
4. The rows name their files by the segmenter only. Before you run a row of the next choice, move the `*_s1_*` files of `outputs/offline/`, `outputs/online/` and `outputs/stability/` aside. An online S1 row that translates reads S1 only from the default folder, so for that row put each choice in the default folder in turn.

## 10. After a change: what to run again

Find your change in the left column. Run again everything in the right column, in the order of §2.1. Before a new training run, move its `latest.pt` (§2.3).

| Change | Run again |
| --- | --- |
| Arm architecture, loss, sampler, or how the translator reads the span | B3 (both arms, as new runs), then their rows 2, 4, 6, 7, 10 and 11 |
| S1 recipe, pool or input types (`pretrain.*` in `configs/bio_pretrain.yaml`, or a new slowest train frame rate); with windows in `pretrain.inputs`, also the window sampler (`data/sampler.py`, and `mode_ratios`, `mode2_subcase_weights` or `jitter` in `dlm.yaml`) | A2, the S1 segmenter scores and span file, rows 5 and 9, then B3 and every row of the arms |
| Moryossef checkpoint or its whole-video inference | The Moryossef segmenter scores, span file and error analysis (B4), then rows 3, 4 and 8 |
| Clean translator of a decoder | The B2b entry of that decoder, the B3 arm of that decoder, and every `--method baseline` row that reads it: rows 1, 3, 5, 7, 8, 9 and 11 (for the clean DLM translator: the rows that you ran with its `--checkpoint`) |
| Learning rate of a stage | That stage and every stage that loads it |
| `augmentation` (`dlm.yaml`; keep the copy in `moryossef26.yaml` equal) | Every pose model (A1, A2, B2a and the B2b edits, B3), then every row. Stage 2 refuses a clean translator stamped with another block |
| Pose converter (`prepare_yt25.py`), `poses.switch_signing_speed` or `poses.switch_raised_offset` | `python prepare_yt25.py --stage all --overwrite --languages ase asf bfi --workers 12` (it needs the shard tars), `report.py data`, then every training stage and every row |
| `poses.handless_unit_share`, `poses.switch_unit_share` or `poses.switch_doubt_share` | No conversion. Every training stage and every row, because the units change |
| `poses.model_fps`, or the grid or body rule of the pose reader | No conversion. Every pose model (A1, A2, B2a and the B2b edits, B3), then every row |
| `poses.pulldown_*`, or the pose reader's test for repeated frames | No conversion. B2a and B3 of the corpora with repeated frames (`how2sign`, `auslan_daily`), then every row of those corpora. Nothing warns |
| The frame geometry of the pose reader (how it maps x to the frame shape) | No conversion. Every Uni-Sign pose model (A2, B2a and the B2b edits, B3), then every row. The Moryossef segmenter needs no new run |
| The `width`/`height` of a video in `video_meta.csv` | No conversion. Every pose model, A1 too, then every row. Nothing warns |
| A dataset's `subtitles.case_lexicon`, or the train captions of a corpus of its pool | The next train load builds that casing file again (copy it to the other machines). Then every translator (B2a, B3) and every text row of those corpora |
| New or changed data in a benchmark folder (poses, captions, split files or sidecars) | `python prepare_yt25.py --stage dedup --languages <the benchmark keys>`, then every training stage and every row of those corpora |
| `languages.<corpus>.released_slt` | B2a of that corpus and its B2b entries, then its B3 and every row that reads them |
| `languages.<corpus>.lowercase_references` | Every text row and report of that corpus, and its B2a and B3 runs (their dev monitors score at this level) |
| Annotation rules in `configs/data.yaml`, or any other change of the labels | `python prepare_yt25.py --stage dedup` for the corpora whose labels changed, then every training stage and every row |
| A caption file (`convert` or `subs` wrote one), a split file or `subtitles.dedup` | `python prepare_yt25.py --stage dedup` for those corpora, then every training stage and every row (after a change of `subtitles.dedup` alone: only when a `copy_of` value changed) |
| δ (also a label rule, so the labels of every corpus) | Every training stage (A1, A2, B2a, B3), then every row |
| K or stride (they also set the attention band and the cap cut window) | A2 and B3, then every row |
| Buffer cap (also S1's longest chunk and the cap cut window) | A2, B2a and B3, then every row |
| Λ_min (also the loader's shortest clean segment) | Every training stage (A1, A2, B2a, B3), then every row |
| A larger `max_text_tokens` in `dlm.yaml` | Nothing, while every target fits the value that the models trained with (when one does not, `train-slt` stops at its start and names the value that it needs). Run again a row whose captions reach about that value in tokens |
| A smaller `max_text_tokens`, or one that changes the batches | B2a for both decoders and their B2b entries, B3, then every row and every boundary sweep that you keep |
| `spd.tau_dec` | B2a of the clean DLM translator and its B2b entry, the DLM arm, then their rows |
| A checkpoint monitor | Every model that the monitor selects, then every row that reads these models |
| Scoring (`metrics.py`, `scoring` in `configs/eval.yaml`) | Every row, every `--segmenter-eval` file, the error types of §7.1 and every `report.py` output. `chunk_gap_s` and `edge_taus_s` also change both monitors (the row above); `chunk_gap_rows_s` changes `report.py results` only |
| FSM rules (`infer/stream.py`) | Rows 8-11 and the stability files. `val_online_score` can select another epoch |
| Offline span rule (`infer/commit_gate.py`) | The span files and the error types, `--segmenter-eval`, and rows 3-7. `val_offline_score` and `val_video_f1` can select another epoch |

The guards that refuse a stale checkpoint, and the checks that only warn, are in [`docs/run_real_data.md`](docs/run_real_data.md) §8. Some code changes have no guard and no stamp: its section "No guard" says what to run again after each one.

## 11. Reference

### 11.1 Configuration

| Config | Drives | Inheritance |
| --- | --- | --- |
| `dlm.yaml` | The DLM arm; the single source of the sampler, augmentation, BIO-head architecture, gate and optimizer keys | — |
| `ar.yaml` | The AR arm (spec §9.3) | `extends: dlm.yaml` (decoder and output folder only) |
| `baseline_eval.yaml` | The clean translator at evaluation, with no gate. Its `decoder` picks the clean translator that `train-slt` trains and the `pretrained_slt` entry that `--method baseline` reads | `extends: dlm.yaml` (gate off) |
| `baseline_train.yaml` | Trains the clean translator of `decoder` (Mode 1 only, no BIO loss) into `checkpoints/baseline_train_<decoder>/<lang>` | `extends: baseline_eval.yaml` |
| `bio_pretrain.yaml` | S1 (`train-bio`): its own rates, its pool and its input types (`pretrain`) | `extends: dlm.yaml` (BIO-head architecture, BIO loss and augmentation) |
| `moryossef26.yaml` | The Moryossef segmenter (`train-moryossef`) | standalone |
| `inference.yaml` | The FSM design constants (below) | standalone |
| `data.yaml` | The corpora and their roots, split files, caption rules, casing files and frame sizes; the translator starts (`pretrained_slt`, `released_slt`); the pose reader (`poses`) | — |
| `eval.yaml` | The scoring of the rows (`scoring`) and the boundary-sweep grid (`boundary_sweep`) | — |

In `data.yaml`, the `splits`, `subtitles` and `pose` blocks have 3 levels: every corpus, then the dataset (`datasets`), then the corpus (`languages`). Each level replaces only the keys that it names. `--language` points every templated path (`${language}`) at the target corpus, so do not edit a config for one corpus.

`configs/inference.yaml` holds one value per constant for every corpus, all in seconds. They are design values, not measurements, and no step writes them. Each reader converts δ and Λ_min to frames with the frame rate of its stream (12 frames at 24 fps).

| Key | Value | Meaning |
| --- | --- | --- |
| `stride_s` | 1.0 s | The FSM segments the buffer again on each stride. |
| `buffer_cap_s` | 40 s | The longest buffer. At the cap, a complete span commits forced (PARTIAL), and an open span is kept for the next buffer, cut (the cap cut) or dropped (spec §7.1). Training windows and S1's chunks are at most this long. |
| `boundary_stability.delta_s` (δ) | 0.5 s | The terminator tolerance, the overlap that the buffer keeps after a commit, and the span-identity tolerance. The label rules also use it ([`docs/data_pipeline.md`](docs/data_pipeline.md)). |
| `boundary_stability.hysteresis_strides` (K) | 3 | The strides that a terminator must stay within δ before its span commits. |
| `span_selection.min_span_s` (Λ_min) | 0.5 s | The shortest unit that the sampler, the gate and the FSM may target, and the shortest clean segment of the loader. Report the recall limit that it causes. |

Settings that follow from other settings have no key. The code computes each one where it uses it, so a new cap, stride, K or frame rate moves them too.

| Setting | Value | Rule |
| --- | --- | --- |
| Cap cut window | [20 s, 38 s] of buffer time | [cap / 2, cap − (K−1) × stride]. Only the FSM uses it: the offline rows have no cap. |
| S1's longest chunk | 40 s | `buffer_cap_s`. |
| BIO-head attention band r | 0.375 s (9 frames) at 24 fps | S1 derives it: the widest band for which a frame reads at most (K−1) × stride = 2 s ahead, at the slowest train frame rate of the pool. Every later model reads S1's stamp. The FSM refuses a stream on which the head would read further ahead. |
| Decode canvas | 448 tokens | The method config's `max_text_tokens`, the training canvas. |

S1 keys in `configs/bio_pretrain.yaml`:

| Key | Value | Meaning |
| --- | --- | --- |
| `pretrain.inputs` | `[chunks, windows, restart_chunks]` | The input types of the S1 train set, in equal shares (§4). `moryossef26.yaml` lists `[chunks]` only: the Moryossef segmenter trains on plain chunks. `--resume` refuses a change. |
| `pretrain.languages` | `[ase, asf, bfi, bobsl_auto]` | The pool of both segmenters (`moryossef26.yaml` holds the same lists). |
| `pretrain.weights`, `pretrain.temperature` | bfi 0.5, bobsl_auto 0.5 (others 1); 0.0 | Each corpus's share of the hours of an epoch: ∝ weight × hours^temperature. With these values each sign language has one third. |
| `pretrain.dev_languages` | `[ase, asf, bfi]` | The corpora whose dev sets choose the checkpoint (`val_video_f1`). |

Stage-2 keys in `configs/dlm.yaml` (`ar.yaml` inherits them):

| Key | Value | Meaning |
| --- | --- | --- |
| `mode_ratios` | mode1 0.45, mode2 0.30, mode3 0.20, mode4 0.05 | The designed mix of window modes, not a measured one. The clean translator uses mode1 1.0. |
| `jitter.context_s` | 4.0 s | The sampler draws the window edges from Uniform(−4, +4) s around the edges of the anchor unit. The clean translator uses 0.0. |
| `jitter.cut_range` | [0.15, 0.85] | The relative depth of a Mode-2 cut, drawn uniformly. |
| `augmentation` | `fps` {`min_factor` 0.625, `max_factor` 1.25}, `frame_dropout` 0.15, `body_part_dropout` 0.1 | One recipe for every trained pose model (`moryossef26.yaml` holds a copy): frame-rate resampling, frame dropout and hand dropout on the train inputs. Dev, evaluation and the FSM read the 24 fps grid with no augmentation. |
| `learning_rate` | 1e-4 | One rate for every module, in one optimizer group. S1 uses 1e-3 for its head and `backbone_lr` 1e-3 for its encoder. The clean translator uses 2e-4. |
| `lambda_bio` | 1.0 | The weight of the BIO loss. |
| `spd.tau_dec` | 0.5 | The one DLM decode threshold. It is a fixed value: do not tune it on dev. |

Values fitted from labels:

| Value | Source |
| --- | --- |
| duration prior | A lognormal fit of the train unit durations of the target corpus. Stage 2 stamps it in its checkpoint, and evaluation decodes with the stamp. The cascades fit it at run time. No config holds it. |
| `bio.class_weights: balanced` | Computed from the label counts when training starts, and logged. |

### 11.2 Artifact folders

Every artifact goes into one folder per kind under `outputs/`, one level deep. The code chooses the folder, except for the gold span files and the boundary sweeps: there the command gives the path (§7.2, §7.3).

| Folder | Holds | Written by |
| --- | --- | --- |
| `gold/` | gold span files | `eval.py --emit-gold-segments` |
| `offline/`, `online/` | the events of the rows and their `_scores.json`, by the `when` of the row | `eval.py --offline`, `--online`, `--segments` |
| `segmenter/` | segmenter scores (`segmenter_eval_*`) and span files (`segmenter_predictions_*`) | `eval.py --segmenter-eval`, `analyze.py --stage segmenter-infer`, `analyze.py --stage chance-spans` |
| `analysis/` | the error analysis of the segmenters (`error_types_*`, `jitter_*`, `mode_ratios_*`) and the boundary sweeps (`boundary_sweep_*`) | `analyze.py --stage segmenter-errors`, `analyze.py --stage boundary-sweep --output` |
| `stability/` | reveal-policy scores (`stability_*`) and every recorded decode (`provisional_*`) | `eval.py --online --stability` |
| `report/` | one folder per product: `results_<lang>_<split>/`, `progress/`, `data/`, `poses/`, `predict/` | `report.py` |

### 11.3 Row artifact names

Every row writes one pair of files under `outputs/offline/` or `outputs/online/`. Four fields name the row, and every field is always present:

```
{when}_{spans}_{decode}_{translator}_{language}_{split}.json          the events
{when}_{spans}_{decode}_{translator}_{language}_{split}_scores.json   what this run scored on them
```

| Field | Values | Meaning |
| --- | --- | --- |
| `when` | `offline`, `online` | The access of the run: `offline` reads the whole stream, and `online` decides from the past alone. Row 11 is `online`, because it reads the committed spans of row 10. |
| `spans` | `gold`, `moryossef`, `s1`, `joint-ar`, `joint-dlm` | The model that cut the video. The spans of an arm carry the arm, so the same-span controls of the two arms are different files. |
| `decode` | `duration`, `plain`, `none` | The segmentation decode that made the spans; `none` is the annotation. |
| `translator` | `clean`, `clean-dlm`, `ar`, `dlm`, `none` | The translator that read the spans (`clean`: the clean AR translator or the released file; `clean-dlm`: the clean DLM translator); `none` is a run with no text. An ablation arm adds its name tokens (§9.1), so it cannot write over the files of the main arm. |

So `offline_moryossef_duration_clean_asf_test.json` holds the spans of the Moryossef segmenter, with the duration decode, read by the clean translator, offline. A zero-shot row writes `<S>-to-<T>` in the language field.

`segmenter_eval_*` files stay separate: they score spans alone, with no translator. One file holds the Moryossef-protocol block (`metrics`), the edge chunk F1 (`edge_chunk`) and the sentence tIoU F1 (`tiou`).

### 11.4 Repository map

```
backbones/       the Uni-Sign pose encoder (a graph network over 69 keypoints, ST-GCN)
poses/           pose files: Uni-Sign normalization, pose loading and video_meta.csv, augmentation
data/            records to samples: loader, BIO labels and text targets (windowing), stage-2 sampler, segmenter chunks, batches
models/          model parts: BIO head, gate (Ω), Uni-Sign front end with mT5, DLM decoder, the full model (streaming_slt), checkpoint checks
train/           training loops: arms and clean translator (slt), S1 (bio_pretrain), losses, epoch loop, several GPUs
moryossef26/     the Moryossef segmenter: model, input, training, whole-video inference
infer/           inference: duration decode, span and commit rules, DLM decode, the FSM (stream), display stability
evaluation/      what eval.py runs: events and gold (predictions), model builds (systems), rows, offline, online, segmenter_eval; the boundary sweep (boundary_sweep)
reporting/       what report.py runs: breakdowns, per-unit outcome tables, figures, audits, debug views
metrics.py       every metric calculation
utils.py         config loading, checkpoint and output folders, device choice
train.py         --stage {smoke-data, train-bio, train-moryossef, train-slt}
eval.py          the rows (--online, --offline, --segments, --predictions), --segmenter-eval, --emit-gold-segments
analyze.py       --stage {dataset-summary, segmenter-infer, segmenter-errors, boundary-sweep, chance-spans}
report.py        {results, progress, data, poses, predict}
prepare_yt25.py  YouTube-SL-25 through SignVerse-2M: shards, pose converter, captions, video_meta.csv, copy check
docs/            membership_gate.md, segmentation_decoding.md, data_pipeline.md, run_real_data.md, implementation_notes.md, literature_notes.md
```
