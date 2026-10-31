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

**Steps:** [1. Setup](#1-setup) · [2. Run order and rules](#2-run-order-and-rules) · [3. Prepare the data](#3-prepare-the-data) · [4. Stage A](#4-stage-a-the-segmenters-a1-a2) · [5. Pretrimmed translators](#5-stage-b-the-pretrimmed-translators-b2a-b2b) · [6. Arms](#6-stage-b-the-arms-b3) · [7. Evaluate](#7-evaluate) · [8. Read the results](#8-read-the-results) · [9. Optional studies](#9-optional-studies) · [10. After a change](#10-after-a-change-what-to-run-again) · [11. Reference](#11-reference)

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
| gold | The caption units of the annotation and their times. | `--spans gold` |
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
| pretrimmed translator | The released Uni-Sign translator, fine-tuned on the pretrimmed caption crops of one corpus: each crop is cut at the true edges of its caption unit. It has no segmentation part. There is one for each decoder: the pretrimmed AR translator and the pretrimmed DLM translator. | `checkpoints/baseline_train_ar/asf/model.pt` |
| arm, stage 2 | A model that segments and translates. Stage 2 (step B3) trains it from S1 and the pretrimmed translator of its decoder. The main arms are the AR arm and the DLM arm. | `checkpoints/dlm/asf/model.pt` |
| two-stream arm, one-stream arm | The 2 architectures of an arm (§6). The two-stream arm has 2 pose encoders: the segmentation encoder starts from S1, and the translator encoder starts from the pretrimmed translator. The one-stream arm has 1 pose encoder, the translator's, and the BIO head reads it too. | `checkpoints/ar_one_stream/asf/model.pt` |
| no-text arm | An arm whose text loss does not reach its segmentation part. It shows what stage 2 gives without the text. Each architecture has its own no-text arm. | `checkpoints/dlm_notext/asf/model.pt`, `checkpoints/ar_one_stream_notext/asf/model.pt` |
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
| row, result table | A row captions whole test videos end to end, from segmentation to text; a no-text row stops at the spans. The result table holds the rows (§8.1). One `eval.py` command runs one row (§7). | Row 10 is the deployed system. |
| no-text row | A row run with `--text none`: it gives the spans alone, with no text (text token `none`). It is not a row of a no-text arm. | `offline_s1_argmax_none_asf_test.json` |
| events file | The file of one row: its timed captions, or with `--text none` its spans alone. Another row can read its spans (`--spans <file>`). | |
| decode | How a segmenter turns its frame scores into spans: `argmax`, `viterbi` or `duration` (§7.1). | |
| cascade | A row in which one model makes the spans and a translator reads them. | Row 3: Moryossef spans, pretrimmed translator. |
| same-span control | A row in which the pretrimmed translator reads the spans of an arm's row, so only the translator changes. | Row 7 reads the spans of row 6. |
| tIoU | Temporal intersection over union: the overlap of two time intervals, divided by their union. | [0, 4] s and [1, 5] s give 3 / 5 = 0.6. |
| main score, τ | The main localization score of a row. Each caption pairs 1 to 1, by tIoU, with a unit or with a run of units signed with no pause between them. A pair counts in a cell (τ, t) of a grid when its tIoU is at least t and both of its edges are within τ of the edges of its run. So τ is a second limit, on top of the tIoU test. The main score is the F1 at τ avg and tIoU avg (§8.3). | A caption that starts 0.3 s late and ends 0.4 s early on a run of 8 s has tIoU 0.91. It counts in all 9 cells. |
| error types | In each cell (τ, t) of the grid, each caption is matched, joined, cut, bridged, offset or phantom, and each unit is matched, joined, cut, bridged, offset or missed (§8.3). | A caption that holds units on both sides of a pause is bridged. |
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
                                      └→ translator pose encoder (from the pretrimmed translator)
                                            → task prompt + mT5 encoder   ← Ω on the self-attention keys
                                            → AR or DLM decoder           ← Ω on the cross-attention
```

- Two pose encoders with separate weights read the same pose frames. The segmentation encoder starts from S1, and the translator encoder starts from the pretrimmed translator (spec §4.7). This is the two-stream arm. The one-stream arm keeps only the translator encoder, and the BIO head reads it too (§6).
- The BIO head is a transformer over time. Each frame attends only to nearby frames, so the head reads only a short time ahead (§11.1). The duration decode scores whole spans with a prior on the unit duration, fitted on the train labels ([`docs/segmentation_decoding.md`](docs/segmentation_decoding.md)).
- The gate Ω selects the first complete span, and the translator encodes only that crop. In training, the text moves the segmentation only at the end of a target that a pause follows, so it cannot reward a merge ([`docs/membership_gate.md`](docs/membership_gate.md) "Current design"). In the one-stream arm, the text also trains the shared encoder that the BIO head reads (§6).
- The translator is mT5. Its encoder reads a fixed task prompt (the instruction text of Uni-Sign), then the pose features. The AR arm decodes token by token, and the DLM arm decodes in blocks (spec §4.5).
- The FSM commits a span when its terminator stays still for K strides. At the cap or at the stream end it forces the commits, and at the cap it cuts an open span that it cannot keep (spec §7).

## The models

This README uses one name for each model. The table gives the name of the same model in commands, configs and output files.

| Model | Step | Training command | Config | Checkpoint | Evaluation flag | Name in output files |
| --- | --- | --- | --- | --- | --- | --- |
| Moryossef segmenter | A1 | `train.py --stage train-moryossef` | `moryossef26.yaml` | `checkpoints/moryossef/multi_ase-asf-bfi-bobsl_auto/model.pt` | `--spans moryossef` (another file: `--segmenter-checkpoint`) | spans `moryossef` |
| S1 | A2 | `train.py --stage train-bio` | `bio_pretrain.yaml` | `checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto/model.pt` | `--spans s1` (another file: `--segmenter-checkpoint`) | spans `s1` |
| pretrimmed AR translator | B2a | `train.py --stage train-slt --slt-config configs/baseline_train.yaml` | `baseline_train.yaml` to train, `baseline_eval.yaml` to evaluate (`decoder: ar`) | `checkpoints/baseline_train_ar/$LANG/model.pt` | `--text pretrimmed` | text `pretrimmed` |
| pretrimmed DLM translator | B2a | the same command, with `decoder: dlm` in `baseline_eval.yaml` for the run (§5) | the same | `checkpoints/baseline_train_dlm/$LANG/model.pt` | `--text pretrimmed --pretrimmed-checkpoint <that file>` | text `pretrimmed-dlm` |
| AR arm | B3 | `train.py --stage train-slt --slt-config configs/ar.yaml` | `ar.yaml` | `checkpoints/ar/$LANG/model.pt` | `--spans ar` (its head), `--text ar` (its translator); another file: `--arm-checkpoint` | spans `joint-ar`, text `ar` |
| DLM arm (the main system) | B3 | `train.py --stage train-slt --slt-config configs/dlm.yaml` | `dlm.yaml` | `checkpoints/dlm/$LANG/model.pt` | `--spans dlm`, `--text dlm`; another file: `--arm-checkpoint` | spans `joint-dlm`, text `dlm` |
| one-stream AR arm, one-stream DLM arm | B3 | the line of the AR arm or the DLM arm, with the config edit of §6 | `ar.yaml` or `dlm.yaml`, with the edit | `checkpoints/ar_one_stream/$LANG/model.pt`, `checkpoints/dlm_one_stream/$LANG/model.pt` | `--spans ar`, `--text ar` (or `dlm`), with `--arm-checkpoint <that file>` | spans `joint-ar-onestream`, text `ar-onestream` (or `dlm`) |

The AR arm and the DLM arm of the table are the two-stream arms. Each arm also has a no-text arm (§6). The names of a no-text arm add the tokens of its config edit: `-detach-noadapter` for a two-stream arm, `-detach` for a one-stream arm (§7.3).

The pretrimmed translators keep their stored names: the configs `baseline_train.yaml` (to train) and `baseline_eval.yaml` (to evaluate), the folders `checkpoints/baseline_train_<decoder>/<lang>`, the checkpoint stamp `architecture: clean_translation`, and the `history.csv` column of their monitor, `val_translation_bleu4`.

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
| `checkpoints/openasl_pose_only_slt.pth` | The released Uni-Sign OpenASL pose-only translator ([Uni-Sign](https://github.com/ZechengLi19/Uni-Sign)) | S1 (A2), and the pretrimmed translators of every corpus except How2Sign (B2a) |
| `checkpoints/how2sign_pose_only_slt.pth` | The released Uni-Sign How2Sign pose-only translator | The pretrimmed translators of How2Sign (B2a) |
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
| B2a | Train the pretrimmed AR translator and the pretrimmed DLM translator | §5 | CUDA GPU | A0 (A0b for a benchmark) | `checkpoints/baseline_train_ar/$LANG/`, `checkpoints/baseline_train_dlm/$LANG/` |
| B2b | Point `configs/data.yaml` at the pretrimmed translators | §5 | every machine that runs B3 or an evaluation | B2a | an edit of `configs/data.yaml` |
| B3 | Train the arms of both architectures, and the no-text arm of each | §6 | CUDA GPU with much memory | A2, B2b | `checkpoints/{ar,dlm,ar_notext,dlm_notext,ar_one_stream,ar_one_stream_notext}/$LANG/` |
| B4 | Segmenter scores and error types | §7.1 | GPU or Mac | A1, A2 | `outputs/offline/`, `outputs/analysis/` |
| B5 | Boundary sweep (optional, not in the paper) | §7.2 | GPU or Mac | B2b, B3 | `outputs/analysis/` |
| B6 | The rows of the result table | §7.3, §7.4, §7.6 | GPU or Mac | A1, B2b, B3 | `outputs/offline/`, `outputs/online/`, `outputs/stability/` |
| B7 | Reports | §7.5 | any machine | B4, B6 | `outputs/report/` |

- A1, A2 and B2a do not depend on each other, so run them at the same time. Run the 2 B2a runs one after the other, or from 2 copies of the repository: the pretrimmed DLM translator needs a config edit (§5).
- The target corpora are `ase`, `asf`, `bfi`, `how2sign`, `auslan_daily` and `bobsl_manual`. `bobsl_auto` only joins the segmenter pool.
- The main path trains 50 models: the 2 segmenters once, and for each target corpus the 2 pretrimmed translators, the 2 two-stream arms with their 2 no-text arms, and the one-stream AR arm with its no-text arm. The 2 one-stream DLM arms are optional. The optional studies of §9 do not change the main results.
- Stage A commands refuse `--language`, because a pool has no target. Every Stage B command needs `--language "$LANG"`. Without it, the code uses the first entry of `active_languages` in `configs/data.yaml` (`asf`).
- The arms are the largest runs, and they need a GPU with much memory. The online rows (§7.4) take much longer than the offline rows (§7.3).

### 2.2 Two machines

Training needs a CUDA GPU, and evaluation can run on a Mac. When you use more than one machine, copy these files between them:

| To | Copy |
| --- | --- |
| each training machine | The repository, and the 3 released files of §1 in `checkpoints/`. Of `data/youtube-sl-25/`: the split CSV, `case_lexicon.en_XX.json`, and in each `<lang>/` folder `poses/`, `subs/` and `video_meta.csv` (not `signverse_shards/`). Of each benchmark folder: `poses/`, `subs/`, the split files, the sidecars (`video_meta*.csv`) and `case_lexicon.en_XX.json` (not `persons/` and not `.cache/`). |
| the machine of B3 | The `model.pt` of S1 (A2) and of the pretrimmed translator of each decoder (B2a), each in its folder. |
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
- Some runs need a config edit while they train: the pretrimmed DLM translator (§5), the one-stream arms and the no-text arms (§6), and the ablation arms (§9.1). While such an edit is in place, start no other run from the same `configs/` folder, because it reads the edit too. To run them at the same time, use a second copy of the repository.
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

- `bobsl_manual` uses BOBSL's official split: 16 train episodes, the 4 Sent-Val episodes (dev) and the 35 Sent-Test episodes (test).
- `bobsl_auto` joins the segmenter pool (§4), and it trains the pretrimmed translators of `bobsl_manual` (§5), as published BOBSL translators train. Its test split is never read.
- The 2 BOBSL corpora share one folder of episodes. Each split file puts only the episodes of its corpus in train, dev or test, and each load prints how many episodes it uses.
- No model sees a dev or test episode of `bobsl_manual` in training: they are in the official val and test splits, and every model trains on the official train split only.
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

**What it does.** It trains the 2 segmenters once, on the pool. Each sign language fills one third of the hours of every epoch: ASL (`ase`), Auslan (`asf`), and BSL (`bfi` with `bobsl_auto`). The dev sets of `ase`, `asf` and `bfi` choose the checkpoint (`val_video_f1`: the main score of §8.3 on the segmenter's offline no-text row of each dev set, then the mean over the 3). The Moryossef segmenter (A1) is the external baseline: it makes the spans of the cascade rows and of the error analysis. S1 (A2) is the start of the segmentation part of every arm.

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

## 5. Stage B: the pretrimmed translators (B2a, B2b)

**What it does.** B2a trains the pretrimmed translators: it fine-tunes a released Uni-Sign translator on the pretrimmed caption crops of one corpus, each cut at the true edges of its caption unit. It does this once for each decoder: the pretrimmed AR translator (cross-entropy) and the pretrimmed DLM translator (the OPUT loss). Each corpus starts from the OpenASL release, except How2Sign, which starts from Uni-Sign's How2Sign release (`released_slt` in `configs/data.yaml`). `bobsl_manual` has no pretrimmed translators of its own: they train on the automatic alignment of the official train split, so run B2a with `--language bobsl_auto` (`floor_corpus` in `configs/data.yaml`). `train-slt` refuses B2a with `--language bobsl_manual`. The `bobsl_auto` runs choose their best epoch on the hand-aligned Sent-Val episodes of `bobsl_manual` (`dev_corpus` in `configs/data.yaml`), as LiTFiC does. `bobsl_auto` is the largest train set, so its B2a runs are the longest. B2b points `configs/data.yaml` at the 2 pretrimmed translators, so the arm and the cascade start from the same translator. B2 does not need Stage A, so run it at the same time.

**Runs on.** A CUDA GPU for B2a. B2b is a config edit.

**Needs.** The data of §3, and the released files of §1.

```bash
# B2a: train the pretrimmed AR translator (baseline_eval.yaml decoder: ar).
python train.py --stage train-slt --language "$LANG" --slt-config configs/baseline_train.yaml
# B2a: train the pretrimmed DLM translator. First set `decoder: dlm` in configs/baseline_eval.yaml.
# Each decoder writes to its own folder, so the two runs never share a model.pt or a latest.pt.
python train.py --stage train-slt --language "$LANG" --slt-config configs/baseline_train.yaml
```

Set `decoder: ar` again after the DLM run. While `baseline_eval.yaml` holds `decoder: dlm`, `--text pretrimmed` with no `--pretrimmed-checkpoint` reads the pretrimmed DLM translator.

**B2b.** In `configs/data.yaml`, find the entry of the corpus. Replace the active `ar` and `dlm` lines of its `pretrained_slt` (they name the released file) with the pretrimmed translators. The comment lines under them hold the new values. For example, for `asf`:

```yaml
languages:
    asf:
        pretrained_slt:
            ar: checkpoints/baseline_train_ar/asf/model.pt
            dlm: checkpoints/baseline_train_dlm/asf/model.pt
```

For `bobsl_manual`, the same lines name the pretrimmed translators of `bobsl_auto` (its comment lines hold them). B2a always starts from a released file, so a later B2a run needs no change of these lines. `train-slt` refuses an arm whose entry names a released file, or the pretrimmed translator of another corpus or of the other decoder.

**Writes.** `checkpoints/baseline_train_ar/$LANG/` and `checkpoints/baseline_train_dlm/$LANG/` (`model.pt`, `best.json`, `latest.pt`, `history.csv`).

**Check before you continue.**

- `val_translation_bleu4` in the `best.json` of each pretrimmed translator is the dev BLEU-4 of the selected epoch. It must be well above the dev BLEU-4 of the released file with no fine-tuning (spec §10.3). This command measures the released file on the same dev crops; read `text_metrics.translation_bleu4` of the one cell in its output file:

```bash
python analyze.py --stage boundary-sweep --text pretrimmed --pretrimmed-checkpoint checkpoints/openasl_pose_only_slt.pth --language "$LANG" --split dev \
    --severity-grid-head=0 --severity-grid-tail=0 --output outputs/analysis/boundary_sweep_released_${LANG}_dev.json
```

- `val_translation_len_ratio` stays near 1. A value far from 1 shows outputs that are too short or too long.
- The `val_translation_loss` of the pretrimmed DLM translator is the OPUT loss. It is noisy, and it does not compare with the cross-entropy of the pretrimmed AR translator. Only `val_translation_bleu4` selects the epoch.
- Run no separate test of the pretrimmed translators: the number that you report for them is row 1 of the result table (§7.3). `report.py results` also gives the pooled corpus BLEU-4 of row 1 (`located_corpus_bleu4`), as a paper on pretrimmed clips computes it. Do not compare it with a published number: the references are our caption units, and Auslan-Daily uses a re-split.

## 6. Stage B: the arms (B3)

**What it does.** It trains segmentation and translation together under the gate (stage 2). An arm has one of 2 architectures. Both architectures get the same runs and the same rows (§7), and the paper reports one of them (§8.2):

- **Two-stream** (the shipped configs): 2 pose encoders with separate weights. The segmentation encoder starts from S1, and the translator encoder starts from the pretrimmed translator (spec §4.7).
- **One-stream**: 1 pose encoder, the pretrimmed translator's. The BIO head and the translator both read it, so the BIO loss and the text loss both train it. S1's head reads it through a linear map that the run fits at its start (spec §4.7, §1.4 rule 13).

For each architecture, train the arm and its no-text arm. The no-text arm is the same run with the text path into the segmentation cut. It is the control of "text improves the segmentation" (claim 1), and the pretrimmed translator on its committed spans is the fair cascade of claim 2 (§8.2). Train the AR arms first, then the DLM arms. The two-stream DLM arm is the main system. The 2 one-stream DLM arms are optional.

**Runs on.** A CUDA GPU with much memory.

**Needs.** S1 (A2) and the B2b entry of the arm's decoder. Train the arms and run every evaluation under one `configs/inference.yaml`. Design: [`docs/membership_gate.md`](docs/membership_gate.md) "Current design", spec §5 and §6.

**The two-stream arms.** They need no config edit:

```bash
# The two-stream AR arm.
python train.py --stage train-slt --language "$LANG" --slt-config configs/ar.yaml
# The two-stream DLM arm (the main system).
python train.py --stage train-slt --language "$LANG" --slt-config configs/dlm.yaml
```

`train_inputs` in `configs/dlm.yaml` lists the input types of the train rows, in equal shares (by default the 3 types of S1). The dev rows are windows. The run prints the counts of the first epoch.

**The other arms.** Each one is the same run with a config edit. For an AR arm, edit `configs/ar.yaml`. For a DLM arm, edit `configs/dlm.yaml`, and write `dlm` in place of `ar` in the folder and in the name.

| Arm | `shared_pose_encoder` | `bio.adapter` | `membership_gate.detach_omega` | `checkpoint.dir` | `wandb.name` |
| --- | --- | --- | --- | --- | --- |
| two-stream no-text arm | `false` (no change) | `false` | `true` | `checkpoints/ar_notext/${language}` | `ar_notext_${language}` |
| one-stream arm | `true` | `false` | `false` (no change) | `checkpoints/ar_one_stream/${language}` | `ar_one_stream_${language}` |
| one-stream no-text arm | `true` | `false` | `true` | `checkpoints/ar_one_stream_notext/${language}` | `ar_one_stream_notext_${language}` |

1. Set the keys of one line of the table. For example, the one-stream no-text AR arm needs these keys in `configs/ar.yaml`:

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

2. Train with the `train-slt` line of the arm's decoder, for example for an AR arm:

```bash
python train.py --stage train-slt --language "$LANG" --slt-config configs/ar.yaml
```

3. Keep the edit until the run is complete: `--resume` refuses a changed config.
4. Undo the edit before you start another run or any evaluation. While the edit is in place, `--spans ar` or `--text ar` with no `--arm-checkpoint` loads this arm, and its files take the names of the two-stream arm. Evaluate each of these arms with `--arm-checkpoint` (§7.3). Its files carry the name tokens of its edit: `detach` and `noadapter` (two-stream no-text arm), `onestream` (one-stream arm), `onestream` and `detach` (one-stream no-text arm).

What each no-text arm removes:

- In the two-stream no-text arm, the text gives no gradient to the segmentation part: there is no end-side term and no adapter.
- In the one-stream no-text arm, the text reads the shared encoder under a stop-gradient. So the text trains no part that the BIO head reads: the shared encoder (with its `pose_proj`) and the head. The text still trains the mT5 encoder and the decoder, and the BIO loss still trains the shared encoder and the head. Only the gradients change: the outputs and the losses are those of the one-stream arm.

If GPU time is short, train the AR no-text arms first: the paper reports the AR arms, and the DLM no-text rows are optional.

**Writes.** `checkpoints/ar/$LANG/` and `checkpoints/dlm/$LANG/` (the two-stream arms), and the `checkpoint.dir` of each edited run, for example `checkpoints/ar_one_stream/$LANG/`.

**Check before you continue.**

- Before the first full run, watch the GPU memory with `nvidia-smi` during the first steps. If the run is near the memory limit, lower `batch_size`, and raise `grad_accum_steps` by the same factor.
- The dev losses `val_bio_loss` and `val_translation_loss` fall over epochs 1 to 3.
- `text_accepted_rate` (the share of supervised rows that read the model's own span) must not fall on one-unit targets. A correct split of a joined target also lowers it, so read it beside `text_pair_rate` (the share of joined targets).
- `text_grad_rate` (the share of rows on which the text moves the end of the span) is above 0. It is 0 only in a no-text arm.
- Each dev evaluation logs the monitor `val_offline_score` = `val_offline_f1` × `val_offline_bleu4`, and `val_offline_events_per_gold`. `val_offline_f1` is the main score (§8.3) of the offline row on dev: the F1 at τ avg and tIoU avg. `val_offline_bleu4` is the DVC BLEU-4 of the pairs that count, the mean over the 9 cells. Report both factors, not only their product.
- `model.pt` stamps the selected epoch as `best_epoch`. `report.py results --history` shows it (§7.5).
- If a dev evaluation runs out of memory, lower `checkpoint.dev_batch_size`. No score depends on it.
- `checkpoint.monitor: val_online_score` selects on the streaming FSM instead. It is the deployed path, but it is much slower.

**Evaluate the last epoch.** `model.pt` holds the best epoch. To evaluate the last epoch, copy its weights out of `latest.pt` with the command below, because eval cannot load that file. Then give `--arm-checkpoint "checkpoints/ar/${LANG}/last.pt"` to the commands of §7 (for the DLM arm, write `dlm` in place of `ar`; for another arm, use its own folder). Its row files take the names of the files of `model.pt`, so move one set aside before you run the other.

```bash
python -c "import torch; s = torch.load('checkpoints/ar/$LANG/latest.pt', map_location='cpu', weights_only=False); torch.save({'model': s['model'], 'meta': s['meta']}, 'checkpoints/ar/$LANG/last.pt')"
```

## 7. Evaluate

Run this section only when training and checkpoint selection are final. All commands read test (`--split test --allow-test`). Run the subsections in order, on a GPU or on a Mac.

One `eval.py` command runs one row. Its flags are the fields of the row's file name, `{when}_{spans}_{decode}_{text}_{corpus}_{split}` (§11.3):

```
python eval.py --offline|--online --spans SPANS --text TEXT [--decode DECODE] --language L --split S
```

| Flag | Values | Meaning |
| --- | --- | --- |
| `--offline`, `--online` | | `--offline` reads each whole video at once. `--online` runs the streaming FSM, which decides from the past alone. |
| `--spans` | `gold`, `moryossef`, `s1`, `ar`, `dlm`, `chance`, or a file | Where the spans come from: the gold units (offline only), a standalone segmenter, the head of an arm, spans of the median train unit length made by no model (offline only), or the spans of an events file (its own provenance says offline or online). |
| `--text` | `none`, `pretrimmed`, `ar`, `dlm` | The translator: no translation, the pretrimmed translator, or the translator of an arm. |
| `--decode` | `argmax`, `viterbi`, `duration` | Only for `--spans moryossef` or `s1` (the 3 decodes are below). By default, the decode that chose the segmenter's best epoch. |

The model files: `--segmenter-checkpoint` (the segmenter; default its pooled folder), `--arm-checkpoint` (the arm; default its `checkpoint.dir`) and `--pretrimmed-checkpoint` (the pretrimmed translator; default the `pretrained_slt` entry in `configs/data.yaml` of the `decoder` in `configs/baseline_eval.yaml`). One command also translates the spans: an arm translates its own spans, and online the FSM gives each commit of a segmenter to the pretrimmed translator. For any other text, the translator reads the spans after the pass (a cascade). A row refuses a combination that is no row (for example `--online --spans gold`, or the head of one arm with the translator of the other) before it loads any caption or pose.

What a row writes and prints. Each row writes its events file and its `_scores.json` into the folder of its access, `outputs/offline/` or `outputs/online/` (or into `--output`), with no folder per kind of row (§11.3). Every row prints the same report, with the main score alone (§8.3):

- A title line gives the main score (the F1 at τ avg and tIoU avg), the rule of the match in short, and the counts of captions, reliable units and chunks.
- One table follows. It has one column per tIoU threshold (0.5, 0.7, 0.9) and `avg`. It has one group of lines per score: precision, recall and F1, then, when the captions have text, the DVC and the SODA-style scores of the pairs that count. Each group has one line per edge τ (0.5, 1 and 2 s), `avg` and `no limit`. The F1 line at τ `avg`, in the column tIoU `avg`, is the main score. The `no limit` lines count the pairs with the tIoU test alone: they are for reference and in no average.
- Below the table is the line with the units that no caption overlaps. The secondary scores are not printed: one line names the key `densevid` of the `_scores.json` (the one-caption-per-unit scores of densevid_eval), and a no-text row adds one line for the key `frame` (the frame scores of Moryossef et al.).

`--predictions` prints the same report and writes no file. `--tiou-thresholds` changes the tIoU columns of the main score and the thresholds of the densevid scores together, so it changes the main score too: a row with thresholds other than `scoring.tiou_thresholds` in `configs/eval.yaml` needs `--output`, and `report.py` refuses its `_scores.json`.

Which command gives which row. Each command also takes `--language "$LANG" --split test --allow-test`:

| Row | Command |
| --- | --- |
| The Moryossef segmenter alone, offline / online | `eval.py --offline --spans moryossef --text none` / `eval.py --online --spans moryossef --text none` |
| S1 alone, offline / online | `eval.py --offline --spans s1 --text none` / `eval.py --online --spans s1 --text none` |
| The Moryossef cascade, offline (row 3) / online (row 8) | `eval.py --offline --spans moryossef --text pretrimmed` / `eval.py --online --spans moryossef --text pretrimmed` |
| The S1 cascade, offline (row 5) / online (row 9) | `eval.py --offline --spans s1 --text pretrimmed` / `eval.py --online --spans s1 --text pretrimmed` |
| The Moryossef spans read by an arm (row 4) | `eval.py --offline --spans moryossef --text ar` (or `dlm`) |
| The true cuts with the pretrimmed translator (row 1) / with an arm (row 2) | `eval.py --offline --spans gold --text pretrimmed` / `eval.py --offline --spans gold --text ar` (or `dlm`) |
| The arm, offline (row 6) / online (row 10) | `eval.py --offline --spans ar --text ar` / `eval.py --online --spans ar --text ar --stability` (or `dlm`) |
| The arm's spans alone, offline / online | `eval.py --offline --spans ar --text none` / `eval.py --online --spans ar --text none` |
| The same-span control, offline (row 7) / online (row 11) | `eval.py --offline --spans ar --text pretrimmed` / `eval.py --online --spans ar --text pretrimmed` |
| The chance spans (§7.7) | `eval.py --offline --spans chance --text none` |
| The spans of a saved row, read by a translator | `eval.py --spans <events file> --text pretrimmed` |
| The scores of a saved row, again (no model run, no new file) | `eval.py --predictions <events file>` |

`--spans ar` and `--text ar` (or `dlm`) read the two-stream arm of `configs/ar.yaml` (or `configs/dlm.yaml`). For the one-stream arms and the no-text arms, add `--arm-checkpoint` to the same commands (§7.3).

### 7.1 Segmenter scores and error types (B4)

**What it does.** It scores each segmenter alone, with no translator, as a no-text row (`--text none`). One pass over each whole video gives the spans, the scores of every row and the frame scores of the Moryossef protocol. Each segmenter chooses its checkpoint under its own decode (`segmenter_decode` in its config file). The error analysis then shows how a real segmenter fails, with the error types of §8.3. On dev, it is the motivation of the window states of stage 2; no training stage reads it.

**The 3 decodes.** A segmenter gives a score to each label (O, B, I) of each frame. `argmax` takes the best label of each frame alone, so a unit can start with I and have no B. `viterbi` finds the best legal BIO path: the label sequence with the highest total score in which every unit starts with B. Where a buffer has no certified start (a stream start, and online most buffers after the cap), two rules apply: the frames that the head itself reads as I from the first frame are the end of a unit that began earlier and get no caption, and the first unit after O frames may start with I, as with `argmax`. `duration` does the same path search and adds a score for the length of each span, from a duration prior fitted on the train units of the corpus (the semi-Markov decode, [`docs/segmentation_decoding.md`](docs/segmentation_decoding.md)). Both segmenters choose their best epoch with `argmax`, so by default their rows decode with `argmax`, offline and online (rows 3, 5, 8 and 9 too). `viterbi` and `duration` are the other 2 decodes of the decode comparison. An arm's head always decodes with `duration`. `--decode` gives another decode, offline and online. Online, `argmax` also takes the best label of each frame, and the FSM reads these labels with the same rules as the other decodes: the frames that it already committed get no label, and after a commit the first new frame starts a unit when its best label is I. With no `--decode`, a row refuses a segmenter whose checkpoint chose its best epoch under another decode than its config names (for example an A1 that stamps `monitor_decode: semi_markov_viterbi`): train it again.

**The decode comparison.** Each segmenter has 6 no-text rows: the 3 decodes offline (here) and online (§7.4). Each legal-path decode (`viterbi`, `duration`) against `argmax` shows what the legal path adds, and `duration` against `viterbi` shows what the length prior alone adds.

**Needs.** The 2 segmenters of §4.

Score the 6 offline segmenter rows:

```bash
# The Moryossef segmenter, with its own decode (argmax), then viterbi and duration.
python eval.py --offline --spans moryossef --text none --language "$LANG" --split test --allow-test
for D in viterbi duration; do
    python eval.py --offline --spans moryossef --text none --decode "$D" --language "$LANG" --split test --allow-test
done
# S1, with its own decode (argmax), then viterbi and duration.
python eval.py --offline --spans s1 --text none --language "$LANG" --split test --allow-test
for D in viterbi duration; do
    python eval.py --offline --spans s1 --text none --decode "$D" --language "$LANG" --split test --allow-test
done
```

Optional: the untrained controls of the Moryossef architecture, the released weights with no training on this data against random weights of the same model. Read them as a pair, never as a baseline. Their file names carry `moryossef-released` and `moryossef-random<seed>`, so they do not write over the row of the fine-tuned segmenter.

```bash
python eval.py --offline --spans moryossef --text none --segmenter-init released --language "$LANG" --split test --allow-test
for SEED in 0 1 2; do
    python eval.py --offline --spans moryossef --text none --segmenter-init random --seed "$SEED" --language "$LANG" --split test --allow-test
done
```

Run the error analysis on the events files of both segmenters:

```bash
python analyze.py --stage segmenter-errors --language "$LANG" --split test --allow-test \
    --predictions outputs/offline/offline_moryossef_argmax_none_${LANG}_test.json
python analyze.py --stage segmenter-errors --language "$LANG" --split test --allow-test \
    --predictions outputs/offline/offline_s1_argmax_none_${LANG}_test.json
```

`segmenter-errors` decodes nothing. It analyzes the spans of the decode that the `eval.py` row used, and the names of its files hold that decode. For the error types of another decode, give the events file of that row. For example, S1 with viterbi:

```bash
python analyze.py --stage segmenter-errors --language "$LANG" --split test --allow-test \
    --predictions outputs/offline/offline_s1_viterbi_none_${LANG}_test.json
```

The motivation statistics, on dev: the error types of the Moryossef segmenter (A1) for one corpus, beside a model-free chance cutter (§7.7). Run them for each corpus as soon as A1 exists. Report each corpus apart, never as one total:

```bash
python eval.py --offline --spans moryossef --text none --language "$LANG" --split dev
python analyze.py --stage segmenter-errors --language "$LANG" --split dev \
    --predictions outputs/offline/offline_moryossef_argmax_none_${LANG}_dev.json
python eval.py --offline --spans chance --text none --language "$LANG" --split dev
python report.py results --language "$LANG" --split dev \
    --events moryossef=outputs/offline/offline_moryossef_argmax_none_${LANG}_dev.json \
    --events chance=outputs/offline/offline_chance_none_none_${LANG}_dev.json --out outputs/report/motivation_${LANG}_dev
```

**Writes.**

- `outputs/offline/offline_<segmenter>_<decode>_none_<lang>_test.json` and its `_scores.json`: the 6 offline segmenter rows. The controls write `offline_moryossef-released_argmax_none_<lang>_test.json` and `offline_moryossef-random<seed>_argmax_none_<lang>_test.json`. Another row can read the spans of each events file (`--spans <file>`).
- `outputs/analysis/error_types_<segmenter>_<decode>_<lang>_<split>.json` (the share and the count of each error type, per cell (τ, tIoU) and their mean over the cells), `jitter_*` (the edge offsets and the cut positions) and `mode_ratios_*` (the window mode that each error type matches, per cell and their mean), with the same name fields.

**Check before you continue.**

- Each no-text row prints the title line with its main score, the table of the main match (τ lines, tIoU columns) and the line that names the key `frame` of its `_scores.json`.
- Quote `main.edge_chunk` of its `_scores.json`: `avg.f1` is the main score, and `by_tau`, `by_tiou` and `cells` hold the rest of the grid (§8.3). For the same checkpoint and decode, they equal the localization scores of rows 3 and 5. The `frame` block (the protocol of Moryossef et al.) is only for comparison with other papers.

### 7.2 Boundary sweep (B5, optional)

**What it does.** It moves the start and the end of each gold unit on a grid (fractions of the unit duration) and translates each moved crop. It is not part of the paper's evidence: the method never translates a cut unit (P1), so the rows measure the real segmenter errors instead (§8.2). Each run translates every unit once for each cell of the grid.

**Needs.** The pretrimmed translators after the B2b edit, and the arms.

`boundary_sweep.severity_grid_rel` in `configs/eval.yaml` sets the grid of the head axis and of the tail axis. The sweep is the full product of the two axes. `--text` names the translator, as in `eval.py`.

```bash
for T in pretrimmed ar dlm; do
    python analyze.py --stage boundary-sweep --text "$T" --language "$LANG" --split test --allow-test \
        --output outputs/analysis/boundary_sweep_${T}_${LANG}.json
done
```

The tail-context sweep in seconds (spec §9.1) adds context after the unit only, and it writes its own files. Write a grid list that starts with a minus sign with `=`, for example `--severity-grid-head=-0.3,0`.

```bash
for T in pretrimmed ar dlm; do
    python analyze.py --stage boundary-sweep --text "$T" --language "$LANG" --split test --allow-test --severity-mode absolute \
        --severity-grid-head=0 --severity-grid-tail=0,0.5,1,1.5,2,3,4,6 --output outputs/analysis/boundary_sweep_${T}_tail_${LANG}.json
done
```

For the pretrimmed DLM translator, give its file with `--pretrimmed-checkpoint`. The sweep refuses an `--output` without `dlm` in its name for it, so it cannot write over the sweep of the pretrimmed AR translator.

```bash
python analyze.py --stage boundary-sweep --text pretrimmed --pretrimmed-checkpoint checkpoints/baseline_train_dlm/${LANG}/model.pt --language "$LANG" \
    --split test --allow-test --output outputs/analysis/boundary_sweep_pretrimmed-dlm_${LANG}.json
```

For the one-stream arm or a no-text arm, give its file with `--arm-checkpoint`. The sweep refuses an `--output` whose name does not hold each name token of the arm (§7.3) without its dash, so it cannot write over the sweep of the two-stream arm. For example, the one-stream AR arm:

```bash
python analyze.py --stage boundary-sweep --text ar --arm-checkpoint "checkpoints/ar_one_stream/${LANG}/model.pt" --language "$LANG" --split test \
    --allow-test --output outputs/analysis/boundary_sweep_ar_onestream_${LANG}.json
```

**Writes.** The `--output` file. `report.py progress` reads `outputs/analysis/boundary_sweep_pretrimmed_<lang>.json` (the pretrimmed AR translator) when it exists.

**Check before you continue.**

- A positive head value starts the window late. A negative tail value ends the window early.
- For the arms, read `gate_skip_rate` and `text_metrics_decoded_only` beside `text_metrics`. A window with a late start has no B, so the FSM would skip it; `gate_skip_rate` counts these windows. Never plot the decoded-only scores alone: they read a smaller and easier subset.
- Read BLEU against the full caption only in the cells with head ≤ 0 and tail ≥ 0. Only these windows hold all the signing of the unit.
- Compare each curve with its own (0,0) cell, and report that cell separately.

### 7.3 Offline rows: rows 1-7 (B6, block A)

**What it does.** The rows caption whole test videos end to end: they find the units, then translate them. In the offline rows, the whole video is available.

**Needs.** The 2 segmenters, the pretrimmed translators after the B2b edit, and the arms of §6.

After the B2b edit, every `--text pretrimmed` command reads the pretrimmed AR translator from `configs/data.yaml`, so it needs no `--pretrimmed-checkpoint`. The DLM arms are compared only with rows of the pretrimmed DLM translator (one decoder family, §8.2). These commands give `--pretrimmed-checkpoint "$DLM_PRETRIMMED"`, and their files carry `pretrimmed-dlm` in place of `pretrimmed`.

**The arms.** `--spans ar` and `--text ar` (or `dlm`) with no `--arm-checkpoint` read the two-stream arm of the config. Every other arm needs `--arm-checkpoint`. Its files carry the name tokens of its stamps, so no arm writes over the files of another arm:

| AR arm | `--arm-checkpoint` | Span field | Text field | Text field of its no-text row |
| --- | --- | --- | --- | --- |
| two-stream arm | none (it reads `checkpoints/ar/$LANG/model.pt`) | `joint-ar` | `ar` | `none` |
| two-stream no-text arm | `checkpoints/ar_notext/$LANG/model.pt` | `joint-ar-detach-noadapter` | `ar-detach-noadapter` | `none-detach-noadapter` |
| one-stream arm | `checkpoints/ar_one_stream/$LANG/model.pt` | `joint-ar-onestream` | `ar-onestream` | `none-onestream` |
| one-stream no-text arm | `checkpoints/ar_one_stream_notext/$LANG/model.pt` | `joint-ar-onestream-detach` | `ar-onestream-detach` | `none-onestream-detach` |

The DLM arms are the same, with `dlm` in place of `ar`. A cascade on the spans of an arm keeps the span field of the arm and takes the text field of its translator (`pretrimmed` or `pretrimmed-dlm`). For example, row 7 of the one-stream AR arm writes `offline_joint-ar-onestream_duration_pretrimmed_<lang>_test.json`.

Set the paths once in each shell:

```bash
DLM_PRETRIMMED=checkpoints/baseline_train_dlm/${LANG}/model.pt
NOTEXT_AR=checkpoints/ar_notext/${LANG}/model.pt
NOTEXT_DLM=checkpoints/dlm_notext/${LANG}/model.pt
ONE_AR=checkpoints/ar_one_stream/${LANG}/model.pt
ONE_NOTEXT_AR=checkpoints/ar_one_stream_notext/${LANG}/model.pt
# Only when you trained the optional one-stream DLM arms (§6):
ONE_DLM=checkpoints/dlm_one_stream/${LANG}/model.pt
ONE_NOTEXT_DLM=checkpoints/dlm_one_stream_notext/${LANG}/model.pt
```

A cascade row (rows 3, 4, 5 and 7) runs the pass of its segmenter or arm again, with no text, and then translates the spans. To translate the spans of a saved row instead, give its events file to `--spans`; the row keeps the name of those spans.

**Rows 1 and 2: the gold units.** Row 1 reads them with the pretrimmed translator, and row 2 with each arm.

```bash
# Row 1.
python eval.py --offline --spans gold --text pretrimmed --language "$LANG" --split test --allow-test
# Row 2: the two-stream AR arm and DLM arm, then the one-stream AR arm.
python eval.py --offline --spans gold --text ar --language "$LANG" --split test --allow-test
python eval.py --offline --spans gold --text dlm --language "$LANG" --split test --allow-test
python eval.py --offline --spans gold --text ar --arm-checkpoint "$ONE_AR" --language "$LANG" --split test --allow-test
```

**Rows 3 and 4: the Moryossef cascades.** The Moryossef segmenter cuts each video as in §7.1: row 3 with the pretrimmed translator, row 4 with each arm.

```bash
# Row 3.
python eval.py --offline --spans moryossef --text pretrimmed --language "$LANG" --split test --allow-test
# Row 4: the two-stream AR arm and DLM arm, then the one-stream AR arm.
python eval.py --offline --spans moryossef --text ar --language "$LANG" --split test --allow-test
python eval.py --offline --spans moryossef --text dlm --language "$LANG" --split test --allow-test
python eval.py --offline --spans moryossef --text ar --arm-checkpoint "$ONE_AR" --language "$LANG" --split test --allow-test
```

**Row 5: the S1 cascade.**

```bash
python eval.py --offline --spans s1 --text pretrimmed --language "$LANG" --split test --allow-test
```

**Row 6: the arm cuts the video itself.** Offline there is no 40 s buffer: each complete span of at least Λ_min is one event, at any length. Only the online rows have the cap cut.

```bash
# The two-stream AR arm and DLM arm, then the one-stream AR arm.
python eval.py --offline --spans ar --text ar --language "$LANG" --split test --allow-test
python eval.py --offline --spans dlm --text dlm --language "$LANG" --split test --allow-test
python eval.py --offline --spans ar --text ar --arm-checkpoint "$ONE_AR" --language "$LANG" --split test --allow-test
```

Optional: row 6 as a no-text row, for claim 1 offline (§8.2). Each arm and its no-text arm cut each video as in row 6 and translate nothing (text field `none`, with the tokens of the arm).

```bash
# The two-stream AR arm and its no-text arm, then the same for DLM.
python eval.py --offline --spans ar --text none --language "$LANG" --split test --allow-test
python eval.py --offline --spans ar --text none --arm-checkpoint "$NOTEXT_AR" --language "$LANG" --split test --allow-test
python eval.py --offline --spans dlm --text none --language "$LANG" --split test --allow-test
python eval.py --offline --spans dlm --text none --arm-checkpoint "$NOTEXT_DLM" --language "$LANG" --split test --allow-test
# The one-stream AR arm and its no-text arm.
python eval.py --offline --spans ar --text none --arm-checkpoint "$ONE_AR" --language "$LANG" --split test --allow-test
python eval.py --offline --spans ar --text none --arm-checkpoint "$ONE_NOTEXT_AR" --language "$LANG" --split test --allow-test
```

**Row 7: the offline same-span control.** The pretrimmed translator of each arm's decoder reads the spans of that arm's row 6.

```bash
# The two-stream AR arm and DLM arm, then the one-stream AR arm.
python eval.py --offline --spans ar --text pretrimmed --language "$LANG" --split test --allow-test
python eval.py --offline --spans dlm --text pretrimmed --pretrimmed-checkpoint "$DLM_PRETRIMMED" --language "$LANG" --split test --allow-test
python eval.py --offline --spans ar --text pretrimmed --arm-checkpoint "$ONE_AR" --language "$LANG" --split test --allow-test
```

**Rows 1, 3 and 5 for the DLM arms.** The same spans, read by the pretrimmed DLM translator:

```bash
for SPANS in gold moryossef s1; do
    python eval.py --offline --spans "$SPANS" --text pretrimmed --pretrimmed-checkpoint "$DLM_PRETRIMMED" --language "$LANG" --split test --allow-test
done
```

**Optional: the one-stream DLM arms.** Rows 2, 4, 6 and 7, and row 6 with no text for the arm and its no-text arm:

```bash
python eval.py --offline --spans gold --text dlm --arm-checkpoint "$ONE_DLM" --language "$LANG" --split test --allow-test
python eval.py --offline --spans moryossef --text dlm --arm-checkpoint "$ONE_DLM" --language "$LANG" --split test --allow-test
python eval.py --offline --spans dlm --text dlm --arm-checkpoint "$ONE_DLM" --language "$LANG" --split test --allow-test
python eval.py --offline --spans dlm --text pretrimmed --arm-checkpoint "$ONE_DLM" --pretrimmed-checkpoint "$DLM_PRETRIMMED" --language "$LANG" --split test --allow-test
python eval.py --offline --spans dlm --text none --arm-checkpoint "$ONE_DLM" --language "$LANG" --split test --allow-test
python eval.py --offline --spans dlm --text none --arm-checkpoint "$ONE_NOTEXT_DLM" --language "$LANG" --split test --allow-test
```

**Writes.** For each row, the events file `outputs/offline/offline_<spans>_<decode>_<text>_<lang>_test.json` and its `_scores.json` beside it (§11.3). The span and text fields carry the tokens of the table above.

**Check before you continue.** Each row prints the table of its main score, and the provenance of each row names its translator checkpoint and, for a cascade, the pass that made its spans (`span_source`). To print the main score of a saved events file again, with no model run and no new `_scores.json`:

```bash
python eval.py --predictions outputs/offline/offline_joint-dlm_duration_dlm_${LANG}_test.json --language "$LANG" --split test --allow-test
```

### 7.4 Online rows: rows 8-11 (B6, block B)

**What it does.** It makes the same comparison when the FSM must decide from the past alone. These rows take much longer than the offline rows.

**Needs.** Arms trained under the same `configs/inference.yaml` that these runs read, and the paths of §7.3 (`DLM_PRETRIMMED` and the arm files) in the shell. As in §7.3, the rows of the DLM arms read the pretrimmed DLM translator (`--pretrimmed-checkpoint "$DLM_PRETRIMMED"`), and every arm other than the two-stream arm of the config needs `--arm-checkpoint`.

**Rows 8 and 9: the online cascades.** The Moryossef segmenter (row 8) or S1 (row 9) runs under the same FSM with its default decode (argmax), and the pretrimmed translator reads its commits.

```bash
# Row 8.
python eval.py --online --spans moryossef --text pretrimmed --language "$LANG" --split test --allow-test
# Row 9.
python eval.py --online --spans s1 --text pretrimmed --language "$LANG" --split test --allow-test
# Rows 8 and 9 for the DLM arm.
python eval.py --online --spans moryossef --text pretrimmed --pretrimmed-checkpoint "$DLM_PRETRIMMED" --language "$LANG" --split test --allow-test
python eval.py --online --spans s1 --text pretrimmed --pretrimmed-checkpoint "$DLM_PRETRIMMED" --language "$LANG" --split test --allow-test
```

**Row 10: the deployed system.** The arm runs under the FSM. `--stability` also records the display trace of §8.4; it does not change the events.

```bash
# The two-stream AR arm and DLM arm, then the one-stream AR arm.
python eval.py --online --spans ar --text ar --stability --language "$LANG" --split test --allow-test
python eval.py --online --spans dlm --text dlm --stability --language "$LANG" --split test --allow-test
python eval.py --online --spans ar --text ar --stability --arm-checkpoint "$ONE_AR" --language "$LANG" --split test --allow-test
```

**Row 11: the online same-span control.** The pretrimmed translator reads the committed spans of each arm. An FSM commit needs only a stable terminator, so the arm's FSM with no text commits the spans of row 10; to read the events file of row 10 instead, give it to `--spans`.

```bash
# The two-stream AR arm and DLM arm, then the one-stream AR arm.
python eval.py --online --spans ar --text pretrimmed --language "$LANG" --split test --allow-test
python eval.py --online --spans dlm --text pretrimmed --pretrimmed-checkpoint "$DLM_PRETRIMMED" --language "$LANG" --split test --allow-test
python eval.py --online --spans ar --text pretrimmed --arm-checkpoint "$ONE_AR" --language "$LANG" --split test --allow-test
```

**Rows 10n and 11n: the no-text arms.** Each no-text arm (§6) runs as row 10, and the pretrimmed translator reads its committed spans as row 11. Row 11n is the fair cascade of claim 2: its segmenter had the same target training as the arm of its architecture, without text.

```bash
# The two-stream no-text AR arm and DLM arm.
python eval.py --online --spans ar --text ar --arm-checkpoint "$NOTEXT_AR" --language "$LANG" --split test --allow-test
python eval.py --online --spans ar --text pretrimmed --arm-checkpoint "$NOTEXT_AR" --language "$LANG" --split test --allow-test
python eval.py --online --spans dlm --text dlm --arm-checkpoint "$NOTEXT_DLM" --language "$LANG" --split test --allow-test
python eval.py --online --spans dlm --text pretrimmed --arm-checkpoint "$NOTEXT_DLM" --pretrimmed-checkpoint "$DLM_PRETRIMMED" --language "$LANG" --split test --allow-test
# The one-stream no-text AR arm.
python eval.py --online --spans ar --text ar --arm-checkpoint "$ONE_NOTEXT_AR" --language "$LANG" --split test --allow-test
python eval.py --online --spans ar --text pretrimmed --arm-checkpoint "$ONE_NOTEXT_AR" --language "$LANG" --split test --allow-test
```

**Rows 10 and 10n with no text (claim 1).** Each arm and its no-text arm run under the FSM and translate nothing (text field `none`, with the tokens of the arm). The pairs show what the text does to the segmentation (§8.2).

```bash
# The two-stream AR arm and its no-text arm, then the same for DLM.
python eval.py --online --spans ar --text none --language "$LANG" --split test --allow-test
python eval.py --online --spans ar --text none --arm-checkpoint "$NOTEXT_AR" --language "$LANG" --split test --allow-test
python eval.py --online --spans dlm --text none --language "$LANG" --split test --allow-test
python eval.py --online --spans dlm --text none --arm-checkpoint "$NOTEXT_DLM" --language "$LANG" --split test --allow-test
# The one-stream AR arm and its no-text arm.
python eval.py --online --spans ar --text none --arm-checkpoint "$ONE_AR" --language "$LANG" --split test --allow-test
python eval.py --online --spans ar --text none --arm-checkpoint "$ONE_NOTEXT_AR" --language "$LANG" --split test --allow-test
```

**Optional: the one-stream DLM arms.** Rows 10, 11, 10n and 11n, and the no-text rows of claim 1:

```bash
python eval.py --online --spans dlm --text dlm --stability --arm-checkpoint "$ONE_DLM" --language "$LANG" --split test --allow-test
python eval.py --online --spans dlm --text pretrimmed --arm-checkpoint "$ONE_DLM" --pretrimmed-checkpoint "$DLM_PRETRIMMED" --language "$LANG" --split test --allow-test
python eval.py --online --spans dlm --text dlm --arm-checkpoint "$ONE_NOTEXT_DLM" --language "$LANG" --split test --allow-test
python eval.py --online --spans dlm --text pretrimmed --arm-checkpoint "$ONE_NOTEXT_DLM" --pretrimmed-checkpoint "$DLM_PRETRIMMED" --language "$LANG" --split test --allow-test
python eval.py --online --spans dlm --text none --arm-checkpoint "$ONE_DLM" --language "$LANG" --split test --allow-test
python eval.py --online --spans dlm --text none --arm-checkpoint "$ONE_NOTEXT_DLM" --language "$LANG" --split test --allow-test
```

**Optional: the segmenters with no text (the decode comparison, online).** These runs give the segmentation of rows 8 and 9 with no translation, and they compare the segmenters and the 3 decodes under one FSM (§7.1 gives what each pair of decodes compares). Their text field is `none`, so they do not write over the rows with text. The offline twins are in §7.1. With no `--decode`, both segmenters decode with argmax, their main decode (as in rows 8 and 9).

```bash
# The Moryossef segmenter, with its default (argmax), then viterbi and duration.
python eval.py --online --spans moryossef --text none --language "$LANG" --split test --allow-test
for D in viterbi duration; do
    python eval.py --online --spans moryossef --text none --decode "$D" --language "$LANG" --split test --allow-test
done
# S1, with its default (argmax), then viterbi and duration.
python eval.py --online --spans s1 --text none --language "$LANG" --split test --allow-test
for D in viterbi duration; do
    python eval.py --online --spans s1 --text none --decode "$D" --language "$LANG" --split test --allow-test
done
```

The `_scores.json` of every online no-text row also holds `frame`, the frame scores of the FSM's tags, with the keys of the offline no-text rows.

**Writes.** For each row, `outputs/online/online_<spans>_<decode>_<text>_<lang>_test.json` and its `_scores.json`. The span and text fields carry the tokens of the arm (§7.3). Row 10 with `--stability` also writes `outputs/stability/stability_<text>_<lang>_test.json` (the scores of the reveal policies) and `outputs/stability/provisional_<text>_<lang>_test.json` (every recorded decode as text), for example `stability_ar-onestream_<lang>_test.json` for the one-stream AR arm.

**Check before you continue.** Each row prints its `gate_stats` counters and the table of its main score. §8.5 lists what to report from them.

### 7.5 Reports (B7)

**What it does.** `report.py results` scores again the systems that you give it, under one protocol. It gives the main score with its grid, the densevid scores, the error types per caption and per unit in each cell (§8.3), the differences against a reference system, case studies and figures. Any events file of a row can be a system. `report.py progress` makes one report over the finished rows of blocks A and B for the arms of the live config (the two-stream arms). Its section on segmentation alone shows every no-text row of the 2 segmenters that exists, offline (§7.1) and online (§7.4), with the 3 decodes, each labelled with its decode and access. Report the one-stream rows, the no-text rows, the zero-shot rows, the ablations and the dev rows with `report.py results --events`.

**Needs.** The events files of §7.3 and §7.4, and the stability files of row 10.

```bash
python report.py results --language "$LANG" --split test --reference online \
    --events "true cuts"=outputs/offline/offline_gold_none_pretrimmed_${LANG}_test.json \
    --events online=outputs/online/online_joint-dlm_duration_dlm_${LANG}_test.json \
    --events offline=outputs/offline/offline_joint-dlm_duration_dlm_${LANG}_test.json \
    --events cascade=outputs/offline/offline_moryossef_argmax_pretrimmed_${LANG}_test.json \
    --stability-file outputs/stability/stability_dlm_${LANG}_test.json \
    --stability-file outputs/stability/stability_ar_${LANG}_test.json
python report.py progress --language "$LANG"
```

Give the rows of the other arms as more systems, for example `--events "one-stream AR, online"=outputs/online/online_joint-ar-onestream_duration_ar-onestream_${LANG}_test.json` with `--stability-file outputs/stability/stability_ar-onestream_${LANG}_test.json`.

Other inputs of `report.py results`: `--boundary-sweep-file` (§7.2), `--history LABEL=PATH` (the `history.csv` of an arm: the training curves and the selected epoch) and `--segmentation-scores LABEL=PATH` (the `_scores.json` of a segmenter's no-text row, offline (§7.1) or online (§7.4)). One report reads the systems of one source corpus.

**Writes.** `outputs/report/results_<lang>_<split>/` (`results_<S>-to-<T>_<split>/` for zero-shot systems), with `results.md`, `fig/*.png` and `tables/*.csv`, and `outputs/report/progress/`. The main tables:

| Table | What it holds |
| --- | --- |
| `edge_chunk.csv` | The main score of each system (`main.edge_chunk` of a row's `_scores.json`): one line per cell (τ, tIoU), then the average of each τ (tIoU `avg`), of each tIoU (τ `avg`) and of all cells (`avg`, `avg`: the main score), then the `no limit` lines. Each line has the precision, recall and F1, the DVC and SODA-style text scores, and the counted captions and units. |
| `densevid.csv` | The densevid scores of each system (one caption per unit): per tIoU threshold and their average (`avg`), with the pooled corpus BLEU-4 `located_corpus_bleu4` (§5). |
| `error_types.csv` | Per system and cell (τ, tIoU), the share of the units and of the captions of each error type. |
| `outcomes_<system>.csv` | One line per unit and cell: its error type, and the caption that the main match counts with it. |
| `differences.csv` | The difference of each system against the reference system: values only, no confidence interval. |
| `boundary.csv` | The start and end errors of the captions that hold exactly one unit. |
| `edge_boundaries.csv` | Per side (start, end), boundary type (abutting, pause, quarantine, stream) and τ: the unit edges that a caption edge of the same side pairs within τ, one to one, and the share of the caption edges paired. |
| `edge_chunk_gaps.csv` | The main score at each chunk gap of `scoring.chunk_gap_rows_s` in `configs/eval.yaml`. |
| `chunk_classes.csv` | The units of each kind of chunk, for example the chunks longer than the cap, and the share of them that the main match counts (the mean over the cells). |
| `caption_shape.csv` | The caption durations, and the units that each caption holds. |
| `cue_breaks.csv` | How often a caption edge inside a unit falls on a junction of 2 display cues, against chance. |
| `train_text_videos.csv`, `edge_chunk_by_train_text.csv` | The eval videos with a unit whose words are those of a train unit, and every main score with and without them. |
| `cap_cut.csv` | The cap-cut captions apart: their count, the units they hold, how many count in each cell, and the main scores without them. |

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
python eval.py --offline --spans gold --text pretrimmed --language "$TGT" --source-language "$SRC" --split test --allow-test
python eval.py --offline --spans moryossef --text none --language "$TGT" --source-language "$SRC" --split test --allow-test
python eval.py --online --spans moryossef --text pretrimmed --language "$TGT" --source-language "$SRC" --split test --allow-test
python eval.py --online --spans s1 --text pretrimmed --language "$TGT" --source-language "$SRC" --split test --allow-test
python eval.py --offline --spans dlm --text dlm --language "$TGT" --source-language "$SRC" --split test --allow-test
python eval.py --online --spans dlm --text dlm --stability --language "$TGT" --source-language "$SRC" --split test --allow-test
```

Every other row of §7.3 and §7.4 takes `--source-language` the same way. For an arm that needs `--arm-checkpoint`, give the file of the source corpus, for example `--arm-checkpoint "checkpoints/ar_one_stream/${SRC}/model.pt"`. A row on the spans of a saved file (`--spans <file>`) refuses spans of another source.

**Writes.** The row files of §7.3 and §7.4, with `<S>-to-<T>` in the language field (§11.3).

**Lowercase scoring.** `auslan_daily` sets `lowercase_references: true`. Its captions are all lowercase, so every text score of its rows compares lowercase text, for every system alike. The training targets do not change.

### 7.7 Chance rows (appendix)

**What it does.** It scores model-free spans per corpus (`--spans chance`): spans of the median length of the corpus's train units, from each stream start to its end. These rows show how much of the main score a corpus gives with no model, so compare systems only inside one corpus.

```bash
python eval.py --offline --spans chance --text none --language "$LANG" --split test --allow-test
```

**Writes.** `outputs/offline/offline_chance_none_none_<lang>_test.json` and its `_scores.json`. `report.py results --events chance=<file>` puts the row in a table. A translator on these spans (`--text pretrimmed`) is a cascade like the others.

## 8. Read the results

### 8.1 The result table

Block A decides offline, with the whole video available. Block B decides online, from the past alone. Block C repeats the rows on a benchmark with the models of the same sign language from YouTube-SL-25 (§7.6). Inside each block, the rows go from gold spans to the cascades, then to the arm and its same-span control. Compare rows across blocks A and B only through row 10 against row 6.

| # | Spans from | Translator | Command | Artifact |
| --- | --- | --- | --- | --- |
| | **A. Offline: the whole video is available** | | | |
| 1 | gold spans (the annotation) | pretrimmed translator | `--offline --spans gold --text pretrimmed` | `offline_gold_none_pretrimmed_…` |
| 2 | gold spans (the annotation) | each arm | `--offline --spans gold --text ar` (or `dlm`) | `offline_gold_none_{ar,dlm}_…` |
| 3 | Moryossef segmenter, argmax decode | pretrimmed translator | `--offline --spans moryossef --text pretrimmed` | `offline_moryossef_argmax_pretrimmed_…` |
| 4 | Moryossef segmenter, argmax decode | each arm | `--offline --spans moryossef --text ar` (or `dlm`) | `offline_moryossef_argmax_{ar,dlm}_…` |
| 5 | S1, argmax decode | pretrimmed translator | `--offline --spans s1 --text pretrimmed` | `offline_s1_argmax_pretrimmed_…` |
| 6 | the BIO head of the arm, on the whole stream; Ω = a hard crop of each span | each arm | `--offline --spans ar --text ar` (or `dlm`) | `offline_joint-ar_duration_ar_…` / `…joint-dlm_duration_dlm_…` |
| 7 | the spans of row 6 | pretrimmed translator | `--offline --spans ar --text pretrimmed` (or `--spans dlm`) | `offline_joint-{ar,dlm}_duration_pretrimmed_…` |
| | **B. Online: the FSM decides from the past alone** | | | |
| 8 | Moryossef segmenter under the FSM, argmax decode | pretrimmed translator | `--online --spans moryossef --text pretrimmed` | `online_moryossef_argmax_pretrimmed_…` |
| 9 | S1 under the FSM, argmax decode | pretrimmed translator | `--online --spans s1 --text pretrimmed` | `online_s1_argmax_pretrimmed_…` |
| 10 | the BIO head of the arm under the FSM | each arm | `--online --spans ar --text ar --stability` (or `dlm`) | `online_joint-ar_duration_ar_…` / `…joint-dlm_duration_dlm_…` |
| 11 | the committed spans of row 10 | pretrimmed translator | `--online --spans ar --text pretrimmed` (or `--spans dlm`) | `online_joint-{ar,dlm}_duration_pretrimmed_…` |
| 10n | the BIO head of the no-text arm under the FSM | each no-text arm | `--online --spans ar --text ar --arm-checkpoint "$NOTEXT_AR"` (or another no-text arm, §7.3) | `online_joint-ar-detach-noadapter_duration_ar-detach-noadapter_…` (two-stream) / `online_joint-ar-onestream-detach_duration_ar-onestream-detach_…` (one-stream) |
| 11n | the committed spans of row 10n (the fair cascade) | pretrimmed translator | `--online --spans ar --text pretrimmed --arm-checkpoint "$NOTEXT_AR"` (or another no-text arm) | `online_joint-ar-detach-noadapter_duration_pretrimmed_…` / `online_joint-ar-onestream-detach_duration_pretrimmed_…` |
| | **C. Zero-shot: models of the same sign language, a benchmark's data** | | | |
| 1-11, 10n, 11n | as above, with the source models | as above | each command with `--source-language S` (§7.6) | the name field `S-to-T`, for example `…_asf-to-auslan_daily_test.json` |

The arm rows (2, 4, 6, 7, 10 and 11) run for each arm of §6, and rows 10n and 11n for the no-text arm of each architecture (§7.3, §7.4). The table shows the files of the two-stream arms, which need no `--arm-checkpoint`. The files of a one-stream arm add `-onestream` to the arm in the span and text fields, for example `online_joint-ar-onestream_duration_ar-onestream_…` (row 10) and `offline_joint-ar-onestream_duration_pretrimmed_…` (row 7). The DLM arms use the same names with `dlm` in place of `ar`.

A row of the pretrimmed DLM translator writes `pretrimmed-dlm` in place of `pretrimmed`, for example `offline_s1_argmax_pretrimmed-dlm_…` and `online_joint-dlm_duration_pretrimmed-dlm_…`. The `--stability` files of its online cascade are `stability_<segmenter>-dlm_…`.

### 8.2 Which rows to compare

Compare rows of one target corpus, one decoder family and one architecture. The full reasons are in spec §9.2 "How to read the table".

The rows follow the story of the paper. A real segmenter makes errors (§7.1), and these errors cost the pretrimmed translator, which trained on crops cut at the true edges (1 → 3 and 1 → 5 offline; 8 and 9 online). The window modes give each error state a label rule, so the arm learns from wrong segments, and one model learns both tasks. The arm then runs offline (row 6) and under the FSM (row 10). The window modes are a label rule, not a claim that the segmenter finds better edges. A segmentation gain of the arm must come from the text, and only the no-text arm isolates the text.

- **Joint training against the cascades: 10 against 11n, 8 and 9 online; 6 against 3 and 5 offline.** This is the deployed claim. 11n is the fair cascade: its segmenter had the same target training as the arm, without text. Take 10n and 11n from the no-text arm of the same architecture as row 10. The gain 10 − 11n has 2 parts that add up exactly: 10 − 11n = (10n − 11n) + (10 − 10n). 10n − 11n changes only the translator on the same committed spans, and 10 − 10n is the share of the text path into the segmentation. Report both parts.
- **Text improves the segmentation: the arm against its no-text arm of the same architecture** (10 against 10n with `--text none`, and 11 against 11n; offline, row 6 of each with `--offline --text none`); also 7 against 5 offline and 11 against 9 online. From S1 to the arm, the data and the training also change, so only the no-text arm shows that the text is the cause. In the two-stream arm, the text reaches the segmentation through the end-side term and the adapter. In the one-stream arm, the text also trains the shared encoder that the BIO head reads, on every row with a text target. Each no-text arm removes all of these paths (§6).
- **The 2 architectures: the one-stream arm against the two-stream arm** of the same decoder, on the same rows: 6 and 10, their rows with `--text none`, and the 2 claims above. The paper reports one architecture. Choose it on dev: run the rows with `--split dev` and without `--allow-test`, and run test only after the choice (§2.3).
- **Zero-shot (block C).** The same pairs on a benchmark, with no label of the benchmark's own annotation in any model: does joint minus cascade keep its sign when the domain changes?
- **Span quality with one translator: 1 → 3 → 5 → 7.** Only the span source changes.
- **Translator at fixed spans: 2 − 1, 4 − 3 and 7 − 6 offline; 10 − 11 online.** These pairs measure the whole training package of the arm. (4 − 3) against (2 − 1) shows if the arm loses less than the pretrimmed translator on the real errors of one segmenter.
- **Access: 10 − 6.** The same arm, online against offline.
- **Online split: 9 → 11 → 10.** 9 → 11 keeps the pretrimmed translator and changes the segmentation. 11 → 10 keeps the committed spans and changes the translator.
- **Which errors change.** Compare the error types (§8.3) of the 2 rows of a pair to see which errors a change removes.
- Compare systems only inside one corpus: the chance rows (§7.7) give each corpus its own main score with no model.

### 8.3 What the scores mean

Each row's `_scores.json` holds, in this order, `provenance`, `main` (`edge_chunk`, `uncaptioned`), `densevid` and, for a no-text row, `frame`. Every kind of row writes these keys, offline and online (§11.3). The full definitions are in spec §9.2.

**The main score** (`main.edge_chunk`). It pairs the captions with the gold units in 3 steps:

1. The reliable units of a stream that follow each other with gaps of at most `scoring.chunk_gap_s` in `configs/eval.yaml` (0.5 s) form a chunk. A caption can pair with one unit, or with a run of consecutive units of one chunk, because a signer marks no boundary between units signed with no pause. A run never crosses a pause.
2. In each video, the captions and the runs pair 1 to 1 by tIoU, once: the highest tIoU first, each caption and each unit used only once, and only pairs with a tIoU of at least the lowest threshold (0.5).
3. A grid of cells (τ, t) decides which pairs count: τ from `scoring.edge_taus_s` (0.5, 1 and 2 s), t from `scoring.tiou_thresholds` (0.5, 0.7 and 0.9). A pair counts in a cell when its tIoU is at least t and both its start and its end are within τ of the start and the end of its run. So τ is a second limit, on the edges of a pair that already passed the tIoU test. It never makes the tIoU test easier.

In each cell, precision = the counted captions / all captions and recall = the units in counted runs / all reliable units, per video, then the mean over the videos; F1 = 2PR / (P + R). An average takes the mean precision and the mean recall of its cells, and its F1 is the F1 of these 2 means, not the mean of the F1 values, as in our NeurIPS paper. There are 3 kinds of average: each τ over the 3 thresholds (`by_tau`), each threshold over the 3 τ values (`by_tiou`), and all 9 cells (`avg`). So each score has 16 values: 9 cells, 3 τ averages, 3 tIoU averages and the average of all cells. **The main score of a row is the F1 at τ avg and tIoU avg: `main.edge_chunk.avg.f1`.** The text scores (`dvc` and `soda`, below) read the same counted pairs.

An example. Units A [10.0, 14.0] s and B [14.2, 18.0] s have a gap of 0.2 s, so they are one chunk.

- A caption at [10.3, 17.6] s pairs with the run A B: tIoU 7.3 / 8.0 = 0.91, start error 0.3 s, end error 0.4 s. It counts in all 9 cells, and its text is scored against the joined text "A B".
- A caption at [9.0, 18.5] s pairs with the run A B: tIoU 8.0 / 9.5 = 0.84, start error 1.0 s, end error 0.5 s. It counts in 4 cells: τ = 1 and 2 s, each with t = 0.5 and 0.7.
- A caption at [20.0, 42.5] s on a unit [20.0, 45.0] s has tIoU 0.9 and an end error of 2.5 s. It passes every tIoU test, but it counts in no cell, because its end is more than 2 s off. With a tIoU test alone, the allowed edge error grows with the length of the caption; τ stops that.

**The "no limit" row** (`main.edge_chunk.no_edge_limit`) is for reference only. It counts the pairs of the same match at each threshold with no edge limit, so it is the tIoU test alone, on runs of units. It is in no average, in no monitor and in no error type. It counts the last caption of the example at every threshold. Its τ is "no limit", not 0: τ = 0 would ask for exact edges. It is not the densevid block (below) either, which pairs one caption with one unit.

**What differs from the protocol of our NeurIPS paper.** The averages are the same: the F1 of the mean precision and the mean recall. There are 3 deliberate differences, and τ is new:

1. A caption can pair with a run of abutting units inside one chunk, not only with one unit.
2. The pairs are 1 to 1. The NeurIPS protocol counts a caption that reaches the threshold with any unit, and a unit that any caption reaches.
3. The scores are per whole video, not per 15 s window.
4. A pair must also have both of its edges within τ.

The thresholds also have no 0.3. So these scores compare with the NeurIPS numbers only loosely.

| Key | What it holds |
| --- | --- |
| `main.edge_chunk.avg` | The average of all 9 cells: `precision`, `recall`, `f1` (the main score), `dvc` and `soda`. |
| `main.edge_chunk.by_tau`, `main.edge_chunk.by_tiou` | The average of each τ over the thresholds, and of each threshold over the τ values, with the same keys. |
| `main.edge_chunk.cells` | The 9 cells, each with its `tau_s` and `tiou`, the same keys, and the counted captions and units (`matched_captions`, `matched_units`). |
| `main.edge_chunk.no_edge_limit` | The reference row: one line per threshold (`by_tiou`) and their average (`avg`). |
| `dvc` | The text scores (BLEU-4, BLEURT, ROUGE-L, METEOR, CIDEr-D) with the DVC rule of densevid_eval: in each video, each caption against the joined text of its counted run, and each other caption against a random garbage text; then the mean over the videos. A missed unit adds no pair. |
| `soda` | A SODA-style text F1 on the same pairs. It charges the captions that do not count and the missed units in one number. It has no CIDEr-D column. |
| `main.uncaptioned` | The reliable units that no caption overlaps. |
| `densevid` | The scores of densevid_eval's pairing (one caption per unit, by tIoU), for comparison with other papers, in the same shape: `by_tiou` (per threshold: the sentence tIoU precision, recall and F1, the DVC of densevid_eval in `dvc`, the counts, the matched tIoU and the emission latency) and `avg` (the same rule of averages). A no-text row has no `dvc`. |
| `frame` | No-text rows only: the frame and run scores of the protocol of Moryossef et al., for comparison with other papers (§11.3). |

Quote the precision and the recall beside the DVC: the DVC charges a caption that does not count, but a missed unit adds no pair. Quote the densevid recall beside a densevid DVC column.

**Error types.** They read the pairs that count in each cell, so they agree with the main score. A caption holds a unit when it covers at least half of the unit. An edge cuts a unit when it lies inside the unit, more than τ from both of its edges. In each cell, each caption and each unit takes the first type whose rule fits. The table uses the units A and B above and a third unit C [20.0, 23.0] s after a 2 s pause. Each example caption is alone in its video, and the cell is τ = 0.5 s, t = 0.5:

| Type | Rule for a caption | Example caption | Rule for a unit |
| --- | --- | --- | --- |
| matched | it counts with a run of one unit | [10.2, 13.8] for A | it is that unit |
| joined | it counts with a run of two or more units of one chunk (correct) | [10.3, 17.6] for A B | it is in that run |
| cut | an edge cuts a unit | [12.5, 17.9]: its start is 2.5 s inside A | a caption edge cuts it |
| bridged | it holds units of two or more chunks (it crosses a pause) | [14.4, 22.8] holds B and C | a caption holds it and a unit of another chunk |
| offset | it holds units of one chunk, but it does not count in this cell: an edge is more than τ from the run edges, its tIoU is below t, or another caption took the match | [19.0, 23.1]: it starts 1 s early, in the pause (it counts at τ = 1 and 2 s, with t = 0.5 or 0.7) | a caption holds it |
| phantom (captions) / missed (units) | it holds no unit and cuts none | [18.4, 19.6], in the pause | no caption holds it or cuts it (every unit of `main.uncaptioned` is missed) |

In each cell, the share of matched and joined captions is the precision of that cell, and the share of matched and joined units is its recall (per video, then the mean over the videos). The other types split the misses. The reports give the shares per cell and their mean over the cells. For example, [20.2, 25.5] has tIoU 0.51 with C, so it passes the tIoU test at t = 0.5. But its end is 2.5 s late, more than every τ, so it is offset in every cell, and only the "no limit" row counts it.

Each table computes BLEU-4 in its own way. The number of the pretrimmed translator on the true cuts is row 1 (§5). The exact rules are in spec §9.2 "BLEU rule".

| Column | Where | What it is |
| --- | --- | --- |
| `translation_bleu4` | `val_translation_bleu4` in training; the `text_metrics` of each boundary-sweep cell | Corpus BLEU-4 over all captions of the set |
| `located_corpus_bleu4` | `report.py results` `tables/densevid.csv` | Corpus BLEU-4 over the tIoU-matched pairs; on row 1 every unit matches, so it is the pooled corpus BLEU-4 of all units |
| `main.edge_chunk.avg.dvc.bleu4` | `_scores.json` | Corpus BLEU-4 on the counted pairs of each video, then the mean over the videos, then over the 9 cells |
| `main.edge_chunk.avg.soda.bleu4` | `_scores.json` | Sentence BLEU-4 of each counted pair, inside the SODA-style F1, then the mean over the 9 cells |
| `densevid.avg.dvc.bleu4` | `_scores.json` | Corpus BLEU-4 on the tIoU pairs of each video, then the mean over the videos, then over the thresholds |

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

**What it does.** It trains each ablation as its own arm, never as a switch at evaluation time. The pretrimmed translator already is the ablation with no misalignment training. The one-stream arms and the no-text arms are not ablations: §6 trains them, and §7 evaluates them.

**Needs.** The same as B3 (§6).

| Ablation | Keys to set | `checkpoint.dir` | Name token |
| --- | --- | --- | --- |
| Seed repeat | `seed: <n>` | `checkpoints/ar_seed<n>/${language}` | `seed<n>` |

1. In `configs/ar.yaml` (or `configs/dlm.yaml`), set the keys of the table, a new `checkpoint.dir` and a new `wandb.name`.
2. Train with the `train-slt` line of §6. Keep the edit until the run is complete: `--resume` refuses a run whose config changed.
3. Undo the edit before any evaluation. While the config holds the edit, `--spans ar` or `--text ar` with no `--arm-checkpoint` loads the ablation arm, and its files can take the names of the main arm and write over them.
4. Evaluate the ablation with the commands of §7. Give `--arm-checkpoint` at the `model.pt` of its folder in each command with `--spans ar|dlm` or `--text ar|dlm`.

A boundary-sweep `--output` path (§7.2) must hold each name token of the arm. For example, for the seed repeat 1:

```bash
python analyze.py --stage boundary-sweep --text ar --arm-checkpoint "checkpoints/ar_seed1/${LANG}/model.pt" --language "$LANG" --split test \
    --allow-test --output outputs/analysis/boundary_sweep_ar_seed1_${LANG}.json
```

**Writes.** The checkpoint folder of each ablation, and row files that carry its name token (§11.3).

**Check before you continue.**

- Spec §11 item 4 gives the rule that decides whether to keep the end-side text term and the adapter (`bio.adapter`, which gives translator features to the BIO head). The rule needs a confidence interval of the dev difference, which the code does not compute: compute it outside the repository.
- A seed repeat shows how much the score of one arm changes from one run to the next. Compare two arms only when their difference is larger than that change.

### 9.2 Segmenter pretraining study (each target corpus)

**What it does.** It measures what the pooled pretraining of §4 gives each segmenter on one target corpus `T`. Each segmenter trains in up to 3 conditions, and all conditions get the same evaluation. The main plan needs none of these runs: every arm starts from the pooled S1 (A2), the cascades use A1 and A2, and the error statistics of §7.1 use A1. Run the study after the main round, and only if the paper makes a claim about the pool.

**Runs on.** A CUDA GPU for the training, any machine for the scores.

**Needs.** The data of §3, A1 and A2 (§4), and the released start weights `checkpoints/openasl_pose_only_slt.pth` (S1) and `checkpoints/moryossef26.safetensors` (Moryossef segmenter).

| Condition | Train data | Start weights | S1 | Moryossef segmenter |
| --- | --- | --- | --- | --- |
| 1. Target only | `T` | the released weights | new run | new run |
| 2. Pooled | the pool of §4 | the released weights | A2 (no new run) | A1 (no new run) |
| 3. Pooled, then target | `T` | the checkpoint of condition 2 | new run | not supported |

What each comparison measures:

- **2 against 1** (both segmenters): the value of the pool. For `ase`, `asf` and `bfi`, the pool holds `T`, so this is the value of the other corpora.
- **3 against 2** (S1): the value of more training on `T`.
- **3 against 1** (S1): pooled-then-target against target only. If condition 3 does not beat condition 1, do not claim that the pool helps.

**Fix the budget first.** Before the first run, write down the maximum epochs, the patience and the batch size, and use the same values for every condition and every `T`. The same number of epochs is not the same amount of training: a pooled epoch and a target epoch hold different numbers of batches. To claim a gain at equal budget, also match the optimizer updates of conditions 1 and 3.

**Set a condition.** A condition is 3 keys in the config file of the segmenter. Change only these keys. With `pretrain.languages: null`, the run trains and chooses its epoch on `T` alone, and `${corpus}` in the folder name becomes `T`.

S1, in `configs/bio_pretrain.yaml`:

| Key | Condition 1 | Condition 2 (as shipped) | Condition 3 |
| --- | --- | --- | --- |
| `pretrain.languages` | `null` | `[ase, asf, bfi, bobsl_auto]` | `null` |
| `checkpoint.from_pretrained` | `checkpoints/openasl_pose_only_slt.pth` | `checkpoints/openasl_pose_only_slt.pth` | `checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto/model.pt` |
| `checkpoint.dir` | `checkpoints/bio_s1_target/${corpus}` | `checkpoints/bio_s1/${corpus}` | `checkpoints/bio_s1_finetune/${corpus}` |
| The checkpoint to score | `checkpoints/bio_s1_target/$T/model.pt` | `checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto/model.pt` | `checkpoints/bio_s1_finetune/$T/model.pt` |

Moryossef segmenter, in `configs/moryossef26.yaml`:

| Key | Condition 1 | Condition 2 (as shipped) |
| --- | --- | --- |
| `pretrain.languages` | `null` | `[ase, asf, bfi, bobsl_auto]` |
| `checkpoint.from_pretrained` | `checkpoints/moryossef26.safetensors` | `checkpoints/moryossef26.safetensors` |
| `checkpoint.dir` | `checkpoints/moryossef_target/${corpus}` | `checkpoints/moryossef/${corpus}` |
| The checkpoint to score | `checkpoints/moryossef_target/$T/model.pt` | `checkpoints/moryossef/multi_ase-asf-bfi-bobsl_auto/model.pt` |

**Rules for every condition.**

- A segmenter checkpoint loads only while its config file holds the pool that it trained with. So score each checkpoint while its condition is set.
- Do not run `train-slt`, a row of §7 or `report.py` while a study condition is set. Stage 2 and the rows find the segmenters through these files, so they would read the study checkpoint and write files under the main names. The scoring commands below name their checkpoint and their own `--output`.
- After each condition, put the shipped values back. Then `git diff configs/bio_pretrain.yaml configs/moryossef26.yaml` must show nothing.
- Choose on dev. Read the test files only after you decide.

Set the target and the budget:

```bash
T=asf          # asf | bfi | ase | how2sign | auslan_daily | bobsl_manual
E=100          # the maximum epochs, fixed before the first run
```

**Step 1: condition 2.** No training: score A2 and A1 as they are.

```bash
python eval.py --offline --spans s1 --text none --decode argmax --language "$T" --split dev \
    --segmenter-checkpoint checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto/model.pt --output outputs/offline/study_case2_s1_${T}_dev.json
python eval.py --offline --spans moryossef --text none --decode argmax --language "$T" --split dev \
    --segmenter-checkpoint checkpoints/moryossef/multi_ase-asf-bfi-bobsl_auto/model.pt --output outputs/offline/study_case2_moryossef_${T}_dev.json
```

**Step 2: condition 1.** Set the condition-1 keys in both files. Train both segmenters, then score them while the keys are still set:

```bash
python train.py --stage train-bio --language "$T" --epochs "$E"
python train.py --stage train-moryossef --language "$T" --epochs "$E"
python eval.py --offline --spans s1 --text none --decode argmax --language "$T" --split dev \
    --segmenter-checkpoint checkpoints/bio_s1_target/${T}/model.pt --output outputs/offline/study_case1_s1_${T}_dev.json
python eval.py --offline --spans moryossef --text none --decode argmax --language "$T" --split dev \
    --segmenter-checkpoint checkpoints/moryossef_target/${T}/model.pt --output outputs/offline/study_case1_moryossef_${T}_dev.json
```

Then put the shipped values back in both files, and check the `git diff`.

**Step 3: condition 3, S1 only.** Set the condition-3 keys in `configs/bio_pretrain.yaml`. Start the run without `--resume`: it loads only the weights of condition 2, with a new optimizer and schedule. If the run stops, run the same command with `--resume`, which continues from `checkpoints/bio_s1_finetune/$T/latest.pt`.

```bash
python train.py --stage train-bio --language "$T" --epochs "$E"
python eval.py --offline --spans s1 --text none --decode argmax --language "$T" --split dev \
    --segmenter-checkpoint checkpoints/bio_s1_finetune/${T}/model.pt --output outputs/offline/study_case3_s1_${T}_dev.json
```

Then put the shipped values back, and check the `git diff`.

**Optional: the error types of a condition.** Run this inside the step of the condition, while its keys are set, on the events file of its scoring command. Set `S` (`s1` or `moryossef`) and `K` (the condition). The names carry the condition, so the error files of A1 and A2 stay:

```bash
python analyze.py --stage segmenter-errors --language "$T" --split dev \
    --predictions outputs/offline/study_case${K}_${S}_${T}_dev.json --output-dir outputs/analysis/study_case${K}_${S}
```

**Test, once, at the end.** For each condition, set its keys again and run its scoring command with `--split test --allow-test`, and `test` in place of `dev` in the output name.

**Writes.** The folders `checkpoints/bio_s1_target/`, `checkpoints/moryossef_target/` and `checkpoints/bio_s1_finetune/`, the scores `outputs/offline/study_case*` (each with its `_scores.json`), and the error files in `outputs/analysis/study_case*/`.

**Check before you continue.**

- Both config files hold the shipped values again: `git diff configs/bio_pretrain.yaml configs/moryossef26.yaml` shows nothing.
- Report, for each condition: the main score, the selected epoch (in `best.json`), the optimizer updates and the training time.
- Everything else is the same in all conditions: the chunk lengths, `pretrain.inputs`, the augmentation, the loss recipe, the attention band and the evaluation. Each condition fits its duration prior on the train labels of `T`.
- These differences remain, and the paper must say them:
  - Condition 2 chose its epoch on the dev sets of `ase`, `asf` and `bfi`, while conditions 1 and 3 chose on the dev set of `T`. For a benchmark `T`, this favours conditions 1 and 3.
  - Each run measures its class weights on its own train records.
  - Only condition 2 has a best-epoch floor (one full rotation of the pool).
  - The Moryossef condition 1 fits its input normalization on the train records of `T`.
- Every condition chooses its epoch with argmax, the decode that both segmenter configs name, and is scored with argmax. The shipped pooled checkpoints (condition 2) chose their epoch with the duration decode, so the scoring commands give `--decode argmax`; with no `--decode` a row refuses such a checkpoint.
- Not supported: the Moryossef condition 3 (its trainer takes start weights only from a `.safetensors` file), and stage 2 or a cascade row with a study segmenter (both find the segmenters through the shipped config files).
- How to read the results: [`docs/membership_gate.md`](docs/membership_gate.md) §4.

### 9.3 S1 input types

**What it does.** It compares choices of `pretrain.inputs` (§4): for example `[chunks]` (the data of the Moryossef segmenter) against the default `[chunks, windows, restart_chunks]`. Dev chunks are plain chunks for every choice, and the checkpoint stamps the list (`pretrain_inputs`).

**Needs.** The same as A2 (§4).

Every choice trains into the same folder, and a run refuses to start while `latest.pt` is in it. So:

1. Train one choice. Then move its folder by hand, for example to `checkpoints/bio_s1/multi_ase-asf-bfi-bobsl_auto_chunks`.
2. Train the next choice. Train the choice that the arms use last, into the default folder: the arms find S1 there.
3. Score a moved run with `--segmenter-checkpoint <folder>/model.pt` and its own `--output` (`eval.py --offline --spans s1 --text none`). Its files save the list as `segmenter_inputs`.
4. The rows name their files by the segmenter only. Give every row of a moved run (`--spans s1`, offline or online, with or without text) its `--segmenter-checkpoint` and its own `--output`, and before an online S1 row with `--stability`, move the `stability_s1_*` files of `outputs/stability/` aside.

## 10. After a change: what to run again

Find your change in the left column. Run again everything in the right column, in the order of §2.1. Before a new training run, move its `latest.pt` (§2.3).

| Change | Run again |
| --- | --- |
| Arm architecture, loss, sampler, or how the translator reads the span | B3 (every arm of §6, as new runs), then their rows 2, 4, 6, 7, 10, 11, 10n and 11n |
| S1 recipe, pool or input types (`pretrain.*` in `configs/bio_pretrain.yaml`, or a new slowest train frame rate); with windows in `pretrain.inputs`, also the window sampler (`data/sampler.py`, and `mode_ratios`, `mode2_subcase_weights` or `jitter` in `dlm.yaml`) | A2, the S1 no-text rows, rows 5 and 9, then B3 and every row of the arms |
| Moryossef checkpoint or its whole-video inference | The Moryossef no-text rows and error analysis (B4), then rows 3, 4 and 8 |
| Pretrimmed translator of a decoder | The B2b entry of that decoder, the B3 arm of that decoder, and every `--text pretrimmed` row that reads it: rows 1, 3, 5, 7, 8, 9 and 11 (for the pretrimmed DLM translator: the rows that you ran with its `--pretrimmed-checkpoint`) |
| Learning rate of a stage | That stage and every stage that loads it |
| `augmentation` (`dlm.yaml`; keep the copy in `moryossef26.yaml` equal) | Every pose model (A1, A2, B2a and the B2b edits, B3), then every row. Stage 2 refuses a pretrimmed translator stamped with another block |
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
| `spd.tau_dec` | B2a of the pretrimmed DLM translator and its B2b entry, the DLM arm, then their rows |
| A checkpoint monitor | Every model that the monitor selects, then every row that reads these models |
| Scoring (`metrics.py`, `scoring` in `configs/eval.yaml`) | Every row (the no-text rows of §7.1 too), the error types of §7.1 and every `report.py` output. `chunk_gap_s`, `edge_taus_s` and `tiou_thresholds` also change both monitors (the row above); `chunk_gap_rows_s` changes `report.py results` only |
| The rule of the main score, or the shape of `_scores.json` | Every saved row of §7, because `report.py` refuses a `_scores.json` of another shape and names the row to run again. Then the error types of §7.1 and every `report.py` output. Both monitors read the main score: `--resume` refuses a run whose `latest.pt` stamps another monitor, so start that run again. A finished `model.pt` still loads for evaluation and as a warm start, but it chose its epoch under the other score (the row "A checkpoint monitor") |
| FSM rules (`infer/stream.py`) | Rows 8-11 and the stability files. `val_online_score` can select another epoch |
| Offline span rule (`infer/commit_gate.py`) | The no-text rows of §7.1, the error types, and rows 3-7. `val_offline_score` and `val_video_f1` can select another epoch |

The guards that refuse a stale checkpoint, and the checks that only warn, are in [`docs/run_real_data.md`](docs/run_real_data.md) §8. Some code changes have no guard and no stamp: its section "No guard" says what to run again after each one.

## 11. Reference

### 11.1 Configuration

| Config | Drives | Inheritance |
| --- | --- | --- |
| `dlm.yaml` | The DLM arm; the single source of the sampler, augmentation, BIO-head architecture, gate and optimizer keys | — |
| `ar.yaml` | The AR arm (spec §9.3) | `extends: dlm.yaml` (decoder and output folder only) |
| `baseline_eval.yaml` | The pretrimmed translator at evaluation, with no gate. Its `decoder` picks the pretrimmed translator that `train-slt` trains and the `pretrained_slt` entry that `--text pretrimmed` reads | `extends: dlm.yaml` (gate off) |
| `baseline_train.yaml` | Trains the pretrimmed translator of `decoder` (Mode 1 only, no BIO loss) into `checkpoints/baseline_train_<decoder>/<lang>` | `extends: baseline_eval.yaml` |
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
| `mode_ratios` | mode1 0.45, mode2 0.30, mode3 0.20, mode4 0.05 | The designed mix of window modes, not a measured one. The pretrimmed translator uses mode1 1.0. |
| `jitter.context_s` | 4.0 s | The sampler draws the window edges from Uniform(−4, +4) s around the edges of the anchor unit. The pretrimmed translator uses 0.0. |
| `jitter.cut_range` | [0.15, 0.85] | The relative depth of a Mode-2 cut, drawn uniformly. |
| `augmentation` | `fps` {`min_factor` 0.625, `max_factor` 1.25}, `frame_dropout` 0.15, `body_part_dropout` 0.1 | One recipe for every trained pose model (`moryossef26.yaml` holds a copy): frame-rate resampling, frame dropout and hand dropout on the train inputs. Dev, evaluation and the FSM read the 24 fps grid with no augmentation. |
| `learning_rate` | 1e-4 | One rate for every module, in one optimizer group. S1 uses 1e-3 for its head and `backbone_lr` 1e-3 for its encoder. The pretrimmed translator uses 2e-4. |
| `lambda_bio` | 1.0 | The weight of the BIO loss. |
| `shared_pose_encoder` | false | The architecture of the arm: false is the two-stream arm, true the one-stream arm (§6). The one-stream arm needs `bio.adapter: false`. |
| `membership_gate.detach_omega` | false | true makes the no-text arm of either architecture (§6). |
| `spd.tau_dec` | 0.5 | The one DLM decode threshold. It is a fixed value: do not tune it on dev. |

Values fitted from labels:

| Value | Source |
| --- | --- |
| duration prior | A lognormal fit of the train unit durations of the target corpus. Stage 2 stamps it in its checkpoint, and evaluation decodes with the stamp. The cascades fit it at run time. No config holds it. |
| `bio.class_weights: balanced` | Computed from the label counts when training starts, and logged. |

### 11.2 Artifact folders

Every artifact goes into one folder per kind under `outputs/`, one level deep. The code chooses the folder, except for the boundary sweeps: there the command gives the path (§7.2).

| Folder | Holds | Written by |
| --- | --- | --- |
| `offline/`, `online/` | the events of the rows and their `_scores.json`, by the `when` of the row (the no-text rows too) | `eval.py` |
| `analysis/` | the error analysis of the segmenters (`error_types_*`, `jitter_*`, `mode_ratios_*`) and the boundary sweeps (`boundary_sweep_*`) | `analyze.py --stage segmenter-errors`, `analyze.py --stage boundary-sweep --output` |
| `stability/` | reveal-policy scores (`stability_*`) and every recorded decode (`provisional_*`) | `eval.py --online --stability` |
| `report/` | one folder per product: `results_<lang>_<split>/`, `progress/`, `data/`, `poses/`, `predict/` | `report.py` |

### 11.3 Row artifact names

Every row writes one pair of files under `outputs/offline/` or `outputs/online/`. The fields of the name are the flags of the row (§7), and every field is always present:

```
{when}_{spans}_{decode}_{text}_{corpus}_{split}.json          the events
{when}_{spans}_{decode}_{text}_{corpus}_{split}_scores.json   what this run scored on them
```

| Field | Values | Meaning |
| --- | --- | --- |
| `when` | `offline`, `online` | `--offline` or `--online`: `offline` reads the whole stream, and `online` decides from the past alone. Row 11 is `online`, because it reads the committed spans of the arm's FSM. A row on the spans of a file takes the `when` of the file. |
| `spans` | `gold`, `moryossef`, `s1`, `joint-ar`, `joint-dlm`, `chance` | `--spans`: what cut the video. The spans of an arm carry the arm (`--spans ar` gives `joint-ar`) and its name tokens (`joint-ar-onestream`, §7.3), so the same-span controls of different arms are different files. The untrained controls of §7.1 are `moryossef-released` and `moryossef-random<seed>`. A row on the spans of a file takes the `spans` of the file. |
| `decode` | `argmax`, `viterbi`, `duration`, `none` | `--decode`: the decode that made the spans (§7.1); an arm's spans are `duration`, and `none` is the annotation or the chance spans. |
| `text` | `pretrimmed`, `pretrimmed-dlm`, `ar`, `dlm`, `none` | `--text`: the translator that read the spans (`pretrimmed`: the pretrimmed AR translator or the released file; `pretrimmed-dlm`: the pretrimmed DLM translator); `none` is a row with no text. Every arm other than the two-stream arm of the config (a one-stream arm, a no-text arm or an ablation arm) adds its name tokens (§7.3, §9.1), also to `none`, so it cannot write over the files of another arm. |
| `corpus` | the corpus key | `--language`; a zero-shot row writes `<S>-to-<T>` (`--source-language S`). |

So `offline_moryossef_argmax_pretrimmed_asf_test.json` holds the spans of the Moryossef segmenter, with the argmax decode, read by the pretrimmed translator, offline.

A no-text row has no text scores. Its `_scores.json` also holds `frame`: the frame and run scores of the protocol of Moryossef et al., a mean over the streams, with the same keys offline and online: `bio_*` (signing against no signing, per frame), `bio_iou`, and `phrase_*` (the O, B and I frames, the runs, their counts, and per tIoU threshold `@t` and their mean `_avg`). Offline they come from the whole-video pass of a segmenter (`bio_*` from the most likely tag of each frame, `phrase_*` from the decode of the row); online from the FSM's tags. `frame_metrics_decode` names the source of each key family. An arm offline, the chance spans and the spans of a file have no frame scores (`null`).

### 11.4 Repository map

```
backbones/       the Uni-Sign pose encoder (a graph network over 69 keypoints, ST-GCN)
poses/           pose files: Uni-Sign normalization, pose loading and video_meta.csv, augmentation
data/            records to samples: loader, BIO labels and text targets (windowing), stage-2 sampler, segmenter chunks, batches
models/          model parts: BIO head, gate (Ω), Uni-Sign front end with mT5, DLM decoder, the full model (streaming_slt), checkpoint checks
train/           training loops: arms and pretrimmed translator (slt), S1 (bio_pretrain), losses, epoch loop, several GPUs
moryossef26/     the Moryossef segmenter: model, input, training, whole-video inference
infer/           inference: duration decode, span and commit rules, DLM decode, the FSM (stream), display stability
evaluation/      what eval.py runs: events and gold (predictions), model builds (systems), rows, offline (with the cascade and the chance spans), online, the decode of a segmenter row and its dev monitor (segmenter_eval); the boundary sweep (boundary_sweep)
reporting/       what report.py runs: breakdowns, per-unit outcome tables, figures, audits, debug views
metrics.py       every metric calculation
utils.py         config loading, checkpoint and output folders, device choice
train.py         --stage {smoke-data, train-bio, train-moryossef, train-slt}
eval.py          one row per command: --offline|--online --spans --text [--decode]; --predictions (score a saved row again)
analyze.py       --stage {dataset-summary, segmenter-errors, boundary-sweep}
report.py        {results, progress, data, poses, predict}
prepare_yt25.py  YouTube-SL-25 through SignVerse-2M: shards, pose converter, captions, video_meta.csv, copy check
docs/            membership_gate.md, segmentation_decoding.md, data_pipeline.md, run_real_data.md, implementation_notes.md, literature_notes.md
```
