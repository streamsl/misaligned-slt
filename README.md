# Misaligned SLT

Pose-only, gloss-free, **streaming** sign-language-to-text translation. The system is designed for sentence-boundary **temporal misalignment** (imperfect segmentation). Its robustness must be measured in the controlled and online evaluations.

**Central rule.** Training windows cover the input cases that the streaming loop can produce; their sampling distribution is not identical to live buffers. A truncated visual input never gets a partial text label (premise **P1**). Every segmenter error becomes a window-shape variation, never a label variation (**P2**). Design spec: [`streaming_slt_prompt.md`](streaming_slt_prompt.md). The segmentation→translation coupling: [`docs/membership_gate.md`](docs/membership_gate.md). Decoder model, algorithm and loss: [`docs/segmentation_decoding.md`](docs/segmentation_decoding.md).

**Data.** YouTube-SL-25: ASL (`ase`), Auslan (`asf`), BSL (`bfi`). All targets are English. All poses come from SignVerse-2M at a uniform 24 fps. The full load-time pipeline (caption cleanup, caption-unit construction, coverage checks, de-duplication, pooling) is documented in [`docs/data_pipeline.md`](docs/data_pipeline.md). Read it before you change anything that touches ground truth.

**Target definition.** B marks a timed caption-unit start, I its interior. A unit can contain several linguistic sentences. This evaluates caption-unit localization, not every signed sentence boundary. Source cue timing remains weak supervision.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- **Keep `transformers==4.57.3` (pinned `<5`).** The DLM decoder drives T5/mBART 4.x internals. On 5.x, mT5 ties `lm_head`↔`shared`; the untied Uni-Sign checkpoint then loads silently wrong and BLEU collapses. The code guards the tie, but the mBART DLM path still needs 4.x.
- **Install the metric backends** (`evaluate`, `sacrebleu`, `rouge`, `nltk`) before you trust a results table. A missing backend warns loudly and reports 0.0. BLEURT is optional (`bleurt` + a local `BLEURT-20` checkpoint).
- GPU recommended for training. Apple MPS runs eval fine (device auto-selects `cuda → mps → cpu`; override with `--device`).

## Architecture

```
raw pose buffer → Uni-Sign normalization ─┬→ segmentation pose encoder (S1) → conv stem + banded RoPE encoder → H
                                          │      → BIO classifier → legal paths + train-fitted duration prior
                                          │                            ├→ first-span membership → Ω
                                          │                            └→ Viterbi span → FSM
                                          └→ translator pose encoder (clean translator) → F
                                                 → task prompt + mT5 encoder  ← Ω on the self-attention keys
                                                 → AR / DLM decoder           ← Ω on the cross-attention
```

- **Pose encoders** ([`backbones/`](backbones)): two Uni-Sign 4-part ST-GCNs with separate weights, each pretrained alone and then loaded together, as in TwoStream-SLT (arXiv 2211.01367; there the two streams share one task and read different inputs, here they read one input for two tasks). `bio_pose_encoder` (segmentation branch) starts from S1; `front_end.pose_encoder` (translator) starts from the clean translator. Stage 2 trains both encoders, like every other module, at one rate (1e-5); no freeze key exists. TwoStream-SLT freezes its S3D backbones in its SLT stage, but its SLT input is the same trimmed clips as its SLR stage. Here the stage-2 windows are not caption crops, and the Uni-Sign body box is computed over the whole window, so the translator encoder sees other inputs than in its clean training. With the gate off, the AR arm's step 0 decodes as the clean translator token for token; the DLM arm adds its `[MASK]` row and block decoder on top. [`poses/`](poses) reproduces Uni-Sign normalization byte-exactly.
- **Front end** ([`models/unisign.py`](models/unisign.py)): the translator pose encoder, two LMs (mT5 default; mBART ablation). `MisalignedSLTModel` takes either, so the heads, sampler, and FSM are written once.
- **Decoder** ([`models/block_diffusion.py`](models/block_diffusion.py), [`models/dmax.py`](models/dmax.py)): BD3LM core + DMax (OPUT training; block decode with SPD self-revision over a block KV cache). The decode follows DMax's `decode_uniform` with five documented deviations (`infer/decode.py`). AR and DLM reuse the pretrained Transformer layers. `block_size: 1` keeps masked-position prediction, OPUT and token revision; it does not select the AR baseline. See [the decoder comparison](docs/literature_notes.md#ar-to-dllm-adaptation--bd3lm).
- **BIO head** ([`models/bio_head.py`](models/bio_head.py)): Conv1d stem, RMSNorm and time-aware RoPE, with attention banded to ±`bio_attention_radius_s` (0.375 s). Its features H feed only the BIO classifier; the translator receives no BIO-head features. Reach R = 4 × 0.375 s + (6 ST-GCN + 4 conv-stem frames)/24 = 1.92 s ≤ (K−1) × stride = 2 s, for fixed normalization and evaluation-mode features. This bounds direct neural look-ahead, not future boundary changes: buffer normalization and the global Viterbi path can still change. The three-observation stability test is a check, not a proof of finality.
- **Membership gate** ([`infer/duration_decode.py`](infer/duration_decode.py), [`models/membership_gate.py`](models/membership_gate.py)): a soft crop. Ω_t = log(eps + (1−eps)·clamp(m_t/P, 0, 1)) on valid, uncommitted frames and log(eps) on committed and padded frames, where m is the exact membership of the selected first span and P the probability of that state (closed first span or open span). A row with P < 1e-6, or with no span and no open start, reads the whole window (Ω = 0). One Ω biases the translator's encoder self-attention keys and the decoder cross-attention. A point mass matches a hard crop up to the eps leak only when the span starts right after the task prompt; at a later start the prompt↔span relative-position offset adds a second difference (mT5's relative-position buckets see other prompt-to-span distances than in a crop). Caption gradients reach the segmentation branch through Ω. The same rule runs during training and streaming.
- **Segmentation decoder** ([design](docs/segmentation_decoding.md)): a restricted semi-Markov CRF with a lognormal duration prior fitted on the target language's TRAIN labels (`DurationModel.fit`), legal BIO transitions and exact Viterbi. No decoder weight is tuned on dev. BIO supervision remains frame CE plus Dice. S1 checkpoint selection uses an untuned legal-only monitor.
- **Streaming FSM** ([`infer/stream.py`](infer/stream.py)): first eligible complete span from duration-aware BIO Viterbi, K-stride hysteresis for every terminator (O or B) with one vote history per candidate span, and explicit forced-partial handling. A commit needs only a stable terminator; token confidences are recorded on the events but do not gate a commit. Committed prefixes cannot be selected again.

**Claim boundary.** S1 is a pretrained segmentation module built from existing components. Duration-aware BIO decoding uses established structured inference and can also process Moryossef logits. It enforces valid transitions but does not establish accurate interior boundaries. S1's CE/Dice objective does not use the caption mask. The proposed joint contribution is first-complete conditioning with incomplete-state handling; its accuracy and latency benefit must be measured. See `docs/membership_gate.md` §4.

## Multi-GPU (torchrun)

Prefix any `train.py` command with `torchrun`; change nothing else:

```bash
torchrun --standalone --nproc-per-node=4 train.py --stage train-slt --language "$LANG" --slt-config configs/dlm.yaml
```

- Config `batch_size` is the GLOBAL batch, split across ranks. A non-divisible batch is a hard error.
- `grad_accum_steps` raises the EFFECTIVE batch to `batch_size * grad_accum_steps` at the activation memory of `batch_size` alone: that many micro-batches make one optimizer step. Each micro-batch is scaled by its group size before backward, so the accumulated gradient is the MEAN over the group and a short final group is a true mean of its own size. That equals one batch of the combined size when the micro-batches hold equally many supervised rows; length bucketing makes the row count vary, so treat it as a close approximation, not an identity. The scheduler counts optimizer steps, so warmup and decay keep the shape they were tuned with, and so does every reported `step` — the progress bar, `history.csv` and wandb — as in the HF Trainer, so `grad_accum_steps: k` shows k-times fewer steps for the same data.
- `mixed_precision: auto` picks bf16 on compute capability ≥ 8. Multi-GPU + fp16 is refused (per-rank scaler drift).
- `latest.pt` is a full resumable snapshot, written every epoch. Continue an interrupted run of the same architecture and settings with `--resume`. Re-running without `--resume` over an existing `latest.pt` is refused.
- CUDA compiles the membership recurrence on first use. Judge speed after warmup. Before another full run after a slowdown, run `python tests/test_membership_runtime.py` on the training GPU. This data-free check measures gate forward/backward time and checks gradients; it writes `outputs/membership_runtime.json`. It does not estimate total epoch time.
- Eval and analysis stay single-process. Offline cascade translation honors `--batch-size`; streaming processes one active buffer per stride. Gradient averaging is explicit, not DDP ([`train/distributed.py`](train/distributed.py) explains why).

## The experiment sequence

Two blocks. **Stage A** (segmentation pretraining) runs once, pooled over all three languages. **Stage B** runs once per target language, in two phases: **TRAIN → EVALUATE**. There is no measurement phase: `configs/inference.yaml` holds fixed design constants (see [Configuration](#configuration)).

|                                   | **Stage A — language-agnostic**               | **Stage B — per language**                                  |
| --------------------------------- | --------------------------------------------- | ----------------------------------------------------------- |
| Runs                              | once, for all languages                       | once per target language                                    |
| `--language`                      | **refused** (a pool has no target)            | required                                                    |
| Produces                          | `checkpoints/{moryossef,bio_s1}/multi_<pool>` | `checkpoints/{baseline_train,ar,dlm}/$LANG`, `outputs/*_$LANG_*` |
| Reads `data.yaml pretrained_slt`? | no — released warm starts only                | yes (`utils.resolve_pretrained`)                            |

Segmentation training reads its explicit warm start, so the B2b clean-translator re-root cannot reach S1. S1 (A2) and the clean baseline (B2a) have no ordering dependency: train them in parallel. The arms (B3) need both. `--language "$LANG"` re-points the dataset and every `${language}` path; no config edits.

Before Stage A, fix the annotation rules in `configs/data.yaml`. The shared loader applies cleaning, cue grouping, comma rendering, selective capitalization and the 10% punctuation threshold to train, dev and test. The capitalization lexicon is fitted on train only and reused for dev/test. Record excluded-video counts and annotation fingerprints; keep these rules fixed while comparing systems. See [the data protocol](docs/data_pipeline.md#3-caption-unit-construction--reconstruct_sentences).

That lexicon is pooled over every corpus in `data.yaml` sharing the target language, so it is released like the split CSV: the first stage run on a machine holding the whole pool writes `data/youtube-sl-25/case_lexicon.en_XX.json`. Copy it alongside the corpus to any machine that holds one language, or that machine cannot render the same references.

```bash
set -e  # Stop the runbook if a training or evaluation command fails.
# ═══ STAGE A — run ONCE. Both segmenters train on the SAME multilingual pool ([ase, asf, bfi]). ═══
# ── A0. One-time data + checkpoints ──
#   Warm start: checkpoints/openasl_pose_only_slt.pth.
#   Prepare poses and captions for all three pool languages:
#   python prepare_yt25.py --stage all --languages ase asf bfi

# ── A1. External Moryossef segmenter — the RQ2 cascade floor ──
#   Their 50 landmarks (no face) under their own normalisation → UNet → RoPE BIO: it reproduces Moryossef 2026
#   rather than adapting it, and shares no input representation with our head, so it stays an independent baseline.
#   Everything else is S1's (A2): the SAME pool and temperature, chunk tiling (fixed chunks of (num_frames - 1)/24 s, which
#   under fps augmentation load the native span that resamples to num_frames frames, as the release does;
#   moryossef26/trainer.py builds data/chunks.py ChunkDataset with release=), augmentation block, BIO labels, dev chunks,
#   monitor and best-epoch floor. So the cascade compares models and input contracts, not data or recipe.
#   The offline whole-video pass overlap-stitches its 1024-frame RoPE windows, as S1 does.
#   Pooled checkpoints live in ${corpus}-named directories (multi_ase-asf-bfi); utils.checkpoint_dir
#   is the one resolver for every reader and writer. Never build a segmenter checkpoint path by hand.
python train.py --stage train-moryossef        # -> checkpoints/moryossef/multi_ase-asf-bfi/model.pt

# ── A2. S1 — segmentation pretraining (pose encoder + BIO head, pooled) ──
#   Sentence boundaries are prosodic and prosody is shared across signed languages, so segmentation pretrains
#   on the pool; translation stays monolingual in Stage B. The encoder trains too; stage 2 loads encoder AND head
#   via bio_head_init into its SEGMENTATION branch (the translator keeps its own encoder).
#   Rates (bio_pretrain.yaml sets both; S1 reads no rate from dlm.yaml): BIO head learning_rate 2e-4 (random init),
#   pose encoder backbone_lr 6e-5 (released Uni-Sign init). The checkpoint stamps learning_rate, backbone_lr and
#   augmentation, so --resume refuses a drift. backbone_lr is an S1-only key.
#   Whole-input chunks, BIO only (data/chunks.py): each epoch tiles every record (a clean segment of a video,
#   docs/data_pipeline.md §5f) from a random phase with chunk
#   lengths U(L_min, C), L_min = δ + K·stride = 3.5 s, C = pretrain_geometry.buffer_cap_s = 40 s. Full frame coverage is not guaranteed: each re-draw (chunk lengths, pool rotation) changes the chunk count by a few
#   percent, but the epoch size is fixed at the first draw, so a longer draw drops a random tail of its shuffled chunks
#   and a shorter one repeats its head. The first and last chunk are clipped by the record ends, and a record shorter
#   than C can still have several chunks; no window modes, jitter or anchors. A chunk edge falls at an arbitrary time, so the
#   cuts are not placed from caption boundaries. Augmentation (train chunks only; dev chunks are never augmented) is
#   the Moryossef 2026 release recipe, one block shared with moryossef26.yaml (poses/augmentation.py): hand dropout
#   0.1 per hand on the raw keypoints, then fps resampling 15-30 and frame dropout 0.15; BIO labels come from the kept
#   timestamps. No rotation or other spatial transform. Only the two segmenters augment: stage 2 and the clean floor
#   use none.
#   The head attention is banded (dlm.yaml bio_attention_radius_s); the checkpoint stamps the band, and stage 2,
#   eval --segmenter-arch s1 and an S1 warm start refuse a mismatch (train.bio_pretrain.require_attention_radius).
#   The pool is temperature-flattened and balanced by SUB-sampling with per-epoch rotation — nothing is replicated,
#   nothing is permanently dropped. The pool balances RECORDS (clean segments); the checkpoint stamps the hours each language
#   contributes per epoch (meta.pretrain_hours). Dev is a balanced fixed sub-sample, its chunks drawn once; test is
#   pooled as-is. train-bio refuses a chunk cap below the deployed inference.yaml buffer_cap_s.
#   Checkpoints stamp their pool (meta.pretrain_pool) and every loader refuses a pool-KEY mismatch. The key alone
#   does not make A1 and A2 comparable: train them from ONE code state, then check that the two checkpoints
#   agree on meta.pretrain_mix and meta.annotation_fingerprint before quoting any cascade result.
#   Monitor = val_phrase_tiou_f1 (complete-span F1@0.5 over the fixed dev chunks, train.helpers.evaluate_bio_chunks;
#   the Moryossef segmenter uses the same function). Compare checkpoints with --segmenter-eval, never by monitor value.
#   Best-epoch floor (both segmenters, pooled runs): no epoch before one full pool rotation (cycle_epochs, computed
#   from the record counts and printed at run time) can become the best checkpoint or count toward early-stopping patience; the checkpoint stamps best_epoch_floor,
#   and a run with fewer epochs than the floor is refused. Batch 32 on 40 s chunks is not yet measured on GPU: if it
#   runs out of memory, lower batch_size and raise grad_accum_steps.
python train.py --stage train-bio              # -> checkpoints/bio_s1/multi_ase-asf-bfi/model.pt

# ═══ STAGE B — per target language. TRAIN (B2, B3) → EVALUATE (B4, B5, B6). ═══
# WHAT A CHANGE INVALIDATES (inference.yaml holds one fixed value per constant; no stage writes it):
#   New joint architecture/loss/sampler -> fresh AR and DLM training (eval and --resume refuse another
#     architecture meta), followed by their RQ1/RQ2 rows.
#   New S1 checkpoint (recipe, pool or bio_attention_radius_s) -> S1 --segmenter-eval, the S1 cascade rows 5 and 9,
#     and fresh AR and DLM (stage 2 loads S1; a band mismatch is refused).
#   New clean translator -> every --method baseline row (RQ1 baseline; RQ2 rows 1, 3, 5, 7, 8, 9) and fresh AR and
#     DLM (their translator starts from it through the B2b re-root).
#   Changed learning rates or augmentation -> fresh training of that stage and of every stage that loads it. Every
#     train-slt checkpoint (arms and clean floor) stamps learning_rate and augmentation; S1 stamps learning_rate,
#     backbone_lr and augmentation; Moryossef stamps learning_rate and augmentation; both segmenters also stamp
#     best_epoch_floor. So --resume refuses a drift.
#     train-slt refuses any augmentation block. The two segmenters share one augmentation block: change both together.
#   Changed pose conversion (the converter in prepare_yt25.py) or poses.min_extra_person_motion (masked_runs apply it at conversion)
#     -> python prepare_yt25.py --stage all --overwrite --languages ase asf bfi (convert, person-counts and subs in
#     order; it needs the shard tars, so re-download any that --delete-tars removed; "Required reruns", step 1), then fresh S1, Moryossef, clean baselines and arms, and every evaluation row. The
#     annotation fingerprint does not hash pose values, so no guard detects stale .npy files or a checkpoint trained on
#     them. It does hash every record id, view duration and unit, so when new masked_runs or empty_runs change the
#     clean segments, every span, events and RQ1 file and clean-baseline warm start made before fails the
#     fingerprint check.
#   Changed poses.handless_unit_share -> no re-conversion (the loader applies it to handless_runs), but new labels:
#     fresh S1, Moryossef, clean baselines and arms, and every evaluation row (the fingerprint hashes every unit's
#     reliable flag, so old span, events and RQ1 files and warm starts fail the fingerprint check).
#   New Moryossef checkpoint, or a change to its whole-video inference -> Moryossef --segmenter-eval, segmenter-infer
#     and segmenter-errors (B4b), and the Moryossef cascade rows 3, 4 and 8.
#   Changed delta, K or stride -> S1 (chunk lengths U(delta + K x stride, C)) and the arms. Every train-slt checkpoint
#     stamps delta; eval refuses a GATED checkpoint whose stamp differs from inference.yaml (the clean floor never
#     reads delta). Keep the band reach <= (K-1) x stride and jitter.context_s = (K+1) x stride.
#   Changed cap -> S1 (C must stay >= the cap), the clean baseline and the arms. Both translators clamp their windows
#     to the cap and stamp it. Eval and the stage-2 warm start refuse a stamp that differs.
#   Changed minimum span (Lambda_min) -> the arms (sampler target, gate and FSM share it), the clean baseline (its
#     sampler target; it stamps Lambda_min and eval refuses a stamp that differs) and every RQ2 event row.
#   Changed inference.yaml translation block -> evaluation only; not a training input. max_text_tokens reaches every
#     RQ1/RQ2 decode. The DLM decode threshold has one source, dlm.yaml spd.tau_dec: stage 2 reads it in the Mode-2a
#     decodes and eval reads it from the method config, so a change means a fresh DLM arm.
#   Changed ground truth -> every dependent training/evaluation stage (the duration prior refits from train labels;
#     the stage-2 warm start refuses a clean translator with another annotation_fingerprint).
LANG=ase      # ase (ASL) | asf (Auslan) | bfi (BSL)
python train.py --stage smoke-data --language "$LANG" --split train --num-samples 64

# ── B2. TRAIN the clean floor — the translator init for every later stage ──
#   Faithful Uni-Sign SLT transfer on a caption-trimmed training view: mode1-only, zero context band, lambda_bio 0,
#   everything trainable in one optimizer group (learning_rate 2e-4), no pose augmentation. Windows clamp to
#   inference.yaml buffer_cap_s; configs/baseline_train.yaml documents why the floor and the arms must train under
#   one cap. The checkpoint stamps that cap. Eval and the stage-2 warm start refuse a clean baseline stamped with
#   another cap, so retrain a baseline from an older cap.
#   ORDER: no dependency on S1 or on any measurement; run it in parallel with Stage A.
#   The reported clean number is NOT a separate step: it is the (0,0) cell of the B5 baseline sweep.
#
# B2a — train (writes checkpoints/baseline_train/$LANG/model.pt)
python train.py --stage train-slt --language "$LANG" --slt-config configs/baseline_train.yaml
#
# B2b — accept, then re-root. Acceptance costs nothing: best.json's val_translation_bleu4 IS dev BLEU;
#   several times the zero-shot ~1 = pass. Watch val_translation_len_ratio for degenerate lengths.
#   Then edit configs/data.yaml: languages.$LANG.pretrained_slt: checkpoints/baseline_train/$LANG/model.pt.
#   That one edit re-points the stage-2 translator (B3) and every later --method baseline eval, so the floor and the
#   arms share one translation init. The stage-2 guard in train-slt checks this warm start. It refuses the released
#   *_pose_only_slt.pth while checkpoints/baseline_train/$LANG/model.pt exists. It refuses a local checkpoint unless
#   it stamps architecture clean_translation, language $LANG, the current annotations and the inference.yaml cap and
#   Λ_min (train.slt.require_translator_init; eval applies the same constants rule). B2a itself always starts from the
#   released file (baseline_train.yaml checkpoint.from_pretrained) and refuses a trained warm start, so the re-root never
#   feeds a baseline retrain: a floor from an older cap or older labels is simply retrained with B2a, then re-rooted.
#   The re-root does NOT re-point Stage A (pooled S1 keeps the released warm start;
#   S1's segmentation-adapted encoder goes only into the arms' segmentation branch).

# ── B3. TRAIN the arms — stage-2 fine-tune under the gate ──
#   Requires A2 (S1) and the B2b re-root. Two pose encoders: the segmentation branch (bio_pose_encoder + BIO head)
#   loads S1; the translator loads the clean translator and receives no BIO-head features, so with the gate off the AR
#   arm's step 0 is the clean translator exactly (the DLM arm adds its [MASK] row and block decoder). Ω (the soft crop,
#   see Architecture) is the only coupling. Both encoders read the same sampled window (the window modes below);
#   only S1 (A2) trains on whole-input chunks.
#   Requires a trained S1 at checkpoint.bio_head_init (train-slt refuses a segmentation branch without one), so the
#   gate couples to a trained head from step 0.
#   Optimization: EVERYTHING trains in one optimizer group at learning_rate = 1e-5: both pose encoders (the translator's
#   front_end.pose_encoder, pose_proj included, and bio_pose_encoder), the BIO head, the LM and the decoder. No freeze
#   key exists. lambda_bio 1.0. Why no freeze: the translator ST-GCN is under 1 % of the parameters, and stage-2
#   windows compute the Uni-Sign body box over the whole window, not over a caption crop as in the clean translator's
#   training. Why 1e-5: a judgment call that matches TwoStream-SLT's translator rate. TwoStream-SLT and MMTLB use
#   1e-5 only for the MLP + translation network and 1e-3 for their visual heads, so they do not support 1e-5 for the
#   pose encoders or the BIO head, and no measurement on this model supports it: watch dev BIO loss and translation CE
#   over epochs 0-3.
#   BatchNorm: the translator encoder's BatchNorm layers run in train mode on the gradient paths. The no-grad decodes
#   that must match inference re-extract the translator features in eval mode: the DLM OPUT rollout
#   (MisalignedSLTModel.eval_encode_memory_fn) and, in both arms, the confidence-bound selection decode and its
#   full-view teacher (_confidence_bound). Cost: one extra translator ST-GCN forward per OPUT group and per CB call.
#   Augmentation: none (dlm.yaml augmentation: null; train-slt refuses a non-null block, and the checkpoint stamps
#   it). The arms and the clean floor then differ only by the method, inference never augments, and the CB student
#   and teacher get the same preprocessing. Every stage-2 window is on the native 24 fps grid, as in the FSM and eval,
#   so delta and Λ_min are plain frame counts (12 and 12).
#   The checkpoint stamps learning_rate and augmentation, so --resume refuses a drift.
#   train-slt refuses lambda_bio > 0 without a segmentation branch and lambda_bio = 0 with one.
#   The trainer scores the initialization on dev as epoch 0, so the selected checkpoint (monitor val_joint_score) is
#   never worse on dev than its start. model.pt stamps best_epoch; a best_epoch of 0 prints a warning (the shipped model
#   is the initialization), and report.py progress prints the stamp per arm.
#   Text routing follows the model's OWN predicted state; GT decides only which text exists (P1):
#     complete GT target (Mode 1/3) contained in the predicted first span (delta tolerance at each end) -> the full-text loss
#       trains the translator and, through Ω, the segmentation branch. The rule has no upper limit on overshoot: a
#       predicted span that merges the target unit with the next unit is covered, and the translator learns to write only
#       the first unit's text from it (a deliberate choice, see Key design decisions);
#     not covered -> the same loss as a CRITIC: translator frozen, its features detached, so only Ω and the
#       segmentation branch learn (logged: text_covered_rate);
#     GT unit right-truncated with a full view AND a predicted CLOSED span that starts inside it (a premature commit
#       the FSM would make) -> the confidence-bound term with Ω detached (logged: cb_premature_rate). It is an
#       UNLIKELIHOOD term, -log(1 - p(t)), on the truncated decode's OWN token t, only at slots where that decode is
#       confident (> tau_cb 0.75) and differs from a full-evidence decode that equals the reference. It names no target
#       token, so the truncated input gets no text target (P1). It only lowers the token confidence that the reveal
#       policies read (B6b); the FSM commit reads no confidence;
#     right-truncated GT + predicted open span -> no text loss. A complete GT row can still train Ω through the critic.
#   Covered and critic rows are separate per-token means, summed (logged: text_critic_loss), so the translator's step
#   does not scale with the covered share of the batch. Through Ω, each group's gradient to the segmentation branch is
#   scaled by its token share n_g/N (N = valid target tokens of the groups that run), so every supervised token pulls
#   the segmentation branch equally; the forward values and the translator's gradient do not change. A critic row runs only when Ω can learn from it:
#     membership_gate.detach_omega: true (the ablation "text does not train segmentation") runs no critic rows;
#     a row whose Ω fell back to the whole window (P < 1e-6, which includes a row with no decoded span and no open
#     start) is never a critic row. MembershipGate marks such a row 'uncertain' in its stats.
#   Sampler: the window modes are a text-coverage schedule with a fixed DESIGNED mix (mode_ratios mode1 0.45, mode2
#   0.30, mode3 0.20, mode4 0.05); labels come from the window content; window edges are drawn from a flat band
#   Uniform(-context_s, +context_s), jitter.context_s = 4 s = (K+1) x stride; Mode-2 cut depths are uniform in
#   jitter.cut_range [0.15, 0.85]. A mode-3 window spans the anchor and its successor. No measured jitter or
#   measured mode mix is ever a training input (B4b is analysis only).
#   Post-commit shift (WindowSampler.materialize): when a quarantined unit ends inside a window before the text
#   target starts, the left edge moves to that unit's end - delta, never to the left, as the FSM's post-commit cut
#   would. The window is then relabelled from the new edge, and a Mode-2a full view takes the same shift.
#   Interrupted runs of this architecture use --resume. Watch val_phrase_tiou_f1, boundary precision/recall,
#   val_translation_bleu4, text_covered_rate and cb_premature_rate together.
python train.py --stage train-slt --language "$LANG" --slt-config configs/ar.yaml    # gated AR de-risk (§9.3)
python train.py --stage train-slt --language "$LANG" --slt-config configs/dlm.yaml   # DLM -> checkpoints/dlm/$LANG

# ── B4. EVALUATE — acceptance checks and diagnostics. Nothing here parameterizes training ──
# B4a — portable decoder control: four decoder cells from the same logits, no extra model training. The duration
#   prior is fitted on the target language's train labels, so no setting is tuned before test.
for S in moryossef s1; do
    for D in plain duration; do
        python eval.py --segmenter-eval --segmenter-arch "$S" --segmenter-decode "$D" \
            --language "$LANG" --split test --allow-test --output outputs/segmenter_eval_${S}_${D}_${LANG}_test.json
    done
done
#   UNTRAINED CONTROLS (optional, read as a PAIR and never as a baseline). The released DGS weights with nothing
#   trained, against the same architecture with nothing transferred; both read the arm's fitted release_stats, so
#   they isolate training from representation. Omit --output: the init token must stay in the filename.
#     python eval.py --segmenter-eval --segmenter-arch moryossef --segmenter-init released \
#         --language "$LANG" --split test --allow-test
#     for K in 0 1 2; do python eval.py --segmenter-eval --segmenter-arch moryossef --segmenter-init random \
#         --seed $K --language "$LANG" --split test --allow-test; done
#   Declare the training data, initialization, normalization and trained context beside each model.
#   Different encoders and data views make this a system comparison, not a controlled test of one layer.
#   Use the three-condition S1 continued-training study below to test the value of pretraining.
#   TWO PROTOCOLS, ONE DECODE: segmenter-eval also scores its spans through the RQ2 code path itself
#   (`rq2_protocol` in the JSON: caption-unit gold, ignore-region filtering, F1 of macro P/R) and prints both
#   numbers per threshold. QUOTE the rq2-protocol number wherever localization is compared across tables — it is
#   identical-by-construction to the cascade rows' segmentation block for the same checkpoint and decode; the
#   Moryossef-protocol block stays for cross-paper comparability only.
#
# B4b — segmenter-error analysis: the analysis that motivates the window-mode design. It is NOT a training input:
#   no training stage reads its files. Stage 2 trains on the designed mode mix and context band (dlm.yaml), S1 on
#   whole-input chunks.
#   It counts matched / over-segmentation / under-segmentation / skipped / phantom events (the Moryossef 2020
#   mapping), reports the boundary offsets (Delta_head, Delta_tail), the Mode-2 cut
#   positions inside over-segmented units, and the mode mix these events imply. It is the paper's evidence that real
#   segmenters produce the window-mode event types. The implied mode1 share is small, which is why the designed mix
#   gives complete units (mode1 + mode3) 65 %.
#   Outputs (analysis only): outputs/a_jitter_<arch>_<lang>_<split>.json, a_mode_ratios_*, a_error_taxonomy_*.
#   On test it needs --allow-test (the taxonomy describes the same split as the results tables):
python analyze.py --stage segmenter-infer --segmenter-decode duration --language "$LANG" --split test --allow-test \
    --output outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json
python analyze.py --stage segmenter-errors --language "$LANG" --split test --allow-test \
    --predictions outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json
#   Guard: segmenter-errors refuses spans whose stamped architecture or BIO decoder differs from this run.

# ── B5. RQ1 — controlled boundary sensitivity ──
#   All three arms run, deliberately. RQ1 is the only CONTROLLED robustness evidence: the baseline row shows the
#   problem; the arms' rows test the central claim. "The arms process misaligned input differently" is the
#   hypothesis under test, not a reason to exempt them; the gate's window-skipping is handled by the
#   (decoded-only, skip-rate) reading below. The baseline's (0,0) TEST cell is the reported clean anchor.
#   Context-appetite diagnostic: --severity-mode absolute
#   --severity-grid-head 0 --severity-grid-tail 0,0.5,1,1.5,2,3,4,6. Hypothesis, not yet measured on the two-stream
#   arms: the span-trained baseline falls with added tail context and the buffer-trained arms rise, because the
#   curve's sign follows the training view.
GRID='--severity-grid-head=-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3 --severity-grid-tail=-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3'
for M in baseline ar dlm; do python eval.py --rq 1 --method $M --language "$LANG" --split test --allow-test \
    $GRID --output outputs/rq1_${M}_${LANG}.json; done
#   ABLATION — one training ablation, evaluated as trained (never an eval-time switch). The clean baseline already
#   doubles as the no-misalignment-training ablation at zero extra cost.
#   "Text improves segmentation": membership_gate.detach_omega: true (the segmentation branch learns from BIO only).
#   Write a method config that extends dlm.yaml and sets only that key, train it like the DLM arm, and pass it to
#   eval as --method-config. Its stem enters the translator token (dlm-<stem>), so it never overwrites the arm's
#   events, scores or stability file, and report.py's closed vocabulary keeps it out of the main results table.
#   Read it under the FSM with --no-translate against row 10 with --no-translate (see "How to read the ladder").
#   The confidence-bound term has no CB-off arm. Its effect is read from the stage-2 training log: cb_premature_rate
#   (Mode-2a rows with a predicted premature commit) and cb_active_count (slots where the term fires).
#   Other controls: --segmenter-decode plain and Moryossef --segmenter-init released/random (B4a), and AR vs DLM
#   (every joint row).

# ── B6. RQ2 — end-to-end DVC. Table order IS run order: block A decides offline, block B decides online. ──
#   Inside each block: oracle spans, then the cascades (Moryossef26, then S1), then the joint model.
#   A stream is one clean segment of a source video (docs/data_pipeline.md §5f). Each row decodes every stream alone
#   on its own clock, and every span file, events file and score uses the source video id and source time.
#   Every row writes outputs/rq2_{when}_{spans}_{decode}_{translator}_${LANG}_test.json and a _scores.json beside it.

# ── A. OFFLINE — the whole video is available. ──
#   A0. Span writers. Every offline cascade row reads one of these files, and each one stamps who cut the video.
python eval.py --emit-gold-segments outputs/gold_${LANG}_test.json --language "$LANG" --split test
python analyze.py --stage segmenter-infer --segmenter-decode duration --segmenter-arch moryossef --language "$LANG" --split test --allow-test \
    --output outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json
python analyze.py --stage segmenter-infer --segmenter-arch s1 --language "$LANG" --split test --allow-test \
    --output outputs/segmenter_predictions_s1_duration_${LANG}_test.json
#   Rows 1-2 — oracle spans. What each translator scores when segmentation is free.
python eval.py --rq 2 --segments outputs/gold_${LANG}_test.json --method baseline --language "$LANG" --split test --allow-test  # row 1
python eval.py --rq 2 --segments outputs/gold_${LANG}_test.json --method ar       --language "$LANG" --split test --allow-test  # row 2 (AR)
python eval.py --rq 2 --segments outputs/gold_${LANG}_test.json --method dlm      --language "$LANG" --split test --allow-test  # row 2 (DLM)
#   Rows 3-4 — the external segmenter's spans.
python eval.py --rq 2 --segments outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json --method baseline --language "$LANG" --split test --allow-test  # row 3
python eval.py --rq 2 --segments outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json --method ar       --language "$LANG" --split test --allow-test  # row 4 (AR)
python eval.py --rq 2 --segments outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json --method dlm      --language "$LANG" --split test --allow-test  # row 4 (DLM)
#   Row 5 — our standalone segmenter's spans, same clean translator: the matched-segmenter cascade floor.
python eval.py --rq 2 --segments outputs/segmenter_predictions_s1_duration_${LANG}_test.json --method baseline --language "$LANG" --split test --allow-test  # row 5
#   Row 6 — the joint model cuts the video itself. Its head segments each WHOLE stream in overlap-stitched chunks at the
#   live inference.yaml cap (eval refuses an arm stamped with another cap), each chunk pose-normalized on its own
#   frames, and one duration Viterbi pass decodes the stitched logits. It drops proposals shorter than Λ_min, as the
#   FSM does, and stamps that floor as generation_min_span_frames. The --segments cascade rows take external spans
#   and stamp no floor.
#   It then captions each proposal in the window [start - delta, end + 1 frame] under a point-mass Ω on the proposal.
#   No sampling at eval: deterministic.
python eval.py --rq 2 --offline --method ar  --language "$LANG" --split test --allow-test   # row 6 (AR)
python eval.py --rq 2 --offline --method dlm --language "$LANG" --split test --allow-test   # row 6 (DLM)
#   Row 7 — same-span control: row 6's own spans, read by the clean translator. This is the only OFFLINE pair
#   that moves the translator alone at the joint model's spans, so it is what separates the two (row 11 is the
#   online pair).
python eval.py --rq 2 --segments outputs/rq2_offline_joint-ar_duration_ar_${LANG}_test.json   --method baseline --language "$LANG" --split test --allow-test  # row 7 (AR)
python eval.py --rq 2 --segments outputs/rq2_offline_joint-dlm_duration_dlm_${LANG}_test.json --method baseline --language "$LANG" --split test --allow-test  # row 7 (DLM)

# ── B. ONLINE — the FSM must decide from the past alone. ──
#   Rows 8-9 — online cascades under the same streaming policy as the arms. The segmenter proposes spans; the shared
#   FSM commits a span when its terminator is stable; the clean translator encodes each committed crop
#   [start, terminator + 1), so the crop holds the terminator frame when the buffer has it. Rows 8-10 are only
#   comparable when all three read one inference.yaml, so run them after the arms train, with the same inference.yaml.
#   --checkpoint names the clean translator; eval checks its stamps against inference.yaml.
#   The segmenter comes from --moryossef-config (row 8) or --bio-config (row 9). Both decode with the target
#   language's train-fitted prior.
python eval.py --rq 2 --stream --segmenter-arch moryossef --method baseline \
    --checkpoint checkpoints/baseline_train/"$LANG"/model.pt \
    --language "$LANG" --split test --allow-test   # row 8
python eval.py --rq 2 --stream --segmenter-arch s1 --method baseline \
    --checkpoint checkpoints/baseline_train/"$LANG"/model.pt \
    --language "$LANG" --split test --allow-test   # row 9
#   Row 10 — the deployed system. --stability replays the reveal policies over the same decodes (B6b reads them).
python eval.py --rq 2 --stream --stability --method ar  --language "$LANG" --split test --allow-test  # row 10 (AR)
python eval.py --rq 2 --stream --stability --method dlm --language "$LANG" --split test --allow-test  # row 10 (DLM)
#   Row 11 — online same-span control: row 10's committed spans, re-translated by the clean translator. The crop is
#   [start, end + 1 frame], so every frame it reads was in the buffer at commit time (a normal commit needs the
#   terminator). One exception: a cap-forced cut of an open span ends at the buffer end, so its crop reads one frame
#   more. The crop uses the span normalization of rows 8-9. The events keep row 10's commit times and forced
#   flags, and the row keeps when=online. So 11 vs 10 moves only the translator conditioning (clean translator on a
#   hard crop against the joint arm's soft crop over the buffer) at the same committed spans.
python eval.py --rq 2 --segments outputs/rq2_online_joint-ar_duration_ar_${LANG}_test.json   --method baseline --language "$LANG" --split test --allow-test  # row 11 (AR)
python eval.py --rq 2 --segments outputs/rq2_online_joint-dlm_duration_dlm_${LANG}_test.json --method baseline --language "$LANG" --split test --allow-test  # row 11 (DLM)

# ── B6b. Display stability — how much earlier could text appear, and at what cost? ──
#   Rows 10/10' already wrote outputs/stability_<translator>_<lang>_<split>.json: --stability replays the
#   stable-prefix policies (commit_only, agreement_nK, confidence_nK_tauT, agreement_nK_confidence_tauT) over the
#   per-stride decodes the FSM already computed. No extra decoding; all policies are monotonic, so the trade is latency
#   against prematurely frozen tokens. This step adds no run.

# ── B6c. System error report — WHERE the deployed system fails, from the events B6 already wrote ──
#   report.py results: per gold sentence {matched, merged, split, missed} and per event {matched, merger, fragment,
#   phantom} (mutually exclusive), rule-selected case studies (never cherry-picked), and every system re-scored under
#   ONE protocol on the SAME gold index (stable gold_id join), with per-system per-gold outcome tables and
#   PAIRED-bootstrap significance vs --reference (mean Δ, 95% CI, p; the bootstrap resamples source VIDEOS: one row on
#   DVC BLEU-4 per video, one row on localized BLEU-4 with the units of one video together) plus the sentence-level flip table (fixed / broken / better / worse). Any RQ2-schema events file
#   is a system: streaming, offline, every cascade row, every ablation — a claimed difference without its CI does not
#   go in the paper. --rq1-file adds the confidence panel, --stability-file the reveal-policy panel.
#   -> outputs/report/<lang>_<split>/{results.md, fig/*.png, tables/*.csv}
python report.py results --language "$LANG" --split test --reference stream \
    --events stream=outputs/rq2_online_joint-dlm_duration_dlm_${LANG}_test.json \
    --events offline=outputs/rq2_offline_joint-dlm_duration_dlm_${LANG}_test.json \
    --events cascade=outputs/rq2_offline_moryossef_duration_clean_${LANG}_test.json
#   Or, over every finished row in outputs/ at once, with no --events list to keep in sync:
python report.py progress --language "$LANG"
#   CORPUS AUDIT — what every preprocessing rule removed or repaired, per split, counted by the loader itself
#   (videos: MT captions, no caption, wrong language, scrolling track, pose coverage, dedup, all-quarantined;
#   cues: rolling duplicates merged, overlaps clamped, end straddlers quarantined; visual cuts: videos cut, clean
#   segments, reliable units lost, frames in masked and empty runs and cut; units with no visible hands quarantined;
#   frames: no detected body).
#   Visual cuts: the video_meta.csv columns masked_runs (a second real body is present and its arms move) and
#   empty_runs (no real body) hold frame runs. The loader cuts every kept video at them, and each clean segment is a
#   record and a stream; a unit inside a run is gone, and a unit across a segment edge is quarantined (reliable=False:
#   no target, no anchor, no gold, UNK on every frame). In a frame with 2 or more real bodies, a hand that no wrist
#   of the kept body claims is zero. One rule covers every language/split, with no video or channel exceptions. These
#   detector-based rules can remove valid frames; they do not prove who is signing.
#   No visible hands: handless_runs (12+ frames where the kept body shows no hand) quarantines a unit with at least
#   poses.handless_unit_share (0.9) of its frames inside them; no frame is cut.
#   Fill video_meta.csv (person counts, masked_runs, empty_runs, frame size) from the source archives, and
#   handless_runs from the pose files, without changing pose files or captions. It measures only rows with a blank
#   value and reads only consolidated archives:
python prepare_yt25.py --stage person-counts --languages ase asf bfi
#   After rebuilding metadata, restore caption provenance without replacing existing captions:
python prepare_yt25.py --stage subs --languages ase asf bfi
#   --stage all includes both steps. Existing NPY files are preserved; rebuilding metadata cannot extend a
#   short source pose timeline. undetected_ratio (a diagnostic) reads source person counts, not a signing-activity detector.
#   Then write the actual per-rule counts to outputs/report/data_audit_<lang>.{md,json}:
python report.py data --language "$LANG"
```

**RQ1 design.** One grid, one corpus, three arms. Normalize each curve to its own (0,0) cell; report the intercept separately. Use `--severity-grid-head=…`/`--severity-grid-tail=…` (with `=` for negative-leading lists); the sweep is their full product.

- **Gated arms: read (decoded-only, skip-rate) pairs, not raw corpus BLEU.** A Δ_head > 0 window has no `B`; the FSM skips that state by design, and force-decoding it collapses the cell's corpus BLEU through the brevity penalty. The table therefore reports `gate_skip_rate` and `text_metrics_decoded_only` beside `text_metrics`. Never plot decoded-only alone: it conditions on a shrinking, easier subset.
- Use full-reference BLEU only for evidence-complete cells (head ≤ 0, tail ≥ 0). Cells that remove target evidence test the skip/confidence policy, not translation quality.

**RQ2 ladder** — two blocks. Block A decides OFFLINE, with the whole video available; block B decides ONLINE, from the past alone. Inside each block the rows run oracle spans, then the cascades (external segmenter, then ours), then the joint model, then its same-span control (row 7 offline, row 11 online). A row is never compared across blocks except through row 10 vs row 6, which is the access delta.

| # | Spans from | Translator | Command | Artifact |
| --- | --- | --- | --- | --- |
| | **A. Offline — the whole video is available** | | | |
| 1 | oracle (the annotation) | clean AR | `--segments gold_…` `--method baseline` | `rq2_offline_gold_none_clean_…` |
| 2 | oracle (the annotation) | joint AR, joint DLM | `--segments gold_…` `--method ar/dlm` | `rq2_offline_gold_none_{ar,dlm}_…` |
| 3 | Moryossef26, duration BIO | clean AR | `--segments …moryossef_duration_…` `--method baseline` | `rq2_offline_moryossef_duration_clean_…` |
| 4 | Moryossef26, duration BIO | joint AR, joint DLM | `--segments …moryossef_duration_…` `--method ar/dlm` | `rq2_offline_moryossef_duration_{ar,dlm}_…` |
| 5 | S1, duration BIO | clean AR | `--segments …s1_duration_…` `--method baseline` | `rq2_offline_s1_duration_clean_…` |
| 6 | the joint head, whole stream; point-mass Ω per span | joint AR, joint DLM | `--offline --method ar/dlm` | `rq2_offline_joint-ar_duration_ar_…` / `…joint-dlm_duration_dlm_…` |
| 7 | row 6's own spans | clean AR | `--segments <row 6's events>` `--method baseline` | `rq2_offline_joint-{ar,dlm}_duration_clean_…` |
| | **B. Online — the FSM decides from the past alone** | | | |
| 8 | Moryossef26 under the FSM | clean AR | `--stream --segmenter-arch moryossef --method baseline` | `rq2_online_moryossef_duration_clean_…` |
| 9 | S1 under the FSM | clean AR | `--stream --segmenter-arch s1 --method baseline` | `rq2_online_s1_duration_clean_…` |
| 10 | the joint head under the FSM | joint AR, joint DLM | `--stream --stability --method ar/dlm` | `rq2_online_joint-ar_duration_ar_…` / `…joint-dlm_duration_dlm_…` |
| 11 | row 10's committed spans | clean AR | `--segments <row 10's events>` `--method baseline` | `rq2_online_joint-{ar,dlm}_duration_clean_…` |

**How to read the ladder.** Compare the same target language and the same decoder family.

- **Span quality at one translator: 1 → 3 → 5 → 7.** Four span sources, one clean translator, one code path, one model, offline throughout. This is the only chain on the ladder in which a single axis moves, so the oracle-to-predicted drop and our-segmenter-against-theirs both come from here.
- **Translator at fixed spans: 2 − 1, 4 − 3, 7 − 6 offline; 10 − 11 online.** The clean translator and a joint arm read identical spans. The two models differ in more than the joint objective (training windows, epochs, learning rate, selection metric, and the gate), so this delta measures the whole training package at fixed boundaries, not the joint loss alone.
- **Access at one system: 10 − 6.** The same trained arm, online against offline. Three mechanisms move together: causal against bidirectional segmentation, the growing capped buffer under the soft Ω against the uncapped window [start − δ, end + 1 frame] under a point-mass Ω, and commit policy. Report it as a system-level access delta.
- **Prior art online: 10 vs 8, 10 vs 9.** An FSM commit needs only a stable terminator, so the translator changes no committed span and no commit time; `segmentation.f1` and latency compare the segmenters under one FSM. `--no-translate` runs the same segmentation with no decode (the output name gets the translator token `none`, so the text rows are not overwritten). Quote the text columns as system comparisons and split them with row 11 (next item).
- **Online split through row 11: 9 → 11 → 10.** Row 11 reads row 10's committed spans with the clean translator. 9 → 11 keeps the clean translator and moves the whole stage-2 effect on segmentation: target-language BIO training on stage-2 windows at the stage-2 rate, the text gradient through Ω, and the FSM commit timing of the joint head. It does not isolate "text improves segmentation"; that claim needs the `membership_gate.detach_omega: true` arm under the FSM with `--no-translate`, against row 10 with `--no-translate`. 11 → 10 keeps the committed spans and their commit times and moves only the translator conditioning: the clean translator on a hard crop against the joint arm's soft crop over the buffer.

Notes that prevent misreading, in brief:

- Every row uses the same annotation-ignore policy (`eval.scoreable_predictions`). Keep all emitted events, including short correct and false predictions. Λ_min is an event-generation rule, not an evaluation filter. Report localization precision and recall beside DVC.
- Rows 1–5, 7 and 11 read supplied spans; only the span _boundaries_ are external. `run_cascade` crops each span as [start, end + 1 frame], so the crop holds the span's terminator frame unless the span ends at the stream end (as row 6's window does). The online cascade (rows 8-9) and `report.py predict` crop the same way. `run_cascade` still runs the full model (BIO head and gate included) on `--method ar/dlm` rows. Rows with `--method baseline` use the ungated clean floor — deliberately a different model.
- Row 1 is scored inside the clean translator's own training view (it trains on GT caption spans), so it is an oracle reference, not a bound on the joint arms.
- Rows 6 and 10 are deterministic: no sampling at eval. Row 6 segments each whole stream, then captions each proposal in its own window; row 10 reads only the capped buffer.
- Every joint row runs twice, once per decoder family: `--method ar` first, then `--method dlm`. The two arms share one row id because they are the same condition under two decoders; the pair isolates the decoder family. The headline table stays DLM.
- There is no "cascade spans + joint translator" online row: `eval.py` pins an online cascade to the clean translator, because the joint arm's commit conditions read its own head. The offline pair 4 vs 3 carries that contrast.

**Metric.** RQ2 uses DVC-style text scoring: every prediction/reference pair above the tIoU threshold is a single-reference instance. Unmatched predictions receive a seeded random garbage reference. Scores are computed per source video, averaged equally across videos, then averaged over the declared tIoU thresholds. BLEU-4 is corpus BLEU over each video's pairs with no smoothing (`smooth_method="none"`), as in COCO BLEU, so a video with no 4-gram match scores 0. `report.py` computes every DVC number from the event files. The repository uses SacreBLEU 13a rather than the original toolkit's PTB tokenizer; describe this as DVC-style matching and aggregation with the stated text scorers, not a byte-identical reproduction.

A missed gold caption creates no DVC pair, so report localization precision, recall and F1 alongside text scores. The companion SODA-style fusion uses one-to-one temporal matching and charges predicted/gold counts. Neither RQ2 BLEU column is numerically interchangeable with RQ1 corpus BLEU. Compare the clean GT-span control and cascades under the same RQ2 scorer, checkpoint, beam count and annotation-ignore policy. SODA-style fusion per video and threshold uses the following counts:

&nbsp;&nbsp;&nbsp;&nbsp;`segmentation.f1 = 2|M|/(n_p+n_g)` · `S = Σ_(i,j)∈M s_ij`, `p = S/n_p`, `r = S/n_g`, `text = 2pr/(p+r)`

- The fusion charges spurious predictions (n_p) and missed gold (n_g) in one number, which is why it accompanies the recall-blind headline. Matched-pairs-only means are rejected outright. BLEU here is smoothed sentence-BLEU; CIDEr has no per-pair form, so it appears in the densevid rows only.
- **Provided-boundary translation** is measured by the GT-span rows (1–2), which are explicit oracle controls. RQ1 (0,0) supplies a clean GT-trimmed clip but gives the joint gate no known-start flag or internal boundary. It is a clean-input anchor, not an identical conditioning path to the supplied-proposal controls.
- **segmenter-eval and RQ2** report the same decode under two protocols, both in the segmenter-eval JSON: the Moryossef-comparable block (per-stream mean of F1s over UNK-masked BIO gold — cross-paper comparability only; a stream is a clean segment, so a cut video counts once per segment, and a stream that is UNK on every frame is left out) and `rq2_protocol` (caption-unit gold per source video, ignore-region filtering, F1 of macro P/R — computed by the RQ2 code path itself, so it matches the cascade rows' segmentation block by construction). Quote `rq2_protocol` wherever localization is compared across tables; the residual delta between the two blocks is gold construction plus aggregation, and it is now a printed pair, not an open question.
- Quote `segmentation.recall` beside any RQ2 text column.
- Significance (`report.py results`) is a paired bootstrap that resamples source VIDEOS, so units of one video stay together, also across its streams. It has two rows per system. `DVC BLEU-4` uses one unit per video (its DVC BLEU-4 averaged over the tIoU thresholds), so its mean delta is the headline difference and extra or phantom events count. `localized BLEU-4` uses one unit per caption unit at τ and charges missed sentences, which DVC does not. One video gives a NaN CI and p = 1.

## S1 continued-training study (`asf` and `bfi`)

Keep `[ase, asf, bfi]` in the multilingual phase. The target language is included throughout that phase. Call its direct evaluation **pooled evaluation without target fine-tuning**, not zero-shot. All three conditions use the same S1 architecture and released pose initialization; the target-only control must remain S1.

| Condition | Training schedule | Purpose |
| --- | --- | --- |
| 1. Target-only | Target-language S1 training | Reference for the value of multilingual training |
| 2. Pooled, no target fine-tune | Train on `[ase, asf, bfi]`; evaluate that checkpoint on each target | Multilingual model before specialization |
| 3. Pooled → target fine-tune | Start from the exact checkpoint in case 2; continue on each target separately | Effect of target specialization after multilingual training |

An example schedule is 50 pooled epochs followed by 20 target epochs. Set the target-only training budget before the comparison; adding phase epoch counts does not produce an equal-compute baseline. These are not equal-compute conditions: pooled and target epochs contain different numbers of batches and target windows. Record optimizer updates, target-language exposure and training time. If claiming equal-budget gains, match optimizer updates and batch size for cases 1 and 3 before running them. Case 2 is the shared intermediate checkpoint for case 3, not another model selected to improve the comparison. A lower case-2 result can reflect a compromise between languages or less target-specific training; it does not by itself establish a code defect.

Keep one `pretrain_geometry.buffer_cap_s` (at least the deployed `buffer_cap_s`) for the three conditions. The duration prior is fitted from the target language's train labels, so the decoder is the same for all three conditions. Keep chunk construction, augmentation, BIO loss, attention band, initialization family and evaluation fixed. Use dev for checkpoint selection, freeze the selection rules before test, and use a distinct output directory per run. The default early-stopping and best-checkpoint settings make the phase lengths upper limits. Report the actual selected epochs.

1. Run the pooled phase with `pretrain_languages: [ase, asf, bfi]`, the released `checkpoint.from_pretrained`, and a distinct `checkpoint.dir`: `python train.py --stage train-bio --epochs 50`. Do not pass `--language` for this phase.
2. Save the selected pooled `model.pt` as the shared case-2 checkpoint. Evaluate it on both targets using the pooled config: `python eval.py --segmenter-eval --segmenter-arch s1 --language <target> --checkpoint <pooled-model.pt> --split test --allow-test --output <case2-output.json>`.
3. For case 1, set `pretrain_languages: null`, retain the released initialization, and choose a distinct target-only directory. Run `python train.py --stage train-bio --language <target> --epochs <target-only-budget>` using the predeclared budget.
4. For case 3, keep `pretrain_languages: null`, set `checkpoint.from_pretrained: <pooled-model.pt>` and a separate target-adaptation directory. Run `python train.py --stage train-bio --language <target> --epochs 20`, without `--resume`.
5. Evaluate cases 1 and 3 with the same target test data and monolingual config. Use distinct `--output` paths. Restore the pooled configuration before main experiments that expect that checkpoint.

Case 3 continues both pose-encoder and BIO-head weights. The current fine-tuning interface starts a fresh optimizer and schedule; it does not carry Adam moments across the phase boundary. `--resume` is for interruption within the same recipe and schedule, not for switching the language pool. The main AR/DLM stages need only the selected S1 initialization; this segmentation study does not require separate translators for each case.

Interpret **3 vs 1** as the test of the pooled-then-target recipe, and **3 vs 2** as the effect of further target training. For the separate claim that text improves localization, compare joint training with the case-3 segmentation-only continuation under the same pooled start, target windows, BIO update budget and checkpoint-selection policy. Comparing only against the unadapted pooled checkpoint does not separate text support from extra target BIO training. If case 3 remains worse than case 1 under the declared protocol, do not claim that multilingual pretraining improved that result. The external Moryossef baseline tests a different visual encoder and cannot replace this same-architecture control. Choose its release by a stated comparison protocol, not by which version scores lower. Credit the 2026 public code even when no separate paper is cited; only our specific additions can be claimed as ours.

### Segmentation evaluation and metric meaning

The main streaming evidence must come from chronological online evaluation with equal observed frames, capacity, stride and latency policy. An offline whole-video result remains a useful reference. The four-mode window test is a controlled diagnostic, not an online event evaluation. Use identical test-video windows for all models, fixed independently of their predictions; do not reuse training examples or choose mode weights after seeing results. For modes 1/3, evaluate eligible complete-sentence localization. In mode 2, distinguish visible-fragment labeling from the decision to wait for a complete sentence. In mode 4, report false spans/false commits. Report modes separately: their training mixture is a sampling design and need not match how often they occur in a real stream. A one-sentence crop can make an all-signing prediction score well by cutting it at the supplied window edges; it does not establish boundary detection or correct commit timing. The current CLI supplies whole-video segmentation and online S1/Moryossef/joint evaluation. Streaming capacity is enforced before encoding. EOF clock advances do not count as new boundary evidence; forced overrides are flagged. A common four-mode benchmark remains a proposed comparison, not an implemented entry point.

| Output | What it measures |
| --- | --- |
| `bio_iou` | Raw-head signing-frame overlap, with B and I combined; averaged per stream over reliable labels |
| `phrase_tiou_f1@t` | One-to-one sentence matches at interval tIoU threshold `t`; averaged per stream |
| `phrase_frame_f1` | Macro F1 across the separate O/B/I frame classes |
| `phrase_segment_count_ratio` | Predicted/gold span count, averaged per stream; not boundary accuracy |

Three consecutive gold sentences can be merged into one predicted span with identical signing-frame coverage. Then `bio_iou` is perfect, while no sentence meets tIoU 0.5 if the three gold sentences have equal length. High `bio_iou` therefore does not imply accurate sentence boundaries. Use sentence precision/recall/F1 for that claim. Do not compute IoU from the already averaged precision/recall: the evaluator computes it within each stream and then averages. Use `rq2_protocol` for RQ2 comparisons. It ignores unsupported annotation regions and retains short emitted predictions.

The implemented mask is not Masked Transformer’s hard/learned confidence mixture. See `docs/membership_gate.md` §1.5 for the equations and the distinction. No self-box mask BCE is used.

## Key design decisions

One line each; the full argument lives at the pointer.

- **Caption units merge whole cues without retiming.** Units can contain several linguistic sentences; BIO labels describe unit membership. Incomplete coverage remains excluded. → `docs/data_pipeline.md`
- **Cross-split de-duplication is train-side** (decontamination convention); the split CSV is mandatory. → `configs/data.yaml`
- **Wrong-language videos are dropped per video by non-Latin script share** (`max_non_latin_ratio`). → `docs/data_pipeline.md` §5b
- **Scrolling caption tracks are dropped per video** (`drop_scrolling_tracks`): when over half the cues start before the previous one ends (YouTube's rolling two-line auto-captions), every cue time is a display event — a line leaves the screen a median 1.9–2.6 s after the next line arrives, 5–10× δ_enc — so punctuation can fix the grouping but not the timestamps. → `docs/data_pipeline.md` §3
- **BIO loss**: class-weighted CE plus binary signing Dice in S1 and joint training. Caption loss reaches the segmentation branch only through Ω; the translator reads no BIO-head features.
- **Terminator = first O-or-B, never "closing O"** — back-to-back sentences have no gap; same rule at training and inference. → spec §5.3
- **Decoder controls**: compare plain and enhanced decoding on both heads from the same logits. Use the enhanced Moryossef cascade as the external comparison and the online S1 cascade as the stage-2 comparison (9 → 11 → 10).
- **First-span coupling**: training sums valid-path probabilities before forming Ω. It does not select a hard training boundary or replace the predicted mask with a GT interval. GT determines only which text exists (a complete or open target); the model's own predicted span decides how that text trains: a covered target trains the translator, an uncovered one trains only Ω and the segmentation branch (critic row), and only when Ω can learn from it (not under `detach_omega`, not on a row whose Ω fell back to the whole window).
- **A merged span counts as covered.** "Covered" means containment with δ tolerance at each end (predicted start ≤ unit start + δ, predicted end ≥ unit end − δ), with no upper limit on overshoot. So a predicted span that merges the target unit with the next unit trains the translator to write only the first unit's text. Reasons: robustness to merge errors is the thesis (the translator reads the first complete unit and disregards the rest, as Modes 1/3 intend); DVC matches one predicted event to one true sentence, so a merged event scores against one reference; Ω still gets the text gradient on covered rows, so the segmentation branch is still pushed to split. Risk: it can teach the model to drop content. Not adopted: send spans longer than the unit by more than δ to the critic group. State this choice in the paper. → `models/streaming_slt.py` `forward_loss`
- **S1 pretrains competence before coupling**, pooled, on whole-input chunks with no window modes, jitter or anchors, so cuts are not placed from caption boundaries. Stage 2 uses the anchored sampler with a designed mode mix and a flat context band. No measured jitter or mode mix is a training input; `segmenter-errors` is the analysis that motivates the mode design. → `configs/bio_pretrain.yaml`, `data/chunks.py`
- **The two segmenters differ only by model and input contract.** The Moryossef arm uses S1's pool, chunk tiling (the same `ChunkDataset` with `release`, fixed 1023/24 s chunks), augmentation block, BIO labels, dev chunks, complete-span monitor and best-epoch floor, and both overlap-stitch the offline whole-video pass. So a cascade difference comes from the model and its input, not from exposure or selection. → `data/chunks.py`, `moryossef26/dataset.py`
- **No epoch before one full pool rotation can ship.** A pooled segmenter run sees every training record (clean segment) only after `cycle_epochs` epochs, computed from the record counts and printed at run time, so earlier epochs are not best candidates and do not count toward patience. → `train/helpers.py` `run_epoch_loop`
- **Stage 2 trains every module at one rate, 1e-5**, in one optimizer group, both pose encoders included; no freeze key exists. S1 trains its head at 2e-4 and its encoder at 6e-5 (`backbone_lr`, an S1-only key); the clean floor trains everything at 2e-4 in one group. `learning_rate` is a required key. Stage 2 requires a trained S1 at `checkpoint.bio_head_init`, so the gate is on from step 0 with no warmup. → `configs/dlm.yaml`, `configs/bio_pretrain.yaml`
- **No pose augmentation in stage 2 or the clean floor** (`augmentation: null`, refused otherwise). Only the two segmenters augment, with one shared recipe (fps 15-30, frame dropout, hand dropout, no rotation). So the arms and the clean floor differ only by the method, inference never augments, and δ and Λ_min are plain frame counts on the native 24 fps grid. → `configs/dlm.yaml`, `train/slt.py`
- **The confidence-bound term is Mode-2a only, and only where the model predicts a premature commit** (a closed span that starts inside the open GT unit) — the FSM never decodes the other states. It is an unlikelihood term (Welleck et al., ICLR 2020): -log(1 − p(t)) on the truncated decode's own token t, at slots where that decode is confident (> `tau_cb`), differs from a full-evidence decode that equals the reference, and agrees with it on every earlier slot. It names no target token: the reference decides only WHERE the truncated view is confidently wrong, never WHAT it should say, so the truncated input gets no text target (P1). Its effect is lower per-token confidence at the first confidently wrong slot (the reveal policy reads it). The FSM commit reads no confidence, so the term does not change commits. Training logs `cb_premature_rate` and `cb_active_count` (slots where the term fires). → spec §5, `train/losses.py` `confidence_bound_loss`
- **Best-checkpoint monitor**: clean translation uses dev BLEU; joint arms use `val_joint_score` (dev-window F1 times BLEU). This is a trade-off score, not DVC or a retention guarantee. Report both components.
- **Text scoring level is declared per language, never sniffed from references** (`char_level_for_target`; `tests/test_scoring_level.py` enforces every call site).
- **Segmenter-error artifacts are keyed by (segmenter, language, split)** so an `--segmenter-arch s1` run or another split can never overwrite the independent measurement (`tests/test_calibration_provenance.py`). They are analysis only; no training stage reads them.
- **Pose timing comes from `video_meta.csv`** (SignVerse resolves to exactly 24 fps). → `docs/run_real_data.md`
- **One body per frame, with short detector flickers removed.** The converter keeps slot 0 (the detector's top-scored person), except where slot 0 jumps to another real body for less than 0.5 s; body size is no guide, because a largest-body rule picks a still bystander. Hands bind to the kept body's wrists; in a frame with 2 or more real bodies, a hand that no wrist claims is zero. → `docs/data_pipeline.md` §5e2
- **Frames with a moving second person or with no person are cut out of every stream.** The converter writes `masked_runs` (runs of frames with 2 or more real bodies, with gaps under 12 frames bridged and at least 12 frames long, scored per 2 s block of 48 frames: a block is cut when the arm variation of the real bodies other than the kept one, in the kept body's shoulder widths, is above 0.7 or cannot be measured, and adjacent cut blocks merge) and `empty_runs` (runs of frames with no real body, gaps under 12 frames bridged, at least 24 frames long) to `video_meta.csv`. On every split the loader cuts each video at both, and each clean segment is a record and a stream (`<vid>@<first frame>`, a pose view). No person or detection rule drops a whole video; `undetected_ratio` is an audit diagnostic only. The loader refuses a `video_meta.csv` without the `masked_runs` and `empty_runs` columns and names the `prepare_yt25.py --stage all --overwrite` command. A unit that crosses a segment edge is quarantined (`reliable=False`, UNK on every frame). In a stage-2 window, a quarantined unit that ends before the text target moves the left edge to its end − δ, as the FSM's post-commit cut would. Captions define no cut: an uncaptioned stretch longer than 8 s stays in its stream as UNK. Scores and artifacts stay on the source video id and source time. → `docs/data_pipeline.md` §5f
- **A caption unit with no visible hands is quarantined, and no frame is cut.** The converter also writes `handless_runs` (runs of at least 12 frames where the kept body shows no hand: fewer than 10 of 21 keypoints above confidence 0.3 on both hands) to `video_meta.csv`. On every split the loader quarantines a reliable unit with at least `poses.handless_unit_share` (0.9) of its frames inside them (credits, a URL, an end card, lyrics over a person whose hands are out of shot). The frames stay in the stream, so the neighbouring units and the transitions into and out of the unit stay intact. The loader refuses a `video_meta.csv` without the `handless_runs` column. → `docs/data_pipeline.md` §5f2

## RQ2 artifact names

Every RQ2 row writes one pair of files under `outputs/`, named by the four fields that identify the row (`eval.rq2_output_stem`), with every field always present:

```
rq2_{when}_{spans}_{decode}_{translator}_{language}_{split}.json          the events
rq2_{when}_{spans}_{decode}_{translator}_{language}_{split}_scores.json   what this run scored on them
```

| field | values | meaning |
|---|---|---|
| `when` | `offline`, `online` | how THIS run decided: `offline` reads the whole stream, `online` decides from the past alone. A re-translation of saved spans is `offline`, except spans an FSM committed online (provenance `when=online`): row 11 keeps `online`, because its crop reads only frames that were in the buffer at commit time (one frame more after a cap-forced cut of an open span) |
| `spans` | `gold`, `moryossef`, `s1`, `joint-ar`, `joint-dlm` | which model cut the video; the joint head carries its arm, so the same-span controls of the two arms are different files |
| `decode` | `duration`, `plain`, `none` | the segmentation decode that produced them; `none` is the annotation |
| `translator` | `clean`, `ar`, `dlm`, `none` | which translator read them; `none` is a segmentation-only run. An ablation appends its config stem (`dlm-<stem>`), so it cannot overwrite the arm it is compared with |

So `rq2_offline_moryossef_duration_clean_asf_test.json` is the external segmenter's spans, duration-decoded, read by the clean translator, offline. A cascade row takes `spans` and `decode` from the span file's own provenance, never from its filename, so an output name cannot drift from what produced it.

`segmenter_eval_*` files stay separate: they score spans ALONE, with no translator, and carry both the Moryossef-comparable block and `rq2_protocol` in one payload.

## Repository map

```
backbones/   Uni-Sign 69-kp 4-part ST-GCN (UniSignPoseEncoder)
poses/       generic pose files: normalize_keypoints_unisign (133→69) · pose_io (load_pose_window, video_meta.csv,
             per-video fps) · augmentation
data/        records → samples: loader (YouTube-SL-25 + pooling + dedup) · windowing (BIO, first-complete-span, χ) ·
             sampler (stage-2 window modes + jitter) · chunks (segmenter whole-input chunks) · batch
models/      block_diffusion · dmax (OPUT + block decode) · membership_gate (soft first-span Ω) · front_end · unisign ·
             streaming_slt (MisalignedSLTModel) · bio_head
train/       slt (AR/DLM trainer) · bio_pretrain (S1) · losses (Dice+CE, CB) · helpers (epoch loop, segmenter dev monitor) ·
             distributed
moryossef26/ faithful external segmenter (raw-kp UNet): model · dataset · trainer · infer. NOT the FSM head.
infer/       duration_decode (semi-Markov BIO Viterbi) · commit_gate · decode (DMax block decode + SPD) · stream (FSM) · stability
metrics.py   BIO monitor · tIoU segments · text metrics (declared scoring level)
utils.py     load_yaml (extends, ${language}, ${corpus}) · checkpoint_dir · pick_device
train.py     --stage {smoke-data, train-bio, train-moryossef, train-slt}
prepare_yt25.py  YouTube-SL-25 via SignVerse-2M → language layout: shards, pose converter, captions, video_meta.csv
eval.py      --rq {1, 2} · --segmenter-eval · --emit-gold-segments · --stability
analyze.py   --stage {dataset-summary, segmenter-infer, segmenter-errors}
report.py    {results, progress, data, poses, predict}: statistics, figures and audits over saved artifacts (outputs/report/)
docs/        membership_gate.md · segmentation_decoding.md · data_pipeline.md · run_real_data.md · implementation_notes.md · literature_notes.md
```

## Configuration

| config                    | drives                                                                                  | inheritance                                      |
| ------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `dlm.yaml`                | the DLM method; single source of truth for sampler, BIO-head arch, gate, optimizer keys | —                                                |
| `inference.yaml`          | FSM design constants: one value per constant for every corpus; no stage writes it       | standalone                                       |
| `ar.yaml`                 | gated AR de-risk (§9.3)                                                                 | `extends: dlm.yaml` (decoder + output dir only)  |
| `baseline_eval.yaml`      | clean baseline: ungated greedy AR, eval-only                                            | `extends: dlm.yaml` (gate/CB off)                |
| `baseline_train.yaml`     | trains the clean floor: mode1-only, zero context band, `lambda_bio: 0`, one rate (2e-4), no augmentation | `extends: baseline_eval.yaml`                    |
| `bio_pretrain.yaml`       | S1 pooled segmentation pretraining on whole-input chunks (`train-bio`); own rates (head 2e-4, encoder 6e-5); the segmenter augmentation block, the same as in `moryossef26.yaml` | `extends: dlm.yaml` (BIO-head arch and BIO loss; not the sampler blocks) |
| `moryossef26.yaml`        | external Moryossef segmenter (`train-moryossef`)                                        | standalone                                       |
| `data.yaml` / `eval.yaml` | corpora, splits, `target_lang`, `pretrained_slt`, subtitle pipeline / RQ grids          | —                                                |

`inference.yaml` constants. One value each for every corpus, in seconds or in frames at 24 fps. They are design values, not measurements. Stage 2 applies no fps augmentation, so every stage-2 window is on the native 24 fps grid, as in the FSM and eval. δ and Λ_min are therefore plain frame counts everywhere:

| constant | value and meaning |
| --- | --- |
| `stride_s` | 1.0 s |
| `buffer_cap_s` | 40 s: a buffer this long is force-committed (flagged PARTIAL). It is a design value meant to cover the train p99 unit duration of every configured corpus (measure that p99 again on the re-converted data before you quote it) and equals the S1 context; every translation recipe (clean floor, arms) clamps its windows here. A hysteresis commit needs δ of lead-in plus (K−1) × stride after the terminator, so a unit longer than about 37.5 s commits only through the forced path. A pause before a unit also uses cap space, so a few units shorter than 37.5 s can be cut open; report the cut-open and forced rates. The left-truncated tail of a force-committed unit is never translated |
| `boundary_stability.delta_enc_frames` (δ) | 12 frames = 0.5 s: terminator-stability tolerance, post-commit overlap, and target-identity tolerance. A design tolerance equal to Λ_min, not a measurement: the terminator noise of the banded S1 is not measured |
| `boundary_stability.hysteresis_strides` (K) | 3: strides a terminator (O or B) must hold still before it commits; the one right-context requirement |
| `span_selection.min_span_frames` (Λ_min) | the shortest unit the sampler, the gate's posterior and the FSM may target. A LABEL-domain floor: 12 frames = 0.5 s, chosen to reject brief flicker; genuine shorter units can be excluded, so report the resulting recall ceiling |
| `translation.max_text_tokens` | 320 (at or above the training canvas). The decode always reads the whole buffer under Ω, the conditioning stage 2 trains; the hard-crop comparison is RQ2 row 11 |

Stage-2 keys in `dlm.yaml` (S1 and the clean floor set their own optimizer values):

| key | value and meaning |
| --- | --- |
| `bio_attention_radius_s` | 0.375 s band on every BIO-head layer; reach 1.92 s ≤ (K−1) × stride. S1 stamps it; a mismatch is refused |
| `mode_ratios` | `{mode1: 0.45, mode2: 0.30, mode3: 0.20, mode4: 0.05}`: a fixed DESIGN mix, not a measured one; complete units (mode1 + mode3) get 65 %. The clean floor sets `{mode1: 1.0}` and the others 0 |
| `jitter.context_s` | 4.0 s = (K+1) × stride: window edges ~ Uniform(−4, +4) s around the anchor's boundaries. The clean floor sets 0.0 |
| `jitter.cut_range` | `[0.15, 0.85]`: uniform relative Mode-2 cut depth |
| `augmentation` | `null`: no pose augmentation in stage 2 or the clean floor; train-slt refuses a non-null block. The two segmenters set one shared block in `bio_pretrain.yaml` and `moryossef26.yaml` |
| `learning_rate` | 1e-5 for every module in one optimizer group, both pose encoders and the BIO head included. Required key (no default). S1: 2e-4 head, 6e-5 encoder (`backbone_lr`, S1 only). Clean floor: 2e-4 for everything |
| `spd.tau_dec` | 0.5, the one DLM commit threshold (stage-2 Mode-2a decodes, dev decoding and eval all read it). A fixed mechanism constant (DMax's math point; spec §12), never dev-tuned. `block_size` bounds the arm's parallelism and is fixed at training; `tau_dec` is the dial inside it, and label smoothing 0.2 puts the per-token loss minimiser at max-prob 0.8, so a threshold at or above it approaches one commit per pass. RQ1 reports its cost as `mean_decoder_passes` (read it at `eval.yaml rq1.batch_size 1`); no speed claim without it |
| `lambda_bio` | 1.0 |
| `membership_gate.detach_omega` | `false`; `true` is the ablation "text does not train segmentation" |

Other derived values:

| value | source |
| --- | --- |
| duration prior | lognormal fit of the target language's TRAIN unit durations (`DurationModel.fit`: `mu_log_s`, `sd_log_s`, `tail_start_s`, `annotation_signature`). Stage 2 stamps it in the checkpoint and eval decodes with the stamp; cascades fit it at run time. Never written to a config |
| `bio_class_weights` | `balanced` — resolved from measured label counts at train start, logged |
| pooled S1 context | `bio_pretrain.yaml pretrain_geometry.buffer_cap_s` = 40 s, the longest chunk C, recorded in the checkpoint. train-bio refuses a value below the deployed `buffer_cap_s` |

Rules that are not optional: no stage writes `inference.yaml` and there are no per-language rows; the sampler, gate and FSM share target eligibility. Every `train-slt` checkpoint (the arms and the clean baseline) stamps δ, Λ_min and the cap. Eval never deploys a checkpoint's own geometry. Every eval path that builds a model checks the checkpoint it actually loads (`eval._build_eval_model` calls `models.checkpointing.require_fixed_constants`). This covers RQ1, the `--segments` cascades, `--offline`, `--stream` and `report.py predict`. For an arm it checks the arm checkpoint. For the clean baseline it checks `--checkpoint`, or `data.yaml pretrained_slt` when there is no `--checkpoint`; this includes the clean translator of an online cascade (rows 8-9). The check refuses a stamped cap, δ (on a gated checkpoint only) or Λ_min that differs from `inference.yaml`, and a wrong `annotation_protocol`. The released Uni-Sign file carries no stamps and passes. Eval runs every row under the live constants. After an edit of `inference.yaml`, retrain before you evaluate. S1, the clean baseline and the arms depend on no measurement, so S1 and the clean baseline can train in parallel.

Λ_min is the one constant whose source matters more than its value. It is the eligibility rule shared by the sampler's target, the gate posterior's first-complete span and the FSM's commit, so it decides which units the system can ever supervise or emit. It comes from the unit-duration distribution, never from a model measurement: δ answers "how far can a terminator estimate move", Λ_min answers "what is the shortest unit we will emit", and tying the second to the first makes the reachable part of the test set a function of a trained head's noise. Re-check the 0.5 s value only if a preprocessing change moves the p1 unit duration below it.

Watch `text_covered_rate` during stage-2 training (the share of complete text targets that the predicted first span contains, with δ tolerance at each end, so a span that merges the target with the next unit counts; the rest train as critic rows when Ω can learn from them) beside `cb_premature_rate` (the share of Mode-2a rows where the head predicts a premature commit). A falling covered rate means the head localizes fewer text targets; find the cause before spending more GPU.

### Normalization and comparison contract

| Path | Frames used to compute the body box |
| --- | --- |
| S1 training | Each chunk |
| Stage-2 training, RQ1 | Supplied training or controlled window |
| Whole-video S1 and offline joint head | Each chunk, at the checkpoint's trained context |
| Streaming joint model | Active raw buffer only |
| Online S1/Moryossef cascades | Active buffer for segmentation; selected raw crop for the clean translator |
| Moryossef training and inference | No body box: per-frame shoulder normalisation and one fixed corpus-fitted standardisation table |

The Moryossef arm trains and infers on the RELEASE input contract — their 50 landmarks (8 body, two 21-point hands, no face), their shoulder normalisation and a corpus-fitted standardisation table — so it shares no input representation with the in-system head except the detector's own confidence gate. The aspect correction uses each video's own frame size from `video_meta.csv` (the `data.yaml` size only when the row has none). Declared deviations: z and vz are identically zero (DWPose has no depth, and those two channels carry 19.1 % of the released first convolution's input-channel weight energy), the shoulder normalisation reduces per frame rather than per video, and the standardisation table is fitted on our pool rather than theirs. Regenerate affected segmenter spans after a normalization change. No comparison requires equal crop boxes across models with different training inputs. See [implementation notes](docs/implementation_notes.md) for the current contracts and [membership gate](docs/membership_gate.md) for the loss and gradient paths.

## Required reruns

Every pose file, segmenter and translator must come from the current code. The pose converter keeps one body per frame with the slot-continuity rule, zeroes an unclaimed hand in a frame with 2 or more real bodies (`docs/data_pipeline.md` §5e2), and writes the `masked_runs` and `empty_runs` columns (§5f) and the `handless_runs` column (§5f2). The annotation fingerprint does not hash pose values, so no guard refuses a `.npy` file from another converter version or a checkpoint trained on one. It hashes every record id, view duration and unit, so the visual cuts give new labels: `eval.require_annotation_match` refuses a span file, events file or RQ1 file made before it, and stage 2 refuses a clean-baseline warm start trained before it. The loader refuses a `video_meta.csv` without the `masked_runs`, `empty_runs` and `handless_runs` columns, so a training stage cannot start on unconverted data. Eval and `--resume` refuse these checkpoints: an S1 checkpoint without the attention-band stamp; a stage-2 checkpoint without `architecture: two_stream_slt`; an S1 or stage-2 `latest.pt` whose rate or `augmentation` stamps differ from the config; a segmenter `latest.pt` whose `augmentation` or `best_epoch_floor` stamp differs from the config; and a clean baseline stamped with a cap other than `inference.yaml`'s. Move a refused run directory away before you train again, because `train.py` refuses to start over an existing `latest.pt` without `--resume`. Run in this order:

| Step | Required action |
| --- | --- |
| 1 | Re-convert the poses of all three languages: `python prepare_yt25.py --stage all --overwrite --languages ase asf bfi`. It runs convert, person-counts and subs in this order (convert rewrites `caption_source`; subs restores it). It needs the shard tars; re-download any that `--delete-tars` removed. Every person-counts line must show `masked_runs 0/N`, `empty_runs 0/N` and `handless_runs 0/N` (person-counts reads only consolidated archives for the masked and empty runs, so a per-frame-layout video gets those columns only from convert). Then write the corpus audit, `python report.py data --language ase --language asf --language bfi`, and report its visual-cut rows and its `units_quarantined_handless` row with the results. |
| 2 | Train S1 (`train-bio`), the Moryossef segmenter (`train-moryossef`) and the clean baseline of each language (B2a) at the 40 s cap, in parallel. Both segmenters use the shared augmentation recipe and the best-epoch floor. |
| 3 | Re-root `data.yaml languages.<lang>.pretrained_slt` to `checkpoints/baseline_train/<lang>/model.pt` (B2b). |
| 4 | Train the AR and DLM arms in fresh directories. |
| 5 | Regenerate every span file (both segmenters decode with the train-fitted prior) and rerun every evaluation row: `--segmenter-eval`, `segmenter-infer` and `segmenter-errors`, cascades, joint offline/online events, RQ1 curves. Use one clean translator for the cascade rows. |

Archive result files before reusing their output paths. Segmenter monitor values are comparable only between runs of the same recipe. Externally supplied spans are explicit fixed-boundary controls.

## Reading BLEU across RQ1 and RQ2

| Output column | Calculation |
| --- | --- |
| `translation_bleu4` | Corpus BLEU over the full RQ1 or generated-dev caption set |
| `densevid_bleu4` | Unsmoothed corpus BLEU within each video's tIoU-paired captions, followed by an equal-weight mean over videos |
| `soda_bleu4` | Smoothed sentence BLEU on one-to-one matches, normalized by predicted/gold counts, then averaged over videos |

These columns answer different questions. Shared text preprocessing does not make their numbers interchangeable. DVC includes garbage references for unmatched predictions and no separate pair for each missed GT sentence. A threshold also changes which references enter the score. Compare a cascade with the existing GT-span baseline row using the same RQ2 column, clean checkpoint, beam count and annotation-ignore policy. Keep RQ1 corpus BLEU as its own robustness result. GT-span captions are an oracle-input control, not a mathematical upper bound on every caption metric. The repository's DVC-style scorer uses SacreBLEU 13a instead of the original toolkit's PTB tokenizer, with COCO's no-smoothing rule; disclose that choice. RQ1 corpus BLEU and SODA sentence BLEU keep sacrebleu's `exp` smoothing.

### Processing windows and sentence spans

Both offline segmenter pipelines follow this sequence:

`video → neural chunks → stitched frame scores → sentence spans → one caption per supplied span`

| Setting or object | What it limits |
| --- | --- |
| S1 trained chunk / Moryossef `num_frames` | Temporal encoder context; chunk edges are not sentence boundaries |
| Offline predicted span | Its predicted start and end; a span can cross many neural chunks |
| Translation `--batch-size` | Number of spans processed together, not the frames inside each span |
| Attention query block | Temporary attention memory; all keys in the supplied span remain visible |
| Streaming buffer cap | Available evidence per FSM step, with explicit forced-partial handling |

The soft duration model does not impose a maximum sentence length. If boundary evidence is weak, an offline prediction can merge many sentences. This is a localization error, not a reason to remove that prediction from evaluation. The S1 and Moryossef offline cascades use the same full-proposal translation path. For a bounded streaming comparison, use both online cascades under matched geometry. Do not add sentence boundaries at neural chunk edges or change only one baseline's caption context after inspecting test results.


The joint `--offline` row also preserves each complete predicted interval. It captions each proposal in the window [start − δ, end + 1 frame] (the FSM's post-commit geometry: δ of committed context, the span, its terminator) under a point-mass Ω on the proposal. That is close to a crop of the span, not exactly one: the δ lead puts the span up to δ frames after the task prompt, so it matches a crop only up to the eps leak and the prompt↔span relative-position offset (see Architecture). There is no capacity cut on a proposal. Here the cap sets only the chunk length of the whole-video BIO pass; it limits the evidence only in `--stream`, and it must not shorten offline event timestamps. Cascade event provenance records the resolved translator checkpoint, including when the CLI uses its default.
