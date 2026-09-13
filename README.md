# Misaligned SLT

Pose-only, gloss-free, **streaming** sign-language-to-text translation. The system is robust to sentence-boundary **temporal misalignment** (imperfect segmentation).

**Central rule.** Training windows match the windows the streaming loop produces. A truncated visual input never gets a partial text label (premise **P1**). Every segmenter error becomes a window-shape variation, never a label variation (**P2**). Design spec: [`streaming_slt_prompt.md`](streaming_slt_prompt.md). The segmentation→translation coupling: [`docs/membership_gate.md`](docs/membership_gate.md). Decoder model, algorithm and loss: [`docs/segmentation_decoding.md`](docs/segmentation_decoding.md).

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
raw pose buffer → normalization → Uni-Sign pose backbone → pose tap F
                                                │
                                                └→ Conv stem + RoPE temporal encoder → H
                                                           ├→ BIO classifier → legal paths
                                                           │                    ├→ first-span probability → Ω
                                                           │                    └→ Viterbi span → FSM
                                                           └→ nonlinear mapper → residual F + mapper(H)
                                                                                 ↓
                                                                    task prompt + mT5 encoder
                                                                                 ↓
                                                                      AR / DLM decoder ← Ω
```

- **Pose encoder** ([`backbones/`](backbones)): Uni-Sign 4-part ST-GCN, loaded from the released `*_pose_only_slt.pth`. [`poses/`](poses) reproduces Uni-Sign normalization byte-exactly.
- **Front end** ([`models/unisign.py`](models/unisign.py)): one pose encoder, two LMs (mT5 default; mBART ablation). `MisalignedSLTModel` takes either, so the heads, sampler, and FSM are written once.
- **Decoder** ([`models/block_diffusion.py`](models/block_diffusion.py), [`models/dmax.py`](models/dmax.py)): BD3LM core + DMax (OPUT training, SPD/DCD inference). Verified faithful to the released DMax/DCD code.
- **Shared temporal encoder** ([`models/bio_head.py`](models/bio_head.py)): Conv1d stem, RMSNorm and time-aware RoPE. Its features train from both segmentation and translation.
- **Membership gate** ([`infer/duration_decode.py`](infer/duration_decode.py), [`models/membership_gate.py`](models/membership_gate.py)): exact probabilities of membership in the first eligible complete span. Caption gradients reach the BIO classifier through the soft mask. The same soft-mask rule runs during training and streaming.
- **Segmentation decoder** ([design](docs/segmentation_decoding.md)): a restricted semi-Markov CRF with fixed duration scores, legal BIO transitions and exact Viterbi. BIO supervision remains frame CE plus Dice. S1 checkpoint selection uses an untuned legal-only monitor; joint conditioning sums probabilities under the calibrated model.
- **Streaming FSM** ([`infer/stream.py`](infer/stream.py)): first eligible complete span from duration-aware BIO Viterbi, boundary hysteresis, commit lag for B terminators, translation confidence and explicit forced-partial handling. Committed prefixes cannot be selected again.

**Claim boundary.** S1 is a pretrained segmentation module built from existing components. Duration-aware BIO decoding uses established structured inference and can also process Moryossef logits. It enforces valid transitions but does not establish accurate interior boundaries. S1's CE/Dice objective does not use the caption mask. The proposed joint contribution is first-complete conditioning with incomplete-state handling; its accuracy and latency benefit must be measured. See `docs/membership_gate.md` §4.

## Multi-GPU (torchrun)

Prefix any `train.py` command with `torchrun`; change nothing else:

```bash
torchrun --standalone --nproc-per-node=4 train.py --stage train-slt --language "$LANG" --slt-config configs/dlm.yaml
```

- Config `batch_size` is the GLOBAL batch, split across ranks. A non-divisible batch is a hard error.
- `mixed_precision: auto` picks bf16 on compute capability ≥ 8. Multi-GPU + fp16 is refused (per-rank scaler drift).
- `latest.pt` is a full resumable snapshot, written every epoch. Continue an interrupted run of the same architecture and settings with `--resume`. Re-running without `--resume` over an existing `latest.pt` is refused.
- CUDA compiles the membership recurrence on first use. Judge speed after warmup. Before another full run after a slowdown, run `python tests/test_membership_runtime.py` on the training GPU. This data-free check measures gate forward/backward time and checks gradients; it writes `outputs/membership_runtime.json`. It does not estimate total epoch time.
- Eval and analysis stay single-process. Offline cascade translation honors `--batch-size`; streaming processes one active buffer per stride. Gradient averaging is explicit, not DDP ([`train/distributed.py`](train/distributed.py) explains why).

## The experiment sequence

Two blocks. **Stage A** (segmentation pretraining) runs once, pooled over all three languages. **Stage B** runs once per target language, in three phases: **MEASURE → TRAIN → EVALUATE**.

|                                   | **Stage A — language-agnostic**               | **Stage B — per language**                                                                       |
| --------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Runs                              | once, for all languages                       | once per target language                                                                         |
| `--language`                      | **refused** (a pool has no target)            | required                                                                                         |
| Produces                          | `checkpoints/{moryossef,bio_s1}/multi_<pool>` | `checkpoints/{baseline_train,ar,dlm}/$LANG`, `outputs/*_$LANG_*`, the `inference.yaml` constants |
| Reads `data.yaml pretrained_slt`? | no — released warm starts only                | yes (`utils.resolve_pretrained`)                                                                 |

Segmentation training reads its explicit warm start, so the B2b clean-translator re-root cannot reach S1. The two blocks are independent. `--language "$LANG"` re-points the dataset and every `${language}` path; no config edits.

Before Stage A, fix the annotation rules in `configs/data.yaml`. The shared loader applies cleaning, cue grouping,
comma rendering, selective capitalization and the 10% punctuation threshold to train, dev and test. The capitalization
lexicon is fitted on train only and reused for dev/test. Record excluded-video counts and annotation fingerprints;
keep these rules fixed while comparing systems. See [the data protocol](docs/data_pipeline.md#3-caption-unit-construction--reconstruct_sentences).

```bash
# ═══ STAGE A — run ONCE. Both segmenters train on the SAME multilingual pool ([ase, asf, bfi]). ═══
# ── A0. One-time data + checkpoints ──
#   Warm start: checkpoints/openasl_pose_only_slt.pth.
#   Prepare poses and captions for all three pool languages:
#   python prepare_data.py --stage all --languages ase asf bfi

# ── A1. External Moryossef segmenter — the RQ2 cascade floor ──
#   Raw keypoints → UNet → RoPE BIO: a different input space from our head, so it stays an independent baseline.
#   It uses the SAME pool and temperature as S1 (configs/moryossef26.yaml), so the cascade compares methods,
#   not data. Pooled checkpoints live in ${corpus}-named directories (multi_ase-asf-bfi); utils.checkpoint_dir
#   is the one resolver for every reader and writer. Never build a segmenter checkpoint path by hand.
python train.py --stage train-moryossef        # -> checkpoints/moryossef/multi_ase-asf-bfi/model.pt

# ── A2. S1 — segmentation pretraining (pose encoder + BIO head, pooled) ──
#   Sentence boundaries are prosodic and prosody is shared across signed languages, so segmentation pretrains
#   on the pool; translation stays monolingual in Stage B. The encoder trains too (backbone_lr); stage 2 loads
#   encoder AND head via bio_head_init. The pool is temperature-flattened and balanced by SUB-sampling with
#   per-epoch rotation — nothing is replicated, nothing is permanently dropped. Dev is a balanced fixed
#   sub-sample; test is pooled as-is. Every design value (fixed mode mix, designed jitter, uniform cuts,
#   auto context = pool train p99 + stride + 1 s, legal-BIO monitor) is documented in configs/bio_pretrain.yaml.
#   Checkpoints stamp their pool (meta.pretrain_pool); every loader refuses a pool mismatch.
#   Monitor = val_mode3_tiou_f1. Compare checkpoints with --segmenter-eval, never by monitor value.
python train.py --stage train-bio              # -> checkpoints/bio_s1/multi_ase-asf-bfi/model.pt

# ═══ STAGE B — per target language. MEASURE (B1) → TRAIN (B2, B3) → EVALUATE (B4, B5, B6). ═══
# WHAT A CHANGE INVALIDATES:
#   New joint architecture/loss -> fresh AR and DLM training, followed by their RQ1/RQ2 evaluations.
#   New S1 checkpoint or legal decode -> S1 evaluation and B1 for every target language.
#   Changed delta/minimum span -> B1b capacity, B1c lag, and the arms that consume those values.
#   Changed capacity -> clean baseline and arms if their training windows change.
#   Changed commit lag only -> streaming evaluation; it is not a training input.
#   Changed ground truth -> every dependent training/calibration/evaluation stage.
LANG=ase      # ase (ASL) | asf (Auslan) | bfi (BSL)
python train.py --stage smoke-data --language "$LANG" --split train --num-samples 64

# ── B1. MEASURE — one duration score model in hard decoding and soft membership ──
# S1 architecture and CE/Dice loss are compatible with an existing pooled S1 checkpoint. Check standalone multi-sentence localization before using it as a joint-training initialization; compatibility does not establish quality.
# tune-decode refuses a segmenter checkpoint whose meta lacks the current annotation_protocol stamp. Retrain S1 (and the
# Moryossef segmenter) on the current caption units before B1; a checkpoint trained under an earlier label protocol cannot be reused.
# B1a — fit train durations and select duration-score weights on dev.
python analyze.py --stage tune-decode --segmenter-arch s1 --language "$LANG" --split dev --write-config
# Then measure boundary tolerance and minimum eligible span under that same score model.
# If the written minimum span changes which sentences can be measured, rerun until the selection is stable.
python analyze.py --stage delta-enc --language "$LANG" --split dev --write-config
# On a large dev set, --num-sentences 3000 uses a fixed random subset.
# B1b — capacity from reliable TRAIN sentence durations, stride and measured tolerance.
python analyze.py --stage buffer-cap --language "$LANG" --split train --write-config
# B1c — commit lag under the resolved capacity/tolerance; duration-score weights stay fixed.
python analyze.py --stage tune-stream --segmenter-arch s1 --language "$LANG" --split dev --write-config
# After B3, lag can be selected again on the deployed head using --checkpoint checkpoints/ar/$LANG/model.pt
# (or the DLM checkpoint). This changes streaming evaluation only. Ties prefer smaller lag.

# ── B2. TRAIN the clean floor — the shared init for every later stage ──
#   Faithful Uni-Sign SLT transfer on a caption-trimmed training view: mode1-only, ~zero jitter, lambda_bio 0.
#   Windows clamp to the B1b cap; configs/baseline_train.yaml documents why a stale cap biases this arm.
#   ORDER: this arm sets lambda_bio 0, so it needs no duration fit — but it reads buffer_cap_s, and B1b computes
#   that from delta_enc, which needs a trained S1. So it cannot start before B1b. After B1b the clean floor and
#   both joint arms are independent and may run in parallel.
#   The reported clean number is NOT a separate step: it is the (0,0) cell of the B5 baseline sweep.
#
# B2a — train (writes checkpoints/baseline_train/$LANG/model.pt)
python train.py --stage train-slt --language "$LANG" --slt-config configs/baseline_train.yaml
#
# B2b — accept, then re-root. Acceptance costs nothing: best.json's val_translation_bleu4 IS dev BLEU;
#   several times the zero-shot ~1 = pass. Watch val_translation_len_ratio for degenerate lengths.
#   Then edit configs/data.yaml: languages.$LANG.pretrained_slt: checkpoints/baseline_train/$LANG/model.pt.
#   That one edit re-points the stage-2 arms (B3) and every later --method baseline eval, so the floor and the
#   arms share one translation init. It does NOT re-point Stage A (pooled S1 keeps the released warm start; the
#   arms deliberately carry S1's segmentation-adapted encoder, which the translation-only floor must not absorb).

# B2c (OPTIONAL) — initial loss-scale probe, per arm and language. 16 train batches, no optimizer update.
#   python analyze.py --stage loss-balance --split train --language asf --slt-config configs/ar.yaml
#   Only useful if lambda_bio is going to be swept: it centres the grid on the measured gradient-scale ratio
#   instead of on 1.0. It is not an optimum (docs/membership_gate.md §1.4). Skip it and keep lambda_bio: 1.0
#   if the sweep is not planned — nothing downstream reads its output.

# ── B3. TRAIN the arms — stage-2 fine-tune under the gate ──
#   Both arms train under the membership gate, on the S1 init, with the B1 constants. Requires the B2b re-root.
#   The S1-pretrained pose encoder and BIO head train at backbone_lr (default 0.3 x learning_rate) — the rule stage 1
#   applies to the released encoder; the LM trains at learning_rate. baseline_train.yaml pins backbone_lr = learning_rate
#   (Uni-Sign's one-rate transfer recipe), so the clean floor is unchanged.
#   Stage 2 shares the pretrained visual temporal features H across BIO classification and a nonlinear V-L mapper.
#   The mapper is residual: F + mapper(H), with its last layer initialized to zero to preserve the warm-start input.
#   Both the shared features and first-span membership carry caption gradients. BIO labels and the first-complete
#   translation target stay fixed. No segmentation reference model or GT-boundary veto trains with the arms.
#   Checkpoints without this mapper cannot resume. Interrupted runs of this architecture use --resume.
#   Watch val_phrase_tiou_f1, boundary precision/recall and val_translation_bleu4 together.
python train.py --stage train-slt --language "$LANG" --slt-config configs/ar.yaml    # gated AR de-risk (§9.3)
python train.py --stage train-slt --language "$LANG" --slt-config configs/dlm.yaml   # DLM -> checkpoints/dlm/$LANG

# ── B4. EVALUATE — calibration, acceptance checks, diagnostics. Nothing here parameterizes training ──
# B4a — portable decoder control: no additional segmentation training.
python analyze.py --stage tune-decode --segmenter-arch moryossef --language "$LANG" --split dev --write-config
#   Freeze the settings before test. Four decoder cells; no extra model training.
for S in moryossef s1; do
    for D in plain duration; do
        python eval.py --segmenter-eval --segmenter-arch "$S" --segmenter-decode "$D" \
            --language "$LANG" --split test --allow-test --output outputs/segmenter_eval_${S}_${D}_${LANG}_test.json
    done
done
#   Declare the training data, initialization, normalization and trained context beside each model.
#   Different encoders and data views make this a system comparison, not a controlled test of one layer.
#   Use the three-condition S1 continued-training study below to test the value of pretraining.
#   TWO PROTOCOLS, ONE DECODE: segmenter-eval also scores its spans through the RQ2 code path itself
#   (`rq2_protocol` in the JSON: caption-unit gold, ignore-region filtering, F1 of macro P/R) and prints both
#   numbers per threshold. QUOTE the rq2-protocol number wherever localization is compared across tables — it is
#   identical-by-construction to the cascade rows' segmentation block for the same checkpoint and decode; the
#   Moryossef-protocol block stays for cross-paper comparability only.
#
# B4b — segmenter-error analysis: the reported taxonomy + the measured-jitter ablation input.
#   NOT on the training path: both stages train on the designed corruption (dlm.yaml jitter block).
#   The taxonomy is the paper's evidence that real segmenters produce exactly the window-mode event types.
#   SPLIT BY ROLE, enforced in code: the REPORTED taxonomy may run on TEST (--allow-test — it describes the same
#   split as the results tables); the jitter/cut ARTIFACTS that feed the measured-jitter ablation are a TRAINING
#   input and stay DEV-ONLY. Artifacts are split-stamped in name and payload, and the training side
#   (data/jitter.py, the sampler's mode_ratios loader) REFUSES a split=test artifact outright.
#   REPORTED taxonomy — TEST, the split the results tables describe (this is pure analysis; nothing trains on it):
python analyze.py --stage segmenter-infer --segmenter-decode duration --language "$LANG" --split test --allow-test \
    --output outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json
python analyze.py --stage segmenter-errors --language "$LANG" --split test --allow-test \
    --predictions outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json
#   ABLATION INPUT — DEV, only when running the measured-jitter ablation (its artifacts feed TRAINING, and the
#   loader refuses a split=test artifact):
python analyze.py --stage segmenter-infer --segmenter-decode duration --language "$LANG" --split dev \
    --output outputs/segmenter_predictions_moryossef_duration_${LANG}_dev.json
python analyze.py --stage segmenter-errors --language "$LANG" --split dev \
    --predictions outputs/segmenter_predictions_moryossef_duration_${LANG}_dev.json
#   Guard: segmenter-errors refuses spans whose stamped architecture or BIO decoder differs from this run.

# ── B5. RQ1 — controlled boundary sensitivity ──
#   All three arms run, deliberately. RQ1 is the only CONTROLLED robustness evidence: the baseline row shows the
#   problem; the arms' rows test the central claim. "The arms process misaligned input differently" is the
#   hypothesis under test, not a reason to exempt them; the gate's window-skipping is handled by the
#   (decoded-only, skip-rate) reading below. The baseline's (0,0) TEST cell is the reported clean anchor.
#   Context-appetite diagnostic: --severity-mode absolute
#   --severity-grid-head 0 --severity-grid-tail 0,0.5,1,1.5,2,3,4,6. The span-trained baseline falls with added
#   tail context; the buffer-trained arms rise. The curve's sign follows the training view — why it never enters
#   the buffer-cap formula.
GRID='--severity-grid-head=-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3 --severity-grid-tail=-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3'
for M in baseline ar dlm; do python eval.py --rq 1 --method $M --language "$LANG" --split test --allow-test \
    $GRID --output outputs/rq1_${M}_${LANG}.json; done
#   ABLATIONS — training ablations, evaluated as trained (never eval-time switches: a gated checkpoint learned
#   under Ω, so ungated decode of it measures nothing). Priority under a page limit — one language, dlm arm; the
#   clean baseline already doubles as the no-misalignment-training ablation at zero extra cost:
#     configs/ablation_nogate.yaml           Ω coupling off (aux BIO loss kept) — the architectural claim
#     configs/ablation_nocb.yaml             confidence-bound off — pair with the B6b stability harness
#     configs/ablation_nos1.yaml             no S1 init (gate warmup 2) — what segmentation pretraining buys
#     configs/ablation_measured_jitter.yaml  measured corruption (needs B4b artifacts) — defends the designed default
#   for A in nogate nocb nos1 measured_jitter; do
#     python train.py --stage train-slt --language "$LANG" --slt-config configs/ablation_${A}.yaml
#     python eval.py --rq 1 --method dlm --method-config configs/ablation_${A}.yaml --language "$LANG" \
#         --split test --allow-test $GRID --output outputs/rq1_dlm_${A}_${LANG}.json
#     python eval.py --rq 2 --stream --method dlm --method-config configs/ablation_${A}.yaml --language "$LANG" --split test --allow-test
#   done
#   Report each vs the full arm at (0,0), the worst evidence-complete cells, and RQ2 rows 7/8. Event/stability
#   filenames auto-append the method-config stem, so ablation evals never overwrite the main arm's artifacts.

# ── B6. RQ2 — end-to-end DVC (rows 1–8 below) ──
python eval.py --emit-gold-segments outputs/gold_${LANG}_test.json --language "$LANG" --split test
python analyze.py --stage segmenter-infer --segmenter-decode duration --segmenter-arch moryossef --language "$LANG" --split test --allow-test \
    --output outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json
python eval.py --rq 2 --segments outputs/gold_${LANG}_test.json --method baseline --language "$LANG" --split test --allow-test  # row 1
python eval.py --rq 2 --segments outputs/gold_${LANG}_test.json --method ar       --language "$LANG" --split test --allow-test  # row 2 AR
python eval.py --rq 2 --segments outputs/gold_${LANG}_test.json --method dlm      --language "$LANG" --split test --allow-test  # row 2 DLM
python eval.py --rq 2 --segments outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json --method baseline --language "$LANG" --split test --allow-test  # row 3
python eval.py --rq 2 --segments outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json --method ar       --language "$LANG" --split test --allow-test  # row 4 AR
python eval.py --rq 2 --segments outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json --method dlm      --language "$LANG" --split test --allow-test  # row 4 DLM
#   Row 5 — the S1-cascade floor (matched-segmenter control): the deployed head's spans, clean AR translator.
#   There is deliberately no "S1 spans + our DLM" row; every contrast it could carry is covered by (2−1), (4−3), (8−7).
python analyze.py --stage segmenter-infer --segmenter-arch s1 --language "$LANG" --split test --allow-test \
    --output outputs/segmenter_predictions_s1_duration_${LANG}_test.json
python eval.py --rq 2 --segments outputs/segmenter_predictions_s1_duration_${LANG}_test.json --method baseline --language "$LANG" --split test --allow-test  # row 5
#   Row 6 — online S1 cascade. S1 proposes spans; the clean translator encodes each candidate crop.
#   The shared FSM checks boundary stability, lag and the translator's confidence before it commits.
#   --match-geometry reads the comparison arm's saved cap, delta and minimum span.
#   Run both rows with the same inference.yaml for stride, lag and confidence threshold.
#   The S1 checkpoint comes from --bio-config. --checkpoint here names the clean translator.
python eval.py --rq 2 --stream --segmenter-arch s1 --method baseline \
    --checkpoint checkpoints/baseline_train/"$LANG"/model.pt --match-geometry checkpoints/ar/"$LANG" \
    --language "$LANG" --split test --allow-test
#   Output: outputs/rq2_stream_events_s1_${LANG}_test.json. --no-translate is a segmentation diagnostic only.
#   Row 9 — online Moryossef cascade, under the same comparison arm's streaming policy.
#   Requires duration_model.moryossef.<language> from its B1a tune-decode run.
#   --moryossef-config names the segmenter configuration; --checkpoint names the clean translator.
python eval.py --rq 2 --stream --segmenter-arch moryossef --method baseline \
    --checkpoint checkpoints/baseline_train/"$LANG"/model.pt --match-geometry checkpoints/ar/"$LANG" \
    --language "$LANG" --split test --allow-test
#   Output: outputs/rq2_stream_events_moryossef_${LANG}_test.json.

#   Rows 7/8: the same trained model, offline (self-segments, one-shot) vs streaming (FSM).
python eval.py --rq 2 --offline --method ar  --language "$LANG" --split test --allow-test   # row 7'
python eval.py --rq 2 --offline --method dlm --language "$LANG" --split test --allow-test   # row 7

# ── B6b. Display stability — how much earlier could text appear, and at what cost? ──
#   --stability replays stable-prefix policies (commit_only, agreement_nK, confidence_nK, both) over the
#   per-stride decodes the FSM already computed. No extra decoding; all policies are monotonic, so the trade is
#   latency vs prematurely-frozen tokens. confidence_n1 is the row that tests the CB claim: run it also against
#   a CB-off checkpoint — if the policy works only on the CB-trained model, the CB term earns a deployment
#   metric, not just BLEU. Results: outputs/stability_<method>_<lang>_<split>.json
#   THE CB PAIRING: rerun --stability with --method-config configs/ablation_nocb.yaml — if confidence_n1 is a
#   good policy only on the CB-trained model, the confidence-bound term earns its place here.
python eval.py --rq 2 --stream --stability --method ar  --language "$LANG" --split test --allow-test  # row 8'
python eval.py --rq 2 --stream --stability --method dlm --language "$LANG" --split test --allow-test  # row 8

# ── B6c. System error report — WHERE the deployed system fails, from the events B6 already wrote ──
#   Per gold sentence {matched, merged, split, missed} and per event {matched, merger, fragment, phantom}
#   (mutually exclusive; forced-PARTIAL counted orthogonally), frequencies x mean sentence-BLEU; sentence-BLEU
#   vs tIoU bins (boundary-induced vs translation-intrinsic loss); duration-binned match rates incl. the
#   over-cap tail; and AUTO-SELECTED case studies (top-k per failure type by reference length — paper exemplars
#   are selected by rule, never cherry-picked). Run per events file (rows 7/8 and any cascade row).
python analyze.py --stage system-errors --language "$LANG" --split test --allow-test \
    --predictions outputs/rq2_stream_events_dlm_${LANG}_test.json
#   Figures (one dashboard per run; --report repeatable to OVERLAY systems — any events file works: stream,
#   offline, every cascade row — so failure profiles are comparable across pipeline stages; --rq1-file adds the
#   confidence-calibration panel, --stability-file the reveal-policy trade-off; case timelines are rendered from
#   the report's rule-selected exemplars):
python visualize.py --what errors --language "$LANG" --split test \
    --report outputs/system_errors_rq2_stream_events_dlm_${LANG}_test.json \
    --rq1-file outputs/rq1_dlm_${LANG}.json
#   FIGURES from the report(s): one dashboard (taxonomy shares, BLEU-vs-tIoU, duration-binned match rates,
#   pair scatter, BLEU-vs-OOV, event taxonomy) + one timeline sheet of the rule-selected case studies.
#   Repeat --report to overlay systems (e.g. stream vs offline vs cascade) on shared axes.
python visualize.py --what errors --language "$LANG" \
    --report outputs/system_errors_rq2_stream_events_dlm_${LANG}_test.json
#   STATISTICS LAYER (report.py) — what a dashboard cannot give: every system re-scored under ONE protocol on the
#   SAME gold index (stable gold_id join), per-system per-gold outcome CSVs (drill into any cell with pandas),
#   PAIRED-bootstrap significance on the per-gold deployment score vs --reference (mean Δ, 95% CI, p), and the
#   sentence-level flip table (fixed / broken / better / worse). Any RQ2-schema events file is a system:
#   streaming, offline, every cascade row, every ablation — a claimed difference without its CI does not go in
#   the paper. -> outputs/report_<lang>_<split>/{outcomes_*.csv, taxonomy.csv, significance.csv, report.md, *.png}
python report.py --language "$LANG" --split test --reference stream \
    --events stream=outputs/rq2_stream_events_dlm_${LANG}_test.json \
    --events offline=outputs/rq2_offline_events_dlm_${LANG}_test.json \
    --events cascade=outputs/rq2_cascade_segmenter_predictions_moryossef_duration_${LANG}_test_baseline.json
```

**RQ1 design.** One grid, one corpus, three arms. Normalize each curve to its own (0,0) cell; report the intercept separately. Use `--severity-grid-head=…`/`--severity-grid-tail=…` (with `=` for negative-leading lists); the sweep is their full product.

- **Gated arms: read (decoded-only, skip-rate) pairs, not raw corpus BLEU.** A Δ_head > 0 window has no `B`; the FSM skips that state by design, and force-decoding it collapses the cell's corpus BLEU through the brevity penalty. The table therefore reports `gate_skip_rate` and `text_metrics_decoded_only` beside `text_metrics`. Never plot decoded-only alone: it conditions on a shrinking, easier subset.
- Use full-reference BLEU only for evidence-complete cells (head ≤ 0, tail ≥ 0). Cells that remove target evidence test the skip/confidence policy, not translation quality.

**RQ2 ladder** — conditions over {segmenter source × translator × mode}; each delta varies one axis:

| # | Span source | Translator | Evaluation |
| --- | --- | --- | --- |
| 1 | GT | Clean AR | `--segments gold_… --method baseline` |
| 2 | GT | Joint AR/DLM | `--segments gold_… --method ar/dlm` |
| 3 | Moryossef26, enhanced BIO | Clean AR | `--segments …moryossef_duration_… --method baseline` |
| 4 | Moryossef26, enhanced BIO | Joint AR/DLM | `--segments …moryossef_duration_… --method ar/dlm` |
| 5 | S1, offline duration BIO | Clean AR | `--segments …s1_duration_… --method baseline` |
| 6 | S1, online FSM | Clean AR | `--stream --segmenter-arch s1 --method baseline --match-geometry <arm>` |
| 7 | Joint head, offline | Joint AR/DLM | `--offline --method ar/dlm` |
| 8 | Joint head, online FSM | Joint AR/DLM | `--stream --method ar/dlm` |
| 9 | Moryossef26, online FSM | Clean AR | `--stream --segmenter-arch moryossef --method baseline --match-geometry <arm>` |

**Required same-span control** — re-translate row 7's saved spans with the clean translator ("clean translator + S2 spans"):

```bash
python eval.py --rq 2 --segments outputs/rq2_offline_events_dlm_${LANG}_test.json \
  --method baseline --language "$LANG" --split test --allow-test
```

**How to read the ladder.** Compare the same target language and decoder family.

- **2 vs 1; 4 vs 3:** translation at fixed oracle or Moryossef spans.
- **5 vs 3:** segmentation pipelines with the same clean translator; report the input/context differences.
- **8 vs 6; 8 vs 9:** the joint system against online S1 and Moryossef cascades under matched streaming geometry and policy. Compare localization separately from text scores.
- **8 vs 7:** the same joint model under online and offline access. This includes the effect of future context.

Notes that prevent misreading, in brief:

- Every row uses the same annotation-ignore policy (`eval.scoreable_predictions`). Keep all emitted events, including short correct and false predictions. Λ_min is an event-generation rule, not an evaluation filter. Report localization precision and recall beside DVC.
- Rows 1–5 use supplied spans; row 6 discovers spans online. For rows 1–5, only the span _boundaries_ are external. `run_cascade` still runs the full model (BIO head and gate included) on `--method dlm/ar` rows. Rows with `--method baseline` use the ungated clean floor — deliberately a different model.
- Row 7 chunks the head at its TRAINED cap (checkpoint meta), overlap-stitched, each chunk pose-normalized on its own frames (the head's training frame), and translates each span in a buffer-shaped window. No whole-video pass, no random sampling at eval; rows 7–8 are deterministic.
- The misaligned AR twin runs the identical commands with `--method ar` (rows 2′,5′,7′,8′) to isolate the decoder family. The headline table stays DLM.

**Metric.** RQ2 uses DVC-style text scoring: every prediction/reference pair above the tIoU threshold is a single-reference instance. Unmatched predictions receive a seeded random garbage reference. Scores are computed per gold video, averaged equally across videos, then averaged over the declared tIoU thresholds. The repository uses SacreBLEU 13a rather than the original toolkit's COCO BLEU/PTB tokenizer; describe this as DVC-style matching and aggregation with the stated text scorers, not a byte-identical reproduction.

A missed gold caption creates no DVC pair, so report localization precision, recall and F1 alongside text scores. The companion SODA-style fusion uses one-to-one temporal matching and charges predicted/gold counts. Neither RQ2 BLEU column is numerically interchangeable with RQ1 corpus BLEU. Compare the clean GT-span control and cascades under the same RQ2 scorer, checkpoint, beam count and annotation-ignore policy. The tune-stream stage selects on segmentation F1 rather than text scores. SODA-style fusion per video and threshold uses the following counts:

&nbsp;&nbsp;&nbsp;&nbsp;`segmentation.f1 = 2|M|/(n_p+n_g)` · `S = Σ_(i,j)∈M s_ij`, `p = S/n_p`, `r = S/n_g`, `text = 2pr/(p+r)`

- The fusion charges spurious predictions (n_p) and missed gold (n_g) in one number, which is why it accompanies the recall-blind headline. Matched-pairs-only means are rejected outright. BLEU here is smoothed sentence-BLEU; CIDEr has no per-pair form, so it appears in the densevid rows only.
- **Provided-boundary translation** is measured by the GT-span rows (1–2), which are explicit oracle controls. RQ1 (0,0) supplies a clean GT-trimmed clip but gives the joint gate no known-start flag or internal boundary. It is a clean-input anchor, not an identical conditioning path to the supplied-proposal controls.
- **segmenter-eval and RQ2** report the same decode under two protocols, both in the segmenter-eval JSON: the Moryossef-comparable block (per-video mean of F1s over UNK-masked BIO gold — cross-paper comparability only) and `rq2_protocol` (caption-unit gold, ignore-region filtering, F1 of macro P/R — computed by the RQ2 code path itself, so it matches the cascade rows' segmentation block by construction). Quote `rq2_protocol` wherever localization is compared across tables; the residual delta between the two blocks is gold construction plus aggregation, and it is now a printed pair, not an open question.
- Quote `segmentation.recall` beside any RQ2 text column, and quote tune-stream's held-out F1, never its selected cell.

## S1 continued-training study (`asf` and `bfi`)

Keep `[ase, asf, bfi]` in the multilingual phase. The target language is included throughout that phase.
Call its direct evaluation **pooled evaluation without target fine-tuning**, not zero-shot.
All three conditions use the same S1 architecture and released pose initialization; the target-only control must remain S1.

| Condition | Training schedule | Purpose |
| --- | --- | --- |
| 1. Target-only | Target-language S1 training | Reference for the value of multilingual training |
| 2. Pooled, no target fine-tune | Train on `[ase, asf, bfi]`; evaluate that checkpoint on each target | Multilingual model before specialization |
| 3. Pooled → target fine-tune | Start from the exact checkpoint in case 2; continue on each target separately | Effect of target specialization after multilingual training |

An example schedule is 50 pooled epochs followed by 20 target epochs. Set the target-only training budget before the comparison; adding phase epoch counts does not produce an equal-compute baseline.
These are not equal-compute conditions: pooled and target epochs contain different numbers of batches and target windows.
Record optimizer updates, target-language exposure and training time. If claiming equal-budget gains, match optimizer updates and batch size for cases 1 and 3 before running them.
Case 2 is the shared intermediate checkpoint for case 3, not another model selected to improve the comparison.
A lower case-2 result can reflect a compromise between languages or less target-specific training; it does not by itself establish a code defect.

Set a common numeric `pretrain_geometry.buffer_cap_s` from training data before the study. Freeze one per-language duration decoder across the three pretraining conditions; do not attribute separate decoder retuning to pretraining. Keep window construction, augmentation, BIO loss, initialization family and evaluation fixed.
Use dev for checkpoint selection, freeze the selection rules before test, and use a distinct output directory per run.
The default early-stopping and best-checkpoint settings make the phase lengths upper limits. Report the actual selected epochs.

1. Run the pooled phase with `pretrain_languages: [ase, asf, bfi]`, the released `checkpoint.from_pretrained`, and a distinct `checkpoint.dir`: `python train.py --stage train-bio --epochs 50`. Do not pass `--language` for this phase.
2. Save the selected pooled `model.pt` as the shared case-2 checkpoint. Evaluate it on both targets using the pooled config: `python eval.py --segmenter-eval --segmenter-arch s1 --language <target> --checkpoint <pooled-model.pt> --split test --allow-test --output <case2-output.json>`.
3. For case 1, set `pretrain_languages: null`, retain the released initialization, and choose a distinct target-only directory. Run `python train.py --stage train-bio --language <target> --epochs <target-only-budget>` using the predeclared budget.
4. For case 3, keep `pretrain_languages: null`, set `checkpoint.from_pretrained: <pooled-model.pt>` and a separate target-adaptation directory. Run `python train.py --stage train-bio --language <target> --epochs 20`, without `--resume`.
5. Evaluate cases 1 and 3 with the same target test data and monolingual config. Use distinct `--output` paths. Restore the pooled configuration before main experiments that expect that checkpoint.

Case 3 continues both pose-encoder and BIO-head weights. The current fine-tuning interface starts a fresh optimizer and schedule; it does not carry Adam moments across the phase boundary.
`--resume` is for interruption within the same recipe and schedule, not for switching the language pool.
The main AR/DLM stages need only the selected S1 initialization; this segmentation study does not require separate translators for each case.

Interpret **3 vs 1** as the test of the pooled-then-target recipe, and **3 vs 2** as the effect of further target training. For the separate claim that text improves localization, compare joint training with the case-3 segmentation-only continuation under the same pooled start, target windows, BIO update budget and checkpoint-selection policy. Comparing only against the unadapted pooled checkpoint does not separate text support from extra target BIO training.
If case 3 remains worse than case 1 under the declared protocol, do not claim that multilingual pretraining improved that result.
The external Moryossef baseline tests a different visual encoder and cannot replace this same-architecture control. Choose its release by a stated comparison protocol, not by which version scores lower. Credit the 2026 public code even when no separate paper is cited; only our specific additions can be claimed as ours.

### Segmentation evaluation and metric meaning

The main streaming evidence must come from chronological online evaluation with equal observed frames, capacity, stride and latency policy.
An offline whole-video result remains a useful reference. The four-mode window test is a controlled diagnostic, not an online event evaluation.
Use identical test-video windows for all models, fixed independently of their predictions; do not reuse training examples or choose mode weights after seeing results.
For modes 1/3, evaluate eligible complete-sentence localization. In mode 2, distinguish visible-fragment labeling from the decision to wait for a complete sentence. In mode 4, report false spans/false commits.
Report modes separately: their training mixture is a sampling design and need not match how often they occur in a real stream. A one-sentence crop can make an all-signing prediction score well by cutting it at the supplied window edges; it does not establish boundary detection or correct commit timing.
The current CLI supplies whole-video segmentation and online S1/Moryossef/joint evaluation.
Streaming capacity is enforced before encoding. EOF clock advances do not count as new boundary evidence; forced overrides are flagged. A common four-mode benchmark remains a proposed comparison, not an implemented entry point.

| Output | What it measures |
| --- | --- |
| `bio_iou` | Raw-head signing-frame overlap, with B and I combined; averaged per video over reliable labels |
| `phrase_tiou_f1@t` | One-to-one sentence matches at interval tIoU threshold `t`; averaged per video |
| `phrase_frame_f1` | Macro F1 across the separate O/B/I frame classes |
| `phrase_segment_count_ratio` | Predicted/gold span count, averaged per video; not boundary accuracy |

Three consecutive gold sentences can be merged into one predicted span with identical signing-frame coverage.
Then `bio_iou` is perfect, while no sentence meets tIoU 0.5 if the three gold sentences have equal length.
High `bio_iou` therefore does not imply accurate sentence boundaries. Use sentence precision/recall/F1 for that claim.
Do not compute IoU from the already averaged precision/recall: the evaluator computes it within each video and then averages.
Use `rq2_protocol` for RQ2 comparisons. It ignores unsupported annotation regions and retains short emitted predictions.

**Before a new joint run:** duration conditioning and the CB eligibility fixes change the training recipe. Start fresh AR/DLM runs; old joint weights are not evidence for this recipe. S1 weights remain usable because its BIO architecture/loss is unchanged. Recalibrate each target language, then reevaluate its cascades and joint models. Keep a clean translator only when its normalization and training capacity remain the same.

The `asf` duration row may be reused only with the S1 checkpoint and data used to select it. Untuned languages are refused. A small diagnostic subset must not write final settings. Decoder fit/selection, boundary tolerance, capacity and lag are separate steps; tuning duration scores alone does not complete streaming calibration.

The implemented mask is not Masked Transformer’s hard/learned confidence mixture. See `docs/membership_gate.md` §1.5 for the equations and the distinction. No self-box mask BCE is used.

## Key design decisions

One line each; the full argument lives at the pointer.

- **Caption units merge whole cues without retiming.** Units can contain several linguistic sentences; BIO labels describe unit membership. Incomplete coverage remains excluded. → `docs/data_pipeline.md`
- **Cross-split de-duplication is train-side** (decontamination convention); the split CSV is mandatory. → `configs/data.yaml`
- **Wrong-language videos are dropped per video by non-Latin script share** (`max_non_latin_ratio`). → `docs/data_pipeline.md` §5b
- **Scrolling tracks with no marked boundary are dropped per video** (`min_marked_boundary_ratio`): a boundary is marked by terminal punctuation or by a POSITIONAL capital (a word the train lexicon says is ordinarily lowercase — `NDIS` does not count). A video goes only when its cues also scroll, i.e. overlap in time, so the boundary is a display event rather than an utterance. Both conditions are needed, and the scrolling half carries the weight: across every bfi video 93 fall below the ratio but only 1 scrolls, and the other 92 are vocabulary and fingerspelling practice whose cues are single signs (`RUBBISH`, `MESS`, `CLEAN`) — real timed boundaries a punctuation-or-capital test cannot see. Dropping on the ratio alone would delete them. 0 (asf), 3 (ase), 1 (bfi) videos. Applied to every split. → `docs/data_pipeline.md` §3
- **BIO loss**: class-weighted CE plus binary signing Dice in S1 and joint training. Caption loss also updates shared temporal features and the BIO classifier through first-span membership.
- **Terminator = first O-or-B, never "closing O"** — back-to-back sentences have no gap; same rule at training and inference. → spec §5.3
- **Decoder controls**: compare plain and enhanced decoding on both heads from the same logits. Use the enhanced Moryossef cascade as the external comparison and the online S1 cascade for the joint-training control.
- **First-span coupling**: training sums valid-path probabilities before forming Ω. It does not select a hard training boundary or replace the predicted mask with a GT interval. GT determines whether the training view has a complete or open target, not its predicted boundaries.
- **S1 pretrains competence before coupling**, pooled, under the designed corruption; both stages share one window distribution. Measured (`segmenter-errors`) calibration is an ablation, not the recipe. → `configs/bio_pretrain.yaml`
- **The confidence-bound term is Mode-2a only** — the FSM never decodes the other truncation states. → spec §5
- **Best-checkpoint monitor**: clean translation uses dev BLEU; joint arms use `val_joint_score` (dev-window F1 times BLEU). This is a trade-off score, not DVC or a retention guarantee. Report both components.
- **Text scoring level is declared per language, never sniffed from references** (`char_level_for_target`; `tests/test_scoring_level.py` enforces every call site).
- **Calibration artifacts are keyed by (segmenter, language)** so an `--segmenter-arch s1` run can never overwrite the independent measurement (`tests/test_calibration_provenance.py`).
- **Pose timing comes from `video_meta.csv`** (SignVerse resolves to exactly 24 fps). → `docs/run_real_data.md`

## Repository map

```
backbones/   Uni-Sign 69-kp 4-part ST-GCN (UniSignPoseEncoder)
poses/       normalize_keypoints_unisign (133→69) · pose_io (load_pose_window, per-video fps) · augmentation
data/        loader (YouTube-SL-25 + pooling + dedup) · windowing (BIO, first-complete-span, χ) · jitter · batch
models/      block_diffusion · dmax (OPUT + SPD/DCD) · membership_gate (soft first-span Ω) · front_end · unisign ·
             streaming_slt (MisalignedSLTModel) · bio_head
train/       slt (AR/DLM trainer) · bio_pretrain (S1) · sampler (window modes) · losses (Dice+CE, CB) · helpers
moryossef26/ faithful external segmenter (raw-kp UNet): model · dataset · trainer · infer. NOT the FSM head.
infer/       duration_decode (semi-Markov BIO Viterbi) · commit_gate · decode (SPD+DCD) · stream (FSM) · stability
metrics.py   BIO monitor · tIoU segments · text metrics (declared scoring level)
utils.py     load_yaml (extends, ${language}, ${corpus}) · checkpoint_dir · pick_device
train.py     --stage {smoke-data, train-bio, train-moryossef, train-slt}
prepare_data.py  SignVerse-2M shards → language layout
eval.py      --rq {1, 2} · --segmenter-eval · --emit-gold-segments · --stability
analyze.py   --stage {dataset-summary, segmenter-infer, tune-decode, tune-stream, segmenter-errors, buffer-cap, delta-enc, loss-balance}
report.py    statistics layer over RQ2 events files: shared-gold-index outcome CSVs per system,
             paired-bootstrap significance + flip tables vs a reference (outputs/report_<lang>_<split>/)
visualize.py --what {poses, losses, predict, errors}
docs/        membership_gate.md · segmentation_decoding.md · data_pipeline.md · run_real_data.md · implementation_notes.md · literature_notes.md
```

## Configuration

| config                    | drives                                                                                  | inheritance                                      |
| ------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `dlm.yaml`                | the DLM method; single source of truth for sampler, BIO-head arch, gate, optimizer keys | —                                                |
| `ar.yaml`                 | gated AR de-risk (§9.3)                                                                 | `extends: dlm.yaml` (decoder + output dir only)  |
| `baseline_eval.yaml`      | clean baseline: ungated greedy AR, eval-only                                            | `extends: dlm.yaml` (gate/CB off)                |
| `baseline_train.yaml`     | trains the clean floor: mode1-only, ~0 jitter, `lambda_bio: 0`                          | `extends: baseline_eval.yaml`                    |
| `bio_pretrain.yaml`       | S1 pooled segmentation pretraining (`train-bio`)                                        | `extends: dlm.yaml` (shared window distribution) |
| `moryossef26.yaml`        | external Moryossef segmenter (`train-moryossef`)                                        | standalone                                       |
| `data.yaml` / `eval.yaml` | corpora, splits, `target_lang`, `pretrained_slt`, subtitle pipeline / RQ grids          | —                                                |

Per-language constants, in the order B1 derives them (`--write-config` writes each):

| constant                          | source                                                                                                                               | writes                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| `duration_model` | B1a `tune-decode`: train duration fit and dev decoder selection | `inference.yaml`, per head/language |
| `delta_enc_frames` (δ) | B1a `delta-enc` (S1 head, deployed decode) | `inference.yaml` per-language row |
| `span_selection.min_span_frames` (Λ_min) | the shortest unit the sampler, the gate's posterior and the FSM may target. A LABEL-domain floor: set it from the unit-duration distribution, not from δ. `delta-enc` falls back to δ+1 when the row is absent, and on a corpus with a large δ that fallback puts real units out of reach (bfi δ=53 ⇒ Λ_min 2.25 s, above 16 % of bfi units) | `inference.yaml` per-language row |
| `commit_lag_s` | B1c `tune-stream --segmenter-arch s1` on dev, lag only | `inference.yaml` per-language row |
| `buffer_cap_s`                    | B1b `buffer-cap --split train` = train-split p99 + stride + δ/fps (label-only, model-free; runs after delta-enc)                     | `inference.yaml`                     |
| `bio_class_weights`               | `balanced` — resolved from measured label counts at train start, logged                                                              | automatic                            |
| pooled S1 context                 | `bio_pretrain.yaml pretrain_geometry.buffer_cap_s: auto` = max over the pool of train p99 + stride + 1 s (covers deployed caps when δ/fps ≤ 1 s; a number is an explicit override); the trained value is pinned in checkpoint meta and B1b refuses a cap above it | train-bio, automatic |

Rules that are not optional: run tune-decode before delta-enc, then delta-enc before buffer-cap (the cap reads δ); buffer-cap runs on the train split (label-only constants are measured on train, model-dependent constants are selected on dev; the stage refuses any other split). One shared `inference.yaml` serves every language — the measured constants live in PER-LANGUAGE rows (delta, minimum span, capacity and commit lag), resolved for the active language at load and refused loudly when the row is missing; the sampler, gate and FSM share target eligibility. A changed annotation fingerprint requires a new duration fit — `require_annotations` refuses to start stage 2 until the fit matches the loaded records, so the calibration chain is on the critical path for every experiment, not only for a decoder change.

Λ_min is the one constant whose source matters more than its value. It is the eligibility rule shared by the sampler's target, the gate posterior's first-complete span and the FSM's commit, so it decides which units the system can ever supervise or emit. Set it from the unit-duration distribution. Deriving it from δ makes ground truth a function of a trained model's boundary noise, and on a corpus with a large δ it silently removes real units: `delta-enc` prints a warning when Λ_min exceeds the p10 unit duration, and that warning must be read.

Watch `gate_anchor_hit_rate` during stage-2 training (the head's own closed span at tIoU ≥ 0.5 with the GT target, over windows that have one): at S1's level and steady = S1 delivered a usable policy and joint training keeps it. Falling = the head is decaying; find the cause before spending more GPU.

### Normalization and comparison contract

| Path | Frames used to compute the body box |
| --- | --- |
| S1 and stage-2 training, RQ1 | Supplied training or controlled window |
| Whole-video S1 and offline joint head | Each chunk, at the checkpoint's trained context |
| Streaming joint model | Active raw buffer only |
| Online S1/Moryossef cascades | Active buffer for segmentation; selected raw crop for the clean translator |
| Offline Moryossef adaptation | Whole video, reused by each training chunk |

The reference Moryossef code uses different landmarks and shoulder/mean-std normalization. Our model uses the Uni-Sign input transform.
A Moryossef checkpoint trained with boxes computed from individual chunks needs retraining for the whole-video-box training contract.
Regenerate affected segmenter spans after a normalization change. For S1 or joint-model changes, recheck the B1 calibration before training dependent arms.
Streaming calibration must use buffer normalization. No comparison requires equal crop boxes across models with different training inputs.
See [implementation notes](docs/implementation_notes.md) for the current contracts and [membership gate](docs/membership_gate.md) for the loss and gradient paths.

## Reruns for the shared temporal / first-span architecture

| Stage | Required action |
| --- | --- |
| Pooled S1 | Existing weights are compatible. Rerun its evaluation and per-language B1 calibration. Retrain only if selecting a new S1 checkpoint. |
| Moryossef | No architecture change in this update. If its training normalization is unchanged, keep the checkpoint and its own calibration. |
| Clean translator | Keep it if its input normalization and training capacity are unchanged. A changed B1b capacity can require retraining. |
| AR and DLM | Train both from S1 plus the clean translation warm start in fresh directories. Do not resume a checkpoint from another architecture. |
| Evaluation | Regenerate S1 spans, online S1 cascade, joint offline/online events and the joint RQ1 curves. Use one clean translator for the cascade rows. |

B1 runs per language for ase, asf and bfi. Archive result files before reusing their output paths.
S1 tuning no longer writes duration parameters. `tune-stream` writes only `commit_lag_s`.
The same first-span probabilities condition captioning during training and streaming. Externally supplied spans are explicit fixed-boundary controls.

Names and runtime-only changes preserve checkpoint weights, duration scores and the loss. A run that already uses the current mapper, duration model and CB recipe can resume from `latest.pt` with the same settings. No recalibration is needed for these changes; the runbook order is unchanged.

The duration calibration fingerprint hashes the numerical parameters only. If an existing saved fingerprint is rejected after a fingerprint-format change, rerun B1 `delta-enc --split dev --write-config` with the same checkpoint and settings. The hash change alone requires no decoder tuning or retraining. If the measured tolerance changes, follow the usual dependent capacity/lag steps.

## Reading BLEU across RQ1 and RQ2

| Output column | Calculation |
| --- | --- |
| `translation_bleu4` | Corpus BLEU over the full RQ1 or generated-dev caption set |
| `densevid_bleu4` | Corpus BLEU within each video's tIoU-paired captions, followed by an equal-weight mean over videos |
| `soda_bleu4` | Smoothed sentence BLEU on one-to-one matches, normalized by predicted/gold counts, then averaged over videos |

These columns answer different questions. Shared text preprocessing does not make their numbers interchangeable. DVC includes garbage references for unmatched predictions and no separate pair for each missed GT sentence. A threshold also changes which references enter the score.
Compare a cascade with the existing GT-span baseline row using the same RQ2 column, clean checkpoint, beam count and annotation-ignore policy. Keep RQ1 corpus BLEU as its own robustness result. GT-span captions are an oracle-input control, not a mathematical upper bound on every caption metric. The repository's DVC-style scorer uses SacreBLEU 13a instead of the original toolkit's COCO BLEU/PTB tokenizer; disclose that choice.

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

The soft duration model does not impose a maximum sentence length. If boundary evidence is weak, an offline prediction can merge many sentences. This is a localization error, not a reason to remove that prediction from evaluation.
The S1 and Moryossef offline cascades use the same full-proposal translation path. For a bounded streaming comparison, use both online cascades under matched geometry. Do not add sentence boundaries at neural chunk edges or change only one baseline's caption context after inspecting test results.


The joint `--offline` row also preserves each complete predicted interval. Its caption window includes the normal lead-in/context and extends when necessary to cover a long proposal. The live buffer cap limits only `--stream`; it must not shorten offline event timestamps. Given-interval captioning computes the interval mask directly, without an unused latent-span pass.
After this correction, rerun joint `--offline` rows if their predicted spans exceed the context extent. Existing S1/Moryossef cascade rows and streaming events are unchanged. No training or decoder recalibration is required. Cascade event provenance records the resolved translator checkpoint, including when the CLI uses its default.
