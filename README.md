# Misaligned SLT

Pose-only, gloss-free, **streaming** sign-language-to-text translation. The system is robust to sentence-boundary **temporal misalignment** (imperfect segmentation).

**Central rule.** Training windows match the windows the streaming loop produces. A truncated visual input never gets a partial text label (premise **P1**). Every segmenter error becomes a window-shape variation, never a label variation (**P2**). Design spec: [`streaming_slt_prompt.md`](streaming_slt_prompt.md). The segmentation→translation coupling: [`docs/membership_gate.md`](docs/membership_gate.md).

**Data.** YouTube-SL-25: ASL (`ase`), Auslan (`asf`), BSL (`bfi`). All targets are English. All poses come from SignVerse-2M at a uniform 24 fps. The full load-time pipeline (caption cleanup, sentence reconstruction, quarantine, de-duplication, pooling) is documented in [`docs/data_pipeline.md`](docs/data_pipeline.md). Read it before you change anything that touches ground truth.

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
DWPose 133-kp stream
  └─ pose_io.load_pose_window        normalize 133 → Uni-Sign 69 (bbox crop-scale, conf-clipped)
       └─ Uni-Sign 4-part pose ST-GCN (69 kp)      per-frame pose tokens (T preserved)
            ├─ Conv1d stem → RoPE-rel-time BIO head ──► phrase B/I/O ──► streaming FSM (terminator = O-or-B)
            │        │                                        │
            │        └──────── membership gate Ω(t) ◄─────────┘  (segmentation belief → decoder cross-attention)
            └─ task prompt prepended → LM encoder (mT5 | mBART) ─► DLM decoder (OPUT train / SPD+DCD infer, gated by Ω)
```

- **Pose encoder** ([`backbones/`](backbones)): Uni-Sign 4-part ST-GCN, loaded from the released `*_pose_only_slt.pth`. [`poses/`](poses) reproduces Uni-Sign normalization byte-exactly.
- **Front end** ([`models/unisign.py`](models/unisign.py)): one pose encoder, two LMs (mT5 default; mBART ablation). `MisalignedSLTModel` takes either, so the heads, sampler, and FSM are written once.
- **Decoder** ([`models/block_diffusion.py`](models/block_diffusion.py), [`models/dmax.py`](models/dmax.py)): BD3LM core + DMax (OPUT training, SPD/DCD inference). Verified faithful to the released DMax/DCD code.
- **BIO head** ([`models/bio_head.py`](models/bio_head.py)): its own RoPE-relative-time transformer with a Conv1d stem, over the pose tap.
- **Membership gate** ([`models/membership_gate.py`](models/membership_gate.py)): Ω(t) = γ·ln(m∨ε) + ln(1−χ+ε), an additive bias on decoder cross-attention. It is the only decode-time channel between the heads; both decoder arms use the same Ω. Full derivation: [`docs/membership_gate.md`](docs/membership_gate.md).
- **Streaming FSM** ([`infer/stream.py`](infer/stream.py)): sawtooth WATCH→TRANSLATE→COMMIT. A span's terminator is the first following `O` or `B`. The target is the first complete span ≥ **Λ_min** frames. A commit cuts the buffer at terminator − δ. The buffer's last sentence is right-censored: its tag stream comes from the FSM's own decode triple (`duration_decode_s1_stream`), a split that opens the censored tail survives only if the closed-tail decode agrees within δ, and a duration-decided `B` terminator commits only once the buffer extends `commit_lag_s` past it (fixed-lag decoding; an `O` terminator commits at once).

## Multi-GPU (torchrun)

Prefix any `train.py` command with `torchrun`; change nothing else:

```bash
torchrun --standalone --nproc-per-node=4 train.py --stage train-slt --language "$LANG" --slt-config configs/dlm.yaml
```

- Config `batch_size` is the GLOBAL batch, split across ranks. A non-divisible batch is a hard error.
- `mixed_precision: auto` picks bf16 on compute capability ≥ 8. Multi-GPU + fp16 is refused (per-rank scaler drift).
- `latest.pt` is a full resumable snapshot, written every epoch. Continue with `--resume`. Re-running without `--resume` over an existing `latest.pt` is refused.
- Eval and analysis stay single-process; use `--batch-size` there. Gradient averaging is explicit, not DDP ([`train/distributed.py`](train/distributed.py) explains why).

## The experiment sequence

Two blocks. **Stage A** (segmentation pretraining) runs once, pooled over all three languages. **Stage B** runs once per target language, in three phases: **MEASURE → TRAIN → EVALUATE**.

|                                   | **Stage A — language-agnostic**               | **Stage B — per language**                                                                       |
| --------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Runs                              | once, for all languages                       | once per target language                                                                         |
| `--language`                      | **refused** (a pool has no target)            | required                                                                                         |
| Produces                          | `checkpoints/{moryossef,bio_s1}/multi_<pool>` | `checkpoints/{baseline_train,ar,dlm}/$LANG`, `outputs/*_$LANG_*`, the `inference.yaml` constants |
| Reads `data.yaml pretrained_slt`? | no — released warm starts only                | yes (`utils.resolve_pretrained`)                                                                 |

Stage A never reads `pretrained_slt`, so the B2b re-root cannot reach it. The two blocks are independent. `--language "$LANG"` re-points the dataset and every `${language}` path; no config edits.

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
#   auto context = pool train p99 + stride + 1 s, plain-argmax monitor) is documented in configs/bio_pretrain.yaml.
#   Checkpoints stamp their pool (meta.pretrain_pool); every loader refuses a pool mismatch.
#   Monitor = val_mode3_tiou_f1. Compare checkpoints with --segmenter-eval, never by monitor value.
python train.py --stage train-bio              # -> checkpoints/bio_s1/multi_ase-asf-bfi/model.pt

# ═══ STAGE B — per target language. MEASURE (B1) → TRAIN (B2, B3) → EVALUATE (B4, B5, B6). ═══
# WHAT A CHANGE INVALIDATES (rerun only what is listed; everything else keeps):
#   change in Stage A (S1 or Moryossef retrained)    -> all of B1, then B3 and every evaluation that uses the arms
#   change in data preprocessing / GT construction   -> everything, Stage A included
#   change in the FSM decode (terminator rule, streaming triple, commit lag, tune-stream) -> B1a', B1b (delta), B1c if
#       delta moved, then B3 (the gate trains under that decode) and rows 7/8, RQ1 for the arms; NOT the clean baseline
#       (no head, no gate), NOT the cascade rows 1-6 (their spans are whole-video), NOT Stage A
#   change in a language's buffer_cap_s (B1c re-measured) -> B2 baseline_train, B3 both arms, B1a' tune-stream and B1b
#       delta-enc for THAT language (every training window is clamped to the cap; the FSM buffer is the cap)
#   change in the arms' checkpoint monitor -> B3 and the arm rows only
#   change in RQ2 scoring (uniform Lambda_min floor, DVC columns) -> re-score events JSONs with --predictions; no training
LANG=ase      # ase (ASL) | asf (Auslan) | bfi (BSL)
python train.py --stage smoke-data --language "$LANG" --split train --num-samples 64   # loader sanity

# ── B1. MEASURE — freeze the constants that parameterize training ──
#   Three rules order everything. (1) A constant that parameterizes training is frozen before any training that
#   reads it: the sampler clamps windows to buffer_cap_s, and the gate trains under δ/Λ_min. (2) Nothing in B1
#   may depend on a trained translator — that would be circular. Every B1 input is Stage A's S1 checkpoint, the
#   data, or an earlier B1 write (B1a → B1b → B1c). (3) A constant that depends only on labels is measured on the
#   TRAIN split (buffer_cap_s); a constant that depends on a model is selected on dev (the decode triples, δ).
#   Stage A's context follows the same rule without a B1 run: `pretrain_geometry.buffer_cap_s: auto` = max over the
#   pool of train p99 + stride + 1 s (the margin covers any δ up to fps frames), so every deployed cap written in
#   B1c fits the S1 context by construction, and B1c refuses a cap above the checkpoint's stamped context.
#   Re-run B1 per language and after any S1 retrain.
#   Constants are written as PER-LANGUAGE rows in inference.yaml (duration_decode convention), so switching
#   $LANG never overwrites another language's calibration; a new language's rows are inserted by
#   --write-config, and a missing row fails loud at load (utils.resolve_inference). The membership gate's
#   δ/Λ_min are DERIVED from these rows at load (train/slt._inject_gate_geometry) — no dlm.yaml mirror.
#
# B1a — deployed semi-Markov decode triple → inference.yaml duration_decode_s1.$LANG.
#   Two dev folds, joint grid. Quote the HELD-OUT estimate, never the selected cell. One forward per video; the
#   whole (w, bias, radius) grid is decoded by one batched DP per video (duration_split_grid), so the sweep
#   costs minutes, not hours, and still equals the deployed scalar decode cell by cell (tests/test_duration_decode_grid.py).
python analyze.py --stage tune-decode --segmenter-arch s1 --language "$LANG" --split dev --write-config
#
# B1a' — the FSM's OWN triple, selected under the STREAMING decode → inference.yaml duration_decode_s1_stream.$LANG.
#   The buffer's last sentence is right-censored, so the whole-video triple is far too permissive in the FSM (a weak
#   P(B) bump near the buffer edge opens a new sentence at almost no cost and the FSM commits sentences in pieces).
#   tune-stream runs the FSM itself, segmentation-only, on the S1 head over two dev folds and sweeps split bias x commit
#   lag (max min-fold F1@0.5); it writes duration_decode_s1_stream.$LANG and boundary_stability.commit_lag_s.$LANG. The
#   FSM, the gate's anchor decode (training + eval) and delta-enc read the stream row; offline/cascade rows keep the
#   whole-video one (two estimators, two triples). The split that opens the buffer's censored tail counts as a terminator
#   only if the closed-tail decode agrees within δ (streaming_split_tags; no knob). Dry runs of any FSM setting:
#   eval.py --rq 2 --stream --no-translate (no decoder calls, minutes). Numbers: docs/implementation_notes.md.
python analyze.py --stage tune-stream --segmenter-arch s1 --language "$LANG" --split dev --write-config
#   AFTER B3 (optional): re-select ONLY the commit lag on the deployed arm's head (the bias is a training input through
#   the gate and stays; the lag is a commit policy): the same command with --checkpoint checkpoints/{ar,dlm}/$LANG/model.pt
#   --grid-bias <pinned bias> --grid-lag 2 3 4 5 (stage-2 layout accepted).
#
# B1b — gate geometry: δ (the S1 head's boundary-noise floor, under the B1a' streaming decode) and Λ_min = δ+1.
#   Indifferent to the data.yaml warm start. If the write changed Λ_min, run once more; it settles in 2 passes.
#   δ is a p90, so on a large dev split `--num-sentences 3000` (a seeded random subset) estimates it in a
#   fraction of the time; re-run only when the S1 checkpoint or the B1a triple changes.
python analyze.py --stage delta-enc --language "$LANG" --write-config
#
# B1c — buffer capacity: buffer_cap_s = TRAIN-split p99 sentence duration + stride_s + δ/fps. Model-free by rule 2,
#   measured on train by rule 3 (a label-only constant; the stage refuses dev and test). A translator-dependent tail
#   elbow never enters the cap; the B5 context-appetite sweep is diagnostic only. Refuses to write a cap above the
#   S1 checkpoint's trained context (rope_eval_chunk_s): retrain S1 with `auto` instead of running the head out of range.
python analyze.py --stage buffer-cap --language "$LANG" --split train --write-config

# ── B2. TRAIN the clean floor — the shared init for every later stage ──
#   Faithful Uni-Sign SLT transfer on a caption-trimmed training view: mode1-only, ~zero jitter, lambda_bio 0.
#   Windows clamp to the B1c cap; configs/baseline_train.yaml documents why a stale cap biases this arm.
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

# ── B3. TRAIN the arms — stage-2 fine-tune under the gate ──
#   Both arms train under the membership gate, on the S1 init, with the B1 constants. Requires the B2b re-root.
#   The S1-pretrained pose encoder and BIO head train at backbone_lr (default 0.3 x learning_rate) — the rule stage 1
#   applies to the released encoder; the LM trains at learning_rate. baseline_train.yaml pins backbone_lr = learning_rate
#   (Uni-Sign's one-rate transfer recipe), so the clean floor is unchanged.
#   The gate anchors on the head's own span, never on GT. Watch gate_anchor_hit_rate and val_phrase_tiou_f1 (the head's
#   dev span quality) beside val_translation_bleu4, the best-checkpoint monitor (dlm.yaml).
python train.py --stage train-slt --language "$LANG" --slt-config configs/ar.yaml    # gated AR de-risk (§9.3)
python train.py --stage train-slt --language "$LANG" --slt-config configs/dlm.yaml   # DLM -> checkpoints/dlm/$LANG

# ── B4. EVALUATE — calibration, acceptance checks, diagnostics. Nothing here parameterizes training ──
# B4a — baseline segmenter's decode triple + the whole-video segmenter table.
#   Each arch gets its OWN tuned triple (sharing one would be the unfair-baseline objection).
#   duration_decode_moryossef feeds only the baseline's eval and RQ2 rows 4/5.
python analyze.py --stage tune-decode --segmenter-arch moryossef --language "$LANG" --split dev --write-config
python eval.py --segmenter-eval --segmenter-arch moryossef --language "$LANG" --split dev
python eval.py --segmenter-eval --segmenter-arch moryossef --language "$LANG" --split test --allow-test
python eval.py --segmenter-eval --segmenter-arch s1 --language "$LANG" --split dev
python eval.py --segmenter-eval --segmenter-arch s1 --language "$LANG" --split test --allow-test
#   READ WITH CARE: the same protocol is not neutral. Moryossef trains and evaluates on whole-video crops at a
#   1024-frame context; S1 trains on truncated, jittered buffer windows and evaluates at its own trained context
#   (checkpoint-pinned; re-chunking a trained head degrades it, measured). Expect the external segmenter to lead
#   this cell; report both models' training input and context beside the number, plus the segment-count ratio.
#   No paper claim needs S1 to win here — a higher cascade floor raises the bar rows 7/8 must clear.
#   Check val_alli_tiou_f1 (the all-I floor) before crediting any duration-decoded F1.
#   TWO PROTOCOLS, ONE DECODE: segmenter-eval also scores its spans through the RQ2 code path itself
#   (`rq2_protocol` in the JSON: caption-span gold, quarantine-dropped events, F1 of macro P/R) and prints both
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
python analyze.py --stage segmenter-infer --language "$LANG" --split test --allow-test --segmenter-decode duration \
    --output outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json
python analyze.py --stage segmenter-errors --language "$LANG" --split test --allow-test \
    --predictions outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json
#   ABLATION INPUT — DEV, only when running the measured-jitter ablation (its artifacts feed TRAINING, and the
#   loader refuses a split=test artifact):
python analyze.py --stage segmenter-infer --language "$LANG" --split dev --segmenter-decode duration \
    --output outputs/segmenter_predictions_moryossef_duration_${LANG}_dev.json
python analyze.py --stage segmenter-errors --language "$LANG" --split dev \
    --predictions outputs/segmenter_predictions_moryossef_duration_${LANG}_dev.json
#   Guard: segmenter-errors refuses spans whose stamped decode differs from the pinned triple.

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
#     configs/ablation_nogate.yaml           Ω coupling off (aux BIO loss kept) — the architectural claim; also the
#                                            head-decay attribution run (overfitting after pooled S1 vs the gate's
#                                            gradient, docs/implementation_notes.md 2026-09-03) — no new config
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

# ── B6. RQ2 — end-to-end DVC (the 8-row ladder; table below) ──
python eval.py --emit-gold-segments outputs/gold_${LANG}_test.json --language "$LANG" --split test    # rows 1/2 spans
python analyze.py --stage segmenter-infer --language "$LANG" --split test --allow-test \
    --output outputs/segmenter_predictions_moryossef_plain_${LANG}_test.json                # row 3 spans (plain argmax)
python analyze.py --stage segmenter-infer --language "$LANG" --split test --allow-test --segmenter-decode duration \
    --output outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json             # rows 4/5 spans
#   Decode parity: rows 7/8 self-segment under the duration decode, so segmenter/deployment deltas use the
#   duration-decoded cascade rows 4/5. Row 3 stays as the published-protocol reference.
python eval.py --rq 2 --segments outputs/gold_${LANG}_test.json --method baseline --language "$LANG" --split test --allow-test  # row 1
python eval.py --rq 2 --segments outputs/gold_${LANG}_test.json --method ar       --language "$LANG" --split test --allow-test  # row 2'
python eval.py --rq 2 --segments outputs/gold_${LANG}_test.json --method dlm      --language "$LANG" --split test --allow-test  # row 2
python eval.py --rq 2 --segments outputs/segmenter_predictions_moryossef_plain_${LANG}_test.json --method baseline --language "$LANG" --split test --allow-test  # row 3
python eval.py --rq 2 --segments outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json --method baseline --language "$LANG" --split test --allow-test  # row 4
python eval.py --rq 2 --segments outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json --method ar       --language "$LANG" --split test --allow-test  # row 5'
python eval.py --rq 2 --segments outputs/segmenter_predictions_moryossef_duration_${LANG}_test.json --method dlm      --language "$LANG" --split test --allow-test  # row 5
#   Row 6 — the S1-cascade floor (matched-segmenter control): the deployed head's spans, clean AR translator.
#   There is deliberately no "S1 spans + our DLM" row; every contrast it could carry is covered by (2−1), (5−4), (8−7).
python analyze.py --stage segmenter-infer --segmenter-arch s1 --segmenter-decode duration --language "$LANG" --split test --allow-test \
    --output outputs/segmenter_predictions_s1_duration_${LANG}_test.json
python eval.py --rq 2 --segments outputs/segmenter_predictions_s1_duration_${LANG}_test.json --method baseline --language "$LANG" --split test --allow-test  # row 6
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

| #   | segmenter (source of spans) | translator        | eval flag (span source)                                              |
| --- | --------------------------- | ----------------- | -------------------------------------------------------------------- |
| 1   | GT (oracle)                 | clean AR          | `--segments gold_${LANG}_test.json --method baseline`                |
| 2   | GT (oracle)                 | misaligned AR/DLM | `--segments gold_${LANG}_test.json --method dlm`                     |
| 3   | Moryossef (plain argmax)    | clean AR          | `--segments …moryossef_plain_${LANG}_test.json --method baseline`    |
| 4   | Moryossef + duration decode | clean AR          | `--segments …moryossef_duration_${LANG}_test.json --method baseline` |
| 5   | Moryossef + duration decode | misaligned AR/DLM | `--segments …moryossef_duration_${LANG}_test.json --method dlm`      |
| 6   | S1 (deployed head, offline) | clean AR          | `--segments …s1_duration_${LANG}_test.json --method baseline`        |
| 7   | final model's own head      | misaligned AR/DLM | `--offline --method dlm`                                             |
| 8   | final model's own head      | misaligned AR/DLM | `--stream --method dlm`                                              |

**Required same-span control** — re-translate row 7's saved spans with the clean translator ("clean translator + S2 spans"):

```bash
python eval.py --rq 2 --segments outputs/rq2_offline_events_dlm_${LANG}_test.json \
  --method baseline --language "$LANG" --split test --allow-test
```

**How to read the ladder.** `(a−b)` = score(row a) − score(row b); positive = the minuend wins.

- **(2−1)** / **(5−4)** — translator at oracle / realistic spans: the robustness tax on clean data, and the robust-translation gain under real segmenter error.
- **(7−5)** — segmenter (Moryossef → our own head), same translator, same mode, same decode.
- **(8−7)** — streaming vs offline deployment of the same model. Offline segments bidirectionally, streaming causally, so (8−7) folds the causal cost into the streaming gain: row 7 is a conservative baseline.
- **(8−4)** — the application headline: full deployed system vs the standard offline cascade.

Notes that prevent misreading, in brief:

- Every row is scored under ONE boundary policy (`eval.scoreable_predictions`): events majority-inside a quarantined region are ignored and events shorter than Λ_min are dropped — cascade spans included, since the deployed FSM can never commit a sub-Λ_min span. Without the shared floor rows 1–6 carry flicker spans rows 7/8 drop by policy, and (7−6) no longer isolates joint training.
- Rows 1–6 are cascades: only the span _boundaries_ are external. `run_cascade` still runs the full model (BIO head and gate included) on `--method dlm/ar` rows. Rows with `--method baseline` use the ungated clean floor — deliberately a different model.
- Row 7 chunks the head at its TRAINED cap (checkpoint meta), overlap-stitched, and translates each span in a buffer-shaped window. No whole-video pass, no random sampling at eval; rows 7–8 are deterministic.
- The misaligned AR twin runs the identical commands with `--method ar` (rows 2′,5′,7′,8′) to isolate the decoder family. The headline table stays DLM.

**Metric.** The RQ2 headline is the original densevid_eval protocol (the metric set of our previous paper: temporal F1 at tIoU {0.3, 0.5, 0.7, 0.9}; BLEU-4, METEOR, ROUGE-L, CIDEr, BLEURT within the localized segments; every text number quoted as the threshold average), implemented FAITHFULLY (metrics.densevid_text_metrics): one evaluation instance per overlapping (prediction, gold) pair — many-to-many, single reference; an unmatched prediction scores against a random lowercase GARBAGE string (length 10–20, seeded for reproducible tables); scoring is per gold video (BLEU-4 corpus-style and CIDEr-D via the official COCO scorer over the video's pairs; ROUGE-L/METEOR/BLEURT per-pair means) then averaged over videos. It is RECALL-BLIND (a missed gold never enters any pair) and precision-heavy (every extra span is a garbage pair inside the video's corpus score), so the SODA fusion (Fujita et al. 2020: one-to-one tIoU matching plus a localization-aware text score) is printed beside it in every table (`--no-densevid` drops the headline rows). Selection stages (tune-decode, tune-stream) select on segmentation F1, never on a text metric: a recall-blind criterion would reward emitting fewer spans. Neither is comparable to RQ1's corpus BLEU: SODA builds on smoothed sentence scores, which run ~1.6x corpus BLEU on identical pairs at this sentence length — the RQ2 runner prints this warning with every table. SODA per video and threshold _t_: n_p predicted spans, n_g gold sentences, M the matched set, s_ij per-pair sentence scores:

&nbsp;&nbsp;&nbsp;&nbsp;`segmentation.f1 = 2|M|/(n_p+n_g)` · `S = Σ_(i,j)∈M s_ij`, `p = S/n_p`, `r = S/n_g`, `text = 2pr/(p+r)`

- The fusion charges spurious predictions (n_p) and missed gold (n_g) in one number, which is why it accompanies the recall-blind headline. Matched-pairs-only means are rejected outright. BLEU here is smoothed sentence-BLEU; CIDEr has no per-pair form, so it appears in the densevid rows only.
- **Translation isolated from localization** lives in the GT-span rows (1–2) and the RQ1 (0,0) cell — the same condition in two metric spaces. Both anchors stay: (0,0) is the corpus-BLEU curve anchor (comparable to trimmed-clip SLT numbers); rows 1–2 are the ladder's oracle ceiling in SODA space, where (2−1) and the row deltas live.
- **segmenter-eval and RQ2** report the same decode under two protocols, both in the segmenter-eval JSON: the Moryossef-comparable block (per-video mean of F1s over UNK-masked BIO gold — cross-paper comparability only) and `rq2_protocol` (caption-span gold, quarantine-dropped events, F1 of macro P/R — computed by the RQ2 code path itself, so it matches the cascade rows' segmentation block by construction). Quote `rq2_protocol` wherever localization is compared across tables; the residual delta between the two blocks is gold construction plus aggregation, and it is now a printed pair, not an open question.
- Quote `segmentation.recall` beside any RQ2 text column, and quote tune-decode's held-out F1, never its selected cell.

## Key design decisions

One line each; the full argument lives at the pointer.

- **Sentence reconstruction merges caption cues, never splits them**; unresolvable units are quarantined (`reliable=False`, frames UNK). Changing it changes ground truth. → `docs/data_pipeline.md`
- **Cross-split de-duplication is train-side** (decontamination convention); the split CSV is mandatory. → `configs/data.yaml`
- **Wrong-language videos are dropped per video by non-Latin script share** (`max_non_latin_ratio`). → `docs/data_pipeline.md` §5b
- **BIO loss is class-weighted (`balanced`, corpus-measured)** — unweighted, the `B` class collapses in both heads. → `configs/dlm.yaml`
- **Terminator = first O-or-B, never "closing O"** — back-to-back sentences have no gap; same rule at training and inference. → spec §5.3
- **Semi-Markov duration decode is inference-time**: split reward = `split_bias + w·logit P(B)` (discriminative emission; `w=0` is the ablation). Two switches, two estimators: `duration_decode_s1` for whole-video decodes (cascade row 6, offline row 7, segmenter-eval) and `duration_decode_s1_stream` for the right-censored FSM buffer (row 8, the gate's anchor decode, delta-enc), each selected on dev by its own stage (tune-decode / tune-stream). → `docs/implementation_notes.md`
- **The membership gate is the coupling** — the only decode-time channel between the heads; query-independent bias, not a mask; the anchor is always the head's own span (no IoU veto, no GT fallback); γ stop-gradiented. → `docs/membership_gate.md`
- **S1 pretrains competence before coupling**, pooled, under the designed corruption; both stages share one window distribution. Measured (`segmenter-errors`) calibration is an ablation, not the recipe. → `configs/bio_pretrain.yaml`
- **The confidence-bound term is Mode-2a only** — the FSM never decodes the other truncation states. → spec §5
- **Best-checkpoint monitor**: clean floor = dev BLEU (not the composite `val_loss`, which rises while BLEU still climbs); gated arms = dev BLEU as well (`val_translation_bleu4`), with `val_phrase_tiou_f1` logged beside it so a decaying head is visible.
- **Text scoring level is declared per language, never sniffed from references** (`char_level_for_target`; `tests/test_scoring_level.py` enforces every call site).
- **Calibration artifacts are keyed by (segmenter, language)** so an `--segmenter-arch s1` run can never overwrite the independent measurement (`tests/test_calibration_provenance.py`).
- **Pose timing comes from `video_meta.csv`** (SignVerse resolves to exactly 24 fps). → `docs/run_real_data.md`

## Repository map

```
backbones/   Uni-Sign 69-kp 4-part ST-GCN (UniSignPoseEncoder)
poses/       normalize_keypoints_unisign (133→69) · pose_io (load_pose_window, per-video fps) · augmentation
data/        loader (YouTube-SL-25 + pooling + dedup) · windowing (BIO, first-complete-span, χ) · jitter · batch
models/      block_diffusion · dmax (OPUT + SPD/DCD) · membership_gate (Ω) · front_end · unisign ·
             streaming_slt (MisalignedSLTModel) · bio_head
train/       slt (AR/DLM trainer) · bio_pretrain (S1) · sampler (window modes) · losses (Dice + CB) · helpers
moryossef26/ faithful external segmenter (raw-kp UNet): model · dataset · trainer · infer. NOT the FSM head.
infer/       commit_gate · duration_decode (semi-Markov re-split) · decode (SPD+DCD) · stream (FSM) · stability
metrics.py   BIO monitor · tIoU segments · text metrics (declared scoring level)
utils.py     load_yaml (extends, ${language}, ${corpus}) · checkpoint_dir · pick_device
train.py     --stage {smoke-data, train-bio, train-moryossef, train-slt}
prepare_data.py  SignVerse-2M shards → language layout
eval.py      --rq {1, 2} · --segmenter-eval · --emit-gold-segments · --stability
analyze.py   --stage {dataset-summary, segmenter-infer, tune-decode, tune-stream, segmenter-errors, buffer-cap, delta-enc, system-errors}
report.py    statistics layer over RQ2 events files: shared-gold-index outcome CSVs per system,
             paired-bootstrap significance + flip tables vs a reference (outputs/report_<lang>_<split>/)
visualize.py --what {poses, losses, predict, errors}
docs/        membership_gate.md · data_pipeline.md · run_real_data.md · implementation_notes.md · literature_notes.md
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
| `inference.yaml`          | FSM constants (frozen, measured), `duration_decode_<arch>` + `duration_decode_s1_stream` triples, commit lag | —                                                |
| `data.yaml` / `eval.yaml` | corpora, splits, `target_lang`, `pretrained_slt`, subtitle pipeline / RQ grids          | —                                                |

Per-language constants, in the order B1 derives them (`--write-config` writes each):

| constant                          | source                                                                                                                               | writes                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| `duration_decode_s1` triple       | B1a `tune-decode --segmenter-arch s1` (two dev folds; quote held-out)                                                                | `inference.yaml`                     |
| `duration_decode_s1_stream` triple, `commit_lag_s` | B1a' `tune-stream --segmenter-arch s1` (the FSM itself, segmentation-only, two dev folds; snap and w inherited from B1a) | `inference.yaml` per-language rows |
| `delta_enc_frames` (δ), Λ_min=δ+1 | B1b `delta-enc` (S1 head, deployed decode; re-run once if Λ_min changed)                                                             | `inference.yaml` per-language row    |
| `buffer_cap_s`                    | B1c `buffer-cap --split train` = train-split p99 + stride + δ/fps (label-only, model-free; runs after delta-enc)                     | `inference.yaml`                     |
| `duration_decode_moryossef`       | B4a `tune-decode --segmenter-arch moryossef` (baseline eval + RQ2 rows 4/5 only)                                                     | `inference.yaml`                     |
| `bio_class_weights`               | `balanced` — resolved from measured label counts at train start, logged                                                              | automatic                            |
| pooled S1 context                 | `bio_pretrain.yaml pretrain_geometry.buffer_cap_s: auto` = max over the pool of train p99 + stride + 1 s (>= every deployed cap by construction; a number is an explicit override); the trained value is pinned in checkpoint meta and B1c refuses a cap above it | train-bio, automatic |

Rules that are not optional: run delta-enc before buffer-cap (the cap reads δ); buffer-cap runs on the train split (label-only constants are measured on train, model-dependent constants are selected on dev; the stage refuses any other split). One shared `inference.yaml` serves every language — the measured constants live in PER-LANGUAGE rows (like `duration_decode_<arch>`), resolved for the active language at load and refused loudly when the row is missing; the gate's δ/Λ_min are derived from the resolved rows, so there is no per-language config pair to maintain and no mirror to drift. A `buffer_cap_s` change never re-chunks an already-trained head — the checkpoint's `rope_eval_chunk_s` wins for that head's evaluation.

Watch `gate_anchor_hit_rate` during stage-2 training (the head's own closed span at tIoU ≥ 0.5 with the GT target, over windows that have one): at S1's level and steady = S1 delivered a usable policy and joint training keeps it. Falling = the head is decaying; find the cause before spending more GPU.
