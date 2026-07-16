"""Stage-2 composition: the shared pose/text front end plus an AR or DLM decoder, and the per-window-mode training loss 
(`MisalignedSLTModel.forward_loss`) that keeps truncated visual inputs free of partial text labels (premise P1)."""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from train.losses import bio_nll_dice_loss, confidence_bound_gate, confidence_bound_loss
from models.bio_head import RoPEBIOHead
from models.front_end import SLTFrontEnd
from models.membership_gate import build_omega, omega_cross_bias
from infer.commit_gate import open_span_start, select_target_span
from data.windowing import BIO


@dataclass
class SLTLossOutput:
    loss: torch.Tensor
    bio_loss: torch.Tensor
    translation_loss: torch.Tensor
    logs: dict[str, torch.Tensor]


class MisalignedSLTModel(nn.Module):
    """Stage-2 model: a pluggable pose/text front end (`models.front_end.SLTFrontEnd`) + a swappable AR or DLM decoder.

    The front end (Uni-Sign pose encoder + mT5 default / mBART ablation) produces, from poses:
    - `bio_tap` (per-frame, length T): read DIRECTLY by `bio_head` (RoPEBIOHead) for phrase B/I/O — the streaming
      buffer's variable length never hits the seq2seq encoder's positions.
    - the encoder memory (`enc_hidden`/`enc_mask`): cross-attended by the translation decoder.

    `decoder="dlm"` builds the front end's block-diffusion decoder (OPUT training / SPD+DCD inference); `decoder="ar"`
    uses the front end's AR seq2seq decoder. Only the decoder family differs between the two arms — front end, BIO head,
    sampler, FSM, and commit gate are identical, which is what makes the AR-vs-DLM comparison a clean test.
    """
    def __init__(
        self, front_end: SLTFrontEnd | None = None, tokenizer=None, decoder: str = "dlm",
        block_size: int = 8, bio_hidden_dim: int = 384, bio_depth: int = 4, bio_nhead: int = 8,
        bio_dropout: float = 0.1, bio_conv_stem_layers: int = 2, pretrained_path: str | None = None,
    ):
        super().__init__()
        # Front end: caller passes one (UniSignMT5FrontEnd / UniSignMBartFrontEnd — both live in models/unisign.py).
        self.front_end = front_end
        self.tokenizer = self.front_end.tokenizer
        self.decoder_type = decoder
        self.bio_head = RoPEBIOHead(
            input_dim=self.front_end.bio_tap_dim, hidden_dim=bio_hidden_dim,
            depth=bio_depth, nhead=bio_nhead, dropout=bio_dropout, num_classes=4,  # B/I/O + padding/UNK
            conv_stem_layers=bio_conv_stem_layers,  # local boundary inductive bias the UNet-less head lacks
        )
        # Load the pretrained front end BEFORE building the DLM decoder: the block-diffusion substrate copies the
        # current decoder/lm into its vocab+1 [MASK] canvas, so the pretrained weights must already be in place
        # (loads the Uni-Sign pose_encoder + mT5/mBART; pose weights come from the released ckpt).
        if pretrained_path: self.front_end.load_pretrained(pretrained_path)
        if decoder == "dlm": self.dlm_decoder = self.front_end.make_dlm_decoder(block_size)
        elif decoder != "ar": raise ValueError(f"Unsupported decoder type: {decoder}")

    def _pad_or_trim_tokens(self, tokens: torch.Tensor, target_len: int) -> torch.Tensor:
        if tokens.shape[1] > target_len: return tokens[:, :target_len]
        if tokens.shape[1] == target_len: return tokens
        pad_id = int(self.tokenizer.pad_token_id)
        pad = torch.full((tokens.shape[0], target_len - tokens.shape[1]), pad_id, dtype=tokens.dtype, device=tokens.device)
        return torch.cat([tokens, pad], dim=1)

    def encode_visual(self, batch: dict):
        # -> (bio_tap, bio_mask, enc_hidden, enc_mask, timestamps)
        bio_tap, bio_mask, timestamps, enc_hidden, enc_mask = self.front_end.encode(
            batch["poses"], batch["frame_mask"], batch.get("timestamps_s"),
        )
        return bio_tap, bio_mask, enc_hidden, enc_mask, timestamps

    @staticmethod
    def _span_iou(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
        # tIoU of 2 [start, terminator) frame spans; 0 if either is missing.
        if a is None or b is None: return 0.0
        lo = max(a[0], b[0]); hi = min(a[1], b[1])
        inter = max(0, hi - lo)
        union = (a[1] - a[0]) + (b[1] - b[0]) - inter
        return inter / union if union > 0 else 0.0

    def build_gate_omega(
        self, bio_logits: torch.Tensor, bio_labels: torch.Tensor | None, frame_mask: torch.Tensor,
        memory_len: int, commit_mask: torch.Tensor | None = None,
        delta: int = 3, eps: float = 1e-4, min_span_frames: int = 0,
        iou_veto: float = 0.5, gt_anchored: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Build the membership-gate cross-attention bias (B,1,1,M) for a batch (docs/membership_gate.md).

        The per-window selected span (s, τ) is ON-POLICY — from the BIO head's OWN argmax (`select_target_span`,
        the SAME rule the streaming FSM uses) — so training and inference see the same imperfect gate (§1.3).
        At training, `bio_labels` (the true B/I/O, already in window-relative frames) provides the GT span for
        the IoU veto (§1.5) — the honest residue of teacher forcing: where the predicted target overlaps GT at
        IoU < `iou_veto`, the span is rebuilt from the GT anchor for that window only (train-only rail, rate
        logged). `gt_anchored=True` forces the GT span (the always-GT ablation). At INFERENCE `bio_labels=None`:
        no GT, no veto — the predicted span is used verbatim, which is exactly the deployed behaviour.
        """
        B, T, _ = bio_logits.shape
        device = bio_logits.device
        lengths = frame_mask.to(device).long().sum(dim=1) if frame_mask is not None else torch.full((B,), T, device=device)
        pred_tags = bio_logits.detach().argmax(dim=-1)  # selection carries no gradient (§0)
        # UNK closes like O — the SAME remap the FSM applies before its selection (infer/stream.py). Without it the
        # gate's anchor selection sees UNK as neither O nor B: a span the FSM terminates stays open here, and the
        # committed span vs the Ω anchor (s, τ, bands, cliff) silently diverge on argmax-UNK frames.
        pred_tags = torch.where(pred_tags == BIO["UNK"], torch.full_like(pred_tags, BIO["O"]), pred_tags)

        starts, terms, has_term = [], [], []
        vetoed, n_gt = 0, 0  # veto rate is over windows that HAVE a GT target (§1.6 diagnostic)
        for b in range(B):
            n = int(lengths[b].item())
            pred = select_target_span(pred_tags[b, :n], min_span_frames)
            if bio_labels is None: span = pred  # inference: on-policy, no veto, no GT
            else:
                gt = select_target_span(bio_labels[b, :n], min_span_frames)
                if gt is not None: n_gt += 1
                if gt_anchored:                 # ablation row: "GT-anchored with ±δ jitter" (gate-doc §3 table)
                    # The ±δ jitter is part of the ablation's definition, not decoration: exact GT anchors would
                    # hand the gate boundary information the deployed on-policy head can never supply, conflating
                    # "teacher-forced m" with "oracle boundaries". Jitter keeps the anchors GT-DERIVED but
                    # δ-imprecise — the same tolerance the gate's ramp/bands are built around.
                    span = gt
                    if gt is not None:
                        vetoed += 1
                        if n > 2 and delta > 0:
                            j_s = int(torch.randint(-int(delta), int(delta) + 1, (1,)).item())
                            j_t = int(torch.randint(-int(delta), int(delta) + 1, (1,)).item())
                            s_j = min(max(int(gt[0]) + j_s, 0), n - 2)
                            t_j = min(max(int(gt[1]) + j_t, s_j + 1), n - 1)
                            span = (s_j, t_j)
                elif gt is not None and (pred is None or self._span_iou(pred, gt) < float(iou_veto)):
                    span = gt; vetoed += 1       # policy failed on a window that had a target → veto to GT
                else: 
                    span = pred
                
            if span is not None:
                starts.append(int(span[0])); terms.append(int(span[1])); has_term.append(True)
            else:
                # No TERMINATED span. If an OPEN (terminator-less) span runs to the buffer edge — Mode-2a
                # right-truncation or a buffer-cap forced commit — anchor Ω at ITS true start s (doc §2.8 forced
                # path: γ≡γ_s, no right cliff → Ω≈0 for the all-I interior). Anchoring at frame 0 instead would
                # sweep the opening B and floor the entire span the gate is meant to OPEN (attention ×0.01).
                # SAME anchoring policy as the terminated branch above (doc §1.4/§1.5): ON-POLICY first — the
                # predicted open start, which is what inference uses — with GT only as the logged veto fallback
                # when the prediction is missing or off by more than δ (the single-endpoint analog of the IoU
                # veto; within δ the ramp/band geometry is unchanged). GT-first here would train the decoder on
                # teacher-forced anchors it never sees deployed, invisibly to the §1.6 veto-rate diagnostic.
                pred_open = open_span_start(pred_tags[b, :n])
                if bio_labels is None: open_s = pred_open  # inference: on-policy, no veto
                else:
                    gt_open = open_span_start(bio_labels[b, :n])
                    if gt_open is not None: n_gt += 1
                    if pred_open is not None and (gt_open is None or abs(pred_open - gt_open) <= int(delta)):
                        open_s = pred_open
                    elif gt_open is not None: open_s = gt_open; vetoed += 1
                    else: open_s = pred_open
                if open_s is not None:
                    starts.append(int(open_s)); terms.append(-1); has_term.append(False)
                else:
                    # Genuinely no span (all-gap / buffer-start-I leftover). This Ω is INERT — never decoded
                    # against: the FSM skips these buffers (all-gap → no target; left-truncated fragment →
                    # 'skip', and the 'translate_partial' ablation decodes UNgated), and in training no-span
                    # rows carry no translation loss. The anchor value is arbitrary; n−1 is just a valid index
                    # (NOT neutral — frames < n−1−δ sit behind the left wall — which is fine only because
                    # nothing reads it).
                    starts.append(max(0, n - 1)); terms.append(-1); has_term.append(False)

        out = build_omega(
            bio_logits,
            starts=torch.tensor(starts, device=device), terminators=torch.tensor(terms, device=device), commit_mask=commit_mask, 
            lengths=lengths, delta=delta, eps=eps, has_terminator=torch.tensor(has_term, device=device),
        )
        omega_bias = omega_cross_bias(out.omega, memory_len=int(memory_len), dtype=bio_logits.dtype)
        stats = {"veto_rate": vetoed / max(1, n_gt), "gamma_s_mean": float(out.gamma_s.mean())}
        return omega_bias, stats


    @torch.no_grad()
    def generate_from_bio_tap(
        self, bio_tap: torch.Tensor, frame_mask: torch.Tensor, max_text_tokens: int = 128, diffusion_steps: int = 64, 
        tau_dec: float = 0.75, spd_top_k: int = 1, spd_renormalize: bool = True, spd_revision: bool = True, temperature: float = 0.0,
        dcd_window_length: int | None = None, dcd_max_window_length: int | None = None, dcd_window_type: str = "sliding",
        dcd_decode_algo: str = "threshold", dcd_decode_param: int | float | None = None, dcd_sample_top_k: int | None = None,
        dcd_top_p: float | None = None, dcd_cache_type: str = "none",  dcd_refresh_count: int = 16,
        decoder_start_token_id: int | None = None, num_beams: int = 1, omega_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc_hidden, enc_mask = self.front_end.encode_memory(bio_tap, frame_mask)
        if self.decoder_type == "dlm":
            result = self.dlm_decoder.generate_spd_dcd(
                enc_hidden=enc_hidden, enc_mask=enc_mask, max_length=max_text_tokens, diffusion_steps=diffusion_steps,
                tau_dec=tau_dec, top_k=spd_top_k, spd_renormalize=spd_renormalize, spd_revision=spd_revision, temperature=temperature,
                window_length=dcd_window_length, max_window_length=dcd_max_window_length, window_type=dcd_window_type,
                decode_algo=dcd_decode_algo, decode_param=dcd_decode_param, sample_top_k=dcd_sample_top_k,
                top_p=dcd_top_p, cache_type=dcd_cache_type, refresh_count=dcd_refresh_count, omega_bias=omega_bias,
            )
            # Report PRODUCED-token confidence only, matching the AR arm's per-produced-token vector: canvas slot 0 
            # is the synthetic BOS (confidence 1.0) and every slot after the first EOS is pad bookkeeping (also 1.0) 
            # — averaged over a max_text_tokens canvas they pin the mean the commit gate reads near 1 for any short 
            # sentence (a ~10-token sentence on a 128 canvas → mean ≥ 0.92 regardless of decode quality, i.e. 
            # `translation_confident` would always fire).
            seq, conf = result.sequences[:, 1:], result.confidence[:, 1:]
            if seq.shape[0] == 1:  # every live caller decodes one window/buffer at a time
                hits = (seq[0] == int(self.dlm_decoder.eos_index)).nonzero(as_tuple=False)
                if hits.numel(): seq, conf = seq[:, : int(hits[0]) + 1], conf[:, : int(hits[0]) + 1]
            return seq, conf

        # AR arm: the front end owns generation (mBART lang-code start / mT5 prompt-conditioned) and returns the
        # REAL per-token confidence (softmax prob of each produced token), aligned with `generated`. `num_beams>1`
        # is the clean-baseline beam search; the SLT AR arm stays greedy (num_beams=1) for the §9.3 contrast.
        # Under `omega_bias`, cross-attention is membership-gated per step (same Ω the DLM decode uses).
        return self.front_end.ar_generate(
            enc_hidden, enc_mask, max_new_tokens=max_text_tokens, num_beams=num_beams,
            decoder_start_id=decoder_start_token_id, omega_bias=omega_bias,
        )


    def _ar_confidence_bound_logits(
        self, bio_tap: torch.Tensor, frame_mask: torch.Tensor, max_len: int, omega_bias=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return gradient-carrying AR logits on the truncated model path.

        `generate_from_bio_tap` is intentionally called under no-grad to choose the AR prefix. The subsequent forward pass
        replays that prefix and keeps gradients in the logits used by the confidence-bound CE term. When the gate is on,
        BOTH the prefix generation and the grad-bearing replay run under the same Ω conditioning (the replay's gradient
        into the BIO logits through Ω is part of the coupling on Mode-2a as well). (AR is not a diffusion decoder, so
        diffusion_steps/tau_dec don't apply here — the DLM arm's CB path uses them, this one doesn't.)
        """
        with torch.no_grad():
            trunc_tokens, _ = self.generate_from_bio_tap(
                bio_tap, frame_mask, max_text_tokens=max(1, max_len - 1), omega_bias=omega_bias
            )
            trunc_tokens = self._pad_or_trim_tokens(trunc_tokens, max_len)

        enc_hidden, enc_mask = self.front_end.encode_memory(bio_tap, frame_mask)
        with self.front_end.ar_omega_context(omega_bias):
            out = self.front_end.lm_model(
                encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden), attention_mask=enc_mask,
                decoder_input_ids=trunc_tokens[:, :-1].contiguous(), use_cache=False, return_dict=True,
            )
        return out.logits, trunc_tokens


    @torch.no_grad()
    def generate_from_poses(
        self, poses: torch.Tensor, frame_mask: torch.Tensor, timestamps_s: torch.Tensor | None = None, *,
        gate_enabled: bool = False, gate_delta: int = 3, gate_eps: float = 1e-4, gate_min_span_frames: int = 0,
        commit_mask: torch.Tensor | None = None, **decode_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Poses → (bio_logits, tokens, confidence). This method owns the BIO tap + the membership gate; every
        decode knob (max_text_tokens / diffusion_steps / tau_dec / spd_* / dcd_* / num_beams / decoder_start_token_id)
        passes straight through to `generate_from_bio_tap` — declared once, there, not re-listed here."""
        bio_tap, mask, timestamps = self.front_end.extract_bio_tap(poses, frame_mask, timestamps_s)
        bio_logits = self.bio_head(bio_tap, timestamps_s=timestamps, frame_mask=mask).logits
        # Membership gate at inference: on-policy span from the head's own argmax (bio_labels=None → no veto),
        # χ from the FSM commit log (single-window RQ1: none). Same Ω the decoder saw in training (§1.3/§2.8).
        # Both arms: the DLM injects Ω in its decode; the AR arm via HF cross-attn hooks (front_end.ar_generate).
        omega_bias = None
        if gate_enabled: omega_bias, _ = self.build_gate_omega(
            bio_logits, None, mask, memory_len=self.front_end.prompt_length() + int(bio_tap.shape[1]),
            commit_mask=commit_mask, delta=gate_delta, eps=gate_eps, min_span_frames=gate_min_span_frames,
        )
        tokens, confidence = self.generate_from_bio_tap(bio_tap, mask, omega_bias=omega_bias, **decode_kwargs)
        return bio_logits, tokens, confidence


    def forward_loss(
        self, batch: dict, lambda_trans: float = 1.0, 
        dice_weight: float = 1.5, bio_class_weights: torch.Tensor | None = None,
        oput_t_low: float = 0.3, oput_t_high: float = 0.8, oput_sample_rollout: bool = False,
        oput_rollout_eval_mode: bool = True, oput_eos_supervision: int = 32,
        confidence_bound_enabled: bool = True, confidence_bound_active: bool = True, confidence_bound_tau: float = 0.75,
        cb_lambda: float = 0.3, verified_full_evidence_gate: bool = True, cb_decode_steps: int = 64,
        cb_dcd_window_length: int | None = None, cb_dcd_max_window_length: int | None = None, cb_dcd_window_type: str = "sliding",
        cb_dcd_decode_algo: str = "threshold", cb_dcd_decode_param: int | float | None = None, cb_dcd_sample_top_k: int | None = None,
        cb_dcd_top_p: float | None = None, cb_dcd_cache_type: str = "none", cb_dcd_refresh_count: int = 16,
        cb_spd_top_k: int = 1, cb_spd_renormalize: bool = True, cb_spd_revision: bool = True, cb_temperature: float = 0.0,
        gate_enabled: bool = False, gate_delta: int = 3, gate_eps: float = 1e-4, gate_min_span_frames: int = 0,
        gate_iou_veto: float = 0.5, gate_gt_anchored: bool = False,
    ) -> SLTLossOutput:
        """Stage-2 training loss for one mixed-mode batch.

        Computes ``L = L_BIO + lambda_trans * L_translation`` where the translation term is routed *per window mode* (the batch carries 
        `mode_names` from the sampler), enforcing premise P1 — a truncated visual input never receives a partial text label:

        - **BIO** (all modes): Dice(1.5)+CE on the phrase head over every in-window frame; padding/UNK ignored.
        - **Mode 1 / Mode 3** (complete-anchor / first-complete-span): translation via **OPUT** under fixed full conditioning 
          (`dlm_decoder.oput_forward`, or plain CE for the AR arm). A hard check raises if any other mode reaches the OPUT path.
        - **Mode 2a** (right-truncated): the **confidence-bound** term only — gated CE toward the model's own no-grad full-evidence decode, 
          at slots where that decode is reference-verified and the truncated decode is confidently disagreeing. Held off during 
          ``confidence_bound_active=False`` (OPUT warmup) and weighted by ``cb_lambda``. See `dlm_decoder.remasked_logits` for why the CE 
          gradient uses 1 cheap re-masked forward rather than back-prop through the decode.
        - **Mode 2b / 2c / Mode 4** (left/both-truncated, all-gap): no translation loss (the model must stay silent / not hallucinate text).

        Per-mode losses logged separately. Returns `SLTLossOutput` with the total, the BIO and translation components, and a `logs` dict.
        """
        bio_tap, bio_mask, enc_hidden, enc_mask, timestamps = self.encode_visual(batch)
        bio_out = self.bio_head(bio_tap, timestamps_s=timestamps, frame_mask=bio_mask)
        bio_loss = bio_nll_dice_loss(bio_out.logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=bio_class_weights)
        translation_loss = bio_tap.sum() * 0.0
        logs: dict[str, torch.Tensor] = {"bio_loss": bio_loss.detach()}
        target_tokens = batch.get("target_tokens")
        supervised = batch.get("translation_supervised")

        # Membership gate (docs/membership_gate.md): build the (B,1,1,M) cross-attention bias Ω from the BIO
        # posteriors + the on-policy selected span, so the decoder is CONDITIONED on the segmentation belief
        # (the coupling — gradient from the translation loss reaches the BIO logits through Ω). BOTH arms: the
        # DLM injects Ω in its manual decode; the AR arm injects the same Ω via HF cross-attn hooks
        # (front_end.ar_omega_context) — so §9.3 is gated-vs-gated, isolating the decoder family. None → pre-gate.
        omega_bias = None
        if gate_enabled:
            omega_bias, gate_stats = self.build_gate_omega(
                bio_out.logits, batch["bio_labels"], batch.get("frame_mask"), memory_len=int(enc_hidden.shape[1]), 
                commit_mask=batch.get("commit_mask"), delta=gate_delta, eps=gate_eps, min_span_frames=gate_min_span_frames,
                iou_veto=gate_iou_veto, gt_anchored=gate_gt_anchored,
            )
            logs["gate_veto_rate"] = bio_tap.new_tensor(gate_stats["veto_rate"])
            logs["gate_gamma_s_mean"] = bio_tap.new_tensor(gate_stats["gamma_s_mean"])

        if target_tokens is not None and supervised is not None and supervised.any():
            idx = supervised.to(device=bio_tap.device).nonzero(as_tuple=False).flatten()
            mode_names = batch.get("mode_names")
            mode_to_indices: dict[str, torch.Tensor] = {}
            if isinstance(mode_names, list):
                idx_list = idx.detach().cpu().tolist()
                supervised_modes = [mode_names[int(i)] for i in idx_list]
                invalid = sorted({mode for mode in supervised_modes if mode not in {"mode1", "mode3"}})
                if invalid: raise ValueError(
                    f"Translation supervision is allowed only for complete-conditioning Mode 1/Mode 3 windows; got {invalid}"
                )
                for mode in ("mode1", "mode3"):
                    selected = [int(i) for i in idx_list if mode_names[int(i)] == mode]
                    if selected: mode_to_indices[mode] = torch.tensor(selected, dtype=torch.long, device=bio_tap.device)

            if self.decoder_type == "ar":
                labels = target_tokens["labels"].to(bio_tap.device)
                if mode_to_indices:
                    weighted_losses: list[tuple[torch.Tensor, torch.Tensor]] = []

                    for mode, mode_idx in mode_to_indices.items():
                        loss = self.front_end.ar_loss(
                            enc_hidden[mode_idx], enc_mask[mode_idx], labels[mode_idx],
                            omega_bias=None if omega_bias is None else omega_bias[mode_idx]
                        )
                        weight = ((labels[mode_idx] != -100) & (labels[mode_idx] != self.tokenizer.pad_token_id)).sum()
                        weight = weight.to(dtype=loss.dtype).clamp(min=1)
                        weighted_losses.append((loss, weight))

                    total_weight = sum(weight for _, weight in weighted_losses).clamp(min=1)
                    translation_loss = sum(loss * weight for loss, weight in weighted_losses) / total_weight
                else:
                    translation_loss = self.front_end.ar_loss(
                        enc_hidden[idx], enc_mask[idx], labels[idx],
                        omega_bias=None if omega_bias is None else omega_bias[idx]
                    )
            else:
                labels = target_tokens["labels"].to(bio_tap.device)
                if mode_to_indices:
                    weighted_losses: list[tuple[torch.Tensor, torch.Tensor]] = []
                    for mode, mode_idx in mode_to_indices.items():
                        dlm_out = self.dlm_decoder.oput_forward(
                            enc_hidden=enc_hidden[mode_idx], enc_mask=enc_mask[mode_idx],
                            labels=labels[mode_idx], t_low=oput_t_low, t_high=oput_t_high,
                            loss_over_all_positions=True, sample_rollout=oput_sample_rollout,
                            rollout_eval_mode=oput_rollout_eval_mode, eos_supervision=oput_eos_supervision,
                            rollout_encode_fn=(
                                self.front_end.eval_encode_memory_fn(bio_tap[mode_idx], bio_mask[mode_idx])
                                if oput_rollout_eval_mode else None
                            ),
                            omega_bias=None if omega_bias is None else omega_bias[mode_idx],
                        )
                        weight = ((labels[mode_idx] != -100) & (labels[mode_idx] != self.tokenizer.pad_token_id)).sum()
                        weight = weight.to(dtype=dlm_out["translation_loss"].dtype).clamp(min=1)
                        weighted_losses.append((dlm_out["translation_loss"], weight))


                    total_weight = sum(weight for _, weight in weighted_losses).clamp(min=1)
                    translation_loss = sum(loss * weight for loss, weight in weighted_losses) / total_weight
                else:
                    dlm_out = self.dlm_decoder.oput_forward(
                        enc_hidden=enc_hidden[idx], enc_mask=enc_mask[idx],
                        labels=labels[idx], t_low=oput_t_low, t_high=oput_t_high,
                        loss_over_all_positions=True, sample_rollout=oput_sample_rollout,
                        rollout_eval_mode=oput_rollout_eval_mode, eos_supervision=oput_eos_supervision,
                        rollout_encode_fn=(
                            self.front_end.eval_encode_memory_fn(bio_tap[idx], bio_mask[idx])
                            if oput_rollout_eval_mode else None
                        ),
                        omega_bias=None if omega_bias is None else omega_bias[idx],
                    )
                    translation_loss = dlm_out["translation_loss"]

        if (confidence_bound_enabled and confidence_bound_active and batch.get("full_evidence") is not None
            and batch.get("full_evidence_indices") is not None and batch["full_evidence_indices"].numel() > 0
            and batch.get("reference_tokens") is not None):
            cb_indices = batch["full_evidence_indices"].to(bio_tap.device)
            full_batch = batch["full_evidence"]
            full_bio_tap, full_mask, full_timestamps = self.front_end.extract_bio_tap(
                full_batch["poses"], full_batch["frame_mask"], full_batch.get("timestamps_s"),
            )
            ref_ids = batch["reference_tokens"]["input_ids"].to(bio_tap.device)[cb_indices]
            ref_mask = batch["reference_tokens"]["attention_mask"].to(bio_tap.device)[cb_indices].bool()
            max_len = ref_ids.shape[1]

            # Membership gate for the Mode-2a CB decodes (both arms, so §9.3 stays symmetric): the truncated
            # decode is gated by Ω from its own posteriors (on-policy); the full-evidence self-target by Ω from
            # the full view's posteriors. None when the gate is off. Both trunc decodes/forwards share cb_omega_trunc.
            cb_omega_trunc = cb_omega_full = None
            if gate_enabled:
                prompt_len = self.front_end.prompt_length()
                chi = batch.get("commit_mask")
                cb_omega_trunc, _ = self.build_gate_omega(
                    bio_out.logits[cb_indices],
                    batch.get("bio_labels")[cb_indices] if batch.get("bio_labels") is not None else None,
                    bio_mask[cb_indices], memory_len=prompt_len + int(bio_tap.shape[1]),
                    commit_mask=chi[cb_indices] if chi is not None else None,
                    delta=gate_delta, eps=gate_eps, min_span_frames=gate_min_span_frames, iou_veto=gate_iou_veto,
                )
                # Real timestamps + mask (previously dropped: RoPE silently fell back to the 50fps-index
                # assumption on the full-evidence view — a time-scale mismatch vs the trunc view's real seconds).
                full_cb_bio_logits = self.bio_head(full_bio_tap, timestamps_s=full_timestamps, frame_mask=full_mask).logits
                cb_omega_full, _ = self.build_gate_omega(
                    full_cb_bio_logits, None, full_mask, memory_len=prompt_len + int(full_bio_tap.shape[1]),
                    # χ on BOTH views (mirroring cb_omega_trunc): the full-evidence rows carry their own commit_mask,
                    # so a committed predecessor tail straddling the left edge is floored here too — else the two
                    # views would differ by a left-edge conditioning change, not only the right-truncation.
                    commit_mask=full_batch.get("commit_mask"),
                    delta=gate_delta, eps=gate_eps, min_span_frames=gate_min_span_frames,
                )

            if self.decoder_type == "dlm":
                # Encode the truncated path ONCE, grad-bearing (the no-grad decode below won't track it; remasked_logits
                # later does). Encoding belongs to the front end, not the decoder (unified enc_hidden interface).
                trunc_enc_hidden, trunc_enc_mask = self.front_end.encode_memory(bio_tap[cb_indices], bio_mask[cb_indices])
                with torch.no_grad():
                    full_enc_hidden, full_enc_mask = self.front_end.encode_memory(full_bio_tap, full_mask)
                    full_decode = self.dlm_decoder.generate_spd_dcd(
                        enc_hidden=full_enc_hidden, enc_mask=full_enc_mask, max_length=max_len,
                        diffusion_steps=cb_decode_steps, tau_dec=confidence_bound_tau, top_k=cb_spd_top_k,
                        spd_renormalize=cb_spd_renormalize, spd_revision=cb_spd_revision, temperature=cb_temperature,
                        window_length=cb_dcd_window_length, max_window_length=cb_dcd_max_window_length, window_type=cb_dcd_window_type,
                        decode_algo=cb_dcd_decode_algo, decode_param=cb_dcd_decode_param, sample_top_k=cb_dcd_sample_top_k,
                        top_p=cb_dcd_top_p, cache_type=cb_dcd_cache_type, refresh_count=cb_dcd_refresh_count, omega_bias=cb_omega_full,
                    )
                    full_tokens = full_decode.sequences

                    # The gate reads the confidence/argmax the *real* decode produces (what DCD sees at inference), so this stays under no_grad.
                    trunc_decode = self.dlm_decoder.decode_spd_dcd(
                        enc_hidden=trunc_enc_hidden, enc_mask=trunc_enc_mask, max_length=max_len,
                        diffusion_steps=cb_decode_steps, tau_dec=confidence_bound_tau, top_k=cb_spd_top_k,
                        spd_renormalize=cb_spd_renormalize, spd_revision=cb_spd_revision, temperature=cb_temperature,
                        window_length=cb_dcd_window_length, max_window_length=cb_dcd_max_window_length, window_type=cb_dcd_window_type,
                        decode_algo=cb_dcd_decode_algo, decode_param=cb_dcd_decode_param, sample_top_k=cb_dcd_sample_top_k,
                        top_p=cb_dcd_top_p, cache_type=cb_dcd_cache_type, refresh_count=cb_dcd_refresh_count, omega_bias=cb_omega_trunc,
                    )
                trunc_tokens = trunc_decode.sequences
                trunc_confidence = trunc_decode.confidence

                # Align the reference to the decode layout. The decode emits [BOS, tok1, ..., eos, ...] while the mBART tokenizer emits
                # [tok1, ..., eos, lang]: decode slot j holds reference slot j-1. Without this shift the verified gate (f_i == r_i)
                # compares misaligned slots and the CB term silently never fires.
                bos_col = torch.full((ref_ids.shape[0], 1), int(self.dlm_decoder.bos_index), dtype=ref_ids.dtype, device=ref_ids.device)
                cb_ref_ids = torch.cat([bos_col, ref_ids[:, :-1]], dim=1)
                cb_ref_mask = torch.cat([torch.ones_like(ref_mask[:, :1]), ref_mask[:, :-1]], dim=1)

                # Gate first (from the real no-grad decode the inference gate reads), then 1 grad-bearing forward with gated slots re-masked 
                # in an otherwise committed sequence — the deferral counterfactual DCD would face (see dlm_decoder.remasked_logits).
                cb_active_mask = confidence_bound_gate(
                    full_tokens=full_tokens, trunc_tokens=trunc_tokens, trunc_confidence=trunc_confidence,
                    reference_tokens=cb_ref_ids, valid_mask=cb_ref_mask, tau_cb=confidence_bound_tau, 
                    verified_full_evidence_gate=verified_full_evidence_gate, pad_token_id=self.tokenizer.pad_token_id,
                )
                if cb_active_mask.any(): trunc_logits = self.dlm_decoder.remasked_logits(
                    enc_hidden=trunc_enc_hidden, enc_mask=trunc_enc_mask, decoded_tokens=trunc_tokens, 
                    remask_positions=cb_active_mask, omega_bias=cb_omega_trunc,
                )
                else: trunc_logits = None
            else:
                # AR Mode-2a: reuse the shared cb_omega_full / cb_omega_trunc built above the arm split, so the
                # DLM and AR CB paths are gated symmetrically (§9.3).
                with torch.no_grad():
                    full_tokens, _ = self.generate_from_bio_tap(
                        full_bio_tap, full_mask, max_text_tokens=max(1, max_len - 1), omega_bias=cb_omega_full
                    )
                    full_tokens = self._pad_or_trim_tokens(full_tokens, max_len)

                trunc_logits, _ = self._ar_confidence_bound_logits(
                    bio_tap[cb_indices], bio_mask[cb_indices], max_len=max_len, omega_bias=cb_omega_trunc,
                )
                # AR decode layout is [lang, tok1, ..., eos] (decoder start fixed to language code, matching mBART's training-time shift), 
                # so after dropping the start token, slot j of full_tokens[:, 1:] lines up with reference slot j ([tok1, ..., eos, lang]); 
                # Loss's min-length slicing trims the dangling lang code. DON'T slice reference too — that shifts verified gate off by 1.
                full_tokens = full_tokens[:, 1:]
                trunc_tokens, trunc_confidence = None, None
                cb_ref_ids, cb_ref_mask, cb_active_mask = ref_ids, ref_mask, None

            if trunc_logits is not None:
                cb = confidence_bound_loss(
                    trunc_logits=trunc_logits, full_tokens=full_tokens, reference_tokens=cb_ref_ids,
                    valid_mask=cb_ref_mask, trunc_tokens=trunc_tokens, trunc_confidence=trunc_confidence,
                    tau_cb=confidence_bound_tau, verified_full_evidence_gate=verified_full_evidence_gate,
                    enabled=True, pad_token_id=self.tokenizer.pad_token_id, active_mask=cb_active_mask,
                )
                translation_loss = translation_loss + float(cb_lambda) * cb.loss

        total = bio_loss + float(lambda_trans) * translation_loss
        logs["translation_loss"] = translation_loss.detach()
        logs["loss"] = total.detach()
        return SLTLossOutput(total, bio_loss, translation_loss, logs)
