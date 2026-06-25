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


@dataclass
class Stage2LossOutput:
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


    @torch.no_grad()
    def generate_from_bio_tap(
        self, bio_tap: torch.Tensor, frame_mask: torch.Tensor, max_text_tokens: int = 128, diffusion_steps: int = 64, 
        tau_dec: float = 0.75, spd_top_k: int = 1, spd_renormalize: bool = True, spd_revision: bool = True, temperature: float = 0.0,
        dcd_window_length: int | None = None, dcd_max_window_length: int | None = None, dcd_window_type: str = "sliding",
        dcd_decode_algo: str = "threshold", dcd_decode_param: int | float | None = None, dcd_sample_top_k: int | None = None,
        dcd_top_p: float | None = None, dcd_cache_type: str = "none",  dcd_refresh_count: int = 16, 
        decoder_start_token_id: int | None = None, num_beams: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc_hidden, enc_mask = self.front_end.encode_memory(bio_tap, frame_mask)
        if self.decoder_type == "dlm":
            result = self.dlm_decoder.generate_spd_dcd(
                enc_hidden=enc_hidden, enc_mask=enc_mask, max_length=max_text_tokens, diffusion_steps=diffusion_steps,
                tau_dec=tau_dec, top_k=spd_top_k, spd_renormalize=spd_renormalize, spd_revision=spd_revision, temperature=temperature,
                window_length=dcd_window_length, max_window_length=dcd_max_window_length, window_type=dcd_window_type,
                decode_algo=dcd_decode_algo, decode_param=dcd_decode_param, sample_top_k=dcd_sample_top_k,
                top_p=dcd_top_p, cache_type=dcd_cache_type, refresh_count=dcd_refresh_count,
            )
            return result.sequences, result.confidence

        # AR arm: the front end owns generation (mBART lang-code start / mT5 prompt-conditioned) and returns the
        # REAL per-token confidence (softmax prob of each produced token), aligned with `generated`. `num_beams>1`
        # is the clean-baseline beam search; the stage-2 AR arm stays greedy (num_beams=1) for the §9.3 contrast.
        return self.front_end.ar_generate(
            enc_hidden, enc_mask, max_new_tokens=max_text_tokens, num_beams=num_beams,
            decoder_start_id=decoder_start_token_id,
        )


    def _ar_confidence_bound_logits(
        self, bio_tap: torch.Tensor, frame_mask: torch.Tensor, 
        max_len: int, diffusion_steps: int, tau_dec: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return gradient-carrying AR logits on the truncated model path.

        `generate_from_bio_tap` is intentionally called under no-grad to choose the AR prefix. The subsequent forward pass 
        replays that prefix and keeps gradients in the logits used by the confidence-bound CE term.
        """
        del diffusion_steps, tau_dec
        with torch.no_grad():
            trunc_tokens, _ = self.generate_from_bio_tap(bio_tap, frame_mask, max_text_tokens=max(1, max_len - 1))
            trunc_tokens = self._pad_or_trim_tokens(trunc_tokens, max_len)

        enc_hidden, enc_mask = self.front_end.encode_memory(bio_tap, frame_mask)
        out = self.front_end.lm_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden), attention_mask=enc_mask,
            decoder_input_ids=trunc_tokens[:, :-1].contiguous(), use_cache=False, return_dict=True,
        )
        return out.logits, trunc_tokens


    @torch.no_grad()
    def generate_from_poses(
        self, poses: torch.Tensor, frame_mask: torch.Tensor, 
        timestamps_s: torch.Tensor | None = None, max_text_tokens: int = 128, diffusion_steps: int = 64, 
        tau_dec: float = 0.75, spd_top_k: int = 1, spd_renormalize: bool = True, spd_revision: bool = True, temperature: float = 0.0,
        dcd_window_length: int | None = None, dcd_max_window_length: int | None = None, dcd_window_type: str = "sliding",
        dcd_decode_algo: str = "threshold", dcd_decode_param: int | float | None = None, dcd_sample_top_k: int | None = None,
        dcd_top_p: float | None = None, dcd_cache_type: str = "none", dcd_refresh_count: int = 16, 
        decoder_start_token_id: int | None = None, num_beams: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bio_tap, mask, ts = self.front_end.extract_bio_tap(poses, frame_mask, timestamps_s)
        bio_logits = self.bio_head(bio_tap, timestamps_s=ts).logits
        tokens, confidence = self.generate_from_bio_tap(
            bio_tap, mask, max_text_tokens=max_text_tokens, diffusion_steps=diffusion_steps, tau_dec=tau_dec,
            spd_top_k=spd_top_k, spd_renormalize=spd_renormalize, spd_revision=spd_revision, temperature=temperature,
            dcd_window_length=dcd_window_length, dcd_max_window_length=dcd_max_window_length, dcd_window_type=dcd_window_type,
            dcd_decode_algo=dcd_decode_algo, dcd_decode_param=dcd_decode_param, dcd_sample_top_k=dcd_sample_top_k, 
            dcd_top_p=dcd_top_p, dcd_cache_type=dcd_cache_type, dcd_refresh_count=dcd_refresh_count, 
            decoder_start_token_id=decoder_start_token_id, num_beams=num_beams,
        )
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
    ) -> Stage2LossOutput:
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

        Per-mode losses logged separately. Returns `Stage2LossOutput` with the total, the BIO and translation components, and a `logs` dict.
        """
        bio_tap, bio_mask, enc_hidden, enc_mask, timestamps = self.encode_visual(batch)
        bio_out = self.bio_head(bio_tap, timestamps_s=timestamps)
        bio_loss = bio_nll_dice_loss(bio_out.logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=bio_class_weights)
        translation_loss = bio_tap.sum() * 0.0
        logs: dict[str, torch.Tensor] = {"bio_loss": bio_loss.detach()}
        target_tokens = batch.get("target_tokens")
        supervised = batch.get("translation_supervised")

        if target_tokens is not None and supervised is not None and supervised.any():
            idx = supervised.to(device=bio_tap.device).nonzero(as_tuple=False).flatten()
            mode_names = batch.get("mode_names")
            mode_to_indices: dict[str, torch.Tensor] = {}
            if isinstance(mode_names, list):
                idx_list = idx.detach().cpu().tolist()
                supervised_modes = [mode_names[int(i)] for i in idx_list]
                invalid = sorted({mode for mode in supervised_modes if mode not in {"mode1", "mode3"}})
                if invalid: raise ValueError(
                    "Translation supervision is allowed only for complete-conditioning Mode 1/Mode 3 windows; got {invalid}"
                )
                for mode in ("mode1", "mode3"):
                    selected = [int(i) for i in idx_list if mode_names[int(i)] == mode]
                    # logs[f"translation_{mode}_count"] = bio_tap.new_tensor(float(len(selected)))
                    if selected: mode_to_indices[mode] = torch.tensor(selected, dtype=torch.long, device=bio_tap.device)

            if self.decoder_type == "ar":
                labels = target_tokens["labels"].to(bio_tap.device)
                if mode_to_indices:
                    weighted_losses: list[tuple[torch.Tensor, torch.Tensor]] = []

                    for mode, mode_idx in mode_to_indices.items():
                        loss = self.front_end.ar_loss(enc_hidden[mode_idx], enc_mask[mode_idx], labels[mode_idx])
                        weight = ((labels[mode_idx] != -100) & (labels[mode_idx] != self.tokenizer.pad_token_id)).sum()
                        weight = weight.to(dtype=loss.dtype).clamp(min=1)
                        weighted_losses.append((loss, weight))
                        # logs[f"ar_{mode}_ce_loss"] = loss.detach()

                    total_weight = sum(weight for _, weight in weighted_losses).clamp(min=1)
                    translation_loss = sum(loss * weight for loss, weight in weighted_losses) / total_weight
                    # logs["ar_ce_loss"] = translation_loss.detach()
                else:
                    translation_loss = self.front_end.ar_loss(enc_hidden[idx], enc_mask[idx], labels[idx])
                    # logs["ar_ce_loss"] = translation_loss.detach()
            else:
                labels = target_tokens["labels"].to(bio_tap.device)
                if mode_to_indices:
                    weighted_losses: list[tuple[torch.Tensor, torch.Tensor]] = []
                    masked_weights: list[tuple[torch.Tensor, torch.Tensor]] = []
                    for mode, mode_idx in mode_to_indices.items():
                        dlm_out = self.dlm_decoder.oput_forward(
                            enc_hidden=enc_hidden[mode_idx], enc_mask=enc_mask[mode_idx],
                            labels=labels[mode_idx], t_low=oput_t_low, t_high=oput_t_high,
                            loss_over_all_positions=True, sample_rollout=oput_sample_rollout,
                            rollout_eval_mode=oput_rollout_eval_mode, eos_supervision=oput_eos_supervision,
                            rollout_encode_fn=(self.front_end.eval_encode_memory_fn(bio_tap[mode_idx], bio_mask[mode_idx])
                                               if oput_rollout_eval_mode else None),
                        )
                        weight = ((labels[mode_idx] != -100) & (labels[mode_idx] != self.tokenizer.pad_token_id)).sum()
                        weight = weight.to(dtype=dlm_out["translation_loss"].dtype).clamp(min=1)
                        weighted_losses.append((dlm_out["translation_loss"], weight))
                        masked_weights.append((dlm_out["oput_masked_fraction"], weight))

                        # logs[f"oput_{mode}_loss"] = dlm_out["translation_loss"].detach()
                        # logs[f"oput_{mode}_mask_loss"] = dlm_out["oput_mask_loss"]
                        # logs[f"oput_{mode}_pred_loss"] = dlm_out["oput_pred_loss"]
                        # logs[f"oput_{mode}_masked_fraction"] = dlm_out["oput_masked_fraction"]

                    total_weight = sum(weight for _, weight in weighted_losses).clamp(min=1)
                    translation_loss = sum(loss * weight for loss, weight in weighted_losses) / total_weight
                    # logs["oput_loss"] = translation_loss.detach()
                    # logs["oput_masked_fraction"] = (sum(value * weight for value, weight in masked_weights) / total_weight).detach()
                else:
                    dlm_out = self.dlm_decoder.oput_forward(
                        enc_hidden=enc_hidden[idx], enc_mask=enc_mask[idx],
                        labels=labels[idx], t_low=oput_t_low, t_high=oput_t_high,
                        loss_over_all_positions=True, sample_rollout=oput_sample_rollout,
                        rollout_eval_mode=oput_rollout_eval_mode, eos_supervision=oput_eos_supervision,
                        rollout_encode_fn=(self.front_end.eval_encode_memory_fn(bio_tap[idx], bio_mask[idx]) 
                                           if oput_rollout_eval_mode else None),
                    )
                    translation_loss = dlm_out["translation_loss"]
                    # logs["oput_loss"] = translation_loss.detach()
                    # logs["oput_masked_fraction"] = dlm_out["oput_masked_fraction"]

        if (confidence_bound_enabled and confidence_bound_active and batch.get("full_evidence") is not None
            and batch.get("full_evidence_indices") is not None and batch["full_evidence_indices"].numel() > 0
            and batch.get("reference_tokens") is not None):
            cb_indices = batch["full_evidence_indices"].to(bio_tap.device)
            full_batch = batch["full_evidence"]
            full_bio_tap, full_mask, _ = self.front_end.extract_bio_tap(
                full_batch["poses"], full_batch["frame_mask"], full_batch.get("timestamps_s"),
            )
            ref_ids = batch["reference_tokens"]["input_ids"].to(bio_tap.device)[cb_indices]
            ref_mask = batch["reference_tokens"]["attention_mask"].to(bio_tap.device)[cb_indices].bool()
            max_len = ref_ids.shape[1]

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
                        top_p=cb_dcd_top_p, cache_type=cb_dcd_cache_type, refresh_count=cb_dcd_refresh_count,
                    )
                    full_tokens = full_decode.sequences

                    # The gate reads the confidence/argmax the *real* decode produces
                    # (what DCD sees at inference), so this stays under no_grad.
                    trunc_decode = self.dlm_decoder.decode_spd_dcd(
                        enc_hidden=trunc_enc_hidden, enc_mask=trunc_enc_mask, max_length=max_len, 
                        diffusion_steps=cb_decode_steps, tau_dec=confidence_bound_tau, top_k=cb_spd_top_k, 
                        spd_renormalize=cb_spd_renormalize, spd_revision=cb_spd_revision, temperature=cb_temperature,
                        window_length=cb_dcd_window_length, max_window_length=cb_dcd_max_window_length, window_type=cb_dcd_window_type,
                        decode_algo=cb_dcd_decode_algo, decode_param=cb_dcd_decode_param, sample_top_k=cb_dcd_sample_top_k,
                        top_p=cb_dcd_top_p, cache_type=cb_dcd_cache_type, refresh_count=cb_dcd_refresh_count,
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
                    enc_hidden=trunc_enc_hidden, enc_mask=trunc_enc_mask,
                    decoded_tokens=trunc_tokens, remask_positions=cb_active_mask,
                )
                else:
                    trunc_logits = None
                    # logs["confidence_bound_loss"] = bio_tap.new_tensor(0.0)
                    # logs["confidence_bound_active"] = bio_tap.new_tensor(0.0)
            else:
                with torch.no_grad():
                    full_tokens, _ = self.generate_from_bio_tap(full_bio_tap, full_mask, max_text_tokens=max(1, max_len - 1))
                    full_tokens = self._pad_or_trim_tokens(full_tokens, max_len)

                trunc_logits, _ = self._ar_confidence_bound_logits(
                    bio_tap[cb_indices], bio_mask[cb_indices], max_len=max_len,
                    diffusion_steps=cb_decode_steps, tau_dec=confidence_bound_tau,
                )
                # AR decode layout is [lang, tok1, ..., eos] (decoder start fixed to language code, matching mBART's training-time shift), so 
                # after dropping the start token, slot j of full_tokens[:, 1:] lines up with reference slot j ([tok1, ..., eos, lang]); the loss's 
                # min-length slicing trims the dangling lang code. Do NOT slice the reference too — that would shift the verified gate off by one.
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
                # logs["confidence_bound_loss"] = cb.loss.detach()
                # logs["confidence_bound_active"] = cb.active_count.detach().float()

        if confidence_bound_enabled and confidence_bound_active and "trunc_decode_logits" in batch and "full_decode_tokens" in batch:
            cb = confidence_bound_loss(
                trunc_logits=batch["trunc_decode_logits"].to(bio_tap.device),
                full_tokens=batch["full_decode_tokens"].to(bio_tap.device),
                reference_tokens=batch.get("reference_tokens", {}).get("input_ids", None).to(bio_tap.device)
                if isinstance(batch.get("reference_tokens"), dict) else None,
                tau_cb=confidence_bound_tau, verified_full_evidence_gate=verified_full_evidence_gate,
                enabled=True, pad_token_id=self.tokenizer.pad_token_id,
            )
            translation_loss = translation_loss + float(cb_lambda) * cb.loss
            # logs["confidence_bound_loss"] = cb.loss.detach()
            # logs["confidence_bound_active"] = cb.active_count.detach().float()

        total = bio_loss + float(lambda_trans) * translation_loss
        logs["translation_loss"] = translation_loss.detach()
        logs["loss"] = total.detach()
        return Stage2LossOutput(total, bio_loss, translation_loss, logs)
