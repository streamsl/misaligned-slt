"""Stage-2 composition: shared pose/text front end + an AR or DLM decoder,
and the per-window-mode training loss (`MisalignedSLTModel.forward_loss`)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from train.helpers import eval_mode
from train.losses import bio_nll_dice_loss, confidence_bound_gate, confidence_bound_loss
from models.bio_head import RoPEBIOHead
from models.front_end import SLTFrontEnd
from models.membership_gate import MembershipGate
from infer.duration_decode import DurationDecoder
from infer.commit_gate import select_target_span


@dataclass
class SLTLossOutput:
    loss: torch.Tensor
    bio_loss: torch.Tensor
    translation_loss: torch.Tensor
    logs: dict[str, torch.Tensor]
    # BIO head logits from this forward (None on the lambda_bio=0 clean-floor path). Exposed so dev eval can score
    # metrics without a second pose-encoder + head forward on identical inputs.
    bio_logits: torch.Tensor | None = None


class MisalignedSLTModel(nn.Module):
    """Stage-2 model: pluggable pose/text front end (`models.front_end.SLTFrontEnd`) + swappable AR or DLM decoder.

    The front end (Uni-Sign pose encoder + mT5 default / mBART ablation) emits from poses:
    - `bio_tap` (per-frame, length T): read DIRECTLY by `bio_head` (RoPEBIOHead) for phrase B/I/O — the streaming
      buffer's variable length never hits the seq2seq encoder's positions.
    - encoder memory (`enc_hidden`/`enc_mask`): cross-attended by the translation decoder.

    `decoder="dlm"` → block-diffusion decoder (OPUT training / block decode with SPD), `"ar"` → AR seq2seq. Nothing else
    differs — front end, BIO head, sampler, FSM, commit gate identical — which is what makes AR-vs-DLM a clean test.
    """
    def __init__(
        self, front_end: SLTFrontEnd, decoder: str = "dlm", block_size: int = 8,
        bio_hidden_dim: int = 384, bio_depth: int = 4, bio_nhead: int = 8, bio_dropout: float = 0.1, 
        bio_conv_stem_layers: int = 2, pretrained_path: str | None = None, shared_temporal: bool = True,
    ):
        super().__init__()
        # Caller passes UniSignMT5FrontEnd / UniSignMBartFrontEnd (models/unisign.py).
        self.front_end = front_end
        self.tokenizer = self.front_end.tokenizer
        self.decoder_type = decoder
        self.duration_model = None
        self.membership_gate = MembershipGate()
        self.bio_head = RoPEBIOHead(
            input_dim=self.front_end.bio_tap_dim, hidden_dim=bio_hidden_dim,
            depth=bio_depth, nhead=bio_nhead, dropout=bio_dropout, num_classes=4,  # B/I/O + padding/UNK
            conv_stem_layers=bio_conv_stem_layers,  # local boundary inductive bias the UNet-less head lacks
        )
        self.vl_mapper = None
        if shared_temporal:
            self.vl_mapper = nn.Sequential(nn.Linear(bio_hidden_dim, self.front_end.bio_tap_dim), nn.GELU(),
                                           nn.Linear(self.front_end.bio_tap_dim, self.front_end.bio_tap_dim))
            # Residual mapping preserves the pretrained translator input at initialization.
            nn.init.zeros_(self.vl_mapper[-1].weight); nn.init.zeros_(self.vl_mapper[-1].bias)
        # Load pretrained BEFORE building the DLM decoder: the substrate copies the current decoder/lm into its
        # vocab+1 [MASK] canvas, so the weights must already be in place.
        if pretrained_path:
            rep = self.front_end.load_pretrained(pretrained_path)
            print(f"slt | front-end warm-start {Path(pretrained_path).name}: {rep['pose_tensors']} pose + "
                  f"{rep['mt5_tensors']} LM tensors (missing {rep['pose_missing']}/{rep['mt5_missing']}, "
                  f"unexpected {rep['pose_unexpected']}/{rep['mt5_unexpected']})", flush=True)
        if decoder == "dlm": self.dlm_decoder = self.front_end.make_dlm_decoder(block_size)
        elif decoder != "ar": raise ValueError(f"Unsupported decoder type: {decoder}")

    def _pad_or_trim_tokens(self, tokens: torch.Tensor, target_len: int) -> torch.Tensor:
        if tokens.shape[1] > target_len: return tokens[:, :target_len]
        if tokens.shape[1] == target_len: return tokens
        pad_id = int(self.tokenizer.pad_token_id)
        pad = torch.full((tokens.shape[0], target_len - tokens.shape[1]), pad_id, dtype=tokens.dtype, device=tokens.device)
        return torch.cat([tokens, pad], dim=1)

    def _cb_decoded_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        # Keep emitted tokens through the first EOS. The synthetic start is never a CB slot,
        # even when its token ID is EOS; padding and all positions after EOS are excluded.
        valid = tokens != int(self.tokenizer.pad_token_id)
        valid[:, 0] = False
        eos = (tokens == int(self.tokenizer.eos_token_id)) & valid
        return valid & (eos.long().cumsum(dim=1) - eos.long() == 0)

    def encode_memory(self, bio_tap, bio_mask, temporal_features=None, timestamps_s=None):
        if self.vl_mapper is not None:
            if temporal_features is None: 
                temporal_features = self.bio_head.encode(bio_tap, timestamps_s=timestamps_s, frame_mask=bio_mask)
            bio_tap = bio_tap + self.vl_mapper(temporal_features)
        return self.front_end.encode_memory(bio_tap, bio_mask)

    def eval_encode_memory_fn(self, bio_tap, bio_mask, timestamps_s=None):
        def encode():
            with eval_mode(self): return self.encode_memory(bio_tap, bio_mask, timestamps_s=timestamps_s)
        return encode

    def encode_visual(self, batch: dict, use_bio: bool = True):
        tap, mask, timestamps = self.front_end.extract_bio_tap(batch["poses"], batch["frame_mask"], batch.get("timestamps_s"))
        out = self.bio_head(tap, timestamps_s=timestamps, frame_mask=mask) if use_bio else None
        memory, memory_mask = self.encode_memory(tap, mask, None if out is None else out.hidden_states, timestamps)
        return tap, mask, memory, memory_mask, timestamps, out

    @staticmethod
    def _log_per_mode(logs, mode_to_indices, idx_list, row_sum, row_valid) -> None:
        # oput_mode1 / oput_mode3 from ONE merged translation forward. `idx_list` is the supervised-row order that
        # forward saw, so a mode's global row ids map onto positions in the returned per-row stats.
        if not mode_to_indices or row_sum is None or row_valid is None: return
        pos = {int(g): p for p, g in enumerate(idx_list)}
        for mode, mode_idx in mode_to_indices.items():
            sel = torch.tensor([pos[int(i)] for i in mode_idx.tolist() if int(i) in pos], dtype=torch.long, device=row_sum.device)
            if sel.numel(): logs[f"oput_{mode}"] = row_sum[sel].sum() / row_valid[sel].sum().clamp(min=1)

    @torch.no_grad()
    def generate_from_bio_tap(
        self, bio_tap: torch.Tensor, frame_mask: torch.Tensor, max_text_tokens: int = 128, tau_dec: float = 0.5,
        spd_top_k: int = 1, spd_renormalize: bool = True, decoder_start_token_id: int | None = None,
        num_beams: int = 1, omega_bias: torch.Tensor | None = None, temporal_features=None, timestamps_s=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc_hidden, enc_mask = self.encode_memory(bio_tap, frame_mask, temporal_features, timestamps_s)
        if self.decoder_type == "dlm":
            result = self.dlm_decoder.generate(
                enc_hidden, enc_mask, max_length=max_text_tokens, threshold=tau_dec, spd_top_k=spd_top_k,
                spd_renormalize=spd_renormalize, omega_bias=omega_bias,
            )
            # Sequential decoder passes of this decode (a batch's count: rows decode together). The AR arm's is
            # its generated length: 1 cached step per token plus the confidence pass. RQ1 reports the mean.
            self.last_decode_passes = int(result.forwards + result.cache_appends)
            # Slice to PRODUCED tokens (AR-arm parity): slot 0 (synthetic BOS) and everything past the first EOS are
            # 1.0 pad bookkeeping, so an unsliced mean pins the commit gate near 1 (~10 tokens on a 128 canvas → ≥
            # 0.92 regardless of quality; `translation_confident` always fires).
            seq, conf = result.sequences[:, 1:], result.confidence[:, 1:]
            if seq.shape[0] == 1:  # every live caller decodes 1 window/buffer at a time
                hits = (seq[0] == int(self.dlm_decoder.eos_index)).nonzero(as_tuple=False)
                if hits.numel(): seq, conf = seq[:, : int(hits[0]) + 1], conf[:, : int(hits[0]) + 1]
            return seq, conf

        # AR arm: the front end owns generation (mBART lang-code start / mT5 prompt-conditioned) and returns REAL
        # per-token confidence. `num_beams>1` is the clean baseline's beam search; the SLT AR arm stays greedy.
        generated, confidence = self.front_end.ar_generate(
            enc_hidden, enc_mask, max_new_tokens=max_text_tokens, num_beams=num_beams,
            decoder_start_id=decoder_start_token_id, omega_bias=omega_bias,
        )
        self.last_decode_passes = int(generated.shape[1])  # start slot + N tokens = N cached steps + 1 confidence pass
        return generated, confidence

    def _ar_confidence_bound_logits( # Gradient-carrying AR logits on the truncated path
        self, bio_tap: torch.Tensor, frame_mask: torch.Tensor, max_len: int, omega_bias=None, timestamps_s=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # `generate_from_bio_tap` picks prefix under no-grad; this forward replays it for gradients the confidence-bound CE needs.
        # Generation and replay share same first-span conditioning.
        # eval_mode: the selection decode must be the distribution inference sees (dropout-free, BN stats untouched)
        with torch.no_grad(), eval_mode(self):
            trunc_tokens, _ = self.generate_from_bio_tap(
                bio_tap, frame_mask, max_text_tokens=max(1, max_len - 1), omega_bias=omega_bias, timestamps_s=timestamps_s
            )
            trunc_tokens = self._pad_or_trim_tokens(trunc_tokens, max_len)

        enc_hidden, enc_mask = self.encode_memory(bio_tap, frame_mask, timestamps_s=timestamps_s)
        with self.front_end.ar_omega_context(omega_bias):
            out = self.front_end.lm_model(
                encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden), attention_mask=enc_mask,
                decoder_input_ids=trunc_tokens[:, :-1].contiguous(), use_cache=False, return_dict=True,
            )
        return out.logits, trunc_tokens


    @torch.no_grad()
    def generate_from_poses(
        self, poses: torch.Tensor, frame_mask: torch.Tensor, timestamps_s: torch.Tensor | None = None, *,
        gate_enabled: bool = False, gate_eps: float = 1e-4, gate_min_span_frames: int = 0, commit_mask: torch.Tensor | None = None, 
        gate_stream_start: bool = False, gate_anchor: torch.Tensor | None = None, **decode_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Poses → (bio_logits, tokens, confidence, gate_skip). Owns the BIO tap + membership gate; decode knobs (max_text_tokens /
        tau_dec / spd_* / num_beams / decoder_start_token_id) pass via to `generate_from_bio_tap`, declared once there. `gate_skip` 
        (B, bool) marks windows the deployed FSM would never decode; all-False when gate is off."""
        bio_tap, mask, timestamps = self.front_end.extract_bio_tap(poses, frame_mask, timestamps_s)
        if getattr(self, "bio_branch_off", False) and not gate_enabled:
            # Clean-floor recipe: head frozen at random init and, gate off, unread — zeros fill the contract slot.
            bio_logits = bio_tap.new_zeros(*bio_tap.shape[:2], 4)
        else:
            bio_output = self.bio_head(bio_tap, timestamps_s=timestamps, frame_mask=mask)
            bio_logits = bio_output.logits
        # Inference gate, χ from the FSM commit log (single-window RQ1: none). Same Ω the decoder saw in
        # training (§1.3/§2.8); the AR arm injects it via HF cross-attn hooks (front_end.ar_generate).
        omega_bias = None
        gate_skip = torch.zeros(poses.shape[0], dtype=torch.bool)
        if gate_enabled:
            omega_bias, gate_stats = self.membership_gate(
                bio_logits, mask, decoder=DurationDecoder(self.duration_model),
                memory_len=self.front_end.prompt_length() + int(bio_tap.shape[1]), commit_mask=commit_mask, eps=gate_eps,
                min_span_frames=gate_min_span_frames, stream_start=gate_stream_start, anchor_override=gate_anchor, timestamps_s=timestamps,
            )
            gate_skip = gate_stats["skip"]
            if gate_anchor is not None: gate_skip[(gate_anchor[:, 0] >= 0).cpu()] = False   # a known span is always decodable
        tokens, confidence = self.generate_from_bio_tap(
            bio_tap, mask, omega_bias=omega_bias, temporal_features=bio_output.hidden_states if self.vl_mapper is not None else None, 
            timestamps_s=timestamps, **decode_kwargs
        )
        # DLM already strips its synthetic BOS in generate_from_bio_tap; the AR arm returns it raw (the Mode-2a
        # replay needs the start slot). Strip here so eval's confidence mean covers only produced tokens, both arms.
        if self.decoder_type == "ar": tokens, confidence = tokens[:, 1:], confidence[:, 1:]
        return bio_logits, tokens, confidence, gate_skip


    @staticmethod
    def _candidate_frames(cands: list[dict], ts: torch.Tensor, n: int) -> list[dict]:
        # Window-relative seconds -> frame indices with the label convention: B = first frame at or after the start,
        # terminator = first frame at or after the end (n when the sentence ends at the window edge).
        t = ts[:n].detach().float().cpu()
        out = []
        for c in cands:
            b = int(torch.searchsorted(t, torch.tensor(float(c["start_s"]), dtype=t.dtype)).item())
            e = int(torch.searchsorted(t, torch.tensor(float(c["end_s"]), dtype=t.dtype)).item())
            out.append({**c, "b_idx": min(b, n), "t_idx": min(e, n)})
        return out


    def gt_target_spans(self, batch: dict, timestamps, bio_mask) -> list | None:
        """Per-row frame span (B, terminator) of the sentence the collated text supervises, from the window's candidate list; None for 
        rows without a text target. Used to identify whether the training target is complete, and for localization diagnostics."""
        cands, firsts = batch.get("candidate_sentences"), batch.get("translation_targets")
        if cands is None or firsts is None or timestamps is None: return None
        out = []
        for b in range(len(cands)):
            first = firsts[b]
            first_text = (first["text"] if isinstance(first, dict) else getattr(first, "text", None)) if first is not None else None
            span = None
            if first_text:
                n = int(bio_mask[b].long().sum())
                for c in self._candidate_frames(cands[b], timestamps[b], n):
                    if c["text"] == first_text and c["t_idx"] > c["b_idx"]: span = (int(c["b_idx"]), int(min(c["t_idx"], n - 1))); break
            out.append(span)
        return out


    def forward_loss(
        self, batch: dict, lambda_trans: float = 1.0, lambda_bio: float = 1.0,
        dice_weight: float = 1.5, bio_class_weights: torch.Tensor | None = None,
        oput_t_low: float = 0.3, oput_t_high: float = 0.8, oput_sample_rollout: bool = False,
        oput_label_smoothing: float = 0.0, oput_rollout_eval_mode: bool = True, oput_eos_supervision: int | None = None,
        cb_enabled: bool = True, cb_active: bool = True, cb_tau: float = 0.75, cb_lambda: float = 0.3, cb_verified_gate: bool = True, 
        cb_belief_gap: bool = True, cb_tau_dec: float = 0.5, cb_spd_top_k: int = 1, cb_spd_renormalize: bool = True,
        gate_enabled: bool = False, gate_eps: float = 1e-4, gate_min_span_frames: int = 0,
    ) -> SLTLossOutput:
        """Stage-2 training loss for one mixed-mode batch.

        ``L = lambda_bio * L_BIO + lambda_trans * L_translation``, translation routed per window mode (`mode_names` from sampler) to 
        enforce premise P1: a truncated visual input never receives a partial text label.

        - BIO (all modes): class-weighted CE plus binary signing Dice; over in-window frames, padding/UNK ignored.
        - Mode 1 / Mode 3 (complete-anchor / first-complete-span): OPUT under fixed full conditioning (`dlm_decoder.oput_forward`; 
          plain CE for AR). Raises if any other mode reaches that path.
        - Mode 2a (right-truncated): confidence-bound term only — gated CE toward the model's own no-grad full-evidence decode, 
          at slots where that decode is reference-verified and the truncated decode confidently disagrees. Off during OPUT warmup 
          (`cb_active=False`), weighted by `cb_lambda`; see `dlm_decoder.remasked_logits` for why its gradient uses 1 re-masked 
          forward, not back-prop through the decode.
        - Mode 2b / 2c / Mode 4 (left/both-truncated, all-gap): BIO only. The FSM does not call the decoder.

        Per-mode losses logged separately.
        """
        bio_tap, bio_mask, enc_hidden, enc_mask, timestamps, bio_out = self.encode_visual(
            batch, use_bio=not (float(lambda_bio) == 0.0 and not gate_enabled)
        )
        bio_loss = bio_nll_dice_loss(
            bio_out.logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=bio_class_weights
        ) if bio_out is not None else bio_tap.new_zeros(())
        translation_loss = bio_tap.sum() * 0.0
        logs: dict[str, torch.Tensor] = {"bio_loss": bio_loss.detach()}

        # REALIZED mode mix (materialize() relabels windows jitter reshapes, so the drawn ratios are not what trains).
        realized = batch.get("mode_names")
        if isinstance(realized, list) and realized:
            for m in ("mode1", "mode2", "mode3", "mode4"):
                logs[f"mode_frac_{m}"] = bio_tap.new_tensor(sum(n == m for n in realized) / len(realized))
        target_tokens = batch.get("target_tokens")
        supervised = batch.get("translation_supervised")

        # Membership gate (docs/membership_gate.md): Ω from BIO posteriors + on-policy span conditions decoder on segmentation belief. 
        # Live first-span probabilities carry caption gradients to BIO classifier. BOTH arms inject same Ω (DLM manual decode, AR via 
        # front_end.ar_omega_context). None → pre-gate.
        omega_bias, gate_stats = None, {}
        if bio_out is not None and gate_enabled:
            gt_spans = self.gt_target_spans(batch, timestamps, bio_mask)
            cb_rows = batch.get("full_evidence_indices")
            if cb_rows is not None and cb_rows.numel():
                if gt_spans is None: gt_spans = [
                    select_target_span(row[:int(n)], max(1, gate_min_span_frames))
                    for row, n in zip(batch["bio_labels"], bio_mask.long().sum(1))
                ]
                for i in cb_rows.tolist(): gt_spans[i] = None
            gate_omega, gate_stats = self.membership_gate.for_supervision(
                bio_out.logits, batch["bio_labels"], bio_mask, decoder=DurationDecoder(self.duration_model), 
                memory_len=int(enc_hidden.shape[1]), commit_mask=batch.get("commit_mask"), eps=gate_eps, 
                min_span_frames=gate_min_span_frames, gt_spans=gt_spans, timestamps_s=timestamps,
            )
            omega_bias = gate_omega
            logs["gate_anchor_hit_rate"] = bio_tap.new_tensor(gate_stats["anchor_hit_rate"])
            logs["gate_closed_probability"] = gate_stats["closed_probability"]
            logs["gate_open_probability"] = gate_stats["open_probability"]

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
                # Merged for the same reason as the DLM arm below: the per-mode split doubled the decoder forwards
                # only to produce 2 log entries. Keeping both arms merged also keeps AR-vs-DLM a clean contrast.
                labels = target_tokens["labels"].to(bio_tap.device)
                translation_loss, row_sum, row_valid = self.front_end.ar_loss(
                    enc_hidden[idx], enc_mask[idx], labels[idx], label_smoothing=oput_label_smoothing,
                    omega_bias=None if omega_bias is None else omega_bias[idx], row_stats=True,
                )
                self._log_per_mode(logs, mode_to_indices, idx_list if isinstance(mode_names, list) else [], row_sum, row_valid)
            else:
                labels = target_tokens["labels"].to(bio_tap.device)
                # ONE merged OPUT call over all supervised rows. Splitting by window mode existed only to log oput_mode1/oput_mode3 
                # separately, and cost 2x everything: OPUT runs 3 decoder passes over [xt|x0] canvas AND (under rollout_eval_mode) 
                # a full LM-encoder pass PER GROUP, at group sizes of ~2 and ~5 rows where GPU is pure launch overhead. The per-mode 
                # logs are recovered below from detached per-row sums. Note the merged loss normalizes over ALL valid tokens at once 
                # rather than averaging 2 group means by label-token count — the same estimator the group means each use internally, 
                # so this is the consistent definition.
                dlm_out = self.dlm_decoder.oput_forward(
                    enc_hidden=enc_hidden[idx], enc_mask=enc_mask[idx],
                    labels=labels[idx], t_low=oput_t_low, t_high=oput_t_high,
                    loss_over_all_positions=True, sample_rollout=oput_sample_rollout, label_smoothing=oput_label_smoothing,
                    rollout_eval_mode=oput_rollout_eval_mode, eos_supervision=int(
                        oput_eos_supervision if oput_eos_supervision is not None else self.dlm_decoder.block_size
                    ),
                    rollout_encode_fn=self.eval_encode_memory_fn(
                        bio_tap[idx], bio_mask[idx], timestamps[idx] if timestamps is not None else None
                    ) if oput_rollout_eval_mode else None,
                    omega_bias=None if omega_bias is None else omega_bias[idx],
                )
                translation_loss = dlm_out["translation_loss"]
                self._log_per_mode(
                    logs, mode_to_indices, idx_list if isinstance(mode_names, list) else [],
                    dlm_out.get("row_loss_sum"), dlm_out.get("row_valid_count")
                )

        if cb_enabled and cb_active and batch.get("full_evidence") is not None and batch.get("full_evidence_indices") is not None \
                      and batch["full_evidence_indices"].numel() > 0 and batch.get("reference_tokens") is not None:
            cb_indices = batch["full_evidence_indices"].to(bio_tap.device)
            full_batch = batch["full_evidence"]
            # no_grad: the full-evidence view feeds only the self-target decode and its Ω, both no-grad. Without this
            # the pose encoder built a graph over the LONGER full-evidence window that nothing ever backwards through.
            # eval_mode: reference-side forward — dropout off, and BN running stats must not absorb the longer
            # full-evidence windows (BN updates in forward even under no_grad).
            with torch.no_grad(), eval_mode(self):
                full_bio_tap, full_mask, full_timestamps = self.front_end.extract_bio_tap(
                    full_batch["poses"], full_batch["frame_mask"], full_batch.get("timestamps_s"),
                )
            ref_ids = batch["reference_tokens"]["input_ids"].to(bio_tap.device)[cb_indices]
            ref_mask = batch["reference_tokens"]["attention_mask"].to(bio_tap.device)[cb_indices].bool()
            max_len = ref_ids.shape[1]

            # Both CB views use their own first-span probabilities under the same state rules.
            cb_omega_trunc = cb_omega_full = None
            if gate_enabled:
                prompt_len = self.front_end.prompt_length()
                cb_omega_trunc = omega_bias[cb_indices] # The main gate already selected the open readout for these Mode-2a rows.
                # Real timestamps + mask, else RoPE assumes 50fps indices on full-evidence view —
                # a time-scale mismatch vs trunc view's real seconds.
                with torch.no_grad(), eval_mode(self):
                    full_cb_bio_logits = self.bio_head(full_bio_tap, timestamps_s=full_timestamps, frame_mask=full_mask).logits
                cb_omega_full, _ = self.membership_gate.for_supervision(
                    full_cb_bio_logits, full_batch.get("bio_labels"), full_mask,
                    decoder=DurationDecoder(self.duration_model), memory_len=prompt_len + int(full_bio_tap.shape[1]),
                    # χ on BOTH views: a committed predecessor tail at the left edge must be
                    # floored here too, else the views differ by more than the right-truncation.
                    commit_mask=full_batch.get("commit_mask"), eps=gate_eps, min_span_frames=gate_min_span_frames,
                    timestamps_s=full_timestamps, gt_spans=self.gt_target_spans(full_batch, full_timestamps, full_mask),
                )

            if self.decoder_type == "dlm": # Encode the trunc path ONCE: the no-grad decode won't track it, remasked_logits will.
                trunc_enc_hidden, trunc_enc_mask = self.encode_memory(
                    bio_tap[cb_indices], bio_mask[cb_indices], timestamps_s=timestamps[cb_indices] if timestamps is not None else None
                )
                # eval_mode: both no-grad decodes are teachers (dlm.yaml oput: the rollout is drawn "the distribution inference sees");
                # the grad path (trunc encode above, remasked_logits below) stays in train mode so dropout regularizes only what trains.
                with torch.no_grad(), eval_mode(self):
                    full_enc_hidden, full_enc_mask = self.encode_memory(full_bio_tap, full_mask, timestamps_s=full_timestamps)
                    # Same decode as inference (`cb_tau_dec` = tau_dec): the self-target is what the deployed decode would emit.
                    full_decode = self.dlm_decoder.generate(
                        full_enc_hidden, full_enc_mask, max_length=max_len, threshold=cb_tau_dec, spd_top_k=cb_spd_top_k,
                        spd_renormalize=cb_spd_renormalize, omega_bias=cb_omega_full,
                    )
                    full_tokens, full_conf = full_decode.sequences, full_decode.confidence

                    # Decode ONLY to pick which slots to re-mask (where the truncated decode disagrees
                    # with the full-evidence one) — its confidence does NOT gate the loss (see below).
                    trunc_decode = self.dlm_decoder.generate(
                        trunc_enc_hidden, trunc_enc_mask, max_length=max_len, threshold=cb_tau_dec, spd_top_k=cb_spd_top_k,
                        spd_renormalize=cb_spd_renormalize, omega_bias=cb_omega_trunc,
                    )
                trunc_decoded = trunc_decode.sequences

                # Align the reference to the decode layout. The decode emits [BOS, tok1, ..., eos, ...] while the mBART tokenizer emits
                # [tok1, ..., eos, lang]: decode slot j holds reference slot j-1. Without this shift the verified gate (f_i == r_i)
                # compares misaligned slots and the CB term silently never fires.
                bos_col = torch.full((ref_ids.shape[0], 1), int(self.dlm_decoder.bos_index), dtype=ref_ids.dtype, device=ref_ids.device)
                cb_ref_ids = torch.cat([bos_col, ref_ids[:, :-1]], dim=1)
                cb_ref_mask = torch.cat([torch.zeros_like(ref_mask[:, :1]), ref_mask[:, :-1]], dim=1)

                # Only actual decoded slots can become remask candidates. Confidence is tested
                # below on the live remasked logits, not on the selection decode's confidence.
                candidate = confidence_bound_gate(
                    full_tokens=full_tokens, trunc_tokens=trunc_decoded, 
                    trunc_confidence=torch.ones_like(trunc_decode.confidence), tau_cb=0.0, reference_tokens=cb_ref_ids, 
                    valid_mask=cb_ref_mask & self._cb_decoded_mask(full_tokens) & self._cb_decoded_mask(trunc_decoded),
                    verified_full_evidence_gate=cb_verified_gate, pad_token_id=self.tokenizer.pad_token_id,
                )
                cb_slot_mask = candidate
                if candidate.any(): trunc_logits = self.dlm_decoder.remasked_logits(
                    enc_hidden=trunc_enc_hidden, enc_mask=trunc_enc_mask, decoded_tokens=trunc_decoded,
                    remask_positions=candidate, omega_bias=cb_omega_trunc,
                )
                else: trunc_logits = None
            else: # AR Mode-2a: reuse the cb_omega_* built above the arm split.
                with torch.no_grad(), eval_mode(self):
                    full_tokens, full_conf = self.generate_from_bio_tap(
                        full_bio_tap, full_mask, max_text_tokens=max(1, max_len - 1), omega_bias=cb_omega_full, timestamps_s=full_timestamps
                    )
                    if full_conf is not None:
                        if full_conf.shape[1] < max_len: full_conf = torch.cat(
                            [full_conf, full_conf.new_zeros((full_conf.shape[0], max_len - full_conf.shape[1]))], dim=1
                        )
                        else: full_conf = full_conf[:, :max_len]
                    full_tokens = self._pad_or_trim_tokens(full_tokens, max_len)

                trunc_logits, trunc_decoded = self._ar_confidence_bound_logits(
                    bio_tap[cb_indices], bio_mask[cb_indices], max_len=max_len, omega_bias=cb_omega_trunc, 
                    timestamps_s=timestamps[cb_indices] if timestamps is not None else None,
                )
                # AR layout [lang, tok1, ..., eos] (start = language code, matching mBART's training shift), so dropping
                # it lines full_tokens[:, 1:] slot j up with reference slot j ([tok1, ..., eos, lang]); the loss's
                # min-length slicing trims the dangling lang code. DON'T slice the reference too — shifts the gate by 1.
                cb_slot_mask = (self._cb_decoded_mask(full_tokens) & self._cb_decoded_mask(trunc_decoded))[:, 1:]
                full_tokens = full_tokens[:, 1:]
                if full_conf is not None: full_conf = full_conf[:, 1:]
                cb_ref_ids, cb_ref_mask = ref_ids, ref_mask

            # Per-mode routing: L = L_OPUT + λ_cb·L_cb, each term under its OWN normalization (OPUT per supervised token;
            # CB per valid Mode-2a slot, inside confidence_bound_loss). A token-pooled fold would put every Mode-2a reference
            # token into a shared denominator, most of them structurally zero-loss, and shrink the clean-span OPUT gradient on
            # nearly every batch (a clean-translation drift). A flat add cannot let 2a dominate: L_cb is a small fraction of
            # L_OPUT at λ_cb=1. Zero-active batches reduce exactly to L_OPUT in both arms.
            if cb_ref_mask is not None:
                cb_loss_val, cb_active_count = translation_loss.new_zeros(()), translation_loss.new_zeros(())
                if trunc_logits is not None:
                    # Restrict the active gate, not the reference-token denominator. Both arms
                    # test confidence and disagreement on the same live logits that CE trains.
                    seq_len = min(trunc_logits.shape[1], full_tokens.shape[1])
                    with torch.no_grad():
                        trunc_confidence, trunc_tokens = trunc_logits[:, :seq_len].softmax(dim=-1).max(dim=-1)
                        cb_active_mask = confidence_bound_gate(
                            full_tokens=full_tokens[:, :seq_len], trunc_tokens=trunc_tokens, trunc_confidence=trunc_confidence,
                            reference_tokens=cb_ref_ids[:, :seq_len], valid_mask=cb_ref_mask[:, :seq_len] & cb_slot_mask[:, :seq_len],
                            tau_cb=cb_tau, verified_full_evidence_gate=cb_verified_gate, pad_token_id=self.tokenizer.pad_token_id,
                        )
                    cb = confidence_bound_loss(
                        trunc_logits=trunc_logits, full_tokens=full_tokens, reference_tokens=cb_ref_ids, valid_mask=cb_ref_mask,
                        trunc_tokens=trunc_tokens, trunc_confidence=trunc_confidence, tau_cb=cb_tau,
                        verified_full_evidence_gate=cb_verified_gate, enabled=True, pad_token_id=self.tokenizer.pad_token_id,
                        active_mask=cb_active_mask, full_confidence=full_conf if cb_belief_gap else None,
                    )
                    cb_loss_val = cb.loss
                    cb_active_count = cb.active_count.detach().to(translation_loss.dtype)
                translation_loss = translation_loss + float(cb_lambda) * cb_loss_val
                logs["cb_loss"] = cb_loss_val.detach()
                logs["cb_active_count"] = cb_active_count

        # `lambda_bio=0` = translation-only: the faithful Uni-Sign SLT recipe (1 label-smoothed CE) for the clean-floor arm.
        # Also methodological — L_BIO teaches the shared pose trunk sentence boundaries, exactly the competence RQ1 claims
        # existing models LACK, so a floor trained with it would understate the misalignment problem.
        total = float(lambda_bio) * bio_loss + float(lambda_trans) * translation_loss
        logs["translation_loss"] = translation_loss.detach()
        logs["loss"] = total.detach()
        return SLTLossOutput(total, bio_loss, translation_loss, logs, bio_logits=bio_out.logits if bio_out is not None else None)
