"""Stage-2 composition: shared pose/text front end + an AR or DLM decoder, 
and the per-window-mode training loss (`MisalignedSLTModel.forward_loss`)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from train.losses import bio_nll_dice_loss, confidence_bound_gate, confidence_bound_loss
from models.bio_head import RoPEBIOHead
from models.front_end import SLTFrontEnd
from models.membership_gate import build_omega, omega_cross_bias
from infer.commit_gate import open_span_start, select_target_span
from infer.duration_decode import deployed_decode_tags


def gate_skip_flags(
    bio_logits: torch.Tensor, frame_mask: torch.Tensor | None, min_span_frames: int = 0, 
    duration_prior=None, timestamps_s: torch.Tensor | None = None, commit_mask: torch.Tensor | None = None, 
    stream_start: bool = False, seam_is_terminator: bool = True,
) -> torch.Tensor:
    """Per-row 'the FSM would SKIP this window' flag (B,) from `deployed_decode_tags`.

    True when the window has NO terminated and NO open span — the all-gap / headless-left-truncated states the FSM never decodes 
    (buffer-start I never opens; no-span Ω inert). Single-window eval (RQ1) force-decodes them and reports the honest split: decoded-only 
    quality + skip rate. Must use the decode build_gate_omega selects on — the re-split changes span *existence*, not just count (interior 
    B's make a headless all-I window selectable) → skip and the Ω anchor can never disagree.
    """
    B, T = bio_logits.shape[:2]
    lengths = frame_mask.long().sum(dim=1) if frame_mask is not None else torch.full((B,), T)
    # EVERY tag-affecting argument must reach this call: build_gate_omega takes them all, so any defaulted here lets the skip flag and the 
    # Ω anchor disagree — the contradiction this docstring forbids, and the defect that made `stream_start` floor a third of RQ1's windows.
    tags = deployed_decode_tags(
        bio_logits, lengths, duration_prior, timestamps_s, commit_mask, stream_start=stream_start, seam_is_terminator=seam_is_terminator
    )
    skip = torch.zeros(B, dtype=torch.bool)
    for b in range(B):
        n = max(1, int(lengths[b].item()))
        row = tags[b, :n]
        # Same χ-frontier filter as build_gate_omega: else a window whose only complete span is already committed
        # reads as decodable while the Ω anchor discards it.
        chi_b = int(commit_mask[b, :n].sum().item()) if commit_mask is not None else 0
        if select_target_span(row, min_span_frames, skip_term_before=chi_b) is None and open_span_start(row) is None: skip[b] = True
    return skip


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

    `decoder="dlm"` → block-diffusion decoder (OPUT training / SPD+DCD inference), `"ar"` → AR seq2seq. Nothing else
    differs — front end, BIO head, sampler, FSM, commit gate identical — which is what makes AR-vs-DLM a clean test.
    """
    def __init__(
        self, front_end: SLTFrontEnd | None = None, tokenizer=None, decoder: str = "dlm", block_size: int = 8, 
        bio_hidden_dim: int = 384, bio_depth: int = 4, bio_nhead: int = 8, bio_dropout: float = 0.1, 
        bio_conv_stem_layers: int = 2, pretrained_path: str | None = None,
    ):
        super().__init__()
        # Caller passes UniSignMT5FrontEnd / UniSignMBartFrontEnd (models/unisign.py).
        self.front_end = front_end
        self.tokenizer = self.front_end.tokenizer
        self.decoder_type = decoder
        self.bio_head = RoPEBIOHead(
            input_dim=self.front_end.bio_tap_dim, hidden_dim=bio_hidden_dim,
            depth=bio_depth, nhead=bio_nhead, dropout=bio_dropout, num_classes=4,  # B/I/O + padding/UNK
            conv_stem_layers=bio_conv_stem_layers,  # local boundary inductive bias the UNet-less head lacks
        )
        # Load pretrained BEFORE building the DLM decoder: the substrate copies the current decoder/lm into its
        # vocab+1 [MASK] canvas, so the weights must already be in place.
        if pretrained_path:
            rep = self.front_end.load_pretrained(pretrained_path)
            print(f"slt | front-end warm-start {Path(pretrained_path).name}: {rep['pose_tensors']} pose + "
                  f"{rep['mt5_tensors']} LM tensors (missing {rep['pose_missing']}/{rep['mt5_missing']}, "
                  f"unexpected {rep['pose_unexpected']}/{rep['mt5_unexpected']})", flush=True)
        if decoder == "dlm": self.dlm_decoder = self.front_end.make_dlm_decoder(block_size)
        elif decoder != "ar": raise ValueError(f"Unsupported decoder type: {decoder}")
        # Semi-Markov duration prior for gate's anchor selection (infer/duration_decode.py); None = raw argmax. Set by train/slt.py, 
        # infer/stream.py and eval.py run_rq1 from the SAME inference.yaml `duration_decode` switch that drives the FSM's re-split.
        self.duration_prior = None

    def _pad_or_trim_tokens(self, tokens: torch.Tensor, target_len: int) -> torch.Tensor:
        if tokens.shape[1] > target_len: return tokens[:, :target_len]
        if tokens.shape[1] == target_len: return tokens
        pad_id = int(self.tokenizer.pad_token_id)
        pad = torch.full((tokens.shape[0], target_len - tokens.shape[1]), pad_id, dtype=tokens.dtype, device=tokens.device)
        return torch.cat([tokens, pad], dim=1)

    def encode_visual(self, batch: dict):
        bio_tap, bio_mask, timestamps, enc_hidden, enc_mask = self.front_end.encode(
            batch["poses"], batch["frame_mask"], batch.get("timestamps_s"),
        )
        return bio_tap, bio_mask, enc_hidden, enc_mask, timestamps

    @staticmethod
    def _log_per_mode(logs, mode_to_indices, idx_list, row_sum, row_valid) -> None:
        # oput_mode1 / oput_mode3 from ONE merged translation forward. `idx_list` is the supervised-row order that
        # forward saw, so a mode's global row ids map onto positions in the returned per-row stats.
        if not mode_to_indices or row_sum is None or row_valid is None: return
        pos = {int(g): p for p, g in enumerate(idx_list)}
        for mode, mode_idx in mode_to_indices.items():
            sel = torch.tensor([pos[int(i)] for i in mode_idx.tolist() if int(i) in pos], dtype=torch.long, device=row_sum.device)
            if sel.numel(): logs[f"oput_{mode}"] = row_sum[sel].sum() / row_valid[sel].sum().clamp(min=1)

    @staticmethod
    def _span_iou(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
        # tIoU of 2 [start, terminator) frame spans.
        if a is None or b is None: return 0.0
        lo = max(a[0], b[0]); hi = min(a[1], b[1])
        inter = max(0, hi - lo)
        union = (a[1] - a[0]) + (b[1] - b[0]) - inter
        return inter / union if union > 0 else 0.0

    def build_gate_omega(
        self, bio_logits: torch.Tensor, bio_labels: torch.Tensor | None, frame_mask: torch.Tensor, memory_len: int, 
        commit_mask: torch.Tensor | None = None, delta: int = 3, eps: float = 1e-4, min_span_frames: int = 0, iou_veto: float = 0.5, 
        gt_anchored: bool = False, timestamps_s: torch.Tensor | None = None, seam_is_terminator: bool = True, stream_start: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Membership-gate cross-attention bias (B,1,1,M) for a batch (docs/membership_gate.md).

        The span (s, τ) is ON-POLICY — the BIO head's OWN argmax via `select_target_span`, the rule the FSM uses.
        Training: `bio_labels` (window-relative B/I/O) gives the GT span for the IoU veto (§1.5); overlap below
        `iou_veto` rebuilds from the GT anchor (train-only rail, rate logged); `gt_anchored=True` forces GT (ablation).
        Inference (`bio_labels=None`): no GT, no veto, predicted span verbatim — deployed behaviour.
        """
        B, T, _ = bio_logits.shape
        device = bio_logits.device
        lengths = frame_mask.to(device).long().sum(dim=1) if frame_mask is not None else torch.full((B,), T, device=device)
        # Without the re-split the gate goes off-policy once the FSM decodes with duration: a back-to-back window's
        # raw argmax is ONE merged run, training the decoder on neighbour-sentence features the deployed gate masks.
        duration_prior = getattr(self, "duration_prior", None)  # getattr: test fakes drive this method unbound
        pred_tags = deployed_decode_tags(
            bio_logits, lengths, duration_prior, timestamps_s, commit_mask,
            seam_is_terminator=seam_is_terminator, stream_start=stream_start,
        )
        starts, terms, has_term = [], [], []
        vetoed, n_gt = 0, 0  # veto rate is over windows that HAVE a GT target (§1.6 diagnostic)
        for b in range(B):
            n = int(lengths[b].item())
            # χ-frontier filter mirroring the FSM (χ = in-buffer frame count): a span terminating before the commit
            # frontier is emitted content, never an anchor candidate.
            chi_b = int(commit_mask[b, :n].sum().item()) if commit_mask is not None else 0
            pred = select_target_span(pred_tags[b, :n], min_span_frames, skip_term_before=chi_b)
            if bio_labels is None: span = pred  # inference: on-policy, no veto, no GT
            else:
                gt = select_target_span(bio_labels[b, :n], min_span_frames)
                if gt is not None: n_gt += 1
                if gt_anchored:                 # ablation row: "GT-anchored with ±δ jitter" (gate-doc §3 table)
                    # Jitter is part of the ablation: exact GT anchors would hand the gate boundary
                    # info the on-policy head can never supply, conflating "teacher-forced m" with
                    # "oracle boundaries". δ-imprecision is the tolerance the gate's ramp/bands assume.
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
                else: span = pred
                
            if span is not None: 
                starts.append(int(span[0])); terms.append(int(span[1])); has_term.append(True)
            else:
                # An OPEN span running to buffer edge (Mode-2a right-truncation, buffer-cap forced commit) anchors Ω at ITS true start 
                # s (doc §2.8 forced path: γ≡γ_s, no right cliff → Ω≈0 for all-I interior); frame 0 would sweep the opening B and floor 
                # the span the gate must OPEN (attention ×0.01). On-policy first; GT only as the logged veto fallback when the prediction 
                # is missing or off by >δ (single-endpoint IoU-veto analog), since GT-first trains on anchors never seen deployed.
                pred_open = open_span_start(pred_tags[b, :n])
                if bio_labels is None: open_s = pred_open  # inference: on-policy, no veto
                else:
                    gt_open = open_span_start(bio_labels[b, :n])
                    if gt_open is not None: n_gt += 1
                    if pred_open is not None and (gt_open is None or abs(pred_open - gt_open) <= int(delta)): open_s = pred_open
                    elif gt_open is not None: open_s = gt_open; vetoed += 1
                    else: open_s = pred_open
                if open_s is not None:
                    starts.append(int(open_s)); terms.append(-1); has_term.append(False)
                else:
                    # Genuinely no span (all-gap / buffer-start-I leftover). INERT Ω — never decoded against: the FSM skips these buffers 
                    # ('translate_partial' decodes UNgated) and no-span rows carry no translation loss. n−1 is valid but NOT neutral 
                    # (frames < n−1−δ sit behind the left wall) — safe only while unread.
                    starts.append(max(0, n - 1)); terms.append(-1); has_term.append(False)

        out = build_omega(
            bio_logits, starts=torch.tensor(starts, device=device), terminators=torch.tensor(terms, device=device), 
            commit_mask=commit_mask, lengths=lengths, delta=delta, eps=eps, has_terminator=torch.tensor(has_term, device=device),
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
        dcd_top_p: float | None = None, dcd_cache_type: str = "none", decoder_start_token_id: int | None = None, 
        num_beams: int = 1, omega_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc_hidden, enc_mask = self.front_end.encode_memory(bio_tap, frame_mask)
        if self.decoder_type == "dlm":
            result = self.dlm_decoder.generate_spd_dcd(
                enc_hidden=enc_hidden, enc_mask=enc_mask, max_length=max_text_tokens, diffusion_steps=diffusion_steps,
                tau_dec=tau_dec, top_k=spd_top_k, spd_renormalize=spd_renormalize, spd_revision=spd_revision, temperature=temperature,
                window_length=dcd_window_length, max_window_length=dcd_max_window_length, window_type=dcd_window_type,
                decode_algo=dcd_decode_algo, decode_param=dcd_decode_param, sample_top_k=dcd_sample_top_k, top_p=dcd_top_p, 
                cache_type=dcd_cache_type, omega_bias=omega_bias,
            )
            # Slice to PRODUCED tokens (AR-arm parity): slot 0 (synthetic BOS) and everything past the first EOS are
            # 1.0 pad bookkeeping, so an unsliced mean pins the commit gate near 1 (~10 tokens on a 128 canvas → ≥
            # 0.92 regardless of quality; `translation_confident` always fires).
            seq, conf = result.sequences[:, 1:], result.confidence[:, 1:]
            if seq.shape[0] == 1:  # every live caller decodes one window/buffer at a time
                hits = (seq[0] == int(self.dlm_decoder.eos_index)).nonzero(as_tuple=False)
                if hits.numel(): seq, conf = seq[:, : int(hits[0]) + 1], conf[:, : int(hits[0]) + 1]
            return seq, conf

        # AR arm: the front end owns generation (mBART lang-code start / mT5 prompt-conditioned) and returns REAL
        # per-token confidence. `num_beams>1` is the clean baseline's beam search; the SLT AR arm stays greedy.
        return self.front_end.ar_generate(
            enc_hidden, enc_mask, max_new_tokens=max_text_tokens, num_beams=num_beams,
            decoder_start_id=decoder_start_token_id, omega_bias=omega_bias,
        )


    def _ar_confidence_bound_logits(
        self, bio_tap: torch.Tensor, frame_mask: torch.Tensor, max_len: int, omega_bias=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gradient-carrying AR logits on the truncated path.

        `generate_from_bio_tap` picks the prefix under no-grad; this forward replays it for the gradients the confidence-bound CE needs. 
        Generation and replay share one Ω — that gradient into the BIO logits is part of the Mode-2a coupling.
        """
        with torch.no_grad():
            trunc_tokens, _ = self.generate_from_bio_tap(bio_tap, frame_mask, max_text_tokens=max(1, max_len - 1), omega_bias=omega_bias)
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
        commit_mask: torch.Tensor | None = None, gate_stream_start: bool = False, gate_use_duration_prior: bool = True, **decode_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Poses → (bio_logits, tokens, confidence, gate_skip). Owns the BIO tap + membership gate; decode knobs (max_text_tokens / 
        diffusion_steps / tau_dec / spd_* / dcd_* / num_beams / decoder_start_token_id) pass through to `generate_from_bio_tap`, 
        declared once there. `gate_skip` (B, bool) marks windows the deployed FSM would never decode; all-False when gate is off."""
        bio_tap, mask, timestamps = self.front_end.extract_bio_tap(poses, frame_mask, timestamps_s)
        if getattr(self, "bio_branch_off", False) and not gate_enabled:
            # Clean-floor recipe: head frozen at random init and, gate off, unread — zeros fill the contract slot.
            bio_logits = bio_tap.new_zeros(*bio_tap.shape[:2], 4)
        else:
            bio_logits = self.bio_head(bio_tap, timestamps_s=timestamps, frame_mask=mask).logits
        # Inference gate (no veto), χ from the FSM commit log (single-window RQ1: none). Same Ω the decoder saw in
        # training (§1.3/§2.8); the AR arm injects it via HF cross-attn hooks (front_end.ar_generate).
        omega_bias = None
        gate_skip = torch.zeros(poses.shape[0], dtype=torch.bool)
        if gate_enabled:
            # gate_stream_start: this window is the 1st buffer of its stream, so a signing frame 0 IS a genuine onset. Without it a
            # window that opens mid-signing can never open a span (buffer-start I does not open), Ω falls to its no-span branch and
            # floors the whole window — on GT-span RQ1 windows, which begin exactly at sentence onset, that fires on a 3rd of them.
            #
            # gate_use_duration_prior=False: pick Ω anchor WITHOUT semi-Markov re-split. The re-split separates back-to-back sentences 
            # in a RUNNING STREAM; its split_bias is a sentence-COUNT prior tuned on whole videos. On a controlled single-anchor window 
            # it manufactures interior splits, 1st fragment becomes the anchor, and Ω's right wall floors the rest of the sentence being 
            # scored. FSM keeps the prior — streams genuinely contain adjacent sentences; a controlled window by construction doesn't.
            prior_hold = getattr(self, "duration_prior", None)
            if not gate_use_duration_prior: self.duration_prior = None
            try:
                omega_bias, _ = self.build_gate_omega(
                    bio_logits, None, mask, memory_len=self.front_end.prompt_length() + int(bio_tap.shape[1]), 
                    commit_mask=commit_mask, delta=gate_delta, eps=gate_eps, min_span_frames=gate_min_span_frames, 
                    timestamps_s=timestamps, stream_start=gate_stream_start,
                )
                gate_skip = gate_skip_flags(
                    bio_logits, mask, min_span_frames=gate_min_span_frames, duration_prior=getattr(self, "duration_prior", None),
                    timestamps_s=timestamps, commit_mask=commit_mask, stream_start=gate_stream_start,
                )
            finally: self.duration_prior = prior_hold
        tokens, confidence = self.generate_from_bio_tap(bio_tap, mask, omega_bias=omega_bias, **decode_kwargs)
        # DLM already strips its synthetic BOS in generate_from_bio_tap; the AR arm returns it raw (the Mode-2a
        # replay needs the start slot). Strip here so eval's confidence mean covers only produced tokens, both arms.
        if self.decoder_type == "ar": tokens, confidence = tokens[:, 1:], confidence[:, 1:]
        return bio_logits, tokens, confidence, gate_skip


    def forward_loss(
        self, batch: dict, lambda_trans: float = 1.0, lambda_bio: float = 1.0,
        dice_weight: float = 1.5, bio_class_weights: torch.Tensor | None = None,
        oput_t_low: float = 0.3, oput_t_high: float = 0.8, oput_sample_rollout: bool = False,
        oput_label_smoothing: float = 0.0, oput_rollout_eval_mode: bool = True, oput_eos_supervision: int | None = None,
        confidence_bound_enabled: bool = True, confidence_bound_active: bool = True, confidence_bound_tau: float = 0.75,
        cb_lambda: float = 0.3, verified_full_evidence_gate: bool = True, cb_decode_steps: int = 64,
        cb_dcd_window_length: int | None = None, cb_dcd_max_window_length: int | None = None, cb_dcd_window_type: str = "sliding",
        cb_dcd_decode_algo: str = "threshold", cb_dcd_decode_param: int | float | None = None, cb_dcd_sample_top_k: int | None = None,
        cb_dcd_top_p: float | None = None, cb_dcd_cache_type: str = "none", cb_spd_top_k: int = 1, cb_spd_renormalize: bool = True, 
        cb_spd_revision: bool = True, cb_temperature: float = 0.0, gate_enabled: bool = False, gate_delta: int = 3, 
        gate_eps: float = 1e-4, gate_min_span_frames: int = 0, gate_iou_veto: float = 0.5, gate_gt_anchored: bool = False,
    ) -> SLTLossOutput:
        """Stage-2 training loss for one mixed-mode batch.

        ``L = lambda_bio * L_BIO + lambda_trans * L_translation``, translation routed per window mode (`mode_names` from
        the sampler) to enforce premise P1: a truncated visual input never receives a partial text label.

        - BIO (all modes): Dice(1.5)+CE over in-window frames; padding/UNK ignored.
        - Mode 1 / Mode 3 (complete-anchor / first-complete-span): OPUT under fixed full conditioning
          (`dlm_decoder.oput_forward`; plain CE for AR). Raises if any other mode reaches that path.
        - Mode 2a (right-truncated): confidence-bound term only — gated CE toward the model's own no-grad full-evidence
          decode, at slots where that decode is reference-verified and the truncated decode confidently disagrees. Off
          during OPUT warmup (``confidence_bound_active=False``), weighted by ``cb_lambda``; see
          `dlm_decoder.remasked_logits` for why its gradient uses 1 re-masked forward, not back-prop through the decode.
        - Mode 2b / 2c / Mode 4 (left/both-truncated, all-gap): no translation loss — the model must stay silent.

        Per-mode losses logged separately.
        """
        bio_tap, bio_mask, enc_hidden, enc_mask, timestamps = self.encode_visual(batch)
        if float(lambda_bio) == 0.0 and not gate_enabled: # Clean-floor recipe
            # BIO branch SKIPPED, not zero-weighted — no head forward, no graph, no backward. 
            # The gate is its only other logits consumer and is off here.
            bio_out, bio_loss = None, bio_tap.new_zeros(())
        else:
            bio_out = self.bio_head(bio_tap, timestamps_s=timestamps, frame_mask=bio_mask)
            bio_loss = bio_nll_dice_loss(bio_out.logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=bio_class_weights)
        translation_loss = bio_tap.sum() * 0.0
        trans_weight = translation_loss.detach() * 0.0  # token weight of the OPUT pool (0 when no supervised windows)
        logs: dict[str, torch.Tensor] = {"bio_loss": bio_loss.detach()}
        # REALIZED mode mix (materialize() relabels windows the jitter reshapes, so the drawn ratios are not what trains). mean_logs over an 
        # epoch gives the realized fractions — the numbers the paper's "trained under the measured error distribution" claim actually refers to.
        realized = batch.get("mode_names")
        if isinstance(realized, list) and realized:
            for m in ("mode1", "mode2", "mode3", "mode4"):
                logs[f"mode_frac_{m}"] = bio_tap.new_tensor(sum(n == m for n in realized) / len(realized))
        target_tokens = batch.get("target_tokens")
        supervised = batch.get("translation_supervised")

        # Membership gate (docs/membership_gate.md): Ω from BIO posteriors + on-policy span conditions the decoder on the segmentation 
        # belief — the coupling, since translation-loss gradient reaches the BIO logits through Ω. BOTH arms inject same Ω (DLM manual
        # decode, AR via front_end.ar_omega_context) → §9.3 stays gated-vs-gated. None → pre-gate.
        omega_bias = None
        if gate_enabled:
            omega_bias, gate_stats = self.build_gate_omega(
                bio_out.logits, batch["bio_labels"], batch.get("frame_mask"), memory_len=int(enc_hidden.shape[1]),
                commit_mask=batch.get("commit_mask"), delta=gate_delta, eps=gate_eps, min_span_frames=gate_min_span_frames,
                iou_veto=gate_iou_veto, gt_anchored=gate_gt_anchored, timestamps_s=timestamps,
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
                # Merged for the same reason as the DLM arm below: the per-mode split doubled the decoder forwards
                # only to produce two log entries. Keeping both arms merged also keeps AR-vs-DLM a clean contrast.
                labels = target_tokens["labels"].to(bio_tap.device)
                translation_loss, row_sum, row_valid = self.front_end.ar_loss(
                    enc_hidden[idx], enc_mask[idx], labels[idx],
                    omega_bias=None if omega_bias is None else omega_bias[idx], row_stats=True,
                )
                trans_weight = ((labels[idx] != -100) & (labels[idx] != self.tokenizer.pad_token_id)).sum()
                trans_weight = trans_weight.to(dtype=translation_loss.dtype).clamp(min=1)
                self._log_per_mode(logs, mode_to_indices, idx_list if isinstance(mode_names, list) else [], row_sum, row_valid)
            else:
                labels = target_tokens["labels"].to(bio_tap.device)
                # ONE merged OPUT call over all supervised rows. Splitting by window mode existed only to log oput_mode1/oput_mode3 
                # separately, and cost 2x everything: OPUT runs 3 decoder passes over [xt|x0] canvas AND (under rollout_eval_mode) 
                # a full LM-encoder pass PER GROUP, at group sizes of ~2 and ~5 rows where GPU is pure launch overhead. The per-mode 
                # logs are recovered below from detached per-row sums. Note the merged loss normalizes over ALL valid tokens at once 
                # rather than averaging 2 group means by label-token count — the same estimator the group means each use internally, 
                # so this is the consistent definition, but it is not bit-identical to the old value.
                dlm_out = self.dlm_decoder.oput_forward(
                    enc_hidden=enc_hidden[idx], enc_mask=enc_mask[idx],
                    labels=labels[idx], t_low=oput_t_low, t_high=oput_t_high,
                    loss_over_all_positions=True, sample_rollout=oput_sample_rollout, label_smoothing=oput_label_smoothing,
                    rollout_eval_mode=oput_rollout_eval_mode, eos_supervision=int(
                        oput_eos_supervision if oput_eos_supervision is not None else self.dlm_decoder.block_size
                    ),
                    rollout_encode_fn=(
                        self.front_end.eval_encode_memory_fn(bio_tap[idx], bio_mask[idx]) if oput_rollout_eval_mode else None
                    ),
                    omega_bias=None if omega_bias is None else omega_bias[idx],
                )
                translation_loss = dlm_out["translation_loss"]
                # Pool weight = L_OPUT's OWN denominator (valid x0 positions, INCLUDING eos_supervision slots), not label-token count: L_OPUT 
                # was averaged over the former, so trans_weight*L_OPUT reproduces the OPUT loss sum exactly. Label count omits ~block_size EOS 
                # slots, which under-weighted OPUT (over-weighted CB) by ~eos/(len+eos) at λ=1. cb_weight already uses cb.loss's denominator.
                trans_weight = dlm_out["row_valid_count"].sum().to(dtype=translation_loss.dtype).clamp(min=1)
                self._log_per_mode(
                    logs, mode_to_indices, idx_list if isinstance(mode_names, list) else [],
                    dlm_out.get("row_loss_sum"), dlm_out.get("row_valid_count")
                )

        if (confidence_bound_enabled and confidence_bound_active and batch.get("full_evidence") is not None
            and batch.get("full_evidence_indices") is not None and batch["full_evidence_indices"].numel() > 0
            and batch.get("reference_tokens") is not None):
            cb_indices = batch["full_evidence_indices"].to(bio_tap.device)
            full_batch = batch["full_evidence"]
            # no_grad: the full-evidence view feeds only the self-target decode and its Ω, both no-grad. Without this
            # the pose encoder built a graph over the LONGER full-evidence window that nothing ever backwards through.
            with torch.no_grad():
                full_bio_tap, full_mask, full_timestamps = self.front_end.extract_bio_tap(
                    full_batch["poses"], full_batch["frame_mask"], full_batch.get("timestamps_s"),
                )
            ref_ids = batch["reference_tokens"]["input_ids"].to(bio_tap.device)[cb_indices]
            ref_mask = batch["reference_tokens"]["attention_mask"].to(bio_tap.device)[cb_indices].bool()
            max_len = ref_ids.shape[1]

            # Gate for the Mode-2a CB decodes, both arms (§9.3 symmetry): the truncated decode uses Ω from its own
            # posteriors (on-policy), the full-evidence self-target Ω from the full view's.
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
                    gt_anchored=gate_gt_anchored,  # must match the main call (:355): else the gt_anchored ablation
                    # trains its CB decode under a differently-anchored Omega than the OPUT rows.
                    timestamps_s=timestamps[cb_indices],  # real fps for the duration re-split (else 24fps fallback)
                )
                # Real timestamps + mask, else RoPE assumes 50fps indices on full-evidence view — 
                # a time-scale mismatch vs trunc view's real seconds.
                with torch.no_grad():
                    full_cb_bio_logits = self.bio_head(full_bio_tap, timestamps_s=full_timestamps, frame_mask=full_mask).logits
                cb_omega_full, _ = self.build_gate_omega(
                    full_cb_bio_logits, None, full_mask, memory_len=prompt_len + int(full_bio_tap.shape[1]),
                    # χ on BOTH views: a committed predecessor tail at the left edge must be
                    # floored here too, else the views differ by more than the right-truncation.
                    commit_mask=full_batch.get("commit_mask"),
                    delta=gate_delta, eps=gate_eps, min_span_frames=gate_min_span_frames,
                    timestamps_s=full_timestamps,
                )

            if self.decoder_type == "dlm": # Encode the trunc path ONCE, grad-bearing: the no-grad decode won't track it, remasked_logits will.
                trunc_enc_hidden, trunc_enc_mask = self.front_end.encode_memory(bio_tap[cb_indices], bio_mask[cb_indices])
                with torch.no_grad():
                    full_enc_hidden, full_enc_mask = self.front_end.encode_memory(full_bio_tap, full_mask)
                    full_decode = self.dlm_decoder.generate_spd_dcd(
                        enc_hidden=full_enc_hidden, enc_mask=full_enc_mask, max_length=max_len,
                        diffusion_steps=cb_decode_steps, tau_dec=confidence_bound_tau, top_k=cb_spd_top_k,
                        spd_renormalize=cb_spd_renormalize, spd_revision=cb_spd_revision, temperature=cb_temperature,
                        window_length=cb_dcd_window_length, max_window_length=cb_dcd_max_window_length, window_type=cb_dcd_window_type,
                        decode_algo=cb_dcd_decode_algo, decode_param=cb_dcd_decode_param, sample_top_k=cb_dcd_sample_top_k,
                        top_p=cb_dcd_top_p, cache_type=cb_dcd_cache_type, omega_bias=cb_omega_full,
                    )
                    full_tokens = full_decode.sequences

                    # Decode ONLY to pick which slots to defer-counterfactual (where the truncated decode disagrees
                    # with the full-evidence one) — its confidence does NOT gate the loss (see below).
                    trunc_decode = self.dlm_decoder.decode_spd_dcd(
                        enc_hidden=trunc_enc_hidden, enc_mask=trunc_enc_mask, max_length=max_len,
                        diffusion_steps=cb_decode_steps, tau_dec=confidence_bound_tau, top_k=cb_spd_top_k,
                        spd_renormalize=cb_spd_renormalize, spd_revision=cb_spd_revision, temperature=cb_temperature,
                        window_length=cb_dcd_window_length, max_window_length=cb_dcd_max_window_length, window_type=cb_dcd_window_type,
                        decode_algo=cb_dcd_decode_algo, decode_param=cb_dcd_decode_param, sample_top_k=cb_dcd_sample_top_k,
                        top_p=cb_dcd_top_p, cache_type=cb_dcd_cache_type, omega_bias=cb_omega_trunc,
                    )
                trunc_decoded = trunc_decode.sequences

                # Align the reference to the decode layout. The decode emits [BOS, tok1, ..., eos, ...] while the mBART tokenizer emits
                # [tok1, ..., eos, lang]: decode slot j holds reference slot j-1. Without this shift the verified gate (f_i == r_i)
                # compares misaligned slots and the CB term silently never fires.
                bos_col = torch.full((ref_ids.shape[0], 1), int(self.dlm_decoder.bos_index), dtype=ref_ids.dtype, device=ref_ids.device)
                cb_ref_ids = torch.cat([bos_col, ref_ids[:, :-1]], dim=1)
                cb_ref_mask = torch.cat([torch.ones_like(ref_mask[:, :1]), ref_mask[:, :-1]], dim=1)

                # CANDIDATE remask positions = disagreement slots (trunc≠full, full==ref, valid) — WHICH slots to feed the deferral 
                # counterfactual. Confidence threshold is NOT applied here; it is applied by confidence_bound_loss on the REMASKED logits, so 
                # the gate reads SAME distribution the CE trains. Gating on the decode confidence instead (the deployed commit path) leaves the 
                # loss free to push the deferral forward to confidently emit unseen continuation — the hallucination the AR arm's shared-logits 
                # fixed point rules out (measured: the DLM deferral forward hallucinated the continuation while the AR arm deferred). A constant 
                # confidence here just reuses the gate's disagreement/verified/pad conditions to pick candidates.
                candidate = confidence_bound_gate(
                    full_tokens=full_tokens, trunc_tokens=trunc_decoded, 
                    trunc_confidence=torch.ones_like(trunc_decode.confidence), tau_cb=0.0, reference_tokens=cb_ref_ids, valid_mask=cb_ref_mask,
                    verified_full_evidence_gate=verified_full_evidence_gate, pad_token_id=self.tokenizer.pad_token_id,
                )
                # trunc_tokens/confidence=None + active_mask=None → confidence_bound_loss derives π and ŷ from
                # trunc_logits itself (the remasked forward), exactly as the AR arm does: gate and CE share one
                # distribution, so CE lowering p(ŷ) lowers the gate confidence and the slot self-deactivates.
                trunc_tokens = trunc_confidence = cb_active_mask = None
                if candidate.any(): trunc_logits = self.dlm_decoder.remasked_logits(
                    enc_hidden=trunc_enc_hidden, enc_mask=trunc_enc_mask, decoded_tokens=trunc_decoded,
                    remask_positions=candidate, omega_bias=cb_omega_trunc,
                )
                else: trunc_logits = None
            else: # AR Mode-2a: reuse the cb_omega_* built above the arm split.
                with torch.no_grad():
                    full_tokens, _ = self.generate_from_bio_tap(
                        full_bio_tap, full_mask, max_text_tokens=max(1, max_len - 1), omega_bias=cb_omega_full
                    )
                    full_tokens = self._pad_or_trim_tokens(full_tokens, max_len)

                trunc_logits, _ = self._ar_confidence_bound_logits(
                    bio_tap[cb_indices], bio_mask[cb_indices], max_len=max_len, omega_bias=cb_omega_trunc,
                )
                # AR layout [lang, tok1, ..., eos] (start = language code, matching mBART's training shift), so dropping
                # it lines full_tokens[:, 1:] slot j up with reference slot j ([tok1, ..., eos, lang]); the loss's
                # min-length slicing trims the dangling lang code. DON'T slice the reference too — shifts the gate by 1.
                full_tokens = full_tokens[:, 1:]
                trunc_tokens, trunc_confidence = None, None
                cb_ref_ids, cb_ref_mask, cb_active_mask = ref_ids, ref_mask, None

            # Fold CB into SAME token-weighted pool as the OPUT modes (L_OPUT for 1/3, L_cb for 2a). A flat add would weight 2a independently 
            # of its batch share — ~9% of windows supplying most of the translation gradient. λ_cb=1.0 then IS the spec composition; ≠1 is an 
            # explicit deviation. The Mode-2a reference tokens ALWAYS enter the denominator (they are valid tokens carrying ZERO loss when no 
            # slot is active), so OPUT normalizes identically whether the gate fired or not — AR and DLM arms stay comparable on zero-active 
            # batches. cb_weight uses SAME [:, :seq_len] slice the loss scores, so cb_weight·L_cb is CB loss SUM exactly in both arms (the AR 
            # layout shift drops the trailing lang-code column). The expensive remasked forward is still skipped when nothing is active.
            if cb_ref_mask is not None:
                seq_len = full_tokens.shape[1] if trunc_logits is None else min(trunc_logits.shape[1], full_tokens.shape[1])
                cb_weight = cb_ref_mask[:, :seq_len].to(dtype=translation_loss.dtype).sum().clamp(min=1)
                cb_loss_val, cb_active_count = translation_loss.new_zeros(()), translation_loss.new_zeros(())
                if trunc_logits is not None:
                    cb = confidence_bound_loss(
                        trunc_logits=trunc_logits, full_tokens=full_tokens, reference_tokens=cb_ref_ids,
                        valid_mask=cb_ref_mask, trunc_tokens=trunc_tokens, trunc_confidence=trunc_confidence,
                        tau_cb=confidence_bound_tau, verified_full_evidence_gate=verified_full_evidence_gate,
                        enabled=True, pad_token_id=self.tokenizer.pad_token_id, active_mask=cb_active_mask,
                    )
                    cb_loss_val = cb.loss
                    cb_active_count = cb.active_count.detach().to(translation_loss.dtype)
                translation_loss = (translation_loss * trans_weight + float(cb_lambda) * cb_weight * cb_loss_val) \
                                                                        / (trans_weight + cb_weight).clamp(min=1)
                logs["cb_loss"] = cb_loss_val.detach()
                logs["cb_active_count"] = cb_active_count

        # `lambda_bio=0` = translation-only: the faithful Uni-Sign SLT recipe (1 label-smoothed CE) for the clean-floor arm. 
        # Also methodological — L_BIO teaches the shared pose trunk sentence boundaries, exactly the competence RQ1-A claims 
        # existing models LACK, so a floor trained with it would understate the misalignment problem.
        total = float(lambda_bio) * bio_loss + float(lambda_trans) * translation_loss
        logs["translation_loss"] = translation_loss.detach()
        logs["loss"] = total.detach()
        return SLTLossOutput(total, bio_loss, translation_loss, logs, bio_logits=bio_out.logits if bio_out is not None else None)
