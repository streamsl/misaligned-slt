"""Stage-2 composition: a segmentation branch (own pose encoder + BIO head), the translator front end, the Ω soft 
crop between them, an AR or DLM decoder, and the stage-2 training loss (`MisalignedSLTModel.forward_loss`)."""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import copy

import torch
import torch.nn as nn
from models.bio_head import RoPEBIOHead
from models.front_end import SLTFrontEnd
from models.membership_gate import MembershipGate

from train.helpers import eval_mode
from train.losses import bio_nll_dice_loss
from infer.duration_decode import DurationDecoder


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

    Two pose streams read the same frames. Each encoder is pretrained alone and then loaded together, as in TwoStream-SLT
    (arXiv 2211.01367 §4); there the 2 streams share 1 task and read different inputs, here they read 1 input for 2 tasks:
    - segmentation: `bio_pose_encoder` (S1's weights) -> `bio_head` (RoPEBIOHead) -> phrase B/I/O logits. It reads 
      per-frame features directly, so the buffer's variable length never hits the seq2seq encoder's positions.
    - translation: `front_end.pose_encoder` (the clean translator's weights) -> task prompt + LM encoder -> decoder.
      Ω (first-span membership from the BIO logits) biases the LM encoder's self-attention keys and the decoder's
      cross-attention, so at a hard membership the translator reads the cropped span: up to the ε leak when the span
      starts right after the prompt, and with a prompt<->span relative-position offset otherwise. Ω is the only
      coupling: no feature path carries whole-window context past the crop.
    
    At step 0 the branches hold S1's and the clean translator's weights; the gate-off AR arm decodes as the clean
    translator token for token (the DLM adds its [MASK] row and block decoder on top).

    `decoder="dlm"` → block-diffusion decoder (OPUT training / block decode with SPD), `"ar"` → AR seq2seq. Nothing else
    differs — front end, BIO head, sampler, FSM, commit gate identical — which is what makes AR-vs-DLM a clean test.
    """
    def __init__(
        self, front_end: SLTFrontEnd, decoder: str = "dlm", block_size: int = 8, bio_hidden_dim: int = 384, bio_depth: int = 4, 
        bio_nhead: int = 8, bio_dropout: float = 0.1, bio_conv_stem_layers: int = 2, bio_attention_radius_s: float | None = None, 
        pretrained_path: str | None = None, segmentation_branch: bool = True,
    ):
        super().__init__()
        # Caller passes UniSignMT5FrontEnd / UniSignMBartFrontEnd (models/unisign.py).
        self.front_end = front_end
        self.tokenizer = self.front_end.tokenizer
        self.decoder_type = decoder
        self.duration_model = None
        self.membership_gate = MembershipGate()
        self.bio_head = RoPEBIOHead( # The clean floor has no segmentation branch: no head and no segmentation pose encoder.
            input_dim=self.front_end.bio_tap_dim, hidden_dim=bio_hidden_dim,
            depth=bio_depth, nhead=bio_nhead, dropout=bio_dropout, num_classes=4,  # B/I/O + padding/UNK
            conv_stem_layers=bio_conv_stem_layers,      # local boundary inductive bias the UNet-less head lacks
            attention_radius_s=bio_attention_radius_s,  # banded attention: bounded look-ahead (dlm.yaml bio_attention_radius_s)
        ) if segmentation_branch else None
        # Load pretrained BEFORE building the DLM decoder: the substrate copies the current decoder/lm into its
        # vocab+1 [MASK] canvas, so the weights must already be in place.
        if pretrained_path:
            rep = self.front_end.load_pretrained(pretrained_path)
            print(f"slt | front-end warm-start {Path(pretrained_path).name}: {rep['pose_tensors']} pose + "
                  f"{rep['mt5_tensors']} LM tensors (missing {rep['pose_missing']}/{rep['mt5_missing']}, unexpected "
                  f"{rep['pose_unexpected']}/{rep['mt5_unexpected']})", flush=True)
        # Segmentation branch: its OWN pose encoder (same architecture, separate weights), so stage 2 can start from S1's encoder+head 
        # AND clean translator's encoder at once. Copied here from the warm start; train/slt.py overwrites it with S1's weights.
        self.bio_pose_encoder = copy.deepcopy(self.front_end.pose_encoder) if segmentation_branch else None
        if decoder == "dlm": self.dlm_decoder = self.front_end.make_dlm_decoder(block_size)
        elif decoder != "ar": raise ValueError(f"Unsupported decoder type: {decoder}")

    def segment(self, poses, frame_mask, timestamps_s=None):
        """Segmentation branch on the same frames the translator reads: its own pose encoder, then the BIO head.
        Returns the head output (logits for the duration decoder and the gate)."""
        if self.bio_pose_encoder is None: raise RuntimeError("the clean-floor model has no segmentation branch")
        if frame_mask is None: frame_mask = torch.ones(poses.shape[:2], dtype=torch.bool, device=poses.device)
        return self.bio_head(self.bio_pose_encoder(poses, frame_mask), timestamps_s=timestamps_s, frame_mask=frame_mask)

    def encode_memory(self, bio_tap, bio_mask, omega_bias=None):
        # Translator memory from its own pose features; Ω (when gated) biases LM encoder's keys and decoder's cross-attention.
        return self.front_end.encode_memory(bio_tap, bio_mask, omega_bias=omega_bias)

    def eval_encode_memory_fn(self, poses, frame_mask, timestamps_s, omega_bias=None):
        def encode(): # DMax's eval-mode rollout: same poses and Ω, with pose features and memory built as inference builds them
            with eval_mode(self):  # (dropout off, BatchNorm on its running statistics).
                bio_tap, bio_mask, _ = self.front_end.extract_bio_tap(poses, frame_mask, timestamps_s)
                return self.encode_memory(bio_tap, bio_mask, omega_bias=omega_bias)
        return encode

    @contextmanager
    def translator_frozen(self, frozen: bool = True):
        """Within the block, forwards record no gradient for the translation branch (pose encoder, LM, DLM decoder): a text loss
        built inside reaches only Ω and the segmentation branch. Flags are restored on exit; the graph keeps the exclusion."""
        params = [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith(("bio_pose_encoder.", "bio_head."))]
        if frozen:
            for p in params: p.requires_grad_(False)
        try: yield
        finally:
            if frozen:
                for p in params: p.requires_grad_(True)

    @torch.no_grad()
    def generate_from_bio_tap(
        self, bio_tap: torch.Tensor, frame_mask: torch.Tensor, max_text_tokens: int = 128, tau_dec: float = 0.5,
        spd_top_k: int = 1, spd_renormalize: bool = True, num_beams: int = 1, omega_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc_hidden, enc_mask = self.encode_memory(bio_tap, frame_mask, omega_bias=omega_bias)
        if self.decoder_type == "dlm":
            result = self.dlm_decoder.generate(
                enc_hidden, enc_mask, max_length=max_text_tokens, threshold=tau_dec, spd_top_k=spd_top_k,
                spd_renormalize=spd_renormalize, omega_bias=omega_bias,
            )
            # Sequential decoder passes of this decode (a batch's count: rows decode together). The AR arm's is
            # its generated length: 1 cached step per token plus the confidence pass. RQ1 reports the mean.
            self.last_decode_passes = int(result.forwards + result.cache_appends)
            # Slice to PRODUCED tokens (AR-arm parity): slot 0 (synthetic BOS) and everything past the first EOS are 1.0 pad bookkeeping, 
            # so an unsliced mean pins the reported confidence near 1 (~10 tokens on a 128 canvas → ≥ 0.92 regardless of quality).
            seq, conf = result.sequences[:, 1:], result.confidence[:, 1:]
            if seq.shape[0] == 1:  # every live caller decodes 1 window/buffer at a time
                hits = (seq[0] == int(self.dlm_decoder.eos_index)).nonzero(as_tuple=False)
                if hits.numel(): seq, conf = seq[:, : int(hits[0]) + 1], conf[:, : int(hits[0]) + 1]
            return seq, conf

        # AR arm: the front end owns generation (mBART lang-code start / mT5 prompt-conditioned) and returns REAL
        # per-token confidence. `num_beams>1` is the clean baseline's beam search; the SLT AR arm stays greedy.
        generated, confidence = self.front_end.ar_generate(
            enc_hidden, enc_mask, max_new_tokens=max_text_tokens, num_beams=num_beams, omega_bias=omega_bias,
        )
        self.last_decode_passes = int(generated.shape[1])  # start slot + N tokens = N cached steps + 1 confidence pass
        return generated, confidence

    @torch.no_grad()
    def generate_from_poses(
        self, poses: torch.Tensor, frame_mask: torch.Tensor, timestamps_s: torch.Tensor | None = None, *,
        gate_enabled: bool = False, gate_eps: float = 1e-4, gate_min_span_frames: int = 0, commit_mask: torch.Tensor | None = None, 
        gate_stream_start: bool = False, gate_anchor: torch.Tensor | None = None, **decode_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Poses → (bio_logits, tokens, confidence, gate_skip). Owns the BIO tap + membership gate; decode knobs 
        (max_text_tokens / tau_dec / spd_* / num_beams) pass via to `generate_from_bio_tap`, declared once there. 
        `gate_skip` (B, bool) marks windows the deployed FSM would never decode; all-False when gate is off."""
        bio_tap, mask, timestamps = self.front_end.extract_bio_tap(poses, frame_mask, timestamps_s)
        if self.bio_pose_encoder is None and not gate_enabled:
            # Clean floor: no segmentation branch and, gate off, no reader — zeros fill the contract slot.
            bio_logits = bio_tap.new_zeros(*bio_tap.shape[:2], 4)
        else:
            bio_output = self.segment(poses, mask, timestamps)
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
        tokens, confidence = self.generate_from_bio_tap(bio_tap, mask, omega_bias=omega_bias, **decode_kwargs)
        # DLM already strips its synthetic BOS in generate_from_bio_tap; the AR arm returns it raw. 
        # Strip here so eval's confidence mean covers only produced tokens, both arms.
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
        self, batch: dict, *, lambda_trans: float = 1.0, lambda_bio: float = 1.0,
        dice_weight: float = 1.5, bio_class_weights: torch.Tensor | None = None,
        oput_t_low: float = 0.3, oput_t_high: float = 0.8, oput_label_smoothing: float,
        gate_enabled: bool = False, gate_eps: float = 1e-4, gate_min_span_frames: int,
        gate_delta_frames: int = 12, gate_detach_omega: bool = False,
    ) -> SLTLossOutput:
        """Stage-2 training loss for one mixed-mode batch: ``L = lambda_bio * L_BIO + lambda_trans * L_translation``.

        The text supervision follows the state the deployed model is in, read from its OWN prediction (the gate's Viterbi first eligible 
        span and closed/open readout, as `MembershipGate.forward` computes at inference). GT decides only which text exists (premise P1: 
        a truncated visual input never receives a partial text label):
        - BIO (all rows): class-weighted CE plus binary signing Dice; padding/UNK ignored.
        - Complete GT target (window's 1st complete eligible unit, Mode 1/3). If the predicted 1st span COVERS the unit (it misses at
          most δ frames of the unit in total, start and end together, and holds at least Λ_min frames of it; it may run past the unit),
          the full-text loss (OPUT for DLM, CE for AR) trains the translator and, via Ω, the segmentation branch. If it does not cover
          it, the translator would read a crop that lacks part of the unit, so the same loss runs as a CRITIC: translation parameters
          receive no gradient from this loss; only Ω and segmentation branch receive its gradient. Coverage is a boundary-tolerance
          test, not proof that all semantic evidence receives high membership.
        - Other rows (truncated, headless, interior, gap): BIO only. A truncated window never receives text (P1).
        
        Gate off (clean floor, no-gate ablation): every complete target trains the translator on the whole window.
        `gate_detach_omega` (ablation): Ω carries no text gradient, so the segmentation branch learns from BIO only.
        """
        # Translator features and the segmentation branch read the same frames through separate pose encoders.
        bio_tap, bio_mask, timestamps = self.front_end.extract_bio_tap(batch["poses"], batch["frame_mask"], batch.get("timestamps_s"))
        bio_out = None
        if self.bio_pose_encoder is not None and (float(lambda_bio) != 0.0 or gate_enabled):
            bio_out = self.segment(batch["poses"], bio_mask, timestamps)
        bio_loss = bio_nll_dice_loss(
            bio_out.logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=bio_class_weights
        ) if bio_out is not None and float(lambda_bio) != 0.0 else bio_tap.new_zeros(())
        translation_loss = bio_tap.sum() * 0.0
        logs: dict[str, torch.Tensor] = {"bio_loss": bio_loss.detach()}

        # REALIZED mode mix (materialize() relabels windows jitter reshapes, so the drawn ratios are not what trains).
        realized = batch.get("mode_names")
        if isinstance(realized, list) and realized:
            for m in ("mode1", "mode2", "mode3", "mode4"):
                logs[f"mode_frac_{m}"] = bio_tap.new_tensor(sum(n == m for n in realized) / len(realized))
        target_tokens = batch.get("target_tokens")
        supervised = batch.get("translation_supervised")

        # The model's own state, exactly as inference reads it (docs/membership_gate.md): Ω from the predicted readout, plus the predicted 
        # 1st span. Λ_min and δ are frame counts on the native 24 fps grid (stage 2 has no fps augmentation), same integers the FSM uses.
        omega_bias, pred = None, None
        if bio_out is not None and gate_enabled:
            omega_bias, gate_stats = self.membership_gate(
                bio_out.logits, bio_mask, memory_len=self.front_end.prompt_length() + int(bio_tap.shape[1]),
                commit_mask=batch.get("commit_mask"), eps=gate_eps, min_span_frames=max(1, gate_min_span_frames), 
                timestamps_s=timestamps, decoder=DurationDecoder(self.duration_model),
            )
            if gate_detach_omega: omega_bias = omega_bias.detach()
            starts, terms, closed = (x.detach().cpu() for x in gate_stats["anchors"])
            pred = [(int(a), int(t)) if bool(c) else None for a, t, c in zip(starts, terms, closed)]
            logs["gate_closed_probability"] = gate_stats["closed_probability"]
            logs["gate_open_probability"] = gate_stats["open_probability"]
            logs["gate_uncertain_rows"] = bio_tap.new_tensor(float(gate_stats["uncertain_rows"]))

        if target_tokens is not None and supervised is not None and supervised.any():
            idx = supervised.to(device=bio_tap.device).nonzero(as_tuple=False).flatten()
            mode_names = batch.get("mode_names")
            idx_list = idx.detach().cpu().tolist()
            if isinstance(mode_names, list):
                invalid = sorted({mode_names[int(i)] for i in idx_list} - {"mode1", "mode3"})
                if invalid: 
                    raise ValueError(f"Translation supervision is allowed only for complete-conditioning Mode 1/3 windows; got {invalid}")

            covered, critic = idx_list, []
            if pred is not None:
                gt = self.gt_target_spans(batch, timestamps, bio_mask)
                lost = [i for i in idx_list if gt is None or gt[i] is None]
                if lost: 
                    raise ValueError(f"supervised rows {lost} carry no locatable target span (candidate_sentences/translation_targets)")

                def covers(i):
                    # The crop may miss at most δ frames of the unit in TOTAL (caption-boundary noise), and must hold at least Λ_min
                    # frames of it. A tolerance of δ at EACH end let a 36-frame unit be covered by its middle 12 frames, so the
                    # translator learned the full text from a third of the signs. Running past the unit is allowed (a merger).
                    p, g = pred[i], gt[i]
                    if p is None: return False
                    overlap = min(p[1], g[1]) - max(p[0], g[0])
                    return overlap >= max(int(gate_min_span_frames), g[1] - g[0] - int(gate_delta_frames), 1)
                
                covered = [i for i in idx_list if covers(i)]
                # A critic row trains only Ω. With Ω detached, or on a row whose Ω fell back to the whole window 
                # (no decoded span, or P < 1e-6: Ω = 0 with no gradient), nothing can learn from it, so it is not run.
                uncertain = gate_stats["uncertain"]
                critic = [] if gate_detach_omega else [i for i in idx_list if not covers(i) and not bool(uncertain[i])]
                logs["text_covered_rate"] = bio_tap.new_tensor(len(covered) / len(idx_list))

            labels = target_tokens["labels"].to(bio_tap.device)
            # Supervised tokens per group, as each loss counts them (AR: labels != -100; DLM: the OPUT canvas mask).
            n_tok = [0.0 if not grp else float((labels[grp] != -100).sum()) if self.decoder_type == "ar" else
                     float(self.dlm_decoder._prepare_x0(labels[grp])[1].sum()) for grp in (covered, critic)]
            n_all = max(1.0, sum(n_tok))
            group_losses, row_stats = [], {}

            for n_g, group, as_critic in ((n_tok[0], covered, False), (n_tok[1], critic, True)):
                if not group: continue
                g = torch.tensor(group, dtype=torch.long, device=bio_tap.device)
                om = None if omega_bias is None else omega_bias[g]
                # Each group is its own per-token mean (below), so through Ω a group's tokens would weigh 1/n_g each. Scaling only
                # the Ω gradient by n_g/N gives every supervised token the same 1/N pull on the segmentation branch; the forward
                # value and the translator's gradients do not change.
                if om is not None and om.requires_grad: om = om.detach() + (n_g / n_all) * (om - om.detach())
                with self.translator_frozen(as_critic):
                    # A critic row's translator input is a constant; only Ω carries its gradient.
                    tap = bio_tap[g].detach() if as_critic else bio_tap[g]
                    enc_hidden, enc_mask = self.encode_memory(tap, bio_mask[g], omega_bias=om)
                    if self.decoder_type == "ar":
                        loss_g, row_sum, row_valid = self.front_end.ar_loss(
                            enc_hidden, enc_mask, labels[g], label_smoothing=oput_label_smoothing, omega_bias=om, row_stats=True,
                        )
                    else:
                        out = self.dlm_decoder.oput_forward(
                            enc_hidden=enc_hidden, enc_mask=enc_mask, labels=labels[g], t_low=oput_t_low, t_high=oput_t_high,
                            label_smoothing=oput_label_smoothing, rollout_encode_fn=self.eval_encode_memory_fn(
                                batch["poses"][g], bio_mask[g], None if timestamps is None else timestamps[g], om,
                            ),
                            omega_bias=om,
                        )
                        loss_g, row_sum, row_valid = out["translation_loss"], out.get("row_loss_sum"), out.get("row_valid_count")

                # Each group is its own per-token mean: the translator's step must not shrink with the share of critic rows,
                # which would tie its effective learning rate to the segmentation quality.
                group_losses.append(loss_g)
                logs["text_critic_loss" if as_critic else "text_covered_loss"] = loss_g.detach()   # per-token CE of each group
                if row_sum is not None and row_valid is not None:
                    for k, i in enumerate(group): row_stats[i] = (row_sum[k], row_valid[k])

            if group_losses: translation_loss = torch.stack(group_losses).sum()
            if isinstance(mode_names, list) and row_stats:
                for mode in ("mode1", "mode3"):
                    rows = [row_stats[i] for i in idx_list if mode_names[int(i)] == mode and i in row_stats]
                    if rows: logs[f"oput_{mode}"] = torch.stack([r[0] for r in rows]).sum() / \
                                                    torch.stack([r[1] for r in rows]).sum().clamp(min=1)

        # `lambda_bio=0` = translation-only: the faithful Uni-Sign SLT recipe (1 label-smoothed CE) for the clean-floor arm.
        total = float(lambda_bio) * bio_loss + float(lambda_trans) * translation_loss
        logs["translation_loss"] = translation_loss.detach()
        logs["loss"] = total.detach()
        return SLTLossOutput(total, bio_loss, translation_loss, logs, bio_logits=bio_out.logits if bio_out is not None else None)
