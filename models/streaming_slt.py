"""Stage-2 composition: shared pose/text front end + an AR or DLM decoder, 
and the per-window-mode training loss (`MisalignedSLTModel.forward_loss`)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from train.helpers import eval_mode
from train.losses import bio_distill_loss, bio_nll_dice_loss, confidence_bound_gate, confidence_bound_loss
from models.bio_head import RoPEBIOHead
from models.front_end import SLTFrontEnd
from models.membership_gate import build_omega, omega_cross_bias
from infer.commit_gate import bio_complete_spans, open_span_start, select_target_span
from infer.duration_decode import deployed_decode_tags
from data.batch import SENTENCE_SEP, inside_anchor_target, tokenize_targets


def gate_skip_flags(
    bio_logits: torch.Tensor, frame_mask: torch.Tensor | None, min_span_frames: int = 0, 
    duration_prior=None, timestamps_s: torch.Tensor | None = None, commit_mask: torch.Tensor | None = None, 
    stream_start: bool = False, seam_is_terminator: bool = True, delta: int = 3,
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
        bio_logits, lengths, duration_prior, timestamps_s, commit_mask, stream_start=stream_start, 
        seam_is_terminator=seam_is_terminator, delta=int(delta),
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
    teacher_logits: torch.Tensor | None = None  # frozen S1 posteriors on the same batch (KD target), for dev diagnostics
    target_counts: torch.Tensor | None = None   # sentences in each row's translation target (0 = unsupervised), for the count readouts
    target_texts: list[str] | None = None       # the (possibly multi-sentence) target text per row, for dev references
    vetoed: torch.Tensor | None = None          # (B,) rows whose Omega was rebuilt from GT: their decode and target do not share a mask


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
        commit_mask: torch.Tensor | None = None, delta: int = 3, eps: float = 1e-4, min_span_frames: int = 0,
        gt_anchored: bool = False, timestamps_s: torch.Tensor | None = None, seam_is_terminator: bool = True,
        stream_start: bool = False, anchor_override: torch.Tensor | None = None, 
        iou_veto: float = 0.5, gt_spans: list | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Membership-gate cross-attention bias (B,1,1,M) for a batch (docs/membership_gate.md).

        The span (s, τ) is ON-POLICY — BIO head's OWN decoded span via `select_target_span`, the rule the FSM uses. Training
        (`bio_labels` given): when that span misses the GT target (tIoU < `iou_veto`, or no closed span where GT has one) Ω is
        teacher-forced to the GT span for that window — the IoU veto, rate logged as `veto_rate`; `anchor_hit_rate` measures the
        policy itself (closed on-policy span at tIoU ≥ 0.5, over windows with a GT target). Inference (`bio_labels=None`) and
        `anchor_override` rows: predicted span verbatim. `gt_anchored=True` forces the GT span with ±δ jitter (ablation).
        """
        B, T, _ = bio_logits.shape
        device = bio_logits.device
        lengths = frame_mask.to(device).long().sum(dim=1) if frame_mask is not None else torch.full((B,), T, device=device)
        # Without the re-split the gate goes off-policy once the FSM decodes with duration: a back-to-back window's
        # raw argmax is ONE merged run, training the decoder on neighbour-sentence features the deployed gate masks.
        duration_prior = getattr(self, "duration_prior", None)  # getattr: test fakes drive this method unbound
        pred_tags = deployed_decode_tags(
            bio_logits, lengths, duration_prior, timestamps_s, commit_mask,
            seam_is_terminator=seam_is_terminator, stream_start=stream_start, delta=int(delta),
        )
        starts, terms, has_term, has_anchor = [], [], [], []
        hits, n_gt = 0, 0        # anchor_hit_rate: closed on-policy span at tIoU >= 0.5, over windows with a closed GT target
        vetoed, n_gt_any = 0, 0  # veto_rate: windows whose Ω was rebuilt from GT, over windows with any GT target (closed or open)
        veto_flags = [False] * B
        for b in range(B):
            n = int(lengths[b].item())
            # χ-frontier filter mirroring the FSM (χ = in-buffer frame count): a span terminating before the commit
            # frontier is emitted content, never an anchor candidate.
            chi_b = int(commit_mask[b, :n].sum().item()) if commit_mask is not None else 0
            span = select_target_span(pred_tags[b, :n], min_span_frames, skip_term_before=chi_b)
            # anchor_override (B,2), start -1 = none: the caller already knows the span (the offline row translates the span
            # it emits), so Ω anchors on it instead of a re-decode; terminator -1 = the span runs to the window edge (open path).
            forced_open = None
            if anchor_override is not None and int(anchor_override[b, 0]) >= 0:
                ov_s, ov_t = int(anchor_override[b, 0]), int(anchor_override[b, 1])
                if ov_t >= 0: span = (ov_s, min(ov_t, n - 1))
                else: span, forced_open = None, ov_s
            if bio_labels is not None:
                # The GT anchor is the sentence the text supervises (batch frames) when the caller has it; the label-derived
                # first-complete span otherwise. fps augmentation can shrink a >= Lambda_min sentence below Lambda_min label
                # frames, and the two rules would then name different sentences.
                gt = gt_spans[b] if gt_spans is not None and gt_spans[b] is not None \
                                 else select_target_span(bio_labels[b, :n], min_span_frames)
                if gt is not None:
                    n_gt += 1; n_gt_any += 1
                    hits += int(span is not None and MisalignedSLTModel._span_iou(span, gt) >= 0.5)
                if gt_anchored:                 # ablation row: "GT-anchored with ±δ jitter" (gate-doc §3 table)
                    # Jitter is part of ablation: exact GT anchors would hand the gate boundary info the on-policy head can never supply, 
                    # conflating "teacher-forced m" with "oracle boundaries". δ-imprecision is the tolerance the gate's ramp/bands assume.
                    span = gt
                    if gt is not None: vetoed += 1; veto_flags[b] = True   # every GT-anchored window counts as rebuilt from GT
                    if gt is not None and n > 2 and delta > 0:
                        j_s = int(torch.randint(-int(delta), int(delta) + 1, (1,)).item())
                        j_t = int(torch.randint(-int(delta), int(delta) + 1, (1,)).item())
                        s_j = min(max(int(gt[0]) + j_s, 0), n - 2)
                        t_j = min(max(int(gt[1]) + j_t, s_j + 1), n - 1)
                        span = (s_j, t_j)
                elif anchor_override is None and gt is not None:
                    # A span that COVERS the target within δ is kept even at low tIoU (a merge: the decoder sees every sentence inside it
                    # and the target text follows). Otherwise the IoU veto teacher-forces Ω to the GT span.
                    covers = span is not None and int(span[0]) <= int(gt[0]) + int(delta) and int(span[1]) >= int(gt[1]) - int(delta)
                    # A kept non-covering span must not show ANOTHER whole sentence: the target would then pair sentence 1's text
                    # with a mask that holds sentence 2 whole. Such a span is a miss and is teacher-forced like any other.
                    shows_other = span is not None and not covers and any(
                        (a, t) != tuple(gt) and t - a >= int(min_span_frames) and a >= int(span[0]) - int(delta) 
                        and t <= int(span[1]) + int(delta) for a, t in bio_complete_spans(bio_labels[b, :n])
                    )
                    if not covers and (span is None or shows_other or MisalignedSLTModel._span_iou(span, gt) < float(iou_veto)):
                        span = gt; vetoed += 1; veto_flags[b] = True

            if span is not None: starts.append(int(span[0])); terms.append(int(span[1])); has_term.append(True); has_anchor.append(True)
            else:
                # An OPEN span running to buffer edge (Mode-2a right-truncation, buffer-cap forced commit) anchors Ω at ITS true start 
                # s (doc §2.8 forced path: γ≡γ_s, no right cliff → Ω≈0 for all-I interior); frame 0 would sweep the opening B and floor 
                # the span the gate must OPEN (attention ×0.01). A buffer-start I-run never opens → no anchor, neutral row.
                open_s = forced_open if forced_open is not None else open_span_start(pred_tags[b, :n])
                if bio_labels is not None and anchor_override is None and not gt_anchored:
                    # Single-endpoint veto for an open span: keep the predicted start when it is within δ of the GT start.
                    gt_open = open_span_start(bio_labels[b, :n])
                    if gt_open is not None: n_gt_any += 1
                    if open_s is not None and (gt_open is None or abs(int(open_s) - int(gt_open)) <= int(delta)): pass
                    elif gt_open is not None: open_s = gt_open; vetoed += 1; veto_flags[b] = True
                if gt_anchored and bio_labels is not None:   # ablation row: GT-anchored open spans too, jittered like the closed case
                    open_s = open_span_start(bio_labels[b, :n])
                    if open_s is not None and n > 2 and delta > 0:
                        open_s = min(max(int(open_s) + int(torch.randint(-int(delta), int(delta) + 1, (1,)).item()), 0), n - 2)
                if open_s is not None:
                    starts.append(int(open_s)); terms.append(-1); has_term.append(False); has_anchor.append(True)
                else: # No target span. Use a placeholder start for build_omega, then make this row neutral below.
                    starts.append(max(0, n - 1)); terms.append(-1); has_term.append(False); has_anchor.append(False)

        starts_t, terms_t = torch.tensor(starts, device=device), torch.tensor(terms, device=device)
        has_term_t = torch.tensor(has_term, device=device)
        out = build_omega(
            bio_logits, starts=starts_t, terminators=terms_t, commit_mask=commit_mask, lengths=lengths, 
            delta=delta, eps=eps, has_terminator=has_term_t,
        )
        anchor_mask = torch.tensor(has_anchor, device=device, dtype=torch.bool)
        omega = torch.where(anchor_mask.view(B, 1), out.omega, torch.zeros_like(out.omega))
        omega_bias = omega_cross_bias(omega, memory_len=int(memory_len), dtype=bio_logits.dtype)
        gamma_mean = out.gamma_s[anchor_mask].mean() if anchor_mask.any() else out.gamma_s.new_zeros(())
        stats = {
            "anchor_hit_rate": hits / max(1, n_gt), "veto_rate": vetoed / max(1, n_gt_any), "gamma_s_mean": float(gamma_mean),
            "anchors": (starts_t, terms_t, has_term_t), "vetoed": torch.tensor(veto_flags, device=device),
        }
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
        Generation and replay share one Ω, built from detached logits (conditioning only).
        """
        # eval_mode: the selection decode must be the distribution inference sees (dropout-free, BN stats untouched);
        # only the grad-bearing replay below trains under dropout.
        with torch.no_grad(), eval_mode(self):
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
        commit_mask: torch.Tensor | None = None, gate_stream_start: bool = False, gate_use_duration_prior: bool = True,
        gate_anchor: torch.Tensor | None = None, **decode_kwargs,
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
        # Inference gate, χ from the FSM commit log (single-window RQ1: none). Same Ω the decoder saw in
        # training (§1.3/§2.8); the AR arm injects it via HF cross-attn hooks (front_end.ar_generate).
        omega_bias = None
        gate_skip = torch.zeros(poses.shape[0], dtype=torch.bool)
        if gate_enabled:
            # gate_stream_start: this window is the 1st buffer of its stream, so a signing frame 0 IS a genuine onset. Without it a
            # window that opens mid-signing can never open a span (buffer-start I does not open), Ω falls to its no-span branch and
            # floors the whole window — on GT-span RQ1 windows, which begin exactly at sentence onset, that fires on a 3rd of them.
            # gate_use_duration_prior=False is a diagnostic switch only: every reported row keeps the prior, because training
            # (build_gate_omega in forward_loss) and the FSM re-split every window before anchoring Ω.
            prior_hold = getattr(self, "duration_prior", None)
            if not gate_use_duration_prior: self.duration_prior = None
            try:
                omega_bias, _ = self.build_gate_omega(
                    bio_logits, None, mask, memory_len=self.front_end.prompt_length() + int(bio_tap.shape[1]), 
                    commit_mask=commit_mask, delta=gate_delta, eps=gate_eps, min_span_frames=gate_min_span_frames, 
                    timestamps_s=timestamps, stream_start=gate_stream_start, anchor_override=gate_anchor,
                )
                gate_skip = gate_skip_flags(
                    bio_logits, mask, min_span_frames=gate_min_span_frames, duration_prior=getattr(self, "duration_prior", None),
                    timestamps_s=timestamps, commit_mask=commit_mask, stream_start=gate_stream_start, delta=int(gate_delta),
                )
            finally: self.duration_prior = prior_hold
            if gate_anchor is not None: gate_skip[(gate_anchor[:, 0] >= 0).cpu()] = False   # a known span is always decodable
        tokens, confidence = self.generate_from_bio_tap(bio_tap, mask, omega_bias=omega_bias, **decode_kwargs)
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
        """Per-row frame span (B, terminator) of the sentence the collated text supervises, from the window's candidate list; 
        None for rows without a text target. The gate's GT anchor, so the veto and the target name the same sentence."""
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


    def _inside_anchor_targets(self, batch: dict, gate_stats: dict, timestamps, bio_mask, delta: int, target_tokens: dict):
        """Translation target = every complete GT sentence inside the (post-veto) anchor widened by delta, joined in time order
        (data.batch SENTENCE_SEP): the decoder is trained on exactly what its mask shows, and the sentence count of its output is
        the merge readout at inference. Containment is tested in frames with the veto's own tolerance. A joined target that would
        not fit the token canvas drops its LAST sentences whole (never a cut sentence: P1) and is logged as target_trunc_rate.
        Rows without a closed anchor, or whose anchor does not show the first-complete sentence whole, keep the single target.
        Returns (target tokens, per-row sentence count; 0 = unsupervised, target texts)."""
        starts, terms, has_term = gate_stats["anchors"]
        supervised, cands, firsts = batch["translation_supervised"], batch["candidate_sentences"], batch["translation_targets"]
        settings = batch.get("target_tokenization")

        if settings is None or self.tokenizer is None or timestamps is None: return target_tokens, None, None
        cap = int(settings.get("max_text_tokens", 128))
        n_tok = lambda text: len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        sep_tok = n_tok(SENTENCE_SEP.strip())
        texts, counts, changed, truncated, multi = [], [], False, 0, 0

        for b in range(len(cands)):
            first = firsts[b]
            first_text = (first["text"] if isinstance(first, dict) else getattr(first, "text", None)) if first is not None else None
            texts.append(first_text or ""); counts.append(1 if first_text else 0)
            if not bool(supervised[b]) or not first_text or not bool(has_term[b]): continue
            n = int(bio_mask[b].long().sum())
            picks = inside_anchor_target(self._candidate_frames(cands[b], timestamps[b], n), first_text,
                                         int(starts[b]) - int(delta), int(terms[b]) + int(delta), lo_key="b_idx", hi_key="t_idx")
            if picks is None: continue
            if len(picks) >= 2:
                multi += 1
                lens = [n_tok(t) for t in picks]
                while len(picks) > 1 and sum(lens) + sep_tok * (len(picks) - 1) + 1 > cap:   # +1: the EOS the canvas appends
                    picks.pop(); lens.pop(); truncated += 1
            texts[-1], counts[-1] = SENTENCE_SEP.join(picks), len(picks); changed = changed or counts[-1] != 1

        self._last_target_trunc = (truncated, multi)
        counts_t = torch.tensor(counts, dtype=torch.long)
        if not changed: return target_tokens, counts_t, texts
        toks = tokenize_targets(self.tokenizer, texts, **settings)
        toks["labels"][~supervised.to(toks["labels"].device).bool()] = -100
        dev = target_tokens["labels"].device
        return {k: v.to(dev) for k, v in toks.items()}, counts_t, texts


    def forward_loss(
        self, batch: dict, lambda_trans: float = 1.0, lambda_bio: float = 1.0,
        dice_weight: float = 1.5, bio_class_weights: torch.Tensor | None = None,
        oput_t_low: float = 0.3, oput_t_high: float = 0.8, oput_sample_rollout: bool = False,
        oput_label_smoothing: float = 0.0, oput_rollout_eval_mode: bool = True, oput_eos_supervision: int | None = None,
        cb_enabled: bool = True, cb_active: bool = True, cb_tau: float = 0.75, cb_lambda: float = 0.3, 
        cb_verified_gate: bool = True, cb_decode_steps: int = 64, cb_belief_gap: bool = True,
        cb_dcd_window_length: int | None = None, cb_dcd_max_window_length: int | None = None, cb_dcd_window_type: str = "sliding",
        cb_dcd_decode_algo: str = "threshold", cb_dcd_decode_param: int | float | None = None, cb_dcd_sample_top_k: int | None = None,
        cb_dcd_top_p: float | None = None, cb_dcd_cache_type: str = "none", cb_spd_top_k: int = 1, cb_spd_renormalize: bool = True, 
        cb_spd_revision: bool = True, cb_temperature: float = 0.0, gate_enabled: bool = False, gate_delta: int = 3, 
        gate_eps: float = 1e-4, gate_min_span_frames: int = 0, gate_gt_anchored: bool = False, gate_iou_veto: float = 0.5,
        bio_distill_alpha: float = 0.2, bio_distill_temperature: float = 2.0,
    ) -> SLTLossOutput:
        """Stage-2 training loss for one mixed-mode batch.

        ``L = lambda_bio * L_BIO + lambda_trans * L_translation``, translation routed per window mode (`mode_names` from sampler) to 
        enforce premise P1: a truncated visual input never receives a partial text label.

        - BIO (all modes): alpha*(CE+Dice on GT) + (1-alpha)*T^2*KL(S1 || head) with an S1 init (bio_distill), else Dice(1.5)+CE;
          over in-window frames, padding/UNK ignored.
        - Mode 1 / Mode 3 (complete-anchor / first-complete-span): OPUT under fixed full conditioning (`dlm_decoder.oput_forward`; 
          plain CE for AR). Raises if any other mode reaches that path.
        - Mode 2a (right-truncated): confidence-bound term only — gated CE toward the model's own no-grad full-evidence decode, 
          at slots where that decode is reference-verified and the truncated decode confidently disagrees. Off during OPUT warmup 
          (`cb_active=False`), weighted by `cb_lambda`; see `dlm_decoder.remasked_logits` for why its gradient uses 1 re-masked 
          forward, not back-prop through the decode.
        - Mode 2b / 2c / Mode 4 (left/both-truncated, all-gap): BIO only. The FSM does not call the decoder.

        Per-mode losses logged separately.
        """
        bio_tap, bio_mask, enc_hidden, enc_mask, timestamps = self.encode_visual(batch)
        teacher_logits = None
        if float(lambda_bio) == 0.0 and not gate_enabled: bio_out, bio_loss = None, bio_tap.new_zeros(()) # Clean-floor recipe
        else:
            bio_out = self.bio_head(bio_tap, timestamps_s=timestamps, frame_mask=bio_mask)
            teacher = getattr(self, "bio_teacher", None)   # frozen S1 (train/slt.py attaches it, non-registered)
            if teacher is not None and float(lambda_bio) != 0.0:
                with torch.no_grad():
                    teacher_logits = teacher.to(bio_tap.device)(batch["poses"], batch["frame_mask"], timestamps_s=timestamps).logits
                # Standard KD: alpha * GT term + (1 - alpha) * T^2 KL to the teacher. The GT term keeps a small, bounded path for
                # boundaries S1 missed; the KL anchors calibration so the monolingual refit cannot drift.
                kd = bio_distill_loss(bio_out.logits, teacher_logits, bio_mask, temperature=bio_distill_temperature)
                a = float(bio_distill_alpha)
                gt_term = bio_nll_dice_loss(
                    bio_out.logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=bio_class_weights
                ) if a > 0.0 else kd.new_zeros(())
                bio_loss = a * gt_term + (1.0 - a) * kd
            else: bio_loss = bio_nll_dice_loss(bio_out.logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=bio_class_weights)

        translation_loss = bio_tap.sum() * 0.0
        logs: dict[str, torch.Tensor] = {"bio_loss": bio_loss.detach()}
        if teacher_logits is not None: logs["bio_kd_loss"] = kd.detach(); logs["bio_gt_loss"] = gt_term.detach()
        # REALIZED mode mix (materialize() relabels windows jitter reshapes, so the drawn ratios are not what trains).
        realized = batch.get("mode_names")
        if isinstance(realized, list) and realized:
            for m in ("mode1", "mode2", "mode3", "mode4"):
                logs[f"mode_frac_{m}"] = bio_tap.new_tensor(sum(n == m for n in realized) / len(realized))
        target_tokens = batch.get("target_tokens")
        supervised = batch.get("translation_supervised")
        target_counts, target_texts = None, None

        # Membership gate (docs/membership_gate.md): Ω from BIO posteriors + on-policy span conditions the decoder on the segmentation
        # belief. Built from DETACHED logits: conditioning only, no translation gradient into the head (the head is trained by its own
        # objective). BOTH arms inject the same Ω (DLM manual decode, AR via front_end.ar_omega_context). None → pre-gate.
        omega_bias = None
        if gate_enabled:
            omega_bias, gate_stats = self.build_gate_omega(
                bio_out.logits.detach(), batch["bio_labels"], batch.get("frame_mask"), memory_len=int(enc_hidden.shape[1]),
                commit_mask=batch.get("commit_mask"), delta=gate_delta, eps=gate_eps, min_span_frames=gate_min_span_frames,
                gt_anchored=gate_gt_anchored, timestamps_s=timestamps, iou_veto=gate_iou_veto,
                gt_spans=self.gt_target_spans(batch, timestamps, bio_mask),
            )
            logs["gate_anchor_hit_rate"] = bio_tap.new_tensor(gate_stats["anchor_hit_rate"])
            logs["gate_veto_rate"] = bio_tap.new_tensor(gate_stats["veto_rate"])
            logs["gate_gamma_s_mean"] = bio_tap.new_tensor(gate_stats["gamma_s_mean"])
            
            if target_tokens is not None and supervised is not None and batch.get("candidate_sentences") is not None:
                target_tokens, target_counts, target_texts = self._inside_anchor_targets(
                    batch, gate_stats, timestamps, bio_mask, gate_delta, target_tokens
                )
                if target_counts is not None:
                    sup = target_counts[supervised.to(target_counts.device)]
                    if sup.numel(): logs["target_multi_rate"] = bio_tap.new_tensor(float((sup >= 2).float().mean()))
                    trunc, multi = getattr(self, "_last_target_trunc", (0, 0))
                    if multi: logs["target_trunc_rate"] = bio_tap.new_tensor(float(trunc) / float(multi))

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
                    enc_hidden[idx], enc_mask[idx], labels[idx],
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
                    rollout_encode_fn=(
                        self.front_end.eval_encode_memory_fn(bio_tap[idx], bio_mask[idx]) if oput_rollout_eval_mode else None
                    ),
                    omega_bias=None if omega_bias is None else omega_bias[idx],
                )
                translation_loss = dlm_out["translation_loss"]
                self._log_per_mode(
                    logs, mode_to_indices, idx_list if isinstance(mode_names, list) else [],
                    dlm_out.get("row_loss_sum"), dlm_out.get("row_valid_count")
                )

        if (cb_enabled and cb_active and batch.get("full_evidence") is not None and batch.get("full_evidence_indices") is not None 
            and batch["full_evidence_indices"].numel() > 0 and batch.get("reference_tokens") is not None):
            cb_indices = batch["full_evidence_indices"].to(bio_tap.device)
            full_batch = batch["full_evidence"]
            # no_grad: the full-evidence view feeds only the self-target decode and its Ω, both no-grad. Without this
            # the pose encoder built a graph over the LONGER full-evidence window that nothing ever backwards through.
            # eval_mode: teacher-side forward — dropout off, and BN running stats must not absorb the longer
            # full-evidence windows (BN updates in forward even under no_grad).
            with torch.no_grad(), eval_mode(self):
                full_bio_tap, full_mask, full_timestamps = self.front_end.extract_bio_tap(
                    full_batch["poses"], full_batch["frame_mask"], full_batch.get("timestamps_s"),
                )
            ref_ids = batch["reference_tokens"]["input_ids"].to(bio_tap.device)[cb_indices]
            ref_mask = batch["reference_tokens"]["attention_mask"].to(bio_tap.device)[cb_indices].bool()
            max_len = ref_ids.shape[1]

            # Gate for the Mode-2a CB decodes, both arms (§9.3 symmetry): the truncated decode uses Ω from its own
            # posteriors, the full-evidence self-target Ω from the full view's — both under the main call's anchoring
            # policy, so the gt_anchored ablation compares two GT-anchored decodes, not GT-anchored vs on-policy.
            cb_omega_trunc = cb_omega_full = None
            if gate_enabled:
                prompt_len = self.front_end.prompt_length()
                chi = batch.get("commit_mask")
                cb_omega_trunc, _ = self.build_gate_omega(
                    bio_out.logits.detach()[cb_indices],
                    batch.get("bio_labels")[cb_indices] if batch.get("bio_labels") is not None else None,
                    bio_mask[cb_indices], memory_len=prompt_len + int(bio_tap.shape[1]),
                    commit_mask=chi[cb_indices] if chi is not None else None,
                    delta=gate_delta, eps=gate_eps, min_span_frames=gate_min_span_frames,
                    gt_anchored=gate_gt_anchored, iou_veto=gate_iou_veto,  # must match the main call: else the gt_anchored ablation
                    # trains its CB decode under a differently-anchored Omega than the OPUT rows.
                    timestamps_s=timestamps[cb_indices],  # real fps for the duration re-split (else 24fps fallback)
                )
                # Real timestamps + mask, else RoPE assumes 50fps indices on full-evidence view — 
                # a time-scale mismatch vs trunc view's real seconds.
                with torch.no_grad(), eval_mode(self):
                    full_cb_bio_logits = self.bio_head(full_bio_tap, timestamps_s=full_timestamps, frame_mask=full_mask).logits
                cb_omega_full, _ = self.build_gate_omega(
                    full_cb_bio_logits, full_batch.get("bio_labels"), full_mask, memory_len=prompt_len + int(full_bio_tap.shape[1]),
                    # χ on BOTH views: a committed predecessor tail at the left edge must be
                    # floored here too, else the views differ by more than the right-truncation.
                    commit_mask=full_batch.get("commit_mask"),
                    delta=gate_delta, eps=gate_eps, min_span_frames=gate_min_span_frames,
                    gt_anchored=gate_gt_anchored, timestamps_s=full_timestamps, iou_veto=gate_iou_veto,
                )

            if self.decoder_type == "dlm": # Encode the trunc path ONCE: the no-grad decode won't track it, remasked_logits will.
                trunc_enc_hidden, trunc_enc_mask = self.front_end.encode_memory(bio_tap[cb_indices], bio_mask[cb_indices])
                # eval_mode: both no-grad decodes are teachers (dlm.yaml oput: the rollout is drawn "the distribution inference sees");
                # the grad path (trunc encode above, remasked_logits below) stays in train mode so dropout regularizes only what trains.
                with torch.no_grad(), eval_mode(self):
                    full_enc_hidden, full_enc_mask = self.front_end.encode_memory(full_bio_tap, full_mask)
                    full_decode = self.dlm_decoder.generate_spd_dcd(
                        enc_hidden=full_enc_hidden, enc_mask=full_enc_mask, max_length=max_len,
                        diffusion_steps=cb_decode_steps, tau_dec=cb_tau, top_k=cb_spd_top_k,
                        spd_renormalize=cb_spd_renormalize, spd_revision=cb_spd_revision, temperature=cb_temperature,
                        window_length=cb_dcd_window_length, max_window_length=cb_dcd_max_window_length, window_type=cb_dcd_window_type,
                        decode_algo=cb_dcd_decode_algo, decode_param=cb_dcd_decode_param, sample_top_k=cb_dcd_sample_top_k,
                        top_p=cb_dcd_top_p, cache_type=cb_dcd_cache_type, omega_bias=cb_omega_full,
                    )
                    full_tokens, full_conf = full_decode.sequences, full_decode.confidence

                    # Decode ONLY to pick which slots to defer-counterfactual (where the truncated decode disagrees
                    # with the full-evidence one) — its confidence does NOT gate the loss (see below).
                    trunc_decode = self.dlm_decoder.decode_spd_dcd(
                        enc_hidden=trunc_enc_hidden, enc_mask=trunc_enc_mask, max_length=max_len,
                        diffusion_steps=cb_decode_steps, tau_dec=cb_tau, top_k=cb_spd_top_k,
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
                # counterfactual. Confidence threshold isn't applied here; it's applied by confidence_bound_loss on REMASKED logits, 
                # so the gate reads SAME distribution the CE trains. Gating on decode confidence instead (deployed commit path) leaves 
                # the loss free to push deferral forward to confidently emit unseen continuation — the hallucination AR arm's shared-logits 
                # fixed point rules out. Constant confidence here just reuses gate's disagreement/verified/pad conditions to pick candidates.
                candidate = confidence_bound_gate(
                    full_tokens=full_tokens, trunc_tokens=trunc_decoded, 
                    trunc_confidence=torch.ones_like(trunc_decode.confidence), tau_cb=0.0, reference_tokens=cb_ref_ids, 
                    valid_mask=cb_ref_mask, verified_full_evidence_gate=cb_verified_gate, pad_token_id=self.tokenizer.pad_token_id,
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
                with torch.no_grad(), eval_mode(self):
                    full_tokens, full_conf = self.generate_from_bio_tap(
                        full_bio_tap, full_mask, max_text_tokens=max(1, max_len - 1), omega_bias=cb_omega_full
                    )
                    if full_conf is not None:
                        if full_conf.shape[1] < max_len: full_conf = torch.cat(
                            [full_conf, full_conf.new_zeros((full_conf.shape[0], max_len - full_conf.shape[1]))], dim=1
                        )
                        else: full_conf = full_conf[:, :max_len]
                    full_tokens = self._pad_or_trim_tokens(full_tokens, max_len)

                trunc_logits, _ = self._ar_confidence_bound_logits(
                    bio_tap[cb_indices], bio_mask[cb_indices], max_len=max_len, omega_bias=cb_omega_trunc,
                )
                # AR layout [lang, tok1, ..., eos] (start = language code, matching mBART's training shift), so dropping
                # it lines full_tokens[:, 1:] slot j up with reference slot j ([tok1, ..., eos, lang]); the loss's
                # min-length slicing trims the dangling lang code. DON'T slice the reference too — shifts the gate by 1.
                full_tokens = full_tokens[:, 1:]
                if full_conf is not None: full_conf = full_conf[:, 1:]
                trunc_tokens, trunc_confidence = None, None
                cb_ref_ids, cb_ref_mask, cb_active_mask = ref_ids, ref_mask, None

            # Per-mode routing: L = L_OPUT + λ_cb·L_cb, each term under its OWN normalization (OPUT per supervised token; 
            # CB per valid Mode-2a slot, inside confidence_bound_loss). A token-pooled fold would put every Mode-2a reference 
            # token into a shared denominator, most of them structurally zero-loss, and shrink the clean-span OPUT gradient on 
            # nearly every batch (a clean-translation drift). A flat add cannot let 2a dominate: L_cb is a small fraction of 
            # L_OPUT at λ_cb=1. Zero-active batches reduce exactly to L_OPUT in both arms.
            if cb_ref_mask is not None:
                cb_loss_val, cb_active_count = translation_loss.new_zeros(()), translation_loss.new_zeros(())
                if trunc_logits is not None:
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
        return SLTLossOutput(total, bio_loss, translation_loss, logs, bio_logits=bio_out.logits if bio_out is not None else None,
                             teacher_logits=teacher_logits, target_counts=target_counts, target_texts=target_texts,
                             vetoed=gate_stats.get("vetoed") if gate_enabled else None)
