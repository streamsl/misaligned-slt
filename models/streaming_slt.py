"""Stage-2 composition: a segmentation branch (own pose encoder, or the translator's in the one-stream arm, + BIO head), the
translator front end, the Ω crop of the selected span between them, an AR or DLM decoder, and the stage-2 training loss
(`MisalignedSLTModel.forward_loss`)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import copy, math

import torch
import torch.nn as nn
from data.windowing import BIO

from models.bio_head import RoPEBIOHead
from models.front_end import SLTFrontEnd
from models.membership_gate import MembershipGate, membership_bias, omega_cross_bias, span_membership

from train.helpers import eval_mode
from train.losses import bio_nll_dice_loss
from infer.duration_decode import DurationDecoder
from utils import UNIT_SEPARATOR

# Stamped by train-slt and required by eval: an arm trained under another routing, crop, read or end term must not load under this Ω.
TEXT_ROUTING = "gt_anchored_span_crop_last_unit_end"


def support_crop(front_end, poses, timestamps_s, bio_tap, frame_mask, omega_bias, eps):
    """The read of every gated decode and text loss: the translator reads Ω's support as a crop. Returns (tap, mask, Ω).

    Support = frames with v > ε, v = (exp(Ω) - ε)/(1 - ε), read from pose columns of Ω, detached. Committed frames and padding are hidden, 
    so they're never in it. [a, b] = 1st and last support frame of the row. Translator pose encoder encodes poses of a..b again as 1 crop, 
    with 0 padding at both crop edges as on a clean-translator crop. Poses are window's, already normalized, so body box stays window's. 
    The tap of row becomes [crop features of a..b, its own features of b+1..T-1]. Columns before a leave the tap, the mask and Ω (prompt 
    columns stay), so the support starts right after the prompt, as a crop does. Frames after b keep their features and their Ω: they stay 
    hidden near ε, and in training an end-side straight-through term on them keeps its gradient. Each row is left-aligned and padded on 
    the right (mask 0, Ω hidden), so the memory is as long as longest row. No Ω (an ungated read) or no support: input is unchanged."""
    if omega_bias is None: return bio_tap, frame_mask, omega_bias
    prompt, t = front_end.prompt_length(), int(bio_tap.shape[1])
    pose_omega = omega_bias[:, 0, 0, prompt:]
    v = (pose_omega.detach().float().exp() - eps) / (1.0 - eps)
    support = (v > eps) & frame_mask.bool()
    if not bool(support.any()): return bio_tap, frame_mask, omega_bias

    device = bio_tap.device
    frames = torch.arange(t, device=device)
    # min/max(1).values, not amin/amax: on MPS (torch 2.13) an int64 amin/amax over a row starts from 0, so it returned a = 0 and no cut.
    a = torch.where(support, frames, t).min(1).values.masked_fill(~support.any(1), 0)
    b = torch.where(support, frames, -1).max(1).values   # -1 on a row without support: an empty crop, so that row keeps its input
    length = b - a + 1
    cols = torch.arange(t - int(a.cpu().min()), device=device)
    src = a[:, None] + cols                        # the window frame that each new column reads
    mask = (src < t) & frame_mask.gather(1, src.clamp(max=t - 1))
    src, rows, n = src.clamp(max=t - 1), torch.arange(len(bio_tap), device=device)[:, None], int(length.cpu().max())
    in_crop = cols < length[:, None]

    # The crop batch repeats each crop's last frame as its padding, as the collator does (data.batch.repeat_last_frame).
    crop_src = torch.minimum(src[:, :n], torch.maximum(a, b)[:, None])
    # The crop pass normalizes with BatchNorm's running statistics in training too, so training reads 
    # the crop features that every inference read gets. The window pass still updates the statistics.
    modules = front_end.modules() if isinstance(front_end, torch.nn.Module) else ()
    norms = [m for m in modules if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training]
    for m in norms: m.eval()
    try: crop = front_end.extract_bio_tap(
        poses[rows, crop_src], in_crop[:, :n], None if timestamps_s is None else timestamps_s[rows, crop_src]
    )[0]
    finally:
        for m in norms: m.train()
    crop = torch.cat((crop, crop.new_zeros(crop.shape[0], len(cols) - n, *crop.shape[2:])), 1)
    tap = torch.where(in_crop.view(*in_crop.shape, *[1] * (crop.dim() - 2)), crop, bio_tap[rows, src])
    pose_omega = pose_omega[rows, src].masked_fill(~mask, math.log(eps))
    return tap, mask, torch.cat((omega_bias[:, 0, 0, :prompt], pose_omega), 1)[:, None, None]


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
    (arXiv 2211.01367 §4). There the 2 streams share 1 task and read different inputs, here they read 1 input for 2 tasks:
    - segmentation: `bio_pose_encoder` (S1's weights) -> `bio_head` (RoPEBIOHead) -> phrase B/I/O logits. It reads per-frame features 
      directly, so the buffer's variable length never hits the seq2seq encoder's positions.
    - translation: `front_end.pose_encoder` (the clean translator's weights) -> task prompt + LM encoder -> decoder. Ω (hard crop of 1st 
      span the BIO logits select) biases the LM encoder's self-attention keys and the decoder's cross-attention. Every gated read encodes 
      Ω's support again as a crop and removes columns before it (`support_crop`), so at a hard membership the translator reads cropped 
      span up to the ε leak. Ω is the only path from segmentation into the translator: no feature path carries whole-window context past 
      the crop. Other couplings run other way: `bio_adapter` (translator features into the BIO head, stop-gradient) and, on pause rows, 
      the end-side text term (`_end_survival`).
    
    At step 0 the branches hold S1's and the clean translator's weights. The gate-off AR arm decodes as the AR clean floor token for
    token, and the gate-off DLM arm as the DLM clean floor (its canvas loads from that floor's file).

    `shared_pose_encoder` (one-stream arm): no `bio_pose_encoder`; ONE pose encoder, the clean translator's, which the BIO head and the
    translator both read and both losses train. train/slt.py folds a least-squares map from its tokens to S1's encoder tokens into the
    head (`fold_s1_stitch`), so step 0 is S1's head behind that map and exactly the clean translator. Its no-text control is the
    two-stream arm's, `membership_gate.detach_omega`: there the text reads that encoder under stop-gradient, so BIO alone trains it.

    `decoder="dlm"` → block-diffusion decoder (OPUT training / block decode with SPD), `"ar"` → AR seq2seq. Nothing else differs — 
    front end, BIO head, sampler, FSM, commit gate identical — which is what makes AR-vs-DLM a clean test.
    """
    # The block (a DLM) and the head shape (a segmentation branch) come from the config (dlm.yaml block_size, bio:), never from a
    # Python value: a build that needs one and omits it fails. A clean AR floor needs none.
    def __init__(
        self, front_end: SLTFrontEnd, *, decoder: str, block_size: int | None = None, bio_hidden_dim: int | None = None,
        bio_depth: int | None = None, bio_nhead: int | None = None, bio_dropout: float | None = None,
        bio_conv_stem_layers: int | None = None, bio_attention_radius_s: float | None = None, 
        pretrained_path: str | None = None, segmentation_branch: bool = True, bio_adapter: bool = False,
        shared_pose_encoder: bool = False,
    ):
        super().__init__()
        # Like bio_adapter, the switch means nothing without a segmentation branch (the clean floor inherits dlm.yaml's value).
        shared_pose_encoder = bool(shared_pose_encoder) and segmentation_branch
        if shared_pose_encoder and bio_adapter: raise ValueError(
            "shared_pose_encoder needs bio_adapter: false: with 1 pose encoder the adapter would add the head's own features to themselves."
        )
        # Caller passes UniSignMT5FrontEnd (models/unisign.py).
        self.front_end = front_end
        self.tokenizer = self.front_end.tokenizer
        self.decoder_type = decoder
        self.duration_model = None
        self.membership_gate = MembershipGate()
        self.bio_head = RoPEBIOHead( # The clean floor has no segmentation branch: no head and no segmentation pose encoder.
            input_dim=self.front_end.bio_tap_dim, hidden_dim=bio_hidden_dim,
            depth=bio_depth, nhead=bio_nhead, dropout=bio_dropout, num_classes=4,  # B/I/O + padding/UNK
            conv_stem_layers=bio_conv_stem_layers,      # local boundary inductive bias the UNet-less head lacks
            attention_radius_s=bio_attention_radius_s,  # banded attention: bounded look-ahead (S1's derived band, its stamp)
        ) if segmentation_branch else None
        # Load pretrained BEFORE building the DLM decoder: the substrate copies the current decoder/lm into its
        # vocab+1 [MASK] canvas, so the weights must already be in place (a DLM floor's own canvas then loads on top, below).
        if pretrained_path:
            rep = self.front_end.load_pretrained(pretrained_path)
            print(f"slt | front-end warm-start {Path(pretrained_path).name}: {rep['pose_tensors']} pose + "
                  f"{rep['mt5_tensors']} LM tensors (missing {rep['pose_missing']}/{rep['mt5_missing']}, unexpected "
                  f"{rep['pose_unexpected']}/{rep['mt5_unexpected']})", flush=True)
        # Segmentation branch: its OWN pose encoder (same architecture, separate weights), so stage 2 can start from S1's encoder+head 
        # AND clean translator's encoder at once. Copied here from the warm start; train/slt.py overwrites it with S1's weights.
        self.shared_pose_encoder = shared_pose_encoder
        self.bio_pose_encoder = copy.deepcopy(self.front_end.pose_encoder) if segmentation_branch and not shared_pose_encoder else None
        # Translator -> segmentation features (dlm.yaml bio.adapter): W LN(sg(v)) into the head's hidden space, added after the head's 
        # input norm (RoPEBIOHead residual), so it weighs against the normalized input whatever S1's token size; zero-initialized, so 
        # the head reads S1's features at step 0.
        self.bio_adapter = None
        if bio_adapter and segmentation_branch:
            dim = self.front_end.bio_tap_dim
            self.bio_adapter = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, int(bio_hidden_dim)))
            nn.init.zeros_(self.bio_adapter[1].weight); nn.init.zeros_(self.bio_adapter[1].bias)
        if decoder == "dlm":
            self.dlm_decoder = self.front_end.make_dlm_decoder(block_size)
            # A DLM clean floor's file also holds its own canvas (the [MASK] row, the input table and the vocab+1 output head), which
            # the front-end load does not read: it replaces the copy of the loaded mT5. A file without one (the released weights,
            # an AR floor) keeps the copy. mmap: only these tensors are read. A partial canvas fails the strict load.
            if pretrained_path:
                blob = torch.load(str(pretrained_path), map_location="cpu", weights_only=True, mmap=True)
                sd = blob.get("model", blob) if isinstance(blob, dict) else {}
                canvas = {k[len("dlm_decoder."):]: v for k, v in sd.items()
                          if k.startswith(("dlm_decoder.embed_tokens.", "dlm_decoder.lm_head."))}
                if canvas: nn.ModuleDict({"embed_tokens": self.dlm_decoder.embed_tokens, "lm_head": self.dlm_decoder.lm_head}
                                         ).load_state_dict(canvas, strict=True)
        elif decoder != "ar": raise ValueError(f"Unsupported decoder type: {decoder}")

    def segment(self, poses, frame_mask, timestamps_s=None, translator_tap=None):
        """Segmentation branch on the same frames the translator reads: its own pose encoder, then BIO head. With `bio_adapter`, the head 
        also reads translator's per-frame features, stop-gradient, as inference builds them: in eval mode the caller's `extract_bio_tap`
        output (`translator_tap`); in training an eval-mode pass (BatchNorm on its running statistics, no dropout), because the
        train-mode window pass normalizes with batch statistics that inference never has. 
        With `shared_pose_encoder`, the head reads those features themselves, WITH gradient (the one encoder is the translator's: BIO
        trains it, and so does text unless detach_omega, `forward_loss`). Returns the head output."""
        if self.bio_head is None: raise RuntimeError("the pretrimmed translator has no segmentation branch")
        if frame_mask is None: frame_mask = torch.ones(poses.shape[:2], dtype=torch.bool, device=poses.device)
        if self.shared_pose_encoder:   # one encoder (the translator's): BIO trains it, and text unless detach_omega (forward_loss)
            features = translator_tap if translator_tap is not None else self.front_end.extract_bio_tap(poses, frame_mask)[0]
            return self.bio_head(features, timestamps_s=timestamps_s, frame_mask=frame_mask)
        features, residual = self.bio_pose_encoder(poses, frame_mask), None
        if self.bio_adapter is not None:
            if translator_tap is None or self.training:
                with torch.no_grad(), eval_mode(self.front_end): translator_tap = self.front_end.extract_bio_tap(poses, frame_mask)[0]
            residual = self.bio_adapter(translator_tap.detach())
        return self.bio_head(features, timestamps_s=timestamps_s, frame_mask=frame_mask, residual=residual)

    def encode_memory(self, bio_tap, bio_mask, omega_bias=None):
        # Translator memory from its own pose features; Ω (when gated) biases LM encoder's keys and decoder's cross-attention.
        return self.front_end.encode_memory(bio_tap, bio_mask, omega_bias=omega_bias)

    def eval_encode_memory_fn(self, poses, frame_mask, timestamps_s, omega_bias, eps):
        def encode(): # DMax's eval-mode rollout: same poses and Ω, with pose features, support crop and memory built as inference
            with eval_mode(self):  # builds them (dropout off, BatchNorm on its running statistics).
                bio_tap, bio_mask, _ = self.front_end.extract_bio_tap(poses, frame_mask, timestamps_s)
                return self.encode_memory(*support_crop(self.front_end, poses, timestamps_s, bio_tap, bio_mask, omega_bias, eps))
        return encode

    @torch.no_grad()
    def generate_from_bio_tap(
        self, bio_tap: torch.Tensor, frame_mask: torch.Tensor, *, max_text_tokens: int, tau_dec: float,
        spd_top_k: int, spd_renormalize: bool, num_beams: int = 1, omega_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc_hidden, enc_mask = self.encode_memory(bio_tap, frame_mask, omega_bias=omega_bias)
        if self.decoder_type == "dlm":
            result = self.dlm_decoder.generate(
                enc_hidden, enc_mask, max_length=max_text_tokens, threshold=tau_dec, spd_top_k=spd_top_k,
                spd_renormalize=spd_renormalize, omega_bias=omega_bias,
            )
            # Sequential decoder passes of this decode (a batch's count: rows decode together). The AR arm's is
            # its generated length: 1 cached step per token plus the confidence pass. The boundary sweep reports the mean.
            self.last_decode_passes = int(result.forwards + result.cache_appends)
            # Slice to PRODUCED tokens (AR-arm parity): slot 0 (synthetic BOS) and everything past the first EOS are 1.0 pad bookkeeping, 
            # so an unsliced mean pins the reported confidence near 1 (~10 tokens on a 128 canvas → ≥ 0.92 regardless of quality).
            seq, conf = result.sequences[:, 1:], result.confidence[:, 1:]
            if seq.shape[0] == 1:  # every live caller decodes 1 window/buffer at a time
                hits = (seq[0] == int(self.dlm_decoder.eos_index)).nonzero(as_tuple=False)
                if hits.numel(): seq, conf = seq[:, : int(hits[0]) + 1], conf[:, : int(hits[0]) + 1]
            return seq, conf

        # AR arm: the front end owns generation (mT5, prompt-conditioned) and returns REAL per-token confidence. 
        # `num_beams>1` is the clean baseline's beam search; the SLT AR arm stays greedy.
        generated, confidence = self.front_end.ar_generate(
            enc_hidden, enc_mask, max_new_tokens=max_text_tokens, num_beams=num_beams, omega_bias=omega_bias,
        )
        self.last_decode_passes = int(generated.shape[1])  # start slot + N tokens = N cached steps + 1 confidence pass
        return generated, confidence

    @torch.no_grad()
    def gated_read(
        self, poses: torch.Tensor, frame_mask: torch.Tensor, timestamps_s: torch.Tensor | None = None, *,
        gate_enabled: bool = False, gate_eps: float | None = None, gate_min_span_frames: int | list[int] | None = None,
        commit_mask: torch.Tensor | None = None, gate_stream_start: bool = False, gate_anchor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Poses → (bio_logits, bio_tap, mask, omega_bias, gate_skip): translator's read of a window as every decode reads it, 
        BIO tap, membership gate and support crop (generate_from_poses).
        `gate_skip` (B, bool) marks windows the deployed FSM would never decode; all-False when gate is off."""
        bio_tap, mask, timestamps = self.front_end.extract_bio_tap(poses, frame_mask, timestamps_s)
        if self.bio_head is None and not gate_enabled:
            # Clean floor: no segmentation branch and, gate off, no reader — zeros fill the contract slot.
            bio_logits = bio_tap.new_zeros(*bio_tap.shape[:2], 4)
        else:
            bio_output = self.segment(poses, mask, timestamps, translator_tap=bio_tap)
            bio_logits = bio_output.logits
        # Inference gate, χ from the FSM commit log (a single window of the boundary sweep: none). Same Ω and the same support-crop 
        # read the decoder saw in training. AR arm injects Ω via HF cross-attn hooks (front_end.ar_generate).
        omega_bias = None
        gate_skip = torch.zeros(poses.shape[0], dtype=torch.bool)
        if gate_enabled:
            omega_bias, gate_stats = self.membership_gate(
                bio_logits, mask, decoder=DurationDecoder(self.duration_model),
                memory_len=self.front_end.prompt_length() + int(bio_tap.shape[1]), commit_mask=commit_mask, eps=gate_eps,
                min_span_frames=gate_min_span_frames, stream_start=gate_stream_start, 
                anchor_override=gate_anchor, timestamps_s=timestamps,
            )
            gate_skip = gate_stats["skip"]
            if gate_anchor is not None: gate_skip[(gate_anchor[:, 0] >= 0).cpu()] = False   # a known span is always decodable
        bio_tap, mask, omega_bias = support_crop(self.front_end, poses, timestamps, bio_tap, mask, omega_bias, gate_eps)
        return bio_logits, bio_tap, mask, omega_bias, gate_skip

    @torch.no_grad()
    def generate_from_poses(
        self, poses: torch.Tensor, frame_mask: torch.Tensor, timestamps_s: torch.Tensor | None = None, *,
        gate_enabled: bool = False, gate_eps: float | None = None, gate_min_span_frames: int | list[int] | None = None, commit_mask: torch.Tensor = None, 
        gate_stream_start: bool = False, gate_anchor: torch.Tensor | None = None, **decode_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Poses → (bio_logits, tokens, confidence, gate_skip). The read (BIO tap, membership gate, support crop) is gated_read. An
        # ungated read (the clean floor) needs no gate value; a gated one fails without gate_eps and gate_min_span_frames.
        # Decode knobs (max_text_tokens / tau_dec / spd_* / num_beams) pass via to `generate_from_bio_tap`, declared once there.
        bio_logits, bio_tap, mask, omega_bias, gate_skip = self.gated_read(
            poses, frame_mask, timestamps_s, gate_enabled=gate_enabled, gate_eps=gate_eps, gate_min_span_frames=gate_min_span_frames,
            commit_mask=commit_mask, gate_stream_start=gate_stream_start, gate_anchor=gate_anchor,
        )
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

    @staticmethod
    def gt_target_spans(batch: dict, timestamps, bio_mask) -> list | None:
        """Per-row frame span (B frame, terminator frame) of the GT unit(s) the collated text supervises, from the window-relative
        times the sampler stores (`target_spans_s`: the times of the B frame and the terminator frame that the labels give, so the
        label convention of `_candidate_frames` finds those frames again); None for rows without a text target. A multi-unit target
        ("A <sep> B") spans its first unit's B frame to its last unit's terminator."""
        spans_s = batch.get("target_spans_s")
        if spans_s is None or timestamps is None: return None
        out = []
        for b, span in enumerate(spans_s):
            if span is None: out.append(None); continue
            n = int(bio_mask[b].long().sum())
            c = MisalignedSLTModel._candidate_frames([{"start_s": span[0], "end_s": span[1]}], timestamps[b], n)[0]
            out.append((int(c["b_idx"]), int(min(c["t_idx"], n - 1))) if c["t_idx"] > c["b_idx"] else None)
        return out

    def _text_crop(self, batch, bio_logits, bio_mask, timestamps, rows, min_span_frames, delta_frames, eps, logs, text_grad=()):
        """Ω (B, 1, 1, M) for supervised `rows`, a hard crop as at inference: the predicted 1st span where it holds GT target within δ,
        else GT target span with its edges moved in training within same tolerance (at most δ frames of the target missing and at most
        δ frames outside it, in total; none in eval mode). The selection reads detached logits. Rows in `text_grad` add a zero-valued
        term that carries gradient of where the span from s' = max(crop start, B frame of the target's LAST unit) ends
        (`_end_survival`): there the text loss trains segmentation branch, never at or before s'."""
        b = len(bio_logits)
        floors = torch.as_tensor(min_span_frames, dtype=torch.long).clamp(min=1).expand(b).tolist()
        deltas = torch.as_tensor(delta_frames, dtype=torch.long).expand(b).tolist()
        with torch.no_grad():
            selected = self.membership_gate.select(
                bio_logits.detach(), bio_mask, commit_mask=batch.get("commit_mask"), min_span_frames=floors,
                timestamps_s=timestamps, decoder=DurationDecoder(self.duration_model),
            )
            p_start, p_term, p_closed = (x.cpu().tolist() for x in selected["anchors"])
            gt = self.gt_target_spans(batch, timestamps, bio_mask)
            lost = [i for i in rows if gt is None or gt[i] is None]
            if lost: raise ValueError(f"supervised rows {lost} carry no locatable target span (target_spans_s)")

            lengths = selected["lengths"].tolist()
            committed = (batch["commit_mask"].long().sum(1).tolist() if batch.get("commit_mask") is not None else [0] * b)
            starts, terms = list(p_start), list(p_term)
            accepted = 0

            for i in rows:
                g, d = gt[i], deltas[i]
                if p_closed[i]:
                    overlap = min(p_term[i], g[1]) - max(p_start[i], g[0])
                    outside = (p_term[i] - p_start[i]) - overlap
                    if overlap >= max(floors[i], g[1] - g[0] - d, 1) and outside <= d: accepted += 1; continue

                a, e = g
                if self.training and d > 0:
                    # Uniform over the edge moves that the acceptance test above would accept: at most d frames of the target
                    # missing and at most d frames outside it, in total (so no fallback crop trains text it cannot see).
                    while True:
                        sa, se = (int(x) for x in torch.randint(-d, d + 1, (2,)))
                        if max(sa, 0) + max(-se, 0) <= d and max(-sa, 0) + max(se, 0) <= d: break
                    a, e = a + sa, e + se
                a = min(max(a, int(committed[i])), lengths[i] - 1); e = min(max(e, a + floors[i]), lengths[i] - 1)
                starts[i], terms[i] = a, e

            # s' of each end-term row: the B frame of the target's last unit (data/sampler.py target_last_start_s, the target's own B
            # frame for 1 unit; the label convention of gt_target_spans), or the crop start where that lies later.
            last_s = batch.get("target_last_starts_s") or [None] * b
            lost = [i for i in text_grad if last_s[i] is None]
            if lost: raise ValueError(f"end-term rows {lost} carry no target_last_starts_s (data/sampler.py materialize)")
            ends_from = list(starts)
            for i in text_grad:
                last = self._candidate_frames([{"start_s": last_s[i], "end_s": last_s[i]}], timestamps[i], int(bio_mask[i].long().sum()))
                ends_from[i] = max(starts[i], int(last[0]["b_idx"]))

            has_term = torch.tensor([True if i in rows else bool(p_closed[i]) for i in range(b)], device=bio_logits.device)
            membership = span_membership(
                torch.tensor(starts, device=bio_logits.device), torch.tensor(terms, device=bio_logits.device), has_term,
                selected["lengths"], bio_logits.shape[1],
            )
        logs["text_accepted_rate"] = bio_logits.new_tensor(accepted / len(rows))
        if text_grad: membership = membership + self._end_survival(bio_logits, bio_mask, timestamps, list(text_grad), ends_from, floors)
        omega = membership_bias(membership, bio_mask, batch.get("commit_mask"), eps=eps)
        return omega_cross_bias(omega, memory_len=self.front_end.prompt_length() + int(bio_logits.shape[1]), dtype=bio_logits.dtype)


    def _end_survival(self, bio_logits, bio_mask, timestamps, rows, starts, floors):
        """(B, T) term, 0 in value, whose gradient is that of v(t) = P(the span that starts at s holds frame t) = P(its terminator >= t),
        t >= s, on `rows` (a straight-through estimator: the forward stays the crop), with s = `starts`: the crop start, or the B frame
        of the target's last unit where that lies later (`_text_crop`). v holds terminator frame because the crop does. Every row reads
        a hard crop (the model's own span or the jittered GT span), so the term is the text loss linearized at that crop and pulls the
        span's end toward the caption's end. v is 1st closed span's membership with frames before s committed and s a known start, so
        text loss moves only where that span ends: not its start, no earlier span, and no boundary inside a joined target."""
        device = bio_logits.device
        idx = torch.tensor(rows, device=device)
        s = torch.tensor([starts[i] for i in rows], device=device)
        frames = torch.arange(bio_logits.shape[1], device=device)[None]
        valid = bio_mask[idx].bool()
        commit = (frames < s[:, None]) & valid   # holds the window's own committed prefix too: the selected start is never before it
        
        # Frame s is a known start; ruling out O there makes 1st span start AT s (B, or I as a certified start). The posterior
        # conditions on 1st ELIGIBLE span, so the span at s must also reach Λ_min: no O or B on s+1..s+Λ_min-1, else a later span
        # becomes 1st eligible one and the text would move ITS start.
        pinned = bio_logits[idx].clone()
        pinned[torch.arange(len(rows), device=device), s, BIO['O']] = -1e4
        floor = torch.tensor([floors[i] for i in rows], device=device)
        short = (frames > s[:, None]) & (frames < (s + floor)[:, None])
        for tag in ('O', 'B'): pinned[..., BIO[tag]] = pinned[..., BIO[tag]].masked_fill(short, -1e4)
        post = self.membership_gate.posterior(
            pinned, valid.long().sum(1), [floors[i] for i in rows], commit, torch.ones(len(rows), dtype=torch.bool, device=device),
            decoder=DurationDecoder(self.duration_model), timestamps_s=None if timestamps is None else timestamps[idx],
        )
        p = post.closed_probability
        v = (post.closed / p.clamp(min=1e-6)[:, None]).clamp(0.0, 1.0) * ((frames >= s[:, None]) & (p[:, None] > 1e-6))
        # post.closed is P(s <= t < term); P(term >= t) is its value one frame earlier for t > s, and 1 at s.
        v = torch.where(frames > s[:, None], torch.cat((v[:, :1], v[:, :-1]), 1), v) * valid
        return v.new_zeros(bio_logits.shape[:2]).index_add(0, idx, v - v.detach())   # the posterior's dtype (float32 under AMP too)


    def forward_loss(
        self, batch: dict, *, lambda_trans: float, lambda_bio: float,
        dice_weight: float, bio_class_weights: torch.Tensor | None = None,
        oput_t_low: float, oput_t_high: float, oput_label_smoothing: float,
        gate_enabled: bool, gate_eps: float, gate_min_span_frames: int | list[int],
        gate_delta_frames: int | list[int], gate_detach_omega: bool = False,
    ) -> SLTLossOutput:
        """Stage-2 training loss for one mixed-mode batch: ``L = lambda_bio * L_BIO + lambda_trans * L_translation``.

        - BIO (all rows): class-weighted CE plus binary signing Dice on the GT labels; padding/UNK ignored.
        - Text (Mode 1/3 rows with a complete GT target: 1 unit "A", or a run of complete units that follow each other with gaps of 
          at most δ, "A <sep> B"): EVERY supervised row trains the translator on its GT text. The crop it reads (Ω, a hard crop from
          `models.membership_gate.span_membership`) is a sample of the segmentation the text belongs to: model's own 1st span when it
          holds GT target within δ (it misses at most δ frames of it in total, holds >= Λ_min frames of it, and holds at most δ frames 
          outside it), otherwise GT target span with its edges moved within that same tolerance (no move in eval mode). The label never 
          depends on the prediction, so the translator's data and labels are stationary, and it trains on segmenter's own spans wherever 
          those are a valid input for the label. The crop's value never carries a gradient. The translator reads Ω's support as a crop 
          (`support_crop`), as every gated decode does. On rows whose target (1 unit, or a joined run) is followed by a pause (more
          than δ from its last unit to the record's next unit, `target_pause_after`), text loss also trains segmentation branch via
          where the span from the last unit's B frame ends (`_end_survival`): frames up to that B frame count as committed, so the text
          can move only the run's end, never a boundary inside it (a gradient there would reward merging A and B), and the text's best
          end there is the caption's end. An abutting successor has no pose evidence, so those rows give no gradient. In the
          one-stream arm (`shared_pose_encoder`) the head reads the translator's own pose encoder, so there the text of EVERY
          supervised row also trains segmentation, through that encoder.
          `gate_detach_omega` is the no-text control of both arms: no end-side term and, in the one-stream arm, the text reads that
          encoder under stop-gradient. Text then trains no module the head reads (that encoder, its pose_proj included, and the
          head) and still trains the LM encoder and the decoder; BIO alone trains the encoder and the head. Only gradients change.
        - Other rows (truncated, headless, interior, gap): BIO only. A truncated window never receives text (P1).

        Gate off (clean floor): every complete target trains the translator on the whole window.
        """
        # Translator features and the segmentation branch read the same frames, through separate pose encoders or the translator's 1.
        bio_tap, bio_mask, timestamps = self.front_end.extract_bio_tap(batch["poses"], batch["frame_mask"], batch.get("timestamps_s"))
        bio_out = None
        if self.bio_head is not None and (float(lambda_bio) != 0.0 or gate_enabled):
            bio_out = self.segment(batch["poses"], bio_mask, timestamps, translator_tap=bio_tap)
        bio_loss = bio_nll_dice_loss(
            bio_out.logits, batch["bio_labels"], dice_weight=dice_weight, class_weights=bio_class_weights
        ) if bio_out is not None and float(lambda_bio) != 0.0 else bio_tap.new_zeros(())
        translation_loss = bio_tap.sum() * 0.0
        # The clean floor (lambda_bio 0) has no segmentation branch, so it logs no BIO loss (a column of zeros).
        logs: dict[str, torch.Tensor] = {"bio_loss": bio_loss.detach()} if float(lambda_bio) != 0.0 else {}
        # Does path (b) carry weight? The adapter's term against the head's normalized input (0 at step 0).
        if bio_out is not None and bio_out.residual_share is not None: logs["adapter_share"] = bio_out.residual_share

        # REALIZED mode mix (materialize() relabels windows jitter reshapes, so the drawn ratios are not what trains).
        realized = batch.get("mode_names")
        if isinstance(realized, list) and realized:
            for m in ("mode1", "mode2", "mode3", "mode4"):
                logs[f"mode_frac_{m}"] = bio_tap.new_tensor(sum(n == m for n in realized) / len(realized))
        target_tokens = batch.get("target_tokens")
        supervised = batch.get("translation_supervised")

        if target_tokens is not None and supervised is not None and supervised.any():
            idx = supervised.to(device=bio_tap.device).nonzero(as_tuple=False).flatten()
            mode_names = batch.get("mode_names")
            idx_list = idx.detach().cpu().tolist()
            if isinstance(mode_names, list):
                invalid = sorted({mode_names[int(i)] for i in idx_list} - {"mode1", "mode3"})
                if invalid: raise ValueError(
                    f"Translation supervision is allowed only for complete-conditioning Mode 1/3 windows; got {invalid}"
                )
            targets = batch.get("translation_targets") or []
            logs["text_pair_rate"] = bio_tap.new_tensor(sum(
                UNIT_SEPARATOR in (t["text"] if isinstance(t, dict) else getattr(t, "text", "")) for t in (targets[i] for i in idx_list)
            ) / len(idx_list))
            omega_bias = None

            if bio_out is not None and gate_enabled:
                pause_after = batch.get("target_pause_after")
                text_grad = [] if gate_detach_omega or pause_after is None or not torch.is_grad_enabled() \
                               else [i for i in idx_list if pause_after[i]]
                logs["text_grad_rate"] = bio_tap.new_tensor(len(text_grad) / len(idx_list))
                omega_bias = self._text_crop(
                    batch, bio_out.logits, bio_mask, timestamps, idx_list, gate_min_span_frames, 
                    gate_delta_frames, gate_eps, logs, text_grad
                )
            labels = target_tokens["labels"].to(bio_tap.device)
            om = None if omega_bias is None else omega_bias[idx]
            poses, ts = batch["poses"][idx], None if timestamps is None else timestamps[idx]
            # The crop pass is on gradient path: the text trains the translator's pose encoder through it
            # (BatchNorm on its running statistics, as every inference read; support_crop). In the one-stream arm's no-text control
            # (shared_pose_encoder with detach_omega) the head reads that encoder, so the text reads the whole read (the crop pass and
            # the window features it keeps) under stop-gradient. Detached after the pass, not a no-grad pass: the same kernels run, so
            # every forward value stays the joint arm's.
            tap, mask, read = support_crop(self.front_end, poses, ts, bio_tap[idx], bio_mask[idx], om, gate_eps)
            if self.shared_pose_encoder and gate_detach_omega: tap = tap.detach()
            enc_hidden, enc_mask = self.encode_memory(tap, mask, omega_bias=read)
            if self.decoder_type == "ar":
                translation_loss, row_sum, row_valid = self.front_end.ar_loss(
                    enc_hidden, enc_mask, labels[idx], label_smoothing=oput_label_smoothing, omega_bias=read, row_stats=True,
                )
            else:
                out = self.dlm_decoder.oput_forward(
                    enc_hidden=enc_hidden, enc_mask=enc_mask, t_low=oput_t_low, t_high=oput_t_high, 
                    labels=labels[idx], label_smoothing=oput_label_smoothing, omega_bias=read,
                    rollout_encode_fn=self.eval_encode_memory_fn(poses, bio_mask[idx], ts, om, gate_eps),
                    
                )
                translation_loss, row_sum, row_valid = out["translation_loss"], out.get("row_loss_sum"), out.get("row_valid_count")
            if isinstance(mode_names, list) and row_sum is not None and row_valid is not None:
                for mode in ("mode1", "mode3"):
                    rows = [k for k, i in enumerate(idx_list) if mode_names[int(i)] == mode]
                    if rows: logs[f"oput_{mode}"] = row_sum[rows].sum() / row_valid[rows].sum().clamp(min=1)

        # `lambda_bio=0` = translation-only: the faithful Uni-Sign SLT recipe (1 label-smoothed CE) for the clean-floor arm.
        total = float(lambda_bio) * bio_loss + float(lambda_trans) * translation_loss
        logs["translation_loss"] = translation_loss.detach()
        logs["loss"] = total.detach()
        return SLTLossOutput(total, bio_loss, translation_loss, logs, bio_logits=bio_out.logits if bio_out is not None else None)
