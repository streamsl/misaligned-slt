"""The abstract SLT front-end contract. Concrete front ends live with their backbone:

  models/unisign.py -> UniSignMT5FrontEnd    (Uni-Sign pose encoder + mT5, with the task prompt; default)
  models/unisign.py -> UniSignMBartFrontEnd  (same pose encoder + prompt + mBART; the mT5-vs-mBART ablation)

`MisalignedSLTModel` (models/streaming_slt.py) and the eval/train entry points depend only on this interface, so adding a 
backbone never touches the shared training/inference code — we implement 1 `SLTFrontEnd` subclass in that backbone's module.

A front end exposes two frame-aligned/sequence views plus the AR + DLM decode hooks:
  extract_bio_tap(poses, frame_mask)  -> (bio_tap, bio_mask, timestamps)   # translator per-frame features, length T
  encode_memory(bio_tap, bio_mask, omega_bias=None) -> (enc_hidden, enc_mask)  # memory, length M; Ω biases encoder keys
  ar_loss / ar_generate                                                    # autoregressive decoder
  make_dlm_decoder(block_size) -> dmax.OPUTBlockDiffusionDecoder           # block-diffusion decoder

`bio_tap` is the TRANSLATOR's per-frame pose feature BEFORE the seq2seq encoder; the segmentation branch reads its own pose 
encoder (MisalignedSLTModel.segment). For Uni-Sign the encoder memory is LONGER than the tap (the task prompt is prepended 
to the pose tokens), so M = prompt_len + T while the tap stays length T (frame-aligned). Callers that need the OPUT 
eval-rollout or the confidence-bound re-encode must use the per-frame mask (length T), not the encoder mask (length M).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from models.dmax import OPUTBlockDiffusionDecoder
from models.membership_gate import CrossAttnOmegaInjector


class SLTFrontEnd(nn.Module):
    """Backbone-agnostic SLT front end. Subclasses set `lm_model`, `tokenizer`, `pad_token_id`, `eos_token_id`,
    `decoder_start_id`, `bio_tap_dim` and implement `extract_bio_tap` / `encode_memory` / `ar_loss` / `make_dlm_decoder` / 
    `load_pretrained`. The AR generation and the Ω injector are shared here."""
    lm_model: nn.Module
    bio_tap_dim: int
    pad_token_id: int
    eos_token_id: int
    decoder_start_id: int | None

    # ── subclass responsibilities ────────────────────────────────────────────
    def extract_bio_tap(self, poses, frame_mask, timestamps_s=None):
        raise NotImplementedError

    def encode_memory(self, bio_tap, bio_mask, omega_bias=None):
        # omega_bias (B,1,1,M): the membership prior, added to the LM encoder's self-attention keys (None = ungated).
        raise NotImplementedError

    def ar_loss(self, enc_hidden, enc_mask, labels, label_smoothing: float = 0.2, omega_bias=None, row_stats: bool = False):
        raise NotImplementedError

    def make_dlm_decoder(self, block_size: int) -> OPUTBlockDiffusionDecoder:
        raise NotImplementedError

    def load_pretrained(self, ckpt_path, strict: bool = True) -> dict[str, int]:
        raise NotImplementedError

    # ── membership gate on the AR path (shared) ───────────────────────────────
    def ar_omega_context(self, omega_bias):
        # Context manager: within it, every AR cross-attention adds Ω (None → identity). Wraps ar_loss /
        # ar_generate / the AR confidence-bound forward so all AR decodes see the same conditioning.
        # 1st call attaches the Ω hooks to AR language model (1 injector per stack, shared with the DLM decoder).
        # DLM arm gates in its own manual decode loop; AR arm reuses HF forward/generate, so Ω rides in on these hooks.
        return CrossAttnOmegaInjector.attach(self.lm_model).with_omega(omega_bias)

    @torch.no_grad()
    def ar_generate(self, enc_hidden, enc_mask, max_new_tokens=128, num_beams=1, omega_bias=None):
        # AR generation over a precomputed encoder memory. Returns (tokens, per-token confidence). 
        # Under `omega_bias`, every cross-attention step is membership-gated (same Ω the DLM decode uses).
        kwargs = {} if self.decoder_start_id is None else {"decoder_start_token_id": self.decoder_start_id}
        with self.ar_omega_context(omega_bias):
            generated = self.lm_model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden), attention_mask=enc_mask,
                max_new_tokens=max_new_tokens, num_beams=num_beams, **kwargs,
            )
            conf = self._teacher_forced_confidence(enc_hidden, enc_mask, generated)  # gated too (fair confidence)
        return generated, conf

    def _teacher_forced_confidence(self, enc_hidden, enc_mask, generated):
        # REAL per-token confidence: the softmax prob the decoder assigns each token it produced 
        # (one teacher-forced pass). (B, L) aligned with `generated`; the start slot is 1.
        if generated.shape[1] < 2: return torch.ones(generated.shape, dtype=torch.float32, device=generated.device)

        # HF generate() EXPANDS the encoder_outputs it receives to batch*num_beams in place, so it cannot be reused; 
        # build a fresh BaseModelOutput from the un-expanded enc_hidden. Teacher-force the GENERATED sequence 
        # (decoder_input_ids = generated[:, :-1]) to read the prob assigned to each produced token.
        out = self.lm_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden), attention_mask=enc_mask,
            decoder_input_ids=generated[:, :-1], use_cache=False, return_dict=True,
        )
        tok_prob = out.logits.log_softmax(dim=-1).gather(-1, generated[:, 1:].unsqueeze(-1)).squeeze(-1).exp()
        start = torch.ones((generated.shape[0], 1), dtype=tok_prob.dtype, device=tok_prob.device)
        return torch.cat([start, tok_prob], dim=1)
