'''Block-diffusion (BD3LM) core — the backbone-agnostic substrate, no DMax, no backbone bindings.

Layering (so each concept has one home):
  models/block_diffusion.py  (this file)  BD3LM core: the abstract `BlockDiffusionDecoder` (masked-diffusion
                                          training over [xt|x0]) and the attention-mask builders.
  models/dmax.py                          DMax extension: `OPUTBlockDiffusionDecoder` (OPUT training + the block
                                          decode with SPD + confidence-bound surrogates) and the OPUT loss helpers.
  infer/decode.py                         the block decode itself (DMax decode_uniform semantics).
  models/unisign.py                       mBART binding: `MBartBlockDiffusionDecoder` (concrete `_decode`).
  models/unisign.py                   mT5 binding:  `MT5BlockDiffusionDecoder` (concrete `_decode`).

A decoder is built per language-model family by subclassing `dmax.OPUTBlockDiffusionDecoder` and implementing
ONLY `_decode` / `__init__`. Conditioning is uniform: every decoder consumes a precomputed encoder memory 
(`enc_hidden`/`enc_mask`); encoding is the front end's job (models/front_end.py).

Implements BD3LM (block diffusion) adapted to mBART / mT5 via the A2D recipe from dLLM:
  - Architecture: the pretrained decoder with block-causal self-attention.
  - Training: the [xt | x0] forward with the BD3LM attention mask (M_BD + M_OBC + M_BC) and repeated
              position IDs. The OBJECTIVE over it is DMax's OPUT (models/dmax.py), not the BD3LM ELBO.
  - Inference: block-by-block denoising over a KV cache of the final blocks (models/dmax.py, infer/decode.py).

Key insight (dLLM A2D, arXiv 2602.22661 takeaway box p.8): AR and diffusion models differ only in training 
objective and attention mask, NOT in architecture. Converting a pretrained decoder to BD3LM requires: 
  1. Replace causal self-attention mask with BD3LM mask during training.
  2. Concatenate noised tokens xt with clean tokens x0 as model input.
  3. Use repeated position IDs [0..L-1, 0..L-1] for both halves.
  4. Read only the xt-half logits; the loss over them is OPUT (models/dmax.py).

BD3LM training mask (2L x 2L) over concatenated [xt | x0] input:
  M_BD:  Block diagonal — within-block self-attention (xt<->xt, x0<->x0).
  M_OBC: Offset block causal — xt attends to x0 from *previous* blocks.
  M_BC:  Block causal — x0 attends to x0 from same and previous blocks.

Inference is block-causal: a committed prefix (clean, cached) and the current block (all MASK initially),
denoised by the DMax rule in infer/decode.py. No sigma/time conditioning (A2D: the model is not time-aware).

References:
  - dLLM paper + A2D recipe: https://arxiv.org/pdf/2602.22661
  - dLLM BD3LMTrainer: dllm/core/trainers/bd3lm.py (BD3LMTrainer.compute_loss)
  - BD3LM paper: https://arxiv.org/pdf/2503.09573
'''
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ════════════════════════════════════════════════════════════════════════════
# Attention-mask builders + canvas-tail supervision
# ════════════════════════════════════════════════════════════════════════════

def supervise_trailing_eos(x0, valid_mask, pad_index, eos_index, block_size=None, max_tokens=32):
    '''Mark padding up to the next block boundary as supervised EOS targets.

    Both reference implementations supervise the canvas tail beyond the sentence end:
      - dLLM AppendEOSBlockWrapper pads input_ids and labels to the next block boundary.
      - DMax keeps at most 32 EOS tokens on its much longer fixed canvas.
    Without this, slots beyond [eos, lang] are never supervised; at inference, masked slots past
    the true sentence end produce arbitrary high-confidence tokens before EOS commits — hallucinated
    tails and corrupted commit-gate confidence. Assumes right-padded sequences (no interior pads).

    Returns (x0, valid_mask) with the tail slots replaced by `eos_index` and marked valid.
    '''
    if max_tokens <= 0 or eos_index is None: return x0, valid_mask
    # Slot 0 is the canvas BOS and counts as content unconditionally: on mT5 the decoder start id IS the pad id, 
    # so a pad test would drop it, start the tail one slot early and stop it one slot short of the boundary — 
    # leaving the final block's last slot unsupervised whenever the label count is a multiple of the block.
    n_content = 1 + (x0[:, 1:] != pad_index).long().sum(dim=1)  # right-padding assumed
    if block_size is not None:
        to_boundary = (-n_content) % int(block_size)
        tail_count = torch.minimum(to_boundary, torch.full_like(to_boundary, int(max_tokens)))
    else:
        tail_count = torch.full_like(n_content, int(max_tokens))
    positions = torch.arange(x0.shape[1], device=x0.device).unsqueeze(0)
    tail = (positions >= n_content.unsqueeze(1)) & (positions < (n_content + tail_count).unsqueeze(1))
    x0 = torch.where(tail, torch.full_like(x0, int(eos_index)), x0)
    return x0, valid_mask | tail


def build_bd3lm_mask(seq_len, block_size, dtype, device):
    '''BD3LM training attention mask for concatenated [xt | x0] input.

    Mirrors _create_bd3lm_attention_mask (dLLM/dllm/core/trainers/bd3lm.py) and block_diff_mask 
    (bd3lms/models/dit.py). For input of length 2*tgt_len, creates a (1, 1, 2L, 2L) mask with 3 components:
    - M_BD:  xt_block_b ↔ all xt in same block b  (bidirectional, noisy self-attn within each block)
    - M_OBC: xt_block_b → x0_block_{0..b-1}       (clean prefix, STRICTLY prior, cross-attn for conditional context)
    - M_BC:  x0_block_b → x0_block_{0..b}         (block-causal over clean copy)
    x0 never attends to xt; inference uses build_block_causal_mask instead.

    Returns: (1, 1, 2L, 2L) float mask: 0 = attend, -inf = masked.
    '''
    n = seq_len
    idx = torch.arange(2 * n, device=device)
    q_idx  = idx[:, None]   # (2L, 1)
    kv_idx = idx[None, :]   # (1, 2L)

    # Indicate whether token belongs to xt or x0
    x0_flag_q  = q_idx  >= n
    x0_flag_kv = kv_idx >= n
    
    # Compute block indices
    block_q  = torch.where(x0_flag_q,  (q_idx  - n) // block_size, q_idx  // block_size)
    block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

    # M_BD: same block, same half (xt-xt or x0-x0)
    # M_OBC: xt queries attend to x0 keys from strictly earlier/previous blocks
    # M_BC: x0 queries attend to x0 keys from same/current or earlier/previous blocks
    block_diagonal      = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q >  block_kv) & x0_flag_kv & ~x0_flag_q
    block_causal        = (block_q >= block_kv) & x0_flag_kv & x0_flag_q

    # Combine Masks
    can_attend = block_diagonal | offset_block_causal | block_causal
    mask = torch.zeros(2 * n, 2 * n, dtype=dtype, device=device)
    mask = mask.masked_fill(~can_attend, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 2L, 2L)


def build_block_causal_mask(batch_size, tgt_len, block_size, dtype, device):
    '''Block-causal (staircase) attention mask.

    Bidirectional within each block; block b can attend to blocks 0..b (causal
    across blocks). Based on dLLM HF quickstart build_staircase_attention_mask.

    Returns: (B, 1, T, T) float mask: 0 = attend, -inf = masked.
    '''
    positions  = torch.arange(tgt_len, device=device)
    block_ids  = positions // block_size     # (T,)
    q_block    = block_ids.view(tgt_len, 1)  # (T, 1)
    k_block    = block_ids.view(1, tgt_len)  # (1, T)
    can_attend = k_block <= q_block          # (T, T): True = can attend
    mask = torch.zeros(tgt_len, tgt_len, dtype=dtype, device=device)
    mask = mask.masked_fill(~can_attend, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, tgt_len, tgt_len)

# ════════════════════════════════════════════════════════════════════════════
# Abstract base: BD3LM core (training + block-diffusion generation)
# ════════════════════════════════════════════════════════════════════════════

class BlockDiffusionDecoder(nn.Module): # Backbone-agnostic BD3LM decoder
    '''BD3LM decoder built on the pretrained AR decoder backbone.

    Replaces the AR decoder with a block diffusion decoder that:
      - Shares the pretrained weights (encoder + decoder layers).
      - Replaces the causal decoder self-attention mask with a block-causal mask.
      - Trains under DMax OPUT (models/dmax.py) over the BD3LM `[xt|x0]` forward; no MDLM ELBO, no noise schedule.

    Subclasses provide `_decode`, `_decode_with_decoder_forward` and `_decoder_stack` and call
    `_init_block_diffusion(...)` from their `__init__` to build the shared vocab+1 `[MASK]` embedding / LM head and
    store the hyper-parameters. Target prep and the BD3LM `[xt|x0]` forward are shared here; the OPUT loss and the 
    block decode live in `models.dmax` + `infer/decode.py`.
    '''
    def _init_block_diffusion(
        self, *, d_model: int, vocab_size: int, embed_source_weight: torch.Tensor, lm_source_weight: torch.Tensor,
        pad_index: int, eos_index: int, bos_index: int, embed_scale: float = 1.0, block_size: int = 4, 
        ignore_bos: bool = True, eos_supervision_tokens: int | None = None,
    ) -> None:
        self.d_model = d_model
        self.embed_scale = float(embed_scale)
        self.block_size = block_size
        self.ignore_bos = ignore_bos
        # Cap for EOS padding to the next block boundary. The default matches dLLM's AppendEOSBlockWrapper.
        self.eos_supervision_tokens = int(eos_supervision_tokens) if eos_supervision_tokens is not None else int(block_size)
        self.neg_infinity = -1e9

        # ── Tokenizer info ───────────────────────────────────────────────────
        self.pad_index = int(pad_index)
        self.eos_index = int(eos_index)
        self.bos_index = int(bos_index)
        self.vocab_size = vocab_size

        # ── Extend vocabulary with a MASK token ─────────────────────────────
        self.mask_token_id = vocab_size # Append [MASK] at index vocab_size so existing token IDs are unchanged

        # ── Extend embedding and language model heads ───────────────────────
        # No padding_idx: on mT5, pad id is also the canvas BOS, and padding_idx would freeze the BOS row the AR arm trains. 
        # A pad slot never reaches a supervised slot under block-causal attention, so its gradient is zero anyway.
        self.embed_tokens = nn.Embedding(vocab_size + 1, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size + 1, bias=False)
        with torch.no_grad():
            self.embed_tokens.weight[:vocab_size].copy_(embed_source_weight)
            self.lm_head.weight[:vocab_size].copy_(lm_source_weight)
            nn.init.normal_(self.embed_tokens.weight[vocab_size:], std=0.02)
            nn.init.zeros_(self.lm_head.weight[vocab_size:])


    # ── Architecture-specific decode (subclass responsibility) ────────────────
    def _decode(
        self, decoder_input_ids: torch.Tensor, enc_hidden: torch.Tensor, enc_mask: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None, omega_bias: torch.Tensor | None = None,
        logits_len: int | None = None,
    ) -> torch.Tensor:
        '''Run the AR decoder backbone with a custom (block-causal / BD3LM) self-attention mask.

        Bypasses the HF decoder.forward causal mask; everything else (positions, norms, cross-attention, FFN)
        is identical to the AR path. Each backbone implements this ONE method (no separate embeds variant).

        Args:
            decoder_input_ids: (B, T) token IDs — always supplied (used for shape and the learned positions).
            enc_hidden / enc_mask: encoder memory + padding mask for cross-attention.
            self_attn_mask: optional (1/B, 1, T, T) float mask; None -> block-causal mask from block_size.
            position_ids: optional (B, T) positions for the BD3LM [xt|x0] repeated geometry; None -> sequential.
            inputs_embeds: optional (B, T, d) decoder-input embeddings that REPLACE embedding decoder_input_ids
                (the SPD soft-embedding mixture). None -> the decoder embeds decoder_input_ids itself.
            omega_bias: optional (B, 1, 1, M) membership-gate bias ADDED to the cross-attention logits, every
                layer and head (models.membership_gate; docs/membership_gate.md §2.9). None -> no gate.
            logits_len: optional prefix length to project through the vocab head; the rest of the canvas runs the
                backbone but skips the |V|-way matmul. Sliced after the final dropout, so the kept positions are
                bit-identical. None -> project every position.
        '''
        raise NotImplementedError

    
    def _prepare_x0( # Clean target construction (BOS-prefix, block-pad, supervised EOS tail)
        self, labels: torch.Tensor, decoder_input_ids: torch.Tensor | None = None, eos_supervision: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = labels.shape[0]
        x0 = labels.clone()
        x0[x0 == -100] = self.pad_index
        if decoder_input_ids is not None: bos = decoder_input_ids[:, :1].to(device=labels.device)
        else: bos = torch.full((batch, 1), self.bos_index, dtype=x0.dtype, device=labels.device)
        x0 = torch.cat([bos, x0], dim=1)             # (B, L+1)

        # Text attention mask: 1 for real tokens, 0 for padding
        valid = (x0 != self.pad_index)           # (B, L+1) bool
        if self.ignore_bos: valid[:, 0] = False  # BOS never masked

        # Align length to a multiple of block_size
        aligned_len = max(1, math.ceil(x0.shape[1] / self.block_size)) * self.block_size
        if x0.shape[1] < aligned_len:
            x0 = F.pad(x0, (0, aligned_len - x0.shape[1]), value=self.pad_index)
            valid = F.pad(valid, (0, aligned_len - valid.shape[1]), value=False)

        return supervise_trailing_eos( # Supervise padding only to next block boundary. Later canvas slots stay ignored.
            x0, valid, pad_index=self.pad_index, eos_index=self.eos_index, block_size=self.block_size,
            max_tokens=self.eos_supervision_tokens if eos_supervision is None else int(eos_supervision)
        )


    def _bd3lm_logits( # BD3LM [xt|x0] forward with repeated effective positions; return xt-half logits (first L).
        self, noisy_ids: torch.Tensor, clean_ids: torch.Tensor, enc_hidden: torch.Tensor, enc_mask: torch.Tensor,
        omega_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # BD3LM attention mask: (1, 1, 2L, 2L)
        batch, length = clean_ids.shape
        bd3lm_mask = build_bd3lm_mask(length, self.block_size, enc_hidden.dtype, clean_ids.device)

        # Repeated position IDs: [0..L-1, 0..L-1] (no sigma/time conditioning — A2D is not time-aware; dLLM
        # BD3LMTrainer passes no sigma, and TimestepEmbedder/AdaLN is DDiT-specific, absent from mBART/mT5).
        base_pos = torch.arange(length, device=clean_ids.device).unsqueeze(0).expand(batch, -1)
        position_ids = torch.cat([base_pos, base_pos], dim=1)  # (B, 2L)

        # BD3LM forward: [xt | x0] with the 3-component mask + SHARED positional embeddings; xt-half logits only.
        # omega_bias (over the M encoder frames) is query-independent, so the same (B,1,1,M) bias applies to all
        # 2L target queries — the [xt|x0] concatenation on the TARGET axis does not touch the cross-attn key axis.
        # logits_len=length: the x0 half conditions the xt half through self-attention but its own logits are
        # never read, so the |V|-way head runs on the xt half only. Halves OPUT's largest matmul, bit-identical.
        return self._decode(
            torch.cat([noisy_ids, clean_ids], dim=1), enc_hidden, enc_mask,
            self_attn_mask=bd3lm_mask, position_ids=position_ids, omega_bias=omega_bias, logits_len=length,
        )  # (B, L, V+1) — xt half

