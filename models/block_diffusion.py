'''Block-diffusion (BD3LM) core — the backbone-agnostic substrate, no DMax, no backbone bindings.

Layering (so each concept has one home):
  models/block_diffusion.py  (this file)  BD3LM core: the abstract `BlockDiffusionDecoder` (masked-diffusion
                                          training over [xt|x0] + block-by-block confidence-remask generation)
                                          and the attention-mask builders.
  models/dmax.py                          DMax extension: `OPUTBlockDiffusionDecoder` (OPUT training + SPD/DCD
                                          inference + confidence-bound surrogates) and the OPUT loss helpers.
  models/unisign.py                       mBART binding: `MBartBlockDiffusionDecoder` (concrete `_decode`).
  models/unisign.py                   mT5 binding:  `MT5BlockDiffusionDecoder` (concrete `_decode`).

A decoder is built per language-model family by subclassing `dmax.OPUTBlockDiffusionDecoder` and implementing
ONLY `_decode` / `__init__`. Conditioning is uniform: every decoder consumes a precomputed encoder memory 
(`enc_hidden`/`enc_mask`); encoding is the front end's job (models/front_end.py).

Implements BD3LM (block diffusion) adapted to mBART via A2D recipe from dLLM, with full bd3lms fidelity:
  - Architecture: pretrained mBART decoder with block-causal self-attention.
  - Training: BD3LM masked diffusion loss with [xt | x0] concatenation,
              BD3LM attention mask (M_BD + M_OBC + M_BC), repeated position IDs,
              and cross-entropy weighted by 1/t at masked positions.
  - Inference: dLLM-style BD3LM semi-AR sampling with confidence-based remasking,
               temperature-controlled Gumbel-max, block-by-block denoising.

Key insight (dLLM A2D, arXiv 2602.22661 takeaway box p.8): AR and diffusion models differ only in training 
objective and attention mask, NOT in architecture. Converting a pretrained decoder to BD3LM requires: 
  1. Replace causal self-attention mask with BD3LM mask during training.
  2. Concatenate noised tokens xt with clean tokens x0 as model input.
  3. Use repeated position IDs [0..L-1, 0..L-1] for both halves.
  4. Compute MDLM masked diffusion loss on only the xt-half logits.

BD3LM training mask (2L x 2L) over concatenated [xt | x0] input:
  M_BD:  Block diagonal — within-block self-attention (xt<->xt, x0<->x0).
  M_OBC: Offset block causal — xt attends to x0 from *previous* blocks.
  M_BC:  Block causal — x0 attends to x0 from same and previous blocks.

Inference (dLLM BD3LMSampler) uses block-causal mask with confidence-based remasking:
  - Block-by-block: committed prefix (clean) + current block (all MASK initially).
  - Inner loop per block: predict tokens, score by confidence, commit top-k, repeat.
  - Linear unmasking schedule: ~(remaining / steps_left) tokens per step.
  - Temperature-controlled Gumbel-max for diverse sampling.
  - No sigma/time conditioning (A2D: model is not time-aware).

References:
  - dLLM paper + A2D recipe: https://arxiv.org/pdf/2602.22661
  - dLLM BD3LMTrainer: dllm/core/trainers/bd3lm.py (BD3LMTrainer.compute_loss)
  - dLLM BD3LMSampler: dllm/core/samplers/bd3lm.py
  - BD3LM paper: https://arxiv.org/pdf/2503.09573
  - LogLinearNoise: bd3lms/noise_schedule.py
'''
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
# Attention-mask builders + canvas-tail supervision
# ════════════════════════════════════════════════════════════════════════════

def supervise_trailing_eos(x0, valid_mask, pad_index, eos_index, max_tokens=32):
    '''Mark the first `max_tokens` padding slots after the sentence as supervised EOS targets.

    Both reference implementations supervise the canvas tail beyond the sentence end:
      - dLLM AppendEOSBlockWrapper (dllm/core/trainers/bd3lm.py) pads input_ids AND labels with
        eos_token_id to the block boundary, so the padded tail is maskable and supervised.
      - DMax process_mdm_sft_example (dFactory/tasks/dataset/data_transform.py) keeps loss on the
        first 32 EOS of the trailing run and sets labels to -100 only after that.
    Without this, slots beyond [eos, lang] are never supervised; at inference, masked slots past
    the true sentence end produce arbitrary high-confidence tokens before EOS commits — hallucinated
    tails and corrupted commit-gate confidence. Assumes right-padded sequences (no interior pads).

    Returns (x0, valid_mask) with the tail slots replaced by `eos_index` and marked valid.
    '''
    if max_tokens <= 0 or eos_index is None: return x0, valid_mask
    n_content = (x0 != pad_index).long().sum(dim=1)  # includes BOS; right-padding assumed
    positions = torch.arange(x0.shape[1], device=x0.device).unsqueeze(0)
    tail = (positions >= n_content.unsqueeze(1)) & (positions < (n_content + int(max_tokens)).unsqueeze(1))
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

    Replaces AR decoder with a block diffusion decoder that:
      - Shares the same mBART pretrained weights (encoder + decoder layers).
      - Replaces the causal decoder self-attention mask with a block-causal mask.
      - Trains with MDLM masked diffusion loss (loglinear noise schedule).
      - Generates via iterative denoising block-by-block at inference time.

    Subclasses provide ONLY the architecture-specific decode (`_decode`) and call `_init_block_diffusion(...)` from 
    their `__init__` to build the shared vocab+1 `[MASK]` embedding / LM head and store the hyper-parameters. 
    Everything else (target prep, noise schedule, BD3LM `[xt|x0]` forward, MDLM loss, and block-by-block sampler) is 
    shared here. DMax (OPUT + SPD/DCD) lives in `models.dmax`.
    '''
    def _init_block_diffusion(
        self, *, d_model: int, vocab_size: int, embed_source_weight: torch.Tensor, lm_source_weight: torch.Tensor,
        pad_index: int, eos_index: int, bos_index: int, embed_scale: float = 1.0,
        block_size: int = 4, sampling_eps_min: float = 1e-3, sampling_eps_max: float = 1.0,
        antithetic_sampling: bool = True, ignore_bos: bool = True, temperature: float = 0.0,
        remasking: str = "low_confidence", steps_per_block: int | None = None, eos_supervision_tokens: int = 32,
    ) -> None:
        self.d_model = d_model
        self.embed_scale = float(embed_scale)
        self.block_size = block_size
        self.sampling_eps_min = sampling_eps_min
        self.sampling_eps_max = sampling_eps_max
        self.antithetic_sampling = antithetic_sampling
        self.ignore_bos = ignore_bos
        self.eos_supervision_tokens = int(eos_supervision_tokens)
        self.neg_infinity = -1e9

        # ── Inference params (dLLM-style) ─────────────────────────────────────
        self.temperature = temperature      # Gumbel noise temperature (0 = greedy argmax)
        self.remasking = remasking          # 'low_confidence' or 'random'
        self.steps_per_block = steps_per_block  # None = auto from diffusion_steps

        # ── Tokenizer info ───────────────────────────────────────────────────
        self.pad_index = int(pad_index)
        self.eos_index = int(eos_index)
        self.bos_index = int(bos_index)
        self.vocab_size = vocab_size

        # ── Extend vocabulary with a MASK token ─────────────────────────────
        self.mask_token_id = vocab_size # Append [MASK] at index vocab_size so existing token IDs are unchanged
        self.mask_index = self.mask_token_id  # Backward-compat alias (SPD/DCD decode)

        # ── Extend embedding and language model heads ───────────────────────
        self.embed_tokens = nn.Embedding(vocab_size + 1, d_model, padding_idx=self.pad_index)
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
        inputs_embeds: torch.Tensor | None = None,
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

        # Supervised EOS tail after [.., eos, lang] (dLLM AppendEOSBlockWrapper / DMax 32-trailing-eos): without it, 
        # slots past the sentence end are never trained and decode to confident garbage before EOS commits, which the 
        # commit gate then reads as hardened. See block_diffusion.supervise_trailing_eos.
        return supervise_trailing_eos(
            x0, valid, pad_index=self.pad_index, eos_index=self.eos_index,
            max_tokens=self.eos_supervision_tokens if eos_supervision is None else int(eos_supervision),
        )


    def _sample_t(self, batch_size: int, num_blocks: int, device: torch.device) -> torch.Tensor:
        # Antithetic per-block timestep sampling; mask x0 -> xt (bd3lms diffusion.py _sample_t).
        t = torch.rand((batch_size, num_blocks), device=device)
        if self.antithetic_sampling:
            offset = torch.arange(batch_size * num_blocks, device=device).float()
            offset = (offset / (batch_size * num_blocks)).view(batch_size, num_blocks)
            t = (t / (batch_size * num_blocks) + offset) % 1.0
        t = t.repeat_interleave(self.block_size, dim=-1)
        return t * (self.sampling_eps_max - self.sampling_eps_min) + self.sampling_eps_min


    def _bd3lm_logits( # BD3LM [xt|x0] forward with repeated effective positions; return xt-half logits (first L).
        self, noisy_ids: torch.Tensor, clean_ids: torch.Tensor, enc_hidden: torch.Tensor, enc_mask: torch.Tensor,
    ) -> torch.Tensor: 
        # BD3LM attention mask: (1, 1, 2L, 2L)
        batch, length = clean_ids.shape
        bd3lm_mask = build_bd3lm_mask(length, self.block_size, enc_hidden.dtype, clean_ids.device)
        
        # Repeated position IDs: [0..L-1, 0..L-1] (no sigma/time conditioning — A2D is not time-aware; dLLM
        # BD3LMTrainer passes no sigma, and TimestepEmbedder/AdaLN is DDiT-specific, absent from mBART/mT5).
        base_pos = torch.arange(length, device=clean_ids.device).unsqueeze(0).expand(batch, -1)
        position_ids = torch.cat([base_pos, base_pos], dim=1)  # (B, 2L)

        # BD3LM forward: [xt | x0] with the 3-component mask + SHARED positional embeddings; xt-half logits only.
        logits = self._decode(
            torch.cat([noisy_ids, clean_ids], dim=1), enc_hidden, enc_mask,
            self_attn_mask=bd3lm_mask, position_ids=position_ids,
        )  # (B, 2L, V+1)
        return logits[:, :length]  # (B, L, V+1), take only first L logits (xt half)


    # ── BD3LM training forward (MDLM loglinear loss on the xt half) ────────────
    def forward(
        self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor, labels: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        '''BD3LM training forward (dLLM A2D) over precomputed encoder conditioning.

        Implements the BD3LM training from dLLM (arxiv.org/abs/2602.22661):
          1. Build x0 (clean target), pad to multiple of block_size.
          2. Sample per-block noise and create xt (noised tokens).
          3. Concatenate [xt | x0] → (B, 2L) with BD3LM attention mask.
          4. Repeated position IDs: [0..L-1, 0..L-1].
          5. Take first L logits (xt positions) for loss computation.
          6. Cross-entropy weighted by 1/t at masked positions only.

        Returns: dict with 'translation_loss' (scalar).
        '''
        device = enc_hidden.device
        B = enc_hidden.shape[0]

        # ── 1. Build x0 (clean target): [BOS, label_tokens...] ───────────────
        x0, text_mask = self._prepare_x0(labels.to(device), decoder_input_ids)
        num_blocks = x0.shape[1] // self.block_size

        # ── 2. Sample noise and create xt (noised tokens) ────────────────────
        t = self._sample_t(B, num_blocks, device) # (B, L) per-block t; mask x0 → xt
        p = t # loglinear schedule: p=t (mask probability)
        rand = torch.rand_like(x0.float())
        masked_mask = (rand < p) & text_mask      # (B, L) bool, True = masked AND valid text position
        xt = torch.where(masked_mask, self.mask_token_id, x0)

        # ── 3. Decoder forward with BD3LM mask ───────────────────────────────
        logits = self._bd3lm_logits(xt, x0, enc_hidden, enc_mask)  # (B, L, V+1)

        # ── 4. Compute weighted cross-entropy loss ────────────────────────────
        loss_weights = 1.0 / t.clamp(min=1e-6)  # (B, L), 1/t per block (MDLM loglinear schedule)
        
        # Substitution parameterization is simply equivalent to cross-entropy
        # with targets = x0 and logits masked to force MASK prediction at masked positions.
        token_nll = F.cross_entropy( 
            logits.transpose(1, 2),             # (B, V+1, L)
            x0,                                 # (B, L) — targets
            reduction='none',                   # (B, L)
        )
        # Mask: only count loss at masked & maskable positions
        loss_mask = masked_mask.float()         # (B, L)
        weighted_nll = token_nll * loss_weights * loss_mask  # Zero out unmasked positions

        # Normalize by total MASKABLE tokens (label != -100), matching dLLM "token" norm.
        # FIX (verified vs dllm/core/trainers/mdlm.py:200-202 & bd3lm.py:230, arXiv 2602.22661):
        # dllm divides by maskable_mask.sum() (all non-pad/non-BOS target positions), NOT by the
        # masked-count. Using loss_mask.sum() (masked only) over-scales the loss and adds per-batch
        # variance. text_mask is the maskable set (valid, non-BOS) defined above.
        translation_loss = weighted_nll.sum() / text_mask.float().sum().clamp(min=1)
        return {'translation_loss': translation_loss}


    # ── Inference (dLLM BD3LMSampler) ───────────────────────────────────────
    @staticmethod
    def _add_gumbel_noise(logits, temperature):
        '''Temperature-controlled Gumbel-max (dLLM samplers/utils.py add_gumbel_noise).
        temperature=0: greedy argmax. Higher temperature: more diverse samples.
        '''
        if temperature == 0: return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise


    @staticmethod
    def _get_num_transfer_tokens(mask_index, steps):
        '''Per-step unmasking schedule (dLLM core/samplers/utils.py).

        Use linear alpha schedule: at each step unmask ~(remaining/steps_left) tokens. 
        Distribute unmasking evenly across steps.

        Args:
            mask_index: (B, L) bool, True at masked positions.
            steps: int, diffusion steps for this block.
        Returns: (B, effective_steps) int64 tensor, tokens to unmask per step.
        '''
        mask_num = mask_index.sum(dim=1, keepdim=True)  # (B, 1)
        B = mask_num.size(0)
        device = mask_index.device
        num_transfer = torch.zeros(B, steps, dtype=torch.int64, device=device)
        
        for i in range(B):
            remaining = mask_num[i, 0].clone()
            for j in range(steps):
                t = (steps - j) / steps
                s = (steps - j - 1) / steps
                if t <= 0: break
                reverse_transfer_prob = 1.0 - (s / t)  # linear: 1 / (steps - j)
                k = torch.round(remaining.float() * reverse_transfer_prob).to(torch.int64)
                k = torch.clamp(k, min=0, max=remaining)
                num_transfer[i, j] = k
                remaining -= k
                if remaining <= 0: break
                
        # Note: because llada is not conditioned on time, this allows us to skip steps with no unmasking (i.e. transfer).
        # Clear all zeros per row (compact) and right-pad with zeros
        # Remove zeros per row, then pad only up to the max length across rows
        rows, max_len = [], 0
        for i in range(B):
            nonzero = num_transfer[i][num_transfer[i] > 0]
            rows.append(nonzero)
            max_len = max(max_len, nonzero.numel())
        return torch.stack([
            torch.cat([r, torch.zeros(max_len - r.numel(), dtype=r.dtype, device=r.device)]) 
            if r.numel() < max_len else r for r in rows
        ], dim=0)


    def _diffusion_step_block(self, logits, x_block, mask_block, num_transfer_step):
        '''One confidence-remask step (dLLM core/samplers/bd3lm.py _diffusion_step_block).

        1. Gumbel-max sample x0 from logits.
        2. Score by confidence (softmax prob or random).
        3. Commit top-k most confident tokens; rest stay MASK.
        '''
        B, L, _ = logits.shape
        device = logits.device
        if not mask_block.any(): return x_block

        logits_noisy = self._add_gumbel_noise(logits, self.temperature)
        x0 = torch.argmax(logits_noisy, dim=-1)  # (B, L)

        if self.remasking == 'low_confidence':
            p = F.softmax(logits.float(), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif self.remasking == 'random': x0_p = torch.rand((B, L), device=device)
        else: raise ValueError(f'Unknown remasking: {self.remasking}')

        # Only masked positions can change
        x0 = torch.where(mask_block, x0, x_block)
        neg_inf = torch.full_like(x0_p, -float('inf'))
        confidence = torch.where(mask_block, x0_p, neg_inf)

        transfer = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(B):
            k = int(num_transfer_step[j].item())
            if k <= 0: continue
            
            valid_count = (confidence[j] > -float('inf')).sum().item()
            if valid_count == 0: continue
            k = min(k, valid_count)
            _, sel = torch.topk(confidence[j], k)
            transfer[j, sel] = True

        x_new = x_block.clone()
        x_new[transfer] = x0[transfer]
        return x_new


    @torch.no_grad()
    def generate(
        self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor, max_length: int = 100, diffusion_steps: int = 128,
    ) -> dict[str, torch.Tensor]:
        '''Vanilla BD3LM block diffusion generation over precomputed encoder conditioning (dLLM BD3LMSampler for A2D).

        Generates text block-by-block with confidence-based remasking:
          1. For each new block: append block_size MASK tokens.
          2. Inner diffusion loop: predict, score confidence, commit top-k,
             re-mask the rest, repeat for steps_per_block iterations.
          3. Move to next block once current is fully denoised.

        Matches dLLM core/samplers/bd3lm.py BD3LMSampler.sample().
        '''
        B = enc_hidden.shape[0]
        device = enc_hidden.device
        num_blocks = max(1, max_length // self.block_size)
        spb = self.steps_per_block or max(1, diffusion_steps // num_blocks)

        # Start with BOS + (block_size-1) MASK tokens
        x = torch.full((B, self.block_size), self.mask_token_id, dtype=torch.long, device=device)
        x[:, 0] = self.bos_index
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for b_idx in range(num_blocks):
            if finished.all(): break

            # Append new MASK block (except first iteration — already initialized)
            if b_idx > 0:
                new_block = torch.full((B, self.block_size), self.mask_token_id, dtype=torch.long, device=device)
                x = torch.cat([x, new_block], dim=1)
            cur_len = self.block_size

            # Compute unmasking schedule for this block
            block_mask = (x[:, -cur_len:] == self.mask_token_id)  # (B, block_size)
            num_transfer = self._get_num_transfer_tokens(block_mask, spb)
            effective_steps = num_transfer.shape[1]

            # Inner diffusion loop
            for i_step in range(effective_steps):
                x_block = x[:, -cur_len:]
                mask_block = (x_block == self.mask_token_id)
                if not mask_block.any(): break

                # Full forward with block-causal mask (no KV cache)
                logits = self._decode(x, enc_hidden, enc_mask)  # (B, T_total, V+1)
                logits_block = logits[:, -cur_len:]             # (B, block_size, V+1)

                # Remasking step
                x_block_new = self._diffusion_step_block(logits_block, x_block, mask_block, num_transfer[:, i_step])
                x[:, -cur_len:] = x_block_new

            # EOS stopping
            if self.eos_index is not None:
                eos_in_block = (x[:, -cur_len:] == self.eos_index).any(dim=1)
                finished = finished | eos_in_block
        return {'sequences': x}