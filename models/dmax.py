'''DMax extension to the BD3LM substrate: OPUT training, block decoding with SPD, and the confidence-bound surrogate.

`OPUTBlockDiffusionDecoder` adds DMax's mechanisms on top of `block_diffusion.BlockDiffusionDecoder`, over a
precomputed encoder memory (`enc_hidden`/`enc_mask`); it stays abstract on `_decode`, `_decode_with_decoder_forward`
and `_decoder_stack`, so the mBART / mT5 bindings (models/unisign.py) supply only the backbone forwards.

  - `oput_forward`     — OPUT two-pass training (mask + on-policy argmax corruption); trains self-correction.
  - `generate`         — block decode (infer/decode.py): DMax's confident-prefix commits, SPD soft state and
                         self-revision inside a block, over a KV cache of the final blocks.
  - `remasked_logits`  — grad-bearing surrogate for the confidence-bound term.

References: DMax (arXiv 2604.08302; dInfer decode_uniform, train_llada2_bd_oput.py); dLLM A2D (arXiv 2602.22661).
'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from models.block_diffusion import BlockDiffusionDecoder
from infer.decode import DecodeResult, block_diffusion_decode
from train.losses import masked_cross_entropy


# ════════════════════════════════════════════════════════════════════════════
# OPUT loss (backbone-agnostic; operates over a fixed conditioning closure)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class OPUTOutput:
    loss: torch.Tensor
    mask_loss: torch.Tensor
    pred_loss: torch.Tensor
    masked_positions: torch.Tensor
    rollout_tokens: torch.Tensor
    # Detached per-row (summed token loss, valid-token count). Lets one merged OPUT call report the per-mode
    # breakdown without paying for a second decoder pass per mode group.
    row_loss_sum: torch.Tensor | None = None
    row_valid_count: torch.Tensor | None = None
    noise: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None  # (t, masked, rollout) drawn here.


def sample_mask_ratio(shape: tuple[int, int], device: torch.device, t_low: float = 0.0, t_high: float = 1.0) -> torch.Tensor:
    # Sample one OPUT/DMax noise level per sequence and broadcast over tokens.
    low, high = float(t_low), float(t_high)
    if high < low: raise ValueError("t_high must be >= t_low")
    batch, length = shape
    ratios = torch.empty((batch, 1), device=device).uniform_(low, high)
    return ratios.expand(batch, length)


def oput_two_pass_loss(
    clean_ids: torch.Tensor, valid_mask: torch.Tensor, decode_fn: Callable[[torch.Tensor], torch.Tensor], mask_token_id: int, 
    t_low: float = 0.3, t_high: float = 0.8, loss_over_all_positions: bool = True, sample_rollout: bool = False, 
    rollout_decode_fn: Callable[[torch.Tensor], torch.Tensor] | None = None, label_smoothing: float = 0.0,
    noise: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
) -> OPUTOutput:
    """DMax-style OPUT over a fixed conditioning closure.

    `decode_fn` must close over fixed, complete conditioning. Called twice: on masked target tokens (L_mask), then on an 
    on-policy corruption from the rollout pass (L_pred). Both supervise recovery of `clean_ids` over all valid positions 
    (DMax §3.1; OPUT SFT transform leaves loss un-restricted to masked positions — see commented `labels[~loss_mask] = -100` 
    in DMax dFactory/.../data_transform.py).

    Rollout is argmax by default (train_llada2_bd_oput.py: `token = semi_logits.argmax(...)`) — exactly what SPD commits at 
    temperature 0, so the model self-corrects errors it will actually make. `sample_rollout=True` is the sampled ablation. 
    DMax rolls out under `model.eval()` + no_grad (train_llada2_bd_oput.py lines 450-472): dropout OFF, matching inference. 
    Pass `rollout_decode_fn` to rerun whole conditioning+decode path in eval; without it rollout reuses masked-pass logits.

    Unlike DMax (per-example mask-vs-pred `flag`, one grad pass), this takes the MEAN of L_mask and L_pred: a sum would
    double the DLM's per-token translation scale against the AR arm's single-pass CE. 2x decoder cost buys lower variance.
    """
    valid_mask = valid_mask.bool()
    replay_pred = None
    if noise is None:  # `noise` replays a previous draw (mask AND rollout) so two conditionings differ in Omega alone
        t = sample_mask_ratio(clean_ids.shape, clean_ids.device, t_low=t_low, t_high=t_high)
        masked = (torch.rand_like(t) < t) & valid_mask
    else: t, masked, replay_pred = noise
    masked_ids = torch.where(masked, torch.full_like(clean_ids, int(mask_token_id)), clean_ids)

    mask_logits = decode_fn(masked_ids)
    with torch.no_grad():
        if replay_pred is not None: pred_ids = replay_pred
        else:
            rollout_logits = rollout_decode_fn(masked_ids) if rollout_decode_fn is not None else mask_logits
            if sample_rollout:
                probs = rollout_logits.softmax(dim=-1)
                rollout = torch.distributions.Categorical(probs=probs).sample()
            else: rollout = rollout_logits.argmax(dim=-1)
            pred_ids = torch.where(masked, rollout, masked_ids)

    pred_logits = decode_fn(pred_ids)
    loss_mask = valid_mask if loss_over_all_positions else masked
    mask_loss = masked_cross_entropy(mask_logits, clean_ids, loss_mask, label_smoothing=label_smoothing)
    pred_loss = masked_cross_entropy(pred_logits, clean_ids, loss_mask, label_smoothing=label_smoothing)
    with torch.no_grad():
        m = loss_mask.to(dtype=mask_logits.dtype)
        rows = 0.5 * sum(
            F.cross_entropy(lg.reshape(-1, lg.shape[-1]), clean_ids.reshape(-1), reduction="none").reshape_as(clean_ids) * m
            for lg in (mask_logits, pred_logits)
        ).sum(dim=1)
    # MEAN of 2 passes, not their sum: the pooled translation loss weighs OPUT rows and Mode-2a CB rows by
    # token count, and the AR arm's CE is single-pass — a summed two-pass OPUT would silently double the DLM's
    # per-token translation scale relative to both (halving CB's share on the DLM arm only).
    return OPUTOutput(
        loss=0.5 * (mask_loss + pred_loss), mask_loss=mask_loss, pred_loss=pred_loss, masked_positions=masked,
        rollout_tokens=pred_ids.detach(), row_loss_sum=rows, row_valid_count=m.sum(dim=1), noise=(t, masked, pred_ids.detach()),
    )


# ════════════════════════════════════════════════════════════════════════════
# Abstract DMax decoder: OPUT training + block decode
# ════════════════════════════════════════════════════════════════════════════

class OPUTBlockDiffusionDecoder(BlockDiffusionDecoder):
    '''BD3LM decoder with DMax's OPUT training and block decoding, over fixed encoder conditioning.

    Abstract on `_decode` (the custom-mask forward that training needs), `_decode_with_decoder_forward` (native HF forward 
    that carries a KV cache) and `_decoder_stack` (the stack the Ω injector hooks); the `MBart`/`MT5` bindings supply all 3. 
    Block decode is cold-start per call — no state crosses streaming strides.
    '''
    def oput_forward(
        self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor, labels: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None, t_low: float = 0.3, t_high: float = 0.8,
        loss_over_all_positions: bool = True, sample_rollout: bool = False,
        rollout_eval_mode: bool = True, eos_supervision: int | None = None,
        rollout_encode_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]] | None = None,
        omega_bias: torch.Tensor | None = None, label_smoothing: float = 0.0,
        noise: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        '''OPUT translation loss under fixed conditioning `enc_hidden`/`enc_mask`.

        The rollout corruption is sampled with dropout OFF: `rollout_encode_fn` re-encodes the whole conditioning
        path in eval; otherwise only the decoder is toggled to eval over the same `enc_hidden`.

        `omega_bias` (models.membership_gate) is FIXED conditioning: identical across both OPUT passes AND the rollout, 
        since it depends on BIO posteriors / encoder features, not target tokens. OPUT corrupts *target*, Ω conditions 
        the *input*, so Ω rides every decode here; its gradient into BIO logits flows via the grad-bearing passes only.
        '''
        x0, valid = self._prepare_x0(labels, decoder_input_ids=decoder_input_ids, eos_supervision=eos_supervision)
        rollout_decode_fn = None
        if rollout_encode_fn is not None:
            def rollout_decode_fn(noisy_ids: torch.Tensor) -> torch.Tensor:
                # Decoder in eval too (caller's rollout_encode_fn covers the encoder): whole path dropout-off.
                was_training = self.training
                self.eval()
                try:
                    r_enc, r_mask = rollout_encode_fn()
                    return self._bd3lm_logits(noisy_ids, x0, r_enc, r_mask, omega_bias=omega_bias)
                finally: self.train(was_training)
        elif rollout_eval_mode:
            # Decoder-only eval rollout: L_mask/L_pred below still share the train-mode `enc_hidden`, 
            # so their conditioning stays fixed.
            def rollout_decode_fn(noisy_ids: torch.Tensor) -> torch.Tensor:
                was_training = self.training
                self.eval()
                try: return self._bd3lm_logits(noisy_ids, x0, enc_hidden, enc_mask, omega_bias=omega_bias)
                finally: self.train(was_training)

        # Conditioning c must be fixed and complete across both passes; `_version` catches in-place mutation.
        enc_version = enc_hidden._version
        out = oput_two_pass_loss(
            clean_ids=x0, valid_mask=valid,
            decode_fn=lambda noisy_ids: self._bd3lm_logits(noisy_ids, x0, enc_hidden, enc_mask, omega_bias=omega_bias),
            mask_token_id=self.mask_token_id, t_low=t_low, t_high=t_high, loss_over_all_positions=loss_over_all_positions, 
            sample_rollout=sample_rollout, rollout_decode_fn=rollout_decode_fn, label_smoothing=label_smoothing, noise=noise
        )
        assert enc_hidden._version == enc_version, "OPUT conditioning mutated between passes (fixed c)"
        return {
            "translation_loss": out.loss, "oput_mask_loss": out.mask_loss.detach(),
            "oput_pred_loss": out.pred_loss.detach(), "oput_masked_fraction": out.masked_positions.float().mean().detach(),
            "row_loss_sum": out.row_loss_sum, "row_valid_count": out.row_valid_count, "noise": out.noise,
        }


    def remasked_logits(
        self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor,
        decoded_tokens: torch.Tensor, remask_positions: torch.Tensor, omega_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        '''Grad-bearing forward on a committed sequence with the gated slots re-masked (confidence-bound surrogate).

        Gated slots (confident-disagreement under the no-grad truncated decode) become `[MASK]`, every other committed token 
        stays in context; 1 block-causal forward gives each gated slot's live conditional belief with the slot still open. 
        Closer to commit-time than an all-`[MASK]` marginal (t = 1, outside OPUT's t ∈ [t_low, t_high]), and costs 1 forward, 
        not back-prop through the decode. No reference text enters the input, so P1 is preserved.
        '''
        remask = remask_positions.to(device=decoded_tokens.device, dtype=torch.bool).clone()
        remask[:, 0] = False  # BOS fixed
        token_ids = torch.where(remask, torch.full_like(decoded_tokens, int(self.mask_token_id)), decoded_tokens)
        return self._decode(token_ids, enc_hidden, enc_mask, omega_bias=omega_bias)


    # ── Block decode over a KV cache of the final blocks (backbone-agnostic) ──
    def _decode_with_decoder_forward(
        self, decoder_input_ids, enc_hidden, enc_mask, self_attn_mask, inputs_embeds=None,
        past_key_values=None, use_cache=False, cache_position=None, logits=True,
    ):
        # Backbone hook: the native HF decoder.forward (KV-cache capable) under a 4D self-attention mask;
        # returns (logits, past_key_values). `logits=False` skips the |V| head (a cache append reads no logits).
        raise NotImplementedError(f"{type(self).__name__} has no cache-capable decoder.forward")

    def _decoder_stack(self) -> torch.nn.Module:
        # Backbone hook: the HF decoder stack (T5Stack / MBartDecoder) the Ω cross-attention injector hooks into.
        raise NotImplementedError(f"{type(self).__name__} exposes no decoder stack")

    def _omega_injector(self):
        # The one injector of this decoder stack (shared with AR arm's hooks on same modules). Pre-hooks
        # are inert outside `with_omega`, so the custom `_decode` path (which adds Ω itself) is untouched.
        from models.membership_gate import CrossAttnOmegaInjector
        return CrossAttnOmegaInjector.attach(self._decoder_stack())

    def _block_decoder(self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor, omega_bias: torch.Tensor | None = None):
        '''`forward` / `finalize` for `block_diffusion_decode`, over a KV cache of the final blocks.

        Exact under block-causal attention, to floating-point accumulation order: Slots of block b see blocks <= b only, so the 
        cached K/V of final blocks + a forward over the block's own slots reproduce full-canvas forward, and slots past the block 
        never influence it. The memory's cross-attention K/V live in same cache and are projected ONCE, on 1st pass of block 0; 
        every later pass reads them back. Ω rides native forward via CrossAttnOmegaInjector — a score-time bias, so cached K/V 
        stay valid — and prefix is built under Ω too, as its hidden states depend on it. Exactness needs eval mode: in train mode, 
        2 paths would draw different dropout masks.
        '''
        assert not self.training, "the block decode is an eval-mode operation (dropout would break cache exactness)"
        state = {"past": None, "length": 0}
        injector = self._omega_injector() if omega_bias is not None else None

        def native(**kw):
            if injector is None: return self._decode_with_decoder_forward(**kw)
            with injector.with_omega(omega_bias): return self._decode_with_decoder_forward(**kw)

        def call(ids, soft, lo, hi, past, use_cache, logits):
            assert state["length"] == lo, "the block decode extends its cache in order"
            # All-attend over [final prefix | block]: bidirectional inside the block, causal across blocks.
            mask = torch.zeros((ids.shape[0], 1, hi - lo, hi), dtype=enc_hidden.dtype, device=ids.device)
            # Always embed here: the canvas embedding carries the [MASK] row (and mBART's scale) the backbone's own does not.
            embeds = self.embed_tokens(ids[:, lo:hi]) if soft is None else soft
            return native(
                decoder_input_ids=ids[:, lo:hi], inputs_embeds=embeds, enc_hidden=enc_hidden, 
                enc_mask=enc_mask, self_attn_mask=mask, past_key_values=past, use_cache=use_cache, 
                cache_position=torch.arange(lo, hi, dtype=torch.long, device=ids.device), logits=logits,
            )

        def forward(ids, soft, lo, hi):
            # HF attention writes a pass's keys into whichever cache it is handed, even at use_cache=False. So write into the 
            # real cache and crop provisional block's self-attention keys back off, rather than copying the prefix per pass. 
            # The memory's cross-attention K/V survive the crop (they are written once and then read-only via `is_updated`), 
            # so block 0's passes stop re-projecting the whole memory.
            logits, past = call(ids, soft, lo, hi, state["past"], use_cache=True, logits=True)
            past.self_attention_cache.crop(lo)
            state["past"] = past
            return logits

        def finalize(ids, lo, hi):
            # 1 hard pass over the finished block appends its final K/V. dInfer instead widens NEXT block's 1st forward to span 
            # [block b | block b+1] and writes block b's K/V from it (generate_uniform.py decode_uniform, need_cross_block_update), 
            # so upstream pays no separate pass and this port trades 1 sequential pass per block for a narrower 1st forward. Its 
            # logits are never read, so the |V| head is skipped.
            _, past = call(ids, None, lo, hi, state["past"], use_cache=True, logits=False)
            state["past"], state["length"] = past, hi

        return forward, finalize

    @torch.no_grad()
    def generate(
        self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor, max_length: int = 128, threshold: float = 0.5,
        spd_top_k: int = 1, spd_renormalize: bool = True, omega_bias: torch.Tensor | None = None,
    ) -> DecodeResult:
        # `omega_bias` is fixed conditioning: identical across every pass of this decode.
        ids = torch.full((enc_hidden.shape[0], int(max_length)), int(self.mask_token_id), dtype=torch.long, device=enc_hidden.device)
        ids[:, 0] = int(self.bos_index)
        forward, finalize = self._block_decoder(enc_hidden, enc_mask, omega_bias=omega_bias)
        return block_diffusion_decode(
            forward, finalize, self.embed_tokens, ids, self.mask_token_id, self.block_size, threshold,
            eos_token_id=self.eos_index, pad_token_id=self.pad_index, spd_top_k=spd_top_k, spd_renormalize=spd_renormalize,
        )
