'''DMax extension to the BD3LM substrate: OPUT training + SPD/DCD inference + the confidence-bound surrogates.

`OPUTBlockDiffusionDecoder` adds DMax's three mechanisms on top of `block_diffusion.BlockDiffusionDecoder`,
all over a precomputed encoder memory (`enc_hidden`/`enc_mask`); it stays abstract on `_decode`, so the concrete
mBART / mT5 bindings (models/unisign.py) supply only the backbone decode.

  - `oput_forward`   — OPUT two-pass training (mask + on-policy argmax corruption), trains self-correction.
  - `decode_spd_dcd` / `generate_spd_dcd` — SPD (renormalized soft state) + DCD (sliding-window commits).
  - `remasked_logits` — grad-bearing surrogate for the confidence-bound term.

References: DMax OPUT + SPD/DCD (train_llada2_bd_oput.py); dLLM A2D (arXiv 2602.22661).
'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import torch
from transformers.cache_utils import EncoderDecoderCache
from models.block_diffusion import BlockDiffusionDecoder, build_block_causal_mask
from infer.decode import SPDDecodeResult, spd_dcd_decode
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


def sample_mask_ratio(shape: tuple[int, int], device: torch.device, t_low: float = 0.0, t_high: float = 1.0) -> torch.Tensor:
    # Sample one OPUT/DMax noise level per sequence and broadcast over tokens.
    low, high = float(t_low), float(t_high)
    if high < low: raise ValueError("t_high must be >= t_low")
    batch, length = shape
    ratios = torch.empty((batch, 1), device=device).uniform_(low, high)
    return ratios.expand(batch, length)


def oput_two_pass_loss(
    clean_ids: torch.Tensor, valid_mask: torch.Tensor,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    mask_token_id: int, t_low: float = 0.3, t_high: float = 0.8,
    loss_over_all_positions: bool = True, sample_rollout: bool = False,
    rollout_module: torch.nn.Module | None = None,
    rollout_decode_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> OPUTOutput:
    """DMax-style OPUT over a fixed conditioning closure.

    `decode_fn` must close over a fixed, complete conditioning tensor. It is called twice: first on masked
    target tokens (L_mask), then on an on-policy target corruption derived from the rollout pass (L_pred). Both passes
    supervise recovery of `clean_ids` over all valid positions (DMax §3.1; the OPUT SFT transform leaves the loss
    un-restricted to masked positions — see the commented `labels[~loss_mask] = -100` in DMax dFactory/.../data_transform.py).

    On-policy rollout uses argmax by default. DMax's training loop (train_llada2_bd_oput.py: `token = semi_logits.argmax(...)`)
    corrupts with the greedy token, i.e. exactly what SPD commits at temperature 0 — so the model is trained to self-correct
    the same errors it will make at inference. `sample_rollout=True` keeps the sampled variant available as an ablation.

    Rollout fidelity: DMax computes the rollout under `model.eval()` + no_grad (train_llada2_bd_oput.py lines 450-472), i.e. 
    with dropout OFF — the corruption is sampled from the same distribution inference will see. For a conditional encoder-decoder, 
    pass `rollout_decode_fn` so the *conditioning path* is also re-run in eval mode; `rollout_module` is the older decoder-only
    fallback that only toggles the closure's existing module state.

    Note: unlike DMax (which gates mask-vs-pred per example via a `flag` and runs one grad pass), this sums L_mask + L_pred 
    each step. In expectation the two are equivalent up to a scale absorbed by the LR; summing trades 2x decoder forward cost 
    for lower gradient variance, acceptable at this model scale.
    """
    valid_mask = valid_mask.bool()
    t = sample_mask_ratio(clean_ids.shape, clean_ids.device, t_low=t_low, t_high=t_high)
    masked = (torch.rand_like(t) < t) & valid_mask
    masked_ids = torch.where(masked, torch.full_like(clean_ids, int(mask_token_id)), clean_ids)

    mask_logits = decode_fn(masked_ids)
    with torch.no_grad():
        if rollout_decode_fn is not None: rollout_logits = rollout_decode_fn(masked_ids)
        elif rollout_module is not None and rollout_module.training:
            rollout_module.eval()
            try: rollout_logits = decode_fn(masked_ids)
            finally: rollout_module.train()
        else: rollout_logits = mask_logits

        if sample_rollout:
            probs = rollout_logits.softmax(dim=-1)
            rollout = torch.distributions.Categorical(probs=probs).sample()
        else: rollout = rollout_logits.argmax(dim=-1)
        pred_ids = torch.where(masked, rollout, masked_ids)

    pred_logits = decode_fn(pred_ids)
    loss_mask = valid_mask if loss_over_all_positions else masked
    mask_loss = masked_cross_entropy(mask_logits, clean_ids, loss_mask)
    pred_loss = masked_cross_entropy(pred_logits, clean_ids, loss_mask)
    return OPUTOutput(
        loss=mask_loss + pred_loss, mask_loss=mask_loss, pred_loss=pred_loss,
        masked_positions=masked, rollout_tokens=pred_ids.detach(),
    )


# ════════════════════════════════════════════════════════════════════════════
# Abstract DMax decoder: OPUT training + SPD/DCD inference
# ════════════════════════════════════════════════════════════════════════════

class OPUTBlockDiffusionDecoder(BlockDiffusionDecoder):
    '''BD3LM decoder with DMax's OPUT training and SPD+DCD inference, over fixed encoder conditioning.

    Still abstract on `_decode`; the concrete `MBart`/`MT5` bindings supply it. DCD prefix KV-cache is shared here and only its 
    `_decode_with_decoder_forward` hook is backbone-specific (MBartDecoder vs T5Stack). BOTH backbones implement that hook and 
    are verified cached == no-cache at a block boundary (Δ<1e-4, `test_prefix_cache_matches_no_cache`): mT5's T5Stack is 
    cache_position-aware, and mBART uses `MBartScaledWordEmbedding` so the cached path scales token embeddings identically to 
    `_decode` (an earlier plain embedding made them disagree — the Δ≈0.5 cache bug, now fixed). `cache_type='none'` is the exact 
    default everywhere; `cache_type='prefix'` + `window_type='static'` turns the cache on.

    Extend BD3LM substrate (`block_diffusion.BlockDiffusionDecoder`) with 3 DMax/DCD mechanisms:
    - `oput_forward`: **OPUT** training (`oput_two_pass_loss`): mask the target, and additionally re-decode an on-policy 
      (argmax) corruption, supervising recovery of the clean target over all positions. Trains self-correction.
    - `decode_spd_dcd` / `generate_spd_dcd` — **SPD+DCD** inference: SPD carries a renormalized soft mask/token embedding 
      state across denoising steps; DCD's sliding window selects which masked slots to commit vs defer. Cold-start per
      call — no state crosses streaming strides.
    - `remasked_logits`: cheap grad-bearing forward used by the confidence-bound term instead of backprop via full decode.

    `[xt|x0]` BD3LM concatenation and block-diff mask match DMax's own block-diff training loop (`train_llada2_bd_oput.py`).
    '''
    def oput_forward(
        self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor, labels: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None, t_low: float = 0.3, t_high: float = 0.8,
        loss_over_all_positions: bool = True, sample_rollout: bool = False,
        rollout_eval_mode: bool = True, eos_supervision: int | None = None,
        rollout_encode_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]] | None = None,
        omega_bias: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        '''OPUT translation loss under fixed conditioning `enc_hidden`/`enc_mask`.

        The gradient-bearing L_mask/L_pred passes score `[noisy | x0]` against `enc_hidden`. The on-policy rollout
        corruption is sampled with dropout OFF (DMax eval-mode rollout): if `rollout_encode_fn` is given the whole
        conditioning path is re-encoded in eval (full fidelity); otherwise only the decoder is toggled to eval over
        the same `enc_hidden` (decoder-side fidelity).

        `omega_bias` (membership gate, models.membership_gate) is part of the FIXED conditioning: identical across
        both OPUT passes AND the rollout (it depends only on the BIO posteriors / encoder features, not the target
        tokens). OPUT corrupts the *target*; Ω conditions the *input* — so it belongs on every decode here, and its
        gradient (into the BIO logits) flows through the two grad-bearing passes, not the no-grad rollout.
        '''
        x0, valid = self._prepare_x0(labels, decoder_input_ids=decoder_input_ids, eos_supervision=eos_supervision)
        rollout_decode_fn = None
        if rollout_encode_fn is not None:
            def rollout_decode_fn(noisy_ids: torch.Tensor) -> torch.Tensor:
                # Eval mode for the decoder too (caller's rollout_encode_fn handles the encoder), so the whole
                # conditioning+decode path runs dropout-off — DMax's eval-mode rollout.
                was_training = self.training
                self.eval()
                try:
                    r_enc, r_mask = rollout_encode_fn()
                    return self._bd3lm_logits(noisy_ids, x0, r_enc, r_mask, omega_bias=omega_bias)
                finally: self.train(was_training)
        elif rollout_eval_mode:
            # DMax OPUT samples the on-policy corruption under model.eval() + no_grad. For this conditional AR port,
            # that means re-running the conditioning path as well as the decoder with dropout off; the gradient-bearing
            # L_mask/L_pred passes below still share the same train-mode `enc_hidden`, so their conditioning remains fixed.
            def rollout_decode_fn(noisy_ids: torch.Tensor) -> torch.Tensor:
                was_training = self.training
                self.eval()
                try: return self._bd3lm_logits(noisy_ids, x0, enc_hidden, enc_mask, omega_bias=omega_bias)
                finally: self.train(was_training)

        out = oput_two_pass_loss(
            clean_ids=x0, valid_mask=valid,
            decode_fn=lambda noisy_ids: self._bd3lm_logits(noisy_ids, x0, enc_hidden, enc_mask, omega_bias=omega_bias),
            mask_token_id=self.mask_token_id, t_low=t_low, t_high=t_high,
            loss_over_all_positions=loss_over_all_positions, sample_rollout=sample_rollout,
            rollout_module=None if rollout_decode_fn is not None else (self if rollout_eval_mode else None),
            rollout_decode_fn=rollout_decode_fn,
        )
        return {
            "translation_loss": out.loss, "oput_mask_loss": out.mask_loss.detach(),
            "oput_pred_loss": out.pred_loss.detach(), "oput_masked_fraction": out.masked_positions.float().mean().detach(),
        }


    def remasked_logits(
        self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor,
        decoded_tokens: torch.Tensor, remask_positions: torch.Tensor, omega_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        '''Grad-bearing forward on a committed sequence with the gated slots re-masked (confidence-bound surrogate).

        The on-policy gradient surrogate for confidence-bound term: gated slots (confident-disagreement under the no-grad 
        truncated decode) are replaced by `[MASK]` while every other committed token stays in context, then 1 block-causal 
        forward yields each gated slot's live conditional belief — exactly the distribution DCD would read had it deferred 
        those slots. This is strictly closer to the decode's commit-time distribution than an all-`[MASK]` marginal (which 
        corresponds to t = 1 corruption that OPUT's t ∈ [t_low, t_high] training never visits), and still costs 1 forward 
        instead of back-prop through the ~128-step decode. No reference text enters the input, so P1 is preserved.
        '''
        remask = remask_positions.to(device=decoded_tokens.device, dtype=torch.bool).clone()
        remask[:, 0] = False  # BOS fixed
        token_ids = torch.where(remask, torch.full_like(decoded_tokens, int(self.mask_token_id)), decoded_tokens)
        return self._decode(token_ids, enc_hidden, enc_mask, omega_bias=omega_bias)


    # ── DCD prefix KV-cache (backbone-agnostic; block-boundary exact) ─────────
    # The cache logic is shared; only `_decode_with_decoder_forward` (the native KV-cache-capable HF decoder.forward)
    # differs per backbone (MBartDecoder vs T5Stack), so each concrete decoder implements ONLY that hook.
    def _decode_with_decoder_forward(
        self, decoder_input_ids, enc_hidden, enc_mask, self_attn_mask, inputs_embeds=None,
        past_key_values=None, use_cache=False, cache_position=None,
    ):
        '''Backbone hook: run the native HF decoder.forward (KV-cache capable) with a 4D self-attention mask; return
        (logits, past_key_values). A backbone that does not implement this disables the prefix cache.'''
        raise NotImplementedError(f"{type(self).__name__} has no cache-capable decoder.forward; use cache_type='none'.")


    def _prefix_static_window_mask(self, batch_size, prefix_len, window_len, dtype, device):
        # All-attend window mask: a window inside one block (after a block-boundary prefix) sees the full committed
        # prefix + its own block bidirectionally — exactly the block-causal view at that point.
        mask = torch.zeros((window_len, prefix_len + window_len), dtype=dtype, device=device)
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, window_len, prefix_len + window_len)


    def _clone_encoder_decoder_cache(self, cache):
        legacy = cache.to_legacy_cache()
        cloned = tuple(tuple(t.detach().clone() for t in layer) for layer in legacy)
        return EncoderDecoderCache.from_legacy_cache(cloned)


    def _make_static_prefix_cache_logits_fn(self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor):
        '''A DCD `logits_fn(ids, soft_embeds, window)` that caches the committed prefix's K/V and forwards only the
        active window against it. Exact ONLY at block boundaries (BD3LM is bidirectional within a block, so a prefix
        that ends mid-block lacks K/V the full forward would expose) and when the window lies inside one block —
        both checked, falling back to the exact full `_decode` otherwise (mirrors DCD window_causal_decode).'''
        state: dict[str, object] = {"prefix_len": None, "prefix_tokens": None, "past": None}

        def logits_fn(ids: torch.Tensor, soft_embeds: torch.Tensor | None, window: tuple[int, int] | None = None) -> torch.Tensor:
            if window is None: return self._decode(ids, enc_hidden, enc_mask)
            left, right = int(window[0]), int(window[1])
            if left <= 0: return self._decode(ids, enc_hidden, enc_mask)
            if left % self.block_size != 0:
                # BD3LM is bidirectional within each block. If the cached prefix ends inside a block, 
                # those prefix K/V states were computed w/o later same-block tokens that the exact full 
                # forward would expose. Prefix cache is exact only at block boundaries; otherwise fall back. 
                # Ref: DCD window_causal_decode block-local cache split + DMax/dLLM BD3LM block-causal mask.
                return self._decode(ids, enc_hidden, enc_mask)

            prefix_tokens = ids[:, :left].detach()
            cached_tokens = state.get("prefix_tokens")
            needs_refresh = (
                state.get("past") is None
                or state.get("prefix_len") != left
                or not torch.equal(cached_tokens, prefix_tokens.cpu())
            )
            if needs_refresh:
                prefix_mask = build_block_causal_mask(ids.shape[0], left, self.block_size, enc_hidden.dtype, ids.device)
                with torch.no_grad():
                    _, past = self._decode_with_decoder_forward(
                        decoder_input_ids=prefix_tokens,
                        enc_hidden=enc_hidden, enc_mask=enc_mask,
                        self_attn_mask=prefix_mask, use_cache=True,
                        cache_position=torch.arange(left, dtype=torch.long, device=ids.device),
                    )
                state["prefix_len"] = left
                state["prefix_tokens"] = prefix_tokens.cpu().clone()
                state["past"] = past

            if left // self.block_size != (right - 1) // self.block_size:
                # The all-attend window mask below is only valid when the window lies inside one attention block 
                # (bidirectional within block, full view of the committed prefix). A spanning window would let an 
                # earlier block attend into a later one — fall back to the exact full block-causal forward instead.
                return self._decode(ids, enc_hidden, enc_mask)

            window_ids = ids[:, left:right]
            window_embeds = soft_embeds[:, left:right] if soft_embeds is not None else None
            window_mask = self._prefix_static_window_mask(ids.shape[0], left, right - left, enc_hidden.dtype, ids.device)
            logits_window, _ = self._decode_with_decoder_forward(
                decoder_input_ids=window_ids, inputs_embeds=window_embeds,
                enc_hidden=enc_hidden, enc_mask=enc_mask,
                self_attn_mask=window_mask, past_key_values=self._clone_encoder_decoder_cache(state["past"]),
                use_cache=False, cache_position=torch.arange(left, right, dtype=torch.long, device=ids.device),
            )
            full_logits = torch.zeros((*ids.shape, logits_window.shape[-1]), dtype=logits_window.dtype, device=ids.device)
            full_logits[:, left:right] = logits_window
            return full_logits

        logits_fn.supports_dcd_cache = True  # type: ignore[attr-defined]
        return logits_fn


    def decode_spd_dcd(
        self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor,
        max_length: int = 128, diffusion_steps: int = 64, tau_dec: float = 0.75, top_k: int = 1,
        spd_renormalize: bool = True, spd_revision: bool = True, temperature: float = 0.0,
        window_length: int | None = None, max_window_length: int | None = None, window_type: str = "sliding",
        decode_algo: str = "threshold", decode_param: int | float | None = None, sample_top_k: int | None = None,
        top_p: float | None = None, cache_type: str = "none", refresh_count: int = 16,
        omega_bias: torch.Tensor | None = None,
    ) -> SPDDecodeResult:
        batch = enc_hidden.shape[0]
        token_ids = torch.full(
            (batch, int(max_length)), int(self.mask_token_id),
            dtype=torch.long, device=enc_hidden.device
        )
        token_ids[:, 0] = int(self.bos_index)

        if cache_type == "prefix" and window_type == "static" and omega_bias is None:
            logits_fn = self._make_static_prefix_cache_logits_fn(enc_hidden, enc_mask)
        elif cache_type == "prefix" and window_type == "static":
            # The prefix KV-cache path does not carry the membership-gate bias (its cached forward bypasses the
            # cross-attn-bias arg). When Ω is active — the Mode-2a CB decode under the gate — fall back to the
            # exact no-cache path so the gated decode is CORRECT (slower, but CB is training-time machinery and
            # decodes a short canvas). Cold-start streaming inference already uses cache_type='none'.
            def logits_fn(ids: torch.Tensor, soft_embeds: torch.Tensor | None) -> torch.Tensor:
                return self._decode(ids, enc_hidden, enc_mask, inputs_embeds=soft_embeds, omega_bias=omega_bias)
        elif cache_type == "none":
            def logits_fn(ids: torch.Tensor, soft_embeds: torch.Tensor | None) -> torch.Tensor:
                # inputs_embeds=None -> _decode embeds `ids` itself; otherwise it uses the SPD soft-embedding mixture.
                # omega_bias is fixed conditioning: identical across every denoising step of this single decode.
                return self._decode(ids, enc_hidden, enc_mask, inputs_embeds=soft_embeds, omega_bias=omega_bias)
        else: raise NotImplementedError(
            "Only cache_type='none' or cache_type='prefix'+window_type='static' are implemented (both verified "
            "for the mT5 and mBART decoders); sliding/dual cache requires a separate verified port."
        )
        return spd_dcd_decode(
            logits_fn=logits_fn, embedding_layer=self.embed_tokens, initial_token_ids=token_ids,
            mask_token_id=self.mask_token_id, steps=diffusion_steps, threshold=tau_dec,
            eos_token_id=self.eos_index, pad_token_id=self.pad_index, temperature=temperature,
            top_k=top_k, spd_renormalize=spd_renormalize, spd_revision=spd_revision,
            window_length=window_length or self.block_size, max_window_length=max_window_length,
            window_type=window_type, decode_algo=decode_algo, decode_param=decode_param,
            sample_top_k=sample_top_k, top_p=top_p, cache_type=cache_type, refresh_count=refresh_count,
            # This decoder is block-causal, so the DCD window must not span attention blocks (see spd_dcd_decode docstring).
            block_size=self.block_size,
        )


    @torch.no_grad()
    def generate_spd_dcd(self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor, **kwargs) -> SPDDecodeResult:
        return self.decode_spd_dcd(enc_hidden=enc_hidden, enc_mask=enc_mask, **kwargs)
