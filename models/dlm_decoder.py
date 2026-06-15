from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math

import torch
import torch.nn.functional as F
from transformers.cache_utils import EncoderDecoderCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from block_diffusion import BlockDiffusionDecoder, build_bd3lm_mask, build_block_causal_mask, supervise_trailing_eos
from infer.decode import SPDDecodeResult, spd_dcd_decode
from train.losses import masked_cross_entropy


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

    `decode_fn` must close over a fixed, complete visual conditioning tensor. It is called twice: first on masked
    target tokens (L_mask), then on an on-policy target corruption derived from the rollout pass (L_pred). Both passes
    supervise recovery of `clean_ids` over all valid positions (DMax §3.1; the OPUT SFT transform leaves the loss
    un-restricted to masked positions — see the commented `labels[~loss_mask] = -100` in DMax dFactory/.../data_transform.py).

    On-policy rollout uses argmax by default. DMax's training loop (train_llada2_bd_oput.py: `token = semi_logits.argmax(...)`)
    corrupts with the greedy token, i.e. exactly what SPD commits at temperature 0 — so the model is trained to self-correct
    the same errors it will make at inference. The prompt §6.2 says "sample"; per the spec's own rule that the DMax code is
    ground truth on divergence, argmax is used. `sample_rollout=True` keeps the sampled variant available as an ablation.

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


class OPUTBlockDiffusionDecoder(BlockDiffusionDecoder):
    """Block-diffusion mBART decoder with DMax's OPUT training and SPD+DCD inference.

    Extends the BD3LM substrate (`block_diffusion.BlockDiffusionDecoder`) with the
    three DMax/DCD mechanisms, all under fixed visual conditioning:

    - `oput_forward` — **OPUT** training (`oput_two_pass_loss`): mask the target,
      and additionally re-decode an on-policy (argmax) corruption, supervising
      recovery of the clean target over all positions. Trains self-correction.
    - `decode_spd_dcd` / `generate_spd_dcd` — **SPD+DCD** inference: SPD carries a
      renormalized soft mask/token embedding state across denoising steps; DCD's
      sliding window selects which masked slots to commit vs defer. Cold-start per
      call — no state crosses streaming strides.
    - `truncated_marginal_logits` — one cheap grad-bearing forward used by the
      confidence-bound term instead of back-prop through the full decode.

    The `[xt | x0]` BD3LM concatenation and block-diffusion mask match DMax's own
    block-diffusion training loop (`train_llada2_bd_oput.py`).
    """
    def _decode_from_embeds(
        self, token_ids: torch.Tensor, raw_inputs_embeds: torch.Tensor,
        enc_hidden: torch.Tensor, enc_mask: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, tgt_len = token_ids.shape
        device = token_ids.device
        dtype = enc_hidden.dtype

        hidden = raw_inputs_embeds * self.embed_scale
        positions = self.mbart_decoder.embed_positions(token_ids)
        hidden = hidden + positions
        hidden = self.mbart_decoder.layernorm_embedding(hidden)
        hidden = F.dropout(hidden, p=self.mbart_decoder.dropout, training=self.training)

        if self_attn_mask is None: self_mask = build_block_causal_mask(batch_size, tgt_len, self.block_size, dtype, device)
        else: self_mask = self_attn_mask.to(dtype=dtype, device=device)
        cross_mask = AttentionMaskConverter._expand_mask(enc_mask, dtype, tgt_len=tgt_len) if enc_mask is not None else None

        for layer in self.mbart_decoder.layers:
            hidden = layer(
                hidden, attention_mask=self_mask,
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=cross_mask,
            )[0]
        if getattr(self.mbart_decoder, "layer_norm", None) is not None: hidden = self.mbart_decoder.layer_norm(hidden)
        return self.lm_head(hidden)


    def _decode_with_decoder_forward(
        self, decoder_input_ids: torch.Tensor,
        enc_hidden: torch.Tensor, enc_mask: torch.Tensor,
        self_attn_mask: torch.Tensor, inputs_embeds: torch.Tensor | None = None,
        past_key_values=None, use_cache: bool = False, cache_position: torch.Tensor | None = None,
    ):
        cross_mask = enc_mask if enc_mask is not None else None
        out = self.mbart_decoder(
            input_ids=None if inputs_embeds is not None else decoder_input_ids,
            inputs_embeds=inputs_embeds, attention_mask=self_attn_mask,
            encoder_hidden_states=enc_hidden, encoder_attention_mask=cross_mask,
            past_key_values=past_key_values, use_cache=use_cache,
            cache_position=cache_position, return_dict=True,
        )
        return self.lm_head(out.last_hidden_state), out.past_key_values


    def _prefix_static_window_mask(
        self, batch_size: int, prefix_len: int, window_len: int,
        dtype: torch.dtype, device: torch.device,
    ) -> torch.Tensor:
        mask = torch.zeros((window_len, prefix_len + window_len), dtype=dtype, device=device)
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, window_len, prefix_len + window_len)


    def _clone_encoder_decoder_cache(self, cache):
        legacy = cache.to_legacy_cache()
        cloned = tuple(tuple(t.detach().clone() for t in layer) for layer in legacy)
        return EncoderDecoderCache.from_legacy_cache(cloned)


    def _make_static_prefix_cache_logits_fn(self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor):
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
            window_embeds = None
            if soft_embeds is not None: window_embeds = soft_embeds[:, left:right]
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


    def _prepare_x0(
        self, labels: torch.Tensor, decoder_input_ids: torch.Tensor | None = None, 
        ignore_index: int = -100, eos_supervision: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = labels.shape[0]
        x0 = labels.clone()
        x0[x0 == ignore_index] = self.pad_index
        if decoder_input_ids is not None: bos = decoder_input_ids[:, :1].to(device=labels.device)
        else: bos = torch.full((batch, 1), self.bos_index, dtype=x0.dtype, device=labels.device)

        x0 = torch.cat([bos, x0], dim=1)
        valid = x0 != self.pad_index
        if self.ignore_bos: valid[:, 0] = False

        aligned_len = max(1, math.ceil(x0.shape[1] / self.block_size)) * self.block_size
        if x0.shape[1] < aligned_len:
            x0 = F.pad(x0, (0, aligned_len - x0.shape[1]), value=self.pad_index)
            valid = F.pad(valid, (0, aligned_len - valid.shape[1]), value=False)
        # Supervised EOS tail after [.., eos, lang] (dLLM AppendEOSBlockWrapper / DMax 32-trailing-eos):
        # without it, slots past the sentence end are never trained and decode to confident garbage
        # before EOS commits, which the commit gate then reads as hardened. See block_diffusion.supervise_trailing_eos.
        return supervise_trailing_eos(
            x0, valid, pad_index=self.pad_index, eos_index=self.eos_index,
            max_tokens=self.eos_supervision_tokens if eos_supervision is None else int(eos_supervision),
        )


    def _bd3lm_logits(
        self, noisy_ids: torch.Tensor, clean_ids: torch.Tensor,
        enc_hidden: torch.Tensor, enc_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = clean_ids.shape
        bd3lm_mask = build_bd3lm_mask(length, self.block_size, enc_hidden.dtype, clean_ids.device)
        base_pos = torch.arange(length, device=clean_ids.device).unsqueeze(0).expand(batch, -1)
        position_ids = torch.cat([base_pos, base_pos], dim=1)
        logits = self._decode(
            torch.cat([noisy_ids, clean_ids], dim=1), enc_hidden, enc_mask,
            self_attn_mask=bd3lm_mask, position_ids=position_ids,
        )
        return logits[:, :length]


    def oput_forward(
        self, input_feature: torch.Tensor, input_lengths: torch.Tensor, labels: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None, t_low: float = 0.3, t_high: float = 0.8,
        loss_over_all_positions: bool = True, sample_rollout: bool = False,
        rollout_eval_mode: bool = True, eos_supervision: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """OPUT translation loss for one complete-conditioning (Mode 1/3) batch.

        Encodes the (fixed, complete) visual features once, builds the BOS-prefixed clean target `x0`, and runs
        `oput_two_pass_loss` over a closure that scores a noisy target via the `[noisy | x0]` BD3LM forward.
        The visual conditioning is byte-identical across both OPUT passes (asserted by the closure capturing
        `enc_hidden`). Returns `translation_loss` plus detached per-term diagnostics.
        """
        x0, valid = self._prepare_x0(labels, decoder_input_ids=decoder_input_ids, eos_supervision=eos_supervision)
        enc_hidden, enc_mask = self._encode_visual(input_feature, input_lengths)
        rollout_decode_fn = None
        if rollout_eval_mode:
            # DMax OPUT samples the on-policy corruption under model.eval() + no_grad. For this conditional mBART port, 
            # that means re-running the visual conditioning path as well as the decoder with dropout off; the gradient-bearing 
            # L_mask/L_pred passes below still share the same train-mode `enc_hidden`, so their conditioning remains fixed.
            def rollout_decode_fn(noisy_ids: torch.Tensor) -> torch.Tensor:
                was_training = self.training
                self.eval()
                try:
                    rollout_enc_hidden, rollout_enc_mask = self._encode_visual(input_feature, input_lengths)
                    return self._bd3lm_logits(noisy_ids, x0, rollout_enc_hidden, rollout_enc_mask)
                finally: self.train(was_training)

        out = oput_two_pass_loss(
            clean_ids=x0, valid_mask=valid,
            decode_fn=lambda noisy_ids: self._bd3lm_logits(noisy_ids, x0, enc_hidden, enc_mask),
            mask_token_id=self.mask_token_id, t_low=t_low, t_high=t_high,
            loss_over_all_positions=loss_over_all_positions,
            sample_rollout=sample_rollout,
            rollout_module=None if rollout_decode_fn is not None else (self if rollout_eval_mode else None),
            rollout_decode_fn=rollout_decode_fn,
        )
        return {
            "translation_loss": out.loss, "oput_mask_loss": out.mask_loss.detach(),
            "oput_pred_loss": out.pred_loss.detach(), "oput_masked_fraction": out.masked_positions.float().mean().detach(),
        }

    def truncated_marginal_logits(self, input_feature: torch.Tensor, input_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
        """One grad-bearing forward giving p(token_j | visual, all-masked target).

        Legacy/ablation variant of the confidence-bound gradient surrogate. The all-`[MASK]` input corresponds 
        to t = 1 corruption, which OPUT's t ∈ [t_low, t_high] training never visits, and its marginal is not 
        the conditional the decode's gate read — prefer `remasked_logits`, which re-masks only the gated slots 
        inside the committed decode (the DCD deferral counterfactual). Kept for the ablation comparison.
        """
        enc_hidden, enc_mask = self._encode_visual(input_feature, input_lengths)
        batch = input_feature.shape[0]
        token_ids = torch.full((batch, int(seq_len)), int(self.mask_token_id), dtype=torch.long, device=input_feature.device)
        token_ids[:, 0] = int(self.bos_index)
        return self._decode(token_ids, enc_hidden, enc_mask)


    def remasked_logits(
        self, input_feature: torch.Tensor, input_lengths: torch.Tensor,
        decoded_tokens: torch.Tensor, remask_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Grad-bearing forward on a decoded sequence with selected slots re-masked.

        The on-policy gradient surrogate for the confidence-bound term: the gated slots (confident-disagreement under the no-grad 
        truncated decode) are replaced by `[MASK]` while every other committed token stays in context, then 1 block-causal forward 
        yields each gated slot's live conditional belief — exactly the distribution DCD would read had it deferred those slots. 
        This is strictly closer to the decode's commit-time distribution than an all-`[MASK]` marginal (which corresponds to t = 1 
        corruption that OPUT's t ∈ [t_low, t_high] training never visits), and still costs 1 forward instead of back-prop through 
        the ~64-step decode. No reference text enters the input, so P1 is preserved.
        """
        enc_hidden, enc_mask = self._encode_visual(input_feature, input_lengths)
        remask = remask_positions.to(device=decoded_tokens.device, dtype=torch.bool).clone()
        remask[:, 0] = False  # BOS is fixed, never re-masked
        token_ids = torch.where(remask, torch.full_like(decoded_tokens, int(self.mask_token_id)), decoded_tokens)
        return self._decode(token_ids, enc_hidden, enc_mask)


    def decode_spd_dcd(
        self, input_feature: torch.Tensor, input_lengths: torch.Tensor, 
        max_length: int = 128, diffusion_steps: int = 64, tau_dec: float = 0.75, top_k: int = 1, 
        spd_renormalize: bool = True, spd_revision: bool = True, temperature: float = 0.0,
        window_length: int | None = None, max_window_length: int | None = None, window_type: str = "sliding",
        decode_algo: str = "threshold", decode_param: int | float | None = None, sample_top_k: int | None = None, 
        top_p: float | None = None, cache_type: str = "none", refresh_count: int = 16, window_block_clip: bool = True,
    ) -> SPDDecodeResult:
        enc_hidden, enc_mask = self._encode_visual(input_feature, input_lengths)
        batch = input_feature.shape[0]
        token_ids = torch.full(
            (batch, int(max_length)), int(self.mask_token_id),
            dtype=torch.long, device=input_feature.device,
        )
        token_ids[:, 0] = int(self.bos_index)

        if cache_type == "prefix" and window_type == "static": 
            logits_fn = self._make_static_prefix_cache_logits_fn(enc_hidden, enc_mask)
        elif cache_type == "none":
            def logits_fn(ids: torch.Tensor, soft_embeds: torch.Tensor | None) -> torch.Tensor:
                if soft_embeds is None: return self._decode(ids, enc_hidden, enc_mask)
                return self._decode_from_embeds(ids, soft_embeds, enc_hidden, enc_mask)
        else: raise NotImplementedError(
            "Only cache_type='prefix' with window_type='static' is currently implemented for the mBART DLM cache; "
            "sliding/dual cache requires a separate verified port."
        )
        return spd_dcd_decode(
            logits_fn=logits_fn, embedding_layer=self.embed_tokens, initial_token_ids=token_ids, 
            mask_token_id=self.mask_token_id, steps=diffusion_steps, threshold=tau_dec,
            eos_token_id=self.eos_index, pad_token_id=self.pad_index, temperature=temperature, 
            top_k=top_k, spd_renormalize=spd_renormalize, spd_revision=spd_revision,
            window_length=window_length or self.block_size, max_window_length=max_window_length,
            window_type=window_type, decode_algo=decode_algo, decode_param=decode_param,
            sample_top_k=sample_top_k, top_p=top_p, cache_type=cache_type, refresh_count=refresh_count,
            # This decoder is block-causal, so the DCD window must not span
            # attention blocks (see spd_dcd_decode docstring). False = ablation.
            block_size=self.block_size if window_block_clip else None,
        )


    @torch.no_grad()
    def generate_spd_dcd(
        self, input_feature: torch.Tensor, input_lengths: torch.Tensor, 
        max_length: int = 128, diffusion_steps: int = 64, tau_dec: float = 0.75, top_k: int = 1, 
        spd_renormalize: bool = True, spd_revision: bool = True, temperature: float = 0.0,
        window_length: int | None = None, max_window_length: int | None = None, window_type: str = "sliding",
        decode_algo: str = "threshold", decode_param: int | float | None = None,  sample_top_k: int | None = None, 
        top_p: float | None = None, cache_type: str = "none", refresh_count: int = 16, window_block_clip: bool = True,
    ) -> SPDDecodeResult:
        return self.decode_spd_dcd(
            input_feature=input_feature, input_lengths=input_lengths,
            max_length=max_length, diffusion_steps=diffusion_steps, tau_dec=tau_dec, top_k=top_k,
            spd_renormalize=spd_renormalize, spd_revision=spd_revision, temperature=temperature,
            window_length=window_length, max_window_length=max_window_length, window_type=window_type,
            decode_algo=decode_algo, decode_param=decode_param, sample_top_k=sample_top_k,
            top_p=top_p, cache_type=cache_type, refresh_count=refresh_count, window_block_clip=window_block_clip,
        )
