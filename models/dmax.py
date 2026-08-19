'''DMax extension to the BD3LM substrate: OPUT training + SPD/DCD inference + the confidence-bound surrogates.

`OPUTBlockDiffusionDecoder` adds DMax's 3 mechanisms on top of `block_diffusion.BlockDiffusionDecoder`, over a
precomputed encoder memory (`enc_hidden`/`enc_mask`); it stays abstract on `_decode`, so the mBART / mT5 bindings
(models/unisign.py) supply only the backbone decode.

  - `oput_forward`   — OPUT two-pass training (mask + on-policy argmax corruption); trains self-correction.
  - `decode_spd_dcd` / `generate_spd_dcd` — SPD (renormalized soft state across denoising steps) + DCD (sliding
    window choosing which masked slots to commit vs defer).
  - `remasked_logits` — grad-bearing surrogate for the confidence-bound term.

References: DMax OPUT + SPD/DCD (train_llada2_bd_oput.py); dLLM A2D (arXiv 2602.22661).
'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
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
    # Detached per-row (summed token loss, valid-token count). Lets one merged OPUT call report the per-mode
    # breakdown the split-by-mode call used to give, without paying for a second decoder pass per mode group.
    row_loss_sum: torch.Tensor | None = None
    row_valid_count: torch.Tensor | None = None


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
) -> OPUTOutput:
    """DMax-style OPUT over a fixed conditioning closure.

    `decode_fn` must close over fixed, complete conditioning. Called twice: on masked target tokens (L_mask), then on
    an on-policy corruption from the rollout pass (L_pred). Both supervise recovery of `clean_ids` over all valid
    positions (DMax §3.1; the OPUT SFT transform leaves the loss un-restricted to masked positions — see the
    commented `labels[~loss_mask] = -100` in DMax dFactory/.../data_transform.py).

    Rollout is argmax by default (train_llada2_bd_oput.py: `token = semi_logits.argmax(...)`) — exactly what SPD
    commits at temperature 0, so the model self-corrects the errors it will actually make. `sample_rollout=True` is
    the sampled ablation. DMax rolls out under `model.eval()` + no_grad (train_llada2_bd_oput.py lines 450-472):
    dropout OFF, matching inference. Pass `rollout_decode_fn` to re-run the whole conditioning+decode path in eval;
    without it the rollout reuses the masked-pass logits.

    Unlike DMax (per-example mask-vs-pred `flag`, one grad pass), this sums L_mask + L_pred: equivalent in
    expectation up to a scale the LR absorbs, trading 2x decoder cost for lower gradient variance.
    """
    valid_mask = valid_mask.bool()
    t = sample_mask_ratio(clean_ids.shape, clean_ids.device, t_low=t_low, t_high=t_high)
    masked = (torch.rand_like(t) < t) & valid_mask
    masked_ids = torch.where(masked, torch.full_like(clean_ids, int(mask_token_id)), clean_ids)

    mask_logits = decode_fn(masked_ids)
    with torch.no_grad():
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
        rollout_tokens=pred_ids.detach(), row_loss_sum=rows, row_valid_count=m.sum(dim=1),
    )


# ════════════════════════════════════════════════════════════════════════════
# Abstract DMax decoder: OPUT training + SPD/DCD inference
# ════════════════════════════════════════════════════════════════════════════

class OPUTBlockDiffusionDecoder(BlockDiffusionDecoder):
    '''BD3LM decoder with DMax's OPUT training and SPD+DCD inference, over fixed encoder conditioning.

    Still abstract on `_decode`; the `MBart`/`MT5` bindings supply it. The DCD prefix KV-cache is shared here; only
    `_decode_with_decoder_forward` is backbone-specific (MBartDecoder vs T5Stack). Both backbones implement it and
    are verified cached == no-cache at a block boundary (Δ<1e-4, `test_prefix_cache_matches_no_cache`): mT5's T5Stack
    is cache_position-aware, and mBART's `MBartScaledWordEmbedding` makes the cached path scale token embeddings
    identically to `_decode` (a plain embedding caused the Δ≈0.5 cache bug, now fixed). `cache_type='none'` is the
    default everywhere; `cache_type='prefix'` + `window_type='static'` turns the cache on.

    SPD/DCD decoding is cold-start per call — no state crosses streaming strides. `[xt|x0]` concatenation and the
    block-diff mask match DMax's own training loop (`train_llada2_bd_oput.py`).
    '''
    def oput_forward(
        self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor, labels: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None, t_low: float = 0.3, t_high: float = 0.8,
        loss_over_all_positions: bool = True, sample_rollout: bool = False,
        rollout_eval_mode: bool = True, eos_supervision: int | None = None,
        rollout_encode_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]] | None = None,
        omega_bias: torch.Tensor | None = None, label_smoothing: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        '''OPUT translation loss under fixed conditioning `enc_hidden`/`enc_mask`.

        The rollout corruption is sampled with dropout OFF: `rollout_encode_fn` re-encodes the whole conditioning
        path in eval; otherwise only the decoder is toggled to eval over the same `enc_hidden`.

        `omega_bias` (models.membership_gate) is FIXED conditioning: identical across both OPUT passes AND the
        rollout, since it depends on the BIO posteriors / encoder features, not the target tokens. OPUT corrupts the
        *target*, Ω conditions the *input*, so Ω rides every decode here; its gradient into the BIO logits flows
        through the grad-bearing passes only.
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
            mask_token_id=self.mask_token_id, t_low=t_low, t_high=t_high,
            loss_over_all_positions=loss_over_all_positions, sample_rollout=sample_rollout,
            rollout_decode_fn=rollout_decode_fn, label_smoothing=label_smoothing
        )
        assert enc_hidden._version == enc_version, "OPUT conditioning mutated between passes (fixed c)"
        return {
            "translation_loss": out.loss, "oput_mask_loss": out.mask_loss.detach(),
            "oput_pred_loss": out.pred_loss.detach(), "oput_masked_fraction": out.masked_positions.float().mean().detach(),
            "row_loss_sum": out.row_loss_sum, "row_valid_count": out.row_valid_count,
        }


    def remasked_logits(
        self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor,
        decoded_tokens: torch.Tensor, remask_positions: torch.Tensor, omega_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        '''Grad-bearing forward on a committed sequence with the gated slots re-masked (confidence-bound surrogate).

        Gated slots (confident-disagreement under the no-grad truncated decode) become `[MASK]`, every other committed
        token stays in context; one block-causal forward gives each gated slot's live conditional belief — what DCD
        would read had it deferred them. Closer to commit-time than an all-`[MASK]` marginal (t = 1, outside OPUT's
        t ∈ [t_low, t_high]), and costs 1 forward, not back-prop through the ~128-step decode. No reference text
        enters the input, so P1 is preserved.
        '''
        remask = remask_positions.to(device=decoded_tokens.device, dtype=torch.bool).clone()
        remask[:, 0] = False  # BOS fixed
        token_ids = torch.where(remask, torch.full_like(decoded_tokens, int(self.mask_token_id)), decoded_tokens)
        return self._decode(token_ids, enc_hidden, enc_mask, omega_bias=omega_bias)


    # ── DCD prefix KV-cache (backbone-agnostic; block-boundary exact) ─────────
    def _decode_with_decoder_forward(
        self, decoder_input_ids, enc_hidden, enc_mask, self_attn_mask, inputs_embeds=None,
        past_key_values=None, use_cache=False, cache_position=None,
    ):
        # Backbone hook: run the native HF decoder.forward (KV-cache capable) with a 4D self-attention mask; 
        # return (logits, past_key_values). A backbone that does not implement this disables the prefix cache.
        raise NotImplementedError(f"{type(self).__name__} has no cache-capable decoder.forward; use cache_type='none'.")

    def _decoder_stack(self) -> torch.nn.Module:
        # Backbone hook: the HF decoder stack (T5Stack / MBartDecoder) `_decode_with_decoder_forward` runs — 
        # also the module the Ω cross-attention injector hooks into.
        raise NotImplementedError(f"{type(self).__name__} exposes no decoder stack; the gated prefix cache is unavailable.")

    def _omega_injector(self):
        # Lazy CrossAttnOmegaInjector, shared by every gated cached decode. Pre-hooks are inert outside `with_omega`,
        # so the custom `_decode` path (which adds Ω itself) is untouched; tight scoping avoids double application.
        inj = getattr(self, "_omega_injector_obj", None)
        if inj is None:
            from models.membership_gate import CrossAttnOmegaInjector
            inj = CrossAttnOmegaInjector(self._decoder_stack())
            self._omega_injector_obj = inj
        return inj


    def _prefix_static_window_mask(self, batch_size, prefix_len, window_len, dtype, device):
        # All-attend mask: inside one block, after a block-boundary prefix, this IS the block-causal view.
        mask = torch.zeros((window_len, prefix_len + window_len), dtype=dtype, device=device)
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, window_len, prefix_len + window_len)


    def _clone_encoder_decoder_cache(self, cache):
        legacy = cache.to_legacy_cache()
        cloned = tuple(tuple(t.detach().clone() for t in layer) for layer in legacy)
        return EncoderDecoderCache.from_legacy_cache(cloned)


    def _make_static_prefix_cache_logits_fn(
        self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor, omega_bias: torch.Tensor | None = None,
    ):
        '''A DCD `logits_fn(ids, soft_embeds, window)` that caches the committed prefix's K/V and forwards only the
        active window against it. Exact ONLY when (a) the prefix ends on a block boundary, (b) the window lies inside
        one block, AND (c) the window reaches that block's end — BD3LM is bidirectional within a block, so a prefix or
        window stopping mid-block lacks K/V the full forward exposes. All three are checked, falling back to the exact
        full `_decode` otherwise (mirrors DCD window_causal_decode). A final partial block (window at the sequence
        end) is exact too: no keys exist past the sequence in either path.

        Ω-compatible: `omega_bias` biases cross-attn SCORES only, so cached K/V are reusable as-is; it rides the native 
        forward via CrossAttnOmegaInjector, the KV-cache-safe mechanism the AR arm uses. Prefix build AND window pass 
        both run under Ω (prefix hiddens depend on Ω via their own cross-attention), matching the gated no-cache `_decode`.'''
        state: dict[str, object] = {"prefix_len": None, "prefix_tokens": None, "past": None}
        injector = self._omega_injector() if omega_bias is not None else None

        def _native_forward(**kw):
            # Fallback `_decode` calls run outside it and take omega_bias explicitly.
            if injector is None: return self._decode_with_decoder_forward(**kw)
            with injector.with_omega(omega_bias): # Tight with_omega scope: hooks bias ONLY this native forward. 
                return self._decode_with_decoder_forward(**kw)

        def logits_fn(ids: torch.Tensor, soft_embeds: torch.Tensor | None, window: tuple[int, int] | None = None) -> torch.Tensor:
            if window is None: return self._decode(ids, enc_hidden, enc_mask, inputs_embeds=soft_embeds, omega_bias=omega_bias)
            left, right = int(window[0]), int(window[1])
            if left <= 0: return self._decode(ids, enc_hidden, enc_mask, inputs_embeds=soft_embeds, omega_bias=omega_bias)
            if left % self.block_size != 0:
                # (a) A prefix ending inside a block has K/V computed without the later same-block tokens the full
                # forward exposes. Ref: DCD window_causal_decode block-local cache split.
                return self._decode(ids, enc_hidden, enc_mask, inputs_embeds=soft_embeds, omega_bias=omega_bias)

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
                    _, past = _native_forward(
                        decoder_input_ids=prefix_tokens,
                        enc_hidden=enc_hidden, enc_mask=enc_mask,
                        self_attn_mask=prefix_mask, use_cache=True,
                        cache_position=torch.arange(left, dtype=torch.long, device=ids.device),
                    )
                state["prefix_len"] = left
                state["prefix_tokens"] = prefix_tokens.cpu().clone()
                state["past"] = past

            if left // self.block_size != (right - 1) // self.block_size:
                # (b) A window spanning blocks would let an earlier block attend into a later one.
                return self._decode(ids, enc_hidden, enc_mask, inputs_embeds=soft_embeds, omega_bias=omega_bias)

            if right % self.block_size != 0 and right != ids.shape[1]:
                # (c) The cached pass exposes keys only up to `right`, so a window short of its block's end omits the
                # [right, block_end) keys the no-cache forward includes. Exact iff `right` is a boundary or seq end.
                return self._decode(ids, enc_hidden, enc_mask, inputs_embeds=soft_embeds, omega_bias=omega_bias)

            window_ids = ids[:, left:right]
            window_embeds = soft_embeds[:, left:right] if soft_embeds is not None else None
            window_mask = self._prefix_static_window_mask(ids.shape[0], left, right - left, enc_hidden.dtype, ids.device)
            logits_window, _ = _native_forward(
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
        top_p: float | None = None, cache_type: str = "none", omega_bias: torch.Tensor | None = None,
    ) -> SPDDecodeResult:
        batch = enc_hidden.shape[0]
        token_ids = torch.full((batch, int(max_length)), int(self.mask_token_id), dtype=torch.long, device=enc_hidden.device)
        token_ids[:, 0] = int(self.bos_index)

        if cache_type == "prefix" and window_type == "static":
            # Ω rides the cached native forward via CrossAttnOmegaInjector (score-time bias; cached K/V are
            # Ω-independent) — same conditioning as the no-cache `_decode`, verified numerically.
            logits_fn = self._make_static_prefix_cache_logits_fn(enc_hidden, enc_mask, omega_bias=omega_bias)
        elif cache_type == "none":
            def logits_fn(ids: torch.Tensor, soft_embeds: torch.Tensor | None) -> torch.Tensor:
                # inputs_embeds=None -> _decode embeds `ids` itself; else it uses the SPD soft-embedding mixture.
                # omega_bias is fixed conditioning: identical across every denoising step of this decode.
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
            sample_top_k=sample_top_k, top_p=top_p, cache_type=cache_type,
            block_size=self.block_size # Block-causal decoder: DCD window must not span attention blocks (see spd_dcd_decode).
        )


    @torch.no_grad()
    def generate_spd_dcd(self, enc_hidden: torch.Tensor, enc_mask: torch.Tensor, **kwargs) -> SPDDecodeResult:
        return self.decode_spd_dcd(enc_hidden=enc_hidden, enc_mask=enc_mask, **kwargs)
