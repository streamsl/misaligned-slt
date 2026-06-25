"""The Uni-Sign backbone stack (Path A), arXiv 2501.15187: pose encoder + mT5 (default) / mBART (ablation).

One module owns everything Uni-Sign so it is easy to find:
  - `MT5BlockDiffusionDecoder` / `MBartBlockDiffusionDecoder` — the per-LM block-diffusion (BD3LM/OPUT/SPD-DCD)
                                  decoder bindings (each implements only `_decode`).
  - `UniSignMT5FrontEnd` / `UniSignMBartFrontEnd` — UniSignPoseEncoder + the verbatim task prompt + the LM encoder
                                  (one front end per LM; the SAME front end serves the AR baseline and the stage-2
                                  AR/DLM model via `MisalignedSLTModel`, so there is no separate per-LM AR wrapper).
  - `load_unisign_pretrained`   — load a released `*_pose_only_slt.pth` into any model carrying the front end.

mT5 has no absolute positions / no embed scale / no layernorm_embedding, and folds the attention mask into the
attention `position_bias` (T5Attention adds a supplied bias straight to the scores, skipping its own relative-bias +
mask compute). The BD3LM `[xt|x0]` repeated-position geometry is recreated by computing the relative bias over the
EFFECTIVE positions [0..L-1, 0..L-1] (`_self_position_bias`). mT5 ties=False so the LM head needs no d_model**-0.5
scaling. Only the decoder objective changes between the AR and DLM arms (dLLM A2D) — the encoder/prompt is identical.
"""
from __future__ import annotations
from pathlib import Path
import math

import torch
import torch.nn.functional as F
from transformers import MT5Config, MT5ForConditionalGeneration, T5Tokenizer
from transformers import MBartConfig, MBartForConditionalGeneration, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.mbart.modeling_mbart import MBartScaledWordEmbedding

from backbones import UniSignPoseEncoder
from models.block_diffusion import build_block_causal_mask
from models.dmax import OPUTBlockDiffusionDecoder
from models.front_end import SLTFrontEnd

__all__ = [
    "MT5BlockDiffusionDecoder", "MBartBlockDiffusionDecoder", "resolve_decoder_start_id",
    "UniSignFrontEndBase", "UniSignMT5FrontEnd", "UniSignMBartFrontEnd",
    "load_unisign_pretrained", "prompt_lang_for_target", "PROMPT_LANG_BY_TARGET",
]
# target_lang (data.yaml) -> the natural-language name Uni-Sign puts in the prompt 
# (Uni_Sign: 'Chinese' if 'CSL' in dataset else 'English'); de_DE = German for PHOENIX.
PROMPT_LANG_BY_TARGET = {"zh_CN": "Chinese", "en_XX": "English", "de_DE": "German"}

def prompt_lang_for_target(target_lang: str | None) -> str:
    return PROMPT_LANG_BY_TARGET.get(str(target_lang or ""), "English")

# ════════════════════════════════════════════════════════════════════════════
# mBART language model: decoder-start resolver + the block-diffusion decoder binding
# ════════════════════════════════════════════════════════════════════════════

def resolve_decoder_start_id(tokenizer) -> int | None:
    """Target-language decoder start for mBART generation.

    HF mBART's shift_tokens_right wraps the LAST non-pad label token — the language code — to position 0, so the
    decoder is trained to start from the language code. Returns that id (else None -> HF uses the model default).
    """
    lang = getattr(tokenizer, "tgt_lang", None) or getattr(tokenizer, "src_lang", None)
    if not lang: return None
    mapping = getattr(tokenizer, "lang_code_to_id", None)
    if mapping and lang in mapping: return int(mapping[lang])
    lang_id = tokenizer.convert_tokens_to_ids(lang)
    return int(lang_id) if lang_id is not None and lang_id != tokenizer.unk_token_id else None


class MBartBlockDiffusionDecoder(OPUTBlockDiffusionDecoder):
    '''BD3LM/OPUT decoder on the pretrained mBART decoder (learned absolute positions, layernorm_embedding,
    sqrt(d_model) token-embedding scale). Includes the DCD prefix-KV-cache decode (block-boundary exact).

    Same constructor shape as `MT5BlockDiffusionDecoder` (model + the three token indices). `decoder_start_id`
    (the DLM canvas BOS) is mBART's target LANGUAGE CODE, not `<s>`: HF mBART's shift_tokens_right maps labels
    [toks, eos, lang] -> decoder inputs [lang, toks, eos], so the pretrained decoder has only ever seen
    sequences starting with the language code.'''
    def __init__(self, mbart_model, pad_index: int, decoder_start_id: int, eos_id: int, block_size: int = 4, **kw):
        super().__init__()
        self.mbart_decoder = mbart_model.model.decoder  # MBartDecoder layers + norms
        # mBART (scale_embedding=True for mbart-large-cc25) scales token embeddings by sqrt(d_model).
        embed_scale = mbart_model.config.d_model ** 0.5 if getattr(mbart_model.config, "scale_embedding", False) else 1.0
        self._init_block_diffusion(
            d_model=mbart_model.config.d_model, vocab_size=mbart_model.config.vocab_size,
            embed_source_weight=mbart_model.model.shared.weight, lm_source_weight=mbart_model.lm_head.weight,
            pad_index=pad_index, eos_index=eos_id, bos_index=decoder_start_id, embed_scale=embed_scale,
            block_size=block_size, **kw,
        )
        # mBART's word embedding scales by sqrt(d_model) INTERNALLY (MBartScaledWordEmbedding). The DLM canvas needs
        # vocab+1 (the MASK row) AND that internal scale so the manual `_decode` AND the HF `decoder.forward` used by
        # the prefix KV-cache scale token embeddings identically (a plain embedding + external scale made the two
        # paths disagree — the cache != no-cache Δ≈0.5 bug). Swap in a scaled embedding and drop the external scale.
        scaled = MBartScaledWordEmbedding(self.vocab_size + 1, self.d_model, self.pad_index, embed_scale=embed_scale)
        with torch.no_grad(): scaled.weight.copy_(self.embed_tokens.weight)
        self.embed_tokens = scaled
        self.embed_scale = 1.0  # scale now lives inside embed_tokens; never apply it again
        self.mbart_decoder.embed_tokens = self.embed_tokens  # avoids a duplicate parameter
        print(
            f"MBartBlockDiffusionDecoder (mBART A2D): d_model={self.d_model}, vocab={self.vocab_size}+1(MASK), "
            f"block_size={self.block_size}, remasking={self.remasking}, temperature={self.temperature}"
        )


    def _decode(self, decoder_input_ids, enc_hidden, enc_mask, self_attn_mask=None, position_ids=None, inputs_embeds=None):
        batch_size, tgt_len = decoder_input_ids.shape
        device = decoder_input_ids.device
        dtype = enc_hidden.dtype
        if inputs_embeds is None: inputs_embeds = self.embed_tokens(decoder_input_ids)
        hidden = inputs_embeds * self.embed_scale
        if position_ids is not None:
            # [xt|x0]: positions for both halves are [0..L-1, 0..L-1] (mBART uses a +2 pad offset on learned positions).
            positions = self.mbart_decoder.embed_positions.weight[position_ids + 2]
        else:
            positions = self.mbart_decoder.embed_positions(decoder_input_ids)

        hidden = hidden + positions
        hidden = self.mbart_decoder.layernorm_embedding(hidden)
        hidden = F.dropout(hidden, p=self.mbart_decoder.dropout, training=self.training)

        # Self-attention mask: BD3LM mask during training or build block-causal mask for inference
        if self_attn_mask is None: self_mask = build_block_causal_mask(batch_size, tgt_len, self.block_size, dtype, device)
        else: self_mask = self_attn_mask.to(dtype=dtype, device=device)
        cross_mask = AttentionMaskConverter._expand_mask(enc_mask, dtype, tgt_len=tgt_len) if enc_mask is not None else None

        # Run each decoder layer (self-attn + cross-attn + FFN)
        for layer in self.mbart_decoder.layers:
            hidden = layer(
                hidden, attention_mask=self_mask, encoder_hidden_states=enc_hidden, encoder_attention_mask=cross_mask
            )[0]
        if getattr(self.mbart_decoder, "layer_norm", None) is not None: hidden = self.mbart_decoder.layer_norm(hidden)
        return self.lm_head(hidden)


    def _decode_with_decoder_forward(
        self, decoder_input_ids, enc_hidden, enc_mask, self_attn_mask, inputs_embeds=None,
        past_key_values=None, use_cache=False, cache_position=None,
    ):
        # DCD prefix KV-cache hook (shared cache logic in dmax.OPUTBlockDiffusionDecoder; only this native KV-cache-capable
        # decoder.forward is backbone-specific). MBartDecoder uses the supplied 4D mask verbatim; token embeddings scale via
        # the MBartScaledWordEmbedding set in __init__, so the cache matches the no-cache `_decode` (verified ~1e-7).
        out = self.mbart_decoder(
            input_ids=None if inputs_embeds is not None else decoder_input_ids,
            inputs_embeds=inputs_embeds, attention_mask=self_attn_mask,
            encoder_hidden_states=enc_hidden, encoder_attention_mask=enc_mask,
            past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, return_dict=True,
        )
        return self.lm_head(out.last_hidden_state), out.past_key_values

# ════════════════════════════════════════════════════════════════════════════
# mT5 block-diffusion decoder
# ════════════════════════════════════════════════════════════════════════════

class MT5BlockDiffusionDecoder(OPUTBlockDiffusionDecoder):
    '''BD3LM/OPUT decoder on the pretrained mT5 decoder. Owns only the decoder canvas (vocab+1 embedding / LM head);
    the caller (UniSignMT5FrontEnd) owns the mT5 encoder over [prompt|pose].'''
    def __init__(self, mt5_model, pad_index: int, decoder_start_id: int, eos_id: int, block_size: int = 4, **kw):
        super().__init__()
        self.mt5 = mt5_model  # MT5ForConditionalGeneration (encoder used by the front end; we use .decoder + .lm_head)
        self.decoder = mt5_model.decoder  # T5Stack
        self.config = mt5_model.config
        self.n_heads = int(self.config.num_heads)
        self.rel_num_buckets = int(self.config.relative_attention_num_buckets)
        self.rel_max_distance = int(self.config.relative_attention_max_distance)
        self._init_block_diffusion(
            d_model=self.config.d_model, vocab_size=self.config.vocab_size,
            embed_source_weight=self.decoder.embed_tokens.weight, lm_source_weight=mt5_model.lm_head.weight,
            pad_index=pad_index, eos_index=eos_id, bos_index=decoder_start_id, embed_scale=1.0,
            block_size=block_size, **kw,
        )
        print(
            f"MT5BlockDiffusionDecoder (mT5 A2D): d_model={self.d_model}, vocab={self.vocab_size}+1(MASK), "
            f"block_size={self.block_size}, remasking={self.remasking}, temperature={self.temperature}"
        )

    def _rel_bias(self, eff_pos: torch.Tensor) -> torch.Tensor:
        # Relative-position self-attention bias over EFFECTIVE positions (reproduces T5Attention.compute_bias but
        # with caller-supplied positions, so [xt|x0] = [0..L-1, 0..L-1] gets the correct relative geometry).
        sa = self.decoder.block[0].layer[0].SelfAttention
        rel = eff_pos[None, :] - eff_pos[:, None]
        bucket = sa._relative_position_bucket(
            rel, bidirectional=False, num_buckets=self.rel_num_buckets, max_distance=self.rel_max_distance,
        )
        values = sa.relative_attention_bias(bucket)  # (Q, K, n_heads)
        return values.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, Q, K)

    def _self_position_bias(self, tgt_len, dtype, device, self_attn_mask, position_ids, batch_size):
        if position_ids is not None: eff = position_ids[0].to(device=device, dtype=torch.long)
        else: eff = torch.arange(tgt_len, device=device, dtype=torch.long)
        rel_bias = self._rel_bias(eff).to(dtype)
        if self_attn_mask is None: add_mask = build_block_causal_mask(batch_size, tgt_len, self.block_size, dtype, device)
        else: add_mask = self_attn_mask.to(dtype=dtype, device=device)
        return rel_bias + add_mask

    def _cross_position_bias(self, enc_mask, dtype):
        # T5 cross-attn has no relative bias, so passing this as encoder_decoder_position_bias skips its mask path.
        m = enc_mask[:, None, None, :].to(dtype)
        return (1.0 - m) * torch.finfo(dtype).min

    def _run_layers(self, hidden, enc_hidden, self_pos_bias, cross_pos_bias):
        # Manual T5Block loop (bypasses T5Block.forward, which eagerly needs cache_position). 
        # Each sub-layer applies its own RMSNorm + residual internally.
        for block in self.decoder.block:
            hidden = block.layer[0](hidden, attention_mask=None, position_bias=self_pos_bias)[0]
            hidden = block.layer[1](hidden, key_value_states=enc_hidden, attention_mask=None, position_bias=cross_pos_bias)[0]
            hidden = block.layer[2](hidden)
        hidden = self.decoder.final_layer_norm(hidden)
        hidden = self.decoder.dropout(hidden)
        return self.lm_head(hidden)  # mT5 ties=False -> no d_model**-0.5 scaling

    def _decode(self, decoder_input_ids, enc_hidden, enc_mask, self_attn_mask=None, position_ids=None, inputs_embeds=None):
        batch_size, tgt_len = decoder_input_ids.shape
        dtype = enc_hidden.dtype
        device = decoder_input_ids.device
        if inputs_embeds is None: inputs_embeds = self.embed_tokens(decoder_input_ids)
        hidden = self.decoder.dropout(inputs_embeds * self.embed_scale)  # embed_scale = 1.0 for mT5; T5Stack entry dropout
        self_pos_bias = self._self_position_bias(tgt_len, dtype, device, self_attn_mask, position_ids, batch_size)
        cross_pos_bias = self._cross_position_bias(enc_mask.to(device), dtype)
        return self._run_layers(hidden, enc_hidden, self_pos_bias, cross_pos_bias)

    def _decode_with_decoder_forward(
        self, decoder_input_ids, enc_hidden, enc_mask, self_attn_mask, inputs_embeds=None,
        past_key_values=None, use_cache=False, cache_position=None,
    ):
        # DCD prefix KV-cache hook (shared cache logic in dmax.OPUTBlockDiffusionDecoder). T5Stack.forward computes its
        # own relative position bias (standard positions, cache_position-aware) and ADDS the supplied 4D mask to it —
        # at a block boundary with the all-attend window mask this equals the no-cache `_decode` (verified numerically).
        out = self.decoder(
            input_ids=None if inputs_embeds is not None else decoder_input_ids,
            inputs_embeds=inputs_embeds, attention_mask=self_attn_mask,
            encoder_hidden_states=enc_hidden, encoder_attention_mask=enc_mask,
            past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, return_dict=True,
        )
        return self.lm_head(out.last_hidden_state), out.past_key_values

# ════════════════════════════════════════════════════════════════════════════
# Uni-Sign front end — shared pose+prompt base; mT5 (primary) / mBART (ablation) LM variants
# ════════════════════════════════════════════════════════════════════════════

class UniSignFrontEndBase(SLTFrontEnd):
    """Shared Uni-Sign front end: UniSignPoseEncoder -> the verbatim task prompt prepended to the pose tokens -> LM encoder. 
    Subclasses bind the language model (mT5 / mBART) and supply only LM-specific bits: `_prompt_token_embeds`, `_run_lm_encoder`, 
    `make_dlm_decoder`, `_load_lm_pretrained`, and `pose_embed_scale` (mBART scales its word embeddings by sqrt(d), so the pose 
    tokens are scaled to match the prompt-embed magnitude in the encoder; mT5 has no embed scale -> 1.0). Only the LM differs 
    between the two — same pose encoder, same prompt — which makes the mT5-vs-mBART comparison a clean LM ablation."""
    prompt_lang: str = "Chinese"
    pose_embed_scale: float = 1.0  # mBART overrides to sqrt(d_model)

    # ── LM-specific hooks ─────────────────────────────────────────────────────
    def _prompt_token_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _run_lm_encoder(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        raise NotImplementedError

    def _load_lm_pretrained(self, sd: dict, strict: bool) -> tuple[int, int, int]:
        return (0, 0, 0)  # (tensors, missing, unexpected); mBART stays at base init (no released Uni-Sign+mBART weights)

    # ── shared ────────────────────────────────────────────────────────────────
    def extract_bio_tap(self, poses, frame_mask, timestamps_s=None):
        if frame_mask is None: frame_mask = torch.ones(poses.shape[:2], dtype=torch.bool, device=poses.device)
        pose_tokens = self.pose_encoder(poses, frame_mask)  # (B, T, d_model), frame-aligned
        return pose_tokens, frame_mask, timestamps_s

    def encode_memory(self, bio_tap, bio_mask):
        # Prepend the LM-embedded task/language prompt to the pose tokens (Uni_Sign forward). Pose tokens are scaled
        # by `pose_embed_scale` to match the prompt-embed magnitude (1.0 for mT5; sqrt(d) for mBART's scaled embeds).
        device = bio_tap.device
        b = bio_tap.shape[0]
        prefix = self.tokenizer(
            [f"Translate sign language video to {self.prompt_lang}: "] * b,
            padding="longest", truncation=True, return_tensors="pt",
        ).to(device)
        prefix_embeds = self._prompt_token_embeds(prefix["input_ids"])
        inputs_embeds = torch.cat([prefix_embeds, bio_tap * self.pose_embed_scale], dim=1)
        attention_mask = torch.cat([prefix["attention_mask"], bio_mask.long()], dim=1)
        enc_out = self._run_lm_encoder(inputs_embeds, attention_mask)
        return enc_out.last_hidden_state, attention_mask

    def ar_loss(self, enc_hidden, enc_mask, labels, label_smoothing: float = 0.2):
        # Uni-Sign uses an external label-smoothed CE (models.py:300) rather than the model's internal loss.
        out = self.lm_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden), 
            attention_mask=enc_mask, labels=labels, return_dict=True,
        )
        return F.cross_entropy(
            out.logits.reshape(-1, out.logits.shape[-1]), labels.reshape(-1).to(out.logits.device),
            ignore_index=-100, label_smoothing=float(label_smoothing),
        )

    def freeze_pose_backbone(self, freeze_projection: bool = False) -> int:
        frozen = 0
        for p in self.pose_encoder.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                frozen += p.numel()
        return frozen

    def load_pretrained(self, ckpt_path, strict: bool = True) -> dict[str, int]:
        """Load a released Uni-Sign `*_pose_only_slt.pth` ({'model': {<pose>.*, mt5_model.*}}): pose keys -> `pose_encoder` 
        (always); the mT5 arm additionally loads `mt5_model.*` -> mT5, the mBART arm loads pose only (mBART LM stays at 
        base init). Call BEFORE building the DLM decoder (it copies the loaded LM into the vocab+1 canvas)."""
        blob = torch.load(str(ckpt_path), map_location="cpu")
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        pose_sd = {k: v for k, v in sd.items() if not k.startswith("mt5_model.")}
        pose_ret = self.pose_encoder.load_state_dict(pose_sd, strict=strict)
        lm_t, lm_m, lm_u = self._load_lm_pretrained(sd, strict)
        return {
            "pose_tensors": len(pose_sd), "mt5_tensors": lm_t,
            "pose_missing": len(pose_ret.missing_keys), "pose_unexpected": len(pose_ret.unexpected_keys),
            "mt5_missing": lm_m, "mt5_unexpected": lm_u,
        }


class UniSignMBartFrontEnd(UniSignFrontEndBase):
    """mBART LM ablation: same Uni-Sign pose encoder + task prompt as the mT5 arm, mBART as the language model.
    mBART's word embedding scales by sqrt(d) (MBartScaledWordEmbedding), so pose tokens are scaled by sqrt(d) to
    match the prompt-embed magnitude in the encoder. Init: pose from the released Uni-Sign ckpt (`load_pretrained`),
    mBART from base (no released Uni-Sign+mBART weights). decoder-start = the mBART target-language code."""
    def __init__(
        self, mbart_name: str = "facebook/mbart-large-cc25", prompt_lang: str = "Chinese",
        target_lang: str = "zh_CN", tokenizer=None, pose_hidden_dim: int = 256, init_mbart_weights: bool = True,
    ):
        super().__init__()
        self.prompt_lang = str(prompt_lang)
        if init_mbart_weights: self.mbart = MBartForConditionalGeneration.from_pretrained(mbart_name)
        else: self.mbart = MBartForConditionalGeneration(MBartConfig.from_pretrained(mbart_name))
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            mbart_name, src_lang=target_lang, tgt_lang=target_lang,
        )
        d_model = int(self.mbart.config.d_model)
        self.pose_embed_scale = math.sqrt(d_model)  # match mBART's sqrt(d) scaled word embeddings
        self.pose_encoder = UniSignPoseEncoder(hidden_dim=int(pose_hidden_dim), out_dim=d_model)
        self.lm_model = self.mbart
        self.bio_tap_dim = d_model
        self.pad_token_id = int(self.mbart.config.pad_token_id)
        self.eos_token_id = int(self.mbart.config.eos_token_id)
        self.decoder_start_id = resolve_decoder_start_id(self.tokenizer)  # mBART lang-code decoder start

    def _prompt_token_embeds(self, input_ids):
        return self.mbart.model.shared(input_ids)  # MBartScaledWordEmbedding -> scaled, matches pose * sqrt(d)

    def _run_lm_encoder(self, inputs_embeds, attention_mask):
        return self.mbart.model.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)

    def make_dlm_decoder(self, block_size: int) -> OPUTBlockDiffusionDecoder:
        # DLM canvas BOS = the mBART target language code (resolve_decoder_start_id); fall back to <s>/0 only if the
        # tokenizer has no language code, mirroring how HF shift_tokens_right starts the pretrained decoder.
        bos = self.decoder_start_id if self.decoder_start_id is not None else (getattr(self.tokenizer, "bos_token_id", None) or 0)
        return MBartBlockDiffusionDecoder(
            self.mbart, pad_index=self.pad_token_id, decoder_start_id=bos, eos_id=self.eos_token_id, block_size=block_size,
        )


class UniSignMT5FrontEnd(UniSignFrontEndBase):
    def __init__(
        self, mt5_name: str = "google/mt5-base", prompt_lang: str = "Chinese", tokenizer=None,
        pose_hidden_dim: int = 256, init_mt5_weights: bool = False,
    ):
        """`init_mt5_weights`: True downloads pretrained mT5-base; False builds from config only (no 2.3GB download)
        — correct when a Uni-Sign checkpoint overwrites every mT5 tensor anyway."""
        super().__init__()
        self.prompt_lang = str(prompt_lang)
        self.pose_embed_scale = 1.0  # mT5 does not scale word embeddings
        if init_mt5_weights: self.mt5 = MT5ForConditionalGeneration.from_pretrained(mt5_name)
        else: self.mt5 = MT5ForConditionalGeneration(MT5Config.from_pretrained(mt5_name))
        self.tokenizer = tokenizer if tokenizer is not None else T5Tokenizer.from_pretrained(mt5_name, legacy=False)
        d_model = int(self.mt5.config.d_model)
        self.pose_encoder = UniSignPoseEncoder(hidden_dim=int(pose_hidden_dim), out_dim=d_model)
        self.lm_model = self.mt5
        self.bio_tap_dim = d_model
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        # mT5 conditions via the encoder prompt (no language-code prefix), so the canvas/AR start is mT5's own
        # decoder_start_token_id (pad for T5) — what HF teacher-forcing shifts labels onto.
        self.decoder_start_id = int(self.mt5.config.decoder_start_token_id)

    def _prompt_token_embeds(self, input_ids):
        return self.mt5.encoder.embed_tokens(input_ids)

    def _run_lm_encoder(self, inputs_embeds, attention_mask):
        return self.mt5.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)

    def make_dlm_decoder(self, block_size: int) -> OPUTBlockDiffusionDecoder:
        return MT5BlockDiffusionDecoder(
            self.mt5, pad_index=self.pad_token_id, decoder_start_id=self.decoder_start_id,
            eos_id=self.eos_token_id, block_size=block_size,
        )

    def _load_lm_pretrained(self, sd, strict):
        mt5_sd = {k[len("mt5_model."):]: v for k, v in sd.items() if k.startswith("mt5_model.")}
        ret = self.mt5.load_state_dict(mt5_sd, strict=strict)
        return len(mt5_sd), len(ret.missing_keys), len(ret.unexpected_keys)


def load_unisign_pretrained(model, ckpt_path: str | Path, strict: bool = True) -> dict[str, int]:
    """Load a released Uni-Sign `*_pose_only_slt.pth` into anything carrying a `UniSignMT5FrontEnd` at `.front_end`
    (the baseline `UniSignMT5SLT` or the stage-2 `MisalignedSLTModel`). Delegates to `UniSignMT5FrontEnd.load_pretrained`."""
    return model.front_end.load_pretrained(ckpt_path, strict=strict)
