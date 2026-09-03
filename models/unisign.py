"""Uni-Sign backbone stack (Path A), arXiv 2501.15187: pose encoder + mT5 (default) / mBART (ablation).

Everything Uni-Sign lives here:
  - `MT5BlockDiffusionDecoder` / `MBartBlockDiffusionDecoder` — per-LM block-diffusion (BD3LM/OPUT/SPD-DCD)
    decoder bindings (each implements only `_decode`).
  - `UniSignMT5FrontEnd` / `UniSignMBartFrontEnd` — pose encoder + task prompt + LM encoder; the SAME front end
    serves the AR baseline and the AR/DLM SLT model via `MisalignedSLTModel`.
  - `load_unisign_pretrained` — load a released `*_pose_only_slt.pth` into a model carrying the front end.

mT5: no absolute positions / embed scale / layernorm_embedding, and the attn mask folds into `position_bias`
(T5Attention adds a supplied bias straight to the scores, skipping its own relative-bias + mask compute), so
BD3LM `[xt|x0]` geometry is rebuilt over EFFECTIVE positions [0..L-1, 0..L-1] (`_self_position_bias`). AR and 
DLM arms differ only in the decoder objective (dLLM A2D).
"""
from __future__ import annotations
from pathlib import Path
import math
import torch
import torch.nn as nn
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
PROMPT_LANG_BY_TARGET = {"en_XX": "English"}

def prompt_lang_for_target(target_lang: str | None) -> str:
    return PROMPT_LANG_BY_TARGET.get(str(target_lang or ""), "English")

# ════════════════════════════════════════════════════════════════════════════
# mBART language model: decoder-start resolver + the block-diffusion decoder binding
# ════════════════════════════════════════════════════════════════════════════

def resolve_decoder_start_id(tokenizer) -> int | None:
    """Target-language decoder start id for mBART; None -> HF model default.

    shift_tokens_right wraps the LAST non-pad label token — the language code — to position 0, so the pretrained
    decoder has only ever started from the language code.
    """
    lang = getattr(tokenizer, "tgt_lang", None) or getattr(tokenizer, "src_lang", None)
    if not lang: return None
    mapping = getattr(tokenizer, "lang_code_to_id", None)
    if mapping and lang in mapping: return int(mapping[lang])
    lang_id = tokenizer.convert_tokens_to_ids(lang)
    return int(lang_id) if lang_id is not None and lang_id != tokenizer.unk_token_id else None


class MBartBlockDiffusionDecoder(OPUTBlockDiffusionDecoder):
    '''BD3LM/OPUT decoder on the pretrained mBART decoder (learned absolute positions, layernorm_embedding,
    sqrt(d_model) embed scale), plus the DCD prefix-KV-cache decode (block-boundary exact). `decoder_start_id` (the DLM 
    canvas BOS) is mBART's target LANGUAGE CODE, not `<s>` — see `resolve_decoder_start_id`.'''
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
        # The canvas needs vocab+1 (MASK row) AND mBART's INTERNAL sqrt(d_model) scale, so `_decode` and the HF
        # `decoder.forward` behind the prefix KV-cache embed identically. Plain embedding + external scale made
        # them disagree (the cache != no-cache Δ≈0.5 bug).
        scaled = MBartScaledWordEmbedding(self.vocab_size + 1, self.d_model, self.pad_index, embed_scale=embed_scale)
        with torch.no_grad(): scaled.weight.copy_(self.embed_tokens.weight)
        self.embed_tokens = scaled
        self.embed_scale = 1.0  # scale lives inside embed_tokens now; never apply it twice
        self.mbart_decoder.embed_tokens = self.embed_tokens  # avoids a duplicate parameter
        print(
            f"MBartBlockDiffusionDecoder (mBART A2D): d_model={self.d_model}, vocab={self.vocab_size}+1(MASK), "
            f"block_size={self.block_size}, remasking={self.remasking}, temperature={self.temperature}"
        )


    def _decode(
        self, decoder_input_ids, enc_hidden, enc_mask, self_attn_mask=None, 
        position_ids=None, inputs_embeds=None, omega_bias=None, logits_len=None,
    ):
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

        # Self-attn: BD3LM mask (training) or block-causal (inference)
        if self_attn_mask is None: self_mask = build_block_causal_mask(batch_size, tgt_len, self.block_size, dtype, device)
        else: self_mask = self_attn_mask.to(dtype=dtype, device=device)
        cross_mask = AttentionMaskConverter._expand_mask(enc_mask, dtype, tgt_len=tgt_len) if enc_mask is not None else None
        # Membership gate: add Ω(t) (query-independent, (B,1,1,M)) to the cross-attn key bias; broadcasts over
        # the query axis, so every layer/head sees it (docs/membership_gate.md §2.9).
        if omega_bias is not None:
            ob = omega_bias.to(dtype=dtype, device=device)
            cross_mask = ob if cross_mask is None else cross_mask + ob

        for layer in self.mbart_decoder.layers:
            hidden = layer(
                hidden, attention_mask=self_mask, encoder_hidden_states=enc_hidden, encoder_attention_mask=cross_mask
            )[0]
        if getattr(self.mbart_decoder, "layer_norm", None) is not None: hidden = self.mbart_decoder.layer_norm(hidden)
        if logits_len is not None: hidden = hidden[:, :logits_len]
        return self.lm_head(hidden)


    def _decode_with_decoder_forward(
        self, decoder_input_ids, enc_hidden, enc_mask, self_attn_mask, inputs_embeds=None,
        past_key_values=None, use_cache=False, cache_position=None,
    ):
        # DCD prefix KV-cache hook (cache logic in dmax; only this KV-cache-capable decoder.forward is backbone-specific). 
        # MBartDecoder takes the 4D mask verbatim and embeds via the MBartScaledWordEmbedding, so the cache matches the 
        # no-cache `_decode` (verified ~1e-7).
        out = self.mbart_decoder(
            input_ids=None if inputs_embeds is not None else decoder_input_ids,
            inputs_embeds=inputs_embeds, attention_mask=self_attn_mask,
            encoder_hidden_states=enc_hidden, encoder_attention_mask=enc_mask,
            past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, return_dict=True,
        )
        return self.lm_head(out.last_hidden_state), out.past_key_values

    def _decoder_stack(self):
        return self.mbart_decoder  # MBartDecoder — the stack the Ω injector hooks (layers[i].encoder_attn)

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
        # T5 scales by d_model**-0.5 before lm_head IFF config.tie_word_embeddings — mirror HF's gate exactly.
        # google/mt5-base ships the flag FALSE, so this resolves to 1.0 there; the gate exists for tied checkpoints.
        self.lm_head_scale = float(self.config.d_model) ** -0.5 if getattr(self.config, "tie_word_embeddings", False) else 1.0
        self._init_block_diffusion(
            d_model=self.config.d_model, vocab_size=self.config.vocab_size, embed_source_weight=self.decoder.embed_tokens.weight, 
            lm_source_weight=mt5_model.lm_head.weight, pad_index=pad_index, eos_index=eos_id, bos_index=decoder_start_id, 
            embed_scale=1.0, block_size=block_size, **kw,
        )
        print(
            f"MT5BlockDiffusionDecoder (mT5 A2D): d_model={self.d_model}, vocab={self.vocab_size}+1(MASK), "
            f"block_size={self.block_size}, remasking={self.remasking}, temperature={self.temperature}"
        )

    def _rel_bias(self, eff_pos: torch.Tensor) -> torch.Tensor:
        # T5Attention.compute_bias with caller-supplied positions: [xt|x0] = [0..L-1, 0..L-1] needs that geometry.
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

    def _run_layers(self, hidden, enc_hidden, self_pos_bias, cross_pos_bias, logits_len=None):
        # Manual T5Block loop: T5Block.forward eagerly needs cache_position. Sub-layers do their own RMSNorm+residual.
        for block in self.decoder.block:
            hidden = block.layer[0](hidden, attention_mask=None, position_bias=self_pos_bias)[0]
            hidden = block.layer[1](hidden, key_value_states=enc_hidden, attention_mask=None, position_bias=cross_pos_bias)[0]
            hidden = block.layer[2](hidden)
        hidden = self.decoder.final_layer_norm(hidden)
        hidden = self.decoder.dropout(hidden)
        # Slice AFTER dropout: the draw covers the full canvas either way, so the kept half is bit-identical.
        if logits_len is not None: hidden = hidden[:, :logits_len]
        return self.lm_head(hidden * self.lm_head_scale)  # T5 tie-flag scaling (see __init__)

    def _decode(
        self, decoder_input_ids, enc_hidden, enc_mask, self_attn_mask=None,
        position_ids=None, inputs_embeds=None, omega_bias=None, logits_len=None,
    ):
        batch_size, tgt_len = decoder_input_ids.shape
        dtype = enc_hidden.dtype
        device = decoder_input_ids.device
        if inputs_embeds is None: inputs_embeds = self.embed_tokens(decoder_input_ids)
        hidden = self.decoder.dropout(inputs_embeds * self.embed_scale)  # embed_scale = 1.0 for mT5; T5Stack entry dropout
        self_pos_bias = self._self_position_bias(tgt_len, dtype, device, self_attn_mask, position_ids, batch_size)
        cross_pos_bias = self._cross_position_bias(enc_mask.to(device), dtype)
        # Membership gate: ADD Ω(t) to the cross-attention key bias (query-independent → every layer and head,
        # every decoder query). Ω already carries the prompt-offset zeros, so it aligns to enc_hidden columns.
        if omega_bias is not None: cross_pos_bias = cross_pos_bias + omega_bias.to(dtype=dtype, device=device)
        return self._run_layers(hidden, enc_hidden, self_pos_bias, cross_pos_bias, logits_len)

    def _decode_with_decoder_forward(
        self, decoder_input_ids, enc_hidden, enc_mask, self_attn_mask, inputs_embeds=None,
        past_key_values=None, use_cache=False, cache_position=None,
    ):
        # DCD prefix KV-cache hook. T5Stack.forward computes its own relative position bias (standard positions,
        # cache_position-aware) and ADDS the supplied 4D mask — at a block boundary with the all-attend window
        # mask this equals the no-cache `_decode` (verified numerically).
        out = self.decoder(
            input_ids=None if inputs_embeds is not None else decoder_input_ids,
            inputs_embeds=inputs_embeds, attention_mask=self_attn_mask,
            encoder_hidden_states=enc_hidden, encoder_attention_mask=enc_mask,
            past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, return_dict=True,
        )
        return self.lm_head(out.last_hidden_state * self.lm_head_scale), out.past_key_values

    def _decoder_stack(self):
        return self.decoder  # T5Stack — the stack the Ω injector hooks (block[i].layer[1])

# ════════════════════════════════════════════════════════════════════════════
# Uni-Sign front end — shared pose+prompt base; mT5 (primary) / mBART (ablation) LM variants
# ════════════════════════════════════════════════════════════════════════════

def released_layout_state(sd: dict) -> dict:
    """Normalize a checkpoint state dict to the RELEASED Uni-Sign layout ({<bare pose keys>, 'mt5_model.*'}).

    Accepts either layout, so ONE artifact serves every consumer:
      - released `*_pose_only_slt.pth` -> as-is;
      - trainer `model.pt` -> re-keyed from front_end.pose_encoder.* / front_end.mt5.*; other keys (bio_head,
        dlm_decoder) dropped as training-arm state, not the transferable front end.
    """
    if not any(k.startswith("front_end.") for k in sd): return sd
    out = {}
    for k, v in sd.items():
        if k.startswith("front_end.pose_encoder."): out[k[len("front_end.pose_encoder."):]] = v
        elif k.startswith("front_end.mt5."): out["mt5_model." + k[len("front_end.mt5."):]] = v
    return out


class UniSignFrontEndBase(SLTFrontEnd):
    """Shared Uni-Sign front end: UniSignPoseEncoder -> verbatim task prompt + pose tokens -> LM encoder.

    Subclasses bind the LM (mT5 / mBART) and supply `_prompt_token_embeds`, `_run_lm_encoder`, `make_dlm_decoder`,
    `_load_lm_pretrained`, `pose_embed_scale` (mBART scales word embeddings by sqrt(d), so pose tokens match the
    prompt-embed magnitude; mT5 -> 1.0). Only the LM differs, so mT5-vs-mBART is a clean ablation."""
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

    def prompt_length(self) -> int:
        # Cached: memory is [prompt | pose], M = prompt_length() + T. Aligns the gate's Ω to the pose columns
        # without an encoder pass just to read a shape.
        if getattr(self, "_prompt_len", None) is None:
            ids = self.tokenizer([f"Translate sign language video to {self.prompt_lang}: "], return_tensors="pt")["input_ids"]
            self._prompt_len = int(ids.shape[1])
        return self._prompt_len

    def encode_memory(self, bio_tap, bio_mask):
        # Prepend the LM-embedded task/language prompt to the pose tokens (Uni_Sign forward).
        device = bio_tap.device
        b = bio_tap.shape[0]
        prefix = self.tokenizer(
            [f"Translate sign language video to {self.prompt_lang}: "] * b, padding="longest", return_tensors="pt"
        ).to(device)
        prefix_embeds = self._prompt_token_embeds(prefix["input_ids"])
        inputs_embeds = torch.cat([prefix_embeds, bio_tap * self.pose_embed_scale], dim=1)
        attention_mask = torch.cat([prefix["attention_mask"], bio_mask.long()], dim=1)
        enc_out = self._run_lm_encoder(inputs_embeds, attention_mask)
        return enc_out.last_hidden_state, attention_mask

    def ar_loss(self, enc_hidden, enc_mask, labels, label_smoothing: float = 0.2, omega_bias=None, row_stats: bool = False):
        # Uni-Sign uses an external label-smoothed CE (models.py:300), not the model's internal loss. `omega_bias`
        # gates cross-attn via HF hooks (CrossAttnOmegaInjector), so the AR de-risk arm trains under the same Ω as
        # the DLM arm and translation loss backprops into the BIO logits through Ω.
        with self.ar_omega_context(omega_bias):
            out = self.lm_model(
                encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
                attention_mask=enc_mask, labels=labels, return_dict=True,
            )
        flat = F.cross_entropy(
            out.logits.reshape(-1, out.logits.shape[-1]), labels.reshape(-1).to(out.logits.device),
            ignore_index=-100, label_smoothing=float(label_smoothing), reduction="none",
        ).reshape_as(labels)
        keep = (labels != -100).to(dtype=flat.dtype)
        loss = (flat * keep).sum() / keep.sum().clamp(min=1)  # == flat-CE mean over non-ignored tokens
        if not row_stats: return loss
        # Detached per-row (summed loss, valid count) so one merged call can report a per-mode breakdown.
        return loss, (flat * keep).detach().sum(dim=1), keep.detach().sum(dim=1)

    def freeze_pose_backbone(self, freeze_projection: bool = False) -> int:
        #Freeze the ST-GCN pose backbone; returns the parameter count frozen. `freeze_projection=False` (default)
        # keeps `pose_encoder.pose_proj` trainable so it can still adapt the frozen GCN features to the LM.
        frozen = 0
        for p in self.pose_encoder.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                frozen += p.numel()
        if not freeze_projection:
            for p in self.pose_encoder.pose_proj.parameters():
                p.requires_grad_(True)
                frozen -= p.numel()
        self._pose_backbone_frozen = True  # `train()` keeps it in eval (see below)
        self.pose_encoder.eval()
        return frozen

    def train(self, mode: bool = True):
        # A frozen pose backbone must stay in eval: BatchNorm2d would otherwise update running_mean/var during
        # SLT training, drifting S2 features off the S1 the BIO head was pretrained on (docs/membership_gate.md §1.4).
        super().train(mode)
        if getattr(self, "_pose_backbone_frozen", False): self.pose_encoder.eval()
        return self

    def load_pretrained(self, ckpt_path, strict: bool = True) -> dict[str, int]:
        """Load a released `*_pose_only_slt.pth` or trainer `model.pt` (normalized by `released_layout_state`):
        pose keys -> `pose_encoder`, mT5 arm also `mt5_model.*` (mBART: pose only). Call BEFORE building the DLM
        decoder — it copies the loaded LM into the vocab+1 canvas."""
        blob = torch.load(str(ckpt_path), map_location="cpu")
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        sd = released_layout_state(sd)
        pose_sd = {k: v for k, v in sd.items() if not k.startswith("mt5_model.")}
        pose_ret = self.pose_encoder.load_state_dict(pose_sd, strict=strict)
        lm_t, lm_m, lm_u = self._load_lm_pretrained(sd, strict)
        return {
            "pose_tensors": len(pose_sd), "mt5_tensors": lm_t,
            "pose_missing": len(pose_ret.missing_keys), "pose_unexpected": len(pose_ret.unexpected_keys),
            "mt5_missing": lm_m, "mt5_unexpected": lm_u,
        }


class UniSignMBartFrontEnd(UniSignFrontEndBase):
    # mBART LM ablation: same pose encoder + task prompt as the mT5 arm. Init: pose from the released Uni-Sign
    # ckpt (`load_pretrained`), mBART from base (no released Uni-Sign+mBART weights); decoder-start = mBART target-language code.
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
        return self.mbart.model.shared(input_ids)  # MBartScaledWordEmbedding — matches pose * sqrt(d)

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
        # `init_mt5_weights=False` builds from config only — correct when a Uni-Sign checkpoint overwrites every mT5 tensor anyway.
        super().__init__()
        self.prompt_lang = str(prompt_lang)
        self.pose_embed_scale = 1.0  # mT5 does not scale word embeddings
        if init_mt5_weights: self.mt5 = MT5ForConditionalGeneration.from_pretrained(mt5_name)
        else: self.mt5 = MT5ForConditionalGeneration(MT5Config.from_pretrained(mt5_name))
        # Defensive untie for TIED configs: if tie_word_embeddings resolved True, the plain constructor aliases
        # lm_head.weight <-> shared.weight, and loading an untied checkpoint would write both keys into one tensor
        # (last write wins) — the encoder would then embed with the LM-head matrix. google/mt5-base ships the flag
        # FALSE, so this guard does not fire there; it protects any tied checkpoint swapped in later. Untie the
        # PARAMETER only — the d_model**-0.5 pre-lm_head scale stays gated on the same flag (lm_head_scale above).
        if self.mt5.lm_head.weight.data_ptr() == self.mt5.shared.weight.data_ptr():
            self.mt5.lm_head = nn.Linear(self.mt5.config.d_model, self.mt5.config.vocab_size, bias=False)
        self.tokenizer = tokenizer if tokenizer is not None else T5Tokenizer.from_pretrained(mt5_name, legacy=False)
        d_model = int(self.mt5.config.d_model)
        self.pose_encoder = UniSignPoseEncoder(hidden_dim=int(pose_hidden_dim), out_dim=d_model)
        self.lm_model = self.mt5
        self.bio_tap_dim = d_model
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        # mT5 conditions via the encoder prompt (no language-code prefix), so the canvas/AR start is
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
    # Load a released `*_pose_only_slt.pth` into anything carrying a `UniSignMT5FrontEnd` at 
    # `.front_end` (`UniSignMT5SLT`, `MisalignedSLTModel`).
    return model.front_end.load_pretrained(ckpt_path, strict=strict)
