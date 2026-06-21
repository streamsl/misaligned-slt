from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones import CoSign1s, DSTANet

from transformers import MBartConfig, MBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
try: from safetensors.torch import load_file as load_safetensors
except Exception: load_safetensors = None # pragma: no cover - optional dependency path


@dataclass
class GFSLTConfig:
    embed_dim: int = 1024
    hidden_size: int = 1024
    temporal_kernel: int = 3
    # The trimmed mBART directory (trim_mbart output). One model serves every role: stage-1 text
    # encoder, the bidirectional visual encoder, and the decoder that becomes the AR/DLM translation
    # decoder. trim_mbart depth-trims it (default 3 enc / 3 dec) so this is a small, fast model.
    mbart_name: str = "facebook/mbart-large-cc25"
    num_keypoints: int = 77 # Selected keypoints fed to the backbone (CoSign: 77; DSTA/MSKA-native: 133)
    input_channels: int = 3 # x, y, confidence
    use_temporal_conv: bool = False
    # Pose backbone: "cosign" = CoSign1s ST-GCN on 77 CoSign-normalised keypoints (default, prior behaviour); "dsta" = MSKA's 4-stream 
    # decoupled spatial-temporal attention (backbones/dsta.py) on 133 MSKA-normalised COCO-WholeBody keypoints. The choice binds the 
    # pose representation the data layer must produce (see poses.pose_io.pose_repr_for_backbone) and MUST be identical across the 
    # stage-1 VLP and stage-2 (the VLP checkpoint carries backbone weights of one specific architecture).
    backbone: str = "cosign"
    # DSTA spatial PE buffers are sized to this many frames; it must exceed the longest window the
    # backbone ever sees (streaming buffer_cap_s * fps). 256 ~ 20s at 12.5 fps (> the 18s buffer cap).
    dsta_num_frame: int = 256
    dsta_dropout: float = 0.1
    # Exclude batch-padded frames from DSTA's BatchNorm statistics (mask-aware BN);
    # False = faithful MSKA path (BN over all frames incl. padding). Only effective for backbone='dsta'.
    dsta_mask_aware: bool = True
    # Scale the post-VLP pose features fed to the mBART encoder by sqrt(d_model), as GFSLT-VLP's gloss_free_model does for its visual 
    # sign embeddings (models.py:307, gated by config ['training']['scale_embedding']). HF MBartEncoder does NOT apply embed_scale to 
    # `inputs_embeds` (only to the embed_tokens path), so the pretrained encoder expects pose features at token- embedding magnitude 
    # (~sqrt(d)), not positional-embedding magnitude. MUST be identical across stage-1 VLP and stage-2 (sourced from stage-1 config)
    # or the VLP-trained encoder sees out-of-distribution input magnitude in stage 2.
    scale_embedding: bool = False
    # Skip mBART TEXT encoder in the pose path: Decoder cross-attends to per-frame VLP features DIRECTLY (BaseModelOutput(post_vlp)), 
    # instead of post_vlp -> mBART encoder -> decoder. This mirrors the decoder-only-captioner design that reached stronger pose BLEU 
    # in the prior project (mBART decoder cross-attending to a from-scratch visual encoder, never routing pose via a text-pretrained 
    # encoder). NB this is NOT encoder_layers=0 (which still adds absolute positions + 2 LayerNorms).
    bypass_mbart_encoder: bool = False


def make_gfslt_config(cfg: dict) -> "GFSLTConfig":
    """Build a GFSLTConfig from a stage config's `backbone:` block — the single architecture source of truth.

    The stage-1 VLP config owns the backbone; stage-2, the baseline, and eval all build their config from the SAME `backbone:` 
    block (passed the stage-1 config) so the architecture cannot drift from the VLP checkpoint. Schema:
      backbone:
        name: cosign | dsta
        embed_dim / hidden_size / scale_embedding / use_temporal_conv
        cosign: { num_keypoints, temporal_kernel }
        dsta:   { num_frame, dropout, mask_aware }
    """
    from utils import mbart_trimmed_dir
    bb = dict(cfg.get("backbone", {}) or {})
    name = str(bb.get("name", "cosign")).lower()
    cosign = dict(bb.get("cosign", {}) or {})
    dsta = dict(bb.get("dsta", {}) or {})
    return GFSLTConfig(
        embed_dim=int(bb.get("embed_dim", 1024)),
        hidden_size=int(bb.get("hidden_size", 1024)),
        temporal_kernel=int(cosign.get("temporal_kernel", 3)),
        mbart_name=mbart_trimmed_dir(cfg),
        use_temporal_conv=bool(bb.get("use_temporal_conv", False)),
        scale_embedding=bool(bb.get("scale_embedding", False)),
        backbone=name,
        num_keypoints=133 if name == "dsta" else int(cosign.get("num_keypoints", 77)),
        dsta_num_frame=int(dsta.get("num_frame", 256)),
        dsta_dropout=float(dsta.get("dropout", 0.1)),
        dsta_mask_aware=bool(dsta.get("mask_aware", True)),
        bypass_mbart_encoder=bool(bb.get("bypass_mbart_encoder", False)),
    )


def _sanitize_generation_config(model: MBartForConditionalGeneration) -> MBartForConditionalGeneration:
    """Force greedy, deterministic decoding and strip None generation params.

    Two reasons:
      1. Crash guard — the depth/vocab-trimmed cc25 carries num_beams / num_return_sequences = None in its generation_config; 
         HF generate() evaluates max(num_beams, num_return_sequences) (transformers generation/utils.py), which raises "'>' 
         not supported between 'int' and 'NoneType'".
      2. Fair AR-vs-DLM comparison — the DLM arm decodes greedily (SPD/DCD at temperature 0), so AR arm must decode greedily too. 
         Inheriting cc25's beam search (num_beams=5) will confound diffusion-vs-AR test by handing AR baseline a free decoding advantage. 
         Beam search can still be requested explicitly per-call (kwargs override generation_config) for a dedicated final-number run.
    """
    gen_cfg = model.generation_config
    gen_cfg.num_beams = 1
    gen_cfg.num_return_sequences = 1
    gen_cfg.do_sample = False
    gen_cfg.early_stopping = False
    return model


def load_gfslt_mbart(model_path: str) -> MBartForConditionalGeneration:
    # Load full mBART even when the trimmed checkpoint was saved decoder-only.
    path = Path(model_path)
    if not path.exists(): return _sanitize_generation_config(MBartForConditionalGeneration.from_pretrained(model_path))
    config = MBartConfig.from_pretrained(model_path)
    config.is_encoder_decoder = True
    config.is_decoder = False
    model = MBartForConditionalGeneration(config)

    safetensors_path = path / "model.safetensors"
    bin_path = path / "pytorch_model.bin"
    if safetensors_path.exists() and load_safetensors is not None: state_dict = load_safetensors(str(safetensors_path))
    elif bin_path.exists(): state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    else: return _sanitize_generation_config(MBartForConditionalGeneration.from_pretrained(model_path))

    if "model.decoder.embed_tokens.weight" in state_dict and "model.shared.weight" not in state_dict:
        state_dict["model.shared.weight"] = state_dict["model.decoder.embed_tokens.weight"]
    model.load_state_dict(state_dict, strict=False)
    return _sanitize_generation_config(model)


class TemporalConv1D(nn.Module): # Legacy GFSLT temporal convolution, optional in this project.
    def __init__(self, input_size: int, hidden_size: int, conv_type: int = 2):
        super().__init__()
        if conv_type == 0: kernel_spec = ["K3"]
        elif conv_type == 1: kernel_spec = ["K5", "P2"]
        elif conv_type == 2: kernel_spec = ["K5", "P2", "K5", "P2"]
        else: raise ValueError(f"Unsupported conv_type={conv_type}")

        layers: list[nn.Module] = []
        for idx, item in enumerate(kernel_spec):
            in_size = input_size if idx == 0 else hidden_size
            if item[0] == "P": layers.append(nn.MaxPool1d(kernel_size=int(item[1]), ceil_mode=False))
            else: layers.extend([
                nn.Conv1d(in_size, hidden_size, kernel_size=int(item[1]), stride=1, padding=0),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
            ])
        self.temporal_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.temporal_conv(x.permute(0, 2, 1)).permute(0, 2, 1)


class PoseFeatureExtractor(nn.Module):
    '''
    Combines CoSign backbone with optional temporal convolution for downsampling.
    Replaces the ResNet-based FeatureExtracter from original GFSLT-VLP.
    '''
    def __init__(self, config: GFSLTConfig, level: str = 'spatial', adaptive: bool = True):
        super().__init__()
        self.config = config
        self.backbone_type = str(getattr(config, "backbone", "cosign")).lower()
        if self.backbone_type == "dsta":
            # MSKA decoupled spatial-temporal attention on MSKA-native 133 keypoints. It keeps T frame-aligned internally 
            # (force_stride1), so the legacy temporal_conv downsampler is incompatible (it would desync BIO/timestamps).
            if bool(config.use_temporal_conv): raise ValueError("backbone='dsta' is frame-aligned; set use_temporal_conv=false.")
            self.dsta = DSTANet(
                hidden_size=config.hidden_size, num_frame=int(config.dsta_num_frame),
                dropout=float(config.dsta_dropout),
            )
            # Exclude padded frames from DSTA's BatchNorms (variable-length windows are batch-zero-padded;
            # the constant features padding produces would otherwise skew BN). False = faithful MSKA path.
            self.dsta_mask_aware = bool(getattr(config, "dsta_mask_aware", True))
            self.use_temporal_conv = False
            self.output_dim = config.hidden_size
            return
        self.cosign = CoSign1s( # CoSign1s ST-GCN backbone for pose feature extraction
            temporal_kernel=config.temporal_kernel, hidden_size=config.hidden_size,
            level=level, adaptive=adaptive
        )
        self.use_temporal_conv = bool(config.use_temporal_conv)
        self.output_dim = config.embed_dim if self.use_temporal_conv else config.hidden_size
        if self.use_temporal_conv: self.temporal_conv = TemporalConv1D(config.hidden_size, config.embed_dim, conv_type=2)

    def forward(
        self, poses: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        timestamps_s: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Both backbones share CoSign1s's contract: (B, T, K, 3) -> (B, T, hidden_size), T preserved.
        if self.backbone_type == "dsta":
            dsta_mask = frame_mask if (frame_mask is not None and self.dsta_mask_aware) else None
            features = self.dsta(poses, frame_mask=dsta_mask)
        else: features = self.cosign(poses)
        if frame_mask is None: frame_mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if not self.use_temporal_conv: return features, frame_mask, timestamps_s

        # Update attention mask after temporal conv (downsampled by factor ~4 with conv_type=2)
        features = self.temporal_conv(features) # (B, T', embed_dim)
        new_len = features.shape[1] # Calculate the temporal reduction factor
        frame_mask = F.adaptive_max_pool1d(frame_mask.float().unsqueeze(1), new_len).squeeze(1).bool()
        if timestamps_s is not None: timestamps_s = F.interpolate(
            timestamps_s.float().unsqueeze(1),
            size=new_len, mode="linear", align_corners=True
        ).squeeze(1)
        return features, frame_mask, timestamps_s


class VisualEncoder(nn.Module): # Visual encoder that projects pose features to mBART embedding space for Stage 2
    def __init__(self, emb_size: int, feature_size: int):
        super().__init__()
        self.src_emb = nn.Linear(feature_size, emb_size)
        self.bn_ac = nn.Sequential(nn.BatchNorm1d(emb_size), nn.ReLU(inplace=True))
        for module in self.modules(): # Initialize weights
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None: nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.src_emb(src)
        return self.bn_ac(src.permute(0, 2, 1)).permute(0, 2, 1)


class GFSLTVisualBackbone(nn.Module): # Reusable visual front end: CoSign -> VLP projection -> mBART encoder.
    def __init__(self, config: GFSLTConfig, mbart: MBartForConditionalGeneration | None = None):
        super().__init__()
        self.config = config
        self.mbart = mbart if mbart is not None else load_gfslt_mbart(config.mbart_name)
        self.pose_frontend = PoseFeatureExtractor(config)
        self.sign_emb = VisualEncoder(self.mbart.config.d_model, self.pose_frontend.output_dim)
        # sqrt(d_model) brings pose features into the token-embedding magnitude the pretrained mBART
        # encoder expects (GFSLT-VLP gloss_free_model.embed_scale); 1.0 = prior behaviour. See GFSLTConfig.
        self.input_embed_scale = self.mbart.config.d_model**0.5 if config.scale_embedding else 1.0

    def freeze_pose_backbone(self, freeze_projection: bool = False) -> int:
        """Freeze the from-scratch CoSign pose backbone for stage 2. On a small corpus, end-to-end fine-tuning
        of the from-scratch GCN can be the main overfitting route. The mBART encoder/decoder stay trainable;
        `freeze_projection=True` additionally freezes the VLP sign projection. Returns # frozen params."""
        frozen = 0
        modules = [self.pose_frontend] + ([self.sign_emb] if freeze_projection else [])
        for module in modules:
            for p in module.parameters():
                if p.requires_grad:
                    p.requires_grad_(False)
                    frozen += p.numel()
        return frozen

    def extract_post_vlp(
        self, poses: torch.Tensor, frame_mask: torch.Tensor,
        timestamps_s: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        features, mask, timestamps = self.pose_frontend(poses, frame_mask, timestamps_s)
        post_vlp = self.sign_emb(features) * self.input_embed_scale
        return post_vlp, mask, timestamps

    def encode(
        self, poses: torch.Tensor, frame_mask: torch.Tensor,
        timestamps_s: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        post_vlp, mask, timestamps = self.extract_post_vlp(poses, frame_mask, timestamps_s)
        if self.config.bypass_mbart_encoder: # Decoder cross-attends to per-frame VLP features directly (no text-encoder in pose path).
            return post_vlp, post_vlp, mask.long(), timestamps
        encoder = self.mbart.model.encoder(inputs_embeds=post_vlp, attention_mask=mask.long(), return_dict=True)
        return post_vlp, encoder.last_hidden_state, mask.long(), timestamps


def resolve_decoder_start_id(tokenizer) -> int | None:
    """Target-language decoder start for mBART generation.

    HF mBART's shift_tokens_right (transformers/models/mbart/modeling_mbart.py) wraps the LAST non-pad token of the labels 
    — the language code — to position 0, so the decoder is trained to start from the language code. The trimmed cc25 config 
    carries no decoder_start_token_id and HF generate would silently fall back to <s>: a train/generation mismatch.
    """
    lang = getattr(tokenizer, "tgt_lang", None) or getattr(tokenizer, "src_lang", None)
    if not lang: return None
    mapping = getattr(tokenizer, "lang_code_to_id", None)
    if mapping and lang in mapping: return int(mapping[lang])
    lang_id = tokenizer.convert_tokens_to_ids(lang)
    return int(lang_id) if lang_id is not None and lang_id != tokenizer.unk_token_id else None


class CleanARSLTModel(nn.Module): # Clean pre-trimmed GFSLT-style AR baseline.
    def __init__(self, config: GFSLTConfig, decoder_start_token_id: int | None = None, label_smoothing: float = 0.0):
        super().__init__()
        mbart = load_gfslt_mbart(config.mbart_name)
        self.visual = GFSLTVisualBackbone(config, mbart=mbart)
        self.mbart = self.visual.mbart
        # Language-code generation start (see resolve_decoder_start_id); None keeps HF defaults.
        self.decoder_start_token_id = decoder_start_token_id
        # GFSLT-VLP's translation CE uses label_smoothing=0.2 (train_slt.py); HF mBART's internal loss has none.
        self.label_smoothing = float(label_smoothing)

    def forward(
        self, poses: torch.Tensor, frame_mask: torch.Tensor,
        labels: torch.Tensor, timestamps_s: torch.Tensor | None = None,
    ):
        _, enc_hidden, enc_mask, _ = self.visual.encode(poses, frame_mask, timestamps_s=timestamps_s)
        out = self.mbart(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
            attention_mask=enc_mask, labels=labels, return_dict=True,
        )
        if self.label_smoothing > 0.0:
            # Recompute the CE with label smoothing from the (already label-aligned) logits, matching GFSLT-VLP's
            # external criterion. -100 is the collator's pad-ignore index, identical to HF's default ignore.
            out.loss = F.cross_entropy(
                out.logits.reshape(-1, out.logits.shape[-1]),
                labels.reshape(-1).to(out.logits.device),
                ignore_index=-100, label_smoothing=self.label_smoothing,
            )
        return out

    @torch.no_grad()
    def generate(
        self, poses: torch.Tensor, frame_mask: torch.Tensor,
        timestamps_s: torch.Tensor | None = None, max_new_tokens: int = 128,
        decoder_start_token_id: int | None = None, **kwargs,
    ) -> torch.Tensor:
        _, enc_hidden, enc_mask, _ = self.visual.encode(poses, frame_mask, timestamps_s=timestamps_s)
        return self.mbart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden), attention_mask=enc_mask, max_new_tokens=max_new_tokens,
            decoder_start_token_id=decoder_start_token_id if decoder_start_token_id is not None else self.decoder_start_token_id,
            **kwargs,
        )

    @torch.no_grad()
    def generate_with_confidence(
        self, poses: torch.Tensor, frame_mask: torch.Tensor, timestamps_s: torch.Tensor | None = None,
        max_new_tokens: int = 128, decoder_start_token_id: int | None = None, per_token_conf: bool = False, **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate, then score the model's REAL confidence in its own output.

        Returns (sequences, mean_confidence[B]) where mean_confidence[i] is the mean per-token softmax prob over the non-pad tokens 
        row i produced. Token_confidence[i, t] is the softmax prob the model assigns to sequences[i, t] (the start slot is 1.0), 
        via 1 teacher-forced pass (decode-strategy-agnostic: identical for greedy & beam).
        """
        _, enc_hidden, enc_mask, _ = self.visual.encode(poses, frame_mask, timestamps_s=timestamps_s)
        sequences = self.mbart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden), attention_mask=enc_mask, max_new_tokens=max_new_tokens,
            decoder_start_token_id=decoder_start_token_id if decoder_start_token_id is not None else self.decoder_start_token_id,
            **kwargs,
        )
        if sequences.shape[1] < 2: return sequences, torch.ones(sequences.shape[0], dtype=torch.float32, device=sequences.device)
        # HF generate() EXPANDS the encoder_outputs it receives to batch*num_beams in place (16 -> 16*4=64),
        # so it cannot be reused. Build a fresh BaseModelOutput from the un-expanded enc_hidden for this pass.
        out = self.mbart(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden), attention_mask=enc_mask, max_new_tokens=max_new_tokens,
            decoder_start_token_id=decoder_start_token_id if decoder_start_token_id is not None else self.decoder_start_token_id,
            **kwargs,
        )
        tok_prob = out.logits.log_softmax(dim=-1).gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1).exp()  # (B, L-1)
        start = torch.ones((sequences.shape[0], 1), dtype=tok_prob.dtype, device=tok_prob.device)
        token_conf = torch.cat([start, tok_prob], dim=1)  # (B, L) aligned with sequences
        valid = (sequences[:, 1:] != int(self.mbart.config.pad_token_id))
        if per_token_conf: return sequences, token_conf
        return sequences, (token_conf[:, 1:] * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)