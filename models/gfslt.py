from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartConfig, MBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from backbones.cosign import CoSign1s

try: from safetensors.torch import load_file as load_safetensors
except Exception: load_safetensors = None # pragma: no cover - optional dependency path


@dataclass
class GFSLTConfig:
    embed_dim: int = 1024
    hidden_size: int = 1024
    temporal_kernel: int = 3
    mbart_name: str = "facebook/mbart-large-cc25"
    num_keypoints: int = 77 # Selected keypoints from CoSign
    input_channels: int = 3 # x, y, confidence
    use_temporal_conv: bool = False


def load_gfslt_mbart(model_path: str) -> MBartForConditionalGeneration:
    # Load full mBART even when the trimmed checkpoint was saved decoder-only.
    path = Path(model_path)
    if not path.exists(): return MBartForConditionalGeneration.from_pretrained(model_path)
    config = MBartConfig.from_pretrained(model_path)
    config.is_encoder_decoder = True
    config.is_decoder = False
    model = MBartForConditionalGeneration(config)

    safetensors_path = path / "model.safetensors"
    bin_path = path / "pytorch_model.bin"
    if safetensors_path.exists() and load_safetensors is not None: state_dict = load_safetensors(str(safetensors_path))
    elif bin_path.exists(): state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    else: return MBartForConditionalGeneration.from_pretrained(model_path)

    if "model.decoder.embed_tokens.weight" in state_dict and "model.shared.weight" not in state_dict:
        state_dict["model.shared.weight"] = state_dict["model.decoder.embed_tokens.weight"]
    model.load_state_dict(state_dict, strict=False)
    return model


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
        features = self.cosign(poses) # CoSign expects (B, T, K, 3) and outputs (B, T, hidden_size)
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
        self.input_embed_scale = 1.0

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
        encoder = self.mbart.model.encoder(inputs_embeds=post_vlp, attention_mask=mask.long(), return_dict=True)
        return post_vlp, encoder.last_hidden_state, mask.long(), timestamps


class CleanARSLTModel(nn.Module): # Clean pre-trimmed GFSLT-style AR baseline.
    def __init__(self, config: GFSLTConfig):
        super().__init__()
        mbart = load_gfslt_mbart(config.mbart_name)
        self.visual = GFSLTVisualBackbone(config, mbart=mbart)
        self.mbart = self.visual.mbart

    def forward(
        self, poses: torch.Tensor, frame_mask: torch.Tensor,
        labels: torch.Tensor, timestamps_s: torch.Tensor | None = None,
    ):
        _, enc_hidden, enc_mask, _ = self.visual.encode(poses, frame_mask, timestamps_s=timestamps_s)
        return self.mbart(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
            attention_mask=enc_mask, labels=labels, return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self, poses: torch.Tensor, frame_mask: torch.Tensor,
        timestamps_s: torch.Tensor | None = None, max_new_tokens: int = 128,
        decoder_start_token_id: int | None = None, **kwargs,
    ) -> torch.Tensor:
        _, enc_hidden, enc_mask, _ = self.visual.encode(poses, frame_mask, timestamps_s=timestamps_s)
        return self.mbart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
            attention_mask=enc_mask, max_new_tokens=max_new_tokens,
            decoder_start_token_id=decoder_start_token_id, **kwargs,
        )