from __future__ import annotations
from collections import namedtuple
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bio_head import ClassifierHead, RoPETransformerEncoderLayer, chunked_rope_encode

ConvDef = namedtuple("ConvDef", ["in_channels", "out_channels", "kernel_size", "stride"])


class Unsqueeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)


class PoseEncoderUNetBlock(nn.Module): # Two-sided temporal UNet block copied from the Moryossef 2026 architecture.
    def __init__(self, input_size: int, output_size: int, convolutions: List[ConvDef]):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        for conv in convolutions:
            if conv.kernel_size % 2 != 1: raise ValueError("Temporal convolution kernel size must be odd")
            if conv.stride & (conv.stride - 1) != 0: raise ValueError("Stride must be a power of 2")
            self.encoder_layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=conv.in_channels, out_channels=conv.out_channels,
                    kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.kernel_size // 2,
                ),
                nn.BatchNorm1d(conv.out_channels),
                nn.SiLU(),
            ))

        stride_to_output_pad = {1: 0, 2: 1, 4: 3, 8: 4}
        self.decoder_layers = nn.ModuleList()
        for conv in reversed(convolutions):
            if conv.stride not in stride_to_output_pad: 
                raise ValueError(f"Stride {conv.stride} not supported for output padding. Manually add it!")
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=conv.out_channels, out_channels=conv.in_channels,
                    kernel_size=conv.kernel_size, stride=conv.stride,
                    padding=conv.kernel_size // 2, output_padding=stride_to_output_pad[conv.stride],
                ),
                nn.BatchNorm1d(conv.in_channels),
                nn.SiLU(),
            ))
        self.fc = nn.Linear(input_size, output_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, input_size, input_channels = x.shape
        # Rearrange to [batch_size * input_size, input_channels, sequence_length]
        x = x.permute(0, 2, 3, 1).contiguous().view(batch * input_size, input_channels, seq_len)
        skip_values = []
        for layer in self.encoder_layers: # Encode values with reducing temporal dimension
            x = layer(x)
            skip_values.append(x)

        for layer in self.decoder_layers: # Decode values with increasing temporal dimension, using skip connections
            skip = skip_values.pop()
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                left = diff // 2
                x = F.pad(x, (left, diff - left)) # Ensure leftover goes on the right
            elif diff < 0: x = x[..., : skip.shape[-1]]
            x = layer(x + skip)

        if x.shape[-1] > seq_len: x = x[..., :seq_len]
        elif x.shape[-1] < seq_len: x = F.pad(x, (0, seq_len - x.shape[-1]))
        _, channels, new_seq_len = x.shape # [batch_size * input_size, output_channels, sequence_length]
        x = x.view(batch, input_size, channels, new_seq_len).permute(0, 3, 1, 2)
        x = x.mean(dim=-1) # Average pool the channel output [batch_size, sequence_length, input_size]
        return self.fc(x)


class MoryossefSegmenter(nn.Module):
    """CNN-medium-attn segmenter with a phrase BIO head.

    Moryossef 2026 jointly trains sign (sub-sentence) and phrase (sentence) BIO heads, but the sign head needs sign-level segment annotations. 
    We retrain on YouTube-SL-25, which only has sentence/caption boundaries, so the sign head has no supervision and is omitted. 
    Analysis A and the RQ2 pipeline-floor baseline consume the phrase head only.
    """
    def __init__(
        self, pose_dims: tuple[int, int] = (69, 3), hidden_dim: int = 384, encoder_depth: int = 4, num_classes: int = 4,
        attn_nhead: int = 8, attn_ff_mult: int = 2, attn_dropout: float = 0.1, num_frames: int = 1024,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.frame_cnn = nn.Sequential(
            PoseEncoderUNetBlock(input_size=pose_dims[0], output_size=hidden_dim, convolutions=[
                ConvDef(in_channels=pose_dims[1], out_channels=16, kernel_size=5, stride=1),
                ConvDef(in_channels=16, out_channels=32, kernel_size=11, stride=1),
                ConvDef(in_channels=32, out_channels=64, kernel_size=21, stride=2),
            ]),
            Unsqueeze(dim=-1),
            PoseEncoderUNetBlock(input_size=hidden_dim, output_size=hidden_dim, convolutions=[
                ConvDef(in_channels=1, out_channels=16, kernel_size=5, stride=1),
                ConvDef(in_channels=16, out_channels=32, kernel_size=11, stride=2),
                ConvDef(in_channels=32, out_channels=64, kernel_size=21, stride=2),
                ConvDef(in_channels=64, out_channels=128, kernel_size=21, stride=2),
            ])
        )
        self.input_norm = nn.RMSNorm(hidden_dim)
        self.encoder_attn = nn.ModuleList([RoPETransformerEncoderLayer(
            hidden_dim=hidden_dim, nhead=attn_nhead,
            dim_feedforward=hidden_dim * attn_ff_mult, dropout=attn_dropout,
        ) for _ in range(encoder_depth)])
        self.phrase_bio_head = ClassifierHead(hidden_dim, num_classes)

    def encode(self, pose_data: torch.Tensor, timestamps_s: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.frame_cnn(pose_data)
        x = self.input_norm(feats.float()).to(feats.dtype)  # fp32 RMSNorm under autocast (see bio_head note)
        if timestamps_s is None: # Assume 50fps when no timestamps provided (1/50s per frame → *50 → 1 unit/frame).
            timestamps_s = torch.arange(x.shape[1], device=x.device, dtype=torch.float32) / RoPETransformerEncoderLayer.REFERENCE_FPS
        # Process in training-size chunks so eval context matches the training distribution.
        return chunked_rope_encode(self.encoder_attn, x, timestamps_s, self.num_frames)

    def forward(self, pose_data: torch.Tensor, timestamps_s: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        encoded = self.encode(pose_data, timestamps_s=timestamps_s)
        return {"phrase": self.phrase_bio_head(encoded)}
