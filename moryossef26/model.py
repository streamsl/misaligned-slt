from __future__ import annotations
from collections import namedtuple
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bio_head import ClassifierHead, RoPETransformerEncoderLayer, chunked_rope_encode

ConvDef = namedtuple("ConvDef", ["in_channels", "out_channels", "kernel_size", "stride"])
_FC_KEY = "frame_cnn.0.fc.weight"

class Unsqueeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)


class PoseEncoderUNetBlock(nn.Module): # Temporal UNet block, copied from Moryossef 2026.
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
                    kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.kernel_size // 2,
                ),
                nn.BatchNorm1d(conv.in_channels),
                nn.SiLU(),
            ))
        self.fc = nn.Linear(input_size, output_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, input_size, input_channels = x.shape # → [batch * input_size, input_channels, seq_len]
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
            x = layer(x + skip)

        _, channels, new_seq_len = x.shape # [batch * input_size, out_channels, seq_len]
        x = x.view(batch, input_size, channels, new_seq_len).permute(0, 3, 1, 2)
        x = x.mean(dim=-1) # pool channels → [batch, seq_len, input_size]
        return self.fc(x)


class MoryossefSegmenter(nn.Module):
    """CNN-medium-attn segmenter with a phrase BIO head.

    The external segmenter for calibration and the RQ2 cascade, not the in-system BIO head. Raw keypoints through a UNet are a different 
    input space from the in-system Uni-Sign features. (docs/membership_gate.md §4, shared components table).

    Moryossef 2026's sign (sub-sentence) head needs sign-level annotations; our corpora carry only sentence boundaries, so it's omitted.
    1 declared deviation remains: the reference zero-pads a final partial chunk to `num_frames` and attends over the pad, while 
    `chunked_rope_encode` runs short chunk. Outputs are bit-identical on whole multiples of `num_frames` and differ only on that tail.
    """
    def __init__(
        self, pose_dims: tuple[int, int] = (50, 6), hidden_dim: int = 384, encoder_depth: int = 4, num_classes: int = 4,
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
        if timestamps_s is None: # No timestamps → assume 50fps (1/50s per frame → *50 → 1 unit/frame).
            timestamps_s = torch.arange(x.shape[1], device=x.device, dtype=torch.float32) / RoPETransformerEncoderLayer.REFERENCE_FPS
        # Training-size chunks so eval context matches the train distribution. Chunking lives INSIDE the model;
        # the inference wrapper just calls forward (moryossef26/infer.py).
        return chunked_rope_encode(self.encoder_attn, x, timestamps_s, self.num_frames)

    def forward(
        self, pose_data: torch.Tensor, frame_mask: torch.Tensor | None = None, timestamps_s: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        # frame_mask is accepted and ignored so the wrapper can call this and BioS1Model with 1 signature. No attention pad mask — 
        # faithful to Moryossef 2026 (README:66: his mask "changes training distribution in a way that does not match inference"), 
        # and safe here: chunks are near-uniform 1024 frames. The in-system RoPE head DOES key-mask padding.
        encoded = self.encode(pose_data, timestamps_s=timestamps_s)
        return {"phrase": self.phrase_bio_head(encoded)}


def load_moryossef_pretrained(model: "MoryossefSegmenter", checkpoint: str | Path) -> dict[str, int]:
    """Warm-start MoryossefSegmenter from the released Moryossef 2026 weights.

    The arm reads released input contract (moryossef26.dataset.to_release_coords: their 50 landmarks, their standardisation, velocity 
    after), so every tensor including keypoint-indexed `frame_cnn.0.fc` loads by shape. 2 renames and 1 drop: `sentence_bio_head.*` 
    → `phrase_bio_head.*`, `sign_bio_head.*` dropped (no sign-level supervision on our corpora), `inv_freq` skipped (see below). 
    This is a warm start, NOT a zero-shot evaluation — `train-moryossef` fine-tunes afterwards. For the untrained measurement, see 
    `eval.py --segmenter-init released`, which reads same coordinates and is a negative control, not a published zero-shot number: 
    the release was trained on MediaPipe poses this project doesn't store. Returns a {loaded, renamed, dropped_sign} summary."""
    from safetensors.torch import load_file
    raw = load_file(str(checkpoint))
    renamed = 0
    src: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("sign_bio_head."): continue
        tk = k.replace("sentence_bio_head.", "phrase_bio_head.")
        if tk != k: renamed += 1
        src[tk] = v

    model_sd = model.state_dict()
    loaded = 0
    for k, tgt in model_sd.items():
        v = src.get(k)
        # inv_freq is derived from the RoPE formula, and the release stores it in BF16: copying it costs up to
        # 1.64 rad of phase at t=1024 reference-fps units. Keep our fp32 buffer.
        if k.endswith("inv_freq"): continue
        if v is not None and tuple(v.shape) == tuple(tgt.shape):
            tgt.copy_(v.to(tgt.dtype))  # BF16 checkpoint → our fp32
            loaded += 1

    # The arm reads THEIR keypoint layout (configs/moryossef26.yaml pose_joints: 50), so every tensor including
    # the keypoint-indexed fc loads by shape. A differing count means a config that this contract cannot serve.
    model.load_state_dict(model_sd, strict=True)
    their_fc, our_fc = src.get(_FC_KEY), model_sd[_FC_KEY]
    if their_fc is not None and tuple(our_fc.shape) != tuple(their_fc.shape): raise ValueError(
        f"{_FC_KEY} is {tuple(our_fc.shape)} but the release is {tuple(their_fc.shape)}: this arm reads the released "
        f"50-keypoint layout (moryossef26.dataset.RELEASE_KP_IDX). Set pose_joints: 50."
    )
    summary = {"loaded": loaded, "renamed": renamed, "dropped_sign": sum(1 for k in raw if k.startswith("sign_bio_head."))}
    print(f"segmenter | warm-started from {checkpoint}: loaded {loaded}, renamed sentence→phrase {renamed}, "
          f"dropped sign head {summary['dropped_sign']} (their 50-keypoint layout, every tensor by shape)", flush=True)
    return summary
