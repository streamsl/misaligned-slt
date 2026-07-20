from __future__ import annotations
from collections import namedtuple
from pathlib import Path
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

    This is the INDEPENDENT analysis instrument (Analysis A/B + RQ2 cascaded baseline), deliberately NOT the
    in-system BIO head: it reads RAW pose keypoints (+ velocity) through a UNet CNN, a *different input space* from
    the in-system head's Uni-Sign encoder features (docs/membership_gate.md §1.4: "weights do not transfer... the
    analysis segmenter keeps its own job"). That input-space independence is what makes Analysis A non-circular.

    Moryossef 2026 jointly trains sign (sub-sentence) and phrase (sentence) BIO heads, but the sign head needs
    sign-level segment annotations. We retrain on YouTube-SL-25 / concatenated corpora, which only carry
    sentence/caption boundaries, so the sign head has no supervision and is omitted — only the phrase head is used.
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
        # Process in training-size chunks so eval context matches the training distribution (chunking lives INSIDE
        # the model via chunked_rope_encode — the inference wrapper just calls forward; see moryossef26/infer.py).
        return chunked_rope_encode(self.encoder_attn, x, timestamps_s, self.num_frames)

    def forward(
        self, pose_data: torch.Tensor, frame_mask: torch.Tensor | None = None, timestamps_s: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        # frame_mask is accepted (and ignored) so the inference wrapper can call MoryossefSegmenter and BioS1Model
        # with 1 uniform signature. No attention pad mask here — faithful to Moryossef 2026 (README:66: his mask
        # "changes training distribution in a way that does not match inference"; removing it was his fix), and
        # sound in THIS model's regime: chunks are near-uniform 1024 frames, so batch padding is negligible.
        # (The in-system RoPE head DOES key-mask padding — its window batches span 0.5s-18s; see models/bio_head.py.)
        encoded = self.encode(pose_data, timestamps_s=timestamps_s)
        return {"phrase": self.phrase_bio_head(encoded)}


_FC_KEY = "frame_cnn.0.fc.weight"
# Part boundaries in each model's keypoint ordering, for the frame_cnn.0.fc [384, num_kp] input projection (column
# j = keypoint j). Their reduce_holistic(pose + L/R hands): 8 upper-body + 21 + 21 = 50 (datasets/common.py body-part
# dropout zeros [8:29] and [29:50]). Ours: poses.normalize_keypoints_unisign order [body 9 | L 21 | R 21 | face 18].
_THEIR_HANDS = {"lhand": (8, 29), "rhand": (29, 50)}
_OUR_HANDS = {"lhand": (9, 30), "rhand": (30, 51)}


def _transfer_hand_fc_columns(their_fc: torch.Tensor, our_fc: torch.Tensor) -> int:
    """Copy left/right-hand fc columns their→our (in place on our_fc). MediaPipe and DWPose hands are both the canonical 21-point 
    topology in SAME order (wrist, thumb..pinky), so the columns map index-for-index. Body (8 MediaPipe upper-body vs 9 COCO body 
    — different points) and face (they have none) are left at our init."""
    cols = 0
    for part, (ts, te) in _THEIR_HANDS.items():
        os_, oe = _OUR_HANDS[part]
        if (te - ts) == (oe - os_):
            our_fc[:, os_:oe].copy_(their_fc[:, ts:te].to(our_fc.dtype))
            cols += oe - os_
    return cols # Return the number of columns transferred (42 for released 50→69 checkpoint)


def load_moryossef_pretrained(model: "MoryossefSegmenter", checkpoint: str | Path) -> dict[str, int]:
    """CROSS-MODALITY warm-start of MoryossefSegmenter from the released Moryossef 2026 weights.

    The released checkpoint is a DIFFERENT POSE MODALITY, so this is a warm-start, NOT zero-shot (fine-tune with
    `train-segmenter` afterwards). Their model: MediaPipe-Holistic 50 keypoints (8 upper-body + 21+21 hands,
    `reduce_holistic`), 3D XYZ + 3D velocity = 6 channels, `normalize_mean_std`. Ours: DWPose/Uni-Sign 69 keypoints
    (incl. 18 face), 2D + confidence + velocity, bbox-normalized. The bridge:
      - `sentence_bio_head.*` → `phrase_bio_head.*`  (their sentence head IS our phrase head)
      - `sign_bio_head.*`     → DROPPED  (no sign-level supervision on our corpora)
      - `frame_cnn.0.fc.weight` [384,50] vs our [384,69] — the only keypoint-count-dependent tensor. We transfer the
        LEFT/RIGHT-HAND columns index-for-index (both are the canonical 21-pt hand topology, same order → 42/69
        columns), and reinit only the body (8→9, different points) + face (they have none) columns. fc.bias loads.
      - everything else (temporal UNet convs incl. BatchNorm stats, RoPE encoder_attn, input_norm, phrase head)
        loads by shape — transferring the learned temporal-boundary inductive bias.
    The conv channel semantics differ (their z vs our confidence) and the normalization differs, so the front-end
    still adapts during fine-tuning; the transfer value is the deep temporal filters + attention + BIO head + hands.
    Returns a {loaded, renamed, dropped_sign, fc_hand_cols, reinit} summary.
    """
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
        if v is not None and tuple(v.shape) == tuple(tgt.shape):
            tgt.copy_(v.to(tgt.dtype))  # BF16 checkpoint → our fp32
            loaded += 1

    # frame_cnn.0.fc.weight: shapes differ (50 vs 69 kp), so the loop above left it at our init. Transfer the hand
    # columns; body+face columns stay reinit. Guarded on the exact released geometry (their 50, our 69).
    fc_hand_cols = 0
    their_fc = src.get(_FC_KEY)
    our_fc = model_sd[_FC_KEY]
    if their_fc is not None and their_fc.shape[0] == our_fc.shape[0] and their_fc.shape[1] == 50 and our_fc.shape[1] == 69:
        fc_hand_cols = _transfer_hand_fc_columns(their_fc, our_fc)

    model.load_state_dict(model_sd, strict=True)
    fc_reinit = "partial (body+face)" if fc_hand_cols else "full"
    summary = {
        "loaded": loaded, "renamed": renamed, "dropped_sign": sum(1 for k in raw if k.startswith("sign_bio_head.")),
        "fc_hand_cols": fc_hand_cols, "reinit": 1,  # frame_cnn.0.fc.weight (partially, if hands transferred)
    }
    print(f"segmenter | warm-started from {checkpoint} (cross-modality: MediaPipe-50 → DWPose-69): "
          f"loaded {loaded}, renamed sentence→phrase {renamed}, dropped sign head {summary['dropped_sign']}, "
          f"{_FC_KEY} hand-columns transferred {fc_hand_cols}/69 ({fc_reinit} reinit)", flush=True)
    return summary
