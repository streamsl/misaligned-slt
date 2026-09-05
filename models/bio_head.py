from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BIOHeadOutput:
    phrase_logits: torch.Tensor
    logits: torch.Tensor
    hidden_states: torch.Tensor


class RoPETransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with Rotary Position Embedding (RoPE).

    RoPE makes attention scores depend on relative position only, so it generalises across chunk boundaries.
    Timestamps are in seconds, scaled by REFERENCE_FPS=50 into "50fps frame units".
    """
    REFERENCE_FPS = 50.0

    def __init__(self, hidden_dim: int = 384, nhead: int = 8, dim_feedforward: int = 768, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % nhead != 0: raise ValueError(f"hidden_dim={hidden_dim} must be divisible by nhead={nhead}")
        self.nhead = int(nhead)
        self.head_dim = int(hidden_dim // nhead)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _compute_rope(self, timestamps_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if timestamps_s.dim() == 1: timestamps_s = timestamps_s.unsqueeze(0)
        freqs = (timestamps_s * self.REFERENCE_FPS).unsqueeze(-1).float() * self.inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)

    def forward(self, x: torch.Tensor, timestamps_s: torch.Tensor | None = None, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, frames, hidden_dim = x.shape
        # Assume 50fps when no timestamps provided (1/50s per frame → *50 → 1 unit/frame).
        if timestamps_s is None: timestamps_s = torch.arange(frames, device=x.device, dtype=torch.float32) / self.REFERENCE_FPS
        if timestamps_s.dim() == 1: timestamps_s = timestamps_s.unsqueeze(0).expand(batch, -1)
        timestamps_s = timestamps_s.to(device=x.device)
        cos, sin = self._compute_rope(timestamps_s)

        # Key-padding mask (True = real frame): excluding padded KEYS makes training attention match dense unbatched inference (collator 
        # contract, data/batch.py). NOT the query-side mask Moryossef 2026 removed (README:66) — his uniform 1024-frame chunks barely 
        # pad; our 0.5s–18s windows would leave ghost keys at train and none at inference, the exact mismatch his rule targets.
        attn_mask = None
        if key_mask is not None:
            km = key_mask.to(device=x.device, dtype=torch.bool)
            km = km | ~km.any(dim=-1, keepdim=True)  # fully-padded row: attend anywhere; outputs are loss-ignored
            attn_mask = km[:, None, None, :]

        # fp32 RMSNorm (cast back): autocast excludes rms_norm, so bf16 input + fp32 weight warns and falls back to
        # the unfused kernel; fp32 matches autocast's LayerNorm semantics.
        h = self.norm1(x.float()).to(x.dtype)
        qkv = self.qkv(h).reshape(batch, frames, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        # Residual (attention-output) dropout, not SDPA's dropout_p: the fused MPS SDPA kernel raises
        # NotImplementedError on in-kernel dropout; negligible difference at p=0.1.
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.attn_drop(self.out_proj(attn.transpose(1, 2).reshape(batch, frames, hidden_dim)))
        return x + self.ffn(self.norm2(x.float()).to(x.dtype))


def chunked_rope_encode(
    layers: nn.ModuleList, x: torch.Tensor, timestamps_s: torch.Tensor | None,
    chunk_size: int | None, key_mask: torch.Tensor | None = None, overlap: bool = False,
) -> torch.Tensor:
    """Run pre-norm RoPE layers over fixed-size chunks (Moryossef chunked inference).

    Shared by `RoPEBIOHead` (BioS1Model) and the analysis `MoryossefSegmenter.encoder_attn`. Each chunk attends only within itself;
    RoPE relative time in seconds keeps positions consistent across boundaries, so train-size chunked eval matches the training
    distribution. `chunk_size=None` (or T <= chunk_size) is one full pass. Timestamps are dim-normalized up front, else 1-D inputs
    are dropped on the chunked path and fall back to the 50fps index assumption, breaking fps-augmented inputs.

    `overlap=True` — OVERLAP-STITCHED chunking for the S1 whole-video paths. Contiguous chunks give frames at a seam context on ONE 
    side only, so a sentence straddling a chunk boundary is decoded from two truncated halves — with ~18s chunks and ~4-6s sentences 
    that is a large fraction of sentences, and it is a decode ARTIFACT, not head error (the FSM never suffers it: its buffer follows 
    commits). Windows advance by half a chunk and each frame keeps the estimate from the window where it is most INTERIOR, so every 
    frame (stream edges aside) has at least a quarter-chunk of context on both sides. Window size stays the TRAINED extent — only the 
    stitching changes. The external Moryossef baseline keeps `overlap=False`: its released protocol is contiguous chunks.
    """
    if timestamps_s is not None:
        if timestamps_s.dim() == 1: timestamps_s = timestamps_s.unsqueeze(0).expand(x.shape[0], -1)
        timestamps_s = timestamps_s.to(device=x.device)

    if chunk_size is None or x.shape[1] <= int(chunk_size):
        for layer in layers: x = layer(x, timestamps_s, key_mask=key_mask)
        return x

    def run(start: int, end: int) -> torch.Tensor:
        chunk = x[:, start:end]
        chunk_ts = timestamps_s[:, start:end] if timestamps_s is not None else None
        chunk_mask = key_mask[:, start:end] if key_mask is not None else None
        for layer in layers: chunk = layer(chunk, chunk_ts, key_mask=chunk_mask)
        return chunk

    chunk_size, T = int(chunk_size), x.shape[1]
    if not overlap: return torch.cat([run(s, min(T, s + chunk_size)) for s in range(0, T, chunk_size)], dim=1)

    out = torch.empty_like(x)
    for start, end, write_from, keep_hi in overlap_windows(T, chunk_size):
        out[:, write_from:keep_hi] = run(start, end)[:, write_from - start : keep_hi - start]
    return out


def overlap_windows(T: int, chunk_size: int) -> list[tuple[int, int, int, int]]:
    """Overlap-stitched windows over T frames as (start, end, write_from, keep_hi): windows advance by half a chunk, each frame
    keeps the estimate from the window where it is most interior (a quarter-chunk of context on both sides), and the last window
    is right-aligned to the full trained extent. One rule for the RoPE chunking and for per-chunk pose normalization."""
    chunk_size, T = int(chunk_size), int(T)
    stride = max(1, chunk_size // 2)
    margin = (chunk_size - stride) // 2
    windows, write_from, start = [], 0, 0
    while write_from < T:
        end = min(T, start + chunk_size)
        if end == T: start = max(0, T - chunk_size)
        keep_hi = T if end == T else end - margin
        windows.append((start, end, write_from, keep_hi))
        write_from, start = keep_hi, start + stride
    return windows


def chunk_normalized_logits(forward, raw_poses: np.ndarray, timestamps: np.ndarray, chunk_size: int | None, device) -> torch.Tensor:
    """Whole-video BIO logits (1, T, C) with pose normalization PER CHUNK at the trained context. The Uni-Sign body scale is one
    scalar over the frames it is given and the head trained on per-window scales, so a whole-video scale is a frame the head never
    saw. `forward(poses (1,t,69,3), frame_mask (1,t), timestamps_s (1,t)) -> logits (1,t,C)`; windows and stitching follow
    `overlap_windows`, so every frame is decoded from the window it is most interior to."""
    from poses import normalize_keypoints_unisign
    if raw_poses.ndim != 3 or raw_poses.shape[1:] != (133, 3): raise ValueError(f"raw (T,133,3) poses expected, got {raw_poses.shape}")
    T = int(raw_poses.shape[0])
    windows = [(0, T, 0, T)] if chunk_size is None or T <= int(chunk_size) else overlap_windows(T, int(chunk_size))
    out = None
    for start, end, write_from, keep_hi in windows:
        poses = torch.as_tensor(normalize_keypoints_unisign(raw_poses[start:end]), dtype=torch.float32, device=device).unsqueeze(0)
        ts = torch.as_tensor(timestamps[start:end], dtype=torch.float32, device=device).unsqueeze(0)
        logits = forward(poses, torch.ones(poses.shape[:2], dtype=torch.bool, device=device), ts)
        if out is None: out = logits.new_empty((1, T, int(logits.shape[-1])))
        out[:, write_from:keep_hi] = logits[:, write_from - start : keep_hi - start]
    return out


class ClassifierHead(nn.Module): # Two-layer MLP classifier.
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalConvStem(nn.Module):
    """Residual stride-1 temporal Conv1d stem restoring the local boundary bias of Moryossef 2026's UNet front-end.

    Moryossef's ablations credit the UNet skip convolutions for boundary precision (EXPERIMENTS.md finding 23 "UNet
    skip connections are critical for sign boundary precision"; finding 8 "CNN naturally detects B"). This head reads
    pose tokens directly with no UNet, so without a stem it is a transformer/BiLSTM tagger — weak on B, which feeds
    the commit gate's I→O δ_enc stability and RQ2 tIoU. 2 conv layers, not the full UNet, to keep the head small.

    Stride 1 + odd kernel + 'same' padding keeps length, so per-frame BIO labels stay aligned. Per-position RMSNorm
    (channels only) is length- and batch-independent: no train/inference mismatch over the growing streaming buffer
    (unlike BatchNorm running stats or a time-pooling GroupNorm). conv_stem_layers=0 disables it (ablation).
    """
    def __init__(self, hidden_dim: int, num_layers: int = 2, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 != 1: raise ValueError(f"conv stem kernel must be odd to preserve length; got {kernel_size}")
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2) 
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:  # (B, T, D), key_mask (B,T) True=real
        # Zero padded positions before AND after each conv: masking the input alone is not enough, since each Conv1d
        # BIAS regenerates nonzero values at padded positions that the NEXT layer convolves into trailing real frames.
        # Held at 0, a real frame's window reads zeros over padding — as in streaming inference ('same' padding
        # zero-pads past sequence end): no train/infer mismatch, no dependence on batch pad length.
        m = None if key_mask is None else key_mask.to(x.dtype).unsqueeze(-1)  # (B, T, 1)
        if m is not None: x = x * m
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x.transpose(1, 2)).transpose(1, 2)            # (B, T, D): local temporal mix
            h = self.dropout(F.gelu(norm(h.float()).to(h.dtype)))  # fp32 RMSNorm under autocast (see layer note)
            if m is not None: h = h * m                            # kill conv-bias values at padded positions
            x = x + h                                              # residual; length preserved
        return x


class RoPEBIOHead(nn.Module):
    """Phrase-level Moryossef-style BIO head over per-frame pose-token features.

    `.logits` aliases the phrase BIO logits the FSM consumes. Moryossef 2026's extra *sign* (sub-sentence) head needs sign-level 
    segment annotations; YouTube-SL-25 has only sentence/caption timestamps, so it is dropped rather than left as an unsupervised 
    dead branch. A residual `TemporalConvStem` precedes the RoPE layers to recover the boundary bias this UNet-less head lacks.
    """
    def __init__(
        self, input_dim: int, hidden_dim: int = 384, depth: int = 4, nhead: int = 8, ff_mult: int = 2, dropout: float = 0.1,
        num_classes: int = 4, chunk_size: int | None = None, conv_stem_layers: int = 2, conv_stem_kernel: int = 5,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        # Runtime flag, set by whole-video S1 paths (moryossef26.infer.whole_video_logits, eval.run_offline): overlap-stitched chunking.
        # False for training (windows fit one chunk; the flag is inert) and for the faithful Moryossef baseline.
        self.chunk_overlap = False
        self.input_proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.RMSNorm(hidden_dim)
        self.conv_stem = TemporalConvStem(
            hidden_dim, num_layers=conv_stem_layers, kernel_size=conv_stem_kernel, dropout=dropout,
        ) if conv_stem_layers > 0 else None
        self.layers = nn.ModuleList([RoPETransformerEncoderLayer(
            hidden_dim=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * ff_mult, dropout=dropout,
        ) for _ in range(depth)])
        self.phrase_bio_head = ClassifierHead(hidden_dim, num_classes)

    def encode(
        self, features: torch.Tensor, timestamps_s: torch.Tensor | None = None, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        proj = self.input_proj(features)
        x = self.input_norm(proj.float()).to(proj.dtype)  # fp32 RMSNorm (see layer note)
        # Conv stem runs on the full sequence BEFORE RoPE chunking: local/translation-equivariant, so it does not
        # reintroduce the absolute-position issue chunking eliminates. Mask-aware (see TemporalConvStem).
        if self.conv_stem is not None: x = self.conv_stem(x, key_mask=frame_mask)
        return chunked_rope_encode(self.layers, x, timestamps_s, self.chunk_size, key_mask=frame_mask, overlap=bool(self.chunk_overlap))

    def forward(
        self, features: torch.Tensor, timestamps_s: torch.Tensor | None = None, frame_mask: torch.Tensor | None = None
    ) -> BIOHeadOutput:
        hidden = self.encode(features, timestamps_s=timestamps_s, frame_mask=frame_mask)
        phrase_logits = self.phrase_bio_head(hidden)
        return BIOHeadOutput(phrase_logits=phrase_logits, logits=phrase_logits, hidden_states=hidden)
