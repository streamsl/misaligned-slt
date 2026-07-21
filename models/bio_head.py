from __future__ import annotations
from dataclasses import dataclass

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

    RoPE rotates Q and K by position-dependent angles so that attention scores depend only on relative position, 
    not absolute — generalises well across chunk boundaries during chunked inference.

    Timestamps are expected in seconds; they are scaled by reference_fps=50 internally so that relative positions are 
    expressed in "50fps frame units" (i.e. 2 frames 0.02s apart → relative position 1, same as consecutive frames at 50fps).
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
        # Scale seconds → 50fps-equivalent frame units before computing frequencies.
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

        # Key-padding mask (True = real frame). Excluding padded KEYS makes training attention semantics identical to dense unbatched 
        # inference ("attend to exactly the real frames") — the collator's stated contract (data/batch.py: "Padded frames stay masked 
        # (attention)"). This is NOT the query/"causal-style" padding mask Moryossef 2026 removed (README:66): his regime (uniform 
        # 1024-frame chunks) has negligible padding, so REMOVING mask achieved train/inference consistency there; our window batches 
        # span 0.5s–18s, where UNMASKED ghost keys are heavy at training & absent at inference — the exact mismatch his rule targets.
        attn_mask = None
        if key_mask is not None:
            km = key_mask.to(device=x.device, dtype=torch.bool)
            km = km | ~km.any(dim=-1, keepdim=True)  # all-padding row (fully-padded chunk): attend anywhere; outputs are loss-ignored
            attn_mask = km[:, None, None, :]

        # RMSNorm in fp32 (cast back after): autocast keeps LayerNorm in fp32 but not rms_norm, so a
        # bf16 input meets the fp32 weight and PyTorch warns + falls back to the unfused kernel.
        # Computing the norm in fp32 matches autocast's LayerNorm semantics and silences the warning.
        h = self.norm1(x.float()).to(x.dtype)
        qkv = self.qkv(h).reshape(batch, frames, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        # Dropout is applied to the attention OUTPUT (residual dropout), not via SDPA's dropout_p: the fused MPS
        # SDPA kernel raises NotImplementedError on in-kernel dropout, and residual dropout is the standard,
        # device-portable placement (negligible difference from attention-weight dropout at p=0.1).
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.attn_drop(self.out_proj(attn.transpose(1, 2).reshape(batch, frames, hidden_dim)))
        return x + self.ffn(self.norm2(x.float()).to(x.dtype))


def chunked_rope_encode(
    layers: nn.ModuleList, x: torch.Tensor, timestamps_s: torch.Tensor | None, 
    chunk_size: int | None, key_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run pre-norm RoPE layers over fixed-size chunks (Moryossef chunked inference).

    Shared by both segmenters' RoPE stacks: the in-system `RoPEBIOHead` (BioS1Model) and the analysis
    `MoryossefSegmenter.encoder_attn`. Each chunk attends only within itself; RoPE relative time in seconds keeps
    positions consistent across chunk boundaries, which is what makes train-size chunked eval match the training distribution. 
    `chunk_size=None` (or T <= chunk_size) is a single full pass. Timestamps are dim-normalized up front so 1-D inputs are not 
    silently dropped on the chunked path (which would fall back to the 50fps index assumption and break fps-augmented inputs).
    """
    if timestamps_s is not None:
        if timestamps_s.dim() == 1: timestamps_s = timestamps_s.unsqueeze(0).expand(x.shape[0], -1)
        timestamps_s = timestamps_s.to(device=x.device)

    if chunk_size is None or x.shape[1] <= int(chunk_size):
        for layer in layers: x = layer(x, timestamps_s, key_mask=key_mask)
        return x

    chunks: list[torch.Tensor] = []
    for start in range(0, x.shape[1], int(chunk_size)):
        end = min(x.shape[1], start + int(chunk_size))
        chunk = x[:, start:end]
        chunk_ts = timestamps_s[:, start:end] if timestamps_s is not None else None
        chunk_mask = key_mask[:, start:end] if key_mask is not None else None
        for layer in layers: chunk = layer(chunk, chunk_ts, key_mask=chunk_mask)
        chunks.append(chunk)
    return torch.cat(chunks, dim=1)


class ClassifierHead(nn.Module): # Two-layer MLP classifier: hidden_dim → hidden_dim → num_classes.
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
    """Residual stride-1 temporal Conv1d stem restoring the local boundary inductive bias of Moryossef 2026's UNet front-end.

    Moryossef 2026's segmenter keeps a full UNet CNN before its RoPE layers, and their ablations attribute precise boundary placement to 
    exactly those skip-connection convolutions (EXPERIMENTS.md finding 23 "UNet skip connections are critical for sign boundary precision"; 
    finding 8 "CNN naturally detects B"). This in-model BIO head reads pose-token features DIRECTLY with no UNet, so without a conv stem 
    it is structurally a transformer/BiLSTM-style tagger, which Moryossef shows is weak on B / boundary localization — and that directly 
    feeds the commit gate's I→O δ_enc stability and RQ2 tIoU. This stem is the cheap middle ground (We wants the head small, so this is 
    2 conv layers, not the full UNet).

    Stride 1 + odd kernel + 'same' padding keeps the sequence length, so per-frame BIO labels stay aligned. Normalization is per-position 
    RMSNorm (over channels only): length- and batch-independent, so it introduces no train/inference mismatch over the growing streaming 
    buffer (unlike BatchNorm running stats or a GroupNorm that pools the time axis). conv_stem_layers=0 disables it (ablation).
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
        # Zero padded positions before AND after each conv so they never leak into real frames: masking the input alone is not enough 
        # since each Conv1d BIAS regenerates a nonzero value at padded positions, which NEXT layer then convolves back into trailing 
        # real frames. With padded positions held at 0, a real frame's conv window reads zeros over padding — identical to streaming 
        # inference, where 'same' padding zero-pads past sequence end (no train/infer mismatch & no dependence on batch padding length).
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

    `.logits` aliases the phrase BIO logits, which the FSM consumes. Moryossef 2026 additionally carries a *sign* (sub-sentence) BIO head,
    but that head is supervised by sign-level temporal segment annotations. YouTube-SL-25 only provides sentence/caption timestamps, so 
    there is no signal to train a sign head; it is removed here rather than left as an unsupervised dead branch.

    A small residual `TemporalConvStem` precedes the RoPE layers to recover the local boundary inductive bias that Moryossef's UNet 
    front-end provides but this UNet-less head otherwise lacks (see that class).
    """
    def __init__(
        self, input_dim: int, hidden_dim: int = 384, depth: int = 4, nhead: int = 8, ff_mult: int = 2, dropout: float = 0.1,
        num_classes: int = 4, chunk_size: int | None = None, conv_stem_layers: int = 2, conv_stem_kernel: int = 5,
    ):
        super().__init__()
        self.chunk_size = chunk_size
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
        x = self.input_norm(proj.float()).to(proj.dtype)  # fp32 RMSNorm under autocast (see layer note)
        # Conv stem runs on the full sequence BEFORE RoPE chunking (it is local/translation-equivariant, so it does not reintroduce the 
        # absolute-position issue chunked RoPE eliminates). It is mask-aware: padded frames are held at 0 so they never leak into real 
        # (loss-computed) frames — making training identical to unbatched streaming inference (see TemporalConvStem).
        if self.conv_stem is not None: x = self.conv_stem(x, key_mask=frame_mask)
        return chunked_rope_encode(self.layers, x, timestamps_s, self.chunk_size, key_mask=frame_mask)

    def forward(
        self, features: torch.Tensor, timestamps_s: torch.Tensor | None = None, 
        frame_mask: torch.Tensor | None = None
    ) -> BIOHeadOutput:
        hidden = self.encode(features, timestamps_s=timestamps_s, frame_mask=frame_mask)
        phrase_logits = self.phrase_bio_head(hidden)
        return BIOHeadOutput(phrase_logits=phrase_logits, logits=phrase_logits, hidden_states=hidden)
