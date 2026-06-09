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

    def forward(self, x: torch.Tensor, timestamps_s: torch.Tensor | None = None) -> torch.Tensor:
        batch, frames, hidden_dim = x.shape
        # Assume 50fps when no timestamps provided (1/50s per frame → *50 → 1 unit/frame).
        if timestamps_s is None: timestamps_s = torch.arange(frames, device=x.device, dtype=torch.float32) / self.REFERENCE_FPS
        if timestamps_s.dim() == 1: timestamps_s = timestamps_s.unsqueeze(0).expand(batch, -1)
        timestamps_s = timestamps_s.to(device=x.device)
        cos, sin = self._compute_rope(timestamps_s)

        h = self.norm1(x)
        qkv = self.qkv(h).reshape(batch, frames, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x + self.out_proj(attn.transpose(1, 2).reshape(batch, frames, hidden_dim))
        return x + self.ffn(self.norm2(x))


def chunked_rope_encode(
    layers: nn.ModuleList, x: torch.Tensor, 
    timestamps_s: torch.Tensor | None, chunk_size: int | None,
) -> torch.Tensor:
    """Run pre-norm RoPE layers over fixed-size chunks (Moryossef chunked inference).

    Shared by `RoPEBIOHead` and `MoryossefSegmenter`. Each chunk attends only within itself; RoPE relative time in seconds keeps 
    positions consistent across chunk boundaries, which is what makes train-size chunked eval match the training distribution. 
    `chunk_size=None` (or T <= chunk_size) is a single full pass. Timestamps are dim-normalized up front so 1-D inputs are not 
    silently dropped on the chunked path (which would fall back to the 50fps index assumption and break fps-augmented inputs).
    """
    if timestamps_s is not None:
        if timestamps_s.dim() == 1: timestamps_s = timestamps_s.unsqueeze(0).expand(x.shape[0], -1)
        timestamps_s = timestamps_s.to(device=x.device)

    if chunk_size is None or x.shape[1] <= int(chunk_size):
        for layer in layers: x = layer(x, timestamps_s)
        return x

    chunks: list[torch.Tensor] = []
    for start in range(0, x.shape[1], int(chunk_size)):
        end = min(x.shape[1], start + int(chunk_size))
        chunk = x[:, start:end]
        chunk_ts = timestamps_s[:, start:end] if timestamps_s is not None else None
        for layer in layers: chunk = layer(chunk, chunk_ts)
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


class RoPEBIOHead(nn.Module):
    """Phrase-level Moryossef-style BIO head over post-VLP visual features.

    `.logits` aliases the phrase BIO logits, which the FSM consumes. Moryossef 2026 additionally carries a *sign* (sub-sentence) BIO head, 
    but that head is supervised by sign-level temporal segment annotations. YouTube-SL-25 only provides sentence/caption timestamps, 
    so there is no signal to train a sign head; it is removed here rather than left as an unsupervised dead branch.
    """
    def __init__(
        self, input_dim: int, hidden_dim: int = 384, depth: int = 4, 
        nhead: int = 8, ff_mult: int = 2, dropout: float = 0.1,
        num_classes: int = 4, chunk_size: int | None = None,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.input_proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.RMSNorm(hidden_dim)
        self.layers = nn.ModuleList([RoPETransformerEncoderLayer(
            hidden_dim=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * ff_mult, dropout=dropout,
        ) for _ in range(depth)])
        self.phrase_bio_head = ClassifierHead(hidden_dim, num_classes)

    def encode(self, features: torch.Tensor, timestamps_s: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_norm(self.input_proj(features))
        return chunked_rope_encode(self.layers, x, timestamps_s, self.chunk_size)

    def forward(self, features: torch.Tensor, timestamps_s: torch.Tensor | None = None) -> BIOHeadOutput:
        hidden = self.encode(features, timestamps_s=timestamps_s)
        phrase_logits = self.phrase_bio_head(hidden)
        return BIOHeadOutput(phrase_logits=phrase_logits, logits=phrase_logits, hidden_states=hidden)
