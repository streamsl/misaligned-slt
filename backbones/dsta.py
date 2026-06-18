"""MSKA DSTA pose backbone (decoupled spatial-temporal attention), ported for streaming SLT.

Faithful adaptation of the keypoint backbone in MSKA (Multi-Stream Keypoint Attention;
github.com/sutwangyan/MSKA, `recognition.py`). MSKA reaches SOTA gloss-supervised SLR/SLT
(29.03 BLEU4 on Phoenix-2014T) with a 4-stream decoupled spatial-temporal attention network whose
spatial relations are LEARNED by self-attention (no fixed skeleton graph), unlike CoSign's ST-GCN.

What is ported vs. dropped (this project is gloss-FREE streaming SLT):
  - PORTED  : `PositionalEncoding` (recognition.py:33-68) and `STAttentionBlock` (recognition.py:71-174) VERBATIM 
    (device-agnostic; `.cuda()` removed), and the 4-stream DSTA` body (recognition.py:177-286) as `DSTANet` below.
  - DROPPED : Per-stream `VisualHead` gloss classifiers, CTC `recognition_loss`, cross-stream KL distillation, and 
    MSKA's own mBART translation network (recognition.py:289-403, model.py, translation.py, vl_mapper.py). 
    Those and ONLY those — need gloss files (gloss2ids.pkl / gloss_embeddings.bin / map_ids.pkl). DSTANet needs none.

Two intentional deviations from MSKA, each a "faithful-to-original = bug-here" case:
  1. TEMPORAL STRIDE forced to 1 (`force_stride1=True`). MSKA's net strides downsample T->T/4 (for CTC over T/4). 
     Here `post_vlp` must stay FRAME-ALIGNED: it feeds the per-frame BIO head (bio_labels + timestamps_s) AND the 
     FSM/streaming buffer. Downsampling would desync those. The temporal convs (`out_nett`, t_kernel 7/3) are KEPT 
     — only the stride-2 downsampling is neutralised.
  2. OUTPUT is the fused per-frame feature projected to `hidden_size` (CoSign1s's exact I/O contract: (B,T,K,3) -> 
     (B,T,hidden_size)), so DSTANet is a drop-in for CoSign. MSKA's VisualHead (fc->bn->pe->FF->layernorm) refinement 
     is intentionally omitted: it's already played downstream by our VLP projection (`VisualEncoder`) + mBART encoder.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
from poses import MSKA_STREAMS_133

# --- MSKA Phoenix-2014T defaults (configs/phoenix-2014t_s2t.yaml: model.RecognitionNetwork.DSTA-Net) ---
# net rows: (in_channels, out_channels, inter_channels, t_kernel, stride). The 2 stride-2 rows give
# MSKA's T/4 downsampling; `force_stride1` neutralises the stride (see module docstring) while keeping
# channels/kernels identical. num_subset is DSTA's default (recognition.py:179).
MSKA_NET: tuple[tuple[int, int, int, int, int], ...] = (
    (64, 64, 16, 7, 2), (64, 64, 16, 3, 1),
    (64, 128, 32, 3, 1), (128, 128, 32, 3, 1),
    (128, 256, 64, 3, 2), (256, 256, 64, 3, 1),
    (256, 256, 64, 3, 1), (256, 256, 64, 3, 1),
)
MSKA_NUM_SUBSET = 6
FUSE_ORDER = ("left", "face", "right", "body") # Concat over channels of these streams, in MSKA's order (recognition.py:283).


class MaskedBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d that, in TRAINING with a per-frame mask (N,1,T,1 bool of valid frames), computes the batch statistics 
    (and running stats) over VALID frames only. MSKA trains on fixed-length clips with no padding; we feed variable-length 
    windows zero-padded to the batch max, and the constant features those padded frames produce (Conv bias -> BN -> LeakyReLU) 
    skew BN. `mask=None` or eval -> EXACTLY nn.BatchNorm2d, so the faithful path is unchanged."""
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None or not self.training: return super().forward(x)
        m = mask.to(x.dtype)                                   # (N,1,T,1)
        count = (m.sum() * x.shape[3]).clamp(min=1.0)          # valid (N,T,V) entries per channel
        mean = (x * m).sum(dim=(0, 2, 3)) / count              # (C,)
        var = ((x - mean.view(1, -1, 1, 1)) ** 2 * m).sum(dim=(0, 2, 3)) / count
        if self.track_running_stats and self.running_mean is not None:
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                unbiased = var * (count / (count - 1)) if count > 1 else var   # BN tracks unbiased var
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased)
                self.num_batches_tracked.add_(1)
        x_hat = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        if self.affine: x_hat = x_hat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x_hat


class _ConvBN(nn.Module):
    """Conv2d + MaskedBatchNorm2d, forwarding an optional frame mask to the norm (replaces the
    nn.Sequential(Conv2d, BatchNorm2d) blocks so the mask can reach the BN)."""
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        self.bn = MaskedBatchNorm2d(conv.out_channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.bn(self.conv(x), mask)


class _Identity(nn.Module):  # mask-accepting identity residual (replaces `lambda x: x`)
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal spatial/temporal positional encoding (MSKA recognition.py:33-68, verbatim).

    Buffer is sized to `time_len` (=`num_frame`); `forward` slices it to the actual T, so `num_frame`
    must be >= the longest sequence the backbone ever sees (asserted in DSTANet.forward).
    """
    def __init__(self, channel: int, joint_num: int, time_len: int, domain: str):
        super().__init__()
        self.joint_num = joint_num
        self.time_len = time_len
        self.domain = domain

        if domain == "temporal": pos_list = [t for t in range(self.time_len) for _ in range(self.joint_num)]
        elif domain == "spatial": pos_list = [j_id for _ in range(self.time_len) for j_id in range(self.joint_num)]
        else: raise ValueError(f"Unsupported PE domain={domain}")

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        pe = torch.zeros(self.time_len * self.joint_num, channel)
        div_term = torch.exp(torch.arange(0, channel, 2).float() * -(math.log(10000.0) / channel))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, C, T, V)
        return x + self.pe[:, :, : x.size(2)]


class STAttentionBlock(nn.Module):
    """Decoupled spatial self-attention + temporal conv block (MSKA recognition.py:71-174, verbatim).

    Spatial: per-frame decoupled self-attention over the V nodes (`num_subset` heads, optional global
    regularisation `attention0s`); temporal: a (t_kernel x 1) conv. Device-agnostic — the only change
    from MSKA is that buffers move with `.to(device)` (no in-module `.cuda()`).
    """
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=2, num_node=27, num_frame=400,
                 kernel_size=1, stride=1, t_kernel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
                 use_temporal_att=False, use_spatial_att=True, attentiondrop=0., use_pes=True, use_pet=False):
        super().__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet

        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            atts = torch.zeros((1, num_subset, num_node, num_node))
            self.register_buffer("atts", atts)
            self.pes = PositionalEncoding(in_channels, num_node, num_frame, "spatial")
            self.ff_nets = _ConvBN(nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True))
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_s: self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node, requires_grad=True)
            self.out_nets = _ConvBN(nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True))
        else: self.out_nets = _ConvBN(nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1))

        padd = int(t_kernel / 2)
        self.out_nett = _ConvBN(nn.Conv2d(out_channels, out_channels, (t_kernel, 1), padding=(padd, 0), bias=True, stride=(stride, 1)))
        if in_channels != out_channels or stride != 1:
            if use_spatial_att: self.downs1 = _ConvBN(nn.Conv2d(in_channels, out_channels, 1, bias=True))
            self.downs2 = _ConvBN(nn.Conv2d(in_channels, out_channels, 1, bias=True))
            if use_temporal_att: self.downt1 = _ConvBN(nn.Conv2d(out_channels, out_channels, 1, 1, bias=True))
            self.downt2 = _ConvBN(nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True))
        else:
            if use_spatial_att: self.downs1 = _Identity()
            self.downs2 = _Identity()
            self.downt2 = _Identity()

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)


    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # mask: (N,1,T,1) bool of valid frames (or None for the faithful, padding-unaware path). Forwarded
        # to every BN; force_stride1 keeps T constant so the same mask is valid through the whole block.
        N, C, T, V = x.size()
        if self.use_spatial_att:
            attention = self.atts
            y = self.pes(x) if self.use_pes else x
            if self.att_s:
                q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                if mask is None: denom = self.inter_channels * T
                else:
                    # Zero padded frames in q,k (the spatial PE made them non-zero) so the temporal sum in
                    # the attention excludes padding, and normalise by each sample's VALID T, not padded T.
                    m5 = mask.view(N, 1, 1, T, 1).to(q.dtype)
                    q, k = q * m5, k * m5
                    denom = self.inter_channels * mask.to(q.dtype).sum(dim=2).clamp(min=1).view(N, 1, 1, 1)
                attention = attention + self.tan(torch.einsum("nsctu,nsctv->nsuv", [q, k]) / denom) * self.alphas

            if self.glo_reg_s: attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            y = torch.einsum("nctu,nsuv->nsctv", [x, attention]).contiguous().view(N, self.num_subset * self.in_channels, T, V)
            y = self.out_nets(y, mask)
            y = self.relu(self.downs1(x, mask) + y)
            y = self.ff_nets(y, mask)
            y = self.relu(self.downs2(x, mask) + y)
        else:
            y = self.out_nets(x, mask)
            y = self.relu(self.downs2(x, mask) + y)
        z = self.out_nett(y, mask)
        return self.relu(self.downt2(y, mask) + z)


class DSTANet(nn.Module):
    """4-stream decoupled spatial-temporal attention backbone (MSKA recognition.py:177-286, `DSTA`).

    Drop-in for `CoSign1s`: forward maps (B, T, K, 3) -> (B, T, hidden_size). Each anatomical stream (left/right/face/body) runs 
    an `input_map` conv + a stack of `STAttentionBlock`s, mean-pools over its nodes, and the four are concatenated (MSKA's `fuse`) 
    and projected to `hidden_size`. The per-stream gloss heads / CTC / distillation of MSKA are dropped (gloss-free). `force_stride1` 
    keeps T frame-aligned (see module docstring). `streams` indexes the K dimension of the input.
    """
    def __init__(
        self, hidden_size: int, streams: dict[str, tuple[int, ...]] | None = None,
        net: tuple[tuple[int, int, int, int, int], ...] = MSKA_NET, num_channel: int = 3,
        num_subset: int = MSKA_NUM_SUBSET, num_frame: int = 256, dropout: float = 0.1,
        attentiondrop: float = 0.1, force_stride1: bool = True,
    ):
        super().__init__()
        streams = streams if streams is not None else MSKA_STREAMS_133
        if tuple(streams.keys()) and set(FUSE_ORDER) - set(streams.keys()):
            raise ValueError(f"DSTANet needs streams {FUSE_ORDER}; got {tuple(streams.keys())}")
            
        self.num_frame = int(num_frame)
        self.force_stride1 = bool(force_stride1)
        in_channels0 = net[0][0]
        self.out_channels = net[-1][1]
        param = dict(num_subset=num_subset, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
                     use_spatial_att=True, use_temporal_att=False, use_pet=False, use_pes=True, attentiondrop=attentiondrop)

        # Stream index buffers (move with the module; index the input's K dimension).
        self.stream_names = tuple(FUSE_ORDER)
        for name in self.stream_names:
            self.register_buffer(f"idx_{name}", torch.tensor(streams[name], dtype=torch.long), persistent=False)

        self.input_maps = nn.ModuleDict({  # Conv+BN (mask-aware) per stream; LeakyReLU applied in _run_stream
            name: _ConvBN(nn.Conv2d(num_channel, in_channels0, 1)) for name in self.stream_names
        })
        self.input_act = nn.LeakyReLU(0.1)
        self.graph_layers = nn.ModuleDict()
        for name in self.stream_names:
            num_node = len(streams[name])
            layers = nn.ModuleList()
            frame = self.num_frame

            for (ci, co, inter, t_kernel, stride) in net:
                s = 1 if self.force_stride1 else stride
                layers.append(STAttentionBlock(
                    ci, co, inter, stride=s, t_kernel=t_kernel, num_node=num_node, num_frame=frame, **param
                ))
                frame = int(frame / s + 0.5)
            self.graph_layers[name] = layers

        self.drop_out = nn.Dropout(dropout)
        # fuse (concat of the 4 streams' out_channels) -> hidden_size, mirroring CoSign1s.fusion (Linear+GELU).
        self.fusion = nn.Sequential(nn.Linear(self.out_channels * len(self.stream_names), hidden_size), nn.GELU())


    def _run_stream(self, x: torch.Tensor, name: str, mask: torch.Tensor | None) -> torch.Tensor:
        # x: (N, C, T, K) -> stream feature (N, T, out_channels). When masked, padded frames are re-zeroed
        # after the input map and every block: this (i) excludes them from BN (MaskedBatchNorm2d) and
        # (ii) makes the temporal conv / spatial attention see standard ZERO-padding at the boundary
        # instead of the garbage the Conv biases would otherwise produce, so valid frames stay clean.
        idx = getattr(self, f"idx_{name}")
        feat = self.input_act(self.input_maps[name](x.index_select(3, idx), mask))
        if mask is not None: feat = feat * mask
        for block in self.graph_layers[name]:
            feat = block(feat, mask)
            if mask is not None: feat = feat * mask
        return feat.permute(0, 2, 1, 3).contiguous().mean(3)  # (N, T, C)


    def forward(self, poses: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        # poses: (B, T, K, 3) -> (B, T, hidden_size). `frame_mask` (B,T) bool excludes padded frames from
        # every BatchNorm (the (a) fix). Only used with force_stride1 (T preserved so the mask stays valid
        # through all blocks); otherwise ignored. None -> faithful, padding-unaware behaviour.
        if poses.dim() != 4 or poses.shape[-1] < 2: raise ValueError(f"DSTANet expects (B, T, K, C>=2); got {tuple(poses.shape)}")
        T = poses.shape[1]
        if T > self.num_frame: raise ValueError(f"DSTANet window length T={T} exceeds num_frame={self.num_frame}; raise "
                                                f"GFSLTConfig.dsta_num_frame above the longest window (buffer_cap_s * fps).")
        mask = None
        if frame_mask is not None and self.force_stride1: mask = frame_mask.to(torch.bool).view(frame_mask.shape[0], 1, T, 1)
        x = poses.permute(0, 3, 1, 2).contiguous()  # (B, C, T, K)
        feats = [self._run_stream(x, name, mask) for name in FUSE_ORDER]  # left, face, right, body
        fuse = torch.cat(feats, dim=-1)  # (B, T, 4 * out_channels)
        return self.fusion(self.drop_out(fuse))
