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


# --- MSKA Phoenix-2014T defaults (configs/phoenix-2014t_s2t.yaml: model.RecognitionNetwork.DSTA-Net) ---
# net rows: (in_channels, out_channels, inter_channels, t_kernel, stride). The two stride-2 rows give
# MSKA's T/4 downsampling; `force_stride1` neutralises the stride (see module docstring) while keeping
# channels/kernels identical. num_subset is DSTA's default (recognition.py:179).
MSKA_NET: tuple[tuple[int, int, int, int, int], ...] = (
    (64, 64, 16, 7, 2), (64, 64, 16, 3, 1),
    (64, 128, 32, 3, 1), (128, 128, 32, 3, 1),
    (128, 256, 64, 3, 2), (256, 256, 64, 3, 1),
    (256, 256, 64, 3, 1), (256, 256, 64, 3, 1),
)
MSKA_NUM_SUBSET = 6

# MSKA's 4 anatomical streams as indices into the FULL 133-keypoint COCO-WholeBody array
# (configs/phoenix-2014t_s2t.yaml). Overlapping by design: hands carry the adjacent arm joints; the
# body stream spans body+both hands+face so its LEARNED attention sees inter-part spatial layout
# (the signal CoSign's per-group root-normalisation destroys — the reason we feed MSKA-native 133).
MSKA_STREAMS_133: dict[str, tuple[int, ...]] = {
    "left": (0, 1, 3, 5, 7, 9, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
             107, 108, 109, 110, 111),                                                   # 27 nodes
    "right": (0, 2, 4, 6, 8, 10, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
              126, 127, 128, 129, 130, 131, 132),                                        # 27 nodes
    "face": (23, 26, 29, 33, 36, 39, 41, 43, 46, 48, 53, 56, 59, 62, 65, 68, 71, 72, 73, 74, 75, 76,
             77, 79, 80, 81),                                                            # 26 nodes
    "body": (0, 1, 3, 5, 7, 9, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
             107, 108, 109, 110, 111, 2, 4, 6, 8, 10, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
             122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 23, 26, 29, 33, 36, 39, 41, 43, 46,
             48, 53, 56, 59, 62, 65, 68, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81),        # 79 nodes
}
# fuse = concat over channels of these streams, in MSKA's order (recognition.py:283).
FUSE_ORDER = ("left", "face", "right", "body")


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
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_s:
                self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node, requires_grad=True)
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else: self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
            nn.BatchNorm2d(out_channels),
        )
        padd = int(t_kernel / 2)
        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (t_kernel, 1), padding=(padd, 0), bias=True, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels or stride != 1:
            if use_spatial_att: self.downs1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if use_temporal_att: self.downt1 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            if use_spatial_att: self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            self.downt2 = lambda x: x

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.size()
        if self.use_spatial_att:
            attention = self.atts
            y = self.pes(x) if self.use_pes else x
            if self.att_s:
                q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                attention = attention + self.tan(
                    torch.einsum("nsctu,nsctv->nsuv", [q, k]) / (self.inter_channels * T)
                ) * self.alphas

            if self.glo_reg_s: attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            y = torch.einsum("nctu,nsuv->nsctv", [x, attention]).contiguous().view(N, self.num_subset * self.in_channels, T, V)
            y = self.out_nets(y)
            y = self.relu(self.downs1(x) + y)
            y = self.ff_nets(y)
            y = self.relu(self.downs2(x) + y)
        else:
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)
        z = self.out_nett(y)
        return self.relu(self.downt2(y) + z)


class DSTANet(nn.Module):
    """4-stream decoupled spatial-temporal attention backbone (MSKA recognition.py:177-286, `DSTA`).

    Drop-in for `CoSign1s`: forward maps (B, T, K, 3) -> (B, T, hidden_size). Each anatomical stream
    (left/right/face/body) runs an `input_map` conv + a stack of `STAttentionBlock`s, mean-pools over
    its nodes, and the four are concatenated (MSKA's `fuse`) and projected to `hidden_size`. The
    per-stream gloss heads / CTC / distillation of MSKA are dropped (gloss-free). `force_stride1` keeps
    T frame-aligned (see module docstring). `streams` indexes the K dimension of the input.
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

        self.input_maps = nn.ModuleDict({
            name: nn.Sequential(nn.Conv2d(num_channel, in_channels0, 1), nn.BatchNorm2d(in_channels0), nn.LeakyReLU(0.1))
            for name in self.stream_names
        })
        self.graph_layers = nn.ModuleDict()
        for name in self.stream_names:
            num_node = len(streams[name])
            layers = nn.ModuleList()
            frame = self.num_frame
            for (ci, co, inter, t_kernel, stride) in net:
                s = 1 if self.force_stride1 else stride
                layers.append(STAttentionBlock(
                    ci, co, inter, stride=s, t_kernel=t_kernel, 
                    num_node=num_node, num_frame=frame, **param
                ))
                frame = int(frame / s + 0.5)
            self.graph_layers[name] = layers
        self.drop_out = nn.Dropout(dropout)
        # fuse (concat of the 4 streams' out_channels) -> hidden_size, mirroring CoSign1s.fusion (Linear+GELU).
        self.fusion = nn.Sequential(nn.Linear(self.out_channels * len(self.stream_names), hidden_size), nn.GELU())


    def _run_stream(self, x: torch.Tensor, name: str) -> torch.Tensor: # x: (N, C, T, K) -> stream feature (N, T, out_channels)
        idx = getattr(self, f"idx_{name}")
        feat = self.input_maps[name](x.index_select(3, idx))
        for block in self.graph_layers[name]: feat = block(feat)
        return feat.permute(0, 2, 1, 3).contiguous().mean(3)  # (N, T, C)


    def forward(self, poses: torch.Tensor) -> torch.Tensor: # poses: (B, T, K, 3) -> (B, T, hidden_size)
        if poses.dim() != 4 or poses.shape[-1] < 2: raise ValueError(f"DSTANet expects (B, T, K, C>=2); got {tuple(poses.shape)}")
        T = poses.shape[1]
        if T > self.num_frame: raise ValueError(f"DSTANet window length T={T} exceeds num_frame={self.num_frame}; raise "
                                                f"GFSLTConfig.dsta_num_frame above the longest window (buffer_cap_s * fps).")
        x = poses.permute(0, 3, 1, 2).contiguous()  # (B, C, T, K)
        feats = [self._run_stream(x, name) for name in FUSE_ORDER]  # left, face, right, body
        fuse = torch.cat(feats, dim=-1)  # (B, T, 4 * out_channels)
        return self.fusion(self.drop_out(fuse))
