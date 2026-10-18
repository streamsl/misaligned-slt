"""Uni-Sign pose-only encoder (the pose branch of `Uni_Sign`), vendored faithfully.

Provenance: ZechengLi19/Uni-Sign @ models.py `Uni_Sign.__init__` / `.forward` pose branch. 
RGB/PGF path (`rgb_support`) is intentionally dropped — this is the released pose-only model.

Faithfulness points that make the released checkpoint load (strict=True) and behave identically:
  * submodule names match Uni_Sign exactly: proj_linear / gcn_modules / fusion_gcn_modules /
    part_para / pose_proj (ModuleDict keys 'body','left','right','face_all').
  * left/right hands SHARE weights (gcn_modules['left']=gcn_modules['right'] etc.) — that is why 
    the upstream checkpoint stores only 'right' hand keys and mirrors them to 'left' at save
    (models.get_requires_grad_dict). Our loader mirrors the same way.
    Unlike upstream, both hands run via the shared modules as ONE batched call, so a train-mode 
    BatchNorm normalizes them with the pooled statistics that eval applies; eval outputs match 
    the two-call forward up to float rounding.
  * body part is processed FIRST and its spatial-GCN feature is injected (detached) into the
    hand/face features BEFORE the temporal GCN: left += body[...,-2], right += body[...,-1],
    face_all += body[...,0]. Order of self.modes is important.
  * NO temporal downsampling (stride 1) -> T preserved -> frame-aligned, so the RoPE BIO head and
    the streaming FSM read per-frame features exactly as before.

Input contract: poses (B, T, 69, 3) with the 69 keypoints in the fixed part order
[body 9 | left 21 | right 21 | face_all 18] and channels (x, y, conf), already normalised by
poses.preprocessing.normalize_keypoints_unisign. Output: (B, T, 768) pose tokens (mT5 d_model).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from .gcn_utils import Graph
from .stgcn_block import get_stgcn_chain

# Fixed split of the 69-keypoint tensor into Uni-Sign parts (must match normalize_keypoints_unisign).
PART_SIZES = {"body": 9, "left": 21, "right": 21, "face_all": 18}
PART_ORDER = ["body", "left", "right", "face_all"]
NUM_KEYPOINTS = sum(PART_SIZES.values())  # 69
# Masked frames after each row of a padded batch run end to end: the temporal conv's padding (kernel 5), 
# so no conv window reads 2 rows.
ROW_GAP = 2
# Longest sequence of rows end to end; a longer batch runs padded. On MPS (torch 2.13) a Linear on a 4-D 
# input whose dim 1 is over 2**16 returns wrong values, and dim 1 of proj_linear's input is that sequence. 
# Training batches stay below it; RQ1 batches can pass it.
MAX_SEQUENCE_FRAMES = 2 ** 16


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # norm_cdf is evaluated on SCALAR bounds (a, b) — use math.erf, not torch.erf (which rejects Python floats).
    norm_cdf = lambda x: (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class UniSignPoseEncoder(nn.Module):
    # 4-part ST-GCN pose encoder -> (B, T, out_dim). out_dim defaults to 768 (mT5-base d_model).
    def __init__(self, hidden_dim: int = 256, out_dim: int = 768):
        super().__init__()
        self.modes = list(PART_ORDER)
        self.hidden_dim = int(hidden_dim)

        self.graph, A = {}, []
        self.proj_linear = nn.ModuleDict()
        for mode in self.modes:
            self.graph[mode] = Graph(layout=f"{mode}", strategy="distance", max_hop=1)
            A.append(torch.tensor(self.graph[mode].A, dtype=torch.float32, requires_grad=False))
            self.proj_linear[mode] = nn.Linear(3, 64)

        self.gcn_modules = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()
        spatial_kernel_size = A[0].size(0)
        # The shared ST-GCN blocks are BatchNorm + ReLU (the faithful Uni-Sign config; `bn` keys match the
        # released pose-only checkpoint).
        for index, mode in enumerate(self.modes):
            self.gcn_modules[mode], final_dim = get_stgcn_chain(
                64, "spatial", (1, spatial_kernel_size), A[index].clone(), True)
            self.fusion_gcn_modules[mode], _ = get_stgcn_chain(
                final_dim, "temporal", (5, spatial_kernel_size), A[index].clone(), True)

        # Left hand shares all weights with the right hand (Uni_Sign models.py:95-97).
        self.gcn_modules["left"] = self.gcn_modules["right"]
        self.fusion_gcn_modules["left"] = self.fusion_gcn_modules["right"]
        self.proj_linear["left"] = self.proj_linear["right"]

        self.part_para = nn.Parameter(torch.zeros(self.hidden_dim * len(self.modes)))
        self.pose_proj = nn.Linear(256 * 4, int(out_dim))
        self.out_dim = int(out_dim)
        self.apply(self._init_weights)


    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            _no_grad_trunc_normal_(m.weight, 0.0, 0.02, -2.0, 2.0)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def _split_parts(self, poses: torch.Tensor) -> dict[str, torch.Tensor]:
        if poses.shape[-2] != NUM_KEYPOINTS:
            raise ValueError(f"UniSignPoseEncoder expects {NUM_KEYPOINTS} keypoints, got {poses.shape[-2]}")
        parts, offset = {}, 0
        for mode in self.modes:
            size = PART_SIZES[mode]
            parts[mode] = poses[..., offset:offset + size, :]
            offset += size
        return parts


    def forward(self, poses: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        """poses: (B, T, 69, 3) -> (B, T, out_dim).

        `frame_mask` (B, T, True = real frame) zeroes padded frames before each TEMPORAL conv (kernel 5, three
        blocks = a +-6 frame reach). Without it a real frame within 6 of the pad mixes in whatever the collator
        wrote there, so its features depend on which batch it landed in — measured at 44-66% relative error on
        those frames, a fixed offset every step reapplies in the same direction. The downstream attention mask
        hides the pad COLUMNS but cannot undo corruption already folded into the real ones, and here the affected
        frames carry the sentence terminator the commit gate reads. With the mask, a padded batch reproduces an
        exact-length forward bit-for-bit; a fully packed batch is unchanged, and the released weights still load
        strict. `None` applies no mask (the upstream Uni-Sign forward).

        In train mode the 33 BatchNorm2d layers take their batch statistics, and update their running statistics, 
        over the real frames only (`stgcn_block.masked_batch_norm`), so the amount of padding in a batch doesn't 
        change a real frame's features (both stage-2 pose encoders and the S1 encoder train). 2 hands take 1 call, 
        so they share its statistics. With no padded frame, and in eval mode, each layer runs as plain BatchNorm2d.

        A padded batch runs as ONE sequence: its rows end to end, each from its 1st to its last real frame, then 
        ROW_GAP masked frames. Each temporal conv reads masked gap as the 0 padding that it reads at a row's edge 
        in the padded batch, and BatchNorm statistics take same real cells. So real frames' outputs, gradients and 
        running statistics equal those of the padded batch up to float rounding, and the encoder computes only each 
        row's real extent and its gap. In train mode each BatchNorm2d runs natively on real frames of that sequence. 
        The output at a frame outside its row's real extent is a copy of an output of that row: it is finite, and 
        every reader masks it. A batch whose sequence would be longer than MAX_SEQUENCE_FRAMES runs padded.
        """
        if frame_mask is None: return self._encode(poses, None)
        mask = frame_mask.bool()
        b, t = mask.shape
        frames = torch.arange(t, device=mask.device)
        first = torch.where(mask, frames, t).min(1).values
        last = torch.where(mask, frames, -1).max(1).values
        width = (last - first + 1).clamp(min=0) + ROW_GAP  # a row's real extent [first, last], then its gap
        real, total = torch.stack((mask.sum(), width.sum())).tolist()

        if real == mask.numel(): return self._encode(poses, None)  # no padded frame: the plain forward
        if real == 0 and self.training: raise ValueError("a train-mode batch needs a real frame for BatchNorm")
        if real == 0 or total > MAX_SEQUENCE_FRAMES: return self._encode(poses, mask)  # the padded forward
        rows = torch.repeat_interleave(torch.arange(b, device=mask.device), width, output_size=total)
        start = width.cumsum(0) - width
        pos = torch.arange(total, device=mask.device) - start[rows]
        src = (first[rows] + pos).clamp(max=t - 1)
        seq = mask[rows, src] & (pos < width[rows] - ROW_GAP)  # gaps are masked. Masked frame inside stays masked
        out = self._encode(poses[rows, src][None], seq[None], seq.nonzero().squeeze(1) if self.training else None)[0]
        return out[start[:, None] + torch.minimum((frames - first[:, None]).clamp(min=0), (width - 1)[:, None])]


    def _encode(
        self, poses: torch.Tensor, frame_mask: torch.Tensor | None, real_frames: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Padded forward. `real_frames`: real frames that every row shares (rows end to end), for native BatchNorm.
        parts = self._split_parts(poses)
        n = poses.shape[0]

        # Left and right share every hand module, so they run as ONE batch: a train-mode BatchNorm then normalizes 
        # both hands with pooled batch statistics that its running statistics hold and eval mode applies (2 calls 
        # would normalize each hand with its own; Wu & Johnson 2021, Table 5). No layer mixes samples in eval mode.
        streams = {
            "body": (parts["body"], frame_mask), 
            "right": (
                torch.cat([parts["left"], parts["right"]]), 
                None if frame_mask is None else frame_mask.repeat(2, 1)
            ),
            "face_all": (parts["face_all"], frame_mask)
        }
        features, body_feat = {}, None
        for part, (x, mask) in streams.items():  # body first: its spatial feature is injected into the others
            proj_feat = self.proj_linear[part](x).permute(0, 3, 1, 2)  # B,C,T,V
            gcn_feat = self.gcn_modules[part](proj_feat, mask, real_frames)  # spatial GCN (kernel-1 in time)
            if part == "body": body_feat = gcn_feat
            elif part == "right":  # left += body[...,-2], right += body[...,-1]
                gcn_feat = gcn_feat + torch.cat([body_feat[..., -2:-1], body_feat[..., -1:]]).detach()
            else: gcn_feat = gcn_feat + body_feat[..., :1].detach()  # face_all += body[...,0]
            gcn_feat = self.fusion_gcn_modules[part](gcn_feat, mask, real_frames)  # temporal GCN (kernel 5, stride 1)
            features[part] = gcn_feat.mean(-1).transpose(1, 2)  # B,T,C

        hands = features["right"]
        inputs_embeds = torch.cat(
            [features["body"], hands[:n], hands[n:], features["face_all"]], dim=-1
        ) + self.part_para
        return self.pose_proj(inputs_embeds)  # B,T,out_dim
