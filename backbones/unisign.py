"""Uni-Sign pose-only encoder (the pose branch of `Uni_Sign`), vendored faithfully.

Provenance: ZechengLi19/Uni-Sign @ models.py `Uni_Sign.__init__` / `.forward` pose branch
(arXiv 2501.15187, ICLR 2025). RGB/PGF path (`rgb_support`) is intentionally dropped — this is
the pose-only model that produced their CSL-Daily dev/test BLEU-4 25.27/25.61.

Faithfulness points that make the released checkpoint load (strict=True) and behave identically:
  * submodule names match Uni_Sign exactly: proj_linear / gcn_modules / fusion_gcn_modules /
    part_para / pose_proj (ModuleDict keys 'body','left','right','face_all').
  * left/right hands SHARE weights (gcn_modules['left']=gcn_modules['right'] etc.) — that is why the
    upstream checkpoint stores only 'right' hand keys and mirrors them to 'left' at save
    (models.get_requires_grad_dict). Our loader mirrors the same way.
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


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # norm_cdf is evaluated on the SCALAR bounds (a, b) — use math.erf, not torch.erf (which rejects Python floats).
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
        """poses: (B, T, 69, 3) -> (B, T, out_dim). `frame_mask` is accepted for interface parity with the
        other backbones but unused: Uni-Sign's ST-GCN has no mask-aware norm, and padded frames are masked
        downstream by the mT5 encoder's attention_mask, exactly as in the upstream model."""
        parts = self._split_parts(poses)
        features = []
        body_feat = None
        
        for part in self.modes:
            proj_feat = self.proj_linear[part](parts[part]).permute(0, 3, 1, 2)  # B,C,T,V
            gcn_feat = self.gcn_modules[part](proj_feat)  # spatial GCN
            if part == "body": body_feat = gcn_feat
            else:
                assert body_feat is not None
                if part == "left": gcn_feat = gcn_feat + body_feat[..., -2][..., None].detach()
                elif part == "right": gcn_feat = gcn_feat + body_feat[..., -1][..., None].detach()
                elif part == "face_all": gcn_feat = gcn_feat + body_feat[..., 0][..., None].detach()

            gcn_feat = self.fusion_gcn_modules[part](gcn_feat)  # temporal GCN (stride 1)
            pool_feat = gcn_feat.mean(-1).transpose(1, 2)  # B,T,C
            features.append(pool_feat)

        inputs_embeds = torch.cat(features, dim=-1) + self.part_para
        return self.pose_proj(inputs_embeds)  # B,T,out_dim
