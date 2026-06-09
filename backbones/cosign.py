import torch
import torch.nn as nn
from .gcn_utils import Graph
from .stgcn_block import get_stgcn_chain
from poses import KPS_MODULES


class CoSign1s(nn.Module):
    def __init__(self, temporal_kernel, hidden_size, level='spatial', adaptive=True):
        super().__init__()
        self.graph, A = {}, {}
        self.gcn_modules = {}
        self.linear = nn.Sequential(nn.Linear(3, 64), nn.GELU())

        self.projections = {}
        for module in KPS_MODULES.keys():
            self.graph[module] = Graph(layout=f'{module}', strategy='distance', max_hop=1)
            A[module] = torch.tensor(self.graph[module].A, dtype=torch.float32, requires_grad=False)
            spatial_kernel_size = A[module].size(0)
            self.gcn_modules[module], final_dim = get_stgcn_chain(
                64, level, (temporal_kernel, spatial_kernel_size),
                A[module].clone(), adaptive
            )
            # Create learnable projection for each part to aggregate keypoint information
            num_keypoints = KPS_MODULES[module]['kps_rel_range'][1] - KPS_MODULES[module]['kps_rel_range'][0]
            self.projections[module] = nn.Linear(final_dim * num_keypoints, final_dim)

        self.gcn_modules = nn.ModuleDict(self.gcn_modules)
        self.projections = nn.ModuleDict(self.projections)
        self.fusion = nn.Sequential(nn.Linear(final_dim * len(KPS_MODULES), hidden_size), nn.GELU())
        self.final_dim = final_dim
    
    
    def process_part_features(self, features):
        feat_list = []
        for module, kps_info in KPS_MODULES.items():
            kps_rng = kps_info['kps_rel_range']
            part_feat = self.gcn_modules[module](features[..., kps_rng[0]: kps_rng[1]])
            
            # Reshape from [B, final_dim, T, num_keypoints] to [B, T, final_dim * num_keypoints]
            B, C, T, K = part_feat.shape
            part_feat = part_feat.permute(0, 2, 1, 3).reshape(B, T, C * K)
            
            # Apply learnable projection to aggregate keypoint information
            projected_feat = self.projections[module](part_feat)  # [B, T, final_dim]
            feat_list.append(projected_feat)
        return torch.cat(feat_list, dim=-1) # Shape: [B, T, final_dim * parts]
    
    
    def forward(self, x):
        # linear stage x.shape: [B(N), T, 77(K), 3(C)]
        static = self.linear(x).permute(0, 3, 1, 2) # [B, 64, T, 77]
        cat_feat = self.process_part_features(static) # [B, T, final_dim * parts]
        return self.fusion(cat_feat)  # [B, T, hidden_size]