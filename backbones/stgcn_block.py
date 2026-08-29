import torch
import torch.nn as nn


class GCN_unit(nn.Module):
    """Spatial graph conv (Uni-Sign ST-GCN: BatchNorm2d + ReLU).

    The normalization submodule is named `bn` to match the released Uni-Sign pose-only checkpoint keys
    (`...gcn.bn.*`, with running stats); conv + adaptive A are the rest.
    """
    def __init__(self, in_channels, out_channels, kernel_size, A, adaptive=True,
                 t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        assert A.size(0) == self.kernel_size
        self.conv = nn.Conv2d(
            in_channels, out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1), padding=(t_padding, 0),
            stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias,
        )
        self.adaptive = adaptive
        if self.adaptive: self.A = nn.Parameter(A.clone())
        else: self.register_buffer('A', A)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, frame_mask=None):
        # Temporally kernel-1 (the t_* args are commented out below), so no frame can reach another: nothing to mask.
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A)).contiguous()
        return self.act(self.bn(x))


class STGCN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, A, adaptive=True, stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        self.gcn = GCN_unit(
            in_channels, out_channels, kernel_sizes[1], A, adaptive, 
            # t_kernel_size=kernel_sizes[0], t_stride=stride, t_padding=(kernel_sizes[0] - 1) // 2
        )
        if kernel_sizes[0] > 1: # temporal_kernel
            self.tcn = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels,
                    kernel_size=(kernel_sizes[0], 1), stride=(stride, 1),
                    padding=((kernel_sizes[0] - 1) // 2, 0),
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            )
        else: self.tcn = nn.Identity()

        if not residual: self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1: self.residual = lambda x: x
        else: self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, frame_mask=None):
        """`frame_mask` (N, T) marks REAL frames. Padded frames are zeroed before the temporal conv.

        The temporal conv has kernel 5 / padding 2, so without this a real frame within 2 steps of the pad reads
        whatever the collator wrote there (repeat-last, i.e. a frozen pose that looks like sustained signing) and
        its output depends on the batch it landed in. Zeroing reproduces exactly what Conv2d's own zero padding
        would contribute at the end of an exact-length sequence, so a padded batch matches an unpadded one.
        """
        m = None if frame_mask is None else frame_mask[:, None, :, None].to(x.dtype)
        if m is not None: x = x * m
        res = self.residual(x)
        x = self.gcn(x, frame_mask)
        # Re-zero AFTER the gcn: its BatchNorm and bias make padded positions non-zero again, and it is the tcn
        # (kernel 5) that would then mix them into real frames.
        if m is not None: x = x * m
        x = self.tcn(x) + res
        return self.act(x)


class STGCNChain(nn.Sequential):
    def __init__(self, in_dim, block_args, kernel_sizes, A, adaptive):
        super(STGCNChain, self).__init__()
        last_dim = in_dim
        for i, [channel, depth] in enumerate(block_args):
            for j in range(depth):
                self.add_module(f'layer{i}_{j}', STGCN_block(last_dim, channel, kernel_sizes, A.clone(), adaptive))
                last_dim = channel

    def forward(self, x, frame_mask=None):
        # nn.Sequential.forward takes one argument and would silently drop the mask, so forward it per block.
        for block in self: x = block(x, frame_mask)
        return x


def get_stgcn_chain(in_dim, level, kernel_size, A, adaptive):
    if level == 'spatial': block_args = [[64, 1], [128, 1], [256, 1]]
    elif level == 'temporal': block_args = [[256, 3]]
    else: raise NotImplementedError
    return STGCNChain(in_dim, block_args, kernel_size, A, adaptive), block_args[-1][0]
