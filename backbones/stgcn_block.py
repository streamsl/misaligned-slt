import torch
import torch.nn as nn


class _MaskedBatchNorm(torch.autograd.Function):
    # Train-mode BatchNorm2d whose fp32 (fp64 for an fp64 x) batch statistics come from the positions that w 
    # (N, 1, T, 1) marks; every position is normalized with them. The backward is the closed form, so autograd 
    # keeps only x, as BatchNorm itself does.
    @staticmethod
    def forward(ctx, x, w, weight, bias, eps):
        xf, w3 = x.to(torch.promote_types(x.dtype, torch.float32)), w[..., 0]
        count = w3.sum() * x.shape[-1]
        mean = (xf.sum(3) * w3).sum((0, 2)) / count
        centered = xf - mean[:, None, None]
        var = (torch.linalg.vector_norm(centered, dim=3).square() * w3).sum((0, 2)) / count
        invstd = torch.rsqrt(var + eps)
        y = torch.addcmul(bias[:, None, None], centered, (invstd * weight)[:, None, None])
        ctx.save_for_backward(x, w, weight, mean, invstd, count)
        ctx.mark_non_differentiable(mean, var, count)
        return y.to(x.dtype), mean, var, count

    @staticmethod
    def backward(ctx, grad_y, *_):
        # dx = scale * (dy - w / count * (sum dy + xhat * sum(dy * xhat))), the sums over every position (all of 
        # them are normalized) and scale = weight * invstd; written as scale * dy - w * (alpha + beta * x) with 
        # per-channel alpha, beta.
        x, w, weight, mean, invstd, count = ctx.saved_tensors
        gy, xf = grad_y.to(mean.dtype), x.to(mean.dtype)
        grad_bias = gy.sum((0, 2, 3))
        grad_weight = invstd * ((gy * xf).sum((0, 2, 3)) - mean * grad_bias)
        scale = invstd * weight
        beta = scale * invstd * grad_weight / count
        alpha = scale * grad_bias / count - beta * mean
        grad_x = torch.addcmul(
            gy * scale[:, None, None], torch.addcmul(alpha[:, None, None], xf, beta[:, None, None]), w, value=-1
        )
        return grad_x.to(x.dtype), None, grad_weight, grad_bias, None


class _PlaceFrames(torch.autograd.Function):
    # y (N, C, R, V) at the `frames` of dim 2 of a (N, C, length, V) zero tensor. Backward keeps only `frames`: 
    # index_copy's backward keeps y, a second full copy per BatchNorm call.
    @staticmethod
    def forward(ctx, y, frames, length):
        ctx.save_for_backward(frames)
        return y.new_zeros(*y.shape[:2], length, y.shape[3]).index_copy_(2, frames, y)

    @staticmethod
    def backward(ctx, grad):
        frames, = ctx.saved_tensors
        return grad.index_select(2, frames), None, None


def masked_batch_norm(
    bn: nn.BatchNorm2d, x: torch.Tensor, frame_mask: torch.Tensor | None, real_frames: torch.Tensor | None = None
) -> torch.Tensor:
    """`bn(x)` on (N, C, T, V), except that train-mode batch statistics come from the REAL frames only.

    With `frame_mask` (N, T; True = real frame) and `bn` in train mode, batch mean and biased variance are taken over 
    every real (n, t) and all joints v, every position is normalized with them, and running statistics move as torch 
    moves them (momentum, or cumulative average when momentum is None; unbiased variance). So the amount of padding 
    in a batch doesn't change a real frame's features or the running statistics. In eval mode or without mask it is 
    `bn(x)` unchanged (bit-identical). The caller passes None for a batch without padded frame and checks if a masked 
    batch holds a real frame (`UniSignPoseEncoder.forward`, once per forward), so no call here waits on the device.

    `real_frames` (the indices of the frames that `frame_mask` marks real in EVERY row; the encoder's rows end to end) 
    runs `bn` itself on those frames: the same statistics, running statistics and gradients through the native kernel. 
    Other frames output 0; no reader uses them, so their gradient is 0, as in the closed form.
    """
    if not bn.training or frame_mask is None: return bn(x)
    if real_frames is not None: return _PlaceFrames.apply(bn(x.index_select(2, real_frames)), real_frames, x.shape[2])
    w = frame_mask[:, None, :, None].to(torch.promote_types(x.dtype, torch.float32))
    y, mean, var, count = _MaskedBatchNorm.apply(x, w, bn.weight, bn.bias, bn.eps)
    if bn.track_running_stats:
        with torch.no_grad():
            bn.num_batches_tracked.add_(1)
            f = 1.0 / float(bn.num_batches_tracked) if bn.momentum is None else bn.momentum
            bn.running_mean.mul_(1 - f).add_(mean, alpha=f)
            bn.running_var.mul_(1 - f).add_(var * (count / (count - 1)), alpha=f)
    return y


def _run(layers, x, frame_mask, real_frames=None):
    # Calls a Sequential's layers in order so that its BatchNorm reads the mask. State-dict keys stay the Sequential's.
    if not isinstance(layers, nn.Sequential): return layers(x)
    for layer in layers:
        x = masked_batch_norm(layer, x, frame_mask, real_frames) if isinstance(layer, nn.BatchNorm2d) else layer(x)
    return x


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

    def forward(self, x, frame_mask=None, real_frames=None):
        # Temporally kernel-1 (the t_* args are commented out below), so no frame can reach another through the conv;
        # the mask only keeps padded frames out of the train-mode BatchNorm statistics.
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A)).contiguous()
        return self.act(masked_batch_norm(self.bn, x, frame_mask, real_frames))


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

    def forward(self, x, frame_mask=None, real_frames=None):
        """`frame_mask` (N, T) marks REAL frames. Padded frames are zeroed before the temporal conv.

        Temporal conv has kernel 5 / padding 2, so without this a real frame within 2 steps of the pad reads whatever 
        the collator wrote there (repeat-last, i.e. a frozen pose that looks like sustained signing) and its output 
        depends on the batch it landed in. Zeroing reproduces exactly what Conv2d's own 0 padding would contribute at 
        the end of an exact-length sequence, so a padded batch matches an unpadded one. In train mode every BatchNorm 
        (gcn.bn, tcn[1], residual[1]) takes its batch statistics from the real frames only (`masked_batch_norm`).
        """
        m = None if frame_mask is None else frame_mask[:, None, :, None].to(x.dtype)
        if m is not None: x = x * m
        res = _run(self.residual, x, frame_mask, real_frames)
        x = self.gcn(x, frame_mask, real_frames)
        # Re-zero AFTER the gcn: its BatchNorm and bias make padded positions non-zero again, and it is the tcn
        # (kernel 5) that would then mix them into real frames.
        if m is not None: x = x * m
        x = _run(self.tcn, x, frame_mask, real_frames) + res
        return self.act(x)


class STGCNChain(nn.Sequential):
    def __init__(self, in_dim, block_args, kernel_sizes, A, adaptive):
        super(STGCNChain, self).__init__()
        last_dim = in_dim
        for i, [channel, depth] in enumerate(block_args):
            for j in range(depth):
                self.add_module(f'layer{i}_{j}', STGCN_block(last_dim, channel, kernel_sizes, A.clone(), adaptive))
                last_dim = channel

    def forward(self, x, frame_mask=None, real_frames=None):
        # nn.Sequential.forward takes one argument and would silently drop the mask, so forward it per block.
        for block in self: x = block(x, frame_mask, real_frames)
        return x


def get_stgcn_chain(in_dim, level, kernel_size, A, adaptive):
    if level == 'spatial': block_args = [[64, 1], [128, 1], [256, 1]]
    elif level == 'temporal': block_args = [[256, 3]]
    else: raise NotImplementedError
    return STGCNChain(in_dim, block_args, kernel_size, A, adaptive), block_args[-1][0]
