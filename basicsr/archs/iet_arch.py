'''
An official Pytorch impl of `From Local Windows to Adaptive Candidates via Individualized Exploratory:
Rethinking Attention for Image Super-Resolution`.

Arxiv: 'https://arxiv.org/abs/2601.08341'
'''

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from fairscale.nn import checkpoint_wrapper
from natten.functional import na2d_qk, na2d_av, na2d
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from basicsr.utils.registry import ARCH_REGISTRY
import smm_cuda

class SMM_QmK(Function):
    """
    A custom PyTorch autograd Function for sparse matrix multiplication (SMM) of
    query (Q) and key (K) matrices, based on given sparse indices.

    This function leverages a CUDA-implemented kernel for efficient computation.

    Forward computation:
        Computes the sparse matrix multiplication using a custom CUDA function.

    Backward computation:
        Computes the gradients of A and B using a CUDA-implemented backward function.
    """

    @staticmethod
    def forward(ctx, A, B, index):
        """
        Forward function for Sparse Matrix Multiplication QmK.

        Args:
            ctx: Autograd context to save tensors for backward computation.
            A: Input tensor A (Query matrix).
            B: Input tensor B (Key matrix).
            index: Index tensor specifying the sparse multiplication positions.

        Returns:
            Tensor: Result of the sparse matrix multiplication.
        """
        # Save input tensors for backward computation
        ctx.save_for_backward(A, B, index)

        # Call the custom CUDA forward function for sparse matrix multiplication
        return smm_cuda.SMM_QmK_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Backward function for Sparse Matrix Multiplication QmK.

        Args:
            ctx: Autograd context to retrieve saved tensors.
            grad_output: Gradient of the output from the forward pass.

        Returns:
            Tuple: Gradients of the inputs A and B, with None for the index as it is not trainable.
        """
        # Retrieve saved tensors from the forward pass
        A, B, index = ctx.saved_tensors

        # Compute gradients using the custom CUDA backward function
        grad_A, grad_B = smm_cuda.SMM_QmK_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )

        # Return gradients for A and B, no gradient for index
        return grad_A, grad_B, None

class SMM_AmV(Function):
    """
    A custom PyTorch autograd Function for sparse matrix multiplication (SMM)
    between an activation matrix (A) and a value matrix (V), guided by sparse indices.

    This function utilizes a CUDA-optimized implementation for efficient computation.

    Forward computation:
        Computes the sparse matrix multiplication using a custom CUDA function.

    Backward computation:
        Computes the gradients of A and B using a CUDA-implemented backward function.
    """

    @staticmethod
    def forward(ctx, A, B, index):
        """
        Forward function for Sparse Matrix Multiplication AmV.

        Args:
            ctx: Autograd context to save tensors for backward computation.
            A: Input tensor A (Activation matrix).
            B: Input tensor B (Value matrix).
            index: Index tensor specifying the sparse multiplication positions.

        Returns:
            Tensor: Result of the sparse matrix multiplication.
        """
        # Save tensors for backward computation
        ctx.save_for_backward(A, B, index)

        # Call the custom CUDA forward function
        return smm_cuda.SMM_AmV_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Backward function for Sparse Matrix Multiplication AmV.

        Args:
            ctx: Autograd context to retrieve saved tensors.
            grad_output: Gradient of the output from the forward pass.

        Returns:
            Tuple: Gradients of the inputs A and B, with None for the index as it is not trainable.
        """
        # Retrieve saved tensors from the forward pass
        A, B, index = ctx.saved_tensors

        # Compute gradients using the custom CUDA backward function
        grad_A, grad_B = smm_cuda.SMM_AmV_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )

        # Return gradients for A and B, no gradient for index
        return grad_A, grad_B, None


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.in_features = in_features
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class ConvFFN_sim(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        head_features = in_features // num_heads

        self.dwconv0 = dwconv(hidden_features=head_features, kernel_size=kernel_size)
        self.fc0 = nn.Linear(head_features*2, head_features)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv1 = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_sim, x_size):
        b, n, c = x.shape
        x = x.view(b, n, self.num_heads, -1).permute(0, 2, 1, 3).reshape(b * self.num_heads, n, -1)
        x_sim = x_sim.view(b, n, self.num_heads, -1).permute(0, 2, 1, 3).reshape(b * self.num_heads, n, -1)

        x = self.fc0(torch.cat([x, x_sim], dim=-1))
        x = self.act(x)
        x = x + self.dwconv0(x, x_size)
        x = x.view(b, self.num_heads, n, -1).permute(0, 2, 1, 3).reshape(b, n, -1)

        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv1(x, x_size)
        x = self.fc2(x)
        return x


def batched_isin(elements: torch.Tensor, test_elements: torch.Tensor) -> torch.Tensor:
    """
    elements: (b, n)
    test_elements: (b, m)
    return: (b, n) bool mask
    """
    test_elements_sorted, _ = test_elements.sort(dim=-1)
    idx = torch.searchsorted(test_elements_sorted, elements, right=False)

    idx = torch.clamp(idx, max=test_elements.size(1) - 1)
    gathered = torch.gather(test_elements_sorted, 1, idx)

    mask = (gathered == elements) & (idx < test_elements.size(1))
    return mask


def get_local_indices_dilation1(h, w, r, device='cuda'):
    assert r % 2 == 1, "only support odd r"
    device = torch.device(device)

    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device),
                                    torch.arange(w, device=device), indexing='ij')
    center_y = grid_y.reshape(-1)
    center_x = grid_x.reshape(-1)

    radius = r // 2
    top = torch.clamp(center_y - radius, 0, h - r)
    left = torch.clamp(center_x - radius, 0, w - r)

    dy = torch.arange(r, device=device)
    dx = torch.arange(r, device=device)
    offset_y, offset_x = torch.meshgrid(dy, dx, indexing='ij')

    neigh_y = top.unsqueeze(1).unsqueeze(2) + offset_y
    neigh_x = left.unsqueeze(1).unsqueeze(2) + offset_x

    neigh_idx = neigh_y * w + neigh_x
    neigh_idx = neigh_idx.reshape(-1, r * r)

    return neigh_idx


def get_local_indices(h, w, r, dilation=1, device='cuda'):
    assert r % 2 == 1, "only support odd r"
    device = torch.device(device)
    result = torch.empty((h * w, r * r), dtype=torch.long, device=device)

    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device),
                                    torch.arange(w, device=device), indexing='ij')

    for dy in range(dilation):
        for dx in range(dilation):
            mask = (grid_y % dilation == dy) & (grid_x % dilation == dx)
            if mask.sum() == 0:
                continue

            sub_flat_idx = (grid_y[mask] * w + grid_x[mask]).flatten()
            h_sub = (h + dilation - 1 - dy) // dilation
            w_sub = (w + dilation - 1 - dx) // dilation

            sub_local_idx = get_local_indices_dilation1(h_sub, w_sub, r, device=device)

            mapped_y = dy + dilation * (sub_local_idx // w_sub)
            mapped_x = dx + dilation * (sub_local_idx % w_sub)
            mapped_idx = mapped_y * w + mapped_x

            result[sub_flat_idx] = mapped_idx

    return result.to(torch.int32)


def merge_and_pad(idx: torch.LongTensor, prop_idx: torch.LongTensor, K3: int):
    """
    Merge idx and prop_idx, deduplicate them, and keep at most K3 unique values.
    Args:
        idx:      (B, N, K1)
        prop_idx: (B, N, K2)
    Returns:
        new_idx:  (B, N, K3)
    """
    B, N, K1 = idx.shape
    K2 = prop_idx.shape[2]
    K = K1 + K2

    # Merge and reshape to (B*N, K).
    combined = torch.cat([idx, prop_idx], dim=-1).reshape(B * N, K)

    # Sort before deduplication.
    sorted_vals = combined.sort(dim=1)[0]

    uniq_mask = torch.ones_like(sorted_vals, dtype=torch.bool)
    uniq_mask[:, 1:] = sorted_vals[:, 1:] != sorted_vals[:, :-1]

    # Build the deduplicated unique elements.
    sorted_vals = sorted_vals.to(torch.int32)
    positions = torch.cumsum(uniq_mask, dim=1, dtype=torch.long) - 1
    uniq_vals = torch.full_like(sorted_vals, -1, dtype=torch.int32)
    uniq_vals.scatter_(1, positions, sorted_vals)

    # Take the first K3 unique elements.
    return uniq_vals[:, :K3].view(B, N, K3)



class IEA(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """
    def __init__(
        self, dim, local_range, sparse_range, num_heads, qkv_bias=True, dilation=1, topk_focus=0, topk_prop_1=0, topk_prop_2=0, topk_prop_3=0, rpi=False,
                 is_first=None, is_last=None, is_first_block=None, is_last_block=None, nattn_dim=-1, block_idx=-1, layer_idx=-1, progressive_layer=None):

        super().__init__()
        self.dim = dim
        self.local_range = local_range  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.dilation=dilation
        head_dim = nattn_dim // num_heads
        self.scale = head_dim**-0.5
        self.eps = 1e-20
        self.topk_focus = topk_focus
        self.topk_prop_1 = topk_prop_1
        self.topk_prop_2 = topk_prop_2
        self.topk_prop_3 = topk_prop_3
        self.is_first = is_first
        self.is_last = is_last
        self.is_first_block = is_first_block
        self.is_last_block = is_last_block
        self.nattn_dim = nattn_dim
        self.block_idx = block_idx
        self.layer_idx = layer_idx
        self.local_range = local_range
        self.sparse_range = sparse_range
        self.progressive_layer = progressive_layer

        # define a parameter table of relative position bias
        if rpi:
            self.relative_position_bias = nn.Parameter(
                torch.zeros(num_heads, local_range * local_range))  # nH, k
            trunc_normal_(self.relative_position_bias, std=.02)

        self.proj = nn.Linear(nattn_dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def add_relative_position_bias(self, attn):
        """
        Add relative position bias to the center region of attn.

        Args:
            attn: Tensor of shape (b, h, H, W, ws^2)

        Returns:
            attn: The modified original tensor or a new tensor, depending on training mode.
        """
        b, h, H, W, ws2 = attn.shape

        assert ws2 == self.local_range * self.local_range, f"local_range != attn last dim: {ws2} != {self.local_range} * {self.local_range}"
        assert self.relative_position_bias.shape == (h, ws2), f"expected (h, ws²), got {self.relative_position_bias.shape}"

        h_start = self.local_range // 2
        h_end = H - self.local_range // 2
        w_start = self.local_range // 2
        w_end = W - self.local_range // 2

        bias = self.relative_position_bias[:, None, None, :]

        if self.training:
            attn = attn.clone()
        attn[:, :, h_start:h_end, w_start:w_end, :] += bias

        return attn

    def forward(self, qkv, v_lepe, params, x_size):
        r"""
        Args:
            qkv: Input query, key, and value tokens with shape of (num_windows*b, n, c*3)
            rpi: Relative position index
            mask (0/-inf):  Mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, n, c3 = qkv.shape
        c = c3 // 3
        h, w = x_size

        lr2 = self.local_range ** 2
        sr2 = self.sparse_range ** 2

        topk = params['idx'].shape[-1]    if params['idx'] is not None else lr2 + sr2

        # calculate similarity map
        if topk == lr2 + sr2:
            qkv = qkv.reshape(b, h, w, 3, self.num_heads, c // self.num_heads).permute(3, 0, 4, 1, 2, 5)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            q = q * self.scale  # b, head, h, w, c

            attn_dense = na2d_qk(q, k, kernel_size=self.local_range, dilation=1)
            attn_sparse = na2d_qk(q, k, kernel_size=self.sparse_range, dilation=self.dilation).reshape(b*self.num_heads, h*w, sr2)  # (B*H, N, r2*r2)

            if hasattr(self, 'relative_position_bias'):
                attn_dense = self.add_relative_position_bias(attn_dense).reshape(b * self.num_heads, h*w, lr2)  # (B*H, N, r1*r1)
            attn_sparse.masked_fill_(params['mask'].unsqueeze(0).expand(b * self.num_heads, -1, -1), float('-inf'))

            attn = torch.cat([attn_dense, attn_sparse], dim=-1)  # (B*H, N, r1^2 + r2^2)

        else:
            qkv = qkv.reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            q = q * self.scale  # b ,h, n, c_h

            q = q.reshape(b * self.num_heads, n, c // self.num_heads)
            k = k.reshape(b * self.num_heads, n, c // self.num_heads)
            k = k.transpose(-2, -1)

            smm_index = params['idx'].int().contiguous()
            attn = SMM_QmK.apply(q, k, smm_index)  # b * self.num_heads, n, topk

        if not self.training:
            attn = torch.softmax(attn, dim=-1, out=attn)
        else:
            attn = self.softmax(attn)

        # cascaded relationship
        if not self.is_first or self.block_idx in self.progressive_layer:
            if self.training:
                attn = attn * params['attn']
                attn = (attn+self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)
            else:
                attn.mul_(params['attn'])
                attn.add_(self.eps)
                denom = attn.sum(dim=-1, keepdim=True).add_(self.eps)
                attn.div_(denom)

        # prune
        if self.topk_focus < topk:
            params['idx'] = params['idx'] if params['idx'] is not None else torch.cat([params['idx_dense'], params['idx_sparse']], dim=-1).unsqueeze(0).expand(b * self.num_heads, -1 , -1)  # (B*H, N, r1^2 + r2^2)
            attn, topk_indices = torch.topk(attn, k=self.topk_focus, dim=-1, largest=True, sorted=True)
            params['idx'] = torch.gather(params['idx'], dim=-1, index=topk_indices)  # b*h, n, topk_focus

        params['attn'] = attn

        # reconstruct
        if self.topk_focus == lr2 + sr2:
            attn = attn.view(b, self.num_heads, h, w, lr2 + sr2)
            x_dense = na2d_av(attn[..., :lr2].contiguous(), v, kernel_size=self.local_range, dilation=1).permute(0, 2, 3, 1, 4).reshape(b, n, c)
            x_sparse = na2d_av(attn[..., lr2:].contiguous(), v, kernel_size=self.sparse_range, dilation=self.dilation).permute(0, 2, 3, 1, 4).reshape(b, n, c)
            x = x_dense + x_sparse
        else:
            v = v.reshape(b * self.num_heads, n, c // self.num_heads)
            if v.dtype != attn.dtype:
                v = v.to(attn.dtype)
            smm_index = params['idx'].int().contiguous()
            x = (SMM_AmV.apply(attn, v, smm_index).view(b, self.num_heads, n, c // self.num_heads)).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x + v_lepe)

        # propagation
        if self.is_last and self.topk_prop_3 > 0:
            with torch.no_grad():
                _, topk_prop_1_idx = torch.topk(params['attn'], k=self.topk_prop_1, dim=-1, largest=True, sorted=True)  # b*h, n, topk_prop_1
                topk_prop_1_idx = torch.gather(params['idx'], dim=2, index=topk_prop_1_idx)  # (b*h, n, topk_prop_1)
                idx_topk2 = topk_prop_1_idx[..., :self.topk_prop_2]  # (b*h, n, topk_prop_2)

                prop_idx = torch.gather(idx_topk2.unsqueeze(2).expand(-1, -1, self.topk_prop_1, -1), dim=1,
                                        index=topk_prop_1_idx.long().unsqueeze(-1).expand(-1, -1, -1, self.topk_prop_2))  # b*h, n, k1, k2
                prop_idx = prop_idx.view(b * self.num_heads, n, self.topk_prop_1 * self.topk_prop_2)  # b*h, n, k1*k2

                params['idx'] = merge_and_pad(params['idx'], prop_idx, self.topk_focus + self.topk_prop_3)

        #ffn
        if self.block_idx > 5:
            self_pos = torch.arange(n, device=x.device).view(1, n).expand(b * self.num_heads, -1)

            _, topk_indices = torch.topk(attn, 2, dim=-1, largest=True, sorted=False)
            first_two_idx = torch.gather(params['idx'], dim=-1, index=topk_indices)

            idx_sim = first_two_idx[..., 0].clone()  # (b*h, n)
            mask = (idx_sim == self_pos)
            idx_sim = torch.where(mask, first_two_idx[..., 1], idx_sim)

            x_sim = x.view(b, n, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3).reshape(-1, n, self.dim // self.num_heads)
            x_sim = torch.gather(x_sim, dim=1, index=idx_sim.long().unsqueeze(-1).expand(-1, -1, self.dim // self.num_heads))
            x_sim = x_sim.view(b, self.num_heads, n, self.dim // self.num_heads).permute(0, 2, 1, 3).reshape(b, n, self.dim)
        else:
            x_sim = None

        return x, x_sim

    def extra_repr(self) -> str:
        return f'dim={self.dim}, local_range={self.local_range}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

    def flops(self, n):
        flops = 0

        # attn = (q @ k.transpose(-2, -1))
        if not self.is_last or not self.is_last_block:
            if not self.is_last:
                flops += n * self.nattn_dim * self.topk_focus
            else:
                flops += n * self.nattn_dim * (self.topk_focus + self.topk_prop_3)
        else:
            flops += n * self.nattn_dim * (self.local_range ** 2 + self.sparse_range ** 2)

        #  x = (attn @ v)
        flops += n * self.topk_focus * self.nattn_dim

        # x = self.proj(x)
        flops += n * self.nattn_dim * self.dim

        return flops


class IETTransformerLayer(nn.Module):
    r"""
    IET Transformer Layer

    Args:
        dim (int): Number of input channels.
        idx (int): Layer index.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        category_size (int): Category size for AC-MSA.
        num_tokens (int): Token number for each token dictionary.
        reducted_dim (int): Reducted dimension number for query and key matrix.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        is_last (bool): True if this layer is the last of a IET Block. Default: False
    """

    def __init__(self,
                 dim,
                 block_idx,
                 layer_idx,
                 input_resolution,
                 num_heads,
                 local_range,
                 sparse_range,
                 convffn_kernel_size,
                 mlp_ratio,
                 dilation,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 topk_focus=0,
                 topk_prop_1=0,
                 topk_prop_2=0,
                 topk_prop_3=0,
                 rpi=False,
                 nattn_dim=240,
                 is_first=None,
                 is_last=None,
                 is_first_block=None,
                 is_last_block=None,
                 progressive_layer=None
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.nattn_dim = nattn_dim if is_first_block else dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.block_idx = block_idx

        self.wqkv = nn.Linear(dim, 3*self.nattn_dim, bias=qkv_bias)


        self.v_LePE = dwconv(hidden_features=self.nattn_dim, kernel_size=convffn_kernel_size)

        self.attn_win = IEA(
            self.dim,
            local_range=local_range,
            sparse_range=sparse_range,
            num_heads=num_heads,
            dilation=dilation,
            qkv_bias=qkv_bias,
            topk_focus=topk_focus,
            topk_prop_1=topk_prop_1,
            topk_prop_2=topk_prop_2,
            topk_prop_3=topk_prop_3,
            rpi=rpi,
            is_first=is_first,
            is_last=is_last,
            is_first_block=is_first_block,
            is_last_block=is_last_block,
            nattn_dim=self.nattn_dim,
            block_idx=block_idx,
            layer_idx=layer_idx,
            progressive_layer=progressive_layer,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        if block_idx <= 5:
            self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, act_layer=act_layer)
        else:
            self.norm3 = norm_layer(dim)
            self.convffn = ConvFFN_sim(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, act_layer=act_layer, num_heads=num_heads)


    def forward(self, x, x_size, params):
        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)

        v_lepe = self.v_LePE(torch.split(qkv, self.nattn_dim, dim=-1)[-1], x_size)

        x, x_sim = self.attn_win(qkv, v_lepe, params, x_size)

        x = shortcut + x

        # FFN
        if x_sim is None:
            x = x + self.convffn(self.norm2(x), x_size)
        else:
            x = x + self.convffn(self.norm2(x), self.norm3(x_sim), x_size)

        return x


    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution

        # qkv = self.wqkv(x)
        flops += self.dim * 3 * self.dim * h * w

        # W-MSA/SW-MSA
        flops += self.attn_win.flops(h * w)

        # mlp
        if self.block_idx <= 5:
            flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
            flops += h * w * self.dim * self.convffn_kernel_size**2 * self.mlp_ratio
        else:
            flops += h * w * self.dim // self.num_heads * self.dim // self.num_heads * 2
            flops += h * w * self.dim // self.num_heads * self.convffn_kernel_size**2

            flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
            flops += h * w * self.dim * self.convffn_kernel_size**2 * self.mlp_ratio

        # lepe
        flops += h * w * self.dim * (self.convffn_kernel_size ** 2)

        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self, input_resolution=None):
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops = h * w * self.dim
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicBlock(nn.Module):
    """ A basic IET Block for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        idx (int): Block index.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        category_size (int): Category size for AC-MSA.
        num_tokens (int): Token number for each token dictionary.
        reducted_dim (int): Reducted dimension number for query and key matrix.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 idx,
                 depth,
                 num_heads,
                 local_range,
                 sparse_range,
                 convffn_kernel_size,
                 mlp_ratio=4.,
                 dilation=1,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 is_last_block=False,
                topk_focus=None,
                topk_prop_1=None,
                topk_prop_2=None,
                topk_prop_3=None,
                nattn_dim=240,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.idx = idx

        self.layers = nn.ModuleList()
        max_num = local_range ** 2 + sparse_range ** 2

        for i in range(depth):
            self.layers.append(
                IETTransformerLayer(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    local_range=local_range,
                    sparse_range=sparse_range,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    dilation=dilation,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    topk_focus=topk_focus[idx][i],
                    topk_prop_1=topk_prop_1[idx],
                    topk_prop_2=topk_prop_2[idx],
                    topk_prop_3=topk_prop_3[idx],
                    rpi=topk_focus[idx][i]==max_num or topk_focus[idx][i-1]==max_num,
                    nattn_dim=nattn_dim,
                    block_idx=idx,
                    layer_idx=i,
                    is_first=i == 0,
                    is_last=i == depth -1,
                    is_first_block=idx == 0,
                    is_last_block=is_last_block,
                    progressive_layer=[i+1 for i, x in enumerate(topk_prop_3) if x == 0]
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        b, n, c = x.shape
        for layer in self.layers:
            # adjust the value of idx_checkpoint to change the number of layers processed by checkpoint_wrapper
            # increase the value of idx_checkpoint could save more GPU memory footprint but slow down the training
            # idx_checkpoint need to be set as at least 4 for eight 24G GPU when training IET
            idx_checkpoint = 0
            if self.use_checkpoint and self.idx < idx_checkpoint:
                layer = checkpoint_wrapper(layer, offload_to_cpu=False)
            x= layer(x, x_size, params)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        if self.downsample is not None:
            flops += self.downsample.flops(input_resolution)
        return flops


class IETB(nn.Module):
    """IET Block (IETB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 idx,
                 input_resolution,
                 depth,
                 num_heads,
                 local_range,
                 sparse_range,
                 convffn_kernel_size,
                 mlp_ratio,
                 dilation,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv',
                 topk_focus=None,
                 topk_prop_1=None,
                 topk_prop_2=None,
                 topk_prop_3=None,
                 nattn_dim=240,
                 is_last=False):
        super(IETB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.residual_group = BasicBlock(
            dim=dim,
            input_resolution=input_resolution,
            idx=idx,
            depth=depth,
            num_heads=num_heads,
            local_range=local_range,
            sparse_range=sparse_range,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            dilation=dilation,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_last_block=is_last,
            topk_focus=topk_focus,
            topk_prop_1=topk_prop_1,
            topk_prop_2=topk_prop_2,
            topk_prop_3=topk_prop_3,
            nattn_dim=nattn_dim,
        )

        self.idx = idx
        if idx in [0, 2, 4, 6, 7]:
            self.conv = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size, params):
        if  hasattr(self, 'conv'):
            return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x
        else:
            return self.residual_group(x, x_size, params) + x

    def flops(self, input_resolution=None):
        flops = 0
        flops += self.residual_group.flops(input_resolution)
        h, w = self.input_resolution if input_resolution is None else input_resolution
        if hasattr(self, 'conv'):
            if len(self.conv) == 1:
                flops += h * w * self.dim * self.dim * 9
            else:
                flops += h * w * self.dim * self.dim * 9 // 2 + h * w * self.dim * self.dim // 16
        flops += self.patch_embed.flops(input_resolution)
        flops += self.patch_unembed.flops(input_resolution)

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.img_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self, input_resolution=None):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        x, y = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * x * y * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 9 * self.num_feat * 9 * x * y
        return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        h, w = self.patches_resolution if input_resolution is None else input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


@ARCH_REGISTRY.register()
class IET(nn.Module):
    r""" IET
        A PyTorch impl of IET for single image super-resolution.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=90,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 local_range=8,
                 sparse_range=8,
                 dilation=1,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 topk_focus=((914, 914, 225, 225),
                             (345, 225, 225, 125),
                             (225, 125, 125, 125),
                             (185, 81, 81, 81),
                             (121, 81, 81, 81),
                             (64, 64, 64, 64),
                             (36, 36, 36, 36),
                             (24, 24, 24, 24)),
                 topk_prop_1=(22, 20, 14, 12, 0, 0, 0, 0),
                 topk_prop_2=(12, 11, 9, 8, 0, 0, 0, 0),
                 topk_prop_3=(120, 100, 60, 40, 0, 0, 0, 0),
                 nattn_dim=240,
                 cur_dim=None,
                 natten_dim=None,
                 **kwargs):
        super().__init__()
        if cur_dim is not None:
            nattn_dim = cur_dim
        if natten_dim is not None:
            nattn_dim = natten_dim
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.num_heads = num_heads
        self.dilation = dilation
        self.local_range = local_range
        self.sparse_range = sparse_range
        self.img_size = img_size
        self.nattn_dim = nattn_dim

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # build Residual IET Blocks (IETB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = IETB(
                dim=embed_dim,
                idx=i_layer,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads,
                local_range=local_range,
                sparse_range=sparse_range,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                dilation=dilation,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                topk_focus=topk_focus,
                topk_prop_1=topk_prop_1,
                topk_prop_2=topk_prop_2,
                topk_prop_3=topk_prop_3,
                nattn_dim=nattn_dim,
                is_last=i_layer == self.num_layers - 1
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

        if self.training:
            self.params = {
                'idx_dense': get_local_indices(img_size, img_size, self.local_range, 1, 'cuda'),
                'idx_sparse': get_local_indices(img_size, img_size, sparse_range, self.dilation, 'cuda'),
                'idx': None,
            }
            self.params['mask'] = batched_isin(self.params['idx_sparse'], self.params['idx_dense'])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])  # torch.Size([1, 48, 336, 512])
        x = self.patch_embed(x)  # torch.Size([1, 172032, 48])
        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        # print(x.shape)  # torch.Size([1, 3, 322, 512])
        # padding
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.sparse_range * self.dilation
        if self.training:
            mod = max(mod, self.img_size)
        if h_ori < mod:
            x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :mod, :]
            h = mod
        else:
            h = h_ori
        if w_ori < mod:
            x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :mod]
            w = mod
        else:
            w = w_ori

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # idx = int(w * int(h * h_ratio) + w * w_ratio)
        if not self.training:
            idx_dense = get_local_indices(h, w, self.local_range, 1, x.device)
            idx_sparse = get_local_indices(h, w, self.sparse_range, self.dilation, x.device)
            params = {
                'idx_dense': idx_dense,
                'idx_sparse': idx_sparse,
                'idx': None,
                    }
            params['mask'] = batched_isin(params['idx_sparse'], params['idx_dense'])
        else:
            params = self.params

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)  # nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
            x = self.conv_after_body(self.forward_features(x, params)) + x  # nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            x = self.upsample(x)  # torch.Size([1, 3, 672, 1024])
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        params['idx'] = None

        # unpadding
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]

        return x

    def flops(self, input_resolution=None):
        flops = 0
        resolution = self.patches_resolution if input_resolution is None else input_resolution
        h, w = resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops(resolution)
        for layer in self.layers:
            flops += layer.flops(resolution)
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        if self.upsampler == 'pixelshuffle':
            flops += self.upsample.flops(resolution)
        else:
            flops += self.upsample.flops(resolution)

        return flops


if __name__ == '__main__':
    model = IET(
            upscale=2,
            img_size=64,
            embed_dim=240,
            topk_focus=[[914, 914, 225, 225],
                        [345, 225, 225, 125],
                        [225, 125, 125, 125],
                        [185, 81, 81, 81],
                        [121, 81, 81, 81],
                        [64, 64, 64, 64],
                        [36, 36, 36, 36],
                        [24, 24, 24, 24]],
            topk_prop_1=[22, 20, 14, 12, 0, 0, 0, 0],
            topk_prop_2=[12, 10, 7, 6, 0, 0, 0, 0],
            topk_prop_3=[120, 100, 60, 40, 0, 0, 0, 0],
            depths=[4, 4, 4, 4, 4, 4, 4, 4,],
            num_heads=6,
            local_range=17,
            sparse_range=25,
            dilation=2,
            convffn_kernel_size=5,
            img_range=1.,
            mlp_ratio=2,
            upsampler='pixelshuffle')

    # Model Size
    total = sum([param.nelement() for param in model.parameters()])
    print("Params: %.1fM" % (total / 1e6))
    # print(model.flops([320, 180]) / 1e12, 'T')
    # print(model.flops([426, 240]) / 1e12, 'T')
    print('FLOPs: %.2fT' % (model.flops([640, 360]) / 1e12))

    # Test
    # _input = torch.randn([2, 3, 64, 64])
    # output = model(_input)
    # print(output.shape)
