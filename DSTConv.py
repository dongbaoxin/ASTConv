import torch.nn as nn
from functools import partial
import pywt
import pywt.data
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import torch
import torch.nn as nn
from functools import partial
import pywt
import pywt.data
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from PIL import Image
import os
from functools import partial

import pywt  # 添加这行导入

import numpy as np

import torch

import torch.nn.functional as F

from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np
import torch
import torch.nn as nn


class GumbelAngleSelector(nn.Module):
    def __init__(self, candidate_angles, tau_start=5.0, tau_end=0.1, anneal_steps=20000):
        super().__init__()
        self.candidate_angles = nn.Parameter(
            torch.tensor(candidate_angles, dtype=torch.float32),
            requires_grad=False
        )
        init_logits = torch.cos(torch.deg2rad(self.candidate_angles - 45))
        self.logits = nn.Parameter(init_logits * 0.1)
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.anneal_steps = anneal_steps
        self.step = 0

    def anneal_temperature(self):
        frac = min(self.step / self.anneal_steps, 1.0)
        self.current_tau = self.tau_start * (1 - frac) + self.tau_end * frac
        self.step += 1

    def forward(self):
        self.anneal_temperature()
        gumbels = -torch.log(-torch.log(torch.rand_like(self.logits)))
        logits = (self.logits + gumbels) / self.current_tau
        probs = F.softmax(logits, dim=-1)

        # 动态选择单个角度
        discrete_probs = torch.zeros_like(probs)
        hard_indices = torch.argmax(probs, dim=-1, keepdim=True)
        discrete_probs.scatter_(-1, hard_indices, 1.0)
        selected_angle = torch.sum(discrete_probs * self.candidate_angles, dim=-1)

        return selected_angle.item(), probs  # 返回标量角度值


def create_dynamic_shearlet_filter(in_size, angle, device='cuda'):
    """生成单个动态剪切波滤波器"""

    def shearlet_kernel(size=5, shear=1.0):
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)
        kernel = np.exp(-(xx ** 2 + (yy - shear * xx) ** 2) / 0.2)
        return kernel / np.linalg.norm(kernel)

    # 计算剪切参数
    shear = np.tan(np.deg2rad(angle)) if abs(angle % 180) != 90 else 1e5
    kernel = shearlet_kernel(5, shear=shear)

    # 生成滤波器组（单通道）
    dec_filters = torch.tensor(kernel, dtype=torch.float32, device=device
        .unsqueeze(0).unsqueeze(0) # 形状变为 [1, 1, 5, 5]
        .repeat(in_size, 1, 1, 1)) # 扩展为 [in_size, 1, 5, 5]


# 计算伪逆（单滤波器特殊处理）
    dec_filters_np = dec_filters.cpu().numpy().reshape(1, -1)  # [1, 25]
    rec_filters_np = np.linalg.pinv(dec_filters_np).reshape(1, 1, 5, 5)  # [1, 1, 5, 5]
    rec_filters = torch.tensor(rec_filters_np, dtype=torch.float32, device=device)

    return dec_filters, rec_filters

# def wavelet_transform(x, filters):
#
#     b, c, h, w = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
#     x = x.reshape(b, c, 4, h // 2, w // 2)
#     return x
#


# def inverse_wavelet_transform(x, filters):
#
#     b, c, _, h_half, w_half = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     x = x.reshape(b, c * 4, h_half, w_half)
#     x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
#     return x

class DynamicWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 bias=True, wt_levels=1, candidate_angles=[0, 20,40,60,80,100,120]):
        super().__init__()
        assert in_channels == out_channels, "输入输出通道数必须相同"

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.device = 'cuda'

        # 角度选择器
        self.angle_selector = GumbelAngleSelector(candidate_angles)

        # 基础卷积层
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding='same', stride=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1], init_scale=1.0)

        # 卷积路径（保持单滤波器结构）
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size,  # 输入输出通道数调整为 in_channels
                      padding='same', stride=1, groups=in_channels, bias=False)
            for _ in range(wt_levels)
        ])
        self.wavelet_scales = nn.ModuleList([
            _ScaleModule([1, in_channels, 1, 1], init_scale=1.0)
            for _ in range(wt_levels)
        ])

        # 步长处理
        if stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x: F.conv2d(x, self.stride_filter, stride=stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        # 动态选择角度并生成滤波器
        selected_angle, probs = self.angle_selector()
        self.dec_filters, self.rec_filters = create_dynamic_shearlet_filter(
            self.in_channels, selected_angle, device=self.device)


        x_wavelet = self.wt_function(x, self.dec_filters)  # 输出形状: [B, C, 1, H/2, W/2]

        # 基础路径
        x_base = self.base_scale(self.base_conv(x))

        wavelet_features = []
        curr = x_wavelet
        for i in range(self.wt_levels):
            curr = self.wavelet_convs[i](curr)  # 输入输出通道数均为 in_channels
            curr = self.wavelet_scales[i](curr)
            wavelet_features.append(curr)

        # 特征融合（保持维度一致）
        fused = x_base + wavelet_features[-1]


        x_recon = self.iwt_function(fused, self.rec_filters)

        # 步长处理
        if self.stride > 1:
            x_recon = self.do_stride(x_recon)

        return x_recon

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        if c_ == c2:
            self.cv2 = DynamicWTConv2d(c_, c2, 5, 1)
        else:
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class DSTConv1(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )



if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)

    # Model
    mobilenet_v1 = DSTConv1(64, 64)

    out = mobilenet_v1(image)
    print(out.size())