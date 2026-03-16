"""Example file, implementing a more or less modern Unet."""

import logging
from math import ceil

import torch
import torch.nn as nn
from jaxtyping import Float as F
from torch import Tensor as T

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _round_filters(filters: int, factor: float, divisor: int = 8, min_depth: int | None = None) -> int:
    if factor == 1.0:
        return filters
    min_depth = min_depth or divisor
    v = filters * factor
    new_v = max(min_depth, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def _round_repeats(repeats: int, factor: float) -> int:
    if factor == 1.0:
        return repeats
    return int(ceil(repeats * factor))


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU() if act else nn.Identity(),
        )

    def forward(self, x: F[T, "B C H W"]) -> F[T, "B C2 H2 W2"]:
        return self.net(x)


class SqueezeExcite(nn.Module):
    """Per-channel learned scaling of the input tensor.

    Ref: [Hu et al., 2019](http://arxiv.org/abs/1709.01507)
    """
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        reduced = max(8, int(channels * se_ratio))
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: F[T, "B C H W"]) -> F[T, "B C H W"]:
        return x * self.net(x)


class MBConv(nn.Module):
    """MobileConvNet block [Sandler et al., 2018](https://doi.org/10.1109/CVPR.2018.00474)."""
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 4, se_ratio: float = 0.25):
        super().__init__()
        mid = int(in_channels * expand_ratio)
        self.use_res = in_channels == out_channels

        self.net = nn.Sequential(
            ConvBNAct(in_channels, mid, kernel_size=1, act=True) if expand_ratio != 1 else nn.Identity(),
            ConvBNAct(mid, mid, kernel_size=3, groups=mid, act=True),
            SqueezeExcite(mid, se_ratio=se_ratio),
            ConvBNAct(mid, out_channels, kernel_size=1, act=False),
        )

    def forward(self, x: F[T, "B C H W"]) -> F[T, "B C2 H W"]:
        y = self.net(x)
        return y + x if self.use_res else y


def _mbconv_stage(in_channels: int, out_channels: int, repeats: int, expand_ratio: int = 4, se_ratio: float = 0.25):
    return nn.Sequential(
        *[
            MBConv(in_channels if i == 0 else out_channels, out_channels, expand_ratio=expand_ratio, se_ratio=se_ratio)
            for i in range(repeats)
        ]
    )


class Unet(nn.Module):
    """Efficient-Unet ([Tan and Le, 2020](http://arxiv.org/abs/1905.11946)) for segmentation, optionally integrates embeddings in the bottleneck."""

    def __init__(
        self,
        in_channels: int,
        factor: float = 1.0,
        pools: tuple[int, int] = (7, 2),
    ):
        super().__init__()

        base_hidden = (64, 128, 256)
        hidden = tuple(_round_filters(h, factor) for h in base_hidden)
        base_repeats = (2, 2, 3)
        repeats = tuple(_round_repeats(r, factor) for r in base_repeats)

        # down-sampling (encoder) path
        self.down1 = _mbconv_stage(in_channels, hidden[0], repeats=repeats[0], expand_ratio=4, se_ratio=0.25)
        self.down2 = _mbconv_stage(hidden[0], hidden[1], repeats=repeats[1], expand_ratio=4, se_ratio=0.25)
        self.down3 = _mbconv_stage(hidden[1], hidden[2], repeats=repeats[2], expand_ratio=4, se_ratio=0.25)
        self.pool1 = nn.MaxPool2d(kernel_size=pools[0], stride=pools[0])
        self.pool2 = nn.MaxPool2d(kernel_size=pools[1], stride=pools[1])

        # up-sampling (decoder) path
        self.up_trans1 = nn.ConvTranspose2d(hidden[2], hidden[1], kernel_size=pools[1], stride=pools[1])
        self.conv_up1 = _mbconv_stage(hidden[1] + hidden[1], hidden[1], repeats=_round_repeats(2, factor), expand_ratio=4, se_ratio=0.25)

        self.up_trans2 = nn.ConvTranspose2d(hidden[1], hidden[0], kernel_size=pools[0], stride=pools[0])
        self.conv_up2 = _mbconv_stage(hidden[0] + hidden[0], hidden[0], repeats=_round_repeats(2, factor), expand_ratio=4, se_ratio=0.25)

        # 1×1 conv to get a single output channel (segmentation target)
        self.out_conv = nn.Conv2d(hidden[0], 1, kernel_size=1)

    def forward(self, x: F[T, "B C H W"]) -> F[T, "B H W"]:
        """Forward patches, integrating information from the FM embeddings.

        Args:
            x(torch.Tensor, [Batch, Channels, Height, Width]): batch instance
            embeddings(torch.Tensor, [B, Height_2, Width_2, Emb]): embeddings from
                foundational model. If `None`, they are not used.
        """
        # Encoder
        c1 = self.down1(x)  # shape: (B, H1, H, W)
        p1 = self.pool1(c1)  # shape: (B, H1, H/2, W/2)
        c2 = self.down2(p1)  # shape: (B, H2, H/2, W/2)
        p2 = self.pool2(c2)  # shape: (B, H2, H/4, W/4)
        c3 = self.down3(p2)  # shape: (B, H3, H/4, W/4)

        # Decoder, with residual connections to the corresponding encoder steps
        u1 = self.up_trans1(c3)  # upsample to (B, H2, H/2, W/2)
        u1 = torch.cat([u1, c2], dim=1)  # shape: (B, H2+H2, H/2, W/2)
        u1 = self.conv_up1(u1)  # shape: (B, H2, H/2, W/2)
        u2 = self.up_trans2(u1)  # upsample to (B, H1, H, W)
        u2 = torch.cat([u2, c1], dim=1)  # shape: (B, H1+H1, H, W)
        u2 = self.conv_up2(u2)  # shape: (B, H1, H, W)

        return self.out_conv(u2).squeeze(1)

    def count_parameters(self):
        total_params = 0
        for _, param in self.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
        return total_params
