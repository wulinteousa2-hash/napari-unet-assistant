from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_conv(dim: int):
    if dim == 2:
        return nn.Conv2d
    if dim == 3:
        return nn.Conv3d
    raise ValueError(f"Unsupported dim={dim}")


def _get_conv_transpose(dim: int):
    if dim == 2:
        return nn.ConvTranspose2d
    if dim == 3:
        return nn.ConvTranspose3d
    raise ValueError(f"Unsupported dim={dim}")


def _get_maxpool(dim: int):
    if dim == 2:
        return nn.MaxPool2d
    if dim == 3:
        return nn.MaxPool3d
    raise ValueError(f"Unsupported dim={dim}")


def _get_batchnorm(dim: int):
    if dim == 2:
        return nn.BatchNorm2d
    if dim == 3:
        return nn.BatchNorm3d
    raise ValueError(f"Unsupported dim={dim}")


def _get_dropout(dim: int):
    if dim == 2:
        return nn.Dropout2d
    if dim == 3:
        return nn.Dropout3d
    raise ValueError(f"Unsupported dim={dim}")


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """
    Pick the largest valid group count <= max_groups that divides num_channels.
    Falls back to 1 group if needed.
    """
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


def _make_norm(dim: int, norm: str, num_channels: int) -> nn.Module:
    norm = norm.lower()
    if norm == "batch":
        bn = _get_batchnorm(dim)
        return bn(num_channels)
    if norm == "group":
        return _make_group_norm(num_channels)
    if norm in {"instance", "inst"}:
        if dim == 2:
            return nn.InstanceNorm2d(num_channels, affine=True)
        if dim == 3:
            return nn.InstanceNorm3d(num_channels, affine=True)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm='{norm}'")


def _match_spatial(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Center-crop x spatially to match ref if needed.
    This is a defensive safeguard for odd shapes or future tiled inference paths.
    """
    if x.shape[2:] == ref.shape[2:]:
        return x

    xs = x.shape[2:]
    rs = ref.shape[2:]
    slices = [slice(None), slice(None)]

    for xdim, rdim in zip(xs, rs):
        if xdim == rdim:
            slices.append(slice(None))
        elif xdim > rdim:
            start = (xdim - rdim) // 2
            end = start + rdim
            slices.append(slice(start, end))
        else:
            # If x is smaller, pad symmetrically
            pad_total = rdim - xdim
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            # F.pad expects reversed spatial order
            # We handle this after loop if needed
            slices.append(slice(None))

    # First crop dimensions where x is larger
    x = x[tuple(slices)]

    # Then pad dimensions where x is smaller
    if x.shape[2:] != ref.shape[2:]:
        pads = []
        for xdim, rdim in zip(reversed(x.shape[2:]), reversed(ref.shape[2:])):
            if xdim < rdim:
                total = rdim - xdim
                before = total // 2
                after = total - before
            else:
                before = 0
                after = 0
            pads.extend([before, after])
        x = F.pad(x, pads)

    return x


class DoubleConv(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        norm: str = "group",
        dropout: float = 0.0,
    ):
        super().__init__()
        conv = _get_conv(dim)
        drop = _get_dropout(dim)

        layers = [
            conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(dim, norm, out_channels),
            nn.ReLU(inplace=True),
            conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(dim, norm, out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(drop(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        norm: str = "group",
        dropout: float = 0.0,
    ):
        super().__init__()
        pool = _get_maxpool(dim)
        self.pool = pool(kernel_size=2, stride=2)
        self.conv = DoubleConv(dim, in_channels, out_channels, norm=norm, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str = "group",
        dropout: float = 0.0,
    ):
        super().__init__()
        conv_t = _get_conv_transpose(dim)
        self.up = conv_t(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(
            dim,
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            norm=norm,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = _match_spatial(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetND(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (32, 64, 128, 256),
        norm: str = "group",
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(features) < 2:
            raise ValueError("features must contain at least 2 levels")

        conv = _get_conv(dim)

        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = tuple(features)
        self.norm = norm
        self.dropout = dropout

        self.stem = DoubleConv(dim, in_channels, features[0], norm=norm, dropout=0.0)

        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            drop_i = dropout if i >= 1 else 0.0
            self.downs.append(
                DownBlock(
                    dim=dim,
                    in_channels=features[i],
                    out_channels=features[i + 1],
                    norm=norm,
                    dropout=drop_i,
                )
            )

        bottleneck_channels = features[-1] * 2
        self.bottleneck = DoubleConv(
            dim=dim,
            in_channels=features[-1],
            out_channels=bottleneck_channels,
            norm=norm,
            dropout=dropout,
        )

        self.ups = nn.ModuleList()
        current_channels = bottleneck_channels
        for skip_channels in reversed(features):
            out_ch = skip_channels
            self.ups.append(
                UpBlock(
                    dim=dim,
                    in_channels=current_channels,
                    skip_channels=skip_channels,
                    out_channels=out_ch,
                    norm=norm,
                    dropout=dropout if out_ch >= features[1] else 0.0,
                )
            )
            current_channels = out_ch

        self.head = conv(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        x = self.stem(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(skips[-1])

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.head(x)


class UNet2D(UNetND):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base: int = 32,
        norm: str = "group",
        dropout: float = 0.0,
    ):
        features = (base, base * 2, base * 4, base * 8)
        super().__init__(
            dim=2,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            norm=norm,
            dropout=dropout,
        )


class UNet3D(UNetND):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base: int = 16,
        norm: str = "group",
        dropout: float = 0.0,
    ):
        features = (base, base * 2, base * 4, base * 8)
        super().__init__(
            dim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            norm=norm,
            dropout=dropout,
        )