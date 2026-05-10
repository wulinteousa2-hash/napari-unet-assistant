from __future__ import annotations

import torch

from ..registry import ModelSpec
from ..unet import UNet2D, UNet3D


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("builtin", "unet", "Built-in plain U-Net"),
        ModelSpec("builtin", "resunet", "Built-in residual U-Net"),
    ]


def build_model(
    *,
    mode: str,
    model_name: str,
    in_channels: int,
    out_channels: int,
    model_base: int | None,
    model_params: dict,
) -> torch.nn.Module:
    if model_name == "resunet":
        raise NotImplementedError(
            "Built-in residual U-Net is listed as a planned model family. "
            "Use MONAI SegResNet now, or add a built-in ResUNet provider implementation."
        )
    if model_name not in {"unet", "unet2d", "unet3d"}:
        raise ValueError(f"Built-in backend does not provide model: {model_name}")
    if mode == "2d":
        return UNet2D(in_channels, out_channels, base=int(model_base or 32))
    if mode == "3d":
        return UNet3D(in_channels, out_channels, base=int(model_base or 16))
    raise ValueError("mode must be '2d' or '3d'")
