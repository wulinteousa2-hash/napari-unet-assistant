from __future__ import annotations

import torch

from ..registry import ModelSpec


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("monai", "unet", "MONAI U-Net", requires="monai"),
        ModelSpec("monai", "segresnet", "MONAI SegResNet", requires="monai"),
        ModelSpec(
            "monai",
            "swinunetr",
            "MONAI SwinUNETR",
            supports_2d=False,
            requires="monai",
            encoders=("swin_transformer",),
            default_encoder="swin_transformer",
        ),
    ]


def _require_monai():
    try:
        from monai.networks import nets
    except Exception as exc:
        raise ImportError(
            "MONAI models require MONAI. Install with: pip install monai"
        ) from exc
    return nets


def _channels(mode: str, model_base: int | None, model_params: dict) -> tuple[int, ...]:
    if "channels" in model_params:
        return tuple(int(v) for v in model_params["channels"])
    base = int(model_base or (32 if mode == "2d" else 16))
    return (base, base * 2, base * 4, base * 8)


def build_model(
    *,
    mode: str,
    model_name: str,
    in_channels: int,
    out_channels: int,
    model_base: int | None,
    model_params: dict,
) -> torch.nn.Module:
    nets = _require_monai()
    spatial_dims = 2 if mode == "2d" else 3
    name = model_name.lower()
    channels = _channels(mode, model_base, model_params)

    if name == "unet":
        return nets.UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=tuple(2 for _ in range(len(channels) - 1)),
            num_res_units=int(model_params.get("num_res_units", 2)),
        )

    if name == "segresnet":
        return nets.SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=int(model_base or (32 if mode == "2d" else 16)),
            blocks_down=tuple(model_params.get("blocks_down", (1, 2, 2, 4))),
            blocks_up=tuple(model_params.get("blocks_up", (1, 1, 1))),
        )

    if name == "swinunetr":
        if mode != "3d":
            raise ValueError("MONAI SwinUNETR is only enabled for 3D training in this plugin.")
        patch_z = int(model_params.get("patch_z", 16))
        patch_xy = int(model_params.get("patch_xy", 128))
        kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "feature_size": int(model_params.get("feature_size", 48)),
        }
        try:
            return nets.SwinUNETR(img_size=(patch_z, patch_xy, patch_xy), **kwargs)
        except TypeError:
            return nets.SwinUNETR(**kwargs)

    raise ValueError(f"MONAI backend does not provide model: {model_name}")
