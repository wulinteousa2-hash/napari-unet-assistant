from __future__ import annotations

import torch

from ..registry import ModelSpec


def model_specs() -> list[ModelSpec]:
    encoders = ("resnet34", "resnet50", "efficientnet-b0", "densenet121", "mobilenet_v2")
    weights = ("none", "imagenet")
    return [
        ModelSpec("smp", "unet", "SMP U-Net", supports_3d=False, requires="segmentation-models-pytorch", encoders=encoders, default_encoder="resnet34", encoder_weights=weights),
        ModelSpec("smp", "unetplusplus", "SMP U-Net++", supports_3d=False, requires="segmentation-models-pytorch", encoders=encoders, default_encoder="resnet34", encoder_weights=weights),
        ModelSpec("smp", "deeplabv3plus", "SMP DeepLabV3+", supports_3d=False, requires="segmentation-models-pytorch", encoders=encoders, default_encoder="resnet34", encoder_weights=weights),
    ]


def _require_smp():
    try:
        import segmentation_models_pytorch as smp
    except Exception as exc:
        raise ImportError(
            "segmentation_models_pytorch models require: pip install segmentation-models-pytorch"
        ) from exc
    return smp


def build_model(
    *,
    mode: str,
    model_name: str,
    in_channels: int,
    out_channels: int,
    model_base: int | None,
    model_params: dict,
) -> torch.nn.Module:
    if mode != "2d":
        raise ValueError("segmentation_models_pytorch only supports 2D models.")

    smp = _require_smp()
    encoder_name = model_params.get("encoder_name", "resnet34")
    encoder_weights = model_params.get("encoder_weights", None)
    if encoder_weights == "none":
        encoder_weights = None
    kwargs = {
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "in_channels": in_channels,
        "classes": out_channels,
        "activation": None,
    }

    name = model_name.lower()
    if name == "unet":
        return smp.Unet(**kwargs)
    if name == "unetplusplus":
        return smp.UnetPlusPlus(**kwargs)
    if name == "deeplabv3plus":
        return smp.DeepLabV3Plus(**kwargs)
    raise ValueError(f"segmentation_models_pytorch backend does not provide model: {model_name}")
