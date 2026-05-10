from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ModelSpec:
    backend: str
    name: str
    label: str
    supports_2d: bool = True
    supports_3d: bool = True
    requires: str = ""
    encoders: tuple[str, ...] = ()
    default_encoder: str = ""
    encoder_weights: tuple[str, ...] = ()


def _providers():
    from .providers import builtin_unet, monai_models, nnunet_adapter, segmentation_models_pytorch

    return {
        "builtin": builtin_unet,
        "monai": monai_models,
        "nnunet": nnunet_adapter,
        "smp": segmentation_models_pytorch,
    }


def available_model_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for provider in _providers().values():
        specs.extend(provider.model_specs())
    return specs


def model_specs_for_backend(backend: str, mode: str | None = None) -> list[ModelSpec]:
    provider = _providers().get(backend)
    if provider is None:
        raise ValueError(f"Unknown model backend: {backend}")

    specs = provider.model_specs()
    if mode == "2d":
        return [s for s in specs if s.supports_2d]
    if mode == "3d":
        return [s for s in specs if s.supports_3d]
    return specs


def default_model_name(backend: str, mode: str) -> str:
    specs = model_specs_for_backend(backend, mode)
    if not specs:
        raise ValueError(f"No {mode.upper()} models are available for backend: {backend}")
    return specs[0].name


def model_spec(backend: str, model_name: str, mode: str | None = None) -> ModelSpec | None:
    for spec in model_specs_for_backend(backend, mode):
        if spec.name == model_name:
            return spec
    return None


def build_model_from_config(cfg: Any) -> torch.nn.Module:
    data = cfg if isinstance(cfg, dict) else cfg.__dict__
    backend = data.get("model_backend") or "builtin"
    provider = _providers().get(backend)
    if provider is None:
        raise ValueError(f"Unknown model backend: {backend}")
    return provider.build_model(
        mode=data["mode_2d_or_3d"],
        model_name=data.get("model_name") or default_model_name(backend, data["mode_2d_or_3d"]),
        in_channels=int(data.get("in_channels", 1)),
        out_channels=int(data["out_channels"]),
        model_base=data.get("model_base"),
        model_params=data.get("model_params") or {},
    )
