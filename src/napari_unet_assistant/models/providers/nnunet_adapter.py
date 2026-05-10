from __future__ import annotations

import torch

from ..registry import ModelSpec


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("nnunet", "nnunetv2", "nnU-Net v2 adapter", requires="nnunetv2"),
    ]


def _require_nnunetv2():
    try:
        import nnunetv2  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "nnU-Net support requires nnunetv2. Install with: pip install nnunetv2"
        ) from exc


def build_model(
    *,
    mode: str,
    model_name: str,
    in_channels: int,
    out_channels: int,
    model_base: int | None,
    model_params: dict,
) -> torch.nn.Module:
    _require_nnunetv2()
    raise NotImplementedError(
        "nnU-Net is a full training/inference pipeline, not only a torch.nn.Module. "
        "This adapter is reserved for a dedicated nnU-Net workflow that exports data "
        "to nnU-Net format, runs nnUNetv2_train/nnUNetv2_predict, and imports results."
    )
