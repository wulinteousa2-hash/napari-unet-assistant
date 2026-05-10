from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


@dataclass
class AugmentationConfig:
    enabled: bool = True
    preset: str = "conservative"
    flip_horizontal: bool = True
    flip_vertical: bool = True
    rotate: bool = True
    rotate_degrees: float = 15.0
    shear: bool = False
    shear_degrees: float = 0.0
    scale: bool = True
    scale_min: float = 0.9
    scale_max: float = 1.1
    brightness: bool = True
    brightness_min: float = 0.9
    brightness_max: float = 1.1
    gaussian_noise: bool = False
    noise_min: float = 0.0
    noise_max: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None):
        if data is None:
            return cls()
        values = asdict(cls())
        values.update({k: v for k, v in data.items() if k in values})
        return cls(**values)

    @classmethod
    def preset_config(cls, preset: str):
        preset = preset.lower()
        if preset == "none":
            return cls(enabled=False, preset="none")
        if preset == "balanced":
            return cls(
                preset="balanced",
                rotate_degrees=25.0,
                shear=True,
                shear_degrees=5.0,
                scale_min=0.85,
                scale_max=1.15,
                brightness_min=0.85,
                brightness_max=1.2,
                gaussian_noise=True,
                noise_min=0.0,
                noise_max=0.015,
            )
        if preset == "strong":
            return cls(
                preset="strong",
                rotate_degrees=45.0,
                shear=True,
                shear_degrees=10.0,
                scale_min=0.75,
                scale_max=1.25,
                brightness_min=0.7,
                brightness_max=1.35,
                gaussian_noise=True,
                noise_min=0.005,
                noise_max=0.03,
            )
        return cls(preset="conservative")


def random_flip_2d(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator):
    if rng.random() < 0.5:
        img = np.flip(img, axis=-2)
        msk = np.flip(msk, axis=-2)
    if rng.random() < 0.5:
        img = np.flip(img, axis=-1)
        msk = np.flip(msk, axis=-1)
    return img.copy(), msk.copy()


def random_rot90_2d(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator):
    k = int(rng.integers(0, 4))
    img = np.rot90(img, k=k, axes=(-2, -1))
    msk = np.rot90(msk, k=k, axes=(-2, -1))
    return img.copy(), msk.copy()


def random_intensity_jitter(img: np.ndarray, rng: np.random.Generator, scale_range=(0.9, 1.1), bias_range=(-0.05, 0.05)):
    s = rng.uniform(*scale_range)
    b = rng.uniform(*bias_range)
    out = img * s + b
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def augment_2d_conservative(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator):
    img, msk = random_flip_2d(img, msk, rng)
    img, msk = random_rot90_2d(img, msk, rng)
    img = random_intensity_jitter(img, rng)
    return img, msk


def augment_3d_conservative(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator):
    # Conservative: only Y/X flips plus mild intensity jitter.
    if rng.random() < 0.5:
        img = np.flip(img, axis=-2)
        msk = np.flip(msk, axis=-2)
    if rng.random() < 0.5:
        img = np.flip(img, axis=-1)
        msk = np.flip(msk, axis=-1)
    img = random_intensity_jitter(img, rng)
    return img.copy(), msk.copy()


def _affine_yx(
    img: np.ndarray,
    msk: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentationConfig,
):
    angle = 0.0
    if cfg.rotate and cfg.rotate_degrees > 0:
        angle = float(rng.uniform(-cfg.rotate_degrees, cfg.rotate_degrees))

    shear = [0.0, 0.0]
    if cfg.shear and cfg.shear_degrees > 0:
        shear = [
            float(rng.uniform(-cfg.shear_degrees, cfg.shear_degrees)),
            float(rng.uniform(-cfg.shear_degrees, cfg.shear_degrees)),
        ]

    scale = 1.0
    if cfg.scale:
        scale_min = min(float(cfg.scale_min), float(cfg.scale_max))
        scale_max = max(float(cfg.scale_min), float(cfg.scale_max))
        if scale_min > 0 and scale_max > 0:
            scale = float(rng.uniform(scale_min, scale_max))

    if angle == 0.0 and shear == [0.0, 0.0] and scale == 1.0:
        return img, msk

    img_t = torch.from_numpy(img.astype(np.float32, copy=False))
    msk_t = torch.from_numpy(msk.astype(np.float32, copy=False))

    squeeze_2d = img_t.ndim == 2
    if squeeze_2d:
        img_t = img_t[None, ...]
        msk_t = msk_t[None, ...]

    img_t = img_t[:, None, ...]
    msk_t = msk_t[:, None, ...]

    img_t = TF.affine(
        img_t,
        angle=angle,
        translate=[0, 0],
        scale=scale,
        shear=shear,
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )
    msk_t = TF.affine(
        msk_t,
        angle=angle,
        translate=[0, 0],
        scale=scale,
        shear=shear,
        interpolation=InterpolationMode.NEAREST,
        fill=0.0,
    )

    img_out = img_t[:, 0, ...].numpy()
    msk_out = msk_t[:, 0, ...].numpy()
    if squeeze_2d:
        img_out = img_out[0]
        msk_out = msk_out[0]
    return img_out.astype(np.float32, copy=False), msk_out.astype(msk.dtype, copy=False)


def _apply_configured_augmentation(
    img: np.ndarray,
    msk: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentationConfig,
):
    if not cfg.enabled:
        return img, msk

    if cfg.flip_vertical and rng.random() < 0.5:
        img = np.flip(img, axis=-2)
        msk = np.flip(msk, axis=-2)
    if cfg.flip_horizontal and rng.random() < 0.5:
        img = np.flip(img, axis=-1)
        msk = np.flip(msk, axis=-1)

    img, msk = _affine_yx(img.copy(), msk.copy(), rng, cfg)

    if cfg.brightness:
        img = random_intensity_jitter(
            img,
            rng,
            scale_range=(float(cfg.brightness_min), float(cfg.brightness_max)),
            bias_range=(-0.03, 0.03),
        )

    if cfg.gaussian_noise:
        noise_min = min(float(cfg.noise_min), float(cfg.noise_max))
        noise_max = max(float(cfg.noise_min), float(cfg.noise_max))
        sigma = float(rng.uniform(max(0.0, noise_min), max(0.0, noise_max)))
        img = np.clip(img + rng.normal(0.0, sigma, size=img.shape), 0.0, 1.0)

    return img.astype(np.float32, copy=False), msk.copy()


def augment_2d(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator, cfg: AugmentationConfig):
    return _apply_configured_augmentation(img, msk, rng, cfg)


def augment_3d(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator, cfg: AugmentationConfig):
    return _apply_configured_augmentation(img, msk, rng, cfg)
