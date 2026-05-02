from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".tif", ".tiff"}
MASK_SUFFIXES = ("_mask", "_m")


@dataclass
class PairRecord:
    key: str
    image_path: str
    mask_path: str


@dataclass
class PairingReport:
    pairs: list[PairRecord]
    unmatched_images: list[str]
    unmatched_masks: list[str]


def _is_tiff(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def _strip_mask_suffix(stem: str) -> str:
    for suf in MASK_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _looks_like_mask(stem: str) -> bool:
    return any(stem.endswith(suf) for suf in MASK_SUFFIXES)


def pair_image_mask_folders(image_dir: str | Path, mask_dir: str | Path) -> PairingReport:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    if not image_dir.is_dir():
        raise ValueError(f"Image folder not found: {image_dir}")
    if not mask_dir.is_dir():
        raise ValueError(f"Mask folder not found: {mask_dir}")

    image_files = sorted([p for p in image_dir.iterdir() if p.is_file() and _is_tiff(p)])
    mask_files = sorted([p for p in mask_dir.iterdir() if p.is_file() and _is_tiff(p)])

    image_map: dict[str, Path] = {}
    for p in image_files:
        if _looks_like_mask(p.stem):
            continue
        image_map[p.stem] = p

    mask_map: dict[str, Path] = {}
    for p in mask_files:
        key = _strip_mask_suffix(p.stem)
        mask_map[key] = p

    pairs: list[PairRecord] = []
    unmatched_images: list[str] = []
    unmatched_masks: list[str] = []

    image_keys = set(image_map.keys())
    mask_keys = set(mask_map.keys())

    common = sorted(image_keys & mask_keys)
    for key in common:
        pairs.append(PairRecord(key=key, image_path=str(image_map[key]), mask_path=str(mask_map[key])))

    for key in sorted(image_keys - mask_keys):
        unmatched_images.append(str(image_map[key]))

    for key in sorted(mask_keys - image_keys):
        unmatched_masks.append(str(mask_map[key]))

    return PairingReport(
        pairs=pairs,
        unmatched_images=unmatched_images,
        unmatched_masks=unmatched_masks,
    )