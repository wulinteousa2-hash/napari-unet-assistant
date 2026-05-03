# napari-unet-assistant

`napari-unet-assistant` is a napari plugin for supervised 2D and 3D U-Net segmentation workflows.

It is designed for users who already have image-mask training data and want to pair files, train a U-Net model, run inference, and inspect results directly inside napari.

This plugin is separate from SAM-based annotation workflows. Its focus is conventional supervised U-Net training from existing image-mask pairs.

## What's new in 0.2.0

- Added smarter image-mask pairing.
- Added support for one-folder mixed datasets.
- Added manual CSV pairing.
- Added `_1` / `_2` pairing support, where `_1` is treated as image and `_2` as mask.
- Added expanded mask-name detection, including `_mask`, `_label`, `_seg`, `_annotation`, `_gt`, and related suffixes.
- Added expanded image-name detection, including `_image`, `_img`, `_raw`, `_input`, and related suffixes.
- Added pair confidence, pairing reason, and shape-check status.
- Added reporting for ambiguous, unmatched, rejected, and shape-mismatched files.

## Highlights

- TIFF-first 2D and 3D U-Net training
- Binary and multiclass segmentation
- Smart image-mask pairing
- Pair review with confidence, reason, and shape-check status
- Patch-based training with configurable patch size and overlap
- Optional empty-mask patch inclusion
- Conservative augmentation
- 80/20 validation split
- Continue training from a previous run
- Single-image and folder inference
- 2D image and 3D volume prediction
- TIFF prediction export
- napari-based visualization and QC

## Smart image-mask pairing

The plugin can pair training data from:

- separate image and mask folders
- one mixed folder containing both images and masks
- a manual CSV file

Supported naming patterns include:

- `sample.tif` + `sample_mask.tif`
- `sample_1.tif` + `sample_2.tif`
- `sample_image.tif` + `sample_mask.tif`
- `sample_raw.tif` + `sample_label.tif`

After scanning, the plugin shows each proposed pair with confidence, reason, and shape-check status. Ambiguous or invalid pairs are reported instead of being silently used for training.

## Manual CSV pairing

For manual pairing, provide a CSV file with one image-mask pair per row.

Required columns:

- `image_path`
- `mask_path`

Optional column:

- `key`

Example:

```csv
key,image_path,mask_path
sample01,/path/to/images/sample01.tif,/path/to/masks/sample01_mask.tif
sample02,/path/to/images/sample02.tif,/path/to/masks/sample02_mask.tif
```

Use absolute paths for the clearest behavior. Relative paths are interpreted from the current working directory.

Each image and mask should have matching spatial dimensions.

## Installation

```bash
pip install git+https://github.com/wulinteousa2-hash/napari-unet-assistant.git
```

For editable development:

```bash
git clone https://github.com/wulinteousa2-hash/napari-unet-assistant.git
cd napari-unet-assistant
pip install -e .
napari
```

## Basic workflow

1. Open napari.
2. Launch **U-Net Assistant**.
3. Choose a pairing mode.
4. Scan and review image-mask pairs.
5. Set training options.
6. Train a 2D or 3D U-Net model.
7. Load a saved run folder.
8. Run inference on new images or volumes.
9. Review prediction masks in napari.

## Supported data

### 2D training

- image: `(Y, X)` grayscale TIFF
- mask: `(Y, X)` label TIFF
- binary masks: `0 = background`, nonzero = foreground
- multiclass masks: integer labels

### 3D training

- image: `(Z, Y, X)` grayscale TIFF
- mask: `(Z, Y, X)` label TIFF
- multiclass masks should use integer labels:
  - `0 = background`
  - `1 = class 1`
  - `2 = class 2`
  - `3 = class 3`

## Patch options

### 2D

XY patch sizes:

- 64
- 128
- 256
- 512
- 1024

Default: `256 x 256`

### 3D

Z patch sizes:

- 8
- 16
- 32
- 64

XY patch sizes:

- 64
- 128
- 256
- 512
- 1024

Default: `16 x 256 x 256`

## Validation

The current training workflow uses a standard train/validation split. The default validation split is 20%.

K-fold controls may exist in the UI/config, but k-fold training is not active in this release.

## Outputs

Each run folder can contain:

- `best_model.pt`
- `config.json`
- `summary.json`
- `history.csv`
- `pairs.csv`
- prediction TIFF outputs

## Current scope

This release is focused on TIFF-based supervised U-Net training and inference.

OME-Zarr, spectral/lambda workflows, and SAM-assisted annotation are intentionally outside the scope of this version.

## Reference

This project builds on U-Net-based nerve morphometry workflows described in:

Moiseev D, Hu B, Li J. Morphometric Analysis of Peripheral Myelinated Nerve Fibers through Deep Learning. *Journal of the Peripheral Nervous System*. 2019;24(1):87-93.  
https://pmc.ncbi.nlm.nih.gov/articles/PMC6420354/
