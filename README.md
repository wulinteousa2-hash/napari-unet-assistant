# napari-unet-assistant

`napari-unet-assistant` is a napari plugin for supervised 2D and 3D U-Net segmentation workflows.

It is designed for users who already have image-mask training data and want to pair files, train a U-Net model, run inference, and inspect results directly inside napari.

This plugin is separate from SAM-based annotation workflows. Its focus is conventional supervised U-Net training from existing image-mask pairs.

## What's new in 0.4.0

- Added recursive dataset-folder auto pairing for TIFF datasets with `images`/`masks`, `raw`/`labels`, mixed-folder, and nested layouts.
- Added folder-name role hints so dataset folders can guide image-mask pairing even when filenames are less explicit.
- Added configurable augmentation presets: `none`, `conservative`, `balanced`, and `strong`.
- Added custom augmentation controls for flips, rotation, shear, scale, brightness jitter, and Gaussian noise.
- Added model-capacity options for standard, large, and xlarge U-Net widths.
- Added a model registry with built-in U-Net, MONAI, nnU-Net, and segmentation-models-pytorch backend hooks.
- Added separate model-family, backbone/encoder, and encoder-weight controls for clearer U-Net variant testing.
- Added saving and loading of augmentation settings in run configuration metadata.
- Added a training stop button that cancels after the current batch, discards the interrupted model state, and clears GPU cache when available.
- Added a U-Net architecture preview for the selected 2D/3D mode and output-channel configuration.
- Improved the training UI layout with clearer tabs and more stable dock-widget sizing.

## Highlights

- TIFF-first 2D and 3D U-Net training
- Binary and multiclass segmentation
- Recursive smart image-mask pairing
- Pair review with confidence, reason, and shape-check status
- Patch-based training with configurable patch size and overlap
- Optional empty-mask patch inclusion
- Configurable augmentation presets and custom augmentation controls
- Standard, large, and xlarge U-Net capacity options
- Optional model backends for MONAI, nnU-Net, and segmentation-models-pytorch
- 80/20 validation split
- Continue training from a previous run
- Training cancellation from the UI
- Single-image and folder inference
- 2D image and 3D volume prediction
- TIFF prediction export
- napari-based visualization and QC

## Smart image-mask pairing

The plugin can pair training data from:

- a dataset root scanned recursively
- separate image and mask folders
- one mixed folder containing both images and masks
- a manual CSV file

Dataset-root auto scan supports common layouts such as:

- `images/` + `masks/`
- `raw/` + `labels/`
- nested TIFF folders under one dataset root
- one mixed folder containing `sample.tif` + `sample_mask.tif`

Supported naming patterns include:

- `sample.tif` + `sample_mask.tif`
- `sample_1.tif` + `sample_2.tif`
- `sample_image.tif` + `sample_mask.tif`
- `sample_raw.tif` + `sample_label.tif`

After scanning, the plugin shows each proposed pair with confidence, reason, and shape-check status. Ambiguous or invalid pairs are reported instead of being silently used for training.

Folder names such as `images`, `raw`, `masks`, and `labels` can also provide role hints during recursive dataset scans.

## Augmentation

Training supports augmentation presets and custom controls:

- `none`: no augmentation
- `conservative`: flips, small rotations/scales, and light brightness jitter
- `balanced`: stronger rotation/scale, shear, brightness jitter, and light Gaussian noise
- `strong`: wider rotation/scale/shear ranges, stronger brightness jitter, and stronger Gaussian noise
- `custom`: user-selected flips, rotation, shear, scale, brightness, and noise settings

The selected augmentation configuration is saved in each run folder's `config.json` and restored when loading run metadata.

## Training controls

The training panel includes a stop button for cancelling an active training run. Cancellation is checked between batches, so the current batch may finish before the run stops.

When a run is stopped, the interrupted model state is discarded and GPU cache is cleared when available.

## Model capacity

Training can use standard, large, or xlarge U-Net widths. For 2D models, these use base channel widths of 32, 64, and 128. Larger models can learn more complex boundaries, but they need more GPU memory and may require a smaller batch size.

## Model backends

The default backend is the built-in U-Net. Optional backends can be installed separately:

```bash
pip install napari-unet-assistant[monai]
pip install napari-unet-assistant[smp]
pip install napari-unet-assistant[nnunet]
pip install napari-unet-assistant[models]
```

The model registry lives under `src/napari_unet_assistant/models/` and separates provider code into `providers/`. MONAI and segmentation-models-pytorch models are regular `torch.nn.Module` backends. nnU-Net is reserved as a pipeline adapter because nnU-Net manages its own data conversion, training, and prediction workflow.

Model selection is split into:

- `Model backend`: implementation source, such as built-in, MONAI, SMP, or nnU-Net
- `Model family`: architecture family, such as U-Net, U-Net++, SegResNet, or DeepLabV3+
- `Backbone / encoder`: feature extractor when the selected family supports one, such as ResNet34, ResNet50, EfficientNet-B0, DenseNet121, or MobileNetV2
- `Encoder weights`: pretrained encoder weights when supported
- `Model capacity`: built-in width preset used by backends that expose width/depth-style capacity

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

K-fold cross-validation is not active in this release.

Each run writes `validation.json` with the active validation mode, split fraction, random seed, total patch count, train patch count, and validation patch count. The training log also reports the same split. Per-epoch validation metrics are written to `history.csv`.

## Outputs

Each run folder can contain:

- `best_model.pt`
- `config.json`
- `summary.json`
- `history.csv`
- `validation.json`
- `pairs.csv`
- prediction TIFF outputs

## Current scope

This release is focused on TIFF-based supervised U-Net training and inference.

OME-Zarr, spectral/lambda workflows, and SAM-assisted annotation are intentionally outside the scope of this version.

## Reference

This project builds on U-Net-based nerve morphometry workflows described in:

Moiseev D, Hu B, Li J. Morphometric Analysis of Peripheral Myelinated Nerve Fibers through Deep Learning. *Journal of the Peripheral Nervous System*. 2019;24(1):87-93.  
https://pmc.ncbi.nlm.nih.gov/articles/PMC6420354/
