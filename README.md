# napari-unet-assistant

`napari-unet-assistant` is a napari plugin for TIFF-based 2D and 3D U-Net segmentation workflows.

It is designed for users who already have paired images and masks and want to train a U-Net model, run inference, and inspect results directly inside napari.

This plugin is different from a SAM-based training assistant. It focuses on conventional supervised U-Net training from existing image-mask pairs.

## Current features

- TIFF-first image and mask pairing
- 2D U-Net training
- 3D U-Net training
- Binary segmentation
- Multiclass segmentation
- Patch-based training
- Optional empty-mask patch inclusion
- Conservative augmentation
- 80/20 validation split
- Continue training from a previous run
- Inference from a saved run folder
- Single-image inference
- Folder inference
- 2D and 3D prediction support
- TIFF prediction output
- napari-based visualization and QC

## Installation
```BASH
pip install git+https://github.com/wulinteousa2-hash/napari-unet-assistant.git
```
OR
```BASH
git clone https://github.com/wulinteousa2-hash/napari-unet-assistant.git
cd napari-unet-assistant
pip install -e .
napari
```

## Intended workflow

1. Prepare an image folder.
2. Prepare a mask folder.
3. Pair images and masks by filename.
4. Train a 2D or 3D U-Net model.
5. Save the best model checkpoint and training metadata.
6. Load a saved run folder.
7. Run inference on new images or volumes.
8. Review prediction masks in napari.

## Supported data

#### 2D training
- image: `(Y, X)` grayscale
- mask: `(Y, X)` grayscale labels
- `0 = background` , `nonzero = foreground`

#### 3D training
- image: `(Z, Y, X)` grayscale
- mask: `(Z, Y, X)` integer labels
- Multiclass masks should use integer labels, 
for example:

- 0 = background
- 1 = class 1
- 2 = class 2
- 3 = class 3 

#### Dtypes observed
- images: `uint8`, `uint16`
- masks: `uint8`

### Current priorities
- TIFF first
- OME-Zarr later
- no spectral/lambda workflow in this phase

### Pairing logic
- image folder and mask folder are selected separately
- mask suffixes supported:
  - `_mask.tif`
  - `_mask.tiff`
  - `_m.tif`
  - `_m.tiff`
- unmatched items are skipped and reported

### Training modes
- 2D binary default
- 3D supports:
  - binary
  - multiclass

### Patch extraction
#### 2D XY patch options
- 64
- 128
- 256
- 512
- 1024

Default:
- 256 x 256

#### 3D patch options
Z:
- 8
- 16
- 32
- 64

XY:
- 64
- 128
- 256
- 512
- 1024

Default:
- 16 x 256 x 256

### Overlap
- overlap is percent
- `0` = no overlap
- `10` = 10% overlap

### Empty-mask patch policy
- user can choose whether to include empty-mask patches

### Backend
- PyTorch only
- CPU fallback required
- CUDA expected
- DGX Spark target

### Validation
- default: 80/20 split
- optional: k-fold

### Checkpoint policy
- save best model only

### Inference
- load from saved run folder
- read JSON config from training
- TIFF output only

### Outputs saved in user-selected run folder
- best model checkpoint
- config JSON
- metrics CSV
- training history CSV
- pair report CSV
- prediction TIFF outputs

## Reference

This project builds on U-Net-based nerve morphometry workflows described in:

Moiseev D, Hu B, Li J. Morphometric Analysis of Peripheral Myelinated Nerve Fibers through Deep Learning. *Journal of the Peripheral Nervous System*. 2019;24(1):87-93. https://pmc.ncbi.nlm.nih.gov/articles/PMC6420354/
