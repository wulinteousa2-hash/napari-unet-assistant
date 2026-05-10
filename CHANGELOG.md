# Changelog

## 0.5.0 - Guided model selection and run summaries

### Added

- Added recursive dataset-folder auto pairing for common `images`/`masks`, `raw`/`labels`, mixed-folder, and nested TIFF dataset layouts.
- Added folder-name role hints so folders named like `images`, `raw`, `masks`, or `labels` can guide pairing even when filenames are less explicit.
- Added automatic selection of the best pairing strategy from recursive scan, two-subfolder scan, and mixed-folder scan results.
- Added configurable augmentation presets: `none`, `conservative`, `balanced`, and `strong`.
- Added custom augmentation controls for horizontal and vertical flips, rotation, shear, scale, brightness jitter, and Gaussian noise.
- Added model-capacity options for standard, large, and xlarge U-Net widths.
- Added a model registry with provider modules for built-in U-Net, MONAI, nnU-Net, and segmentation-models-pytorch backends.
- Added separate model-family, backbone/encoder, and encoder-weight controls for comparing U-Net variants more clearly.
- Added optional install extras for `monai`, `nnunet`, `smp`, and `models`.
- Added saving and loading of augmentation settings in run configuration metadata.
- Added a training stop button that requests cancellation, discards the interrupted model state, and clears GPU cache when available.
- Added a U-Net architecture preview for the selected 2D/3D mode and output-channel configuration.

### Changed

- Changed auto pairing to focus on scanning one dataset root recursively instead of mixing dataset-folder and two-folder inputs in the same mode.
- Improved one-folder pairing by using the recursive dataset auto-pairing logic.
- Improved role-suffix stripping so repeated trailing role tokens can be removed from filenames before matching.
- Reworked the training panel into clearer tabs and more stable Qt sizing so controls fit better in the dock widget.
- Replaced the fixed conservative augmentation path with configuration-driven augmentation used by both 2D and 3D datasets.
- Updated inference loading so checkpoints are restored with the saved U-Net base-channel width.
- Updated training and inference to rebuild models through the registry from saved backend/model metadata.
- Clarified validation behavior by keeping split validation as the active mode and writing `validation.json` for each run.
- Added training-start model/configuration logging and a human-readable `run_summary.txt` in each run folder.
- Added short tooltips across pairing, training, validation, augmentation, architecture, inference, and results controls.
- Improved training startup and shutdown handling by releasing old model memory before a new run and after cancellation or errors.

### Notes

- This version remains focused on TIFF-based supervised U-Net training and inference.
- OME-Zarr, spectral/lambda workflows, and SAM-assisted annotation remain outside the scope of this version.

## 0.2.0 - Smart pairing update

### Added

- Added smarter image-mask pairing for supervised U-Net training datasets.
- Added auto pairing mode.
- Added support for one-folder mixed datasets containing both images and masks.
- Added manual CSV pairing with `image_path` and `mask_path` columns.
- Added optional `key` column support for manual CSV pairing.
- Added `_1` / `_2` pairing support, where `_1` is treated as image and `_2` as mask.
- Added expanded mask-name detection, including `_mask`, `_m`, `_label`, `_labels`, `_seg`, `_annotation`, `_gt`, `_groundtruth`, `_binary`, and related terms.
- Added expanded image-name detection, including `_image`, `_img`, `_raw`, `_input`, `_source`, `_ch0`, and related terms.
- Added pair confidence scoring.
- Added pairing reason reporting.
- Added image/mask shape quick-check during pair scanning.
- Added reporting for ambiguous, unmatched, rejected, and shape-mismatched files.
- Added saved pairing reports, including `pairs.csv`, `unmatched_images.csv`, `unmatched_masks.csv`, and `pairing_warnings.csv`.

### Changed

- Improved dataset pairing from simple suffix matching to a reviewable smart-pairing workflow.
- Updated README to clarify supported pairing modes, CSV format, and validation status.

### Notes

- K-fold controls may exist in the UI/config, but k-fold training is not active in this release.
- This version remains focused on TIFF-based supervised U-Net training and inference.

## 0.1.0 - Initial public release

### Added

- Initial `napari-unet-assistant` plugin release.
- Added napari dock widget for U-Net segmentation workflows.
- Added TIFF-based image and mask folder pairing.
- Added support for paired 2D image-mask training.
- Added support for paired 3D image-mask training.
- Added 2D U-Net model.
- Added 3D U-Net model.
- Added binary segmentation training.
- Added multiclass segmentation training.
- Added patch-based training for 2D and 3D datasets.
- Added configurable XY patch size.
- Added configurable Z patch size for 3D training.
- Added configurable patch overlap.
- Added optional empty-mask patch inclusion.
- Added conservative augmentation.
- Added validation split training workflow.
- Added Dice, IoU, and F1 metrics.
- Added best-checkpoint saving based on validation Dice score.
- Added saved run folder structure with model weights, config, summary, history, and pair reports.
- Added continue-training workflow from a previous run folder.
- Added compatibility checks for resumed training.
- Added inference from saved run folders.
- Added single-image inference.
- Added folder-based inference.
- Added single-volume 3D inference.
- Added folder-based 3D inference.
- Added TIFF prediction export.
- Added optional loading of predictions into napari.

### Notes

- This version focused on TIFF-first supervised U-Net training and inference.
- The public plugin/project name became `napari-unet-assistant`.
- SAM-assisted annotation and persistent SAM-to-U-Net project workflows were intentionally outside this release.
