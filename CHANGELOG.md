# Changelog

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
