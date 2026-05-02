```markdown
# Changelog

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

- This version is focused on TIFF-first supervised U-Net training and inference.
- The public plugin/project name is now `napari-unet-assistant`.
- SAM-assisted annotation and persistent SAM-to-U-Net project workflows are intentionally outside 