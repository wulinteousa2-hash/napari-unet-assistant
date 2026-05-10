from __future__ import annotations

import csv
from pathlib import Path

import torch
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QPlainTextEdit, QFileDialog, QLineEdit,
    QComboBox, QSpinBox, QCheckBox, QListWidget, QAbstractItemView,
    QScrollArea, QProgressBar, QToolButton, QTabWidget, QDoubleSpinBox,
    QSizePolicy,
)
import napari
from napari.utils.notifications import show_info, show_warning
from napari.qt.threading import thread_worker

from ..inference.predictor import (
    load_run_metadata,
    predict_single_from_run_folder,
    predict_folder_from_run_folder,
)
from ..io.loaders import ensure_numpy, load_image_any
from ..io.pairing import (
    pair_auto,
    pair_dataset_folder_auto,
    pair_image_mask_folders,
    pair_mixed_folder,
    pair_from_csv,
)
from ..io.writers import ensure_dir, save_csv_rows, save_json, save_tiff
from ..training.augment import AugmentationConfig
from ..training.trainer import TrainConfig
from ..utils.config import RunConfig


_PANEL_MIN_WIDTH = 520
_PANEL_PREFERRED_WIDTH = 620


def _get_viewer(napari_viewer=None):
    if napari_viewer is not None:
        return napari_viewer
    try:
        return napari.current_viewer()
    except Exception:
        return None


def _gpu_status_label() -> QLabel:
    lbl = QLabel()
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        lbl.setText("GPU: CUDA ON")
        lbl.setStyleSheet(
            "QLabel { background-color: #1f7a1f; color: white; padding: 4px; font-weight: bold; }"
        )
    else:
        lbl.setText("GPU: CUDA OFF / CPU")
        lbl.setStyleSheet(
            "QLabel { background-color: #9b1c1c; color: white; padding: 4px; font-weight: bold; }"
        )
    return lbl


def _double_spin(value: float, minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setDecimals(3 if step < 0.01 else 2)
    spin.setValue(value)
    return spin


def _make_width_neutral(widget: QWidget, vertical_policy=QSizePolicy.Fixed):
    widget.setMinimumWidth(0)
    widget.setSizePolicy(QSizePolicy.Ignored, vertical_policy)
    return widget


def _stabilize_combo(combo: QComboBox):
    combo.setMinimumContentsLength(8)
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    _make_width_neutral(combo)


def _stabilize_line_edit(line_edit: QLineEdit):
    line_edit.setMinimumWidth(0)
    line_edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)


def _stabilize_plain_text(edit: QPlainTextEdit):
    edit.setLineWrapMode(QPlainTextEdit.WidgetWidth)
    edit.setMinimumWidth(0)
    edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)


def _stabilize_list(list_widget: QListWidget):
    list_widget.setMinimumWidth(0)
    list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    list_widget.setTextElideMode(Qt.ElideMiddle)
    list_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)


def _stabilize_label(label: QLabel):
    label.setMinimumWidth(0)
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)


def _stabilize_progress(progress: QProgressBar):
    progress.setMinimumWidth(0)
    progress.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)


def _stabilize_form(form: QFormLayout):
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    form.setLabelAlignment(Qt.AlignLeft)


def _stabilize_row(row: QWidget):
    row.setMinimumWidth(0)
    row.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)


def _mirror_text_as_tooltip(line_edit: QLineEdit):
    line_edit.textChanged.connect(line_edit.setToolTip)


def _unet_architecture_summary(mode: str, in_channels: int, out_channels: int) -> str:
    base = 32 if mode == "2d" else 16
    conv = "Conv2d" if mode == "2d" else "Conv3d"
    pool = "MaxPool2d" if mode == "2d" else "MaxPool3d"
    up = "ConvTranspose2d" if mode == "2d" else "ConvTranspose3d"
    spatial = "Y, X" if mode == "2d" else "Z, Y, X"

    rows = [
        f"Input: {in_channels} channel(s), patch axes {spatial}",
        "",
        f"enc1: {conv} block {in_channels} -> {base}",
        f"pool1: {pool} x2",
        f"enc2: {conv} block {base} -> {base * 2}",
        f"pool2: {pool} x2",
        f"enc3: {conv} block {base * 2} -> {base * 4}",
        f"pool3: {pool} x2",
        f"bottleneck: {conv} block {base * 4} -> {base * 8}",
        "",
        f"up3: {up} {base * 8} -> {base * 4}, concatenate enc3",
        f"dec3: {conv} block {base * 8} -> {base * 4}",
        f"up2: {up} {base * 4} -> {base * 2}, concatenate enc2",
        f"dec2: {conv} block {base * 4} -> {base * 2}",
        f"up1: {up} {base * 2} -> {base}, concatenate enc1",
        f"dec1: {conv} block {base * 2} -> {base}",
        "",
        f"head: 1x1 {conv} {base} -> {out_channels}",
    ]
    return "\n".join(rows)


class _StepSection(QWidget):
    _ACCENT = ("#6b5131", "#d6a75f")

    def __init__(self, step: int, title: str, hint: str = "", parent=None):
        super().__init__(parent)
        self._step = step
        self._title = title
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        badge_color, accent_color = self._ACCENT

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(4)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self._toggle = QToolButton()
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setText("-")
        self._toggle.setFixedWidth(24)
        self._toggle.setToolTip(f"Collapse or expand step {step}")
        self._toggle.setStyleSheet(
            "QToolButton { "
            f"border-color: {accent_color}; "
            f"color: {accent_color}; "
            "}"
        )

        step_badge = QLabel(f"Step {step}")
        step_badge.setObjectName("stepBadge")
        step_badge.setStyleSheet(
            "QLabel#stepBadge { "
            f"background-color: {badge_color}; "
            f"border-color: {accent_color}; "
            "}"
        )

        title_label = QLabel(title)
        title_label.setObjectName("stepTitle")
        _stabilize_label(title_label)

        header_layout.addWidget(self._toggle)
        header_layout.addWidget(step_badge)
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        self.body = QGroupBox(hint)
        self.body.setObjectName("stepBody")
        self.body.setMinimumWidth(0)
        self.body.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.body.setStyleSheet(
            "QGroupBox#stepBody { "
            f"border-color: {badge_color}; "
            f"border-left-color: {accent_color}; "
            "}"
        )

        root_layout.addWidget(header)
        root_layout.addWidget(self.body)

        self._toggle.toggled.connect(self._set_expanded)

    def set_content_layout(self, layout):
        self.body.setLayout(layout)

    def _set_expanded(self, expanded: bool):
        self.body.setVisible(expanded)
        self._toggle.setText("-" if expanded else "+")


def _apply_step_style(widget: QWidget):
    widget.setStyleSheet("""
        QWidget {
            color: #d8dee9;
            font-size: 13px;
        }
        QScrollArea {
            border: 0;
        }
        QLabel#stepBadge {
            background-color: #3f5872;
            border: 1px solid #6d8cac;
            border-radius: 8px;
            color: #f1f5f9;
            font-weight: 700;
            padding: 2px 14px;
        }
        QLabel#stepTitle {
            color: #f3f4f6;
            font-weight: 700;
            padding-left: 4px;
        }
        QToolButton {
            background-color: #0f1b2a;
            border: 1px solid #7da5ca;
            border-radius: 8px;
            color: #f1f5f9;
            font-weight: 700;
            padding: 1px 6px;
        }
        QGroupBox#stepBody {
            border: 1px solid #6b5131;
            border-left: 4px solid #6d8cac;
            border-radius: 8px;
            margin-top: 10px;
            padding: 12px 10px 10px 10px;
        }
        QGroupBox#stepBody::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 18px;
            padding: 1px 10px;
            background-color: #111827;
            border-radius: 4px;
            color: #b7c9de;
        }
        QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QListWidget {
            background-color: #0f1723;
            border: 1px solid #4a3d2d;
            border-radius: 6px;
            color: #e5e7eb;
            padding: 5px;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus, QListWidget:focus {
            border-color: #d6a75f;
        }
        QPushButton {
            background-color: #2f2a24;
            border: 1px solid #6b5131;
            border-radius: 6px;
            color: #f3f4f6;
            font-weight: 700;
            padding: 6px 10px;
        }
        QPushButton:hover {
            background-color: #3b3025;
            border-color: #d6a75f;
        }
        QPushButton:pressed {
            background-color: #6b5131;
        }
        QPushButton:disabled {
            color: #8b949e;
            background-color: #20242a;
            border-color: #343a43;
        }
        QProgressBar {
            background-color: #0f1723;
            border: 1px solid #4a3d2d;
            border-radius: 6px;
            color: #e5e7eb;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #d6a75f;
            border-radius: 5px;
        }
    """)


def _read_pairs_csv(csv_path: Path) -> list[dict]:
    rows = []
    if not csv_path.exists():
        return rows

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "image_path" in row and "mask_path" in row:
                rows.append({
                    "key": row.get("key", ""),
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                })
    return rows


def _deduplicate_pairs(pairs: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for p in pairs:
        key = (p["image_path"], p["mask_path"])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def toolbox_widget(napari_viewer=None) -> QWidget:
    viewer = _get_viewer(napari_viewer)

    root = QWidget()
    root.setMinimumWidth(_PANEL_MIN_WIDTH)
    root.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    root.resize(_PANEL_PREFERRED_WIDTH, root.height())
    _apply_step_style(root)
    root_layout = QVBoxLayout(root)

    if viewer is None:
        root_layout.addWidget(QLabel("No active napari viewer found."))
        return root

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    root_layout.addWidget(scroll)

    container = QWidget()
    container.setMinimumWidth(_PANEL_MIN_WIDTH)
    container.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    layout = QVBoxLayout(container)
    scroll.setWidget(container)

    state = {
        "pair_report": None,
        "model": None,
        "history": None,
        "run_dir": None,
        "run_cfg": None,
        "train_worker": None,
        "stop_training_requested": False,
        "infer_worker": None,
        "is_training": False,
        "is_inference": False,
        "loaded_infer_cfg": None,
        "loaded_resume_cfg": None,
    }

    # -------------------------
    # GPU STATUS
    # -------------------------
    layout.addWidget(_gpu_status_label())

    # -------------------------
    # DATASET PAIRING
    # -------------------------
    g_data = _StepSection(
        1,
        "Dataset pairing",
        "Choose matching image and mask folders, scan pairs, then load examples for review.",
    )
    data_layout = QFormLayout()
    _stabilize_form(data_layout)

    pairing_mode_combo = QComboBox()
    pairing_mode_combo.addItems([
        "Auto scan dataset folder",
        "Two folders: images + masks",
        "One folder: mixed images/masks",
        "Manual CSV",
    ])

    dataset_dir_edit = QLineEdit()
    image_dir_edit = QLineEdit()
    mask_dir_edit = QLineEdit()
    csv_pairing_edit = QLineEdit()
    output_dir_edit = QLineEdit()
    for line_edit in [
        dataset_dir_edit, image_dir_edit, mask_dir_edit,
        csv_pairing_edit, output_dir_edit,
    ]:
        _stabilize_line_edit(line_edit)
        _mirror_text_as_tooltip(line_edit)

    btn_browse_dataset_dir = QPushButton("Browse dataset folder")
    btn_browse_image_dir = QPushButton("Browse image folder")
    btn_browse_mask_dir = QPushButton("Browse mask folder")
    btn_browse_csv_pairing = QPushButton("Browse CSV")
    btn_browse_output_dir = QPushButton("Browse output folder")
    btn_scan_pairs = QPushButton("Scan pairs")
    btn_load_selected_pair = QPushButton("Load selected pair(s) into napari")

    validate_shapes_chk = QCheckBox("Quick-check image/mask shapes during scan")
    validate_shapes_chk.setChecked(True)

    pair_summary_label = QLabel("No pair scan yet.")
    pair_summary_label.setStyleSheet(
        "QLabel { background-color: #273344; color: white; padding: 4px; font-weight: bold; }"
    )
    _stabilize_label(pair_summary_label)

    pair_list = QListWidget()
    pair_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    _stabilize_list(pair_list)

    row_dataset = QWidget()
    _stabilize_row(row_dataset)
    row_dataset_l = QHBoxLayout(row_dataset)
    row_dataset_l.setContentsMargins(0, 0, 0, 0)
    row_dataset_l.addWidget(dataset_dir_edit, 1)
    row_dataset_l.addWidget(btn_browse_dataset_dir)

    row_img = QWidget()
    _stabilize_row(row_img)
    row_img_l = QHBoxLayout(row_img)
    row_img_l.setContentsMargins(0, 0, 0, 0)
    row_img_l.addWidget(image_dir_edit, 1)
    row_img_l.addWidget(btn_browse_image_dir)

    row_msk = QWidget()
    _stabilize_row(row_msk)
    row_msk_l = QHBoxLayout(row_msk)
    row_msk_l.setContentsMargins(0, 0, 0, 0)
    row_msk_l.addWidget(mask_dir_edit, 1)
    row_msk_l.addWidget(btn_browse_mask_dir)

    row_csv = QWidget()
    _stabilize_row(row_csv)
    row_csv_l = QHBoxLayout(row_csv)
    row_csv_l.setContentsMargins(0, 0, 0, 0)
    row_csv_l.addWidget(csv_pairing_edit, 1)
    row_csv_l.addWidget(btn_browse_csv_pairing)

    row_out = QWidget()
    _stabilize_row(row_out)
    row_out_l = QHBoxLayout(row_out)
    row_out_l.setContentsMargins(0, 0, 0, 0)
    row_out_l.addWidget(output_dir_edit, 1)
    row_out_l.addWidget(btn_browse_output_dir)

    auto_pairing_hint = QLabel(
        "Choose the dataset root. Auto scan searches TIFFs recursively and detects common "
        "images/masks, raw/labels, or mixed filename pairs."
    )
    _stabilize_label(auto_pairing_hint)

    data_layout.addRow("Pairing mode:", pairing_mode_combo)
    data_layout.addRow(auto_pairing_hint)
    data_layout.addRow("Dataset folder:", row_dataset)
    data_layout.addRow("Images:", row_img)
    data_layout.addRow("Masks:", row_msk)
    data_layout.addRow("Pair CSV:", row_csv)
    data_layout.addRow("Run output:", row_out)
    data_layout.addRow(validate_shapes_chk)
    data_layout.addRow(btn_scan_pairs)
    data_layout.addRow("Summary:", pair_summary_label)
    data_layout.addRow("Paired files quick-check:", pair_list)
    data_layout.addRow(btn_load_selected_pair)
    g_data.set_content_layout(data_layout)

    layout.addWidget(g_data)

    # -------------------------
    # TRAINING OPTIONS
    # -------------------------
    g_train = _StepSection(
        2,
        "Training options",
        "Set model type, patching, validation, and whether this is a new or continued run.",
    )
    train_layout = QVBoxLayout()

    training_mode_combo = QComboBox()
    training_mode_combo.addItems(["new training", "continue training"])

    resume_run_edit = QLineEdit()
    _stabilize_line_edit(resume_run_edit)
    _mirror_text_as_tooltip(resume_run_edit)
    btn_browse_resume_run = QPushButton("Browse previous run")
    btn_load_resume_meta = QPushButton("Load resume metadata")

    row_resume = QWidget()
    _stabilize_row(row_resume)
    row_resume_l = QHBoxLayout(row_resume)
    row_resume_l.setContentsMargins(0, 0, 0, 0)
    row_resume_l.addWidget(resume_run_edit, 1)
    row_resume_l.addWidget(btn_browse_resume_run)
    row_resume_l.addWidget(btn_load_resume_meta)

    resume_summary_box = QPlainTextEdit()
    resume_summary_box.setReadOnly(True)
    resume_summary_box.setPlaceholderText("Previous run metadata will appear here...")
    resume_summary_box.setMaximumHeight(110)
    _stabilize_plain_text(resume_summary_box)

    resume_data_policy_combo = QComboBox()
    resume_data_policy_combo.addItems([
        "use new data only",
        "merge previous + new data",
    ])

    mode_combo = QComboBox()
    mode_combo.addItems(["2d", "3d"])

    task_combo = QComboBox()
    task_combo.addItems(["binary", "multiclass"])

    num_classes_spin = QSpinBox()
    num_classes_spin.setRange(1, 255)
    num_classes_spin.setValue(5)

    model_combo = QComboBox()
    model_combo.addItems(["unet2d", "unet3d"])

    patch_xy_combo = QComboBox()
    patch_xy_combo.addItems(["64", "128", "256", "512", "1024"])
    patch_xy_combo.setCurrentText("256")

    patch_z_combo = QComboBox()
    patch_z_combo.addItems(["8", "16", "32", "64"])
    patch_z_combo.setCurrentText("16")

    overlap_spin = QSpinBox()
    overlap_spin.setRange(0, 90)
    overlap_spin.setValue(0)

    include_empty_chk = QCheckBox("Include empty-mask patches")
    include_empty_chk.setChecked(False)

    augment_enable_chk = QCheckBox("Enable data augmentation")
    augment_enable_chk.setChecked(True)

    augment_preset_combo = QComboBox()
    augment_preset_combo.addItems(["conservative", "balanced", "strong", "custom", "none"])

    augment_flip_h_chk = QCheckBox("Flip horizontally")
    augment_flip_h_chk.setChecked(True)
    augment_flip_v_chk = QCheckBox("Flip vertically")
    augment_flip_v_chk.setChecked(True)
    augment_rotate_chk = QCheckBox("Rotate")
    augment_rotate_chk.setChecked(True)
    augment_rotate_spin = _double_spin(15.0, 0.0, 90.0, 1.0)
    augment_shear_chk = QCheckBox("Shear")
    augment_shear_spin = _double_spin(0.0, 0.0, 45.0, 1.0)
    augment_scale_chk = QCheckBox("Scale")
    augment_scale_chk.setChecked(True)
    augment_scale_min_spin = _double_spin(0.90, 0.10, 2.00, 0.05)
    augment_scale_max_spin = _double_spin(1.10, 0.10, 2.00, 0.05)
    augment_brightness_chk = QCheckBox("Brightness")
    augment_brightness_chk.setChecked(True)
    augment_brightness_min_spin = _double_spin(0.90, 0.10, 3.00, 0.05)
    augment_brightness_max_spin = _double_spin(1.10, 0.10, 3.00, 0.05)
    augment_noise_chk = QCheckBox("Gaussian noise")
    augment_noise_min_spin = _double_spin(0.000, 0.000, 0.500, 0.005)
    augment_noise_max_spin = _double_spin(0.010, 0.000, 0.500, 0.005)
    btn_reset_augmentation = QPushButton("Reset augmentation preset")

    epochs_spin = QSpinBox()
    epochs_spin.setRange(1, 10000)
    epochs_spin.setValue(5)

    batch_spin = QSpinBox()
    batch_spin.setRange(1, 256)
    batch_spin.setValue(2)

    val_mode_combo = QComboBox()
    val_mode_combo.addItems(["split", "kfold"])

    val_split_combo = QComboBox()
    val_split_combo.addItems(["0.1", "0.2", "0.25", "0.3"])
    val_split_combo.setCurrentText("0.2")

    for combo in [
        pairing_mode_combo, training_mode_combo, resume_data_policy_combo,
        mode_combo, task_combo, model_combo, patch_xy_combo, patch_z_combo,
        augment_preset_combo, val_mode_combo, val_split_combo,
    ]:
        _stabilize_combo(combo)

    kfold_spin = QSpinBox()
    kfold_spin.setRange(2, 10)
    kfold_spin.setValue(5)

    btn_start_train = QPushButton("Start training")
    btn_stop_train = QPushButton("Stop training")
    btn_stop_train.setEnabled(False)

    train_status_label = QLabel("Idle")
    train_status_label.setStyleSheet(
        "QLabel { background-color: #444; color: white; padding: 4px; font-weight: bold; }"
    )
    _stabilize_label(train_status_label)

    train_progress = QProgressBar()
    train_progress.setMinimum(0)
    train_progress.setMaximum(100)
    train_progress.setValue(0)
    train_progress.setFormat("Idle")
    _stabilize_progress(train_progress)

    train_tabs = QTabWidget()
    train_tabs.setMinimumWidth(0)
    train_tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    run_tab = QWidget()
    run_form = QFormLayout(run_tab)
    _stabilize_form(run_form)
    run_form.addRow("Run type:", training_mode_combo)
    run_form.addRow("Resume run folder:", row_resume)
    run_form.addRow("Resume summary:", resume_summary_box)
    run_form.addRow("Resume data policy:", resume_data_policy_combo)
    run_form.addRow("Data shape:", mode_combo)
    run_form.addRow("Segmentation task:", task_combo)
    run_form.addRow("Num classes (incl. background):", num_classes_spin)
    run_form.addRow("Model:", model_combo)
    run_form.addRow("Epochs:", epochs_spin)
    run_form.addRow("Batch size:", batch_spin)

    patch_tab = QWidget()
    patch_form = QFormLayout(patch_tab)
    _stabilize_form(patch_form)
    patch_form.addRow("Patch XY:", patch_xy_combo)
    patch_form.addRow("Patch Z (3D):", patch_z_combo)
    patch_form.addRow("Overlap %:", overlap_spin)
    patch_form.addRow(include_empty_chk)

    validation_tab = QWidget()
    validation_form = QFormLayout(validation_tab)
    _stabilize_form(validation_form)
    validation_form.addRow("Validation mode:", val_mode_combo)
    validation_form.addRow("Validation split:", val_split_combo)
    validation_form.addRow("K folds:", kfold_spin)

    augmentation_tab = QWidget()
    augmentation_form = QFormLayout(augmentation_tab)
    _stabilize_form(augmentation_form)
    scale_row = QWidget()
    _stabilize_row(scale_row)
    scale_row_l = QHBoxLayout(scale_row)
    scale_row_l.setContentsMargins(0, 0, 0, 0)
    scale_row_l.addWidget(QLabel("min"))
    scale_row_l.addWidget(augment_scale_min_spin)
    scale_row_l.addWidget(QLabel("max"))
    scale_row_l.addWidget(augment_scale_max_spin)
    brightness_row = QWidget()
    _stabilize_row(brightness_row)
    brightness_row_l = QHBoxLayout(brightness_row)
    brightness_row_l.setContentsMargins(0, 0, 0, 0)
    brightness_row_l.addWidget(QLabel("min"))
    brightness_row_l.addWidget(augment_brightness_min_spin)
    brightness_row_l.addWidget(QLabel("max"))
    brightness_row_l.addWidget(augment_brightness_max_spin)
    noise_row = QWidget()
    _stabilize_row(noise_row)
    noise_row_l = QHBoxLayout(noise_row)
    noise_row_l.setContentsMargins(0, 0, 0, 0)
    noise_row_l.addWidget(QLabel("min"))
    noise_row_l.addWidget(augment_noise_min_spin)
    noise_row_l.addWidget(QLabel("max"))
    noise_row_l.addWidget(augment_noise_max_spin)

    augmentation_form.addRow(augment_enable_chk)
    augmentation_form.addRow("Preset:", augment_preset_combo)
    augmentation_form.addRow(augment_flip_h_chk)
    augmentation_form.addRow(augment_flip_v_chk)
    augmentation_form.addRow(augment_rotate_chk, augment_rotate_spin)
    augmentation_form.addRow(augment_shear_chk, augment_shear_spin)
    augmentation_form.addRow(augment_scale_chk, scale_row)
    augmentation_form.addRow(augment_brightness_chk, brightness_row)
    augmentation_form.addRow(augment_noise_chk, noise_row)
    augmentation_form.addRow(btn_reset_augmentation)

    architecture_tab = QWidget()
    architecture_form = QFormLayout(architecture_tab)
    _stabilize_form(architecture_form)
    architecture_summary_box = QPlainTextEdit()
    architecture_summary_box.setReadOnly(True)
    architecture_summary_box.setMinimumHeight(260)
    _stabilize_plain_text(architecture_summary_box)
    btn_refresh_architecture = QPushButton("Refresh architecture preview")
    architecture_form.addRow("Current U-Net:", architecture_summary_box)
    architecture_form.addRow(btn_refresh_architecture)

    train_tabs.addTab(run_tab, "Run")
    train_tabs.addTab(patch_tab, "Patching")
    train_tabs.addTab(validation_tab, "Validation")
    train_tabs.addTab(augmentation_tab, "Augmentation")
    train_tabs.addTab(architecture_tab, "Architecture")

    train_layout.addWidget(train_tabs)
    train_layout.addWidget(QLabel("Status:"))
    train_layout.addWidget(train_status_label)
    train_layout.addWidget(QLabel("Progress:"))
    train_layout.addWidget(train_progress)
    train_buttons = QWidget()
    _stabilize_row(train_buttons)
    train_buttons_l = QHBoxLayout(train_buttons)
    train_buttons_l.setContentsMargins(0, 0, 0, 0)
    train_buttons_l.addWidget(btn_start_train)
    train_buttons_l.addWidget(btn_stop_train)
    train_layout.addWidget(train_buttons)
    g_train.set_content_layout(train_layout)

    layout.addWidget(g_train)

    # -------------------------
    # INFERENCE
    # -------------------------
    g_inf = _StepSection(
        3,
        "Inference",
        "Load a saved training run, choose new data, and create prediction masks.",
    )
    inf_layout = QFormLayout()
    _stabilize_form(inf_layout)

    run_dir_edit = QLineEdit()
    _stabilize_line_edit(run_dir_edit)
    _mirror_text_as_tooltip(run_dir_edit)
    btn_browse_run_dir = QPushButton("Browse run folder")
    btn_load_run_meta = QPushButton("Load run metadata")

    row_run = QWidget()
    _stabilize_row(row_run)
    row_run_l = QHBoxLayout(row_run)
    row_run_l.setContentsMargins(0, 0, 0, 0)
    row_run_l.addWidget(run_dir_edit, 1)
    row_run_l.addWidget(btn_browse_run_dir)
    row_run_l.addWidget(btn_load_run_meta)

    model_summary_box = QPlainTextEdit()
    model_summary_box.setReadOnly(True)
    model_summary_box.setPlaceholderText("Run metadata will appear here...")
    model_summary_box.setMaximumHeight(110)
    _stabilize_plain_text(model_summary_box)

    infer_mode_combo = QComboBox()
    infer_mode_combo.addItems([
        "single image",
        "folder of images",
        "single 3D volume",
        "folder of 3D volumes",
    ])

    infer_input_label = QLabel("Image file:")
    _stabilize_label(infer_input_label)
    infer_input_edit = QLineEdit()
    _stabilize_line_edit(infer_input_edit)
    _mirror_text_as_tooltip(infer_input_edit)
    btn_browse_infer_input = QPushButton("Browse")

    row_input = QWidget()
    _stabilize_row(row_input)
    row_input_l = QHBoxLayout(row_input)
    row_input_l.setContentsMargins(0, 0, 0, 0)
    row_input_l.addWidget(infer_input_edit, 1)
    row_input_l.addWidget(btn_browse_infer_input)

    infer_strategy_combo = QComboBox()
    infer_strategy_combo.addItems(["auto", "full", "tiled"])

    infer_output_edit = QLineEdit()
    _stabilize_line_edit(infer_output_edit)
    _mirror_text_as_tooltip(infer_output_edit)
    btn_browse_infer_output = QPushButton("Browse output folder")

    row_output = QWidget()
    _stabilize_row(row_output)
    row_output_l = QHBoxLayout(row_output)
    row_output_l.setContentsMargins(0, 0, 0, 0)
    row_output_l.addWidget(infer_output_edit, 1)
    row_output_l.addWidget(btn_browse_infer_output)

    infer_load_chk = QCheckBox("Also load prediction into napari")
    infer_load_chk.setChecked(True)

    infer_overwrite_chk = QCheckBox("Overwrite existing prediction TIFF")
    infer_overwrite_chk.setChecked(False)

    infer_status_label = QLabel("Idle")
    infer_status_label.setStyleSheet(
        "QLabel { background-color: #444; color: white; padding: 4px; font-weight: bold; }"
    )
    _stabilize_label(infer_status_label)

    infer_progress = QProgressBar()
    infer_progress.setMinimum(0)
    infer_progress.setMaximum(100)
    infer_progress.setValue(0)
    infer_progress.setFormat("Idle")
    _stabilize_progress(infer_progress)

    btn_run_infer = QPushButton("Run inference")

    infer_log_box = QPlainTextEdit()
    infer_log_box.setReadOnly(True)
    infer_log_box.setPlaceholderText("Inference logs appear here...")
    infer_log_box.setMaximumHeight(140)
    _stabilize_plain_text(infer_log_box)

    for combo in [infer_mode_combo, infer_strategy_combo]:
        _stabilize_combo(combo)

    inf_layout.addRow("Run folder:", row_run)
    inf_layout.addRow("Model summary:", model_summary_box)
    inf_layout.addRow("Inference mode:", infer_mode_combo)
    inf_layout.addRow(infer_input_label, row_input)
    inf_layout.addRow("Strategy:", infer_strategy_combo)
    inf_layout.addRow("Output folder:", row_output)
    inf_layout.addRow(infer_load_chk)
    inf_layout.addRow(infer_overwrite_chk)
    inf_layout.addRow("Status:", infer_status_label)
    inf_layout.addRow("Progress:", infer_progress)
    inf_layout.addRow(btn_run_infer)
    inf_layout.addRow("Log:", infer_log_box)
    g_inf.set_content_layout(inf_layout)

    layout.addWidget(g_inf)

    # -------------------------
    # RESULTS
    # -------------------------
    g_res = _StepSection(
        4,
        "Results",
        "Review pair reports, training metrics, output paths, and run messages.",
    )
    res_layout = QVBoxLayout()
    results_box = QPlainTextEdit()
    results_box.setReadOnly(True)
    results_box.setPlaceholderText("Training logs, pair reports, and global results appear here...")
    _stabilize_plain_text(results_box)
    res_layout.addWidget(results_box)
    g_res.set_content_layout(res_layout)
    layout.addWidget(g_res)

    layout.addStretch(1)

    def log(txt: str):
        results_box.appendPlainText(txt)
        sb = results_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def infer_log(txt: str):
        infer_log_box.appendPlainText(txt)
        sb = infer_log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _release_training_memory(clear_model: bool = False):
        if clear_model:
            state["model"] = None
            state["history"] = None
        state["train_worker"] = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def set_training_ui(running: bool):
        state["is_training"] = running
        btn_start_train.setEnabled(not running)
        btn_stop_train.setEnabled(running)

        if running:
            btn_start_train.setText("Training...")
            btn_stop_train.setText("Stop training")
            train_status_label.setText("Running")
            train_status_label.setStyleSheet(
                "QLabel { background-color: #b36b00; color: white; padding: 4px; font-weight: bold; }"
            )
            train_progress.setValue(0)
            train_progress.setFormat("Starting...")
        else:
            btn_start_train.setText("Start training")
            btn_stop_train.setText("Stop training")
            train_status_label.setText("Idle")
            train_status_label.setStyleSheet(
                "QLabel { background-color: #444; color: white; padding: 4px; font-weight: bold; }"
            )
            train_progress.setValue(0)
            train_progress.setFormat("Idle")

    def set_infer_ui(running: bool):
        state["is_inference"] = running
        btn_run_infer.setEnabled(not running)

        if running:
            btn_run_infer.setText("Running...")
            infer_status_label.setText("Running")
            infer_status_label.setStyleSheet(
                "QLabel { background-color: #b36b00; color: white; padding: 4px; font-weight: bold; }"
            )
            infer_progress.setValue(0)
            infer_progress.setFormat("Starting...")
        else:
            btn_run_infer.setText("Run inference")
            infer_status_label.setText("Idle")
            infer_status_label.setStyleSheet(
                "QLabel { background-color: #444; color: white; padding: 4px; font-weight: bold; }"
            )
            infer_progress.setValue(0)
            infer_progress.setFormat("Idle")

    def browse_dir(line_edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(root, "Select folder")
        if d:
            line_edit.setText(d)

    def browse_tiff_file(line_edit: QLineEdit):
        f, _ = QFileDialog.getOpenFileName(root, "Select file", "", "TIFF (*.tif *.tiff)")
        if f:
            line_edit.setText(f)

    def _sync_mode_ui():
        mode = mode_combo.currentText()
        if mode == "2d":
            model_combo.setCurrentText("unet2d")
            patch_z_combo.setEnabled(False)
        else:
            model_combo.setCurrentText("unet3d")
            patch_z_combo.setEnabled(True)

    def _sync_task_ui():
        task = task_combo.currentText()
        if task == "binary":
            num_classes_spin.setEnabled(False)
            num_classes_spin.setValue(1)
        else:
            num_classes_spin.setEnabled(True)
            if num_classes_spin.value() < 2:
                num_classes_spin.setValue(2)

    def _sync_infer_ui():
        mode = infer_mode_combo.currentText()

        if mode == "single image":
            infer_input_label.setText("Image file:")
            infer_load_chk.setChecked(True)
            btn_browse_infer_input.setText("Browse file")
        elif mode == "folder of images":
            infer_input_label.setText("Image folder:")
            infer_load_chk.setChecked(False)
            btn_browse_infer_input.setText("Browse folder")
        elif mode == "single 3D volume":
            infer_input_label.setText("Volume file:")
            infer_load_chk.setChecked(True)
            btn_browse_infer_input.setText("Browse file")
        else:
            infer_input_label.setText("Volume folder:")
            infer_load_chk.setChecked(False)
            btn_browse_infer_input.setText("Browse folder")

    def _sync_training_mode_ui():
        is_continue = training_mode_combo.currentText() == "continue training"
        resume_run_edit.setEnabled(is_continue)
        btn_browse_resume_run.setEnabled(is_continue)
        btn_load_resume_meta.setEnabled(is_continue)
        resume_summary_box.setEnabled(is_continue)
        resume_data_policy_combo.setEnabled(is_continue)

    def _set_augmentation_controls(cfg: AugmentationConfig):
        widgets = [
            augment_enable_chk, augment_flip_h_chk, augment_flip_v_chk,
            augment_rotate_chk, augment_rotate_spin, augment_shear_chk,
            augment_shear_spin, augment_scale_chk, augment_scale_min_spin,
            augment_scale_max_spin, augment_brightness_chk,
            augment_brightness_min_spin, augment_brightness_max_spin,
            augment_noise_chk, augment_noise_min_spin, augment_noise_max_spin,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        augment_enable_chk.setChecked(cfg.enabled)
        augment_flip_h_chk.setChecked(cfg.flip_horizontal)
        augment_flip_v_chk.setChecked(cfg.flip_vertical)
        augment_rotate_chk.setChecked(cfg.rotate)
        augment_rotate_spin.setValue(cfg.rotate_degrees)
        augment_shear_chk.setChecked(cfg.shear)
        augment_shear_spin.setValue(cfg.shear_degrees)
        augment_scale_chk.setChecked(cfg.scale)
        augment_scale_min_spin.setValue(cfg.scale_min)
        augment_scale_max_spin.setValue(cfg.scale_max)
        augment_brightness_chk.setChecked(cfg.brightness)
        augment_brightness_min_spin.setValue(cfg.brightness_min)
        augment_brightness_max_spin.setValue(cfg.brightness_max)
        augment_noise_chk.setChecked(cfg.gaussian_noise)
        augment_noise_min_spin.setValue(cfg.noise_min)
        augment_noise_max_spin.setValue(cfg.noise_max)
        for widget in widgets:
            widget.blockSignals(False)

    def _current_augmentation_config() -> AugmentationConfig:
        return AugmentationConfig(
            enabled=bool(augment_enable_chk.isChecked()) and augment_preset_combo.currentText() != "none",
            preset=augment_preset_combo.currentText(),
            flip_horizontal=bool(augment_flip_h_chk.isChecked()),
            flip_vertical=bool(augment_flip_v_chk.isChecked()),
            rotate=bool(augment_rotate_chk.isChecked()),
            rotate_degrees=float(augment_rotate_spin.value()),
            shear=bool(augment_shear_chk.isChecked()),
            shear_degrees=float(augment_shear_spin.value()),
            scale=bool(augment_scale_chk.isChecked()),
            scale_min=float(augment_scale_min_spin.value()),
            scale_max=float(augment_scale_max_spin.value()),
            brightness=bool(augment_brightness_chk.isChecked()),
            brightness_min=float(augment_brightness_min_spin.value()),
            brightness_max=float(augment_brightness_max_spin.value()),
            gaussian_noise=bool(augment_noise_chk.isChecked()),
            noise_min=float(augment_noise_min_spin.value()),
            noise_max=float(augment_noise_max_spin.value()),
        )

    def _apply_augmentation_preset(*_args):
        preset = augment_preset_combo.currentText()
        if preset == "custom":
            return
        _set_augmentation_controls(AugmentationConfig.preset_config(preset))

    def _mark_augmentation_custom(*_args):
        if augment_preset_combo.currentText() != "custom":
            augment_preset_combo.blockSignals(True)
            augment_preset_combo.setCurrentText("custom")
            augment_preset_combo.blockSignals(False)

    def _refresh_architecture_preview(*_args):
        out_channels = 1 if task_combo.currentText() == "binary" else int(num_classes_spin.value())
        architecture_summary_box.setPlainText(
            _unet_architecture_summary(mode_combo.currentText(), 1, out_channels)
        )

    mode_combo.currentTextChanged.connect(_sync_mode_ui)
    task_combo.currentTextChanged.connect(_sync_task_ui)
    infer_mode_combo.currentTextChanged.connect(_sync_infer_ui)
    training_mode_combo.currentTextChanged.connect(_sync_training_mode_ui)
    augment_preset_combo.currentTextChanged.connect(_apply_augmentation_preset)
    btn_reset_augmentation.clicked.connect(_apply_augmentation_preset)
    btn_refresh_architecture.clicked.connect(_refresh_architecture_preview)
    mode_combo.currentTextChanged.connect(_refresh_architecture_preview)
    task_combo.currentTextChanged.connect(_refresh_architecture_preview)
    num_classes_spin.valueChanged.connect(_refresh_architecture_preview)

    for widget in [
        augment_enable_chk, augment_flip_h_chk, augment_flip_v_chk,
        augment_rotate_chk, augment_shear_chk, augment_scale_chk,
        augment_brightness_chk, augment_noise_chk,
    ]:
        widget.toggled.connect(_mark_augmentation_custom)

    for widget in [
        augment_rotate_spin, augment_shear_spin, augment_scale_min_spin,
        augment_scale_max_spin, augment_brightness_min_spin,
        augment_brightness_max_spin, augment_noise_min_spin, augment_noise_max_spin,
    ]:
        widget.valueChanged.connect(_mark_augmentation_custom)

    def _sync_pairing_ui():
        mode = pairing_mode_combo.currentText()
        is_auto = mode == "Auto scan dataset folder"
        is_two = mode == "Two folders: images + masks"
        is_one = mode == "One folder: mixed images/masks"
        is_csv = mode == "Manual CSV"

        dataset_dir_edit.setEnabled(is_auto or is_one)
        btn_browse_dataset_dir.setEnabled(is_auto or is_one)
        row_dataset.setVisible(is_auto or is_one)
        auto_pairing_hint.setVisible(is_auto)

        image_dir_edit.setEnabled(is_two)
        btn_browse_image_dir.setEnabled(is_two)
        row_img.setVisible(is_two)
        mask_dir_edit.setEnabled(is_two)
        btn_browse_mask_dir.setEnabled(is_two)
        row_msk.setVisible(is_two)

        csv_pairing_edit.setEnabled(is_csv)
        btn_browse_csv_pairing.setEnabled(is_csv)
        row_csv.setVisible(is_csv)

    pairing_mode_combo.currentTextChanged.connect(_sync_pairing_ui)

    _sync_mode_ui()
    _sync_task_ui()
    _sync_infer_ui()
    _sync_training_mode_ui()
    _apply_augmentation_preset()
    _refresh_architecture_preview()
    _sync_pairing_ui()

    def _pair_status_icon(status: str) -> str:
        if status == "ok":
            return "OK"
        if status == "ambiguous":
            return "WARN"
        if status == "shape_mismatch":
            return "FAIL"
        return "CHECK"

    def _pair_row_text(rec) -> str:
        img_name = Path(rec.image_path).name
        msk_name = Path(rec.mask_path).name
        shape_txt = ""
        if getattr(rec, "image_shape", "") or getattr(rec, "mask_shape", ""):
            shape_txt = f" | shape {rec.image_shape} -> {rec.mask_shape}"
        return (
            f"{_pair_status_icon(rec.status)} | {rec.key} | "
            f"conf={rec.confidence} | {img_name} -> {msk_name}"
            f"{shape_txt} | {rec.reason}"
        )

    def scan_pairs():
        mode = pairing_mode_combo.currentText()
        dataset_dir = dataset_dir_edit.text().strip()
        image_dir = image_dir_edit.text().strip()
        mask_dir = mask_dir_edit.text().strip()
        csv_path = csv_pairing_edit.text().strip()
        validate_shapes = bool(validate_shapes_chk.isChecked())

        try:
            if mode == "Auto scan dataset folder":
                if not dataset_dir:
                    raise ValueError("Choose a dataset folder first.")
                report = pair_auto(
                    dataset_dir=dataset_dir,
                    validate_shapes=validate_shapes,
                )
            elif mode == "Two folders: images + masks":
                if not image_dir or not mask_dir:
                    raise ValueError("Choose image folder and mask folder first.")
                report = pair_image_mask_folders(
                    image_dir,
                    mask_dir,
                    validate_shapes=validate_shapes,
                )
            elif mode == "One folder: mixed images/masks":
                if not dataset_dir:
                    raise ValueError("Choose a dataset folder first.")
                report = pair_dataset_folder_auto(
                    dataset_dir,
                    validate_shapes=validate_shapes,
                )
            else:
                if not csv_path:
                    raise ValueError("Choose a pair CSV first.")
                report = pair_from_csv(csv_path, validate_shapes=validate_shapes)
        except Exception as e:
            show_warning(str(e))
            log(f"Pair scan failed: {e}")
            return

        state["pair_report"] = report

        pair_list.clear()
        usable = 0
        for rec in report.pairs:
            pair_list.addItem(_pair_row_text(rec))
            if rec.status == "ok":
                usable += 1

        summary = (
            f"Mode: {getattr(report, 'mode_used', mode)} | "
            f"usable OK pairs: {usable} | all pairs: {len(report.pairs)} | "
            f"ambiguous: {len(getattr(report, 'ambiguous', []))} | "
            f"rejected: {len(getattr(report, 'rejected', []))} | "
            f"unmatched images: {len(report.unmatched_images)} | "
            f"unmatched masks: {len(report.unmatched_masks)}"
        )
        pair_summary_label.setText(summary)

        log(summary)
        for x in getattr(report, "ambiguous", [])[:20]:
            log(f"Ambiguous: {x}")
        for x in getattr(report, "rejected", [])[:20]:
            log(f"Rejected: {x}")

        if output_dir_edit.text().strip():
            out_dir = ensure_dir(output_dir_edit.text().strip())
            rows = [[
                p.key, p.image_path, p.mask_path,
                getattr(p, "status", ""), getattr(p, "confidence", ""),
                getattr(p, "reason", ""), getattr(p, "image_shape", ""),
                getattr(p, "mask_shape", ""), getattr(p, "shape_ok", ""),
            ] for p in report.pairs]
            save_csv_rows(
                out_dir / "pairs.csv",
                ["key", "image_path", "mask_path", "status", "confidence", "reason", "image_shape", "mask_shape", "shape_ok"],
                rows,
            )

            rows_img = [[x] for x in report.unmatched_images]
            save_csv_rows(out_dir / "unmatched_images.csv", ["image_path"], rows_img)

            rows_msk = [[x] for x in report.unmatched_masks]
            save_csv_rows(out_dir / "unmatched_masks.csv", ["mask_path"], rows_msk)

            save_csv_rows(
                out_dir / "pairing_warnings.csv",
                ["type", "message"],
                [["ambiguous", x] for x in getattr(report, "ambiguous", [])] +
                [["rejected", x] for x in getattr(report, "rejected", [])],
            )

        show_info("Pair scan complete. Review paired files before training.")

    def load_selected_pair():
        report = state["pair_report"]
        if report is None:
            show_warning("Scan pairs first.")
            return

        items = pair_list.selectedItems()
        if not items:
            show_warning("Select one or more pairs first.")
            return

        loaded = 0
        for item in items:
            row = pair_list.row(item)
            rec = report.pairs[row]

            img = load_image_any(rec.image_path)
            msk = ensure_numpy(load_image_any(rec.mask_path)).astype("int32")

            viewer.add_image(img, name=f"{rec.key}_img")
            viewer.add_labels(msk, name=f"{rec.key}_mask")
            loaded += 1

        log(f"Loaded {loaded} pair(s) into napari.")
        show_info(f"Loaded {loaded} pair(s).")

    def load_resume_meta():
        run_dir = resume_run_edit.text().strip()
        if not run_dir:
            show_warning("Choose a previous run folder first.")
            return

        try:
            cfg, summary = load_run_metadata(run_dir)
            state["loaded_resume_cfg"] = cfg

            # auto-fill training controls from previous run
            mode_combo.setCurrentText(cfg.get("mode_2d_or_3d", "2d"))
            task_combo.setCurrentText(cfg.get("task_type", "binary"))
            model_combo.setCurrentText(cfg.get("model_name", "unet2d"))
            patch_xy_combo.setCurrentText(str(cfg.get("patch_xy", 256)))

            patch_z_val = cfg.get("patch_z", None)
            if patch_z_val is not None:
                patch_z_combo.setCurrentText(str(patch_z_val))

            out_channels = int(cfg.get("out_channels", 1))
            num_classes_spin.setValue(out_channels if out_channels >= 1 else 1)
            aug_cfg = cfg.get("augmentation")
            if isinstance(aug_cfg, dict):
                aug = AugmentationConfig.from_dict(aug_cfg)
                augment_preset_combo.setCurrentText(
                    aug.preset if aug.preset in {"none", "conservative", "balanced", "strong", "custom"} else "custom"
                )
                _set_augmentation_controls(aug)

            lines = [
                f"mode_2d_or_3d: {cfg.get('mode_2d_or_3d')}",
                f"task_type: {cfg.get('task_type')}",
                f"model_name: {cfg.get('model_name')}",
                f"patch_xy: {cfg.get('patch_xy')}",
                f"patch_z: {cfg.get('patch_z')}",
                f"out_channels: {cfg.get('out_channels')}",
                f"augmentation: {cfg.get('augmentation', {}).get('preset', 'not recorded') if isinstance(cfg.get('augmentation'), dict) else 'not recorded'}",
            ]
            if summary:
                lines.extend([
                    "",
                    f"best epoch: {summary.get('epoch')}",
                    f"best val_dice: {summary.get('val_dice')}",
                    f"best val_iou: {summary.get('val_iou')}",
                    f"best val_f1: {summary.get('val_f1')}",
                ])

            resume_summary_box.setPlainText("\n".join(str(x) for x in lines))
            log(f"Loaded resume metadata from: {run_dir}")
            show_info("Resume metadata loaded.")

        except Exception as e:
            show_warning(str(e))
            log(f"Load resume metadata failed: {e}")

    def _build_run_config() -> RunConfig:
        mode = mode_combo.currentText()
        task = task_combo.currentText()
        model_name = model_combo.currentText()
        output_dir = output_dir_edit.text().strip()

        if not output_dir:
            raise ValueError("Choose an output folder first.")

        report = state["pair_report"]
        if report is None or len(report.pairs) == 0:
            raise ValueError("No valid pairs found.")

        if task == "binary":
            out_channels = 1
        else:
            out_channels = int(num_classes_spin.value())

        return RunConfig(
            mode_2d_or_3d=mode,
            task_type=task,
            model_name=model_name,
            in_channels=1,
            out_channels=out_channels,
            patch_xy=int(patch_xy_combo.currentText()),
            patch_z=None if mode == "2d" else int(patch_z_combo.currentText()),
            overlap_percent=int(overlap_spin.value()),
            include_empty_mask=bool(include_empty_chk.isChecked()),
            batch_size=int(batch_spin.value()),
            epochs=int(epochs_spin.value()),
            learning_rate=1e-3,
            val_mode=val_mode_combo.currentText(),
            val_split=float(val_split_combo.currentText()),
            k_folds=int(kfold_spin.value()),
            use_gpu=torch.cuda.is_available(),
            image_dir=image_dir_edit.text().strip(),
            mask_dir=mask_dir_edit.text().strip(),
            output_dir=output_dir,
        )

    def _validate_resume_compatibility(new_cfg: RunConfig, resume_cfg: dict):
        checks = [
            ("mode_2d_or_3d", new_cfg.mode_2d_or_3d, resume_cfg.get("mode_2d_or_3d")),
            ("task_type", new_cfg.task_type, resume_cfg.get("task_type")),
            ("model_name", new_cfg.model_name, resume_cfg.get("model_name")),
            ("out_channels", int(new_cfg.out_channels), int(resume_cfg.get("out_channels", -1))),
        ]

        # same patching setup is the safest first implementation
        checks.append(("patch_xy", int(new_cfg.patch_xy), int(resume_cfg.get("patch_xy", -1))))
        if new_cfg.mode_2d_or_3d == "3d":
            checks.append(("patch_z", int(new_cfg.patch_z), int(resume_cfg.get("patch_z", -1))))

        mismatches = []
        for key, current, previous in checks:
            if current != previous:
                mismatches.append(f"{key}: current={current}, previous={previous}")

        if mismatches:
            raise ValueError(
                "Continue training requires compatible settings.\n" +
                "\n".join(mismatches)
            )

    def _collect_training_pairs():
        report = state["pair_report"]
        if report is None or len(report.pairs) == 0:
            raise ValueError("Scan valid pairs first.")

        current_pairs = [
            {"key": p.key, "image_path": p.image_path, "mask_path": p.mask_path}
            for p in report.pairs
            if getattr(p, "status", "ok") == "ok"
        ]

        if not current_pairs:
            raise ValueError("No usable OK pairs selected. Review the pairing result table first.")

        if training_mode_combo.currentText() == "new training":
            return current_pairs

        resume_cfg = state["loaded_resume_cfg"]
        if resume_cfg is None:
            raise ValueError("Load resume metadata first for continue training.")

        policy = resume_data_policy_combo.currentText()
        if policy == "use new data only":
            return current_pairs

        prev_pairs_path = Path(resume_run_edit.text().strip()) / "pairs.csv"
        prev_pairs = _read_pairs_csv(prev_pairs_path)
        if not prev_pairs:
            raise ValueError(
                f"Could not read previous pairs from: {prev_pairs_path}\n"
                "Merge previous + new data requires previous pairs.csv."
            )

        merged = _deduplicate_pairs(prev_pairs + current_pairs)
        return merged

    def start_training():
        try:
            if state["is_training"]:
                show_warning("Training is already running.")
                return

            state["stop_training_requested"] = False
            _release_training_memory(clear_model=True)

            cfg = _build_run_config()
            if cfg.task_type == "multiclass" and cfg.out_channels < 2:
                raise ValueError("Multiclass requires at least 2 classes including background.")

            is_continue = training_mode_combo.currentText() == "continue training"
            resume_run_dir = resume_run_edit.text().strip() if is_continue else None
            resume_cfg = state["loaded_resume_cfg"] if is_continue else None

            if is_continue:
                if not resume_run_dir:
                    raise ValueError("Choose a previous run folder for continue training.")
                if resume_cfg is None:
                    raise ValueError("Load resume metadata first.")
                _validate_resume_compatibility(cfg, resume_cfg)

            run_dir = ensure_dir(cfg.output_dir)
            state["run_dir"] = run_dir
            state["run_cfg"] = cfg

            training_pairs = _collect_training_pairs()
            image_paths = [p["image_path"] for p in training_pairs]
            mask_paths = [p["mask_path"] for p in training_pairs]
            augmentation_cfg = _current_augmentation_config()

            log(f"Training pairs used: {len(training_pairs)}")

            # save actual training pairs used in this run
            save_csv_rows(
                run_dir / "pairs.csv",
                ["key", "image_path", "mask_path"],
                [[p.get("key", ""), p["image_path"], p["mask_path"]] for p in training_pairs],
            )

            train_cfg = TrainConfig(
                mode_2d_or_3d=cfg.mode_2d_or_3d,
                task_type=cfg.task_type,
                in_channels=cfg.in_channels,
                out_channels=cfg.out_channels,
                batch_size=cfg.batch_size,
                epochs=cfg.epochs,
                lr=cfg.learning_rate,
                val_split=cfg.val_split,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            ds_kwargs = {
                "mode_2d_or_3d": cfg.mode_2d_or_3d,
                "task_type": cfg.task_type,
                "patch_xy": cfg.patch_xy,
                "patch_z": cfg.patch_z,
                "overlap_percent": cfg.overlap_percent,
                "include_empty_mask": cfg.include_empty_mask,
                "augment": bool(augmentation_cfg.enabled),
                "augment_config": augmentation_cfg.to_dict(),
                "cache_arrays": True,
            }

            cfg_dict = cfg.to_dict()
            cfg_dict.update({
                "training_mode": training_mode_combo.currentText(),
                "resume_from_run_dir": resume_run_dir if is_continue else "",
                "resume_data_policy": resume_data_policy_combo.currentText() if is_continue else "",
                "augmentation": ds_kwargs["augment_config"],
            })
            save_json(run_dir / "config.json", cfg_dict)

            set_training_ui(True)
            if is_continue:
                log(f"Starting continue training from: {resume_run_dir}")
            else:
                log("Starting new training...")

            @thread_worker(start_thread=False)
            def _train():
                from ..training.datasets import PatchDataset
                from ..training.trainer import (
                    TrainingCancelled,
                    build_model,
                    build_loss,
                    _run_epoch,
                )
                from torch.utils.data import DataLoader, random_split
                import torch

                def _should_stop():
                    return bool(state.get("stop_training_requested", False))

                if _should_stop():
                    raise TrainingCancelled("Training stopped by user.")

                yield {
                    "kind": "status",
                    "message": "Building PatchDataset..."
                }

                ds = PatchDataset(
                    image_paths=image_paths,
                    mask_paths=mask_paths,
                    **ds_kwargs,
                )

                n_total = len(ds)
                if _should_stop():
                    del ds
                    raise TrainingCancelled("Training stopped by user.")

                yield {
                    "kind": "status",
                    "message": f"PatchDataset ready. Total samples: {n_total}"
                }

                if n_total <= 0:
                    raise ValueError(
                        "PatchDataset contains 0 samples. "
                        "Check image/mask shapes, patch size, overlap, and empty-mask filtering."
                    )

                n_val = max(1, int(round(n_total * train_cfg.val_split)))
                n_train = max(1, n_total - n_val)
                if n_train + n_val > n_total:
                    n_val = n_total - n_train

                yield {
                    "kind": "status",
                    "message": f"Splitting dataset: train={n_train}, val={n_val}"
                }

                train_ds, val_ds = random_split(
                    ds,
                    [n_train, n_val],
                    generator=torch.Generator().manual_seed(42),
                )

                train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)

                yield {
                    "kind": "status",
                    "message": (
                        f"DataLoaders ready. "
                        f"train batches={len(train_loader)}, val batches={len(val_loader)}, "
                        f"batch_size={train_cfg.batch_size}, device={train_cfg.device}"
                    )
                }

                model = build_model(train_cfg).to(train_cfg.device)

                if is_continue:
                    ckpt_path = Path(resume_run_dir) / "best_model.pt"
                    if not ckpt_path.exists():
                        raise ValueError(f"Checkpoint not found: {ckpt_path}")
                    state_dict = torch.load(ckpt_path, map_location=train_cfg.device)
                    model.load_state_dict(state_dict)
                    yield {
                        "kind": "status",
                        "message": f"Loaded pretrained weights from: {ckpt_path}"
                    }

                loss_fn = build_loss(train_cfg)
                optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

                yield {
                    "kind": "status",
                    "message": f"Model ready on {train_cfg.device}. Starting epochs..."
                }

                history = []
                best_val_dice = -1.0
                best_state = None

                for epoch in range(train_cfg.epochs):
                    if _should_stop():
                        raise TrainingCancelled("Training stopped by user.")

                    yield {
                        "kind": "status",
                        "message": f"Epoch {epoch + 1}/{train_cfg.epochs} started..."
                    }

                    train_loss, train_metrics = _run_epoch(
                        model,
                        train_loader,
                        loss_fn,
                        optimizer,
                        train_cfg,
                        train=True,
                        should_stop=_should_stop,
                    )
                    val_loss, val_metrics = _run_epoch(
                        model,
                        val_loader,
                        loss_fn,
                        optimizer,
                        train_cfg,
                        train=False,
                        should_stop=_should_stop,
                    )

                    row = {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_dice": train_metrics["dice"],
                        "val_dice": val_metrics["dice"],
                        "train_iou": train_metrics["iou"],
                        "val_iou": val_metrics["iou"],
                        "train_f1": train_metrics["f1"],
                        "val_f1": val_metrics["f1"],
                    }
                    history.append(row)

                    if val_metrics["dice"] > best_val_dice:
                        best_val_dice = val_metrics["dice"]
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                    yield {
                        "kind": "epoch",
                        "epoch": epoch + 1,
                        "total": train_cfg.epochs,
                        "row": row,
                    }

                if best_state is not None:
                    model.load_state_dict(best_state)

                return model, history

            worker = _train()
            state["train_worker"] = worker

            def _mark_training_stopped():
                train_status_label.setText("Stopped")
                train_status_label.setStyleSheet(
                    "QLabel { background-color: #6b5131; color: white; padding: 4px; font-weight: bold; }"
                )
                train_progress.setFormat("Stopped")
                btn_start_train.setEnabled(True)
                btn_stop_train.setEnabled(False)
                btn_start_train.setText("Start training")
                btn_stop_train.setText("Stop training")
                state["is_training"] = False
                state["stop_training_requested"] = False
                _release_training_memory(clear_model=True)

            def _on_yielded(payload):
                if payload.get("kind") == "status":
                    msg = payload["message"]
                    log(msg)
                    train_progress.setFormat(msg)
                    return

                if payload.get("kind") == "epoch":
                    epoch = payload["epoch"]
                    total = payload["total"]
                    r = payload["row"]

                    pct = int(round(100 * epoch / max(total, 1)))
                    train_progress.setValue(pct)
                    train_progress.setFormat(f"Epoch {epoch}/{total} | val Dice {r['val_dice']:.4f}")

                    log(
                        f"Epoch {epoch}/{total} | "
                        f"train_loss={r['train_loss']:.4f} val_loss={r['val_loss']:.4f} | "
                        f"train_dice={r['train_dice']:.4f} val_dice={r['val_dice']:.4f} | "
                        f"train_iou={r['train_iou']:.4f} val_iou={r['val_iou']:.4f} | "
                        f"train_f1={r['train_f1']:.4f} val_f1={r['val_f1']:.4f}"
                    )

            def _on_returned(result):
                if state.get("stop_training_requested", False):
                    log("Training stopped by user. Model state was discarded and GPU cache was cleared if available.")
                    show_info("Training stopped.")
                    _mark_training_stopped()
                    return

                model, history = result
                state["model"] = model
                state["history"] = history

                torch.save(model.state_dict(), run_dir / "best_model.pt")

                rows = []
                for r in history:
                    rows.append([
                        r["epoch"],
                        r["train_loss"], r["val_loss"],
                        r["train_dice"], r["val_dice"],
                        r["train_iou"], r["val_iou"],
                        r["train_f1"], r["val_f1"],
                    ])

                save_csv_rows(
                    run_dir / "history.csv",
                    ["epoch", "train_loss", "val_loss", "train_dice", "val_dice", "train_iou", "val_iou", "train_f1", "val_f1"],
                    rows,
                )

                best = max(history, key=lambda x: x["val_dice"])
                save_json(run_dir / "summary.json", best)

                train_status_label.setText("Complete")
                train_status_label.setStyleSheet(
                    "QLabel { background-color: #1f7a1f; color: white; padding: 4px; font-weight: bold; }"
                )
                train_progress.setValue(100)
                train_progress.setFormat(f"Done | best val Dice {best['val_dice']:.4f}")

                log(f"Training complete. Best val Dice={best['val_dice']:.4f}")
                show_info("Training complete.")
                btn_start_train.setEnabled(True)
                btn_stop_train.setEnabled(False)
                btn_start_train.setText("Start training")
                state["is_training"] = False
                state["train_worker"] = None
                state["stop_training_requested"] = False

            def _on_error(exc):
                cancelled = exc.__class__.__name__ == "TrainingCancelled" or "stopped by user" in str(exc).lower()
                if cancelled:
                    log("Training stopped by user. Model state was discarded and GPU cache was cleared if available.")
                    show_info("Training stopped.")
                    _mark_training_stopped()
                    return

                train_status_label.setText("Error")
                train_status_label.setStyleSheet(
                    "QLabel { background-color: #9b1c1c; color: white; padding: 4px; font-weight: bold; }"
                )
                train_progress.setFormat("Error")
                log(f"Training error: {exc}")
                show_warning(str(exc))
                btn_start_train.setEnabled(True)
                btn_stop_train.setEnabled(False)
                btn_start_train.setText("Start training")
                state["is_training"] = False
                state["stop_training_requested"] = False
                _release_training_memory(clear_model=True)

            def _on_finished():
                if state.get("stop_training_requested", False) and state.get("is_training", False):
                    log("Training worker stopped. Model state was discarded and GPU cache was cleared if available.")
                    _mark_training_stopped()

            worker.yielded.connect(_on_yielded)
            worker.returned.connect(_on_returned)
            worker.errored.connect(_on_error)
            if hasattr(worker, "finished"):
                worker.finished.connect(_on_finished)
            worker.start()

        except Exception as e:
            set_training_ui(False)
            show_warning(str(e))
            log(f"Start training failed: {e}")

    def stop_training():
        if not state.get("is_training", False):
            return

        state["stop_training_requested"] = True
        btn_stop_train.setEnabled(False)
        btn_stop_train.setText("Stopping...")
        train_status_label.setText("Stopping")
        train_status_label.setStyleSheet(
            "QLabel { background-color: #6b5131; color: white; padding: 4px; font-weight: bold; }"
        )
        train_progress.setFormat("Stopping after current batch...")
        log("Stop requested. Training will stop after the current batch.")

        worker = state.get("train_worker")
        if worker is not None:
            try:
                worker.quit()
            except Exception:
                pass

    def load_run_meta():
        run_dir = run_dir_edit.text().strip()
        if not run_dir:
            show_warning("Choose a run folder first.")
            return

        try:
            cfg, summary = load_run_metadata(run_dir)
            state["loaded_infer_cfg"] = cfg

            lines = [
                f"mode_2d_or_3d: {cfg.get('mode_2d_or_3d')}",
                f"task_type: {cfg.get('task_type')}",
                f"model_name: {cfg.get('model_name')}",
                f"patch_xy: {cfg.get('patch_xy')}",
                f"patch_z: {cfg.get('patch_z')}",
                f"out_channels: {cfg.get('out_channels')}",
            ]
            if summary:
                lines.extend([
                    "",
                    f"best epoch: {summary.get('epoch')}",
                    f"best val_dice: {summary.get('val_dice')}",
                    f"best val_iou: {summary.get('val_iou')}",
                    f"best val_f1: {summary.get('val_f1')}",
                ])

            model_summary_box.setPlainText("\n".join(str(x) for x in lines))

            pred_dir = Path(run_dir) / "predictions"
            infer_output_edit.setText(str(pred_dir))

            show_info("Run metadata loaded.")

        except Exception as e:
            show_warning(str(e))
            infer_log(f"Load metadata failed: {e}")

    def browse_infer_input():
        mode = infer_mode_combo.currentText()
        if "folder" in mode:
            browse_dir(infer_input_edit)
        else:
            browse_tiff_file(infer_input_edit)

    def _validate_infer_mode_with_run(cfg: dict, infer_mode: str):
        run_mode = cfg.get("mode_2d_or_3d")
        if run_mode == "2d" and infer_mode not in {"single image", "folder of images"}:
            raise ValueError("Loaded run folder is a 2D model. Use single image or folder of images.")
        if run_mode == "3d" and infer_mode not in {"single 3D volume", "folder of 3D volumes"}:
            raise ValueError("Loaded run folder is a 3D model. Use single 3D volume or folder of 3D volumes.")

    def run_inference():
        run_dir = run_dir_edit.text().strip()
        infer_input = infer_input_edit.text().strip()
        infer_output = infer_output_edit.text().strip()
        infer_mode = infer_mode_combo.currentText()
        strategy = infer_strategy_combo.currentText()
        overwrite = bool(infer_overwrite_chk.isChecked())
        load_into_napari = bool(infer_load_chk.isChecked())

        if not run_dir:
            show_warning("Choose a run folder.")
            return
        if not infer_input:
            show_warning("Choose an input file or folder.")
            return
        if not infer_output:
            show_warning("Choose an output folder.")
            return

        try:
            cfg, _ = load_run_metadata(run_dir)
            _validate_infer_mode_with_run(cfg, infer_mode)
            output_dir = ensure_dir(infer_output)

            set_infer_ui(True)
            infer_log_box.clear()
            infer_log("Starting inference...")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            if infer_mode in {"single image", "single 3D volume"}:
                @thread_worker(start_thread=False)
                def _infer_single():
                    pred, cfg2, used_strategy = predict_single_from_run_folder(
                        run_dir=run_dir,
                        image_path=infer_input,
                        device=device,
                        strategy=strategy,
                    )
                    return pred, cfg2, used_strategy

                worker = _infer_single()
                state["infer_worker"] = worker

                def _on_single_returned(result):
                    pred, cfg2, used_strategy = result

                    out_path = Path(output_dir) / f"{Path(infer_input).stem}_pred.tif"
                    if out_path.exists() and not overwrite:
                        infer_log(f"Skipped existing output: {out_path}")
                    else:
                        save_tiff(out_path, pred)
                        infer_log(f"Saved prediction TIFF: {out_path}")

                    if load_into_napari:
                        viewer.add_labels(pred.astype("int32"), name=f"{Path(infer_input).stem}_pred")

                    infer_status_label.setText("Complete")
                    infer_status_label.setStyleSheet(
                        "QLabel { background-color: #1f7a1f; color: white; padding: 4px; font-weight: bold; }"
                    )
                    infer_progress.setValue(100)
                    infer_progress.setFormat(f"Done | {used_strategy}")
                    infer_log(f"Inference complete. Strategy used: {used_strategy}")
                    show_info("Inference complete.")
                    btn_run_infer.setEnabled(True)
                    btn_run_infer.setText("Run inference")
                    state["is_inference"] = False

                def _on_single_error(exc):
                    infer_status_label.setText("Error")
                    infer_status_label.setStyleSheet(
                        "QLabel { background-color: #9b1c1c; color: white; padding: 4px; font-weight: bold; }"
                    )
                    infer_progress.setFormat("Error")
                    infer_log(f"Inference error: {exc}")
                    show_warning(str(exc))
                    btn_run_infer.setEnabled(True)
                    btn_run_infer.setText("Run inference")
                    state["is_inference"] = False

                worker.returned.connect(_on_single_returned)
                worker.errored.connect(_on_single_error)
                worker.start()
                return

            @thread_worker(start_thread=False)
            def _infer_folder():
                report, cfg2 = predict_folder_from_run_folder(
                    run_dir=run_dir,
                    input_dir=infer_input,
                    output_dir=output_dir,
                    device=device,
                    strategy=strategy,
                    overwrite=overwrite,
                )
                return report, cfg2

            worker = _infer_folder()
            state["infer_worker"] = worker

            def _on_folder_returned(result):
                report, cfg2 = result

                total = len(report)
                ok = sum(1 for r in report if r["status"] == "ok")
                skipped = sum(1 for r in report if r["status"] == "skipped_exists")
                errors = total - ok - skipped

                for i, r in enumerate(report, start=1):
                    pct = int(round(100 * i / max(total, 1)))
                    infer_progress.setValue(pct)
                    infer_progress.setFormat(f"{i}/{total}")
                    infer_log(f"{r['status']} | {r['input']} -> {r['output']} | {r['strategy']}")

                    if load_into_napari and r["status"] == "ok":
                        try:
                            pred = ensure_numpy(load_image_any(r["output"]))
                            viewer.add_labels(pred.astype("int32"), name=f"{Path(r['output']).stem}")
                        except Exception as e:
                            infer_log(f"Preview load failed: {r['output']} | {e}")

                save_csv_rows(
                    Path(output_dir) / "inference_report.csv",
                    ["input", "output", "status", "strategy"],
                    [[r["input"], r["output"], r["status"], r["strategy"]] for r in report],
                )

                infer_status_label.setText("Complete")
                infer_status_label.setStyleSheet(
                    "QLabel { background-color: #1f7a1f; color: white; padding: 4px; font-weight: bold; }"
                )
                infer_progress.setValue(100)
                infer_progress.setFormat(f"Done | ok={ok} skipped={skipped} errors={errors}")
                infer_log(f"Batch inference complete. ok={ok}, skipped={skipped}, errors={errors}")
                show_info("Batch inference complete.")
                btn_run_infer.setEnabled(True)
                btn_run_infer.setText("Run inference")
                state["is_inference"] = False

            def _on_folder_error(exc):
                infer_status_label.setText("Error")
                infer_status_label.setStyleSheet(
                    "QLabel { background-color: #9b1c1c; color: white; padding: 4px; font-weight: bold; }"
                )
                infer_progress.setFormat("Error")
                infer_log(f"Inference error: {exc}")
                show_warning(str(exc))
                btn_run_infer.setEnabled(True)
                btn_run_infer.setText("Run inference")
                state["is_inference"] = False

            worker.returned.connect(_on_folder_returned)
            worker.errored.connect(_on_folder_error)
            worker.start()

        except Exception as e:
            set_infer_ui(False)
            show_warning(str(e))
            infer_log(f"Start inference failed: {e}")

    btn_browse_dataset_dir.clicked.connect(lambda: browse_dir(dataset_dir_edit))
    btn_browse_image_dir.clicked.connect(lambda: browse_dir(image_dir_edit))
    btn_browse_mask_dir.clicked.connect(lambda: browse_dir(mask_dir_edit))

    def browse_csv_file(line_edit: QLineEdit):
        f, _ = QFileDialog.getOpenFileName(root, "Select pair CSV", "", "CSV (*.csv)")
        if f:
            line_edit.setText(f)

    btn_browse_csv_pairing.clicked.connect(lambda: browse_csv_file(csv_pairing_edit))
    btn_browse_output_dir.clicked.connect(lambda: browse_dir(output_dir_edit))
    btn_scan_pairs.clicked.connect(scan_pairs)
    btn_load_selected_pair.clicked.connect(load_selected_pair)

    btn_browse_resume_run.clicked.connect(lambda: browse_dir(resume_run_edit))
    btn_load_resume_meta.clicked.connect(load_resume_meta)

    btn_start_train.clicked.connect(start_training)
    btn_stop_train.clicked.connect(stop_training)

    btn_browse_run_dir.clicked.connect(lambda: browse_dir(run_dir_edit))
    btn_load_run_meta.clicked.connect(load_run_meta)
    btn_browse_infer_input.clicked.connect(browse_infer_input)
    btn_browse_infer_output.clicked.connect(lambda: browse_dir(infer_output_edit))
    btn_run_infer.clicked.connect(run_inference)

    return root
