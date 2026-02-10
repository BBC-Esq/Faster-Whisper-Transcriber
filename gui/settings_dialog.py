from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QFormLayout,
    QWidget,
)

from core.models.metadata import ModelMetadata
from gui.styles import update_button_property
from core.logging_config import get_logger

logger = get_logger(__name__)


class SettingsDialog(QDialog):
    model_update_requested = Signal(str, str, str)
    reset_dimensions_requested = Signal()

    def __init__(
        self,
        parent: QWidget | None,
        cuda_available: bool,
        supported_quantizations: dict[str, list[str]],
        current_settings: dict[str, str],
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(380)

        self.cuda_available = cuda_available
        self.supported_quantizations = supported_quantizations
        self.current_settings = dict(current_settings)

        self._build_ui()
        self._setup_connections()
        self._populate_from_settings()
        self._check_for_changes()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(ModelMetadata.get_all_model_names())
        form.addRow("Model", self.model_dropdown)

        self.device_dropdown = QComboBox()
        devices = ["cpu", "cuda"] if self.cuda_available else ["cpu"]
        self.device_dropdown.addItems(devices)
        form.addRow("Device", self.device_dropdown)

        self.quantization_dropdown = QComboBox()
        form.addRow("Precision", self.quantization_dropdown)

        layout.addLayout(form)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        self.update_btn = QPushButton("Update Settings")
        self.update_btn.setObjectName("updateButton")
        self.update_btn.setEnabled(False)
        self.update_btn.clicked.connect(self._on_update_clicked)
        button_row.addWidget(self.update_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        button_row.addWidget(close_btn)

        layout.addLayout(button_row)

        reset_row = QHBoxLayout()
        reset_row.setSpacing(10)

        reset_btn = QPushButton("Reset Window Dimensions")
        reset_btn.setToolTip("Reset the main window and clipboard window to their default sizes and positions")
        reset_btn.clicked.connect(self._on_reset_dimensions_clicked)
        reset_row.addWidget(reset_btn)

        reset_row.addStretch(1)

        layout.addLayout(reset_row)

    def _setup_connections(self) -> None:
        self.model_dropdown.currentTextChanged.connect(self._update_quantization_options)
        self.device_dropdown.currentTextChanged.connect(self._update_quantization_options)
        self.model_dropdown.currentTextChanged.connect(self._check_for_changes)
        self.device_dropdown.currentTextChanged.connect(self._check_for_changes)
        self.quantization_dropdown.currentTextChanged.connect(self._check_for_changes)

    def _populate_from_settings(self) -> None:
        self.model_dropdown.setCurrentText(self.current_settings.get("model_name", "base.en"))
        self.device_dropdown.setCurrentText(self.current_settings.get("device_type", "cpu"))
        self._update_quantization_options()
        self.quantization_dropdown.setCurrentText(
            self.current_settings.get("quantization_type", "float32")
        )

    def _update_quantization_options(self) -> None:
        model = self.model_dropdown.currentText()
        device = self.device_dropdown.currentText()
        opts = ModelMetadata.get_quantization_options(model, device, self.supported_quantizations)

        self.quantization_dropdown.blockSignals(True)
        current = self.quantization_dropdown.currentText()
        self.quantization_dropdown.clear()
        self.quantization_dropdown.addItems(opts)
        if current in opts:
            self.quantization_dropdown.setCurrentText(current)
        elif opts:
            self.quantization_dropdown.setCurrentText(opts[0])
        self.quantization_dropdown.blockSignals(False)
        self._check_for_changes()

    def _check_for_changes(self) -> None:
        current = {
            "model_name": self.model_dropdown.currentText(),
            "quantization_type": self.quantization_dropdown.currentText(),
            "device_type": self.device_dropdown.currentText(),
        }
        has_changes = current != self.current_settings
        self.update_btn.setEnabled(has_changes)
        self.update_btn.setText("Reload Model" if has_changes else "Update Settings")
        update_button_property(self.update_btn, "changed", has_changes)

    def _on_update_clicked(self) -> None:
        model = self.model_dropdown.currentText()
        quant = self.quantization_dropdown.currentText()
        device = self.device_dropdown.currentText()
        self.model_update_requested.emit(model, quant, device)
        self.accept()

    def _on_reset_dimensions_clicked(self) -> None:
        self.reset_dimensions_requested.emit()