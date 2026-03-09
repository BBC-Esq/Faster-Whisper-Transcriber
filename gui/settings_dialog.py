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
    QGroupBox,
)

from core.models.metadata import ModelMetadata
from core.audio.device_utils import get_input_devices
from gui.styles import update_button_property
from core.logging_config import get_logger

logger = get_logger(__name__)


class SettingsDialog(QDialog):
    model_update_requested = Signal(str, str, str)
    audio_device_changed = Signal(str, str)
    task_mode_changed = Signal(str)

    def __init__(
        self,
        parent: QWidget | None,
        cuda_available: bool,
        supported_quantizations: dict[str, list[str]],
        current_settings: dict[str, str],
        current_task_mode: str = "transcribe",
        current_audio_device: dict[str, str] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(300)
        self.resize(300, self.sizeHint().height())

        self.cuda_available = cuda_available
        self.supported_quantizations = supported_quantizations
        self.current_settings = dict(current_settings)
        self.current_task_mode = current_task_mode
        self.current_audio_device = current_audio_device or {"name": "", "hostapi": ""}

        self._input_devices = get_input_devices()

        self._build_ui()
        self._setup_connections()
        self._populate_from_settings()
        self._check_for_changes()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        model_group = QGroupBox("Whisper Model")
        model_form = QFormLayout(model_group)
        model_form.setHorizontalSpacing(12)
        model_form.setVerticalSpacing(10)

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(ModelMetadata.get_all_model_names())
        model_form.addRow("Model", self.model_dropdown)

        self.device_dropdown = QComboBox()
        devices = ["cpu", "cuda"] if self.cuda_available else ["cpu"]
        self.device_dropdown.addItems(devices)
        model_form.addRow("Device", self.device_dropdown)

        self.quantization_dropdown = QComboBox()
        model_form.addRow("Precision", self.quantization_dropdown)

        layout.addWidget(model_group)

        task_group = QGroupBox("Task")
        task_form = QFormLayout(task_group)
        task_form.setHorizontalSpacing(12)
        task_form.setVerticalSpacing(10)

        self.task_dropdown = QComboBox()
        task_form.addRow("Mode", self.task_dropdown)

        layout.addWidget(task_group)

        audio_group = QGroupBox("Audio Input")
        audio_form = QFormLayout(audio_group)
        audio_form.setHorizontalSpacing(12)
        audio_form.setVerticalSpacing(10)

        self.audio_device_dropdown = QComboBox()
        self.audio_device_dropdown.addItem("System Default", None)
        for dev in self._input_devices:
            display = f"{dev['name']} ({dev['hostapi']})"
            self.audio_device_dropdown.addItem(display, dev)
        audio_form.addRow("Input Device", self.audio_device_dropdown)

        layout.addWidget(audio_group)

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

        self.update_btn.setFixedHeight(35)
        close_btn.setFixedHeight(35)
        layout.addLayout(button_row)

    def _setup_connections(self) -> None:
        self.model_dropdown.currentTextChanged.connect(self._update_quantization_options)
        self.device_dropdown.currentTextChanged.connect(self._update_quantization_options)
        self.model_dropdown.currentTextChanged.connect(self._update_task_availability)
        self.model_dropdown.currentTextChanged.connect(self._check_for_changes)
        self.device_dropdown.currentTextChanged.connect(self._check_for_changes)
        self.quantization_dropdown.currentTextChanged.connect(self._check_for_changes)
        self.task_dropdown.currentTextChanged.connect(self._check_for_changes)
        self.audio_device_dropdown.currentIndexChanged.connect(self._check_for_changes)

    def _populate_from_settings(self) -> None:
        self.model_dropdown.setCurrentText(self.current_settings.get("model_name", "base.en"))
        self.device_dropdown.setCurrentText(self.current_settings.get("device_type", "cpu"))
        self._update_quantization_options()
        self.quantization_dropdown.setCurrentText(
            self.current_settings.get("quantization_type", "float32")
        )
        self._update_task_availability()
        self.task_dropdown.setCurrentText(self.current_task_mode)
        self._select_audio_device()

    def _select_audio_device(self) -> None:
        saved_name = self.current_audio_device.get("name", "")
        saved_hostapi = self.current_audio_device.get("hostapi", "")

        if not saved_name:
            self.audio_device_dropdown.setCurrentIndex(0)
            return

        for i in range(1, self.audio_device_dropdown.count()):
            data = self.audio_device_dropdown.itemData(i)
            if data and data["name"] == saved_name:
                if saved_hostapi and data["hostapi"] == saved_hostapi:
                    self.audio_device_dropdown.setCurrentIndex(i)
                    return

        for i in range(1, self.audio_device_dropdown.count()):
            data = self.audio_device_dropdown.itemData(i)
            if data and data["name"] == saved_name:
                self.audio_device_dropdown.setCurrentIndex(i)
                return

        self.audio_device_dropdown.setCurrentIndex(0)

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

    def _update_task_availability(self) -> None:
        model = self.model_dropdown.currentText()
        can_translate = ModelMetadata.supports_translation(model)

        self.task_dropdown.blockSignals(True)
        current = self.task_dropdown.currentText()
        self.task_dropdown.clear()
        self.task_dropdown.addItem("transcribe")
        if can_translate:
            self.task_dropdown.addItem("translate")

        if current == "translate" and can_translate:
            self.task_dropdown.setCurrentText("translate")
        else:
            self.task_dropdown.setCurrentText("transcribe")
        self.task_dropdown.blockSignals(False)
        self._check_for_changes()

    def _model_settings_changed(self) -> bool:
        current = {
            "model_name": self.model_dropdown.currentText(),
            "quantization_type": self.quantization_dropdown.currentText(),
            "device_type": self.device_dropdown.currentText(),
        }
        return current != self.current_settings

    def _task_mode_selection_changed(self) -> bool:
        return self.task_dropdown.currentText() != self.current_task_mode

    def _audio_device_selection_changed(self) -> bool:
        data = self.audio_device_dropdown.currentData()
        if data is None:
            return bool(self.current_audio_device.get("name", ""))
        return (
            data["name"] != self.current_audio_device.get("name", "")
            or data["hostapi"] != self.current_audio_device.get("hostapi", "")
        )

    def _check_for_changes(self) -> None:
        model_changed = self._model_settings_changed()
        task_changed = self._task_mode_selection_changed()
        audio_changed = self._audio_device_selection_changed()
        has_changes = model_changed or task_changed or audio_changed
        self.update_btn.setEnabled(has_changes)
        if model_changed:
            self.update_btn.setText("Reload Model")
        else:
            self.update_btn.setText("Update Settings")
        update_button_property(self.update_btn, "changed", has_changes)

    def _on_update_clicked(self) -> None:
        if self._model_settings_changed():
            model = self.model_dropdown.currentText()
            quant = self.quantization_dropdown.currentText()
            device = self.device_dropdown.currentText()
            self.model_update_requested.emit(model, quant, device)

        if self._task_mode_selection_changed():
            self.task_mode_changed.emit(self.task_dropdown.currentText())

        if self._audio_device_selection_changed():
            data = self.audio_device_dropdown.currentData()
            if data is None:
                self.audio_device_changed.emit("", "")
            else:
                self.audio_device_changed.emit(data["name"], data["hostapi"])

        self.accept()
