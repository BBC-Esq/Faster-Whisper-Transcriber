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
    QCheckBox,
    QSpinBox,
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
    whisper_settings_changed = Signal(object)

    def __init__(
        self,
        parent: QWidget | None,
        cuda_available: bool,
        supported_quantizations: dict[str, list[str]],
        current_settings: dict[str, str],
        current_task_mode: str = "transcribe",
        current_audio_device: dict[str, str] | None = None,
        current_whisper_settings: dict | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.resize(600, self.sizeHint().height())

        self.cuda_available = cuda_available
        self.supported_quantizations = supported_quantizations
        self.current_settings = dict(current_settings)
        self.current_task_mode = current_task_mode
        self.current_audio_device = current_audio_device or {"name": "", "hostapi": ""}
        self.current_whisper_settings = current_whisper_settings or {
            "without_timestamps": True,
            "word_timestamps": False,
            "beam_size": 5,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }

        self._input_devices = get_input_devices()

        self._build_ui()
        self._setup_connections()
        self._populate_from_settings()
        self._check_for_changes()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(16, 16, 16, 16)
        outer_layout.setSpacing(12)

        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(16)

        left_column = QVBoxLayout()
        left_column.setSpacing(12)

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

        left_column.addWidget(model_group)

        task_group = QGroupBox("Task")
        task_form = QFormLayout(task_group)
        task_form.setHorizontalSpacing(12)
        task_form.setVerticalSpacing(10)

        self.task_dropdown = QComboBox()
        task_form.addRow("Mode", self.task_dropdown)

        left_column.addWidget(task_group)

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

        left_column.addWidget(audio_group)
        left_column.addStretch(1)

        columns_layout.addLayout(left_column, 1)

        right_column = QVBoxLayout()
        right_column.setSpacing(12)

        whisper_group = QGroupBox("Faster Whisper Settings")
        whisper_form = QFormLayout(whisper_group)
        whisper_form.setHorizontalSpacing(12)
        whisper_form.setVerticalSpacing(10)

        self.without_timestamps_cb = QCheckBox()
        self.without_timestamps_cb.setToolTip(
            "Skip timestamp generation for faster output"
        )
        whisper_form.addRow("Without Timestamps", self.without_timestamps_cb)

        self.word_timestamps_cb = QCheckBox()
        self.word_timestamps_cb.setToolTip(
            "Extract start/end time for each individual word"
        )
        whisper_form.addRow("Word Timestamps", self.word_timestamps_cb)

        self.beam_size_spin = QSpinBox()
        self.beam_size_spin.setRange(1, 20)
        self.beam_size_spin.setToolTip(
            "Number of beams for decoding (higher = more accurate, slower)"
        )
        whisper_form.addRow("Beam Size", self.beam_size_spin)

        self.vad_filter_cb = QCheckBox()
        self.vad_filter_cb.setToolTip(
            "Filter out non-speech segments using Silero VAD"
        )
        whisper_form.addRow("VAD Filter", self.vad_filter_cb)

        self.condition_on_previous_cb = QCheckBox()
        self.condition_on_previous_cb.setToolTip(
            "Use previous output as context for the next segment"
        )
        whisper_form.addRow("Condition on Previous", self.condition_on_previous_cb)

        right_column.addWidget(whisper_group)
        right_column.addStretch(1)

        columns_layout.addLayout(right_column, 1)

        outer_layout.addLayout(columns_layout)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        self.update_btn = QPushButton("Update Settings")
        self.update_btn.setObjectName("updateButton")
        self.update_btn.setEnabled(False)
        self.update_btn.clicked.connect(self._on_update_clicked)
        button_row.addWidget(self.update_btn)

        close_btn = QPushButton("Close")
        close_btn.setObjectName("closeButton")
        close_btn.clicked.connect(self.reject)
        button_row.addWidget(close_btn)

        self.update_btn.setFixedHeight(35)
        close_btn.setFixedHeight(35)
        outer_layout.addLayout(button_row)

    def _setup_connections(self) -> None:
        self.model_dropdown.currentTextChanged.connect(self._update_quantization_options)
        self.device_dropdown.currentTextChanged.connect(self._update_quantization_options)
        self.model_dropdown.currentTextChanged.connect(self._update_task_availability)
        self.model_dropdown.currentTextChanged.connect(self._check_for_changes)
        self.device_dropdown.currentTextChanged.connect(self._check_for_changes)
        self.quantization_dropdown.currentTextChanged.connect(self._check_for_changes)
        self.task_dropdown.currentTextChanged.connect(self._check_for_changes)
        self.audio_device_dropdown.currentIndexChanged.connect(self._check_for_changes)
        self.without_timestamps_cb.toggled.connect(self._check_for_changes)
        self.word_timestamps_cb.toggled.connect(self._check_for_changes)
        self.beam_size_spin.valueChanged.connect(self._check_for_changes)
        self.vad_filter_cb.toggled.connect(self._check_for_changes)
        self.condition_on_previous_cb.toggled.connect(self._check_for_changes)

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

        self.without_timestamps_cb.setChecked(
            self.current_whisper_settings.get("without_timestamps", False)
        )
        self.word_timestamps_cb.setChecked(
            self.current_whisper_settings.get("word_timestamps", False)
        )
        self.beam_size_spin.setValue(
            self.current_whisper_settings.get("beam_size", 5)
        )
        self.vad_filter_cb.setChecked(
            self.current_whisper_settings.get("vad_filter", False)
        )
        self.condition_on_previous_cb.setChecked(
            self.current_whisper_settings.get("condition_on_previous_text", True)
        )

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

    def _whisper_settings_selection_changed(self) -> bool:
        current = {
            "without_timestamps": self.without_timestamps_cb.isChecked(),
            "word_timestamps": self.word_timestamps_cb.isChecked(),
            "beam_size": self.beam_size_spin.value(),
            "vad_filter": self.vad_filter_cb.isChecked(),
            "condition_on_previous_text": self.condition_on_previous_cb.isChecked(),
        }
        return current != self.current_whisper_settings

    def _check_for_changes(self) -> None:
        model_changed = self._model_settings_changed()
        task_changed = self._task_mode_selection_changed()
        audio_changed = self._audio_device_selection_changed()
        whisper_changed = self._whisper_settings_selection_changed()
        has_changes = model_changed or task_changed or audio_changed or whisper_changed
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

        if self._whisper_settings_selection_changed():
            settings = {
                "without_timestamps": self.without_timestamps_cb.isChecked(),
                "word_timestamps": self.word_timestamps_cb.isChecked(),
                "beam_size": self.beam_size_spin.value(),
                "vad_filter": self.vad_filter_cb.isChecked(),
                "condition_on_previous_text": self.condition_on_previous_cb.isChecked(),
            }
            self.whisper_settings_changed.emit(settings)

        self.accept()
