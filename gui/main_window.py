"""
Main application window.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QHBoxLayout,
    QGroupBox,
    QRadioButton,
)

from core.quantization import CheckQuantizationSupport
from core.controller import TranscriberController
from config.manager import config_manager
from gui.styles import apply_recording_button_style, apply_update_button_style, apply_clipboard_button_style
from gui.clipboard_window import ClipboardWindow


class MainWindow(QWidget):

    MODEL_CHOICES = [
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v3",
        "large-v3-turbo",
        "distil-whisper-small.en",
        "distil-whisper-medium.en",
        "distil-whisper-large-v3",
    ]
    
    TRANSLATION_CAPABLE_MODELS = [
        "tiny",
        "base",
        "small",
        "medium",
        "large-v3",
    ]

    def __init__(self, cuda_available: bool = False):
        super().__init__()

        config_settings = config_manager.get_model_settings()
        self.loaded_model_settings = config_settings.copy()
        
        self.task_mode = config_manager.get_value("task_mode", "transcribe")

        self.controller = TranscriberController()
        self.supported_quantizations: dict[str, list[str]] = {"cpu": [], "cuda": []}
        self.is_recording = False

        layout = QVBoxLayout(self)

        self.clipboard_window = ClipboardWindow(self)

        self.status_label = QLabel("")
        font = self.status_label.font()
        font.setPointSize(13)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)

        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self._toggle_recording)
        layout.addWidget(self.record_button)

        task_group = QGroupBox("Mode")
        task_layout = QHBoxLayout()
        
        self.transcribe_radio = QRadioButton("Transcribe")
        self.translate_radio = QRadioButton("Translate to English")
        
        if self.task_mode == "transcribe":
            self.transcribe_radio.setChecked(True)
        else:
            self.translate_radio.setChecked(True)
        
        self.transcribe_radio.toggled.connect(self._on_task_mode_changed)
        self.translate_radio.toggled.connect(self._on_task_mode_changed)
        
        task_layout.addWidget(self.transcribe_radio)
        task_layout.addWidget(self.translate_radio)
        task_group.setLayout(task_layout)
        layout.addWidget(task_group)

        self.clipboard_button = QPushButton("Hide Clipboard")
        self.clipboard_button.clicked.connect(self._toggle_clipboard)
        layout.addWidget(self.clipboard_button)

        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        row = QHBoxLayout()

        row.addWidget(QLabel("Model"))
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(self.MODEL_CHOICES)
        row.addWidget(self.model_dropdown)

        row.addWidget(QLabel("Quantization"))
        self.quantization_dropdown = QComboBox()
        row.addWidget(self.quantization_dropdown)

        row.addWidget(QLabel("Device"))
        self.device_dropdown = QComboBox()
        self.device_dropdown.addItems(["cpu", "cuda"] if cuda_available else ["cpu"])
        row.addWidget(self.device_dropdown)

        settings_layout.addLayout(row)

        self.update_model_btn = QPushButton("Update Settings")
        self.update_model_btn.clicked.connect(self.update_model)
        settings_layout.addWidget(self.update_model_btn)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        self.setFixedSize(425, 280)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)

        self._load_config()

        if not self.supported_quantizations.get("cpu") or not self.supported_quantizations.get("cuda"):
            quantization_checker = CheckQuantizationSupport()
            quantization_checker.update_supported_quantizations()
            self._load_config()

        self.device_dropdown.currentTextChanged.connect(self.update_quantization_options)
        self.model_dropdown.currentTextChanged.connect(self.update_quantization_options)

        self.model_dropdown.currentTextChanged.connect(self._on_dropdown_changed)
        self.quantization_dropdown.currentTextChanged.connect(self._on_dropdown_changed)
        self.device_dropdown.currentTextChanged.connect(self._on_dropdown_changed)
        
        self.update_quantization_options()

        self.controller.update_status_signal.connect(self.update_status)
        self.controller.enable_widgets_signal.connect(self.set_widgets_enabled)
        self.controller.text_ready_signal.connect(self.update_clipboard)
        self.controller.model_loaded_signal.connect(self._on_model_loaded_success)

    def _load_config(self) -> None:
        config = config_manager.load_config()
        
        model = config["model_name"]
        quant = config["quantization_type"]
        device = config["device_type"]
        self.supported_quantizations = config["supported_quantizations"]
        show_clipboard = config["show_clipboard_window"]
        self.task_mode = config.get("task_mode", "transcribe")

        if model in self.MODEL_CHOICES:
            self.model_dropdown.setCurrentText(model)
        self.device_dropdown.setCurrentText(device)
        self.update_quantization_options()
        if quant in [self.quantization_dropdown.itemText(i) for i in range(self.quantization_dropdown.count())]:
            self.quantization_dropdown.setCurrentText(quant)

        self.clipboard_window.setVisible(show_clipboard)
        self._update_clipboard_button_text()
        self._update_translation_availability(model)

    def _save_clipboard_setting(self, show_clipboard: bool) -> None:
        config_manager.set_value("show_clipboard_window", show_clipboard)

    def _is_translation_capable(self, model_name: str) -> bool:
        return model_name in self.TRANSLATION_CAPABLE_MODELS

    def _update_translation_availability(self, model_name: str) -> None:
        translation_capable = self._is_translation_capable(model_name)
        
        self.translate_radio.setEnabled(translation_capable)
        
        if not translation_capable:
            self.translate_radio.setToolTip("Translation not supported by the model currently loaded")
            if self.translate_radio.isChecked():
                self.transcribe_radio.setChecked(True)
                self.task_mode = "transcribe"
                config_manager.set_value("task_mode", self.task_mode)
                self.controller.set_task_mode(self.task_mode)
        else:
            self.translate_radio.setToolTip("")

    @Slot()
    def _on_dropdown_changed(self) -> None:
        self._update_button_state()

    @Slot()
    def _on_task_mode_changed(self) -> None:
        if self.transcribe_radio.isChecked():
            self.task_mode = "transcribe"
        else:
            self.task_mode = "translate"
        
        config_manager.set_value("task_mode", self.task_mode)
        self.controller.set_task_mode(self.task_mode)

    def _update_button_state(self) -> None:
        current_selections = {
            "model_name": self.model_dropdown.currentText(),
            "quantization_type": self.quantization_dropdown.currentText(),
            "device_type": self.device_dropdown.currentText()
        }

        has_changes = current_selections != self.loaded_model_settings

        if has_changes:
            self.update_model_btn.setText("Reload Model to Apply Changes")
            apply_update_button_style(self.update_model_btn, True)
        else:
            self.update_model_btn.setText("Update Settings")
            apply_update_button_style(self.update_model_btn, False)

    @Slot(str, str, str)
    def _on_model_loaded_success(self, model_name: str, quantization_type: str, device_type: str) -> None:
        self.loaded_model_settings = {
            "model_name": model_name,
            "quantization_type": quantization_type,
            "device_type": device_type
        }
        self._update_button_state()
        self._update_translation_availability(model_name)

    @Slot()
    def _toggle_recording(self) -> None:
        if not self.is_recording:
            self.controller.start_recording()
            self.is_recording = True
            self.record_button.setText("Recording...click again to stop and transcribe")
        else:
            self.controller.stop_recording()
            self.is_recording = False
            self.record_button.setText("Start Recording")

        apply_recording_button_style(self.record_button, self.is_recording)

    @Slot()
    def _toggle_clipboard(self) -> None:
        is_visible = self.clipboard_window.isVisible()
        self.clipboard_window.setVisible(not is_visible)
        self._save_clipboard_setting(not is_visible)
        self._update_clipboard_button_text()

    def _update_clipboard_button_text(self) -> None:
        if self.clipboard_window.isVisible():
            self.clipboard_button.setText("Hide Clipboard")
        else:
            self.clipboard_button.setText("Show Clipboard")

    def update_clipboard(self, text: str) -> None:
        self.clipboard_window.update_text(text)
        if self.is_recording:
            self.is_recording = False
            self.record_button.setText("Start Recording")
            apply_recording_button_style(self.record_button, self.is_recording)

    def get_quantization_options(self, model: str, device: str) -> list[str]:
        distil = {
            "distil-whisper-small.en": ["float16", "bfloat16", "float32"],
            "distil-whisper-medium.en": ["float16", "bfloat16", "float32"],
            "distil-whisper-large-v2": ["float16", "float32"],
            "distil-whisper-large-v3": ["float16", "bfloat16", "float32"],
        }

        special_models = {
            "large-v3-turbo": ["float16", "bfloat16", "float32"],
        }

        if model in special_models:
            options = special_models[model]
        else:
            options = distil.get(model, self.supported_quantizations.get(device, []))

        if device == "cpu":
            options = [opt for opt in options if opt not in ["float16", "bfloat16"]]

        return options

    @Slot()
    def update_quantization_options(self) -> None:
        model = self.model_dropdown.currentText()
        device = self.device_dropdown.currentText()
        opts = self.get_quantization_options(model, device)

        self.quantization_dropdown.blockSignals(True)
        current_text = self.quantization_dropdown.currentText()
        self.quantization_dropdown.clear()
        self.quantization_dropdown.addItems(opts)

        if current_text in opts:
            self.quantization_dropdown.setCurrentText(current_text)
        elif opts:
            self.quantization_dropdown.setCurrentText(opts[0])

        self.quantization_dropdown.blockSignals(False)

        self._update_button_state()

    def update_model(self) -> None:
        self.controller.update_model(
            self.model_dropdown.currentText(),
            self.quantization_dropdown.currentText(),
            self.device_dropdown.currentText(),
        )

    @Slot(str)
    def update_status(self, text: str) -> None:
        self.status_label.setText(text)

    @Slot(bool)
    def set_widgets_enabled(self, enabled: bool) -> None:
        self.record_button.setEnabled(enabled)
        self.clipboard_button.setEnabled(enabled)
        self.model_dropdown.setEnabled(enabled)
        self.quantization_dropdown.setEnabled(enabled)
        self.device_dropdown.setEnabled(enabled)
        self.update_model_btn.setEnabled(enabled)
        self.transcribe_radio.setEnabled(enabled)
        
        current_model = self.model_dropdown.currentText()
        self.translate_radio.setEnabled(enabled and self._is_translation_capable(current_model))

        if not enabled and self.is_recording:
            self.is_recording = False
            self.record_button.setText("Start Recording")
            apply_recording_button_style(self.record_button, self.is_recording)

    def moveEvent(self, event):
        if self.clipboard_window.isVisible():
            self.clipboard_window.move(self.x() - self.clipboard_window.width(), self.y())
        super().moveEvent(event)

    def closeEvent(self, event):
        self._save_clipboard_setting(self.clipboard_window.isVisible())
        self.clipboard_window.close()
        self.controller.stop_all_threads()
        super().closeEvent(event)