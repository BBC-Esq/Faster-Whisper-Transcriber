from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot, QRect, QEvent, Signal, QSettings, QByteArray
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QGroupBox,
    QRadioButton,
    QMessageBox,
    QFileDialog,
    QFormLayout,
    QSizePolicy,
    QDialog,
    QDialogButtonBox,
    QSpinBox,
    QCheckBox,
)

from core.quantization import CheckQuantizationSupport
from core.controller import TranscriberController
from core.models.metadata import ModelMetadata
from core.hotkeys import GlobalHotkey
from config.manager import config_manager
from gui.styles import APP_STYLESHEET, update_button_property
from gui.clipboard_window import ClipboardSideWindow
from core.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac", ".amr", ".asf", ".avi", ".flac", ".m4a",
    ".mkv", ".mp3", ".mp4", ".wav", ".webm", ".wma",
}

# QSettings keys
SETTINGS_GEOMETRY = "window/geometry"
SETTINGS_CLIPBOARD_GEOMETRY = "clipboard/geometry"
SETTINGS_CLIPBOARD_DOCKED = "clipboard/docked"
SETTINGS_CLIPBOARD_VISIBLE = "clipboard/visible"
SETTINGS_CLIPBOARD_ALWAYS_ON_TOP = "clipboard/always_on_top"
SETTINGS_APPEND_MODE = "clipboard/append_mode"
SETTINGS_MODEL = "model/name"
SETTINGS_DEVICE = "model/device"
SETTINGS_QUANTIZATION = "model/quantization"
SETTINGS_TASK_MODE = "model/task_mode"


class BatchSizeDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, default_value: int = 16):
        super().__init__(parent)
        self.setWindowTitle("Batch Size")
        self.setModal(True)
        self.setFixedWidth(360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Set batch size for file transcription.\nUse 1 for non-batched transcription."))

        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(QLabel("Batch size"))
        self.spin = QSpinBox()
        self.spin.setRange(1, 256)
        self.spin.setValue(max(1, int(default_value)))
        row.addWidget(self.spin, 1)
        layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def batch_size(self) -> int:
        return self.spin.value()


class MainWindow(QMainWindow):
    hotkey_toggle_recording = Signal()

    # Default values for settings
    DEFAULTS = {
        "model": "base.en",
        "device": "cpu",
        "quantization": "float32",
        "task_mode": "transcribe",
        "append_mode": False,
        "clipboard_visible": False,
        "clipboard_always_on_top": True,
        "clipboard_docked": True,
    }

    def __init__(self, cuda_available: bool = False):
        super().__init__()

        self.setWindowTitle("Faster Whisper Transcriber")
        self.setStyleSheet(APP_STYLESHEET)

        # Initialize QSettings
        self.settings = QSettings("FasterWhisperTranscriber", "Transcriber")

        self.loaded_model_settings = config_manager.get_model_settings()
        self.task_mode = config_manager.get_value("task_mode", "transcribe")
        self.controller = TranscriberController()
        self.supported_quantizations: dict[str, list[str]] = {"cpu": [], "cuda": []}
        self.is_recording = False
        self.cuda_available = cuda_available
        self._clipboard_visible = False
        self._toggleable_widgets: list[QWidget] = []

        self.clipboard_window = ClipboardSideWindow(None, width=360)
        self.clipboard_window.user_closed.connect(self._on_clipboard_closed)
        self.clipboard_window.docked_changed.connect(lambda d: logger.debug(f"Clipboard docked: {d}"))
        self.clipboard_window.always_on_top_changed.connect(self._on_clipboard_always_on_top_changed)

        self._build_ui()
        self._setup_connections()

        # Load quantization support first (needed for validation)
        self._load_quantization_support()

        # Restore state from QSettings with validation
        self._restore_state()

        self.hotkey_toggle_recording.connect(self._toggle_recording, Qt.QueuedConnection)
        self.global_hotkey = GlobalHotkey(lambda: self.hotkey_toggle_recording.emit())
        self.global_hotkey.start()

        self.setAcceptDrops(True)
        if app := QApplication.instance():
            app.installEventFilter(self)

        logger.info("MainWindow initialized")

    def _load_quantization_support(self) -> None:
        """Load quantization support, updating if necessary."""
        try:
            config = config_manager.load_config()
            self.supported_quantizations = config.get("supported_quantizations", {"cpu": [], "cuda": []})

            # Update if not populated
            if not self.supported_quantizations.get("cpu") or (
                self.cuda_available and not self.supported_quantizations.get("cuda")
            ):
                CheckQuantizationSupport().update_supported_quantizations()
                config = config_manager.load_config()
                self.supported_quantizations = config.get("supported_quantizations", {"cpu": [], "cuda": []})
        except Exception as e:
            logger.error(f"Failed to load quantization support: {e}")
            self.supported_quantizations = {"cpu": ["float32"], "cuda": []}

    def _validate_model(self, model_name: str) -> str:
        """Validate model name, return default if invalid."""
        valid_models = ModelMetadata.get_all_model_names()
        if model_name in valid_models:
            return model_name
        logger.warning(f"Invalid model '{model_name}', falling back to default")
        return self.DEFAULTS["model"]

    def _validate_device(self, device: str) -> str:
        """Validate device, return default if invalid (e.g., CUDA no longer available)."""
        if device == "cuda" and not self.cuda_available:
            logger.warning("CUDA no longer available, falling back to CPU")
            return "cpu"
        if device in ["cpu", "cuda"]:
            return device
        logger.warning(f"Invalid device '{device}', falling back to default")
        return self.DEFAULTS["device"]

    def _validate_quantization(self, quantization: str, model_name: str, device: str) -> str:
        """Validate quantization for the given model/device, return default if invalid."""
        available = ModelMetadata.get_quantization_options(model_name, device, self.supported_quantizations)
        if quantization in available:
            return quantization
        if available:
            logger.warning(f"Quantization '{quantization}' not available for {model_name}/{device}, using {available[0]}")
            return available[0]
        logger.warning(f"No quantizations available for {model_name}/{device}, using float32")
        return "float32"

    def _validate_task_mode(self, task_mode: str, model_name: str) -> str:
        """Validate task mode for the given model."""
        if task_mode == "translate" and not ModelMetadata.supports_translation(model_name):
            logger.warning(f"Model '{model_name}' doesn't support translation, falling back to transcribe")
            return "transcribe"
        if task_mode in ["transcribe", "translate"]:
            return task_mode
        return self.DEFAULTS["task_mode"]

    def _restore_state(self) -> None:
        """Restore application state from QSettings with validation and fallbacks."""
        logger.info("Restoring application state from QSettings")

        # Restore main window geometry
        geometry = self.settings.value(SETTINGS_GEOMETRY)
        if geometry and isinstance(geometry, QByteArray):
            if not self.restoreGeometry(geometry):
                logger.warning("Failed to restore main window geometry, using defaults")
                self.resize(425, 280)
                self._center_on_screen()
        else:
            self.resize(425, 280)
            self._center_on_screen()

        # Restore and validate model settings
        saved_model = self.settings.value(SETTINGS_MODEL, self.DEFAULTS["model"])
        saved_device = self.settings.value(SETTINGS_DEVICE, self.DEFAULTS["device"])
        saved_quant = self.settings.value(SETTINGS_QUANTIZATION, self.DEFAULTS["quantization"])
        saved_task = self.settings.value(SETTINGS_TASK_MODE, self.DEFAULTS["task_mode"])

        # Validate with fallbacks
        model = self._validate_model(saved_model)
        device = self._validate_device(saved_device)
        quantization = self._validate_quantization(saved_quant, model, device)
        task_mode = self._validate_task_mode(saved_task, model)

        # Apply validated settings to UI
        self.model_dropdown.setCurrentText(model)
        self.device_dropdown.setCurrentText(device)
        self.update_quantization_options()
        self.quantization_dropdown.setCurrentText(quantization)

        self.task_mode = task_mode
        self.transcribe_radio.setChecked(task_mode == "transcribe")
        self.translate_radio.setChecked(task_mode != "transcribe")
        self._update_translation_availability(model)

        # Restore clipboard settings
        append_mode = self.settings.value(SETTINGS_APPEND_MODE, self.DEFAULTS["append_mode"], type=bool)
        self.append_checkbox.blockSignals(True)
        self.append_checkbox.setChecked(append_mode)
        self.append_checkbox.blockSignals(False)
        self.clipboard_window.set_append_mode(append_mode)

        always_on_top = self.settings.value(SETTINGS_CLIPBOARD_ALWAYS_ON_TOP, self.DEFAULTS["clipboard_always_on_top"], type=bool)
        self.clipboard_window.set_always_on_top(always_on_top)

        clipboard_docked = self.settings.value(SETTINGS_CLIPBOARD_DOCKED, self.DEFAULTS["clipboard_docked"], type=bool)
        self._clipboard_visible = self.settings.value(SETTINGS_CLIPBOARD_VISIBLE, self.DEFAULTS["clipboard_visible"], type=bool)

        # Restore clipboard window geometry if undocked
        if self._clipboard_visible:
            if clipboard_docked:
                self.clipboard_window.show_docked(self._host_rect_global(), gap=10, animate=False)
            else:
                # Restore undocked position
                clip_geometry = self.settings.value(SETTINGS_CLIPBOARD_GEOMETRY)
                if clip_geometry and isinstance(clip_geometry, QByteArray):
                    self.clipboard_window.set_docked(False)
                    self.clipboard_window.restoreGeometry(clip_geometry)
                    self.clipboard_window.show()
                else:
                    # Fallback to docked if geometry restoration fails
                    self.clipboard_window.show_docked(self._host_rect_global(), gap=10, animate=False)

        self._update_clipboard_button_text()

        # Update config manager with validated settings
        try:
            config_manager.set_model_settings(model, quantization, device)
            config_manager.set_value("task_mode", task_mode)
            config_manager.set_value("clipboard_append_mode", append_mode)
            config_manager.set_value("show_clipboard_window", self._clipboard_visible)
        except Exception as e:
            logger.warning(f"Failed to sync config manager: {e}")

        logger.info(f"State restored: model={model}, device={device}, quant={quantization}, task={task_mode}")

    def _save_state(self) -> None:
        """Save application state to QSettings."""
        logger.info("Saving application state to QSettings")

        # Save main window geometry
        self.settings.setValue(SETTINGS_GEOMETRY, self.saveGeometry())

        # Save model settings
        self.settings.setValue(SETTINGS_MODEL, self.model_dropdown.currentText())
        self.settings.setValue(SETTINGS_DEVICE, self.device_dropdown.currentText())
        self.settings.setValue(SETTINGS_QUANTIZATION, self.quantization_dropdown.currentText())
        self.settings.setValue(SETTINGS_TASK_MODE, self.task_mode)

        # Save clipboard settings
        self.settings.setValue(SETTINGS_APPEND_MODE, self.append_checkbox.isChecked())
        self.settings.setValue(SETTINGS_CLIPBOARD_VISIBLE, self._clipboard_visible)
        self.settings.setValue(SETTINGS_CLIPBOARD_ALWAYS_ON_TOP, self.clipboard_window.is_always_on_top())
        self.settings.setValue(SETTINGS_CLIPBOARD_DOCKED, self.clipboard_window.is_docked())

        # Save clipboard geometry if undocked
        if not self.clipboard_window.is_docked():
            self.settings.setValue(SETTINGS_CLIPBOARD_GEOMETRY, self.clipboard_window.saveGeometry())

        self.settings.sync()
        logger.debug("State saved successfully")

    def _center_on_screen(self) -> None:
        """Center the window on the primary screen."""
        if screen := QApplication.primaryScreen():
            screen_geometry = screen.availableGeometry()
            x = (screen_geometry.width() - self.width()) // 2 + screen_geometry.x()
            y = (screen_geometry.height() - self.height()) // 2 + screen_geometry.y()
            self.move(x, y)

    def _on_clipboard_always_on_top_changed(self, value: bool) -> None:
        """Handle clipboard always-on-top change."""
        self.settings.setValue(SETTINGS_CLIPBOARD_ALWAYS_ON_TOP, value)

    def _setup_connections(self) -> None:
        self.device_dropdown.currentTextChanged.connect(self.update_quantization_options)
        self.model_dropdown.currentTextChanged.connect(self.update_quantization_options)
        self.model_dropdown.currentTextChanged.connect(self._on_dropdown_changed)
        self.quantization_dropdown.currentTextChanged.connect(self._on_dropdown_changed)
        self.device_dropdown.currentTextChanged.connect(self._on_dropdown_changed)

        self.controller.update_status_signal.connect(self.statusBar().showMessage)
        self.controller.enable_widgets_signal.connect(self.set_widgets_enabled)
        self.controller.text_ready_signal.connect(self._on_transcription_ready)
        self.controller.model_loaded_signal.connect(self._on_model_loaded_success)
        self.controller.error_occurred.connect(self._show_error_dialog)

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(5, 5, 5, 5)
        root.setSpacing(8)

        root.addWidget(self._build_actions_group())
        root.addWidget(self._build_settings_group())

        self.statusBar().showMessage("Ready")
        self.statusBar().setSizeGripEnabled(True)

        self.resize(425, 280)
        self.setMinimumSize(425, 280)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)

    def _build_actions_group(self) -> QGroupBox:
        group = QGroupBox("")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(8)

        self.record_button = QPushButton("Start Recording")
        self.record_button.setObjectName("recordButton")
        self.record_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.record_button.setToolTip("Click or press F9 to toggle recording")
        self.record_button.clicked.connect(self._toggle_recording)
        buttons_row.addWidget(self.record_button, 2)
        self._register_toggleable_widget(self.record_button)

        self.transcribe_file_button = QPushButton("Transcribe Audio File")
        self.transcribe_file_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.transcribe_file_button.setToolTip("Select an audio file to transcribe")
        self.transcribe_file_button.clicked.connect(self._select_and_transcribe_file)
        buttons_row.addWidget(self.transcribe_file_button, 1)
        self._register_toggleable_widget(self.transcribe_file_button)

        layout.addLayout(buttons_row)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)

        mode_label = QLabel("Mode")
        mode_label.setMinimumWidth(48)
        mode_row.addWidget(mode_label)

        self.transcribe_radio = QRadioButton("Transcribe")
        self.translate_radio = QRadioButton("Translate")
        self.transcribe_radio.toggled.connect(self._on_task_mode_changed)
        self._register_toggleable_widget(self.transcribe_radio)

        mode_row.addWidget(self.transcribe_radio)
        mode_row.addWidget(self.translate_radio)
        mode_row.addStretch(1)

        self.clipboard_button = QPushButton("Show Clipboard")
        self.clipboard_button.setObjectName("clipboardButton")
        self.clipboard_button.setToolTip("Show or hide the clipboard panel")
        self.clipboard_button.setMinimumWidth(130)
        self.clipboard_button.clicked.connect(self._toggle_clipboard)
        mode_row.addWidget(self.clipboard_button)
        self._register_toggleable_widget(self.clipboard_button)

        layout.addLayout(mode_row)
        return group

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox("")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(ModelMetadata.get_all_model_names())
        self.model_dropdown.setToolTip("Choose a Whisper model")
        self._register_toggleable_widget(self.model_dropdown)

        self.loaded_model_label = QLabel("Not loaded")
        self.loaded_model_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.loaded_model_label.setToolTip("Currently loaded model")
        self.loaded_model_label.setStyleSheet("color: rgba(34, 197, 94, 0.95); font-weight: 600;")

        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        model_row.addWidget(self.model_dropdown, 1)
        model_row.addWidget(self.loaded_model_label, 0)

        form_top = QFormLayout()
        form_top.setContentsMargins(0, 0, 0, 0)
        form_top.setHorizontalSpacing(10)
        form_top.setVerticalSpacing(6)
        form_top.addRow("Model", model_row)
        layout.addLayout(form_top)

        self.device_dropdown = QComboBox()
        self.device_dropdown.addItems(["cpu", "cuda"] if self.cuda_available else ["cpu"])
        self.device_dropdown.setToolTip("Choose the compute device")
        self._register_toggleable_widget(self.device_dropdown)

        self.quantization_dropdown = QComboBox()
        self.quantization_dropdown.setToolTip("Choose the compute type (quantization)")
        self._register_toggleable_widget(self.quantization_dropdown)

        self.append_checkbox = QCheckBox("Append")
        self.append_checkbox.setToolTip("Append new transcriptions instead of replacing")
        self.append_checkbox.toggled.connect(self._on_append_toggled)
        self._register_toggleable_widget(self.append_checkbox)

        row = QHBoxLayout()
        row.setSpacing(10)

        for label, widget in [("Device", self.device_dropdown), ("Quantization", self.quantization_dropdown)]:
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setHorizontalSpacing(10)
            form.setVerticalSpacing(6)
            form.addRow(label, widget)
            row.addLayout(form, 1)

        row.addWidget(self.append_checkbox, 0)
        layout.addLayout(row)

        self.update_model_btn = QPushButton("Update Settings")
        self.update_model_btn.setObjectName("updateButton")
        self.update_model_btn.setToolTip("Load the selected model and settings")
        self.update_model_btn.clicked.connect(self._update_model)
        layout.addWidget(self.update_model_btn)
        self._register_toggleable_widget(self.update_model_btn)

        return group

    def _host_rect_global(self) -> QRect:
        pos = self.mapToGlobal(self.rect().topLeft())
        return QRect(pos.x(), pos.y(), self.width(), self.height())

    def _save_config(self, key: str, value) -> None:
        try:
            config_manager.set_value(key, value)
        except Exception as e:
            logger.warning(f"Failed to save {key}: {e}")

    def _update_translation_availability(self, model_name: str) -> None:
        can_translate = ModelMetadata.supports_translation(model_name)
        self.translate_radio.setEnabled(can_translate)
        self.translate_radio.setToolTip("" if can_translate else "Translation not supported by the selected model")

        if not can_translate and self.translate_radio.isChecked():
            self.transcribe_radio.setChecked(True)
            self.task_mode = "transcribe"
            self._save_config("task_mode", self.task_mode)
            self.controller.set_task_mode(self.task_mode)

    @Slot()
    def _on_clipboard_closed(self) -> None:
        self._clipboard_visible = False
        self._update_clipboard_button_text()
        self._save_config("show_clipboard_window", False)

    @Slot(bool)
    def _on_append_toggled(self, checked: bool) -> None:
        self.clipboard_window.set_append_mode(checked)
        self._save_config("clipboard_append_mode", checked)

    @Slot(str, str)
    def _show_error_dialog(self, title: str, message: str) -> None:
        logger.error(f"Error: {title} - {message}")
        QMessageBox.warning(self, title, message)

    @Slot()
    def _on_dropdown_changed(self) -> None:
        current = {
            "model_name": self.model_dropdown.currentText(),
            "quantization_type": self.quantization_dropdown.currentText(),
            "device_type": self.device_dropdown.currentText(),
        }
        has_changes = current != self.loaded_model_settings
        self.update_model_btn.setText("Reload Model to Apply Changes" if has_changes else "Update Settings")
        update_button_property(self.update_model_btn, "changed", has_changes)

    @Slot()
    def _on_task_mode_changed(self) -> None:
        self.task_mode = "transcribe" if self.transcribe_radio.isChecked() else "translate"
        self._save_config("task_mode", self.task_mode)
        self.controller.set_task_mode(self.task_mode)

    @Slot(str, str, str)
    def _on_model_loaded_success(self, model_name: str, quant: str, device: str) -> None:
        self.loaded_model_settings = {"model_name": model_name, "quantization_type": quant, "device_type": device}
        self.loaded_model_label.setText(model_name)
        self._on_dropdown_changed()
        self._update_translation_availability(model_name)

    @Slot()
    def _toggle_recording(self) -> None:
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.controller.start_recording()
            self.record_button.setText("Click to Stop and Transcribe")
        else:
            self.controller.stop_recording()
            self.record_button.setText("Start Recording")
        update_button_property(self.record_button, "recording", self.is_recording)

    def _is_supported_audio_file(self, path: str) -> bool:
        try:
            return Path(path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        except Exception:
            return False

    def _transcribe_specific_file(self, file_path: str) -> None:
        if not file_path or not self._is_supported_audio_file(file_path):
            self.statusBar().showMessage("Unsupported file type")
            QMessageBox.warning(self, "Unsupported File", "Please select a supported audio file.")
            return

        if not self.transcribe_file_button.isEnabled():
            self.statusBar().showMessage("Busy")
            return

        dlg = BatchSizeDialog(self, default_value=16)
        if dlg.exec() == QDialog.Accepted:
            logger.info(f"Transcribing: {file_path} (batch={dlg.batch_size()})")
            self.controller.transcribe_file(file_path, batch_size=dlg.batch_size())

    @Slot()
    def _select_and_transcribe_file(self) -> None:
        exts = " ".join(f"*{ext}" for ext in SUPPORTED_AUDIO_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", f"Audio Files ({exts});;All Files (*)")
        if path:
            self._transcribe_specific_file(path)

    @Slot()
    def _toggle_clipboard(self) -> None:
        self._clipboard_visible = not self._clipboard_visible
        self._update_clipboard_button_text()
        self._save_config("show_clipboard_window", self._clipboard_visible)

        host_rect = self._host_rect_global()
        if self._clipboard_visible:
            if self.clipboard_window.is_docked():
                self.clipboard_window.show_docked(host_rect, gap=10)
            else:
                self.clipboard_window.show()
                self.clipboard_window.raise_()
        else:
            self.clipboard_window.hide_animated(host_rect, gap=10)

    def _update_clipboard_button_text(self) -> None:
        self.clipboard_button.setText("Hide Clipboard" if self._clipboard_visible else "Show Clipboard")

    def _register_toggleable_widget(self, widget: QWidget) -> None:
        self._toggleable_widgets.append(widget)

    @Slot(str)
    def _on_transcription_ready(self, text: str) -> None:
        self.clipboard_window.add_transcription(text)

        if app := QApplication.instance():
            clip_text = self.clipboard_window.get_full_text() if self.clipboard_window.is_append_mode() else text
            app.clipboard().setText(clip_text)

        if self.is_recording:
            self.is_recording = False
            self.record_button.setText("Start Recording")
            update_button_property(self.record_button, "recording", False)

    @Slot()
    def update_quantization_options(self) -> None:
        model = self.model_dropdown.currentText()
        device = self.device_dropdown.currentText()
        opts = ModelMetadata.get_quantization_options(model, device, self.supported_quantizations)

        self.quantization_dropdown.blockSignals(True)
        current = self.quantization_dropdown.currentText()
        self.quantization_dropdown.clear()
        self.quantization_dropdown.addItems(opts)
        self.quantization_dropdown.setCurrentText(current if current in opts else (opts[0] if opts else ""))
        self.quantization_dropdown.blockSignals(False)

        self._on_dropdown_changed()

    @Slot()
    def _update_model(self) -> None:
        self.loaded_model_label.setText("Loading...")
        self.controller.update_model(
            self.model_dropdown.currentText(),
            self.quantization_dropdown.currentText(),
            self.device_dropdown.currentText(),
        )

    @Slot(bool)
    def set_widgets_enabled(self, enabled: bool) -> None:
        for widget in self._toggleable_widgets:
            widget.setEnabled(enabled)

        self.translate_radio.setEnabled(enabled and ModelMetadata.supports_translation(self.model_dropdown.currentText()))

        if not enabled and self.is_recording:
            self.is_recording = False
            self.record_button.setText("Start Recording")
            update_button_property(self.record_button, "recording", False)

    def _extract_first_supported_drop(self, event) -> str | None:
        if not (md := event.mimeData()) or not md.hasUrls():
            return None
        for url in md.urls():
            if url.isLocalFile() and self._is_supported_audio_file(p := url.toLocalFile()):
                return p
        return None

    def eventFilter(self, obj, event):
        et = event.type()
        if et in (QEvent.DragEnter, QEvent.DragMove):
            if self._extract_first_supported_drop(event):
                event.acceptProposedAction()
                return True
        elif et == QEvent.Drop:
            if p := self._extract_first_supported_drop(event):
                event.acceptProposedAction()
                self._transcribe_specific_file(p)
                return True
        return super().eventFilter(obj, event)

    def _sync_clipboard_position(self) -> None:
        host_rect = self._host_rect_global()
        self.clipboard_window.update_host_rect(host_rect)
        if self.clipboard_window.isVisible() and self.clipboard_window.is_docked():
            self.clipboard_window.reposition_to_host(host_rect, gap=10)

    def moveEvent(self, event):
        super().moveEvent(event)
        self._sync_clipboard_position()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_clipboard_position()

    def closeEvent(self, event):
        # Save state before closing
        self._save_state()

        self._save_config("show_clipboard_window", self._clipboard_visible)
        self.clipboard_window.close()
        self.controller.stop_all_threads()
        if hasattr(self, "global_hotkey"):
            self.global_hotkey.stop()
        logger.info("Application closing")
        super().closeEvent(event)