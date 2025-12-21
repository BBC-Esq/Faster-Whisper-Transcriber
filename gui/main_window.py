from __future__ import annotations

from PySide6.QtCore import Qt, Slot, QRect
from PySide6.QtWidgets import (
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
)

from core.quantization import CheckQuantizationSupport
from core.controller import TranscriberController
from core.models.metadata import ModelMetadata
from config.manager import config_manager
from gui.styles import APP_STYLESHEET, apply_recording_button_style, apply_update_button_style, apply_clipboard_button_style
from gui.clipboard_window import ClipboardSideWindow
from core.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = [
    ".aac",
    ".amr",
    ".asf",
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mp3",
    ".mp4",
    ".wav",
    ".webm",
    ".wma",
]


class BatchSizeDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, default_value: int = 16):
        super().__init__(parent)
        self.setWindowTitle("Batch Size")
        self.setModal(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        label = QLabel("Set batch size for file transcription.\nUse 1 for non-batched transcription.")
        label.setWordWrap(True)
        root.addWidget(label)

        row = QHBoxLayout()
        row.setSpacing(10)

        batch_label = QLabel("Batch size")
        self.spin = QSpinBox()
        self.spin.setRange(1, 256)
        self.spin.setValue(max(1, int(default_value)))
        self.spin.setSingleStep(1)

        row.addWidget(batch_label)
        row.addWidget(self.spin, 1)
        root.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.setFixedWidth(360)

    def batch_size(self) -> int:
        return int(self.spin.value())


class MainWindow(QMainWindow):
    def __init__(self, cuda_available: bool = False):
        super().__init__()

        self.setWindowTitle("Faster Whisper Transcriber")
        self.setStyleSheet(APP_STYLESHEET)

        config_settings = config_manager.get_model_settings()
        self.loaded_model_settings = config_settings.copy()

        self.task_mode = config_manager.get_value("task_mode", "transcribe")

        self.controller = TranscriberController()
        self.supported_quantizations: dict[str, list[str]] = {"cpu": [], "cuda": []}
        self.is_recording = False
        self.cuda_available = cuda_available

        self.clipboard_window = ClipboardSideWindow(self, width=360)
        self.clipboard_window.switch_side_requested.connect(self._switch_clipboard_side)
        self.clipboard_window.user_closed.connect(self._on_clipboard_closed)

        self._clipboard_target_visible = False
        self._clipboard_side = "right"

        self._build_ui()
        self._load_config()

        if not self.supported_quantizations.get("cpu") or (self.cuda_available and not self.supported_quantizations.get("cuda")):
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
        self.controller.error_occurred.connect(self._show_error_dialog)

        logger.info("MainWindow initialized")

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(5, 5, 5, 5)
        root.setSpacing(8)

        actions_group = QGroupBox("")
        actions_layout = QVBoxLayout()
        actions_layout.setContentsMargins(2, 2, 2, 2)
        actions_layout.setSpacing(8)

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(8)

        self.record_button = QPushButton("Start Recording")
        self.record_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.record_button.setToolTip("Start recording from your microphone. Click again to stop and transcribe.")
        self.record_button.clicked.connect(self._toggle_recording)
        buttons_row.addWidget(self.record_button, 2)

        self.transcribe_file_button = QPushButton("Transcribe Audio File")
        self.transcribe_file_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.transcribe_file_button.setToolTip("Select an audio file to transcribe.")
        self.transcribe_file_button.clicked.connect(self._select_and_transcribe_file)
        buttons_row.addWidget(self.transcribe_file_button, 1)

        actions_layout.addLayout(buttons_row)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)

        mode_label = QLabel("Mode")
        mode_label.setMinimumWidth(48)
        mode_row.addWidget(mode_label)

        self.transcribe_radio = QRadioButton("Transcribe")
        self.translate_radio = QRadioButton("Translate")

        self.transcribe_radio.toggled.connect(self._on_task_mode_changed)
        self.translate_radio.toggled.connect(self._on_task_mode_changed)

        mode_row.addWidget(self.transcribe_radio)
        mode_row.addWidget(self.translate_radio)
        mode_row.addStretch(1)

        self.clipboard_button = QPushButton("Show Clipboard")
        self.clipboard_button.setToolTip("Show or hide the clipboard panel")
        self.clipboard_button.setMinimumWidth(130)
        apply_clipboard_button_style(self.clipboard_button)
        self.clipboard_button.clicked.connect(self._toggle_clipboard)
        mode_row.addWidget(self.clipboard_button)

        actions_layout.addLayout(mode_row)

        actions_group.setLayout(actions_layout)
        root.addWidget(actions_group)

        settings_group = QGroupBox("")
        settings_layout = QVBoxLayout()
        settings_layout.setContentsMargins(2, 2, 2, 2)
        settings_layout.setSpacing(8)

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(ModelMetadata.get_all_model_names())
        self.model_dropdown.setToolTip("Choose a Whisper model.")

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
        form_top.setFormAlignment(Qt.AlignTop)
        form_top.setLabelAlignment(Qt.AlignLeft)
        form_top.addRow("Model", model_row)
        settings_layout.addLayout(form_top)

        self.device_dropdown = QComboBox()
        self.device_dropdown.addItems(["cpu", "cuda"] if self.cuda_available else ["cpu"])
        self.device_dropdown.setToolTip("Choose the compute device.")

        self.quantization_dropdown = QComboBox()
        self.quantization_dropdown.setToolTip("Choose the compute type (quantization).")

        row = QHBoxLayout()
        row.setSpacing(10)

        left_form = QFormLayout()
        left_form.setContentsMargins(0, 0, 0, 0)
        left_form.setHorizontalSpacing(10)
        left_form.setVerticalSpacing(6)
        left_form.setLabelAlignment(Qt.AlignLeft)
        left_form.addRow("Device", self.device_dropdown)

        right_form = QFormLayout()
        right_form.setContentsMargins(0, 0, 0, 0)
        right_form.setHorizontalSpacing(10)
        right_form.setVerticalSpacing(6)
        right_form.setLabelAlignment(Qt.AlignLeft)
        right_form.addRow("Quantization", self.quantization_dropdown)

        row.addLayout(left_form, 1)
        row.addLayout(right_form, 1)
        settings_layout.addLayout(row)

        self.update_model_btn = QPushButton("Update Settings")
        self.update_model_btn.setToolTip("Load the selected model and settings.")
        self.update_model_btn.clicked.connect(self.update_model)
        settings_layout.addWidget(self.update_model_btn)

        settings_group.setLayout(settings_layout)
        root.addWidget(settings_group)

        self.statusBar().showMessage("Ready")
        self.statusBar().setSizeGripEnabled(True)

        apply_recording_button_style(self.record_button, False)

        self.resize(425, 280)
        self.setMinimumSize(425, 280)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)

    @Slot()
    def _on_clipboard_closed(self) -> None:
        self._clipboard_target_visible = False
        self._update_clipboard_button_text()
        self._save_clipboard_setting(False)

    def _load_config(self) -> None:
        try:
            config = config_manager.load_config()

            model = config["model_name"]
            quant = config["quantization_type"]
            device = config["device_type"]
            self.supported_quantizations = config["supported_quantizations"]
            show_clipboard = config["show_clipboard_window"]
            self.task_mode = config.get("task_mode", "transcribe")

            model_names = ModelMetadata.get_all_model_names()
            if model in model_names:
                self.model_dropdown.setCurrentText(model)

            if device in [self.device_dropdown.itemText(i) for i in range(self.device_dropdown.count())]:
                self.device_dropdown.setCurrentText(device)

            self.update_quantization_options()

            if quant in [self.quantization_dropdown.itemText(i) for i in range(self.quantization_dropdown.count())]:
                self.quantization_dropdown.setCurrentText(quant)

            if self.task_mode == "transcribe":
                self.transcribe_radio.setChecked(True)
            else:
                self.translate_radio.setChecked(True)

            self._update_translation_availability(self.model_dropdown.currentText())

            self._clipboard_target_visible = bool(show_clipboard)
            if self._clipboard_target_visible:
                self._show_clipboard(animate=False)
            else:
                self.clipboard_window.hide()
                self._update_clipboard_button_text()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def _save_clipboard_setting(self, show_clipboard: bool) -> None:
        try:
            config_manager.set_value("show_clipboard_window", show_clipboard)
        except Exception as e:
            logger.warning(f"Failed to save clipboard setting: {e}")

    def _update_loaded_model_label(self, model_name: str) -> None:
        self.loaded_model_label.setText(model_name)

    def _update_translation_availability(self, model_name: str) -> None:
        translation_capable = ModelMetadata.supports_translation(model_name)

        self.translate_radio.setEnabled(translation_capable)

        if not translation_capable:
            self.translate_radio.setToolTip("Translation not supported by the selected model.")
            if self.translate_radio.isChecked():
                self.transcribe_radio.setChecked(True)
                self.task_mode = "transcribe"
                try:
                    config_manager.set_value("task_mode", self.task_mode)
                except Exception as e:
                    logger.warning(f"Failed to save task mode: {e}")
                self.controller.set_task_mode(self.task_mode)
        else:
            self.translate_radio.setToolTip("")

    @Slot(str, str)
    def _show_error_dialog(self, title: str, message: str) -> None:
        logger.error(f"Error dialog shown - {title}: {message}")
        QMessageBox.warning(self, title, message)

    @Slot()
    def _on_dropdown_changed(self) -> None:
        self._update_button_state()

    @Slot()
    def _on_task_mode_changed(self) -> None:
        if self.transcribe_radio.isChecked():
            self.task_mode = "transcribe"
        else:
            self.task_mode = "translate"

        try:
            config_manager.set_value("task_mode", self.task_mode)
        except Exception as e:
            logger.warning(f"Failed to save task mode: {e}")

        self.controller.set_task_mode(self.task_mode)

    def _update_button_state(self) -> None:
        current_selections = {
            "model_name": self.model_dropdown.currentText(),
            "quantization_type": self.quantization_dropdown.currentText(),
            "device_type": self.device_dropdown.currentText(),
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
            "device_type": device_type,
        }
        self._update_loaded_model_label(model_name)
        self._update_button_state()
        self._update_translation_availability(model_name)

    @Slot()
    def _toggle_recording(self) -> None:
        if not self.is_recording:
            self.controller.start_recording()
            self.is_recording = True
            self.record_button.setText("Click to Stop and Transcribe")
        else:
            self.controller.stop_recording()
            self.is_recording = False
            self.record_button.setText("Start Recording")

        apply_recording_button_style(self.record_button, self.is_recording)

    @Slot()
    def _select_and_transcribe_file(self) -> None:
        extensions_filter = " ".join(f"*{ext}" for ext in SUPPORTED_AUDIO_EXTENSIONS)
        file_filter = f"Audio Files ({extensions_filter});;All Files (*)"

        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", file_filter)

        if not file_path:
            return

        dlg = BatchSizeDialog(self, default_value=16)
        if dlg.exec() != QDialog.Accepted:
            return

        batch_size = dlg.batch_size()
        logger.info(f"Selected file for transcription: {file_path} (batch_size={batch_size})")
        self.controller.transcribe_file(file_path, batch_size=batch_size)

    def _host_rect_global(self) -> QRect:
        top_left = self.mapToGlobal(self.rect().topLeft())
        return QRect(top_left.x(), top_left.y(), self.width(), self.height())

    def _show_clipboard(self, animate: bool = True) -> None:
        self._clipboard_target_visible = True
        self._update_clipboard_button_text()
        self.clipboard_window.dock_and_reset_size()
        host_rect = self._host_rect_global()
        self.clipboard_window.show_beside(host_rect, gap=10, animate=animate, side=self._clipboard_side)
        self._save_clipboard_setting(True)

    def _hide_clipboard(self, animate: bool = True) -> None:
        self._clipboard_target_visible = False
        self._update_clipboard_button_text()
        host_rect = self._host_rect_global()
        self.clipboard_window.hide_away(host_rect, gap=10, animate=animate)
        self._save_clipboard_setting(False)

    @Slot()
    def _toggle_clipboard(self) -> None:
        if self._clipboard_target_visible:
            self._hide_clipboard(animate=True)
        else:
            self._show_clipboard(animate=True)

    @Slot()
    def _switch_clipboard_side(self) -> None:
        self._clipboard_side = "left" if self._clipboard_side == "right" else "right"
        self.clipboard_window.set_side(self._clipboard_side)
        if self.clipboard_window.isVisible() and self.clipboard_window.is_docked():
            self.clipboard_window.move_to_side(self._host_rect_global(), self._clipboard_side, gap=10, animate=True)

    def _update_clipboard_button_text(self) -> None:
        self.clipboard_button.setText("Hide Clipboard" if self._clipboard_target_visible else "Show Clipboard")

    def update_clipboard(self, text: str) -> None:
        self.clipboard_window.update_text(text)
        if self.is_recording:
            self.is_recording = False
            self.record_button.setText("Start Recording")
            apply_recording_button_style(self.record_button, self.is_recording)

    @Slot()
    def update_quantization_options(self) -> None:
        model = self.model_dropdown.currentText()
        device = self.device_dropdown.currentText()
        opts = ModelMetadata.get_quantization_options(model, device, self.supported_quantizations)

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
        self.loaded_model_label.setText("Loading...")
        self.controller.update_model(
            self.model_dropdown.currentText(),
            self.quantization_dropdown.currentText(),
            self.device_dropdown.currentText(),
        )

    @Slot(str)
    def update_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    @Slot(bool)
    def set_widgets_enabled(self, enabled: bool) -> None:
        self.record_button.setEnabled(enabled)
        self.transcribe_file_button.setEnabled(enabled)
        self.clipboard_button.setEnabled(enabled)
        self.model_dropdown.setEnabled(enabled)
        self.quantization_dropdown.setEnabled(enabled)
        self.device_dropdown.setEnabled(enabled)
        self.update_model_btn.setEnabled(enabled)
        self.transcribe_radio.setEnabled(enabled)

        current_model = self.model_dropdown.currentText()
        self.translate_radio.setEnabled(enabled and ModelMetadata.supports_translation(current_model))

        if not enabled and self.is_recording:
            self.is_recording = False
            self.record_button.setText("Start Recording")
            apply_recording_button_style(self.record_button, self.is_recording)

    def moveEvent(self, event):
        super().moveEvent(event)
        if self.clipboard_window.isVisible() and self.clipboard_window.is_docked():
            self.clipboard_window.reposition_to_host(self._host_rect_global(), gap=10)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.clipboard_window.isVisible() and self.clipboard_window.is_docked():
            self.clipboard_window.reposition_to_host(self._host_rect_global(), gap=10)

    def closeEvent(self, event):
        self._save_clipboard_setting(self._clipboard_target_visible)
        self.clipboard_window.hide()
        self.controller.stop_all_threads()
        logger.info("Application closing")
        super().closeEvent(event)
