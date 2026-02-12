from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot, QRect, QEvent, Signal, QSettings, QByteArray, QTimer
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
    QMessageBox,
    QFileDialog,
    QSizePolicy,
    QDialog,
    QDialogButtonBox,
    QSpinBox,
    QCheckBox,
    QMenuBar,
    QMenu,
    QProgressBar,
)

from core.quantization import CheckQuantizationSupport
from core.controller import TranscriberController
from core.models.metadata import ModelMetadata
from core.hotkeys import GlobalHotkey
from config.manager import config_manager
from gui.styles import update_button_property
from gui.clipboard_window import ClipboardSideWindow
from gui.settings_dialog import SettingsDialog
from gui.visualizations import WaveformButton
from core.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac", ".amr", ".asf", ".avi", ".flac", ".m4a",
    ".mkv", ".mp3", ".mp4", ".wav", ".webm", ".wma",
}

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

_DOCK_GAP = 10

_DEFAULT_MAIN_WIDTH = 350
_DEFAULT_MAIN_HEIGHT = 180
_DEFAULT_CLIPBOARD_WIDTH = 350
_DEFAULT_CLIPBOARD_HEIGHT = 180


class BatchSizeDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, default_value: int = 16):
        super().__init__(parent)
        self.setWindowTitle("Batch Size")
        self.setModal(True)
        self.setFixedWidth(350)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        layout.addWidget(
            QLabel(
                "Set batch size for file transcription.\n"
                "Use 1 for non-batched transcription."
            )
        )

        row = QHBoxLayout()
        row.setSpacing(5)
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

        self.settings = QSettings("FasterWhisperTranscriber", "Transcriber")

        self.loaded_model_settings = config_manager.get_model_settings()
        self.task_mode = config_manager.get_value("task_mode", "transcribe")
        self.controller = TranscriberController()
        self.supported_quantizations: dict[str, list[str]] = {"cpu": [], "cuda": []}
        self.is_recording = False
        self.cuda_available = cuda_available
        self._clipboard_visible = False
        self._toggleable_widgets: list[QWidget] = []
        self._model_is_loaded = False
        self._is_loading_model = False
        self._download_total_bytes = 0

        self.clipboard_window = ClipboardSideWindow(None, width=_DEFAULT_CLIPBOARD_WIDTH)
        self.clipboard_window.user_closed.connect(self._on_clipboard_closed)
        self.clipboard_window.docked_changed.connect(
            lambda d: logger.debug(f"Clipboard docked: {d}")
        )
        self.clipboard_window.always_on_top_changed.connect(
            self._on_clipboard_always_on_top_changed
        )

        self._build_ui()
        self._setup_connections()

        self._load_quantization_support()

        self.set_widgets_enabled(False)

        self._restore_state()

        self.hotkey_toggle_recording.connect(self._toggle_recording, Qt.QueuedConnection)
        self.global_hotkey = GlobalHotkey(lambda: self.hotkey_toggle_recording.emit())
        self.global_hotkey.start()

        self._sample_timer = QTimer(self)
        self._sample_timer.setInterval(50)
        self._sample_timer.timeout.connect(self._feed_audio_samples)

        self.setAcceptDrops(True)
        if app := QApplication.instance():
            app.installEventFilter(self)

        logger.info("MainWindow initialized")

    def _load_quantization_support(self) -> None:
        try:
            config = config_manager.load_config()
            self.supported_quantizations = config.get(
                "supported_quantizations", {"cpu": [], "cuda": []}
            )

            if not self.supported_quantizations.get("cpu") or (
                self.cuda_available
                and not self.supported_quantizations.get("cuda")
            ):
                CheckQuantizationSupport().update_supported_quantizations()
                config = config_manager.load_config()
                self.supported_quantizations = config.get(
                    "supported_quantizations", {"cpu": [], "cuda": []}
                )
        except Exception as e:
            logger.error(f"Failed to load precision support: {e}")
            self.supported_quantizations = {"cpu": ["float32"], "cuda": []}

    def _validate_model(self, model_name: str) -> str:
        valid_models = ModelMetadata.get_all_model_names()
        if model_name in valid_models:
            return model_name
        logger.warning(f"Invalid model '{model_name}', falling back to default")
        return self.DEFAULTS["model"]

    def _validate_device(self, device: str) -> str:
        if device == "cuda" and not self.cuda_available:
            logger.warning("CUDA no longer available, falling back to CPU")
            return "cpu"
        if device in ["cpu", "cuda"]:
            return device
        logger.warning(f"Invalid device '{device}', falling back to default")
        return self.DEFAULTS["device"]

    def _validate_quantization(
        self, quantization: str, model_name: str, device: str
    ) -> str:
        available = ModelMetadata.get_quantization_options(
            model_name, device, self.supported_quantizations
        )
        if quantization in available:
            return quantization
        if available:
            logger.warning(
                f"Precision '{quantization}' not available for "
                f"{model_name}/{device}, using {available[0]}"
            )
            return available[0]
        logger.warning(
            f"No quantizations available for {model_name}/{device}, using float32"
        )
        return "float32"

    def _validate_task_mode(self, task_mode: str, model_name: str) -> str:
        if task_mode == "translate" and not ModelMetadata.supports_translation(
            model_name
        ):
            logger.warning(
                f"Model '{model_name}' doesn't support translation, "
                f"falling back to transcribe"
            )
            return "transcribe"
        if task_mode in ["transcribe", "translate"]:
            return task_mode
        return self.DEFAULTS["task_mode"]

    def _restore_state(self) -> None:
        logger.info("Restoring application state from QSettings")

        geometry = self.settings.value(SETTINGS_GEOMETRY)
        if geometry and isinstance(geometry, QByteArray):
            if not self.restoreGeometry(geometry):
                logger.warning(
                    "Failed to restore main window geometry, using defaults"
                )
                self.resize(_DEFAULT_MAIN_WIDTH, _DEFAULT_MAIN_HEIGHT)
                self._center_on_screen()
        else:
            self.resize(_DEFAULT_MAIN_WIDTH, _DEFAULT_MAIN_HEIGHT)
            self._center_on_screen()

        saved_model = self.settings.value(SETTINGS_MODEL, self.DEFAULTS["model"])
        saved_device = self.settings.value(SETTINGS_DEVICE, self.DEFAULTS["device"])
        saved_quant = self.settings.value(
            SETTINGS_QUANTIZATION, self.DEFAULTS["quantization"]
        )
        saved_task = self.settings.value(
            SETTINGS_TASK_MODE, self.DEFAULTS["task_mode"]
        )

        model = self._validate_model(saved_model)
        device = self._validate_device(saved_device)
        quantization = self._validate_quantization(saved_quant, model, device)
        task_mode = self._validate_task_mode(saved_task, model)

        self.loaded_model_settings = {
            "model_name": model,
            "quantization_type": quantization,
            "device_type": device,
        }

        self.task_mode = task_mode
        self._update_translation_availability(model)
        self.task_combo.blockSignals(True)
        self.task_combo.setCurrentText(f"{task_mode.capitalize()} Mode")
        self.task_combo.blockSignals(False)

        append_mode = self.settings.value(
            SETTINGS_APPEND_MODE, self.DEFAULTS["append_mode"], type=bool
        )
        self.clipboard_window.set_append_mode(append_mode)

        always_on_top = self.settings.value(
            SETTINGS_CLIPBOARD_ALWAYS_ON_TOP,
            self.DEFAULTS["clipboard_always_on_top"],
            type=bool,
        )
        self.clipboard_window.set_always_on_top(always_on_top)

        clipboard_docked = self.settings.value(
            SETTINGS_CLIPBOARD_DOCKED, self.DEFAULTS["clipboard_docked"], type=bool
        )
        self._clipboard_visible = self.settings.value(
            SETTINGS_CLIPBOARD_VISIBLE,
            self.DEFAULTS["clipboard_visible"],
            type=bool,
        )

        if self._clipboard_visible:
            if clipboard_docked:
                self.clipboard_window.show_docked(
                    self._host_rect_global(), gap=_DOCK_GAP, animate=False
                )
            else:
                clip_geometry = self.settings.value(SETTINGS_CLIPBOARD_GEOMETRY)
                if clip_geometry and isinstance(clip_geometry, QByteArray):
                    self.clipboard_window.set_docked(False)
                    self.clipboard_window.restoreGeometry(clip_geometry)
                    self.clipboard_window.show()
                else:
                    self.clipboard_window.show_docked(
                        self._host_rect_global(), gap=_DOCK_GAP, animate=False
                    )

        self._update_clipboard_button_text()

        try:
            config_manager.set_model_settings(model, quantization, device)
            config_manager.set_value("task_mode", task_mode)
            config_manager.set_value("clipboard_append_mode", append_mode)
            config_manager.set_value("show_clipboard_window", self._clipboard_visible)
        except Exception as e:
            logger.warning(f"Failed to sync config manager: {e}")

        logger.info(
            f"State restored: model={model}, device={device}, "
            f"quant={quantization}, task={task_mode}"
        )

        self._update_model_status("No model loaded")
        self.controller.update_model(model, quantization, device)

    def _save_state(self) -> None:
        logger.info("Saving application state to QSettings")

        self.settings.setValue(SETTINGS_GEOMETRY, self.saveGeometry())

        self.settings.setValue(
            SETTINGS_MODEL,
            self.loaded_model_settings.get("model_name", self.DEFAULTS["model"]),
        )
        self.settings.setValue(
            SETTINGS_DEVICE,
            self.loaded_model_settings.get("device_type", self.DEFAULTS["device"]),
        )
        self.settings.setValue(
            SETTINGS_QUANTIZATION,
            self.loaded_model_settings.get(
                "quantization_type", self.DEFAULTS["quantization"]
            ),
        )
        self.settings.setValue(SETTINGS_TASK_MODE, self.task_mode)

        self.settings.setValue(
            SETTINGS_APPEND_MODE, self.clipboard_window.is_append_mode()
        )
        self.settings.setValue(SETTINGS_CLIPBOARD_VISIBLE, self._clipboard_visible)
        self.settings.setValue(
            SETTINGS_CLIPBOARD_ALWAYS_ON_TOP,
            self.clipboard_window.is_always_on_top(),
        )
        self.settings.setValue(
            SETTINGS_CLIPBOARD_DOCKED, self.clipboard_window.is_docked()
        )

        if not self.clipboard_window.is_docked():
            self.settings.setValue(
                SETTINGS_CLIPBOARD_GEOMETRY, self.clipboard_window.saveGeometry()
            )

        self.settings.sync()
        logger.debug("State saved successfully")

    def _center_on_screen(self) -> None:
        if screen := QApplication.primaryScreen():
            screen_geometry = screen.availableGeometry()
            x = (screen_geometry.width() - self.width()) // 2 + screen_geometry.x()
            y = (screen_geometry.height() - self.height()) // 2 + screen_geometry.y()
            self.move(x, y)

    def _on_clipboard_always_on_top_changed(self, value: bool) -> None:
        self.settings.setValue(SETTINGS_CLIPBOARD_ALWAYS_ON_TOP, value)

    def _setup_connections(self) -> None:
        self.clipboard_window.append_mode_changed.connect(
            self._on_append_mode_changed
        )

        self.controller.update_button_signal.connect(self._on_button_text_update)
        self.controller.enable_widgets_signal.connect(self.set_widgets_enabled)
        self.controller.text_ready_signal.connect(self._on_transcription_ready)
        self.controller.model_loaded_signal.connect(self._on_model_loaded_success)
        self.controller.error_occurred.connect(self._show_error_dialog)
        self.controller.transcription_cancelled_signal.connect(
            self._on_transcription_cancelled
        )

        self.controller.model_download_started.connect(
            self._on_model_download_started
        )
        self.controller.model_download_progress.connect(
            self._on_model_download_progress
        )
        self.controller.model_download_finished.connect(
            self._on_model_download_finished
        )
        self.controller.model_download_cancelled.connect(
            self._on_model_download_cancelled
        )
        self.controller.model_loading_started.connect(
            self._on_model_loading_started
        )

    def _build_ui(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        self.settings_action = file_menu.addAction("Settings")
        self.settings_action.triggered.connect(self._open_settings_dialog)
        self._toggleable_widgets.append(self.settings_action)

        central = QWidget(self)
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(5, 5, 5, 5)
        root.setSpacing(8)

        root.addWidget(self._build_main_group())
        root.addStretch(1)

        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setMinimumWidth(100)

        self.download_progress_bar = QProgressBar()
        self.download_progress_bar.setFixedWidth(130)
        self.download_progress_bar.setFixedHeight(14)
        self.download_progress_bar.setRange(0, 10000)
        self.download_progress_bar.setValue(0)
        self.download_progress_bar.setTextVisible(False)
        self.download_progress_bar.setVisible(False)

        self.cancel_download_button = QPushButton("Cancel")
        self.cancel_download_button.setFixedWidth(55)
        self.cancel_download_button.setFixedHeight(18)
        self.cancel_download_button.setVisible(False)
        self.cancel_download_button.clicked.connect(self._cancel_model_download)

        self.statusBar().addWidget(self.model_status_label, 1)
        self.statusBar().addPermanentWidget(self.download_progress_bar)
        self.statusBar().addPermanentWidget(self.cancel_download_button)
        self.statusBar().setSizeGripEnabled(True)

        self.resize(_DEFAULT_MAIN_WIDTH, _DEFAULT_MAIN_HEIGHT)
        self.setMinimumSize(280, 180)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)

    def _build_main_group(self) -> QGroupBox:
        group = QGroupBox("")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        left_col = QVBoxLayout()
        left_col.setSpacing(6)

        self.task_combo = QComboBox()
        self.task_combo.addItems(["Transcribe Mode", "Translate Mode"])
        self.task_combo.setToolTip("Choose transcription or translation mode")
        self.task_combo.currentTextChanged.connect(self._on_task_mode_changed)
        self._register_toggleable_widget(self.task_combo)
        left_col.addWidget(self.task_combo)

        self.transcribe_file_button = QPushButton("Transcribe File")
        self.transcribe_file_button.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        self.transcribe_file_button.setToolTip("Select an audio file to transcribe")
        self.transcribe_file_button.clicked.connect(
            self._select_and_transcribe_file
        )
        self._register_toggleable_widget(self.transcribe_file_button)
        left_col.addWidget(self.transcribe_file_button)

        self.clipboard_button = QPushButton("Show Clipboard")
        self.clipboard_button.setObjectName("clipboardButton")
        self.clipboard_button.setToolTip("Show or hide the clipboard panel")
        self.clipboard_button.clicked.connect(self._toggle_clipboard)
        self._register_toggleable_widget(self.clipboard_button)
        left_col.addWidget(self.clipboard_button)

        layout.addLayout(left_col)

        self.record_button = WaveformButton()
        self.record_button.setObjectName("recordButton")
        self.record_button.setText("Click to Record")
        self.record_button.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.record_button.setToolTip("Click or press F9 to toggle recording")
        self.record_button.clicked.connect(self._toggle_recording)
        self._register_toggleable_widget(self.record_button)
        layout.addWidget(self.record_button, 1)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding
        )
        self.cancel_button.setToolTip("Cancel the current transcription")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel_transcription)
        layout.addWidget(self.cancel_button)

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
        self.task_combo.blockSignals(True)
        current = self.task_combo.currentText()
        self.task_combo.clear()
        if can_translate:
            self.task_combo.addItems(["Transcribe Mode", "Translate Mode"])
            self.task_combo.setCurrentText(
                current
                if current in ["Transcribe Mode", "Translate Mode"]
                else "Transcribe Mode"
            )
        else:
            self.task_combo.addItems(["Transcribe Mode"])
            if current == "Translate Mode":
                self.task_mode = "transcribe"
                self._save_config("task_mode", self.task_mode)
                self.controller.set_task_mode(self.task_mode)
        self.task_combo.blockSignals(False)

    def _update_model_status(self, text: str) -> None:
        self.model_status_label.setText(text)

    def _show_current_model_status(self) -> None:
        if self._model_is_loaded:
            name = self.loaded_model_settings.get("model_name", "")
            quant = self.loaded_model_settings.get("quantization_type", "")
            device = self.loaded_model_settings.get("device_type", "")
            self._update_model_status(f"{name} ({quant} / {device})")
        else:
            self._update_model_status("No model loaded")

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        if num_bytes < 1024:
            return f"{num_bytes} B"
        elif num_bytes < 1024 * 1024:
            return f"{num_bytes / 1024:.1f} KB"
        elif num_bytes < 1024 * 1024 * 1024:
            return f"{num_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"

    @Slot(str, int)
    def _on_model_download_started(self, model_name: str, total_bytes: int) -> None:
        self._is_loading_model = True
        self._download_total_bytes = total_bytes
        self.download_progress_bar.setValue(0)
        self.download_progress_bar.setVisible(True)
        self.cancel_download_button.setVisible(True)
        total_str = self._format_bytes(total_bytes)
        self._update_model_status(f"Downloading {model_name}... 0 B / {total_str}")

    @Slot(int, int)
    def _on_model_download_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            pct = int(downloaded / total * 10000)
            self.download_progress_bar.setValue(pct)
        dl_str = self._format_bytes(downloaded)
        total_str = self._format_bytes(total)
        self._update_model_status(f"Downloading... {dl_str} / {total_str}")

    @Slot(str)
    def _on_model_download_finished(self, model_name: str) -> None:
        self.download_progress_bar.setVisible(False)
        self.cancel_download_button.setVisible(False)
        self._update_model_status(f"Loading {model_name}...")

    @Slot()
    def _on_model_download_cancelled(self) -> None:
        self._is_loading_model = False
        self.download_progress_bar.setVisible(False)
        self.cancel_download_button.setVisible(False)
        self._show_current_model_status()

    @Slot(str)
    def _on_model_loading_started(self, model_name: str) -> None:
        self._is_loading_model = True
        self.download_progress_bar.setVisible(False)
        self.cancel_download_button.setVisible(False)
        self._update_model_status(f"Loading {model_name}...")

    @Slot()
    def _cancel_model_download(self) -> None:
        self.cancel_download_button.setEnabled(False)
        self.controller.cancel_model_loading()

    @Slot()
    def _open_settings_dialog(self) -> None:
        dlg = SettingsDialog(
            parent=self,
            cuda_available=self.cuda_available,
            supported_quantizations=self.supported_quantizations,
            current_settings=self.loaded_model_settings,
        )
        dlg.model_update_requested.connect(self._on_settings_update_requested)
        dlg.reset_dimensions_requested.connect(self._reset_window_dimensions)
        dlg.exec()

    @Slot(str, str, str)
    def _on_settings_update_requested(
        self, model: str, quant: str, device: str
    ) -> None:
        self.controller.update_model(model, quant, device)

    @Slot()
    def _reset_window_dimensions(self) -> None:
        logger.info("Resetting window dimensions to defaults")

        self.settings.remove(SETTINGS_GEOMETRY)
        self.settings.remove(SETTINGS_CLIPBOARD_GEOMETRY)
        self.settings.remove(SETTINGS_CLIPBOARD_DOCKED)
        self.settings.sync()

        self.resize(_DEFAULT_MAIN_WIDTH, _DEFAULT_MAIN_HEIGHT)
        self._center_on_screen()

        self.clipboard_window.resize(
            _DEFAULT_CLIPBOARD_WIDTH, _DEFAULT_CLIPBOARD_HEIGHT
        )

        if self._clipboard_visible:
            self.clipboard_window.dock_to_host(
                self._host_rect_global(), gap=_DOCK_GAP, animate=True
            )
        else:
            self.clipboard_window.set_docked(True)

    @Slot(str)
    def _on_button_text_update(self, text: str) -> None:
        self.record_button.setText(text)

        if (
            "Transcribing" in text
            and "Done" not in text
            and "Failed" not in text
            and "Cancelled" not in text
        ):
            self.record_button.set_state(WaveformButton.TRANSCRIBING)
            self.cancel_button.setVisible(True)
            self.cancel_button.setEnabled(True)

    @Slot()
    def _on_clipboard_closed(self) -> None:
        self._clipboard_visible = False
        self._update_clipboard_button_text()
        self._save_config("show_clipboard_window", False)

    @Slot(bool)
    def _on_append_mode_changed(self, checked: bool) -> None:
        self._save_config("clipboard_append_mode", checked)

    @Slot(str, str)
    def _show_error_dialog(self, title: str, message: str) -> None:
        logger.error(f"Error: {title} - {message}")
        self._is_loading_model = False
        self.download_progress_bar.setVisible(False)
        self.cancel_download_button.setVisible(False)
        self.cancel_download_button.setEnabled(True)
        self._show_current_model_status()
        if "model" in title.lower():
            QMessageBox.critical(self, title, message)
        else:
            QMessageBox.warning(self, title, message)

    @Slot()
    def _on_task_mode_changed(self) -> None:
        self.task_mode = (
            "translate"
            if "Translate" in self.task_combo.currentText()
            else "transcribe"
        )
        self._save_config("task_mode", self.task_mode)
        self.controller.set_task_mode(self.task_mode)

    @Slot(str, str, str)
    def _on_model_loaded_success(
        self, model_name: str, quant: str, device: str
    ) -> None:
        self._is_loading_model = False
        self._model_is_loaded = True
        self.loaded_model_settings = {
            "model_name": model_name,
            "quantization_type": quant,
            "device_type": device,
        }
        self._update_translation_availability(model_name)
        self._show_current_model_status()
        self.download_progress_bar.setVisible(False)
        self.cancel_download_button.setVisible(False)
        self.cancel_download_button.setEnabled(True)

    @Slot()
    def _toggle_recording(self) -> None:
        if self.controller.is_transcribing():
            return

        if not self.record_button.isEnabled():
            return

        self.is_recording = not self.is_recording
        if self.is_recording:
            self.controller.start_recording()
            self.record_button.setText("Recording...")
            self.record_button.set_state(WaveformButton.RECORDING)
            self._sample_timer.start()
        else:
            self._sample_timer.stop()
            self.controller.stop_recording()
            self.record_button.setText("Processing...")
        update_button_property(self.record_button, "recording", self.is_recording)

    @Slot()
    def _feed_audio_samples(self) -> None:
        if self.is_recording:
            samples = self.controller.audio_manager.get_latest_samples()
            if samples is not None:
                self.record_button.update_waveform(samples)

    @Slot()
    def _cancel_transcription(self) -> None:
        self.controller.cancel_transcription()
        self.cancel_button.setEnabled(False)

    @Slot()
    def _on_transcription_cancelled(self) -> None:
        self.record_button.set_state(WaveformButton.IDLE)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setVisible(False)

    def _is_supported_audio_file(self, path: str) -> bool:
        try:
            return Path(path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        except Exception:
            return False

    def _transcribe_specific_file(self, file_path: str) -> None:
        if not file_path or not self._is_supported_audio_file(file_path):
            QMessageBox.warning(
                self, "Unsupported File", "Please select a supported audio file."
            )
            return

        if not self.transcribe_file_button.isEnabled():
            return

        dlg = BatchSizeDialog(self, default_value=16)
        if dlg.exec() == QDialog.Accepted:
            logger.info(
                f"Transcribing: {file_path} (batch={dlg.batch_size()})"
            )
            self.controller.transcribe_file(
                file_path, batch_size=dlg.batch_size()
            )

    @Slot()
    def _select_and_transcribe_file(self) -> None:
        exts = " ".join(f"*{ext}" for ext in SUPPORTED_AUDIO_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", f"Audio Files ({exts});;All Files (*)"
        )
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
                self.clipboard_window.show_docked(host_rect, gap=_DOCK_GAP)
            else:
                self.clipboard_window.show()
                self.clipboard_window.raise_()
        else:
            self.clipboard_window.hide_animated(host_rect, gap=_DOCK_GAP)

    def _update_clipboard_button_text(self) -> None:
        self.clipboard_button.setText(
            "Hide Clipboard" if self._clipboard_visible else "Show Clipboard"
        )

    def _register_toggleable_widget(self, widget: QWidget) -> None:
        self._toggleable_widgets.append(widget)

    @Slot(str)
    def _on_transcription_ready(self, text: str) -> None:
        self.record_button.set_state(WaveformButton.IDLE)

        self.clipboard_window.add_transcription(text)

        if app := QApplication.instance():
            clip_text = (
                self.clipboard_window.get_full_text()
                if self.clipboard_window.is_append_mode()
                else text
            )
            app.clipboard().setText(clip_text)

        if self.is_recording:
            self.is_recording = False
            update_button_property(self.record_button, "recording", False)

    @Slot(bool)
    def set_widgets_enabled(self, enabled: bool) -> None:
        for widget in self._toggleable_widgets:
            if hasattr(widget, "setEnabled"):
                widget.setEnabled(enabled)

        if enabled:
            self.cancel_button.setVisible(False)
            self.cancel_button.setEnabled(True)

        if not enabled and self.is_recording:
            self._sample_timer.stop()
            self.controller.stop_recording()
            self.is_recording = False
            update_button_property(self.record_button, "recording", False)

        if enabled and self.record_button.get_state() != WaveformButton.IDLE:
            self.record_button.set_state(WaveformButton.IDLE)

    def _extract_first_supported_drop(self, event) -> str | None:
        if not (md := event.mimeData()) or not md.hasUrls():
            return None
        for url in md.urls():
            if url.isLocalFile() and self._is_supported_audio_file(
                p := url.toLocalFile()
            ):
                return p
        return None

    def eventFilter(self, obj, event):
        if obj is not self and not (
            isinstance(obj, QWidget) and self.isAncestorOf(obj)
        ):
            return super().eventFilter(obj, event)

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
        if (
            self.clipboard_window.isVisible()
            and self.clipboard_window.is_docked()
        ):
            self.clipboard_window.reposition_to_host(host_rect, gap=_DOCK_GAP)

    def moveEvent(self, event):
        super().moveEvent(event)
        self._sync_clipboard_position()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_clipboard_position()

    def closeEvent(self, event):
        logger.info("Application closing")

        self._sample_timer.stop()
        self.record_button.stop()

        self.controller.audio_manager.audio_ready.disconnect()
        self.controller.transcription_service.transcription_error.disconnect()
        self.controller.error_occurred.disconnect()

        self._save_state()
        self._save_config("show_clipboard_window", self._clipboard_visible)
        self.clipboard_window.close()
        self.controller.stop_all_threads()

        if hasattr(self, "global_hotkey"):
            self.global_hotkey.stop()

        super().closeEvent(event)