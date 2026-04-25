from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
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
    QLabel,
    QSpinBox,
    QLineEdit,
    QMessageBox,
    QApplication,
    QDialogButtonBox,
)

from core.models.metadata import ModelMetadata
from core.audio.device_utils import get_input_devices
from gui.styles import update_button_property
from gui.file_panel import FileTypesDialog, SUPPORTED_AUDIO_EXTENSIONS, ToggleSwitch
from core.logging_config import get_logger

logger = get_logger(__name__)


class SpeakerLabelsDialog(QDialog):
    def __init__(self, parent: QWidget | None, labels: list[str]):
        super().__init__(parent)
        self.setWindowTitle("Speaker Labels")
        self.setModal(True)
        self.setMinimumWidth(400)
        self._defaults = ["Lawyer", "Client"]
        self._labels = list(labels or self._defaults)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._count_spin = QSpinBox()
        self._count_spin.setRange(2, 8)
        self._count_spin.setValue(max(2, min(8, len(self._labels))))
        self._count_spin.valueChanged.connect(self._refresh_visible_rows)

        form = QFormLayout()
        form.addRow("Expected Speakers", self._count_spin)
        self._rows: list[tuple[QLabel, QLineEdit]] = []
        for idx in range(8):
            default = self._default_label(idx)
            value = self._labels[idx] if idx < len(self._labels) else default
            label = QLabel(f"Speaker {idx + 1}")
            edit = QLineEdit(value)
            form.addRow(label, edit)
            self._rows.append((label, edit))
        layout.addLayout(form)
        self._refresh_visible_rows()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_labels(self) -> list[str]:
        labels = []
        for idx, (_, edit) in enumerate(self._rows[: self._count_spin.value()]):
            labels.append(edit.text().strip() or self._default_label(idx))
        return labels

    def _refresh_visible_rows(self) -> None:
        count = self._count_spin.value()
        for idx, (label, edit) in enumerate(self._rows):
            visible = idx < count
            label.setVisible(visible)
            edit.setVisible(visible)

    def _default_label(self, idx: int) -> str:
        if idx < len(self._defaults):
            return self._defaults[idx]
        return f"Speaker {idx + 1}"


class VoiceEnrollmentDialog(QDialog):
    _PROFILE_KEY = "speaker_1"

    def __init__(
        self,
        parent: QWidget | None,
        labels: list[str],
        profiles: dict,
        device_provider,
    ):
        super().__init__(parent)
        self.setWindowTitle("Voice Enrollment")
        self.setModal(True)
        self.setMinimumWidth(460)

        self._labels = labels
        self._profiles = {
            key: list(value)
            for key, value in (profiles or {}).items()
            if key == self._PROFILE_KEY and isinstance(value, list)
        }
        self._device_provider = device_provider

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        hint = QLabel(
            "Record 5-6 seconds of your voice. The app uses this only when it is a confident match."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        row = QHBoxLayout()
        record_btn = QPushButton("Record")
        record_btn.clicked.connect(self._record_profile)
        row.addWidget(record_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_profile)
        row.addWidget(clear_btn)

        self._status_label = QLabel()
        row.addWidget(self._status_label, 1)
        layout.addLayout(row)

        self._refresh_status()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def get_profiles(self) -> dict:
        return dict(self._profiles)

    def _refresh_status(self) -> None:
        self._status_label.setText(
            "Enrolled" if self._PROFILE_KEY in self._profiles else "No sample"
        )

    def _clear_profile(self) -> None:
        self._profiles.pop(self._PROFILE_KEY, None)
        self._refresh_status()

    def _record_profile(self) -> None:
        QMessageBox.information(
            self,
            "Voice Enrollment",
            "After you click OK, speak naturally for about 6 seconds.",
        )

        try:
            import numpy as np
            import sounddevice as sd
            from core.transcription.client_call import build_voice_profile_from_samples

            device_id = self._device_provider()
            sample_rate = self._choose_sample_rate(sd, device_id)
            frames = int(sample_rate * 6)

            QApplication.setOverrideCursor(Qt.WaitCursor)
            cursor_set = True
            QApplication.processEvents()
            recording = sd.rec(
                frames,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=device_id,
            )
            sd.wait()
            samples = np.asarray(recording).reshape(-1)
            self._profiles[self._PROFILE_KEY] = build_voice_profile_from_samples(
                samples, sample_rate
            )
            self._refresh_status()
            QMessageBox.information(self, "Voice Enrollment", "Saved your voice sample.")
        except Exception as e:
            logger.exception("Voice enrollment failed")
            QMessageBox.warning(
                self, "Voice Enrollment", f"Could not save voice sample:\n{e}"
            )
        finally:
            if "cursor_set" in locals():
                QApplication.restoreOverrideCursor()

    @staticmethod
    def _choose_sample_rate(sd, device_id: int | None) -> int:
        for sample_rate in (16000, 44100, 48000):
            try:
                sd.check_input_settings(
                    device=device_id,
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                )
                return sample_rate
            except Exception:
                continue
        return 44100


class SettingsDialog(QDialog):
    model_update_requested = Signal(str, str, str)
    audio_device_changed = Signal(str, str)
    task_mode_changed = Signal(str)
    whisper_settings_changed = Signal(object)
    speaker_settings_changed = Signal(object)
    server_mode_changed = Signal(bool, int)

    file_types_changed = Signal(object)

    def __init__(
        self,
        parent: QWidget | None,
        cuda_available: bool,
        supported_quantizations: dict[str, list[str]],
        current_settings: dict[str, str],
        current_task_mode: str = "transcribe",
        current_audio_device: dict[str, str] | None = None,
        current_whisper_settings: dict | None = None,
        current_speaker_labels: list[str] | None = None,
        current_speaker_profiles: dict | None = None,
        current_ext_checked: dict[str, bool] | None = None,
        current_server_settings: dict | None = None,
        is_busy_check=None,
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
        self.current_speaker_labels = list(
            current_speaker_labels or ["Lawyer", "Client"]
        )
        while len(self.current_speaker_labels) < 2:
            self.current_speaker_labels.append(
                ["Lawyer", "Client"][len(self.current_speaker_labels)]
            )
        self.current_speaker_profiles = dict(current_speaker_profiles or {})
        self.current_server_settings = current_server_settings or {
            "server_mode_enabled": False,
            "server_port": 8765,
        }
        self._is_busy_check = is_busy_check or (lambda: False)
        self._ext_checked = current_ext_checked or {
            ext: True for ext in SUPPORTED_AUDIO_EXTENSIONS
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

        model_group = QGroupBox("Speech Model")
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

        self.include_timestamps_cb = QCheckBox()
        self.include_timestamps_cb.setToolTip(
            "Include segment timestamps in output (always enabled for SRT/VTT)"
        )
        whisper_form.addRow("Include Timestamps", self.include_timestamps_cb)

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

        speaker_group = QGroupBox("Speaker Labels")
        speaker_layout = QHBoxLayout(speaker_group)
        speaker_layout.setSpacing(8)

        self._speaker_labels_btn = QPushButton("Speaker Labels...")
        self._speaker_labels_btn.setFixedHeight(28)
        self._speaker_labels_btn.clicked.connect(self._open_speaker_labels_dialog)
        speaker_layout.addWidget(self._speaker_labels_btn)

        self._voice_enrollment_btn = QPushButton("Voice Enrollment...")
        self._voice_enrollment_btn.setFixedHeight(28)
        self._voice_enrollment_btn.clicked.connect(
            self._open_voice_enrollment_dialog
        )
        speaker_layout.addWidget(self._voice_enrollment_btn)

        right_column.addWidget(speaker_group)

        server_group = QGroupBox("Server Mode")
        server_vbox = QVBoxLayout(server_group)
        server_vbox.setContentsMargins(12, 10, 12, 10)
        server_vbox.setSpacing(8)

        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(6)
        self._server_off_label = QLabel("Off")
        self._server_off_label.setStyleSheet("font-size: 11px;")
        toggle_row.addWidget(self._server_off_label)
        self.server_mode_toggle = ToggleSwitch()
        toggle_row.addWidget(self.server_mode_toggle)
        self._server_on_label = QLabel("On")
        self._server_on_label.setStyleSheet("font-size: 11px;")
        toggle_row.addWidget(self._server_on_label)
        toggle_row.addSpacing(16)
        toggle_row.addWidget(QLabel("Port:"))
        self.server_port_spin = QSpinBox()
        self.server_port_spin.setRange(1024, 65535)
        self.server_port_spin.setToolTip(
            "TCP port for the HTTP API (default 8765)."
        )
        toggle_row.addWidget(self.server_port_spin)
        toggle_row.addStretch(1)
        server_vbox.addLayout(toggle_row)

        server_hint = QLabel(
            "<qt>When On, an HTTP API is exposed at <code>http://0.0.0.0:&lt;port&gt;</code>. "
            "Endpoints: <code>/transcribe</code>, <code>/transcribe/raw</code>, "
            "<code>/models</code>, <code>/status</code>, <code>/health</code>.</qt>"
        )
        server_hint.setWordWrap(True)
        server_hint.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        server_vbox.addWidget(server_hint)

        right_column.addWidget(server_group)

        file_types_row = QHBoxLayout()
        file_types_row.addStretch(1)
        self._file_types_btn = QPushButton("File Types...")
        self._file_types_btn.setFixedHeight(28)
        self._file_types_btn.setFixedWidth(100)
        self._file_types_btn.setToolTip("Configure which audio/video file types to include in batch processing")
        self._file_types_btn.clicked.connect(self._open_file_types_dialog)
        file_types_row.addWidget(self._file_types_btn)

        self._guide_btn = QPushButton("Guide")
        self._guide_btn.setFixedHeight(28)
        self._guide_btn.setFixedWidth(100)
        self._guide_btn.setToolTip(
            "<qt>Open the HTML user guide for the<br>"
            "Server API in your default browser.</qt>"
        )
        self._guide_btn.clicked.connect(self._open_server_guide)
        file_types_row.addWidget(self._guide_btn)

        right_column.addLayout(file_types_row)

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
        self.include_timestamps_cb.toggled.connect(self._check_for_changes)
        self.beam_size_spin.valueChanged.connect(self._check_for_changes)
        self.vad_filter_cb.toggled.connect(self._check_for_changes)
        self.condition_on_previous_cb.toggled.connect(self._check_for_changes)
        self.server_mode_toggle.toggled.connect(self._check_for_changes)
        self.server_port_spin.valueChanged.connect(self._check_for_changes)

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

        self.include_timestamps_cb.setChecked(
            self.current_whisper_settings.get("include_timestamps", False)
        )
        self.beam_size_spin.setValue(
            self.current_whisper_settings.get("beam_size", 5)
        )
        self.vad_filter_cb.setChecked(
            self.current_whisper_settings.get("vad_filter", True)
        )
        self.condition_on_previous_cb.setChecked(
            self.current_whisper_settings.get("condition_on_previous_text", False)
        )

        self.server_mode_toggle.blockSignals(True)
        self.server_mode_toggle.setChecked(
            bool(self.current_server_settings.get("server_mode_enabled", False))
        )
        self.server_mode_toggle.blockSignals(False)
        self.server_port_spin.blockSignals(True)
        self.server_port_spin.setValue(
            int(self.current_server_settings.get("server_port", 8765))
        )
        self.server_port_spin.blockSignals(False)
        self._apply_server_mode_lock()

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
            "include_timestamps": self.include_timestamps_cb.isChecked(),
            "beam_size": self.beam_size_spin.value(),
            "vad_filter": self.vad_filter_cb.isChecked(),
            "condition_on_previous_text": self.condition_on_previous_cb.isChecked(),
        }
        return current != self.current_whisper_settings

    def _server_settings_selection_changed(self) -> bool:
        current = {
            "server_mode_enabled": self.server_mode_toggle.isChecked(),
            "server_port": self.server_port_spin.value(),
        }
        return current != self.current_server_settings

    def _apply_server_mode_lock(self) -> None:
        server_on = bool(self.current_server_settings.get("server_mode_enabled", False))
        locked_widgets = [
            self.model_dropdown,
            self.device_dropdown,
            self.quantization_dropdown,
            self.task_dropdown,
            self.audio_device_dropdown,
            self.include_timestamps_cb,
            self.beam_size_spin,
            self.vad_filter_cb,
            self.condition_on_previous_cb,
            self._speaker_labels_btn,
            self._voice_enrollment_btn,
        ]
        for w in locked_widgets:
            w.setEnabled(not server_on)

    def _check_for_changes(self) -> None:
        model_changed = self._model_settings_changed()
        task_changed = self._task_mode_selection_changed()
        audio_changed = self._audio_device_selection_changed()
        whisper_changed = self._whisper_settings_selection_changed()
        server_changed = self._server_settings_selection_changed()
        has_changes = (
            model_changed or task_changed or audio_changed
            or whisper_changed or server_changed
        )
        self.update_btn.setEnabled(has_changes)
        if model_changed:
            self.update_btn.setText("Reload Model")
        else:
            self.update_btn.setText("Update Settings")
        update_button_property(self.update_btn, "changed", has_changes)

    def _open_speaker_labels_dialog(self) -> None:
        dlg = SpeakerLabelsDialog(self, self.current_speaker_labels)
        if dlg.exec() == QDialog.Accepted:
            self.current_speaker_labels = dlg.get_labels()
            self.speaker_settings_changed.emit({
                "speaker_labels": self.current_speaker_labels,
                "speaker_voice_profiles": self.current_speaker_profiles,
            })

    def _open_voice_enrollment_dialog(self) -> None:
        dlg = VoiceEnrollmentDialog(
            self,
            self.current_speaker_labels,
            self.current_speaker_profiles,
            self._selected_audio_device_id,
        )
        if dlg.exec() == QDialog.Accepted:
            self.current_speaker_profiles = dlg.get_profiles()
            self.speaker_settings_changed.emit({
                "speaker_labels": self.current_speaker_labels,
                "speaker_voice_profiles": self.current_speaker_profiles,
            })

    def _selected_audio_device_id(self) -> int | None:
        data = self.audio_device_dropdown.currentData()
        if isinstance(data, dict):
            return data.get("id")
        return None

    def _open_file_types_dialog(self) -> None:
        dlg = FileTypesDialog(self, self._ext_checked)
        if dlg.exec() == QDialog.Accepted:
            self._ext_checked = dlg.get_checked()
            self.file_types_changed.emit(self._ext_checked)

    def _open_server_guide(self) -> None:
        guide_path = Path(__file__).parent.parent / "guides" / "SERVER_API_GUIDE.html"
        if not guide_path.is_file():
            logger.warning(f"Guide not found at {guide_path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(guide_path.resolve())))

    def _on_update_clicked(self) -> None:
        if self._server_settings_selection_changed():
            wants_server_on = self.server_mode_toggle.isChecked()
            currently_on = bool(
                self.current_server_settings.get("server_mode_enabled", False)
            )
            if wants_server_on and not currently_on and self._is_busy_check():
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Busy",
                    "A transcription or batch job is currently running. "
                    "Wait for it to finish before turning Server Mode on.",
                )
                self.server_mode_toggle.blockSignals(True)
                self.server_mode_toggle.setChecked(False)
                self.server_mode_toggle.blockSignals(False)

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
                "include_timestamps": self.include_timestamps_cb.isChecked(),
                "beam_size": self.beam_size_spin.value(),
                "vad_filter": self.vad_filter_cb.isChecked(),
                "condition_on_previous_text": self.condition_on_previous_cb.isChecked(),
            }
            self.whisper_settings_changed.emit(settings)

        if self._server_settings_selection_changed():
            self.server_mode_changed.emit(
                self.server_mode_toggle.isChecked(),
                self.server_port_spin.value(),
            )

        self.accept()
