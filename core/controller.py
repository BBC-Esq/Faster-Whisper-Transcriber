from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QApplication

from config.manager import config_manager
from core.models.manager import ModelManager
from core.audio.manager import AudioManager
from core.audio.device_utils import get_optimal_audio_settings
from core.transcription.service import TranscriptionService
from core.logging_config import get_logger

logger = get_logger(__name__)


class TranscriberController(QObject):
    update_status_signal = Signal(str)
    enable_widgets_signal = Signal(bool)
    text_ready_signal = Signal(str)
    model_loaded_signal = Signal(str, str, str)
    error_occurred = Signal(str, str)

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        audio_manager: Optional[AudioManager] = None,
        transcription_service: Optional[TranscriptionService] = None,
    ):
        super().__init__()

        samplerate, channels, dtype = get_optimal_audio_settings()
        logger.info(f"Audio settings: {samplerate} Hz, {channels} ch, {dtype}")

        self.model_manager = model_manager or ModelManager()
        self.audio_manager = audio_manager or AudioManager(samplerate, channels, dtype)

        task_mode = config_manager.get_value("task_mode", "transcribe")
        curate_enabled = config_manager.get_value("curate_transcription", True)
        self.transcription_service = transcription_service or TranscriptionService(
            curate_text_enabled=curate_enabled,
            task_mode=task_mode
        )
        self.transcription_service.set_model_version_provider(self._get_current_model_version)

        self._connect_signals()
        self._load_settings()
        logger.info("TranscriberController initialized")

    def _get_current_model_version(self) -> str | None:
        _, version = self.model_manager.get_model()
        return version

    def set_task_mode(self, mode: str) -> None:
        self.transcription_service.set_task_mode(mode)
        self.update_status_signal.emit(f"Mode: {mode.capitalize()}")

    def _connect_signals(self) -> None:
        self.model_manager.model_loaded.connect(self._on_model_loaded)
        self.model_manager.model_error.connect(self._on_model_error)

        self.audio_manager.recording_started.connect(
            lambda: self.update_status_signal.emit("Recording...")
        )
        self.audio_manager.audio_ready.connect(self._on_audio_ready)
        self.audio_manager.audio_error.connect(self._on_audio_error)

        self.transcription_service.transcription_started.connect(
            lambda: self.update_status_signal.emit("Transcribing...")
        )
        self.transcription_service.transcription_progress.connect(self._on_transcription_progress)
        self.transcription_service.transcription_completed.connect(self._on_transcription_completed)
        self.transcription_service.transcription_error.connect(self._on_transcription_error)

    def update_model(self, model_name: str, quant: str, device: str) -> None:
        self.enable_widgets_signal.emit(False)
        self.update_status_signal.emit(f"Loading model {model_name}...")
        self.model_manager.load_model(model_name, quant, device)

    def start_recording(self) -> None:
        if not self.audio_manager.start_recording():
            self.update_status_signal.emit("Already recording")

    def stop_recording(self) -> None:
        self.audio_manager.stop_recording()

    def transcribe_file(self, file_path: str, batch_size: int | None = None) -> None:
        model, model_version = self.model_manager.get_model()
        if model and model_version:
            self.enable_widgets_signal.emit(False)
            self.update_status_signal.emit(f"Transcribing {Path(file_path).name}...")
            self.transcription_service.transcribe_file(
                model,
                model_version,
                file_path,
                is_temp_file=False,
                batch_size=batch_size,
            )
        else:
            self.update_status_signal.emit("No model loaded")
            self.error_occurred.emit("Transcription Error", "No model is loaded to process audio")

    @Slot(str, str, str)
    def _on_model_loaded(self, name: str, quant: str, device: str) -> None:
        try:
            config_manager.set_model_settings(name, quant, device)
        except Exception as e:
            logger.warning(f"Failed to save model settings: {e}")

        self.update_status_signal.emit(f"Model {name} ready on {device}")
        self.enable_widgets_signal.emit(True)
        self.model_loaded_signal.emit(name, quant, device)

    @Slot(str)
    def _on_model_error(self, error: str) -> None:
        logger.error(f"Model error: {error}")
        self.update_status_signal.emit("Model load failed")
        self.enable_widgets_signal.emit(True)
        self.error_occurred.emit("Model Error", error)

    @Slot(str)
    def _on_audio_ready(self, audio_file: str) -> None:
        model, model_version = self.model_manager.get_model()
        if model and model_version:
            self.transcription_service.transcribe_file(
                model,
                model_version,
                audio_file,
                is_temp_file=True,
                batch_size=None,
            )
        else:
            self.update_status_signal.emit("No model loaded")
            self.enable_widgets_signal.emit(True)
            self.error_occurred.emit("Audio Error", "No model is loaded to process audio")

    @Slot(str)
    def _on_audio_error(self, error: str) -> None:
        logger.error(f"Audio error: {error}")
        self.update_status_signal.emit("Recording failed")
        self.enable_widgets_signal.emit(True)
        self.error_occurred.emit("Audio Error", error)

    @Slot(int, int, float)
    def _on_transcription_progress(self, segment_num: int, total_segments: int, percent: float) -> None:
        if percent >= 0:
            self.update_status_signal.emit(f"Transcribing... {percent:.0f}% (segment {segment_num})")
        else:
            self.update_status_signal.emit(f"Transcribing... segment {segment_num}")

    @Slot(str)
    def _on_transcription_completed(self, text: str) -> None:
        app = QApplication.instance()
        if app:
            app.clipboard().setText(text)

        self.text_ready_signal.emit(text)
        self.update_status_signal.emit("Done")
        self.enable_widgets_signal.emit(True)

    @Slot(str)
    def _on_transcription_error(self, error: str) -> None:
        logger.error(f"Transcription error: {error}")
        self.update_status_signal.emit("Transcription failed")
        self.enable_widgets_signal.emit(True)
        self.error_occurred.emit("Transcription Error", error)

    def _load_settings(self) -> None:
        settings = config_manager.get_model_settings()
        self.update_model(
            settings["model_name"],
            settings["quantization_type"],
            settings["device_type"]
        )

    def stop_all_threads(self) -> None:
        logger.info("Stopping all threads")
        self.audio_manager.cleanup()
        self.transcription_service.cleanup()
        self.model_manager.cleanup()
        logger.info("All threads stopped")
