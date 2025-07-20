# core/controller.py
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QApplication

from config.manager import config_manager
from core.models.manager import ModelManager
from core.audio.manager import AudioManager
from core.transcription.service import TranscriptionService

class TranscriberController(QObject):
    update_status_signal = Signal(str)
    enable_widgets_signal = Signal(bool)
    text_ready_signal = Signal(str)
    model_loaded_signal = Signal(str, str, str)

    def __init__(self, samplerate: int = 44_100, channels: int = 1, dtype: str = "int16"):
        super().__init__()

        self.model_manager = ModelManager()
        self.audio_manager = AudioManager(samplerate, channels, dtype)
        self.transcription_service = TranscriptionService(curate_text_enabled=True)

        self._connect_signals()

        self._load_settings()

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

    @Slot(str, str, str)
    def _on_model_loaded(self, name: str, quant: str, device: str) -> None:
        config_manager.set_model_settings(name, quant, device)
        self.update_status_signal.emit(f"Model {name} ready on {device}")
        self.enable_widgets_signal.emit(True)
        self.model_loaded_signal.emit(name, quant, device)

    @Slot(str)
    def _on_model_error(self, error: str) -> None:
        self.update_status_signal.emit(error)
        self.enable_widgets_signal.emit(True)

    @Slot(str)
    def _on_audio_ready(self, audio_file: str) -> None:
        model, expected_id = self.model_manager.get_model()
        if model and expected_id:
            self.transcription_service.transcribe_file(model, expected_id, audio_file)
        else:
            self.update_status_signal.emit("No model loaded")
            self.enable_widgets_signal.emit(True)

    @Slot(str)
    def _on_audio_error(self, error: str) -> None:
        self.update_status_signal.emit(error)
        self.enable_widgets_signal.emit(True)

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
        self.update_status_signal.emit(error)
        self.enable_widgets_signal.emit(True)

    def _load_settings(self) -> None:
        settings = config_manager.get_model_settings()
        self.update_model(
            settings["model_name"],
            settings["quantization_type"],
            settings["device_type"]
        )

    def stop_all_threads(self) -> None:
        self.audio_manager.cleanup()
        self.transcription_service.cleanup()
        self.model_manager.cleanup()