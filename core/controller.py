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
    update_button_signal = Signal(str)
    enable_widgets_signal = Signal(bool)
    text_ready_signal = Signal(str)
    result_ready_signal = Signal(object)
    model_loaded_signal = Signal(str, str, str)
    error_occurred = Signal(str, str)
    transcription_cancelled_signal = Signal()
    model_download_started = Signal(str, object)
    model_download_progress = Signal(object, object)
    model_download_finished = Signal(str)
    model_download_cancelled = Signal()
    model_loading_started = Signal(str)

    batch_progress = Signal(int, int, str)
    batch_completed = Signal(str)
    batch_error = Signal(str)

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        audio_manager: Optional[AudioManager] = None,
        transcription_service: Optional[TranscriptionService] = None,
        audio_device_id: int | None = None,
    ):
        super().__init__()

        samplerate, channels, dtype = get_optimal_audio_settings(audio_device_id)
        logger.info(f"Audio settings: {samplerate} Hz, {channels} ch, {dtype}, device={audio_device_id}")

        self.model_manager = model_manager or ModelManager()
        self.audio_manager = audio_manager or AudioManager(samplerate, channels, dtype, device_id=audio_device_id)

        task_mode = config_manager.get_value("task_mode", "transcribe")
        curate_enabled = config_manager.get_value("curate_transcription", True)
        self.transcription_service = transcription_service or TranscriptionService(
            curate_text_enabled=curate_enabled,
            task_mode=task_mode,
        )
        self.transcription_service.set_model_version_provider(
            self._get_current_model_version
        )
        self._load_whisper_params()

        self._batch_processor = None

        self._connect_signals()
        logger.info("TranscriberController initialized")

    def _get_current_model_version(self) -> str | None:
        _, version = self.model_manager.get_model()
        return version

    def _load_whisper_params(self) -> None:
        include_timestamps = config_manager.get_value("include_timestamps", False)
        params = {
            "without_timestamps": not include_timestamps,
            "word_timestamps": False,
            "beam_size": config_manager.get_value("beam_size", 5),
            "vad_filter": config_manager.get_value("vad_filter", False),
            "condition_on_previous_text": config_manager.get_value("condition_on_previous_text", True),
        }
        self.transcription_service.set_whisper_params(params)

    def set_whisper_params(self, params: dict) -> None:
        self.transcription_service.set_whisper_params(params)

    def set_task_mode(self, mode: str) -> None:
        self.transcription_service.set_task_mode(mode)

    def set_audio_device(self, device_id: int | None) -> None:
        samplerate, channels, dtype = get_optimal_audio_settings(device_id)
        self.audio_manager.set_device(device_id, samplerate, channels, dtype)
        logger.info(f"Audio device updated: id={device_id}, {samplerate} Hz, {channels} ch")

    def _connect_signals(self) -> None:
        self.model_manager.model_loaded.connect(self._on_model_loaded)
        self.model_manager.model_error.connect(self._on_model_error)
        self.model_manager.download_started.connect(self._on_download_started)
        self.model_manager.download_progress.connect(self._on_download_progress)
        self.model_manager.download_finished.connect(self._on_download_finished)
        self.model_manager.download_cancelled.connect(self._on_download_cancelled)
        self.model_manager.loading_started.connect(self._on_loading_started)

        self.audio_manager.recording_started.connect(
            lambda: self.update_button_signal.emit("Recording...")
        )
        self.audio_manager.audio_ready.connect(self._on_audio_ready)
        self.audio_manager.audio_error.connect(self._on_audio_error)

        self.transcription_service.transcription_started.connect(
            lambda: self.update_button_signal.emit("Transcribing...")
        )
        self.transcription_service.transcription_progress.connect(
            self._on_transcription_progress
        )
        self.transcription_service.transcription_completed.connect(
            self._on_transcription_completed
        )
        self.transcription_service.transcription_completed_with_result.connect(
            self._on_transcription_completed_with_result
        )
        self.transcription_service.transcription_error.connect(
            self._on_transcription_error
        )
        self.transcription_service.transcription_cancelled.connect(
            self._on_transcription_cancelled
        )

    def update_model(self, model_name: str, quant: str, device: str) -> None:
        self.enable_widgets_signal.emit(False)
        self.model_manager.load_model(model_name, quant, device)

    def cancel_model_loading(self) -> None:
        self.model_manager.cancel_loading()

    def start_recording(self) -> bool:
        if not self.audio_manager.start_recording():
            self.update_button_signal.emit("Already recording")
            return False
        return True

    def stop_recording(self) -> None:
        self.audio_manager.stop_recording()

    def cancel_transcription(self) -> bool:
        if self.transcription_service.cancel_transcription():
            self.update_button_signal.emit("Cancelling...")
            return True
        return False

    def is_transcribing(self) -> bool:
        return self.transcription_service.is_transcribing()

    def transcribe_file(self, file_path: str, batch_size: int | None = None) -> None:
        model, model_version = self.model_manager.get_model()
        if model and model_version:
            self.enable_widgets_signal.emit(False)
            self.update_button_signal.emit(
                f"Transcribing {Path(file_path).name}..."
            )
            self.transcription_service.transcribe_file(
                model,
                model_version,
                file_path,
                is_temp_file=False,
                batch_size=batch_size,
            )
        else:
            self.error_occurred.emit(
                "Transcription Error", "No model is loaded to process audio"
            )

    # --- Batch processing ---

    def start_batch_processing(self, files, output_format, output_directory,
                                batch_size, task_mode, whisper_params) -> None:
        from core.transcription.batch_processor import BatchProcessor

        model, model_version = self.model_manager.get_model()
        if not model or not model_version:
            self.batch_error.emit("No model is loaded to process audio")
            return

        self._batch_processor = BatchProcessor(
            files=files,
            model=model,
            output_format=output_format,
            output_directory=output_directory,
            batch_size=batch_size,
            task_mode=task_mode,
            whisper_params=whisper_params,
        )
        self._batch_processor.progress.connect(self._on_batch_progress)
        self._batch_processor.finished.connect(self._on_batch_completed)
        self._batch_processor.error.connect(self._on_batch_error)
        self._batch_processor.start()

    def stop_batch_processing(self) -> None:
        if self._batch_processor:
            self._batch_processor.request_stop()

    def is_batch_processing(self) -> bool:
        return self._batch_processor is not None and self._batch_processor.isRunning()

    @Slot(int, int, str)
    def _on_batch_progress(self, current: int, total: int, message: str) -> None:
        self.batch_progress.emit(current, total, message)

    @Slot(str)
    def _on_batch_completed(self, message: str) -> None:
        self._batch_processor = None
        self.batch_completed.emit(message)

    @Slot(str)
    def _on_batch_error(self, message: str) -> None:
        self.batch_error.emit(message)

    # --- Model signals ---

    @Slot(str, object)
    def _on_download_started(self, model_name: str, total_bytes: int) -> None:
        self.model_download_started.emit(model_name, total_bytes)

    @Slot(object, object)
    def _on_download_progress(self, downloaded: int, total: int) -> None:
        self.model_download_progress.emit(downloaded, total)

    @Slot(str)
    def _on_download_finished(self, model_name: str) -> None:
        self.model_download_finished.emit(model_name)

    @Slot()
    def _on_download_cancelled(self) -> None:
        self.enable_widgets_signal.emit(True)
        self.model_download_cancelled.emit()

    @Slot(str)
    def _on_loading_started(self, model_name: str) -> None:
        self.model_loading_started.emit(model_name)

    @Slot(str, str, str)
    def _on_model_loaded(self, name: str, quant: str, device: str) -> None:
        try:
            config_manager.set_model_settings(name, quant, device)
        except Exception as e:
            logger.warning(f"Failed to save model settings: {e}")

        self.enable_widgets_signal.emit(True)
        self.model_loaded_signal.emit(name, quant, device)

    @Slot(str)
    def _on_model_error(self, error: str) -> None:
        logger.error(f"Model error: {error}")
        self.enable_widgets_signal.emit(True)
        self.error_occurred.emit("Model Error", error)

    # --- Audio signals ---

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
            self.enable_widgets_signal.emit(True)
            self.error_occurred.emit(
                "Audio Error", "No model is loaded to process audio"
            )

    @Slot(str)
    def _on_audio_error(self, error: str) -> None:
        logger.error(f"Audio error: {error}")
        self.enable_widgets_signal.emit(True)
        self.error_occurred.emit("Audio Error", error)

    # --- Transcription signals ---

    @Slot(int, int, float)
    def _on_transcription_progress(
        self, segment_num: int, total_segments: int, percent: float
    ) -> None:
        if percent >= 0:
            self.update_button_signal.emit(f"Transcribing... {percent:.0f}%")
        else:
            self.update_button_signal.emit(f"Transcribing... segment {segment_num}")

    @Slot(str)
    def _on_transcription_completed(self, text: str) -> None:
        self.text_ready_signal.emit(text)
        self.update_button_signal.emit("Click to Record")
        self.enable_widgets_signal.emit(True)

    @Slot(object)
    def _on_transcription_completed_with_result(self, result: object) -> None:
        self.result_ready_signal.emit(result)

    @Slot(str)
    def _on_transcription_error(self, error: str) -> None:
        logger.error(f"Transcription error: {error}")
        self.update_button_signal.emit("Transcription Failed---Click to Record")
        self.enable_widgets_signal.emit(True)
        self.error_occurred.emit("Transcription Error", error)

    @Slot()
    def _on_transcription_cancelled(self) -> None:
        self.update_button_signal.emit("Click to Record")
        self.enable_widgets_signal.emit(True)
        self.transcription_cancelled_signal.emit()

    def stop_all_threads(self) -> None:
        import time as _time
        logger.info("Stopping all threads")

        if self._batch_processor and self._batch_processor.isRunning():
            _t = _time.perf_counter()
            self._batch_processor.request_stop()
            self._batch_processor.wait(5000)
            logger.info(f"[SHUTDOWN] batch_processor.stop(): {_time.perf_counter() - _t:.3f}s")

        _t = _time.perf_counter()
        self.model_manager.cancel_loading()
        logger.info(f"[SHUTDOWN] model_manager.cancel_loading(): {_time.perf_counter() - _t:.3f}s")

        _t = _time.perf_counter()
        self.transcription_service.cancel_transcription()
        logger.info(f"[SHUTDOWN] transcription_service.cancel_transcription(): {_time.perf_counter() - _t:.3f}s")

        _t = _time.perf_counter()
        self.audio_manager.cleanup()
        logger.info(f"[SHUTDOWN] audio_manager.cleanup(): {_time.perf_counter() - _t:.3f}s")

        _t = _time.perf_counter()
        self.transcription_service.cleanup()
        logger.info(f"[SHUTDOWN] transcription_service.cleanup(): {_time.perf_counter() - _t:.3f}s")

        _t = _time.perf_counter()
        self.model_manager.cleanup()
        logger.info(f"[SHUTDOWN] model_manager.cleanup(): {_time.perf_counter() - _t:.3f}s")

        logger.info("All threads stopped")
