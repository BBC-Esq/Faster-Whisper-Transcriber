from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

from core.audio.recording import RecordingThread
from core.temp_file_manager import temp_file_manager
from core.logging_config import get_logger

logger = get_logger(__name__)


class AudioManager(QObject):
    recording_started = Signal()
    recording_stopped = Signal()
    audio_ready = Signal(str)
    audio_error = Signal(str)

    def __init__(self, samplerate: int = 44_100, channels: int = 1, dtype: str = "int16"):
        super().__init__()
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self._recording_thread: Optional[RecordingThread] = None
        self._current_temp_file: Optional[str] = None

    def start_recording(self) -> bool:
        if self._recording_thread and self._recording_thread.isRunning():
            logger.warning("Attempted to start recording while already recording")
            return False

        try:
            path = temp_file_manager.create_temp_wav()
            self._current_temp_file = str(path)

            self._recording_thread = RecordingThread(
                output_path=path,
                samplerate=self.samplerate,
                channels=self.channels,
                dtype=self.dtype,
            )
            self._recording_thread.recording_error.connect(self._on_recording_error)
            self._recording_thread.recording_finished.connect(self._on_recording_finished)
            self._recording_thread.start()

            self.recording_started.emit()
            logger.info("Recording started")
            return True
        except Exception as e:
            logger.exception("Failed to start recording")
            self.audio_error.emit(f"Failed to start recording: {e}")
            return False

    @Slot(str)
    def _on_recording_error(self, error: str) -> None:
        logger.error(f"Recording error: {error}")
        self.audio_error.emit(error)

    @Slot(str)
    def _on_recording_finished(self, audio_file: str) -> None:
        try:
            self._current_temp_file = str(audio_file)
            logger.info(f"Audio saved to: {audio_file}")
            self.audio_ready.emit(str(audio_file))
        except Exception as e:
            logger.exception("Unexpected error finishing audio")
            self.audio_error.emit(f"Failed to finalize audio: {e}")

    def stop_recording(self) -> None:
        if self._recording_thread and self._recording_thread.isRunning():
            self._recording_thread.stop()
            logger.info("Recording stop requested")
            self.recording_stopped.emit()

    def cleanup(self) -> None:
        if self._recording_thread and self._recording_thread.isRunning():
            self._recording_thread.stop()

            if not self._recording_thread.wait_for_cleanup(timeout_ms=3000):
                logger.warning("Recording thread cleanup taking longer than expected, waiting...")
                self._recording_thread.wait(2000)

            if self._recording_thread.isRunning():
                logger.error("Recording thread did not stop gracefully, forcing termination")
                self._recording_thread.terminate()
                self._recording_thread.wait(1000)

        if self._current_temp_file:
            temp_file_manager.release(self._current_temp_file)
            self._current_temp_file = None

        logger.debug("AudioManager cleanup complete")