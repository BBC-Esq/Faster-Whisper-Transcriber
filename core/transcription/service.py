from __future__ import annotations

from typing import Optional
from pathlib import Path

from PySide6.QtCore import QObject, Signal, QThread

from core.temp_file_manager import temp_file_manager
from core.logging_config import get_logger
from core.exceptions import TranscriptionError

logger = get_logger(__name__)


class _TranscriptionThread(QThread):
    transcription_done = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        model,
        expected_id: int,
        audio_file: str | Path,
        task_mode: str = "transcribe",
        is_temp_file: bool = True
    ) -> None:
        super().__init__()
        self.model = model
        self.expected_id = expected_id
        self.audio_file = Path(audio_file)
        self.task_mode = task_mode
        self.is_temp_file = is_temp_file

    def run(self) -> None:
        try:
            if id(self.model) != self.expected_id:
                logger.warning("Model changed during transcription setup")
                return
                
            if self.isInterruptionRequested():
                return
            
            logger.info(f"Starting transcription: {self.audio_file}")
            segments, _ = self.model.transcribe(
                str(self.audio_file),
                language=None,
                task=self.task_mode
            )
            
            if self.isInterruptionRequested():
                return
            
            text = "\n".join(s.text for s in segments)
            logger.info("Transcription completed successfully")
            self.transcription_done.emit(text)
            
        except Exception as e:
            logger.exception("Transcription failed")
            self.error_occurred.emit(f"Transcription failed: {e}")
        finally:
            if self.is_temp_file:
                temp_file_manager.release(self.audio_file)


class TranscriptionService(QObject):
    transcription_started = Signal()
    transcription_completed = Signal(str)
    transcription_error = Signal(str)

    def __init__(self, curate_text_enabled: bool = False, task_mode: str = "transcribe"):
        super().__init__()
        self.curate_enabled = curate_text_enabled
        self.task_mode = task_mode
        self._transcription_thread: Optional[_TranscriptionThread] = None

    def transcribe_file(
        self,
        model,
        expected_id: int,
        audio_file: str | Path,
        is_temp_file: bool = True
    ) -> None:
        if not model:
            error_msg = "No model available for transcription"
            logger.error(error_msg)
            self.transcription_error.emit(error_msg)
            if is_temp_file:
                temp_file_manager.release(Path(audio_file))
            return

        try:
            self._transcription_thread = _TranscriptionThread(
                model, expected_id, str(audio_file), self.task_mode, is_temp_file
            )
            self._transcription_thread.transcription_done.connect(self._on_transcription_done)
            self._transcription_thread.error_occurred.connect(self._on_transcription_error)
            self._transcription_thread.start()
            self.transcription_started.emit()
        except Exception as e:
            logger.exception("Failed to start transcription thread")
            self.transcription_error.emit(f"Failed to start transcription: {e}")
            if is_temp_file:
                temp_file_manager.release(Path(audio_file))

    def _on_transcription_done(self, text: str) -> None:
        if self.curate_enabled:
            try:
                from core.text.curation import curate_text
                text = curate_text(text)
            except Exception as e:
                logger.warning(f"Text curation failed: {e}")

        text = "\n".join(line.lstrip() for line in text.splitlines())
        self.transcription_completed.emit(text)

    def _on_transcription_error(self, error: str) -> None:
        logger.error(f"Transcription error: {error}")
        self.transcription_error.emit(error)

    def set_task_mode(self, mode: str) -> None:
        self.task_mode = mode
        logger.debug(f"Task mode set to: {mode}")

    def set_curation_enabled(self, enabled: bool) -> None:
        self.curate_enabled = enabled

    def cleanup(self) -> None:
        if self._transcription_thread and self._transcription_thread.isRunning():
            self._transcription_thread.requestInterruption()
            self._transcription_thread.wait(5000)
            if self._transcription_thread.isRunning():
                logger.warning("Transcription thread did not stop in time")
                self._transcription_thread.terminate()
        logger.debug("TranscriptionService cleanup complete")