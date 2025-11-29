# core/transcription/service.py
from typing import Optional
from pathlib import Path
from PySide6.QtCore import QObject, Signal, QThread
import logging

logger = logging.getLogger(__name__)

class _TranscriptionThread(QThread):
    transcription_done = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, model, expected_id: int, audio_file: str | Path, task_mode: str = "transcribe") -> None:
        super().__init__()
        self.model = model
        self.expected_id = expected_id
        self.audio_file = str(audio_file)
        self.task_mode = task_mode

    def run(self) -> None:
        try:
            if id(self.model) != self.expected_id or self.isInterruptionRequested():
                return
            segments, _ = self.model.transcribe(
                self.audio_file,
                language=None,
                task=self.task_mode
            )
            if self.isInterruptionRequested():
                return
            self.transcription_done.emit("\n".join(s.text for s in segments))
        except Exception as exc:
            self.error_occurred.emit(f"Transcription failed: {exc}")
        finally:
            try:
                Path(self.audio_file).unlink(missing_ok=True)
            except OSError:
                pass

class TranscriptionService(QObject):
    transcription_started = Signal()
    transcription_completed = Signal(str)
    transcription_error = Signal(str)

    def __init__(self, curate_text_enabled: bool = False, task_mode: str = "transcribe"):
        super().__init__()
        self.curate_enabled = curate_text_enabled
        self.task_mode = task_mode
        self._transcription_thread: Optional[_TranscriptionThread] = None

    def transcribe_file(self, model, expected_id: int, audio_file: str | Path) -> None:
        if not model:
            self.transcription_error.emit("No model available")
            return

        self._transcription_thread = _TranscriptionThread(
            model, expected_id, str(audio_file), self.task_mode
        )
        self._transcription_thread.transcription_done.connect(self._on_transcription_done)
        self._transcription_thread.error_occurred.connect(self.transcription_error)
        self._transcription_thread.start()
        self.transcription_started.emit()

    def _on_transcription_done(self, text: str) -> None:
        if self.curate_enabled:
            try:
                from core.text.curation import curate_text
                text = curate_text(text)
            except Exception as exc:
                logger.warning("Curate failed: %s", exc)

        text = "\n".join(line.lstrip() for line in text.splitlines())
        self.transcription_completed.emit(text)

    def set_task_mode(self, mode: str) -> None:
        self.task_mode = mode

    def set_curation_enabled(self, enabled: bool) -> None:
        self.curate_enabled = enabled

    def cleanup(self) -> None:
        if self._transcription_thread and self._transcription_thread.isRunning():
            self._transcription_thread.requestInterruption()
            self._transcription_thread.wait()