from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from faster_whisper import BatchedInferencePipeline
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool

from core.audio.wav_for_whisper import try_load_wav_for_faster_whisper
from core.temp_file_manager import temp_file_manager
from core.text.curation import curate_text
from core.output.writers import SegmentData, TranscriptionResult
from core.logging_config import get_logger

logger = get_logger(__name__)


class _TranscriberSignals(QObject):
    transcription_done = Signal(str)
    transcription_done_with_result = Signal(object)
    progress_updated = Signal(int, int, float)
    error_occurred = Signal(str)
    cancelled = Signal()


class _TranscriptionRunnable(QRunnable):
    def __init__(
        self,
        model,
        model_version: str,
        audio_file: str | Path,
        task_mode: str = "transcribe",
        is_temp_file: bool = True,
        batch_size: int | None = None,
        get_current_version_func=None,
        cancel_event: threading.Event | None = None,
        whisper_params: dict | None = None,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.model = model
        self.model_version = model_version
        self.audio_file = Path(audio_file)
        self.task_mode = task_mode
        self.is_temp_file = is_temp_file
        self.batch_size = batch_size
        self.get_current_version = get_current_version_func
        self.cancel_event = cancel_event or threading.Event()
        self.signals = _TranscriberSignals()
        self.whisper_params = whisper_params or {}

    def _is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def run(self) -> None:
        try:
            if self._is_cancelled():
                logger.info("Transcription cancelled before starting")
                self.signals.cancelled.emit()
                return

            if self.get_current_version and self.get_current_version() != self.model_version:
                logger.warning("Model changed during transcription setup")
                self.signals.cancelled.emit()
                return

            logger.info(f"Starting transcription: {self.audio_file}")

            whisper_sr = self.model.feature_extractor.sampling_rate
            audio_input: str | np.ndarray = try_load_wav_for_faster_whisper(
                self.audio_file, target_sr=whisper_sr
            )
            if audio_input is None:
                audio_input = str(self.audio_file)

            extra_kwargs = {
                "without_timestamps": self.whisper_params.get("without_timestamps", True),
                "word_timestamps": self.whisper_params.get("word_timestamps", False),
                "beam_size": self.whisper_params.get("beam_size", 5),
                "vad_filter": self.whisper_params.get("vad_filter", True),
                "condition_on_previous_text": self.whisper_params.get("condition_on_previous_text", False),
            }

            if self.batch_size is not None and int(self.batch_size) > 1:
                batched_model = BatchedInferencePipeline(model=self.model)
                segments, info = batched_model.transcribe(
                    audio_input,
                    language=None,
                    task=self.task_mode,
                    batch_size=int(self.batch_size),
                    **extra_kwargs,
                )
            else:
                segments, info = self.model.transcribe(
                    audio_input,
                    language=None,
                    task=self.task_mode,
                    **extra_kwargs,
                )

            total_duration = info.duration if info and hasattr(info, 'duration') else 0

            text_parts = []
            segment_data_list = []
            segment_count = 0

            for segment in segments:
                if self._is_cancelled():
                    logger.info(f"Transcription cancelled after {segment_count} segments")
                    self.signals.cancelled.emit()
                    return

                segment_count += 1
                text_parts.append(segment.text.lstrip())
                segment_data_list.append(
                    SegmentData(
                        start=segment.start,
                        end=segment.end,
                        text=segment.text,
                    )
                )

                if total_duration > 0:
                    progress_percent = min(100, (segment.end / total_duration) * 100)
                    self.signals.progress_updated.emit(segment_count, -1, progress_percent)
                else:
                    self.signals.progress_updated.emit(segment_count, -1, -1)

            text = "\n".join(text_parts)
            logger.info(f"Transcription completed successfully ({segment_count} segments)")

            result = TranscriptionResult(
                text=text,
                segments=segment_data_list,
                language=info.language if info and hasattr(info, 'language') else None,
                duration=info.duration if info and hasattr(info, 'duration') else None,
                source_file=self.audio_file,
            )

            self.signals.transcription_done.emit(text)
            self.signals.transcription_done_with_result.emit(result)

        except Exception as e:
            if self._is_cancelled():
                logger.info("Transcription cancelled during processing")
                self.signals.cancelled.emit()
            else:
                logger.exception("Transcription failed")
                self.signals.error_occurred.emit(f"Transcription failed: {e}")
        finally:
            if self.is_temp_file:
                temp_file_manager.release(self.audio_file)


class TranscriptionService(QObject):
    transcription_started = Signal()
    transcription_completed = Signal(str)
    transcription_completed_with_result = Signal(object)
    transcription_progress = Signal(int, int, float)
    transcription_error = Signal(str)
    transcription_cancelled = Signal()

    def __init__(self, curate_text_enabled: bool = False, task_mode: str = "transcribe"):
        super().__init__()
        self.curate_enabled = curate_text_enabled
        self.task_mode = task_mode
        self._thread_pool = QThreadPool.globalInstance()
        self._get_model_version_func = None
        self._cancel_event: threading.Event | None = None
        self._is_transcribing = False
        self._whisper_params: dict = {}

    def set_model_version_provider(self, func) -> None:
        self._get_model_version_func = func

    def set_whisper_params(self, params: dict) -> None:
        self._whisper_params = dict(params)
        logger.debug(f"Whisper params updated: {self._whisper_params}")

    def is_transcribing(self) -> bool:
        return self._is_transcribing

    def cancel_transcription(self) -> bool:
        if self._cancel_event and self._is_transcribing:
            logger.info("Cancellation requested")
            self._cancel_event.set()
            return True
        return False

    def transcribe_file(
        self,
        model,
        model_version: str,
        audio_file: str | Path,
        is_temp_file: bool = True,
        batch_size: int | None = None,
    ) -> None:
        if not model:
            error_msg = "No model available for transcription"
            logger.error(error_msg)
            self.transcription_error.emit(error_msg)
            if is_temp_file:
                temp_file_manager.release(Path(audio_file))
            return

        try:
            self._cancel_event = threading.Event()
            self._is_transcribing = True

            runnable = _TranscriptionRunnable(
                model=model,
                model_version=model_version,
                audio_file=str(audio_file),
                task_mode=self.task_mode,
                is_temp_file=is_temp_file,
                batch_size=batch_size,
                get_current_version_func=self._get_model_version_func,
                cancel_event=self._cancel_event,
                whisper_params=self._whisper_params,
            )
            runnable.signals.transcription_done.connect(self._on_transcription_done)
            runnable.signals.transcription_done_with_result.connect(
                self._on_transcription_done_with_result
            )
            runnable.signals.progress_updated.connect(self._on_progress_updated)
            runnable.signals.error_occurred.connect(self._on_transcription_error)
            runnable.signals.cancelled.connect(self._on_transcription_cancelled)
            self._thread_pool.start(runnable)
            self.transcription_started.emit()
        except Exception as e:
            logger.exception("Failed to start transcription")
            self._is_transcribing = False
            self.transcription_error.emit(f"Failed to start transcription: {e}")
            if is_temp_file:
                temp_file_manager.release(Path(audio_file))

    def _on_transcription_done(self, text: str) -> None:
        self._is_transcribing = False
        if self.curate_enabled:
            try:
                text = curate_text(text)
            except Exception as e:
                logger.warning(f"Text curation failed: {e}")

        self.transcription_completed.emit(text)

    def _on_transcription_done_with_result(self, result: object) -> None:
        self.transcription_completed_with_result.emit(result)

    def _on_progress_updated(self, segment_num: int, total_segments: int, percent: float) -> None:
        self.transcription_progress.emit(segment_num, total_segments, percent)

    def _on_transcription_error(self, error: str) -> None:
        self._is_transcribing = False
        logger.error(f"Transcription error: {error}")
        self.transcription_error.emit(error)

    def _on_transcription_cancelled(self) -> None:
        self._is_transcribing = False
        logger.info("Transcription was cancelled")
        self.transcription_cancelled.emit()

    def set_task_mode(self, mode: str) -> None:
        self.task_mode = mode
        logger.debug(f"Task mode set to: {mode}")

    def set_curation_enabled(self, enabled: bool) -> None:
        self.curate_enabled = enabled

    def cleanup(self) -> None:
        import time as _time

        _t = _time.perf_counter()
        if self._cancel_event:
            self._cancel_event.set()
        logger.info(f"[SHUTDOWN]   TS cancel_event.set(): {_time.perf_counter() - _t:.3f}s")

        _t = _time.perf_counter()
        self._thread_pool.waitForDone(5000)
        logger.info(f"[SHUTDOWN]   TS waitForDone(5000): {_time.perf_counter() - _t:.3f}s")

        logger.debug("TranscriptionService cleanup complete")
