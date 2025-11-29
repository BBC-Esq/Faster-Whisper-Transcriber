from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from typing import Iterator

import sounddevice as sd
from PySide6.QtCore import QThread, Signal

from core.logging_config import get_logger
from core.exceptions import AudioRecordingError

logger = get_logger(__name__)


class RecordingThread(QThread):

    update_status_signal = Signal(str)
    recording_error = Signal(str)
    recording_finished = Signal()

    def __init__(self, samplerate: int = 44_100, channels: int = 1, dtype: str = "int16") -> None:
        super().__init__()
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.buffer: queue.Queue = queue.Queue()
        self._stream_error: str | None = None

    @contextmanager
    def _audio_stream(self) -> Iterator[None]:
        stream = None
        try:
            stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback,
            )
            with stream:
                yield
        except sd.PortAudioError as e:
            logger.error(f"Audio device error: {e}")
            raise AudioRecordingError(f"Audio device error: {e}") from e
        except Exception as e:
            logger.error(f"Failed to create audio stream: {e}")
            raise AudioRecordingError(f"Failed to create audio stream: {e}") from e
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception as e:
                    logger.warning(f"Error closing audio stream: {e}")

    def _audio_callback(self, indata, frames, timestamp, status) -> None:
        if status:
            logger.warning(f"Audio callback status: {status}")
            self._stream_error = str(status)
        self.buffer.put(indata.copy())

    def run(self) -> None:
        self.update_status_signal.emit("Recording.")
        try:
            with self._audio_stream():
                gate = threading.Event()
                while not self.isInterruptionRequested():
                    gate.wait(timeout=0.1)
                    if self._stream_error:
                        logger.warning(f"Stream error during recording: {self._stream_error}")
        except AudioRecordingError as e:
            self.recording_error.emit(str(e))
        except Exception as e:
            logger.exception("Unexpected recording error")
            self.recording_error.emit(f"Recording error: {e}")
        finally:
            self.recording_finished.emit()

    def stop(self) -> None:
        self.requestInterruption()

    @staticmethod
    def _sample_width_from_dtype(dtype: str) -> int:
        return {"int16": 2, "int32": 4, "float32": 4}.get(dtype, 2)

    def get_buffer_contents(self) -> list:
        contents = []
        while not self.buffer.empty():
            try:
                contents.append(self.buffer.get_nowait())
            except queue.Empty:
                break
        return contents

    def clear_buffer(self) -> None:
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break