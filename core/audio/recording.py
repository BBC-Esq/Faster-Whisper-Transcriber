from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional
import wave

import sounddevice as sd
from PySide6.QtCore import QThread, Signal

from core.logging_config import get_logger
from core.exceptions import AudioRecordingError

logger = get_logger(__name__)


class RecordingThread(QThread):
    update_status_signal = Signal(str)
    recording_error = Signal(str)
    recording_finished = Signal(str)

    def __init__(
        self,
        output_path: str | Path,
        samplerate: int = 44_100,
        channels: int = 1,
        dtype: str = "int16",
        latency: str = "high",
    ) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.latency = latency

        self._audio_q: queue.Queue[bytes] = queue.Queue()
        self._overflow_count: int = 0
        self._stream_error: Optional[str] = None
        self._stop_event = threading.Event()
        self._cleanup_complete = threading.Event()

    @contextmanager
    def _audio_stream(self) -> Iterator[sd.RawInputStream]:
        try:
            stream = sd.RawInputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype=self.dtype,
                latency=self.latency,
                callback=self._audio_callback,
            )
            with stream:
                yield stream
        except sd.PortAudioError as e:
            logger.error(f"Audio device error: {e}")
            raise AudioRecordingError(f"Audio device error: {e}") from e
        except Exception as e:
            logger.error(f"Failed to create audio stream: {e}")
            raise AudioRecordingError(f"Failed to create audio stream: {e}") from e

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            msg = str(status)
            self._stream_error = msg
            self._overflow_count += 1
        try:
            self._audio_q.put_nowait(bytes(indata))
        except queue.Full:
            self._overflow_count += 1

    def run(self) -> None:
        self.update_status_signal.emit("Recording.")
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            with wave.open(str(self.output_path), "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self._sample_width_from_dtype(self.dtype))
                wf.setframerate(self.samplerate)

                with self._audio_stream():
                    while not self._should_stop():
                        try:
                            chunk = self._audio_q.get(timeout=0.2)
                            wf.writeframes(chunk)
                        except queue.Empty:
                            pass

                    while True:
                        try:
                            chunk = self._audio_q.get_nowait()
                            wf.writeframes(chunk)
                        except queue.Empty:
                            break

            if self._overflow_count > 0:
                logger.warning(
                    f"Recording completed with {self._overflow_count} overflow/underflow events. "
                    f"Last status: {self._stream_error}"
                )
                self.recording_error.emit(
                    f"Audio stream had {self._overflow_count} overflow/underflow events. "
                    f"Some audio may be missing. Try increasing latency/buffer."
                )

            self.recording_finished.emit(str(self.output_path))

        except AudioRecordingError as e:
            self.recording_error.emit(str(e))
        except Exception as e:
            logger.exception("Unexpected recording error")
            self.recording_error.emit(f"Recording error: {e}")
        finally:
            self._cleanup_complete.set()

    def _should_stop(self) -> bool:
        return self._stop_event.is_set() or self.isInterruptionRequested()

    def stop(self) -> None:
        self._stop_event.set()
        self.requestInterruption()

    def wait_for_cleanup(self, timeout_ms: int = 5000) -> bool:
        return self._cleanup_complete.wait(timeout=timeout_ms / 1000.0)

    @staticmethod
    def _sample_width_from_dtype(dtype: str) -> int:
        return {"int16": 2, "int32": 4, "float32": 4}.get(dtype, 2)
