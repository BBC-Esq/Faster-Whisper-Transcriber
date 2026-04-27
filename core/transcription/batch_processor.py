from __future__ import annotations

from pathlib import Path
from threading import Event

from faster_whisper import BatchedInferencePipeline
from PySide6.QtCore import QThread, Signal, QElapsedTimer

from core.output.writers import SegmentData, TranscriptionResult, write_output
from core.logging_config import get_logger

logger = get_logger(__name__)


def _is_oom_error(exc: Exception) -> bool:
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "out of memory" in msg or ("cuda" in msg and "alloc" in msg):
            return True
    return False


def _deduplicated_output_path(output_dir: Path, stem: str, suffix: str,
                               seen: dict[str, int]) -> Path:
    key = stem.lower()
    if key in seen:
        seen[key] += 1
        return output_dir / f"{stem}_{seen[key]}{suffix}"
    else:
        seen[key] = 0
        return output_dir / f"{stem}{suffix}"


class BatchProcessor(QThread):

    progress = Signal(int, int, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        files: list[Path],
        model,
        output_format: str,
        output_directory: str | None,
        batch_size: int,
        task_mode: str,
        whisper_params: dict,
    ):
        super().__init__()
        self.files = files
        self.model = model
        self.output_format = output_format
        self.output_directory = output_directory
        self.batch_size = batch_size
        self.task_mode = task_mode
        self.whisper_params = whisper_params
        self.stop_requested = Event()

    def request_stop(self) -> None:
        self.stop_requested.set()

    def run(self) -> None:
        timer = QElapsedTimer()
        timer.start()

        seen_names: dict[str, int] = {}

        try:
            batched_model = BatchedInferencePipeline(model=self.model)
            total_files = len(self.files)

            extra_kwargs = {
                "without_timestamps": self.whisper_params.get("without_timestamps", True),
                "word_timestamps": self.whisper_params.get("word_timestamps", False),
                "beam_size": self.whisper_params.get("beam_size", 5),
                "condition_on_previous_text": self.whisper_params.get("condition_on_previous_text", False),
            }

            extra_kwargs["vad_filter"] = True
            extra_kwargs["vad_parameters"] = dict(
                threshold=0.0008,
                neg_threshold=0.0001,
                min_speech_duration_ms=500,
                max_speech_duration_s=30,
                min_silence_duration_ms=1000,
                speech_pad_ms=500,
            )

            for idx, audio_file in enumerate(self.files, 1):
                if self.stop_requested.is_set():
                    break

                self.progress.emit(idx, total_files, f"Processing {audio_file.name}")

                try:
                    segments, info = batched_model.transcribe(
                        str(audio_file),
                        language=None,
                        task=self.task_mode,
                        batch_size=self.batch_size,
                        **extra_kwargs,
                    )

                    segment_list = []
                    text_parts = []
                    for segment in segments:
                        if self.stop_requested.is_set():
                            break
                        segment_list.append(
                            SegmentData(
                                start=segment.start,
                                end=segment.end,
                                text=segment.text,
                            )
                        )
                        text_parts.append(segment.text.lstrip())

                    if self.stop_requested.is_set():
                        break

                    result = TranscriptionResult(
                        text="\n".join(text_parts),
                        segments=segment_list,
                        language=info.language if info and hasattr(info, "language") else None,
                        duration=info.duration if info and hasattr(info, "duration") else None,
                        source_file=audio_file,
                    )

                    out_suffix = f".{self.output_format}"
                    if self.output_directory:
                        out_dir = Path(self.output_directory)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        output_file = _deduplicated_output_path(
                            out_dir, audio_file.stem, out_suffix, seen_names
                        )
                    else:
                        output_file = audio_file.with_suffix(out_suffix)

                    write_output(result, output_file, self.output_format)

                    self.progress.emit(
                        idx, total_files, f"Completed {audio_file.name}"
                    )

                except Exception as e:
                    if _is_oom_error(e):
                        self.error.emit(
                            f"GPU out of memory processing {audio_file.name}: {e}\n"
                            "Stopping batch. Try a smaller model or reduce batch size."
                        )
                        logger.error("OOM error, stopping batch: %s", e)
                        break
                    self.error.emit(f"Error processing {audio_file.name}: {e}")
                    logger.error("Error processing %s: %s", audio_file.name, e)

        except Exception as e:
            self.error.emit(f"Processing failed: {e}")
            logger.exception("Batch processing failed")

        finally:
            elapsed = timer.elapsed() / 1000.0
            self.finished.emit(f"Processing time: {elapsed:.2f} seconds")
