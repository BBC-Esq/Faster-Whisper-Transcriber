from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import tempfile
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Dict, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.server_settings import TranscriptionSettings
from core.models.metadata import ModelMetadata

logger = logging.getLogger(__name__)

SR = 16000


class AppState:
    model_manager: Any = None
    default_settings: Optional[TranscriptionSettings] = None
    transcription_active: bool = False
    queue: Optional[asyncio.Queue] = None
    worker_task: Optional[asyncio.Task] = None
    cancel_event: Event = Event()


_state = AppState()


def set_app_state(*, model_manager, default_settings: TranscriptionSettings) -> None:
    _state.model_manager = model_manager
    _state.default_settings = default_settings
    _state.cancel_event.clear()
    _state.transcription_active = False


@dataclass
class WorkItem:
    audio_path: Path
    settings: TranscriptionSettings
    model_info: Dict[str, Any]
    future: asyncio.Future
    cleanup_path: bool = True


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int = SR) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    n_samples = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n_samples)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        if audio.shape[0] <= audio.shape[-1]:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=-1)
    audio = audio.flatten().astype(np.float32)
    if audio.max() > 1.0 or audio.min() < -1.0:
        max_val = max(abs(audio.max()), abs(audio.min()))
        if max_val > 0:
            if np.issubdtype(audio.dtype, np.integer) or max_val > 10:
                audio = audio / 32768.0
    return audio


def _write_wav(audio: np.ndarray, sr: int) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)

    with wave.open(str(tmp_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    return tmp_path


def _detect_format(filename: Optional[str], audio_format: str) -> str:
    if audio_format and audio_format != "auto":
        return audio_format

    if filename:
        ext = Path(filename).suffix.lower()
        if ext == ".npy":
            return "numpy"
        if ext == ".pt":
            return "tensor"
        audio_exts = {
            ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac",
            ".wma", ".webm", ".mp4", ".mkv", ".avi", ".asf", ".amr",
        }
        if ext in audio_exts:
            return "file"

    return "file"


def _normalize_to_wav(
    data: bytes,
    filename: Optional[str] = None,
    audio_format: str = "auto",
    sample_rate: int = SR,
    dtype: str = "float32",
) -> Path:
    """Accept one of several audio payload types, return a path to a 16 kHz mono WAV."""
    fmt = _detect_format(filename, audio_format)

    if fmt == "numpy":
        buf = io.BytesIO(data)
        audio = np.load(buf, allow_pickle=False)
        audio = _to_mono_float32(audio)
        audio = _resample(audio, sample_rate, SR)
        return _write_wav(audio, SR)

    if fmt == "tensor":
        import torch
        buf = io.BytesIO(data)
        tensor = torch.load(buf, map_location="cpu", weights_only=True)
        audio = tensor.numpy()
        audio = _to_mono_float32(audio)
        audio = _resample(audio, sample_rate, SR)
        return _write_wav(audio, SR)

    if fmt == "pcm":
        np_dtype = {
            "float32": np.float32, "float64": np.float64,
            "int16": np.int16, "int32": np.int32,
        }.get(dtype, np.float32)
        audio = np.frombuffer(data, dtype=np_dtype)
        audio = _to_mono_float32(audio)
        audio = _resample(audio, sample_rate, SR)
        return _write_wav(audio, SR)

    # Raw audio container file: write it verbatim and let faster-whisper decode
    # it via PyAV.
    suffix = Path(filename).suffix if filename else ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = Path(tmp.name)
    try:
        tmp.write(data)
        tmp.close()
    except Exception:
        tmp.close()
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return tmp_path


def _do_transcription(item: WorkItem) -> Dict[str, Any]:
    start_time = time.perf_counter()

    model_info = item.model_info
    model_name = model_info["name"]
    quantization = model_info["quantization_type"]

    try:
        model = _state.model_manager.get_or_load_model_sync(
            model_name=model_name,
            quantization=quantization,
            device=item.settings.device,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    if model is None:
        raise RuntimeError("Failed to load model")

    from faster_whisper import BatchedInferencePipeline

    extra_kwargs = {
        "without_timestamps": not item.settings.include_timestamps,
        "word_timestamps": item.settings.word_timestamps,
        "beam_size": item.settings.beam_size,
        "vad_filter": item.settings.vad_filter,
        "condition_on_previous_text": item.settings.condition_on_previous_text,
    }

    language = item.settings.language or None

    use_batched_pipeline = not getattr(model, "expects_file_path", False)

    if item.settings.batch_size and item.settings.batch_size > 1 and use_batched_pipeline:
        extra_kwargs["vad_filter"] = True
        extra_kwargs["vad_parameters"] = dict(
            threshold=0.0008,
            neg_threshold=0.0001,
            min_speech_duration_ms=500,
            max_speech_duration_s=30,
            min_silence_duration_ms=1000,
            speech_pad_ms=500,
        )
        batched_model = BatchedInferencePipeline(model=model)
        segments_iter, info = batched_model.transcribe(
            str(item.audio_path),
            language=language,
            task=item.settings.task_mode,
            batch_size=int(item.settings.batch_size),
            **extra_kwargs,
        )
    else:
        segments_iter, info = model.transcribe(
            str(item.audio_path),
            language=language,
            task=item.settings.task_mode,
            **extra_kwargs,
        )

    text_parts: list[str] = []
    segments_out: list[dict] = []
    for seg in segments_iter:
        text_parts.append(seg.text.lstrip())
        if item.settings.include_timestamps:
            segments_out.append(
                {
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "text": seg.text,
                }
            )
    text = "\n".join(text_parts)

    detected_lang = info.language if info and hasattr(info, "language") else language
    duration = info.duration if info and hasattr(info, "duration") else None

    elapsed = time.perf_counter() - start_time

    return {
        "text": text,
        "segments": segments_out,
        "language": detected_lang,
        "duration": duration,
        "task": item.settings.task_mode,
        "model_used": f"{model_name} - {quantization}",
        "processing_time_seconds": round(elapsed, 3),
    }


async def _queue_worker():
    while True:
        item: WorkItem = await _state.queue.get()
        _state.transcription_active = True
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _do_transcription, item)
            if not item.future.done():
                item.future.set_result(result)
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            if not item.future.done():
                item.future.set_exception(e)
        finally:
            _state.transcription_active = False
            if item.cleanup_path:
                try:
                    os.remove(item.audio_path)
                except OSError:
                    pass
            _state.queue.task_done()


def _resolve_model_key(
    model_name: Optional[str],
    quantization: Optional[str],
    defaults: TranscriptionSettings,
) -> Tuple[str, Dict[str, Any]]:
    default_key = defaults.model_key
    default_name = default_key.split(" - ")[0] if " - " in default_key else default_key
    default_quant = default_key.split(" - ")[1] if " - " in default_key else ""

    target_name = model_name or default_name
    target_quant = quantization or default_quant

    if target_name not in ModelMetadata.get_all_model_names():
        raise ValueError(
            f"Unknown model '{target_name}'. "
            f"Available: {ModelMetadata.get_all_model_names()}"
        )

    info = {
        "name": target_name,
        "quantization_type": target_quant,
        "supports_translation": ModelMetadata.supports_translation(target_name),
    }
    return f"{target_name} - {target_quant}", info


def _build_settings(
    model_name: Optional[str],
    quantization: Optional[str],
    device: Optional[str],
    output_format: Optional[str],
    language: Optional[str],
    task_mode: Optional[str],
    include_timestamps: Optional[bool],
    word_timestamps: Optional[bool],
    beam_size: Optional[int],
    vad_filter: Optional[bool],
    condition_on_previous_text: Optional[bool],
    batch_size: Optional[int],
) -> Tuple[TranscriptionSettings, Dict[str, Any]]:
    defaults = _state.default_settings

    model_key, model_info = _resolve_model_key(model_name, quantization, defaults)

    settings = TranscriptionSettings(
        model_key=model_key,
        device=device or defaults.device,
        task_mode=task_mode or defaults.task_mode,
        language=language if language is not None else defaults.language,
        output_format=output_format or defaults.output_format,
        include_timestamps=(
            include_timestamps if include_timestamps is not None
            else defaults.include_timestamps
        ),
        word_timestamps=(
            word_timestamps if word_timestamps is not None
            else defaults.word_timestamps
        ),
        beam_size=beam_size if beam_size is not None else defaults.beam_size,
        vad_filter=(
            vad_filter if vad_filter is not None else defaults.vad_filter
        ),
        condition_on_previous_text=(
            condition_on_previous_text
            if condition_on_previous_text is not None
            else defaults.condition_on_previous_text
        ),
        batch_size=batch_size if batch_size is not None else defaults.batch_size,
        recursive=False,
        selected_extensions=[],
    )
    return settings, model_info


class RawTranscribeRequest(BaseModel):
    audio_data: str
    audio_format: str = "numpy"
    sample_rate: int = 16000
    dtype: str = "float32"
    model: Optional[str] = None
    quantization: Optional[str] = None
    device: Optional[str] = None
    output_format: Optional[str] = None
    language: Optional[str] = None
    task_mode: Optional[str] = None
    include_timestamps: Optional[bool] = None
    word_timestamps: Optional[bool] = None
    beam_size: Optional[int] = None
    vad_filter: Optional[bool] = None
    condition_on_previous_text: Optional[bool] = None
    batch_size: Optional[int] = None


def create_app() -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _state.queue = asyncio.Queue()
        _state.cancel_event.clear()
        _state.worker_task = asyncio.create_task(_queue_worker())
        logger.info("Transcription queue worker started")
        yield
        if _state.worker_task:
            _state.worker_task.cancel()
            try:
                await _state.worker_task
            except asyncio.CancelledError:
                pass
        if _state.queue:
            while not _state.queue.empty():
                try:
                    item = _state.queue.get_nowait()
                    if not item.future.done():
                        item.future.set_exception(
                            RuntimeError("Server shutting down")
                        )
                except asyncio.QueueEmpty:
                    break
        logger.info("Transcription server shut down")

    app = FastAPI(
        title="Faster-Whisper Transcriber API",
        description=(
            "Transcribe audio using the faster-whisper (CTranslate2) backend. "
            "Accepts audio files, numpy arrays, PyTorch tensors, raw PCM, "
            "and base64-encoded data."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/models")
    async def models():
        result = {}
        for name in ModelMetadata.get_all_model_names():
            result[name] = {
                "name": name,
                "supports_translation": ModelMetadata.supports_translation(name),
            }
        return result

    @app.get("/status")
    async def status():
        return {
            "server_running": True,
            "queue_depth": _state.queue.qsize() if _state.queue else 0,
            "transcription_active": _state.transcription_active,
        }

    @app.post("/transcribe")
    async def transcribe(
        audio: UploadFile = File(...),
        audio_format: Optional[str] = Form("auto"),
        sample_rate: Optional[int] = Form(SR),
        dtype: Optional[str] = Form("float32"),
        model: Optional[str] = Form(None),
        quantization: Optional[str] = Form(None),
        device: Optional[str] = Form(None),
        output_format: Optional[str] = Form(None),
        language: Optional[str] = Form(None),
        task_mode: Optional[str] = Form(None),
        include_timestamps: Optional[bool] = Form(None),
        word_timestamps: Optional[bool] = Form(None),
        beam_size: Optional[int] = Form(None),
        vad_filter: Optional[bool] = Form(None),
        condition_on_previous_text: Optional[bool] = Form(None),
        batch_size: Optional[int] = Form(None),
    ):
        try:
            data = await audio.read()
            if not data:
                raise HTTPException(status_code=400, detail="Empty audio data")

            audio_path = _normalize_to_wav(
                data,
                filename=audio.filename,
                audio_format=audio_format,
                sample_rate=sample_rate,
                dtype=dtype,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process audio: {e}",
            )

        try:
            settings, model_info = _build_settings(
                model, quantization, device, output_format, language, task_mode,
                include_timestamps, word_timestamps, beam_size, vad_filter,
                condition_on_previous_text, batch_size,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        item = WorkItem(
            audio_path=audio_path,
            settings=settings,
            model_info=model_info,
            future=future,
        )
        await _state.queue.put(item)

        try:
            result = await future
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

        return result

    @app.post("/transcribe/raw")
    async def transcribe_raw(request: RawTranscribeRequest):
        try:
            data = base64.b64decode(request.audio_data)
            if not data:
                raise HTTPException(status_code=400, detail="Empty audio data")

            audio_path = _normalize_to_wav(
                data,
                filename=None,
                audio_format=request.audio_format,
                sample_rate=request.sample_rate,
                dtype=request.dtype,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode/process audio: {e}",
            )

        try:
            settings, model_info = _build_settings(
                request.model, request.quantization, request.device,
                request.output_format, request.language, request.task_mode,
                request.include_timestamps, request.word_timestamps,
                request.beam_size, request.vad_filter,
                request.condition_on_previous_text, request.batch_size,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        item = WorkItem(
            audio_path=audio_path,
            settings=settings,
            model_info=model_info,
            future=future,
        )
        await _state.queue.put(item)

        try:
            result = await future
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

        return result

    return app
