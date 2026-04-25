from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Callable

import numpy as np
import psutil
from faster_whisper import WhisperModel
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm.auto import tqdm

from core.logging_config import get_logger
from core.exceptions import ModelLoadError

logger = get_logger(__name__)

PARAKEET_ONNX_MODEL_NAME = "parakeet-tdt-0.6b-v3-onnx"
PARAKEET_ONNX_ASR_NAME = "nemo-parakeet-tdt-0.6b-v3"


def is_parakeet_onnx_model(model_name: str) -> bool:
    return model_name == PARAKEET_ONNX_MODEL_NAME


class _NullWriter:
    def write(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass


def _ensure_streams() -> None:
    if sys.stdout is None:
        sys.stdout = _NullWriter()
    if sys.stderr is None:
        sys.stderr = _NullWriter()


class _ProgressTqdm(tqdm):
    def __init__(
        self,
        *args,
        progress_callback=None,
        completed_bytes=0,
        total_all_bytes=0,
        **kwargs,
    ):
        self._progress_callback = progress_callback
        self._completed_bytes = completed_bytes
        self._total_all_bytes = total_all_bytes
        kwargs.pop("name", None)
        if "file" in kwargs and kwargs["file"] is None:
            kwargs["file"] = _NullWriter()
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        super().update(n)
        if self._progress_callback and self._total_all_bytes > 0:
            self._progress_callback(
                self._completed_bytes + int(self.n), self._total_all_bytes
            )


def _make_tqdm_class(callback, completed, total_all):
    class _BoundTqdm(_ProgressTqdm):
        def __init__(self, *args, **kwargs):
            kwargs["progress_callback"] = callback
            kwargs["completed_bytes"] = completed
            kwargs["total_all_bytes"] = total_all
            super().__init__(*args, **kwargs)

    return _BoundTqdm


def _make_repo_string(model_name: str, quantization_type: str) -> str:
    if model_name.startswith("distil-whisper"):
        return f"ctranslate2-4you/{model_name}-ct2-{quantization_type}"
    return f"ctranslate2-4you/whisper-{model_name}-ct2-{quantization_type}"


class ParakeetOnnxModel:
    expects_file_path = True

    def __init__(self, text_model, segment_model=None, providers: list[str] | None = None) -> None:
        self._text_model = text_model
        self._timestamp_model = text_model.with_timestamps()
        self._segment_model = segment_model
        self.provider_names = list(providers or [])
        self.engine_name = "Parakeet ONNX"
        self.sample_rate = 16000

    def transcribe(
        self,
        audio_input,
        language=None,
        task: str = "transcribe",
        **_kwargs,
    ):
        if task == "translate":
            raise ModelLoadError("Parakeet ONNX supports transcription only")

        audio = self._load_audio(audio_input)
        duration = float(audio.shape[0] / self.sample_rate) if audio.size else 0.0

        if self._segment_model is not None:
            try:
                segments = self._recognize_vad_segments(audio)
            except Exception as exc:
                logger.warning(f"Parakeet VAD segmentation failed, using timestamps: {exc}")
                segments = self._recognize_timestamp_segments(audio, duration)
        else:
            segments = self._recognize_timestamp_segments(audio, duration)

        info = SimpleNamespace(duration=duration, language=None)
        return iter(segments), info

    def _load_audio(self, audio_input) -> np.ndarray:
        if isinstance(audio_input, np.ndarray):
            audio = np.asarray(audio_input, dtype=np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            return audio.reshape(-1).astype(np.float32, copy=False)

        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(audio_input), sampling_rate=self.sample_rate)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio.reshape(-1).astype(np.float32, copy=False)

    def _recognize_vad_segments(self, audio: np.ndarray) -> list[SimpleNamespace]:
        results = list(
            self._segment_model.recognize(
                audio,
                sample_rate=self.sample_rate,
                channel="mean",
            )
        )
        segments = [
            SimpleNamespace(
                start=float(result.start),
                end=float(result.end),
                text=str(result.text).strip(),
            )
            for result in results
            if str(result.text).strip()
        ]
        if not segments:
            return self._recognize_timestamp_segments(
                audio, float(audio.shape[0] / self.sample_rate)
            )
        return segments

    def _recognize_timestamp_segments(
        self,
        audio: np.ndarray,
        duration: float,
    ) -> list[SimpleNamespace]:
        result = self._timestamp_model.recognize(
            audio,
            sample_rate=self.sample_rate,
            channel="mean",
        )
        text = str(getattr(result, "text", "")).strip()
        tokens = getattr(result, "tokens", None) or []
        timestamps = getattr(result, "timestamps", None) or []

        if not text:
            return []
        if not tokens or not timestamps or len(tokens) != len(timestamps):
            return [SimpleNamespace(start=0.0, end=duration, text=text)]

        segments: list[SimpleNamespace] = []
        start = float(timestamps[0])
        last_time = start
        parts: list[str] = []

        for token, ts in zip(tokens, timestamps):
            token_text = str(token)
            current_time = float(ts)
            parts.append(token_text)
            last_time = current_time
            if token_text.strip().endswith((".", "?", "!")) or current_time - start >= 12.0:
                chunk = "".join(parts).strip()
                if chunk:
                    segments.append(
                        SimpleNamespace(
                            start=max(0.0, start),
                            end=min(duration, max(current_time, start + 0.25)),
                            text=chunk,
                        )
                    )
                parts = []
                start = current_time

        chunk = "".join(parts).strip()
        if chunk:
            segments.append(
                SimpleNamespace(
                    start=max(0.0, start),
                    end=min(duration, max(last_time, start + 0.25)),
                    text=chunk,
                )
            )

        return segments or [SimpleNamespace(start=0.0, end=duration, text=text)]


def _onnx_cache_root() -> Path:
    return Path(__file__).resolve().parents[2] / "model_cache" / "onnx"


def _configure_onnx_hf_cache(cache_root: Path) -> None:
    hf_home = cache_root / "hf_home"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_XET_CACHE"] = str(hf_home / "xet")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    hf_home.mkdir(parents=True, exist_ok=True)


def _clear_incomplete_onnx_cache(path: Path, required_files: tuple[str, ...]) -> None:
    if not path.exists():
        return
    if all((path / filename).is_file() for filename in required_files):
        return
    logger.warning(f"Clearing incomplete ONNX model cache: {path}")
    shutil.rmtree(path, onerror=_on_rmtree_error)
    if path.exists():
        _force_remove_tree(path)


def _onnx_providers(device_type: str) -> list[str]:
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        available = set()

    if device_type == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device_type == "cuda":
        logger.warning("ONNX Runtime CUDA provider unavailable; using CPU provider")
    return ["CPUExecutionProvider"]


def _onnx_session_options():
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    return sess_options


def load_parakeet_onnx_model(device_type: str = "cpu") -> ParakeetOnnxModel:
    try:
        import onnx_asr
    except Exception as exc:
        raise ModelLoadError(
            "Parakeet ONNX requires onnx-asr and onnxruntime-gpu. "
            "Run the installer with Parakeet ONNX support enabled."
        ) from exc

    providers = _onnx_providers(device_type)
    cache_root = _onnx_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    _configure_onnx_hf_cache(cache_root)
    model_dir = cache_root / "parakeet-tdt-0.6b-v3"
    vad_dir = cache_root / "silero-vad"
    _clear_incomplete_onnx_cache(
        model_dir,
        ("config.json", "encoder-model.onnx", "decoder_joint-model.onnx", "vocab.txt"),
    )
    _clear_incomplete_onnx_cache(vad_dir, ("silero_vad.onnx",))

    try:
        logger.info(f"Loading Parakeet ONNX with providers: {providers}")
        text_model = onnx_asr.load_model(
            PARAKEET_ONNX_ASR_NAME,
            path=model_dir,
            sess_options=_onnx_session_options(),
            providers=providers,
        )
    except Exception as exc:
        raise ModelLoadError(f"Error loading Parakeet ONNX model: {exc}") from exc

    segment_model = None
    try:
        vad = onnx_asr.load_vad(
            "silero",
            path=vad_dir,
            sess_options=_onnx_session_options(),
            providers=["CPUExecutionProvider"],
        )
        segment_model = text_model.with_vad(
            vad,
            max_speech_duration_s=25,
            min_silence_duration_ms=350,
            speech_pad_ms=80,
        )
    except Exception as exc:
        logger.warning(f"Parakeet ONNX VAD unavailable; using token timestamps: {exc}")

    return ParakeetOnnxModel(text_model, segment_model, providers)


def _get_local_model_dir(repo_id: str) -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        base = Path(HF_HUB_CACHE)
    except Exception:
        base = Path.home() / ".cache" / "huggingface" / "hub"
    return base / "local_copies" / repo_id.replace("/", "--")


def check_model_cached(repo_id: str) -> Optional[str]:
    normal_path = None
    try:
        normal_path = snapshot_download(repo_id, local_files_only=True)
    except OSError as e:
        logger.debug(
            f"Cache check hit OS error for {repo_id}: {e}. "
            f"Attempting manual cache path resolution."
        )
        normal_path = _resolve_cache_path(repo_id)
    except Exception:
        pass

    if normal_path and validate_model_path(normal_path):
        return normal_path

    local_dir = _get_local_model_dir(repo_id)
    if local_dir.is_dir() and validate_model_path(str(local_dir)):
        return str(local_dir)

    if normal_path:
        return normal_path

    return None


def _resolve_cache_path(repo_id: str) -> Optional[str]:
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                for revision in repo.revisions:
                    return str(revision.snapshot_path)
    except Exception:
        pass
    return None


def _is_file_accessible(filepath: Path) -> bool:
    try:
        with open(filepath, "rb") as f:
            f.read(1)
        return True
    except (OSError, IOError):
        return False


def validate_model_path(path: str) -> bool:
    model_bin = Path(path) / "model.bin"
    return _is_file_accessible(model_bin)


def _on_rmtree_error(func, path, _exc_info):
    try:
        os.chmod(path, 0o777)
        func(path)
    except Exception:
        try:
            os.unlink(path)
        except Exception:
            pass


def _force_remove_tree(path: Path) -> None:
    try:
        entries = os.listdir(str(path))
    except OSError:
        return
    for entry in entries:
        entry_path = os.path.join(str(path), entry)
        try:
            os.unlink(entry_path)
        except OSError:
            _force_remove_tree(Path(entry_path))
    try:
        os.rmdir(str(path))
    except OSError:
        pass


def _get_repo_cache_path(repo_id: str) -> Optional[Path]:
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                return Path(repo.repo_path)
    except Exception:
        pass

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        dir_name = "models--" + repo_id.replace("/", "--")
        candidate = Path(HF_HUB_CACHE) / dir_name
        if candidate.exists():
            return candidate
    except Exception:
        pass

    return None


def _clear_corrupted_cache(repo_id: str) -> None:
    target = _get_repo_cache_path(repo_id)
    if target is None:
        logger.warning(f"Could not locate cache directory for {repo_id}")
        return

    logger.info(f"Clearing entire model cache: {target}")
    shutil.rmtree(target, onerror=_on_rmtree_error)

    if target.exists():
        logger.warning(f"rmtree incomplete, forcing file-by-file removal")
        _force_remove_tree(target)

    if target.exists():
        logger.warning(
            f"Cache directory still exists after forced removal: {target}"
        )
    else:
        logger.info(f"Successfully cleared model cache: {target}")


def get_repo_file_info(repo_id: str) -> list[tuple[str, int]]:
    api = HfApi()
    info = api.repo_info(repo_id, repo_type="model", files_metadata=True)
    files = []
    for sibling in info.siblings:
        size = sibling.size if sibling.size is not None else 0
        files.append((sibling.rfilename, size))
    files.sort(key=lambda x: x[1])
    return files


def get_missing_files(
    repo_id: str,
    files_info: list[tuple[str, int]],
    cached_path: Optional[str] = None,
) -> tuple[Optional[str], list[tuple[str, int]]]:
    if cached_path is None:
        try:
            cached_path = snapshot_download(repo_id, local_files_only=True)
        except OSError:
            cached_path = _resolve_cache_path(repo_id)
            if cached_path is None:
                return None, list(files_info)
        except Exception:
            return None, list(files_info)

    missing = []
    for filename, size in files_info:
        filepath = Path(cached_path) / filename
        if not _is_file_accessible(filepath):
            missing.append((filename, size))

    if missing:
        return None, missing
    return cached_path, []


def download_model_files(
    repo_id: str,
    files_info: list[tuple[str, int]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    _ensure_streams()

    total_bytes = sum(size for _, size in files_info)
    downloaded_bytes = 0

    for filename, size in files_info:
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Download cancelled")

        try:
            tqdm_cls = (
                _make_tqdm_class(progress_callback, downloaded_bytes, total_bytes)
                if progress_callback
                else None
            )
            dl_kwargs = {"repo_id": repo_id, "filename": filename}
            if tqdm_cls:
                dl_kwargs["tqdm_class"] = tqdm_cls
            hf_hub_download(**dl_kwargs)
        except Exception as file_err:
            logger.warning(
                f"Per-file download failed for '{filename}': {file_err}. "
                f"Falling back to snapshot_download."
            )
            _ensure_streams()
            try:
                local_path = snapshot_download(repo_id)
            except Exception as snap_err:
                raise snap_err from file_err
            if progress_callback:
                progress_callback(total_bytes, total_bytes)
            return local_path

        downloaded_bytes += size

        if progress_callback:
            progress_callback(downloaded_bytes, total_bytes)

    try:
        local_path = snapshot_download(repo_id, local_files_only=True)
    except OSError:
        local_path = _resolve_cache_path(repo_id)
        if local_path is None:
            raise ModelLoadError(
                f"Files downloaded but cache path could not be resolved for {repo_id}"
            )

    if validate_model_path(local_path):
        return local_path

    logger.warning(
        f"Cache symlinks appear broken for {repo_id}, "
        f"downloading to local directory without symlinks"
    )
    _clear_corrupted_cache(repo_id)
    _ensure_streams()

    try:
        local_dir = _get_local_model_dir(repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id, local_dir=str(local_dir))
        local_path = str(local_dir)
    except Exception as e:
        raise ModelLoadError(
            f"Failed to download model files for {repo_id}: {e}"
        ) from e

    if not validate_model_path(local_path):
        raise ModelLoadError(
            f"Model files for {repo_id} could not be downloaded "
            f"successfully. Please manually delete the cache "
            f"directory and try again."
        )

    return local_path


def load_model(
    model_path: str,
    quantization_type: str = "float32",
    device_type: str = "cpu",
    cpu_threads: Optional[int] = None,
) -> WhisperModel:
    logger.info(f"Loading Whisper model from {model_path} on {device_type}")

    if cpu_threads is None:
        cpu_threads = psutil.cpu_count(logical=False) or 1

    try:
        model = WhisperModel(
            model_path,
            device=device_type,
            compute_type=quantization_type,
            cpu_threads=cpu_threads,
        )
    except Exception as e:
        logger.exception(f"Failed to load model from {model_path}")
        raise ModelLoadError(f"Error loading model: {e}") from e

    logger.info(f"Model ready from {model_path}")
    return model
