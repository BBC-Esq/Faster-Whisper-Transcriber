from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Callable

import psutil
from faster_whisper import WhisperModel
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from core.logging_config import get_logger
from core.exceptions import ModelLoadError

logger = get_logger(__name__)


def _make_repo_string(model_name: str, quantization_type: str) -> str:
    if model_name.startswith("distil-whisper"):
        return f"ctranslate2-4you/{model_name}-ct2-{quantization_type}"
    return f"ctranslate2-4you/whisper-{model_name}-ct2-{quantization_type}"


def check_model_cached(repo_id: str) -> Optional[str]:
    try:
        local_path = snapshot_download(repo_id, local_files_only=True)
        return local_path
    except Exception:
        return None


def get_repo_file_info(repo_id: str) -> list[tuple[str, int]]:
    api = HfApi()
    info = api.repo_info(repo_id, repo_type="model")
    files = []
    for sibling in info.siblings:
        size = sibling.size if sibling.size is not None else 0
        files.append((sibling.rfilename, size))
    files.sort(key=lambda x: x[1])
    return files


def get_missing_files(
    repo_id: str, files_info: list[tuple[str, int]]
) -> tuple[Optional[str], list[tuple[str, int]]]:
    try:
        local_path = snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return None, list(files_info)

    missing = []
    for filename, size in files_info:
        filepath = Path(local_path) / filename
        if not filepath.exists():
            missing.append((filename, size))

    if missing:
        return None, missing
    return local_path, []


def download_model_files(
    repo_id: str,
    files_info: list[tuple[str, int]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    total_bytes = sum(size for _, size in files_info)
    downloaded_bytes = 0

    for filename, size in files_info:
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Download cancelled")

        try:
            hf_hub_download(repo_id, filename)
        except Exception as file_err:
            logger.warning(
                f"Per-file download failed for '{filename}': {file_err}. "
                f"Falling back to snapshot_download."
            )
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

    local_path = snapshot_download(repo_id, local_files_only=True)
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