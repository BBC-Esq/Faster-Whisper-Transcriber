from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Callable

import psutil
from faster_whisper import WhisperModel
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from core.logging_config import get_logger
from core.exceptions import ModelLoadError

logger = get_logger(__name__)


def _ensure_streams() -> None:
    """Ensure sys.stdout and sys.stderr are valid before calling huggingface_hub.

    When running via pythonw.exe (no console), these streams can be None or can
    become invalid mid-session. Libraries like tqdm crash with
    ``'NoneType' object has no attribute 'write'`` when this happens.
    """
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _make_repo_string(model_name: str, quantization_type: str) -> str:
    if model_name.startswith("distil-whisper"):
        return f"ctranslate2-4you/{model_name}-ct2-{quantization_type}"
    return f"ctranslate2-4you/whisper-{model_name}-ct2-{quantization_type}"


def check_model_cached(repo_id: str) -> Optional[str]:
    try:
        local_path = snapshot_download(repo_id, local_files_only=True)
        return local_path
    except OSError as e:
        logger.debug(
            f"Cache check hit OS error for {repo_id}: {e}. "
            f"Attempting manual cache path resolution."
        )
        return _resolve_cache_path(repo_id)
    except Exception:
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
    except OSError:
        local_path = _resolve_cache_path(repo_id)
        if local_path is None:
            return None, list(files_info)
    except Exception:
        return None, list(files_info)

    missing = []
    for filename, size in files_info:
        try:
            filepath = Path(local_path) / filename
            if not filepath.exists():
                missing.append((filename, size))
        except OSError as e:
            logger.debug(
                f"Cannot traverse '{filename}' in cache (OS error: {e}), "
                f"assuming present (symlink/reparse point)"
            )

    if missing:
        return None, missing
    return local_path, []


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
            hf_hub_download(repo_id, filename)
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