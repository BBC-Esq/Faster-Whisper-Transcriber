from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Optional, Callable

import psutil
from faster_whisper import WhisperModel
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm.auto import tqdm

from core.logging_config import get_logger
from core.exceptions import ModelLoadError

logger = get_logger(__name__)


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
        filepath = Path(local_path) / filename
        if not _is_file_accessible(filepath):
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
        f"Cache still corrupted after download for {repo_id}, "
        f"clearing snapshot and forcing full re-download"
    )
    _clear_corrupted_cache(repo_id)
    _ensure_streams()

    try:
        local_path = snapshot_download(repo_id)
    except Exception as e:
        raise ModelLoadError(
            f"Failed to repair corrupted cache for {repo_id}: {e}"
        ) from e

    if not validate_model_path(local_path):
        raise ModelLoadError(
            f"Model cache for {repo_id} is corrupted and could not be "
            f"repaired automatically. Please manually delete the cache "
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
