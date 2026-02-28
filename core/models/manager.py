from __future__ import annotations

from typing import Optional
import gc
import uuid
import threading

from PySide6.QtCore import QObject, Signal, QMutex, QMutexLocker, QRunnable, QThreadPool

from core.models.loader import (
    _make_repo_string,
    check_model_cached,
    get_repo_file_info,
    get_missing_files,
    download_model_files,
    load_model,
)
from core.logging_config import get_logger
from core.exceptions import ModelLoadError

logger = get_logger(__name__)


def _unload_model(model) -> None:
    try:
        if hasattr(model, "model") and hasattr(model.model, "unload_model"):
            model.model.unload_model()
    except Exception as e:
        logger.debug(f"Model unload call failed (non-critical): {e}")


_NETWORK_ERROR_TERMS = [
    "connection",
    "network",
    "resolve",
    "urlerror",
    "timeout",
    "unreachable",
    "dns",
    "socket",
    "offline",
]


def _is_network_error(exception: Exception) -> bool:
    msg = str(exception).lower()
    return any(term in msg for term in _NETWORK_ERROR_TERMS)


class _LoaderSignals(QObject):
    model_loaded = Signal(object, str, str, str, str)
    error_occurred = Signal(str, str)
    download_started = Signal(str, int, str)
    download_progress = Signal(int, int, str)
    download_finished = Signal(str, str)
    download_cancelled = Signal(str)
    loading_started = Signal(str, str)


class _ModelLoaderRunnable(QRunnable):
    def __init__(
        self,
        model_name: str,
        quant_type: str,
        device: str,
        model_version: str,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.model_name = model_name
        self.quant_type = quant_type
        self.device = device
        self.model_version = model_version
        self.cancel_event = cancel_event
        self.signals = _LoaderSignals()

    def run(self) -> None:
        try:
            repo_id = _make_repo_string(self.model_name, self.quant_type)

            if self.cancel_event.is_set():
                self.signals.download_cancelled.emit(self.model_version)
                return

            local_path = self._resolve_model_files(repo_id)
            if local_path is None:
                return

            if self.cancel_event.is_set():
                self.signals.download_cancelled.emit(self.model_version)
                return

            self.signals.loading_started.emit(self.model_name, self.model_version)

            model = load_model(local_path, self.quant_type, self.device)
            self.signals.model_loaded.emit(
                model,
                self.model_name,
                self.quant_type,
                self.device,
                self.model_version,
            )
        except ModelLoadError as e:
            logger.error(f"Model load error: {e}")
            self.signals.error_occurred.emit(str(e), self.model_version)
        except Exception as e:
            if self.cancel_event.is_set():
                self.signals.download_cancelled.emit(self.model_version)
            else:
                logger.exception("Unexpected error loading model")
                self.signals.error_occurred.emit(
                    f"Unexpected error: {e}", self.model_version
                )

    def _resolve_model_files(self, repo_id: str) -> Optional[str]:
        cached_path = check_model_cached(repo_id)

        if cached_path:
            files_info = None
            try:
                files_info = get_repo_file_info(repo_id)
            except Exception as e:
                if _is_network_error(e):
                    logger.info(
                        f"Offline but found cached model for "
                        f"'{self.model_name}', using cache as-is"
                    )
                    return cached_path
                self.signals.error_occurred.emit(
                    f"Failed to get model info for '{self.model_name}': {e}",
                    self.model_version,
                )
                return None

            if self.cancel_event.is_set():
                self.signals.download_cancelled.emit(self.model_version)
                return None

            _, missing_files = get_missing_files(repo_id, files_info)

            if not missing_files:
                return cached_path

            return self._download_files(repo_id, missing_files)

        try:
            files_info = get_repo_file_info(repo_id)
        except Exception as e:
            if _is_network_error(e):
                self.signals.error_occurred.emit(
                    f"Cannot download model '{self.model_name}': "
                    f"No internet connection. Please connect to the "
                    f"internet or select a previously downloaded model.",
                    self.model_version,
                )
            else:
                self.signals.error_occurred.emit(
                    f"Failed to get model info for '{self.model_name}': {e}",
                    self.model_version,
                )
            return None

        if self.cancel_event.is_set():
            self.signals.download_cancelled.emit(self.model_version)
            return None

        _, missing_files = get_missing_files(repo_id, files_info)

        if not missing_files:
            return check_model_cached(repo_id)

        return self._download_files(repo_id, missing_files)

    def _download_files(
        self, repo_id: str, files_to_download: list[tuple[str, int]]
    ) -> Optional[str]:
        total_bytes = sum(size for _, size in files_to_download)
        self.signals.download_started.emit(
            self.model_name, total_bytes, self.model_version
        )

        try:
            local_path = download_model_files(
                repo_id,
                files_to_download,
                progress_callback=self._on_download_progress,
                cancel_event=self.cancel_event,
            )
        except InterruptedError:
            self.signals.download_cancelled.emit(self.model_version)
            return None
        except Exception as e:
            if _is_network_error(e):
                self.signals.error_occurred.emit(
                    f"Download failed for '{self.model_name}': "
                    f"Network connection lost. Please check your "
                    f"internet connection and try again.",
                    self.model_version,
                )
            else:
                self.signals.error_occurred.emit(
                    f"Download failed for '{self.model_name}': {e}",
                    self.model_version,
                )
            return None

        self.signals.download_finished.emit(self.model_name, self.model_version)
        return local_path

    def _on_download_progress(self, downloaded: int, total: int) -> None:
        if not self.cancel_event.is_set():
            self.signals.download_progress.emit(downloaded, total, self.model_version)


class ModelManager(QObject):
    model_loaded = Signal(str, str, str)
    model_error = Signal(str)
    download_started = Signal(str, int)
    download_progress = Signal(int, int)
    download_finished = Signal(str)
    download_cancelled = Signal()
    loading_started = Signal(str)

    def __init__(self):
        super().__init__()
        self._model = None
        self._model_version: Optional[str] = None
        self._pending_version: Optional[str] = None
        self._model_mutex = QMutex()
        self._thread_pool = QThreadPool.globalInstance()
        self._current_settings = {}
        self._cancel_event: Optional[threading.Event] = None

    def load_model(self, model_name: str, quant: str, device: str) -> None:
        logger.info(f"Requesting model load: {model_name}, {quant}, {device}")

        if self._cancel_event:
            self._cancel_event.set()

        new_version = str(uuid.uuid4())
        self._pending_version = new_version
        self._cancel_event = threading.Event()

        runnable = _ModelLoaderRunnable(
            model_name, quant, device, new_version, self._cancel_event
        )
        runnable.signals.model_loaded.connect(self._on_model_loaded)
        runnable.signals.error_occurred.connect(self._on_model_error)
        runnable.signals.download_started.connect(self._on_download_started)
        runnable.signals.download_progress.connect(self._on_download_progress)
        runnable.signals.download_finished.connect(self._on_download_finished)
        runnable.signals.download_cancelled.connect(self._on_download_cancelled)
        runnable.signals.loading_started.connect(self._on_loading_started)
        self._thread_pool.start(runnable)

    def cancel_loading(self) -> None:
        if self._cancel_event:
            self._cancel_event.set()

    def get_model(self) -> tuple[Optional[object], Optional[str]]:
        with QMutexLocker(self._model_mutex):
            return self._model, self._model_version

    def _on_download_started(
        self, model_name: str, total_bytes: int, version: str
    ) -> None:
        if version == self._pending_version:
            self.download_started.emit(model_name, total_bytes)

    def _on_download_progress(
        self, downloaded: int, total: int, version: str
    ) -> None:
        if version == self._pending_version:
            self.download_progress.emit(downloaded, total)

    def _on_download_finished(self, model_name: str, version: str) -> None:
        if version == self._pending_version:
            self.download_finished.emit(model_name)

    def _on_download_cancelled(self, version: str) -> None:
        if version == self._pending_version:
            self.download_cancelled.emit()

    def _on_loading_started(self, model_name: str, version: str) -> None:
        if version == self._pending_version:
            self.loading_started.emit(model_name)

    def _on_model_loaded(
        self, model, name: str, quant: str, device: str, version: str
    ) -> None:
        if version != self._pending_version:
            logger.info(f"Ignoring stale model load (version {version})")
            _unload_model(model)
            del model
            gc.collect()
            return

        with QMutexLocker(self._model_mutex):
            if self._model is not None:
                _unload_model(self._model)
                del self._model
                gc.collect()
            self._model = model
            self._model_version = version

        self._current_settings = {
            "model_name": name,
            "quantization_type": quant,
            "device_type": device,
        }
        logger.info(f"Model loaded successfully: {name}")
        self.model_loaded.emit(name, quant, device)

    def _on_model_error(self, error: str, version: str) -> None:
        if version == self._pending_version:
            logger.error(f"Model error: {error}")
            self.model_error.emit(error)

    def cleanup(self) -> None:
        if self._cancel_event:
            self._cancel_event.set()
        self._thread_pool.waitForDone(5000)
        with QMutexLocker(self._model_mutex):
            if self._model is not None:
                _unload_model(self._model)
                del self._model
                self._model = None
                self._model_version = None
                gc.collect()
        logger.debug("ModelManager cleanup complete")
