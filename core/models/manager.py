from __future__ import annotations

from typing import Optional
import gc

from PySide6.QtCore import QObject, Signal, QMutex, QRunnable, QThreadPool

from core.models.loader import load_model
from core.logging_config import get_logger
from core.exceptions import ModelLoadError

logger = get_logger(__name__)


class _LoaderSignals(QObject):
    model_loaded = Signal(object, str, str, str)
    error_occurred = Signal(str)


class _ModelLoaderRunnable(QRunnable):
    def __init__(self, model_name: str, quant_type: str, device: str) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.model_name = model_name
        self.quant_type = quant_type
        self.device = device
        self.signals = _LoaderSignals()

    def run(self) -> None:
        try:
            model = load_model(self.model_name, self.quant_type, self.device)
            self.signals.model_loaded.emit(
                model, self.model_name, self.quant_type, self.device
            )
        except ModelLoadError as e:
            logger.error(f"Model load error: {e}")
            self.signals.error_occurred.emit(str(e))
        except Exception as e:
            logger.exception("Unexpected error loading model")
            self.signals.error_occurred.emit(f"Unexpected error: {e}")


class ModelManager(QObject):
    model_loaded = Signal(str, str, str)
    model_error = Signal(str)
    
    def __init__(self):
        super().__init__()
        self._model = None
        self._model_mutex = QMutex()
        self._thread_pool = QThreadPool.globalInstance()
        self._current_settings = {}
    
    def load_model(self, model_name: str, quant: str, device: str) -> None:
        logger.info(f"Requesting model load: {model_name}, {quant}, {device}")
        runnable = _ModelLoaderRunnable(model_name, quant, device)
        runnable.signals.model_loaded.connect(self._on_model_loaded)
        runnable.signals.error_occurred.connect(self._on_model_error)
        self._thread_pool.start(runnable)
    
    def get_model(self):
        self._model_mutex.lock()
        try:
            model = self._model
            expected_id = id(model) if model else None
            return model, expected_id
        finally:
            self._model_mutex.unlock()
    
    def _on_model_loaded(self, model, name: str, quant: str, device: str) -> None:
        self._model_mutex.lock()
        try:
            if self._model is not None:
                del self._model
                gc.collect()
            self._model = model
        finally:
            self._model_mutex.unlock()
        
        self._current_settings = {
            "model_name": name,
            "quantization_type": quant, 
            "device_type": device
        }
        logger.info(f"Model loaded successfully: {name}")
        self.model_loaded.emit(name, quant, device)
    
    def _on_model_error(self, error: str) -> None:
        logger.error(f"Model error: {error}")
        self.model_error.emit(error)
    
    def cleanup(self) -> None:
        self._model_mutex.lock()
        try:
            if self._model is not None:
                del self._model
                self._model = None
                gc.collect()
        finally:
            self._model_mutex.unlock()
        logger.debug("ModelManager cleanup complete")