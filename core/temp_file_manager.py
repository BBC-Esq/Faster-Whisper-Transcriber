from __future__ import annotations
import tempfile
import atexit
from pathlib import Path
from typing import Set
from threading import Lock

from core.logging_config import get_logger

logger = get_logger(__name__)

class TempFileManager:

    _instance: "TempFileManager | None" = None
    _lock = Lock()

    def __new__(cls) -> "TempFileManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._files: Set[Path] = set()
        self._files_lock = Lock()
        atexit.register(self.cleanup_all)
        self._initialized = True

    def create_temp_wav(self) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            path = Path(tf.name)

        with self._files_lock:
            self._files.add(path)

        logger.debug(f"Created temp file: {path}")
        return path

    def release(self, path: Path) -> bool:
        path = Path(path)

        with self._files_lock:
            if path in self._files:
                self._files.discard(path)

        try:
            if path.exists():
                path.unlink()
                logger.debug(f"Deleted temp file: {path}")
                return True
        except OSError as e:
            logger.warning(f"Failed to delete temp file {path}: {e}")

        return False

    def cleanup_all(self) -> None:
        with self._files_lock:
            files_to_clean = list(self._files)
            self._files.clear()

        for path in files_to_clean:
            try:
                if path.exists():
                    path.unlink()
                    logger.debug(f"Cleanup: deleted {path}")
            except OSError as e:
                logger.warning(f"Cleanup: failed to delete {path}: {e}")

temp_file_manager = TempFileManager()