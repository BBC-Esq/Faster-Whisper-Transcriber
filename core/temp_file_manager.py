from __future__ import annotations
import tempfile
import atexit
from pathlib import Path
from typing import Set
from threading import Lock

from core.logging_config import get_logger

logger = get_logger(__name__)


class TempFileManager:

    def __init__(self) -> None:
        self._files: Set[Path] = set()
        self._files_lock = Lock()
        atexit.register(self.cleanup_all)

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
            if path not in self._files:
                logger.warning(f"Attempted to release untracked temp file: {path}")
                return False
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