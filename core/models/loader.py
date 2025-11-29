from __future__ import annotations

from typing import Optional

import psutil
from faster_whisper import WhisperModel

from core.logging_config import get_logger
from core.exceptions import ModelLoadError

logger = get_logger(__name__)


def _make_repo_string(model_name: str, quantization_type: str) -> str:
    if model_name.startswith("distil-whisper"):
        return f"ctranslate2-4you/{model_name}-ct2-{quantization_type}"
    return f"ctranslate2-4you/whisper-{model_name}-ct2-{quantization_type}"


def load_model(
    model_name: str,
    quantization_type: str = "float32",
    device_type: str = "cpu",
    cpu_threads: Optional[int] = None,
) -> WhisperModel:

    repo = _make_repo_string(model_name, quantization_type)
    logger.info(f"Loading Whisper model {repo} on {device_type}")

    if cpu_threads is None:
        cpu_threads = psutil.cpu_count(logical=False) or 1

    try:
        model = WhisperModel(
            repo,
            device=device_type,
            compute_type=quantization_type,
            cpu_threads=cpu_threads,
        )
    except Exception as e:
        logger.exception(f"Failed to load model {repo}")
        raise ModelLoadError(f"Error loading model {repo}: {e}") from e

    logger.info(f"Model {repo} ready")
    return model


class ModelLoader:

    def __init__(
        self,
        model_name: str,
        quantization_type: str = "int8",
        device_type: str = "cpu",
        cpu_threads: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.quantization_type = quantization_type
        self.device_type = device_type
        self.cpu_threads = cpu_threads

    def __call__(self) -> WhisperModel:
        return load_model(
            self.model_name,
            self.quantization_type,
            self.device_type,
            self.cpu_threads,
        )