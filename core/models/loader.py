from __future__ import annotations

from typing import Optional

import psutil
from faster_whisper import WhisperModel

from core.logging_config import get_logger
from core.exceptions import ModelLoadError
from core.models.download import (
    download_model_to_local,
    get_local_model_path,
)

logger = get_logger(__name__)


def _make_repo_string(model_name: str, quantization_type: str) -> str:
    if model_name.startswith("distil-whisper"):
        return f"ctranslate2-4you/{model_name}-ct2-{quantization_type}"
    return f"ctranslate2-4you/whisper-{model_name}-ct2-{quantization_type}"


def _extract_model_name_from_repo(repo_id: str) -> str:
    """Extract model name from repository ID.
    
    Args:
        repo_id: Repository ID (e.g., "ctranslate2-4you/whisper-large-v3-ct2-bfloat16")
    
    Returns:
        Model name (e.g., "whisper-large-v3-ct2-bfloat16")
    """
    return repo_id.split('/')[-1]


def load_model(
    model_name: str,
    quantization_type: str = "float32",
    device_type: str = "cpu",
    cpu_threads: Optional[int] = None,
) -> WhisperModel:
    """Load a Whisper model from local storage or download if needed.
    
    Args:
        model_name: Model name (e.g., "large-v3")
        quantization_type: Quantization type (e.g., "float32", "bfloat16")
        device_type: Device type ("cpu" or "cuda")
        cpu_threads: Number of CPU threads (auto-detected if None)
    
    Returns:
        Loaded WhisperModel instance
    
    Raises:
        ModelLoadError: If model loading fails
    """
    repo_id = _make_repo_string(model_name, quantization_type)
    local_name = _extract_model_name_from_repo(repo_id)
    
    logger.info(f"Loading Whisper model {repo_id} on {device_type}")
    
    # Check if model exists locally
    model_path = get_local_model_path(local_name)
    
    # If not found locally, download it
    if model_path is None:
        logger.info(f"Model {local_name} not found locally, downloading...")
        try:
            model_path = download_model_to_local(repo_id, local_name)
        except Exception as e:
            logger.exception(f"Failed to download model {repo_id}")
            raise ModelLoadError(f"Failed to download model {repo_id}: {e}") from e
    
    if cpu_threads is None:
        cpu_threads = psutil.cpu_count(logical=False) or 1

    try:
        # Load model from local path
        model = WhisperModel(
            str(model_path),  # Use local path instead of repo ID
            device=device_type,
            compute_type=quantization_type,
            cpu_threads=cpu_threads,
        )
    except Exception as e:
        logger.exception(f"Failed to load model from {model_path}")
        raise ModelLoadError(f"Error loading model from {model_path}: {e}") from e

    logger.info(f"Model {repo_id} ready from {model_path}")
    return model