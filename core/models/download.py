"""Module for downloading and managing Whisper models locally."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from core.logging_config import get_logger

logger = get_logger(__name__)


def _get_base_directory() -> Path:
    """Get base directory of the application."""
    if getattr(sys, 'frozen', False):
        # PyInstaller frozen mode (compiled exe)
        return Path(sys.executable).parent
    else:
        # Development mode
        return Path(__file__).parent.parent.parent


def get_models_directory() -> Path:
    """Get the directory for storing models.
    
    Priority:
    1. Custom path from config (if set)
    2. Default: models/ next to exe (frozen) or in project root (development)
    
    Returns:
        Path to models directory
    """
    from config.manager import config_manager
    
    # Check for custom path in config
    custom_path = config_manager.get_models_directory()
    
    if custom_path:
        logger.info(f"Using custom models directory from config: {custom_path}")
        path = Path(custom_path)
        
        # If relative path, resolve based on base directory
        if not path.is_absolute():
            base_dir = _get_base_directory()
            path = base_dir / path
            logger.debug(f"Resolved relative path to: {path}")
        
        # Expand environment variables and user home directory
        path = Path(os.path.expanduser(os.path.expandvars(str(path))))
        
        return path
    
    # Default: models/ in project root or next to exe
    base_dir = _get_base_directory()
    return base_dir / "models"


def get_local_model_path(model_name: str) -> Path | None:
    """Get path to a local model if it exists and is valid.
    
    Args:
        model_name: Name of the model (e.g., "whisper-large-v3-ct2-bfloat16")
    
    Returns:
        Path to model directory if exists and valid, None otherwise
    """
    models_dir = get_models_directory()
    model_path = models_dir / model_name
    
    if model_path.exists() and verify_model_integrity(model_path):
        return model_path
    
    return None


def list_local_models() -> list[str]:
    """List all available local models.
    
    Returns:
        List of model names that exist locally and are valid
    """
    models_dir = get_models_directory()
    
    if not models_dir.exists():
        return []
    
    models = []
    for path in models_dir.iterdir():
        if path.is_dir() and verify_model_integrity(path):
            models.append(path.name)
    
    return models


def verify_model_integrity(model_path: Path) -> bool:
    """Verify that a model directory contains a valid CTranslate2 model.
    
    Args:
        model_path: Path to model directory
    
    Returns:
        True if model is valid, False otherwise
    """
    try:
        import ctranslate2
        return ctranslate2.contains_model(str(model_path))
    except Exception as e:
        logger.debug(f"Model integrity check failed for {model_path}: {e}")
        return False


def download_model_to_local(repo_id: str, model_name: str) -> Path:
    """Download a model from HuggingFace Hub to local models directory.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "ctranslate2-4you/whisper-large-v3-ct2-bfloat16")
        model_name: Local name for the model (e.g., "whisper-large-v3-ct2-bfloat16")
    
    Returns:
        Path to the downloaded model directory
    
    Raises:
        Exception: If download fails
    """
    from huggingface_hub import snapshot_download
    
    models_dir = get_models_directory()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / model_name
    
    # Check if model already exists and is valid
    if model_path.exists() and verify_model_integrity(model_path):
        logger.info(f"Model {model_name} already exists locally at {model_path}")
        return model_path
    
    logger.info(f"Downloading model {repo_id} to {model_path}")
    
    try:
        # Download model directly to folder (no cache structure, no symlinks)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,  # IMPORTANT: Use real files, not symlinks
        )
        
        logger.info(f"Model {repo_id} downloaded successfully to {model_path}")
        
        # Verify integrity after download
        if not verify_model_integrity(model_path):
            logger.error(f"Downloaded model {model_name} failed integrity check")
            raise Exception(f"Model {model_name} integrity check failed after download")
        
        return model_path
        
    except Exception as e:
        logger.exception(f"Failed to download model {repo_id}")
        # Clean up partial download if it exists
        if model_path.exists():
            import shutil
            try:
                shutil.rmtree(model_path)
                logger.debug(f"Cleaned up partial download at {model_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up partial download: {cleanup_error}")
        raise Exception(f"Failed to download model {repo_id}: {e}") from e
