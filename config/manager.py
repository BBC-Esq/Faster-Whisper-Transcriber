"""
Centralized configuration management for the Whisper transcriber.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from utils import get_resource_path

logger = logging.getLogger(__name__)


class ConfigManager:

    DEFAULT_CONFIG = {
        "model_name": "base.en",
        "quantization_type": "float32", 
        "device_type": "cpu",
        "task_mode": "transcribe",
        "show_clipboard_window": True,
        "supported_quantizations": {
            "cpu": [],
            "cuda": []
        }
    }

    def __init__(self):
        self._config_path = Path(get_resource_path("config.yaml"))
        self._config_cache: Optional[Dict[str, Any]] = None

    @property
    def config_path(self) -> Path:
        return self._config_path

    def load_config(self) -> Dict[str, Any]:
        if self._config_cache is None:
            self._config_cache = self._load_from_file()
        return self._config_cache.copy()

    def _load_from_file(self) -> Dict[str, Any]:
        try:
            with self._config_path.open() as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.info("Config file not found, using defaults")
            config = {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            config = {}

        merged_config = self.DEFAULT_CONFIG.copy()
        self._deep_update(merged_config, config)
        return merged_config

    def save_config(self, config: Dict[str, Any]) -> None:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with self._config_path.open("w") as f:
                yaml.safe_dump(config, f, sort_keys=False)

            self._config_cache = config.copy()
            logger.debug("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def update_config(self, updates: Dict[str, Any]) -> None:
        config = self.load_config()
        self._deep_update(config, updates)
        self.save_config(config)

    def get_value(self, key: str, default: Any = None) -> Any:
        config = self.load_config()
        return config.get(key, default)

    def set_value(self, key: str, value: Any) -> None:
        self.update_config({key: value})

    def get_model_settings(self) -> Dict[str, str]:
        config = self.load_config()
        return {
            "model_name": config["model_name"],
            "quantization_type": config["quantization_type"],
            "device_type": config["device_type"]
        }

    def set_model_settings(self, model_name: str, quantization_type: str, device_type: str) -> None:
        self.update_config({
            "model_name": model_name,
            "quantization_type": quantization_type,
            "device_type": device_type
        })

    def get_supported_quantizations(self) -> Dict[str, list[str]]:
        return self.get_value("supported_quantizations", {"cpu": [], "cuda": []})

    def set_supported_quantizations(self, device: str, quantizations: list[str]) -> None:
        current = self.get_supported_quantizations()
        current[device] = quantizations
        self.set_value("supported_quantizations", current)

    def invalidate_cache(self) -> None:
        self._config_cache = None

    @staticmethod
    def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                ConfigManager._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

config_manager = ConfigManager()