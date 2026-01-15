from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, List, Set

import yaml

from utils import get_resource_path
from core.logging_config import get_logger
from core.exceptions import ConfigurationError

logger = get_logger(__name__)


class ConfigManager:

    VALID_OPTIONS = {
        "device_types": {"cpu", "cuda"},
        "task_modes": {"transcribe", "translate"},
        "quantization_types": {
            "int8", "int8_float16", "int8_float32", "int8_bfloat16",
            "int16", "float16", "float32", "bfloat16"
        }
    }

    DEFAULT_CONFIG = {
        "model_name": "base.en",
        "quantization_type": "float32",
        "device_type": "cpu",
        "task_mode": "transcribe",
        "show_clipboard_window": True,
        "supported_quantizations": {"cpu": [], "cuda": []},
        "curate_transcription": True
    }

    VALIDATION_SCHEMA = {
        "model_name": {"type": str, "validator": "_validate_model_name"},
        "device_type": {"type": str, "options": "device_types", "lowercase": True},
        "quantization_type": {"type": str, "options": "quantization_types"},
        "task_mode": {"type": str, "options": "task_modes", "lowercase": True},
        "show_clipboard_window": {"type": bool},
        "curate_transcription": {"type": bool},
    }

    def __init__(self):
        self._config_path = Path(get_resource_path("config.yaml"))
        self._config_cache: Optional[Dict[str, Any]] = None

    @property
    def config_path(self) -> Path:
        return self._config_path

    @staticmethod
    def _get_valid_model_names() -> Set[str]:
        try:
            from core.models.metadata import ModelMetadata
            return set(ModelMetadata.get_all_model_names())
        except ImportError:
            logger.warning("Could not import ModelMetadata for validation")
            return set()

    def _validate_model_name(self, value: Any) -> str:
        valid_models = self._get_valid_model_names()
        if valid_models and value not in valid_models:
            return self.DEFAULT_CONFIG["model_name"]
        return value

    def load_config(self) -> Dict[str, Any]:
        if self._config_cache is None:
            self._config_cache = self._load_from_file()
        return copy.deepcopy(self._config_cache)

    def _load_from_file(self) -> Dict[str, Any]:
        loaded_config = {}

        try:
            if self._config_path.exists():
                with self._config_path.open() as f:
                    loaded_config = yaml.safe_load(f) or {}
            else:
                logger.info("Config file not found, creating with defaults")
        except (yaml.YAMLError, Exception) as e:
            logger.error(f"Error loading config: {e}. Reverting to defaults.")
            loaded_config = {}

        if not isinstance(loaded_config, dict):
            loaded_config = {}

        merged_config = copy.deepcopy(self.DEFAULT_CONFIG)
        self._deep_update(merged_config, loaded_config)
        self._validate_and_sanitize(merged_config)

        return merged_config

    def _validate_and_sanitize(self, config: Dict[str, Any]) -> None:
        for key, schema in self.VALIDATION_SCHEMA.items():
            value = config.get(key)
            default = self.DEFAULT_CONFIG[key]

            if not isinstance(value, schema["type"]):
                logger.warning(f"Invalid type for {key}, reverting to default")
                config[key] = default
                continue

            if schema.get("lowercase") and isinstance(value, str):
                value = value.lower()
                config[key] = value

            if "options" in schema:
                valid_opts = self.VALID_OPTIONS[schema["options"]]
                if value not in valid_opts:
                    logger.warning(f"Invalid {key} '{value}', reverting to default")
                    config[key] = default
                    continue

            if "validator" in schema:
                config[key] = getattr(self, schema["validator"])(value)

        self._validate_supported_quantizations(config)

    def _validate_supported_quantizations(self, config: Dict[str, Any]) -> None:
        key = "supported_quantizations"
        if not isinstance(config[key], dict):
            config[key] = copy.deepcopy(self.DEFAULT_CONFIG[key])
            return

        for device in self.VALID_OPTIONS["device_types"]:
            if device not in config[key] or not isinstance(config[key][device], list):
                config[key][device] = []
            else:
                config[key][device] = [
                    q for q in config[key][device]
                    if isinstance(q, str) and q in self.VALID_OPTIONS["quantization_types"]
                ]

    def save_config(self, config: Dict[str, Any]) -> None:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            validated_config = copy.deepcopy(config)
            self._validate_and_sanitize(validated_config)

            with self._config_path.open("w") as f:
                yaml.safe_dump(validated_config, f, sort_keys=False)

            self._config_cache = validated_config
            logger.debug("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise ConfigurationError(f"Failed to save configuration: {e}") from e

    def update_config(self, updates: Dict[str, Any]) -> None:
        config = self.load_config()
        self._deep_update(config, updates)
        self.save_config(config)

    def get_value(self, key: str, default: Any = None) -> Any:
        return self.load_config().get(key, default)

    def set_value(self, key: str, value: Any) -> None:
        self.update_config({key: value})

    def get_model_settings(self) -> Dict[str, str]:
        config = self.load_config()
        return {k: config[k] for k in ["model_name", "quantization_type", "device_type"]}

    def set_model_settings(self, model_name: str, quantization_type: str, device_type: str) -> None:
        self.update_config({
            "model_name": model_name,
            "quantization_type": quantization_type,
            "device_type": device_type
        })

    def get_supported_quantizations(self) -> Dict[str, List[str]]:
        return self.get_value("supported_quantizations", {"cpu": [], "cuda": []})

    def set_supported_quantizations(self, device: str, quantizations: List[str]) -> None:
        if device not in self.VALID_OPTIONS["device_types"]:
            return
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