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
        "supported_quantizations": {
            "cpu": [],
            "cuda": []
        },
        "curate_transcription": True
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
        except yaml.YAMLError as e:
            logger.error(f"Corrupt config.yaml detected: {e}. Reverting to defaults.")
            loaded_config = {}
        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}. Reverting to defaults.")
            loaded_config = {}

        if not isinstance(loaded_config, dict):
            logger.warning(
                f"Config file root must be a mapping/dict, got {type(loaded_config).__name__}. "
                f"Reverting to defaults."
            )
            loaded_config = {}

        merged_config = copy.deepcopy(self.DEFAULT_CONFIG)
        self._deep_update(merged_config, loaded_config)
        self._validate_and_sanitize(merged_config)

        return merged_config

    def _validate_and_sanitize(self, config: Dict[str, Any]) -> None:
        valid_models = self._get_valid_model_names()
        if valid_models and config["model_name"] not in valid_models:
            logger.warning(
                f"Invalid model_name '{config['model_name']}' in config. "
                f"Reverting to default: '{self.DEFAULT_CONFIG['model_name']}'"
            )
            config["model_name"] = self.DEFAULT_CONFIG["model_name"]

        if not isinstance(config["device_type"], str):
            logger.warning(
                f"Invalid type for device_type: expected string, got {type(config['device_type']).__name__}. "
                f"Reverting to default: '{self.DEFAULT_CONFIG['device_type']}'"
            )
            config["device_type"] = self.DEFAULT_CONFIG["device_type"]
        else:
            device_lower = config["device_type"].lower()
            if device_lower not in self.VALID_OPTIONS["device_types"]:
                logger.warning(
                    f"Invalid device_type '{config['device_type']}'. "
                    f"Reverting to default: '{self.DEFAULT_CONFIG['device_type']}'"
                )
                config["device_type"] = self.DEFAULT_CONFIG["device_type"]
            else:
                config["device_type"] = device_lower

        if not isinstance(config["quantization_type"], str):
            logger.warning(
                f"Invalid type for quantization_type: expected string, got {type(config['quantization_type']).__name__}. "
                f"Reverting to default: '{self.DEFAULT_CONFIG['quantization_type']}'"
            )
            config["quantization_type"] = self.DEFAULT_CONFIG["quantization_type"]
        elif config["quantization_type"] not in self.VALID_OPTIONS["quantization_types"]:
            logger.warning(
                f"Unknown quantization_type '{config['quantization_type']}'. "
                f"Reverting to default: '{self.DEFAULT_CONFIG['quantization_type']}'"
            )
            config["quantization_type"] = self.DEFAULT_CONFIG["quantization_type"]

        if not isinstance(config["task_mode"], str):
            logger.warning(
                f"Invalid type for task_mode: expected string, got {type(config['task_mode']).__name__}. "
                f"Reverting to default: '{self.DEFAULT_CONFIG['task_mode']}'"
            )
            config["task_mode"] = self.DEFAULT_CONFIG["task_mode"]
        else:
            task_lower = config["task_mode"].lower()
            if task_lower not in self.VALID_OPTIONS["task_modes"]:
                logger.warning(
                    f"Invalid task_mode '{config['task_mode']}'. "
                    f"Reverting to default: '{self.DEFAULT_CONFIG['task_mode']}'"
                )
                config["task_mode"] = self.DEFAULT_CONFIG["task_mode"]
            else:
                config["task_mode"] = task_lower

        for key in ["show_clipboard_window", "curate_transcription"]:
            if not isinstance(config[key], bool):
                logger.warning(
                    f"Invalid type for {key}: expected boolean, got {type(config[key]).__name__}. "
                    f"Reverting to default: {self.DEFAULT_CONFIG[key]}"
                )
                config[key] = self.DEFAULT_CONFIG[key]

        if not isinstance(config["supported_quantizations"], dict):
            logger.warning(
                f"Invalid type for supported_quantizations: expected dict, got {type(config['supported_quantizations']).__name__}. "
                f"Reverting to default."
            )
            config["supported_quantizations"] = copy.deepcopy(self.DEFAULT_CONFIG["supported_quantizations"])
        else:
            for device in self.VALID_OPTIONS["device_types"]:
                if device not in config["supported_quantizations"]:
                    config["supported_quantizations"][device] = []
                elif not isinstance(config["supported_quantizations"][device], list):
                    logger.warning(
                        f"Invalid type for supported_quantizations.{device}: expected list, "
                        f"got {type(config['supported_quantizations'][device]).__name__}. Reverting to empty list."
                    )
                    config["supported_quantizations"][device] = []
                else:
                    valid_quants = [
                        q for q in config["supported_quantizations"][device] 
                        if isinstance(q, str) and q in self.VALID_OPTIONS["quantization_types"]
                    ]
                    if len(valid_quants) != len(config["supported_quantizations"][device]):
                        logger.warning(
                            f"Invalid entries in supported_quantizations.{device} removed."
                        )
                    config["supported_quantizations"][device] = valid_quants

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

    def get_supported_quantizations(self) -> Dict[str, List[str]]:
        return self.get_value("supported_quantizations", {"cpu": [], "cuda": []})

    def set_supported_quantizations(self, device: str, quantizations: List[str]) -> None:
        if device not in self.VALID_OPTIONS["device_types"]:
            logger.warning(f"Invalid device '{device}' ignored in set_supported_quantizations()")
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