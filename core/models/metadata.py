from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set

@dataclass
class ModelInfo:
    name: str
    supports_translation: bool
    quantization_overrides: Dict[str, List[str]] | None = None

class ModelMetadata:

    _MODELS: List[ModelInfo] = [
        ModelInfo("tiny", True),
        ModelInfo("tiny.en", False),
        ModelInfo("base", True),
        ModelInfo("base.en", False),
        ModelInfo("small", True),
        ModelInfo("small.en", False),
        ModelInfo("medium", True),
        ModelInfo("medium.en", False),
        ModelInfo("large-v3", True),
        ModelInfo("large-v3-turbo", False, {"cpu": ["float32"], "cuda": ["float16", "bfloat16", "float32"]}),
        ModelInfo("distil-whisper-small.en", False, {"cpu": ["float32"], "cuda": ["float16", "bfloat16", "float32"]}),
        ModelInfo("distil-whisper-medium.en", False, {"cpu": ["float32"], "cuda": ["float16", "bfloat16", "float32"]}),
        ModelInfo("distil-whisper-large-v3", False, {"cpu": ["float32"], "cuda": ["float16", "bfloat16", "float32"]}),
    ]

    _MODEL_MAP: Dict[str, ModelInfo] = {m.name: m for m in _MODELS}

    @classmethod
    def get_all_model_names(cls) -> List[str]:
        return [m.name for m in cls._MODELS]

    @classmethod
    def get_model_info(cls, name: str) -> ModelInfo | None:
        return cls._MODEL_MAP.get(name)

    @classmethod
    def supports_translation(cls, name: str) -> bool:
        info = cls._MODEL_MAP.get(name)
        return info.supports_translation if info else False

    @classmethod
    def get_quantization_options(cls, model_name: str, device: str, supported_quantizations: Dict[str, List[str]]) -> List[str]:
        info = cls._MODEL_MAP.get(model_name)

        if info and info.quantization_overrides:
            # Model has quantization restrictions - use intersection with GPU capabilities
            model_options = info.quantization_overrides.get(device, [])
            gpu_options = supported_quantizations.get(device, [])
            # Show only what BOTH model AND GPU support
            options = [opt for opt in model_options if opt in gpu_options]
        else:
            # No model restrictions - use all GPU-supported quantizations
            options = supported_quantizations.get(device, [])

        if device == "cpu":
            options = [opt for opt in options if opt not in ["float16", "bfloat16"]]

        return options