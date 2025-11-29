import ctranslate2

from config.manager import config_manager
from core.logging_config import get_logger

logger = get_logger(__name__)


class CheckQuantizationSupport:

    excluded_types = ['int16', 'int8', 'int8_float32', 'int8_float16', 'int8_bfloat16']

    def has_cuda_device(self) -> bool:
        try:
            cuda_device_count = ctranslate2.get_cuda_device_count()
            return cuda_device_count > 0
        except Exception as e:
            logger.warning(f"Failed to check CUDA devices: {e}")
            return False

    def get_supported_quantizations_cuda(self) -> list[str]:
        try:
            cuda_quantizations = ctranslate2.get_supported_compute_types("cuda")
            return [q for q in cuda_quantizations if q not in self.excluded_types]
        except Exception as e:
            logger.warning(f"Failed to get CUDA quantizations: {e}")
            return []

    def get_supported_quantizations_cpu(self) -> list[str]:
        try:
            cpu_quantizations = ctranslate2.get_supported_compute_types("cpu")
            return [q for q in cpu_quantizations if q not in self.excluded_types]
        except Exception as e:
            logger.warning(f"Failed to get CPU quantizations: {e}")
            return ["float32"]

    def update_supported_quantizations(self) -> None:
        try:
            cpu_quantizations = self.get_supported_quantizations_cpu()
            config_manager.set_supported_quantizations("cpu", cpu_quantizations)
            logger.info(f"CPU quantizations: {cpu_quantizations}")

            if self.has_cuda_device():
                cuda_quantizations = self.get_supported_quantizations_cuda()
                config_manager.set_supported_quantizations("cuda", cuda_quantizations)
                logger.info(f"CUDA quantizations: {cuda_quantizations}")
        except Exception as e:
            logger.error(f"Failed to update quantization support: {e}")