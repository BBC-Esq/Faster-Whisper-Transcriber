from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import psutil


@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    ram_usage_percent: float
    gpu_utilization: Optional[float] = None
    vram_usage_percent: Optional[float] = None
    power_usage_percent: Optional[float] = None


class SystemMonitor:

    def __init__(self):
        self._nvml = None
        self.gpu_handle = None
        self.has_nvidia = self._init_nvidia()

    def _init_nvidia(self) -> bool:
        try:
            import pynvml
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                self._nvml = pynvml
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return True
        except Exception:
            pass
        return False

    def collect_cpu_metrics(self) -> float:
        percentages = psutil.cpu_percent(interval=0, percpu=True)
        return sum(percentages) / len(percentages) if percentages else 0

    def collect_ram_metrics(self) -> float:
        return psutil.virtual_memory().percent

    def collect_gpu_metrics(self) -> tuple[float, float, float]:
        if not self.gpu_handle:
            return 0, 0, 0
        try:
            memory_info = self._nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            gpu_utilization = self._nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
            vram_percent = (memory_info.used / memory_info.total) * 100 if memory_info.total > 0 else 0
            try:
                power_usage = self._nvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
                power_limit = self._nvml.nvmlDeviceGetPowerManagementLimit(self.gpu_handle) / 1000.0
                power_percent = (power_usage / power_limit) * 100 if power_limit > 0 else 0
            except Exception:
                power_percent = 0
            return gpu_utilization, vram_percent, power_percent
        except Exception:
            return 0, 0, 0

    def shutdown(self) -> None:
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
            self.gpu_handle = None
            self.has_nvidia = False

    def collect_all_metrics(self) -> SystemMetrics:
        cpu_usage = self.collect_cpu_metrics()
        ram_percent = self.collect_ram_metrics()

        if self.has_nvidia:
            gpu_util, vram_percent, power_percent = self.collect_gpu_metrics()
        else:
            gpu_util = vram_percent = power_percent = None

        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            ram_usage_percent=ram_percent,
            gpu_utilization=gpu_util,
            vram_usage_percent=vram_percent,
            power_usage_percent=power_percent,
        )
