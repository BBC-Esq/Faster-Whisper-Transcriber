from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from core.monitoring.system_metrics import SystemMonitor, SystemMetrics


class MetricsCollector(QThread):

    metrics_updated = Signal(object)

    def __init__(self, interval_ms: int = 1000):
        super().__init__()
        self._interval_ms = interval_ms
        self._running = False
        self._monitor: SystemMonitor | None = None

    @property
    def has_nvidia(self) -> bool:
        return self._monitor.has_nvidia if self._monitor else False

    def run(self) -> None:
        # Initialize monitor on the worker thread so pynvml.nvmlInit()
        # is called from the same thread that will query the GPU.
        self._monitor = SystemMonitor()
        self._running = True
        while self._running:
            metrics = self._monitor.collect_all_metrics()
            self.metrics_updated.emit(metrics)
            self.msleep(self._interval_ms)

    def stop(self) -> None:
        self._running = False
        self.wait(2000)
        if self._monitor:
            self._monitor.shutdown()
