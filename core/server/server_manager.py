from __future__ import annotations

import threading
from typing import Optional

from PySide6.QtCore import QObject, Signal

from config.server_settings import TranscriptionSettings
from core.logging_config import get_logger

logger = get_logger(__name__)


class ServerManager(QObject):
    server_started = Signal(int)
    server_stopped = Signal()
    server_error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._port: int = 0

    def start_server(
        self, port: int, model_manager, default_settings: TranscriptionSettings
    ) -> None:
        if self.is_running():
            self.server_error.emit("Server is already running")
            return

        try:
            import uvicorn
            from core.server.api_server import create_app, set_app_state

            set_app_state(
                model_manager=model_manager,
                default_settings=default_settings,
            )

            app = create_app()
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=port,
                log_level="warning",
                log_config=None,
                access_log=False,
            )
            self._server = uvicorn.Server(config)
            self._port = port

            self._thread = threading.Thread(
                target=self._run_server,
                daemon=True,
                name="uvicorn-server",
            )
            self._thread.start()

            logger.info(f"Server starting on port {port}")
            self.server_started.emit(port)

        except Exception as e:
            logger.error(f"Failed to start server: {e}", exc_info=True)
            self.server_error.emit(str(e))

    def _run_server(self) -> None:
        try:
            self._server.run()
        except Exception as e:
            logger.error(f"Server thread error: {e}", exc_info=True)

    def stop_server(self) -> None:
        was_running = self.is_running()

        if self._server is not None:
            self._server.should_exit = True

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Server thread did not stop within timeout")
            self._thread = None

        self._server = None

        if was_running:
            logger.info("Server stopped")
            self.server_stopped.emit()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_transcription_active(self) -> bool:
        from core.server.api_server import _state
        return _state.transcription_active

    def cleanup(self) -> None:
        if self.is_running():
            from core.server.api_server import _state
            _state.cancel_event.set()
            self.stop_server()

    @property
    def port(self) -> int:
        return self._port
