from __future__ import annotations

from pynput import keyboard

from core.logging_config import get_logger

logger = get_logger(__name__)


class GlobalHotkey:
    def __init__(self, on_toggle):
        self.on_toggle = on_toggle
        self.listener = None

    def start(self) -> bool:
        try:
            def on_press(key):
                if key == keyboard.Key.f9:
                    self.on_toggle()

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.daemon = True
            self.listener.start()
            logger.info("Global hotkey listener started (F9 to toggle recording)")
            return True
        except Exception as e:
            logger.error(f"Failed to start global hotkey listener: {e}")
            return False

    def stop(self) -> None:
        if self.listener is not None:
            import time as _time
            try:
                _t = _time.perf_counter()
                self.listener.stop()
                logger.info(f"[SHUTDOWN] listener.stop(): {_time.perf_counter() - _t:.3f}s")
                _t = _time.perf_counter()
                self.listener.join(timeout=2.0)
                logger.info(f"[SHUTDOWN] listener.join(): {_time.perf_counter() - _t:.3f}s (alive={self.listener.is_alive()})")
            except Exception as e:
                logger.warning(f"Error stopping hotkey listener: {e}")
            finally:
                self.listener = None