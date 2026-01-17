from __future__ import annotations

import contextlib
import warnings
import sys
import signal

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*pkg_resources is deprecated as an API.*"
)


from core.cuda_setup import setup_cuda_if_available
_cuda_paths_configured = setup_cuda_if_available()

from PySide6.QtWidgets import QApplication, QMessageBox

from core.logging_config import setup_logging, get_logger
from core.temp_file_manager import temp_file_manager
from gui.main_window import MainWindow


def _install_sigint_handler() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def _global_exception_handler(exc_type, exc_value, exc_tb):
    logger = get_logger(__name__)
    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))

    app = QApplication.instance()
    if app:
        QMessageBox.critical(
            None,
            "Critical Error",
            f"An unexpected error occurred:\n\n{exc_value}\n\nPlease check the log file for details."
        )


def run_gui() -> None:
    log_file = setup_logging()
    logger = get_logger(__name__)
    logger.info("Application starting")
    logger.info(f"Log file: {log_file}")
    logger.info(f"CUDA paths configured: {_cuda_paths_configured}")

    sys.excepthook = _global_exception_handler

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    _install_sigint_handler()

    cuda_ok = False
    with contextlib.suppress(ImportError):
        import torch
        cuda_ok = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_ok}")
        if cuda_ok:
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    try:
        window = MainWindow(cuda_available=cuda_ok)
        window.show()

        exit_code = app.exec()

        temp_file_manager.cleanup_all()
        logger.info(f"Application exiting with code {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        QMessageBox.critical(
            None,
            "Startup Error",
            f"Failed to start application:\n\n{e}"
        )
        sys.exit(1)


def main() -> None:
    run_gui()


if __name__ == "__main__":
    main()
