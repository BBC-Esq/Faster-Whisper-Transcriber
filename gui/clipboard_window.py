from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTextEdit,
    QApplication,
)

from core.logging_config import get_logger

logger = get_logger(__name__)


class ClipboardWindow(QWidget):

    def __init__(self, main_window: QWidget | None = None):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Current Transcription")

        layout = QVBoxLayout(self)
        self.text_display = QTextEdit()
        layout.addWidget(self.text_display)

        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self._copy_to_clipboard)
        layout.addWidget(self.copy_button)

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.resize(300, self.main_window.height() if main_window else 250)
        self.hide()

    def update_text(self, text: str) -> None:
        self.text_display.setText(text)

    def update_history(self, text: str) -> None:
        current = self.text_display.toPlainText()
        self.text_display.setText(f"{text}\n\n{current}" if current else text)

    def showEvent(self, event):
        if self.main_window:
            geo = self.main_window.geometry()
            self.setGeometry(
                geo.x() - self.width(),
                geo.y(),
                self.width(),
                geo.height(),
            )
        super().showEvent(event)

    def _copy_to_clipboard(self) -> None:
        try:
            app = QApplication.instance()
            if app:
                text = self.text_display.toPlainText()
                app.clipboard().setText(text)
                logger.debug("Text copied to clipboard")
        except Exception as e:
            logger.warning(f"Failed to copy to clipboard: {e}")