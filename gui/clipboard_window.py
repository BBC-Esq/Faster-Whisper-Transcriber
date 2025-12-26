from __future__ import annotations

from PySide6.QtCore import Qt, Slot, QRect, QEasingCurve, QPropertyAnimation, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QApplication

from core.logging_config import get_logger

logger = get_logger(__name__)


class ClipboardPanel(QWidget):
    switch_side_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setPlaceholderText("Your transcription will appear here.")
        layout.addWidget(self.text_display, 1)

        actions = QHBoxLayout()
        actions.setSpacing(10)

        self.copy_button = QPushButton("Copy")
        self.copy_button.setToolTip("Copy the current text to your clipboard")
        self.copy_button.clicked.connect(self._copy_to_clipboard)
        actions.addWidget(self.copy_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Clear the clipboard panel")
        self.clear_button.clicked.connect(self.clear_text)
        actions.addWidget(self.clear_button)

        actions.addStretch(1)

        self.switch_side_button = QPushButton("←")
        self.switch_side_button.setToolTip("Move clipboard to the other side")
        self.switch_side_button.setFixedWidth(36)
        self.switch_side_button.clicked.connect(self._request_switch_side)
        actions.addWidget(self.switch_side_button)

        layout.addLayout(actions)

    def set_side(self, side: str) -> None:
        self.switch_side_button.setText("←" if side == "right" else "→")

    def update_text(self, text: str) -> None:
        self.text_display.setPlainText(text)

    def update_history(self, text: str) -> None:
        current = self.text_display.toPlainText().strip()
        self.text_display.setPlainText(f"{text}\n\n{current}" if current else text)

    @Slot()
    def clear_text(self) -> None:
        self.text_display.clear()

    @Slot()
    def _copy_to_clipboard(self) -> None:
        try:
            app = QApplication.instance()
            if not app:
                return
            text = self.text_display.toPlainText()
            app.clipboard().setText(text)
            logger.debug("ClipboardPanel copied text to clipboard")
        except Exception as e:
            logger.warning(f"ClipboardPanel failed to copy: {e}")

    @Slot()
    def _request_switch_side(self) -> None:
        self.switch_side_requested.emit()


class ClipboardSideWindow(QWidget):
    switch_side_requested = Signal()
    user_closed = Signal()

    def __init__(self, parent: QWidget | None = None, width: int = 360):
        super().__init__(parent)

        self.setWindowTitle("Clipboard")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )

        self._panel = ClipboardPanel(self)
        self._panel.switch_side_requested.connect(self.switch_side_requested)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._panel)

        self._desired_width = width
        self._default_height = 280

        self._animation: QPropertyAnimation | None = None
        self._side: str = "right"
        self._panel.set_side(self._side)

        self._docked = True
        self._internal_geometry_change = False

        self.setMinimumSize(285, 260)
        self.resize(self._desired_width, self._default_height)
        self.hide()

    def is_docked(self) -> bool:
        return self._docked

    def dock_and_reset_size(self) -> None:
        self._docked = True
        self.resize(self._desired_width, self._default_height)

    def update_text(self, text: str) -> None:
        self._panel.update_text(text)

    def update_history(self, text: str) -> None:
        self._panel.update_history(text)

    def clear_text(self) -> None:
        self._panel.clear_text()

    def side(self) -> str:
        return self._side

    def set_side(self, side: str) -> None:
        side = "left" if side == "left" else "right"
        self._side = side
        self._panel.set_side(side)

    def _stop_animation(self) -> None:
        if self._animation:
            self._animation.stop()
            self._animation = None

    def _compute_rects(self, host_rect: QRect, gap: int, side: str, w: int, h: int) -> tuple[QRect, QRect]:
        h = max(self.minimumHeight(), h)
        w = max(self.minimumWidth(), w)

        if side == "left":
            end_x = host_rect.left() - gap - w
            end_rect = QRect(end_x, host_rect.top(), w, h)
            start_rect = QRect(end_rect.left() - 24, end_rect.top(), end_rect.width(), end_rect.height())
        else:
            end_x = host_rect.right() + gap
            end_rect = QRect(end_x, host_rect.top(), w, h)
            start_rect = QRect(end_rect.left() + 24, end_rect.top(), end_rect.width(), end_rect.height())

        return start_rect, end_rect

    def show_beside(self, host_rect: QRect, gap: int = 10, animate: bool = True, side: str | None = None) -> None:
        self._stop_animation()

        if side is not None:
            self.set_side(side)

        w = self.width()
        h = max(host_rect.height(), self.height(), self.minimumHeight())
        start_rect, end_rect = self._compute_rects(host_rect, gap, self._side, w, h)

        self._internal_geometry_change = True
        try:
            self.setGeometry(start_rect)
            self.show()
        finally:
            self._internal_geometry_change = False

        if not animate:
            self._internal_geometry_change = True
            try:
                self.setGeometry(end_rect)
            finally:
                self._internal_geometry_change = False
            return

        anim = QPropertyAnimation(self, b"geometry", self)
        anim.setDuration(180)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(start_rect)
        anim.setEndValue(end_rect)
        self._animation = anim

        def _done():
            self._internal_geometry_change = False

        self._internal_geometry_change = True
        anim.finished.connect(_done)
        anim.start()

    def hide_away(self, host_rect: QRect, gap: int = 10, animate: bool = True) -> None:
        self._stop_animation()

        if not self.isVisible():
            return

        current = self.geometry()
        w = current.width()
        h = current.height()
        _, end_rect = self._compute_rects(host_rect, gap, self._side, w, h)

        if self._side == "left":
            end_rect = QRect(end_rect.left() - 24, end_rect.top(), end_rect.width(), end_rect.height())
        else:
            end_rect = QRect(end_rect.left() + 24, end_rect.top(), end_rect.width(), end_rect.height())

        if not animate:
            self.hide()
            return

        anim = QPropertyAnimation(self, b"geometry", self)
        anim.setDuration(160)
        anim.setEasingCurve(QEasingCurve.InCubic)
        anim.setStartValue(current)
        anim.setEndValue(end_rect)
        anim.finished.connect(self.hide)
        self._animation = anim

        def _done():
            self._internal_geometry_change = False

        self._internal_geometry_change = True
        anim.finished.connect(_done)
        anim.start()

    def reposition_to_host(self, host_rect: QRect, gap: int = 10) -> None:
        if not self.isVisible() or not self._docked:
            return

        self._stop_animation()
        current = self.geometry()
        w = current.width()
        h = max(host_rect.height(), current.height(), self.minimumHeight())
        _, end_rect = self._compute_rects(host_rect, gap, self._side, w, h)

        self._internal_geometry_change = True
        try:
            self.setGeometry(end_rect)
        finally:
            self._internal_geometry_change = False

    def move_to_side(self, host_rect: QRect, new_side: str, gap: int = 10, animate: bool = True) -> None:
        new_side = "left" if new_side == "left" else "right"
        self.set_side(new_side)

        if not self.isVisible():
            return

        self._docked = True
        self._internal_geometry_change = True
        try:
            self.resize(self._desired_width, self._default_height)
        finally:
            self._internal_geometry_change = False

        self._stop_animation()
        current = self.geometry()
        w = self.width()
        h = max(host_rect.height(), self.height(), self.minimumHeight())
        _, end_rect = self._compute_rects(host_rect, gap, self._side, w, h)

        if not animate:
            self._internal_geometry_change = True
            try:
                self.setGeometry(end_rect)
            finally:
                self._internal_geometry_change = False
            return

        anim = QPropertyAnimation(self, b"geometry", self)
        anim.setDuration(220)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(current)
        anim.setEndValue(end_rect)
        self._animation = anim

        def _done():
            self._internal_geometry_change = False

        self._internal_geometry_change = True
        anim.finished.connect(_done)
        anim.start()

    def closeEvent(self, event):
        self.user_closed.emit()
        event.ignore()
        self.hide()

    def moveEvent(self, event):
        super().moveEvent(event)
        if self.isVisible() and not self._internal_geometry_change and event.spontaneous():
            self._docked = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.isVisible() and not self._internal_geometry_change and event.spontaneous():
            self._docked = False
