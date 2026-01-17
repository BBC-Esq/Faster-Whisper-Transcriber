from __future__ import annotations

from PySide6.QtCore import Qt, Slot, QRect, QEasingCurve, QPropertyAnimation, Signal, QPoint, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QApplication
from PySide6.QtGui import QMoveEvent

from core.logging_config import get_logger

logger = get_logger(__name__)


class ClipboardPanel(QWidget):

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

        layout.addLayout(actions)

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


class ClipboardSideWindow(QWidget):
    user_closed = Signal()
    docked_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None, width: int = 360):
        super().__init__(parent)

        self.setWindowTitle("Clipboard")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )

        self._panel = ClipboardPanel(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._panel)

        self._bottom_bar = QHBoxLayout()
        self._bottom_bar.setContentsMargins(12, 0, 12, 12)
        self._bottom_bar.setSpacing(10)

        self._bottom_bar.addStretch(1)

        self._dock_button = QPushButton("Dock")
        self._dock_button.setToolTip("Re-attach to main window")
        self._dock_button.setFixedWidth(60)
        self._dock_button.clicked.connect(self._request_dock)
        self._dock_button.hide()
        self._bottom_bar.addWidget(self._dock_button)

        layout.addLayout(self._bottom_bar)

        self._desired_width = width
        self._default_height = 280

        self._animation: QPropertyAnimation | None = None
        self._side: str = "right"

        self._docked = True
        self._internal_move = False
        self._last_host_rect: QRect | None = None
        self._drag_start_pos: QPoint | None = None

        self.setMinimumSize(285, 200)
        self.resize(self._desired_width, self._default_height)
        self.hide()

    def is_docked(self) -> bool:
        return self._docked

    def set_docked(self, docked: bool) -> None:
        if self._docked == docked:
            return
        self._docked = docked
        self._dock_button.setVisible(not docked)
        self.docked_changed.emit(docked)

    def update_text(self, text: str) -> None:
        self._panel.update_text(text)

    def update_history(self, text: str) -> None:
        self._panel.update_history(text)

    def clear_text(self) -> None:
        self._panel.clear_text()

    def side(self) -> str:
        return self._side

    def set_side(self, side: str) -> None:
        self._side = "left" if side == "left" else "right"

    def _stop_animation(self) -> None:
        if self._animation:
            self._animation.stop()
            self._animation = None

    def _compute_docked_rect(self, host_rect: QRect, gap: int, w: int, h: int) -> QRect:
        h = max(self.minimumHeight(), h)
        w = max(self.minimumWidth(), w)

        if self._side == "left":
            x = host_rect.left() - gap - w
        else:
            x = host_rect.right() + gap

        return QRect(x, host_rect.top(), w, h)

    def show_docked(self, host_rect: QRect, gap: int = 10, animate: bool = True) -> None:
        self._stop_animation()
        self._last_host_rect = host_rect

        w = self._desired_width
        h = max(host_rect.height(), self._default_height, self.minimumHeight())
        end_rect = self._compute_docked_rect(host_rect, gap, w, h)

        if self._side == "left":
            start_rect = QRect(end_rect.left() - 30, end_rect.top(), end_rect.width(), end_rect.height())
        else:
            start_rect = QRect(end_rect.left() + 30, end_rect.top(), end_rect.width(), end_rect.height())

        self._internal_move = True
        self.setGeometry(start_rect)
        self.show()
        self._internal_move = False

        self.set_docked(True)

        if not animate:
            self._internal_move = True
            self.setGeometry(end_rect)
            self._internal_move = False
            return

        anim = QPropertyAnimation(self, b"geometry", self)
        anim.setDuration(180)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(start_rect)
        anim.setEndValue(end_rect)
        self._animation = anim

        def on_finished():
            self._internal_move = False

        self._internal_move = True
        anim.finished.connect(on_finished)
        anim.start()

    def hide_animated(self, host_rect: QRect, gap: int = 10, animate: bool = True) -> None:
        self._stop_animation()

        if not self.isVisible():
            return

        if not animate:
            self.hide()
            return

        current = self.geometry()

        if self._side == "left":
            end_rect = QRect(current.left() - 30, current.top(), current.width(), current.height())
        else:
            end_rect = QRect(current.left() + 30, current.top(), current.width(), current.height())

        anim = QPropertyAnimation(self, b"geometry", self)
        anim.setDuration(160)
        anim.setEasingCurve(QEasingCurve.InCubic)
        anim.setStartValue(current)
        anim.setEndValue(end_rect)
        anim.finished.connect(self.hide)
        self._animation = anim

        self._internal_move = True
        anim.finished.connect(lambda: setattr(self, '_internal_move', False))
        anim.start()

    def reposition_to_host(self, host_rect: QRect, gap: int = 10) -> None:
        if not self.isVisible() or not self._docked:
            return

        self._stop_animation()
        self._last_host_rect = host_rect

        current = self.geometry()
        w = current.width()
        h = max(host_rect.height(), current.height(), self.minimumHeight())
        new_rect = self._compute_docked_rect(host_rect, gap, w, h)

        self._internal_move = True
        self.setGeometry(new_rect)
        self._internal_move = False

    def dock_to_host(self, host_rect: QRect, gap: int = 10, animate: bool = True) -> None:
        self._stop_animation()
        self._last_host_rect = host_rect

        w = self._desired_width
        h = max(host_rect.height(), self._default_height, self.minimumHeight())
        end_rect = self._compute_docked_rect(host_rect, gap, w, h)

        self.set_docked(True)

        if not animate or not self.isVisible():
            self._internal_move = True
            self.setGeometry(end_rect)
            self.resize(self._desired_width, h)
            self._internal_move = False
            if not self.isVisible():
                self.show()
            return

        current = self.geometry()

        anim = QPropertyAnimation(self, b"geometry", self)
        anim.setDuration(200)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(current)
        anim.setEndValue(end_rect)
        self._animation = anim

        def on_finished():
            self._internal_move = False

        self._internal_move = True
        anim.finished.connect(on_finished)
        anim.start()

    def moveEvent(self, event: QMoveEvent) -> None:
        super().moveEvent(event)

        if self._internal_move:
            return

        if self._docked and self.isVisible():
            self.set_docked(False)
            logger.debug("Clipboard window detached by user drag")

    def closeEvent(self, event):
        self.user_closed.emit()
        event.ignore()
        self.hide()

    @Slot()
    def _request_dock(self) -> None:
        if self._last_host_rect:
            self.dock_to_host(self._last_host_rect, gap=10, animate=True)