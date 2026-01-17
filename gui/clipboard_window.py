from __future__ import annotations

from PySide6.QtCore import Qt, Slot, QRect, QEasingCurve, QPropertyAnimation, Signal, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QApplication, QCheckBox
from PySide6.QtGui import QMoveEvent

from core.logging_config import get_logger

logger = get_logger(__name__)


class ClipboardSideWindow(QWidget):
    user_closed = Signal()
    docked_changed = Signal(bool)
    always_on_top_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None, width: int = 360):
        super().__init__(parent)

        self._always_on_top = True
        self._append_mode = False
        self._docked = True
        self._internal_move = False
        self._internal_move_count = 0
        self._animation: QPropertyAnimation | None = None
        self._side: str = "right"
        self._last_host_rect: QRect | None = None
        self._desired_width = width
        self._default_height = 280

        self.setWindowTitle("Clipboard")
        self._apply_window_flags()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._text_display = QTextEdit()
        self._text_display.setReadOnly(True)
        self._text_display.setPlaceholderText("Your transcription will appear here.")
        layout.addWidget(self._text_display, 1)

        actions = QHBoxLayout()
        actions.setSpacing(10)

        copy_btn = QPushButton("Copy")
        copy_btn.setToolTip("Copy the current text to your clipboard")
        copy_btn.clicked.connect(self._copy_to_clipboard)
        actions.addWidget(copy_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Clear the clipboard panel")
        clear_btn.clicked.connect(self._text_display.clear)
        actions.addWidget(clear_btn)

        actions.addStretch(1)

        layout.addLayout(actions)

        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 0, 0, 0)
        bottom_bar.setSpacing(10)

        self._on_top_checkbox = QCheckBox("Always on Top")
        self._on_top_checkbox.setToolTip("Keep this window above other windows")
        self._on_top_checkbox.setChecked(self._always_on_top)
        self._on_top_checkbox.toggled.connect(self._on_always_on_top_toggled)
        bottom_bar.addWidget(self._on_top_checkbox)

        bottom_bar.addStretch(1)

        self._dock_button = QPushButton("Dock")
        self._dock_button.setToolTip("Re-attach to main window")
        self._dock_button.setFixedWidth(60)
        self._dock_button.clicked.connect(self._request_dock)
        self._dock_button.hide()
        bottom_bar.addWidget(self._dock_button)

        layout.addLayout(bottom_bar)

        self.setMinimumSize(285, 200)
        self.resize(self._desired_width, self._default_height)
        self.hide()

    def _apply_window_flags(self) -> None:
        flags = Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint
        if self._always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)

    def _copy_to_clipboard(self) -> None:
        app = QApplication.instance()
        if app:
            app.clipboard().setText(self._text_display.toPlainText())

    @Slot(bool)
    def _on_always_on_top_toggled(self, checked: bool) -> None:
        self._always_on_top = checked
        was_visible = self.isVisible()
        self._apply_window_flags()
        if was_visible:
            self.show()
        self.always_on_top_changed.emit(checked)

    def set_always_on_top(self, enabled: bool) -> None:
        if self._always_on_top == enabled:
            return
        self._always_on_top = enabled
        self._on_top_checkbox.blockSignals(True)
        self._on_top_checkbox.setChecked(enabled)
        self._on_top_checkbox.blockSignals(False)
        was_visible = self.isVisible()
        self._apply_window_flags()
        if was_visible:
            self.show()

    def is_always_on_top(self) -> bool:
        return self._always_on_top

    def set_append_mode(self, enabled: bool) -> None:
        self._append_mode = enabled

    def is_append_mode(self) -> bool:
        return self._append_mode

    def add_transcription(self, text: str) -> None:
        if self._append_mode:
            current = self._text_display.toPlainText().strip()
            self._text_display.setPlainText(f"{current}\n\n{text}" if current else text)
        else:
            self._text_display.setPlainText(text)

    def get_full_text(self) -> str:
        return self._text_display.toPlainText()

    def clear_text(self) -> None:
        self._text_display.clear()

    def is_docked(self) -> bool:
        return self._docked

    def set_docked(self, docked: bool) -> None:
        if self._docked == docked:
            return
        self._docked = docked
        self._dock_button.setVisible(not docked)
        self.docked_changed.emit(docked)

    def update_host_rect(self, host_rect: QRect) -> None:
        self._last_host_rect = host_rect

    def _stop_animation(self) -> None:
        if self._animation:
            self._animation.stop()
            self._animation = None

    def _begin_internal_move(self) -> None:
        self._internal_move_count += 1
        self._internal_move = True

    def _end_internal_move(self) -> None:
        self._internal_move_count = max(0, self._internal_move_count - 1)
        if self._internal_move_count == 0:
            self._internal_move = False

    def _schedule_end_internal_move(self) -> None:
        QTimer.singleShot(50, self._end_internal_move)

    def _compute_docked_rect(self, host_rect: QRect, gap: int, w: int, h: int) -> QRect:
        h = max(self.minimumHeight(), h)
        w = max(self.minimumWidth(), w)
        x = host_rect.left() - gap - w if self._side == "left" else host_rect.right() + gap
        return QRect(x, host_rect.top(), w, h)

    def _run_animation(self, start: QRect, end: QRect, duration: int, curve: QEasingCurve.Type, on_finish: list | None = None) -> None:
        anim = QPropertyAnimation(self, b"geometry", self)
        anim.setDuration(duration)
        anim.setEasingCurve(curve)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.finished.connect(self._schedule_end_internal_move)
        for callback in (on_finish or []):
            anim.finished.connect(callback)
        self._animation = anim
        anim.start()

    def show_docked(self, host_rect: QRect, gap: int = 10, animate: bool = True) -> None:
        self._stop_animation()
        self._last_host_rect = host_rect

        w = self._desired_width
        h = max(host_rect.height(), self._default_height, self.minimumHeight())
        end_rect = self._compute_docked_rect(host_rect, gap, w, h)
        offset = -30 if self._side == "left" else 30
        start_rect = QRect(end_rect.left() + offset, end_rect.top(), end_rect.width(), end_rect.height())

        self._begin_internal_move()
        self.setGeometry(start_rect)
        self.show()
        self.set_docked(True)

        if animate:
            self._run_animation(start_rect, end_rect, 180, QEasingCurve.OutCubic)
        else:
            self.setGeometry(end_rect)
            self._schedule_end_internal_move()

    def hide_animated(self, host_rect: QRect, gap: int = 10, animate: bool = True) -> None:
        self._stop_animation()
        if not self.isVisible():
            return

        if not animate:
            self.hide()
            return

        current = self.geometry()
        offset = -30 if self._side == "left" else 30
        end_rect = QRect(current.left() + offset, current.top(), current.width(), current.height())

        self._begin_internal_move()
        self._run_animation(current, end_rect, 160, QEasingCurve.InCubic, [self.hide])

    def reposition_to_host(self, host_rect: QRect, gap: int = 10) -> None:
        if not self.isVisible() or not self._docked:
            return

        self._stop_animation()
        self._last_host_rect = host_rect

        current = self.geometry()
        h = max(host_rect.height(), current.height(), self.minimumHeight())
        new_rect = self._compute_docked_rect(host_rect, gap, current.width(), h)

        self._begin_internal_move()
        self.setGeometry(new_rect)
        self._schedule_end_internal_move()

    def dock_to_host(self, host_rect: QRect, gap: int = 10, animate: bool = True) -> None:
        self._stop_animation()
        self._last_host_rect = host_rect

        h = max(host_rect.height(), self._default_height, self.minimumHeight())
        end_rect = self._compute_docked_rect(host_rect, gap, self._desired_width, h)

        self.set_docked(True)
        self._begin_internal_move()

        if animate and self.isVisible():
            self._run_animation(self.geometry(), end_rect, 200, QEasingCurve.OutCubic)
        else:
            self.setGeometry(end_rect)
            self.resize(self._desired_width, h)
            self._schedule_end_internal_move()
            if not self.isVisible():
                self.show()

    def moveEvent(self, event: QMoveEvent) -> None:
        super().moveEvent(event)
        if not self._internal_move and self._docked and self.isVisible():
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