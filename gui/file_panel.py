from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot, QRect, QEasingCurve, QPropertyAnimation, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QCheckBox, QFileDialog,
    QGridLayout, QMessageBox, QApplication, QDialog,
)
from PySide6.QtGui import QMoveEvent, QPainter, QColor
from PySide6.QtCore import QSize as _QSize, Property

from core.transcription.file_scanner import FileScanner
from core.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = [
    ".aac", ".amr", ".asf", ".avi", ".flac", ".m4a",
    ".mkv", ".mp3", ".mp4", ".ogg", ".wav", ".webm", ".wma",
]

OUTPUT_FORMATS = ["txt", "srt", "vtt", "json"]

OUTPUT_MODES = [
    ("Clipboard only", "clipboard"),
    ("Save to source directory", "save_to_source"),
    ("Save to source + clipboard", "save_and_clipboard"),
    ("Save to custom directory", "save_to_custom"),
]


class ToggleSwitch(QCheckBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(36, 18)

    def sizeHint(self):
        return _QSize(36, 18)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Track
        track_color = QColor("#4a90d9") if self.isChecked() else QColor("#555555")
        p.setPen(Qt.NoPen)
        p.setBrush(track_color)
        p.drawRoundedRect(0, 2, 36, 14, 7, 7)

        # Thumb
        p.setBrush(QColor("#ffffff"))
        thumb_x = 20 if self.isChecked() else 2
        p.drawEllipse(thumb_x, 1, 16, 16)
        p.end()

    def hitButton(self, pos):
        return self.rect().contains(pos)


class FileTypesDialog(QDialog):
    def __init__(self, parent: QWidget | None, checked: dict[str, bool]):
        super().__init__(parent)
        self.setWindowTitle("File Types")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Select audio/video file types to include:"))

        grid = QGridLayout()
        grid.setSpacing(2)
        self._checkboxes: dict[str, QCheckBox] = {}
        for i, ext in enumerate(SUPPORTED_AUDIO_EXTENSIONS):
            cb = QCheckBox(ext)
            cb.setChecked(checked.get(ext, True))
            self._checkboxes[ext] = cb
            grid.addWidget(cb, i // 4, i % 4)
        layout.addLayout(grid)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        all_btn = QPushButton("All")
        all_btn.setFixedWidth(50)
        all_btn.clicked.connect(lambda: self._set_all(True))
        btn_row.addWidget(all_btn)
        none_btn = QPushButton("None")
        none_btn.setFixedWidth(50)
        none_btn.clicked.connect(lambda: self._set_all(False))
        btn_row.addWidget(none_btn)
        ok_btn = QPushButton("OK")
        ok_btn.setFixedWidth(60)
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    def _set_all(self, checked: bool) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(checked)

    def get_checked(self) -> dict[str, bool]:
        return {ext: cb.isChecked() for ext, cb in self._checkboxes.items()}

    def get_selected_extensions(self) -> list[str]:
        return [ext for ext, cb in self._checkboxes.items() if cb.isChecked()]


class FilePanelWindow(QWidget):
    user_closed = Signal()
    docked_changed = Signal(bool)

    transcribe_file_requested = Signal(str, int, str, str, str)
    batch_start_requested = Signal(list, str, str, int, str)
    batch_stop_requested = Signal()

    def __init__(self, parent: QWidget | None = None, width: int = 280):
        super().__init__(parent)

        self._docked = True
        self._internal_move = False
        self._internal_move_count = 0
        self._animation: QPropertyAnimation | None = None
        self._side: str = "left"
        self._last_host_rect: QRect | None = None
        self._desired_width = width
        self._default_height = 160

        self._selected_path: str = ""
        self._custom_output_dir: str = ""
        self._is_processing = False
        self._main_window_ref = None
        self._ext_checked: dict[str, bool] = {ext: True for ext in SUPPORTED_AUDIO_EXTENSIONS}

        self.setWindowTitle("File Transcription")
        self._apply_window_flags()

        self._scanner = FileScanner()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)

        # Row 1: Single [toggle] Multi | Recursive | Select
        top_row = QHBoxLayout()
        top_row.setSpacing(4)
        self._single_label = QLabel("Single")
        self._single_label.setStyleSheet("font-size: 11px;")
        top_row.addWidget(self._single_label)
        self._mode_toggle = ToggleSwitch()
        self._mode_toggle.toggled.connect(self._on_mode_changed)
        top_row.addWidget(self._mode_toggle)
        self._multi_label = QLabel("Multi")
        self._multi_label.setStyleSheet("font-size: 11px;")
        top_row.addWidget(self._multi_label)
        top_row.addSpacing(6)
        self._recursive_cb = QCheckBox("Recursive")
        self._recursive_cb.setEnabled(False)
        self._recursive_cb.toggled.connect(self._update_status)
        top_row.addWidget(self._recursive_cb)
        self._file_types_btn = QPushButton("File Types...")
        self._file_types_btn.setObjectName("fileTypesButton")
        self._file_types_btn.setFixedHeight(22)
        self._file_types_btn.setFixedWidth(75)
        self._file_types_btn.clicked.connect(self._open_file_types_dialog)
        top_row.addWidget(self._file_types_btn)
        top_row.addStretch(1)
        self._select_btn = QPushButton("Select...")
        self._select_btn.setObjectName("fileSelectButton")
        self._select_btn.setFixedHeight(22)
        self._select_btn.setFixedWidth(60)
        self._select_btn.clicked.connect(self._on_select_clicked)
        top_row.addWidget(self._select_btn)
        layout.addLayout(top_row)

        # Row 2: Path label
        self._path_label = QLabel("No file selected")
        self._path_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(self._path_label)

        # Settings grid: Batch size + Format on row 0, Output on row 1
        grid = QGridLayout()
        grid.setSpacing(3)

        grid.addWidget(QLabel("Batch size:"), 0, 0)
        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 128)
        self._batch_size.setValue(16)
        grid.addWidget(self._batch_size, 0, 1)

        grid.addWidget(QLabel("Format:"), 0, 2)
        self._format_combo = QComboBox()
        self._format_combo.addItems(OUTPUT_FORMATS)
        grid.addWidget(self._format_combo, 0, 3)

        grid.addWidget(QLabel("Output:"), 1, 0)
        self._output_mode = QComboBox()
        for display, _ in OUTPUT_MODES:
            self._output_mode.addItem(display)
        self._output_mode.currentIndexChanged.connect(self._on_output_mode_changed)
        grid.addWidget(self._output_mode, 1, 1, 1, 3)

        layout.addLayout(grid)

        # Custom output directory (hidden by default)
        self._custom_dir_btn = QPushButton("Output directory...")
        self._custom_dir_btn.setFixedHeight(22)
        self._custom_dir_btn.clicked.connect(self._select_custom_dir)
        self._custom_dir_btn.setVisible(False)
        layout.addWidget(self._custom_dir_btn)

        # Status label
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(self._status_label)

        # Start / Stop / Dock buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        btn_row.setContentsMargins(0, 0, 0, 0)
        self._start_btn = QPushButton("Start")
        self._start_btn.setObjectName("fileStartButton")
        self._start_btn.setFixedHeight(26)
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("fileStopButton")
        self._stop_btn.setFixedHeight(26)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)
        self._dock_button = QPushButton("Dock")
        self._dock_button.setObjectName("fileDockButton")
        self._dock_button.setToolTip("Re-attach to main window")
        self._dock_button.setFixedWidth(50)
        self._dock_button.setFixedHeight(26)
        self._dock_button.clicked.connect(self._request_dock)
        self._dock_button.hide()
        btn_row.addWidget(self._dock_button)
        layout.addLayout(btn_row)

        self.setMinimumSize(270, 160)
        self.resize(self._desired_width, self._default_height)
        self.hide()

    def _apply_window_flags(self) -> None:
        flags = Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint
        flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)

    def _is_single_mode(self) -> bool:
        return not self._mode_toggle.isChecked()

    # --- Mode toggle ---

    @Slot(bool)
    def _on_mode_changed(self, is_multi: bool) -> None:
        self._selected_path = ""
        self._path_label.setText("No directory selected" if is_multi else "No file selected")
        self._recursive_cb.setEnabled(is_multi)
        self._start_btn.setEnabled(False)
        self._status_label.setText("")

        # Disable clipboard-related output modes in multi mode
        model = self._output_mode.model()
        for i, (_, mode_key) in enumerate(OUTPUT_MODES):
            item = model.item(i)
            if mode_key in ("clipboard", "save_and_clipboard"):
                if is_multi:
                    item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                else:
                    item.setFlags(item.flags() | Qt.ItemIsEnabled)

        # If current selection is now disabled, switch to a valid one
        if is_multi:
            current_mode = OUTPUT_MODES[self._output_mode.currentIndex()][1]
            if current_mode in ("clipboard", "save_and_clipboard"):
                self._output_mode.setCurrentIndex(
                    next(i for i, (_, k) in enumerate(OUTPUT_MODES) if k == "save_to_source")
                )

    # --- Select file/directory ---

    @Slot()
    def _on_select_clicked(self) -> None:
        if self._is_single_mode():
            exts = " ".join(f"*{ext}" for ext in SUPPORTED_AUDIO_EXTENSIONS)
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Audio File", "", f"Audio Files ({exts});;All Files (*)"
            )
            if path:
                self._selected_path = path
                name = Path(path).name
                self._path_label.setText(name)
                self._path_label.setToolTip(path)
                self._start_btn.setEnabled(True)
                self._status_label.setText("")
        else:
            d = QFileDialog.getExistingDirectory(self, "Select Directory to Scan")
            if d:
                self._selected_path = d
                self._path_label.setText(d)
                self._path_label.setToolTip(d)
                self._update_status()

    # --- File types dialog ---

    @Slot()
    def _open_file_types_dialog(self) -> None:
        dlg = FileTypesDialog(self, self._ext_checked)
        if dlg.exec() == QDialog.Accepted:
            self._ext_checked = dlg.get_checked()
            self._update_status()

    def get_ext_checked(self) -> dict[str, bool]:
        return dict(self._ext_checked)

    def set_ext_checked(self, checked: dict[str, bool]) -> None:
        self._ext_checked = dict(checked)

    # --- State accessors for QSettings persistence ---

    def get_panel_state(self) -> dict:
        return {
            "multi_mode": self._mode_toggle.isChecked(),
            "recursive": self._recursive_cb.isChecked(),
            "batch_size": self._batch_size.value(),
            "format": self._format_combo.currentText(),
            "output_mode_index": self._output_mode.currentIndex(),
            "custom_output_dir": self._custom_output_dir,
            "selected_path": self._selected_path,
        }

    def restore_panel_state(self, state: dict) -> None:
        # Restore mode toggle (triggers _on_mode_changed)
        multi = state.get("multi_mode", False)
        self._mode_toggle.setChecked(multi)

        self._recursive_cb.setChecked(state.get("recursive", False))
        self._batch_size.setValue(state.get("batch_size", 16))

        fmt = state.get("format", "txt")
        idx = self._format_combo.findText(fmt)
        if idx >= 0:
            self._format_combo.setCurrentIndex(idx)

        output_idx = state.get("output_mode_index", 0)
        if 0 <= output_idx < self._output_mode.count():
            self._output_mode.setCurrentIndex(output_idx)

        custom_dir = state.get("custom_output_dir", "")
        if custom_dir and Path(custom_dir).is_dir():
            self._custom_output_dir = custom_dir
            self._custom_dir_btn.setText(f"Dir: {Path(custom_dir).name}")
            self._custom_dir_btn.setToolTip(custom_dir)

        # Restore selected path only if it still exists
        selected = state.get("selected_path", "")
        if selected:
            p = Path(selected)
            if multi and p.is_dir():
                self._selected_path = selected
                self._path_label.setText(selected)
                self._path_label.setToolTip(selected)
                self._update_status()
            elif not multi and p.is_file():
                self._selected_path = selected
                self._path_label.setText(p.name)
                self._path_label.setToolTip(selected)
                self._start_btn.setEnabled(True)

    def set_main_window(self, main_window) -> None:
        self._main_window_ref = main_window

    # --- Output mode ---

    @Slot(int)
    def _on_output_mode_changed(self, index: int) -> None:
        mode = OUTPUT_MODES[index][1]
        self._custom_dir_btn.setVisible(mode == "save_to_custom")

    @Slot()
    def _select_custom_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._custom_output_dir = d
            self._custom_dir_btn.setText(f"Dir: {Path(d).name}")
            self._custom_dir_btn.setToolTip(d)

    # --- Status update ---

    @Slot()
    def _update_status(self) -> None:
        self._start_btn.setEnabled(bool(self._selected_path) and not self._is_processing)

    # --- Start / Stop ---

    @Slot()
    def _on_start(self) -> None:
        if not self._selected_path:
            return

        mode_idx = self._output_mode.currentIndex()
        mode = OUTPUT_MODES[mode_idx][1]
        fmt = self._format_combo.currentText()
        batch_size = self._batch_size.value()

        output_dir = ""
        if mode == "save_to_custom":
            output_dir = self._custom_output_dir
            if not output_dir:
                QMessageBox.warning(self, "No Directory", "Please select an output directory first.")
                return

        if self._is_single_mode():
            self._is_processing = True
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._status_label.setText("Transcribing...")
            self.transcribe_file_requested.emit(
                self._selected_path, batch_size, mode, fmt, output_dir
            )
        else:
            extensions = [ext for ext, on in self._ext_checked.items() if on]
            files = self._scanner.scan_directory(
                Path(self._selected_path), extensions, self._recursive_cb.isChecked()
            )
            if not files:
                QMessageBox.information(self, "No Files", "No matching files found.")
                return

            confirm = QMessageBox.question(
                self, "Start Batch",
                f"{len(files)} files found. Proceed?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return

            self._is_processing = True
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._status_label.setText("Starting...")

            # Get task_mode from main window
            task_mode = "transcribe"
            try:
                main_win = self._main_window_ref
                if main_win and hasattr(main_win, "task_mode"):
                    task_mode = main_win.task_mode
            except Exception:
                pass

            self.batch_start_requested.emit(files, fmt, output_dir, batch_size, task_mode)

    @Slot()
    def _on_stop(self) -> None:
        self.batch_stop_requested.emit()

    @Slot(int, int, str)
    def update_batch_progress(self, current: int, total: int, message: str) -> None:
        self._status_label.setText(f"[{current}/{total}] {message}")

    @Slot(str)
    def on_batch_completed(self, message: str) -> None:
        self._is_processing = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_label.setText(message)

    @Slot(str)
    def on_batch_error(self, message: str) -> None:
        self._status_label.setText(f"Error: {message}")

    def on_single_file_done(self) -> None:
        self._is_processing = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_label.setText("Done")

    # --- Docking (mirrors ClipboardSideWindow) ---

    def _stop_animation(self) -> None:
        if self._animation:
            was_running = self._animation.state() == QPropertyAnimation.Running
            self._animation.stop()
            self._animation.deleteLater()
            self._animation = None
            if was_running:
                self._end_internal_move()

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
        x = host_rect.left() - gap - w
        return QRect(x, host_rect.top(), w, h)

    def _run_animation(self, start: QRect, end: QRect, duration: int,
                       curve: QEasingCurve.Type, on_finish: list | None = None) -> None:
        self._stop_animation()
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
        start_rect = QRect(end_rect.left() - 30, end_rect.top(), end_rect.width(), end_rect.height())

        self._begin_internal_move()
        self.setGeometry(start_rect)
        self.show()
        self.set_docked(True)

        if animate:
            self._run_animation(start_rect, end_rect, 160, QEasingCurve.OutCubic)
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
        end_rect = QRect(current.left() - 30, current.top(), current.width(), current.height())

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

    def moveEvent(self, event: QMoveEvent) -> None:
        super().moveEvent(event)
        if not self._internal_move and self._docked and self.isVisible():
            self.set_docked(False)
            logger.debug("File panel detached by user drag")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._internal_move and self._docked and self.isVisible():
            self.set_docked(False)
            logger.debug("File panel detached by user resize")

    def closeEvent(self, event):
        self.user_closed.emit()
        event.ignore()
        self.hide()

    @Slot()
    def _request_dock(self) -> None:
        if self._last_host_rect:
            self.dock_to_host(self._last_host_rect, gap=10, animate=True)
