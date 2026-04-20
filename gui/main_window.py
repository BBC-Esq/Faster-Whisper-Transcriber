from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Slot, QRect, QEvent, Signal, QSettings, QByteArray, QTimer, QPointF, QSize
from PySide6.QtGui import QIcon, QPixmap, QPainter, QPen, QColor, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QMessageBox,
    QFileDialog,
    QSizePolicy,
    QDialog,
    QDialogButtonBox,
    QSpinBox,
    QProgressBar,
)
from PySide6.QtSvg import QSvgRenderer

from core.quantization import CheckQuantizationSupport
from core.controller import TranscriberController
from core.models.metadata import ModelMetadata
from core.audio.device_utils import find_device_id_by_name
from core.hotkeys import GlobalHotkey
from core.output.writers import write_output
from core.monitoring.collectors import MetricsCollector
from core.server.server_manager import ServerManager
from config.manager import config_manager
from config.server_settings import TranscriptionSettings as ServerTranscriptionSettings
from gui.styles import update_button_property
from gui.clipboard_window import ClipboardSideWindow
from gui.file_panel import FilePanelWindow
from gui.settings_dialog import SettingsDialog
from gui.visualizations import WaveformButton
from core.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac", ".amr", ".asf", ".avi", ".flac", ".m4a",
    ".mkv", ".mp3", ".mp4", ".wav", ".webm", ".wma",
}

SETTINGS_GEOMETRY = "window/geometry"
SETTINGS_CLIPBOARD_GEOMETRY = "clipboard/geometry"
SETTINGS_CLIPBOARD_DOCKED = "clipboard/docked"
SETTINGS_CLIPBOARD_VISIBLE = "clipboard/visible"
SETTINGS_CLIPBOARD_ALWAYS_ON_TOP = "clipboard/always_on_top"
SETTINGS_APPEND_MODE = "clipboard/append_mode"
SETTINGS_MODEL = "model/name"
SETTINGS_DEVICE = "model/device"
SETTINGS_QUANTIZATION = "model/quantization"
SETTINGS_TASK_MODE = "model/task_mode"
SETTINGS_AUDIO_DEVICE_NAME = "audio/device_name"
SETTINGS_AUDIO_DEVICE_HOSTAPI = "audio/device_hostapi"
SETTINGS_FILE_PANEL_VISIBLE = "file_panel/visible"
SETTINGS_FILE_PANEL_DOCKED = "file_panel/docked"
SETTINGS_FILE_PANEL_GEOMETRY = "file_panel/geometry"
SETTINGS_FILE_PANEL_MULTI_MODE = "file_panel/multi_mode"
SETTINGS_FILE_PANEL_RECURSIVE = "file_panel/recursive"
SETTINGS_FILE_PANEL_BATCH_SIZE = "file_panel/batch_size"
SETTINGS_FILE_PANEL_FORMAT = "file_panel/format"
SETTINGS_FILE_PANEL_OUTPUT_MODE = "file_panel/output_mode"
SETTINGS_FILE_PANEL_CUSTOM_DIR = "file_panel/custom_output_dir"
SETTINGS_FILE_PANEL_EXT_CHECKED = "file_panel/ext_checked"
SETTINGS_FILE_PANEL_SELECTED_PATH = "file_panel/selected_path"

_DOCK_GAP = 10

_DEFAULT_MAIN_WIDTH = 320
_DEFAULT_MAIN_HEIGHT = 190
_DEFAULT_CLIPBOARD_WIDTH = 250
_DEFAULT_CLIPBOARD_HEIGHT = 160


def _create_settings_icon():
    svg = b"""
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path fill-rule="evenodd" clip-rule="evenodd" d="M12 8.25C9.92894 8.25 8.25 9.92893 8.25 12C8.25 14.0711 9.92894 15.75 12 15.75C14.0711 15.75 15.75 14.0711 15.75 12C15.75 9.92893 14.0711 8.25 12 8.25ZM9.75 12C9.75 10.7574 10.7574 9.75 12 9.75C13.2426 9.75 14.25 10.7574 14.25 12C14.25 13.2426 13.2426 14.25 12 14.25C10.7574 14.25 9.75 13.2426 9.75 12Z" fill="#c8c8c8"/>
        <path fill-rule="evenodd" clip-rule="evenodd" d="M11.9747 1.25C11.5303 1.24999 11.1592 1.24999 10.8546 1.27077C10.5375 1.29241 10.238 1.33905 9.94761 1.45933C9.27379 1.73844 8.73843 2.27379 8.45932 2.94762C8.31402 3.29842 8.27467 3.66812 8.25964 4.06996C8.24756 4.39299 8.08454 4.66251 7.84395 4.80141C7.60337 4.94031 7.28845 4.94673 7.00266 4.79568C6.64714 4.60777 6.30729 4.45699 5.93083 4.40743C5.20773 4.31223 4.47642 4.50819 3.89779 4.95219C3.64843 5.14353 3.45827 5.3796 3.28099 5.6434C3.11068 5.89681 2.92517 6.21815 2.70294 6.60307L2.67769 6.64681C2.45545 7.03172 2.26993 7.35304 2.13562 7.62723C1.99581 7.91267 1.88644 8.19539 1.84541 8.50701C1.75021 9.23012 1.94617 9.96142 2.39016 10.5401C2.62128 10.8412 2.92173 11.0602 3.26217 11.2741C3.53595 11.4461 3.68788 11.7221 3.68786 12C3.68785 12.2778 3.53592 12.5538 3.26217 12.7258C2.92169 12.9397 2.62121 13.1587 2.39007 13.4599C1.94607 14.0385 1.75012 14.7698 1.84531 15.4929C1.88634 15.8045 1.99571 16.0873 2.13552 16.3727C2.26983 16.6469 2.45535 16.9682 2.67758 17.3531L2.70284 17.3969C2.92507 17.7818 3.11058 18.1031 3.28089 18.3565C3.45817 18.6203 3.64833 18.8564 3.89769 19.0477C4.47632 19.4917 5.20763 19.6877 5.93073 19.5925C6.30717 19.5429 6.647 19.3922 7.0025 19.2043C7.28833 19.0532 7.60329 19.0596 7.8439 19.1986C8.08452 19.3375 8.24756 19.607 8.25964 19.9301C8.27467 20.3319 8.31403 20.7016 8.45932 21.0524C8.73843 21.7262 9.27379 22.2616 9.94761 22.5407C10.238 22.661 10.5375 22.7076 10.8546 22.7292C11.1592 22.75 11.5303 22.75 11.9747 22.75H12.0252C12.4697 22.75 12.8407 22.75 13.1454 22.7292C13.4625 22.7076 13.762 22.661 14.0524 22.5407C14.7262 22.2616 15.2616 21.7262 15.5407 21.0524C15.686 20.7016 15.7253 20.3319 15.7403 19.93C15.7524 19.607 15.9154 19.3375 16.156 19.1985C16.3966 19.0596 16.7116 19.0532 16.9974 19.2042C17.3529 19.3921 17.6927 19.5429 18.0692 19.5924C18.7923 19.6876 19.5236 19.4917 20.1022 19.0477C20.3516 18.8563 20.5417 18.6203 20.719 18.3565C20.8893 18.1031 21.0748 17.7818 21.297 17.3969L21.3223 17.3531C21.5445 16.9682 21.7301 16.6468 21.8644 16.3726C22.0042 16.0872 22.1135 15.8045 22.1546 15.4929C22.2498 14.7697 22.0538 14.0384 21.6098 13.4598C21.3787 13.1586 21.0782 12.9397 20.7378 12.7258C20.464 12.5538 20.3121 12.2778 20.3121 11.9999C20.3121 11.7221 20.464 11.4462 20.7377 11.2742C21.0783 11.0603 21.3788 10.8414 21.6099 10.5401C22.0539 9.96149 22.2499 9.23019 22.1547 8.50708C22.1136 8.19546 22.0043 7.91274 21.8645 7.6273C21.7302 7.35313 21.5447 7.03183 21.3224 6.64695L21.2972 6.60318C21.0749 6.21825 20.8894 5.89688 20.7191 5.64347C20.5418 5.37967 20.3517 5.1436 20.1023 4.95225C19.5237 4.50826 18.7924 4.3123 18.0692 4.4075C17.6928 4.45706 17.353 4.60782 16.9975 4.79572C16.7117 4.94679 16.3967 4.94036 16.1561 4.80144C15.9155 4.66253 15.7524 4.39297 15.7403 4.06991C15.7253 3.66808 15.686 3.2984 15.5407 2.94762C15.2616 2.27379 14.7262 1.73844 14.0524 1.45933C13.762 1.33905 13.4625 1.29241 13.1454 1.27077C12.8407 1.24999 12.4697 1.24999 12.0252 1.25H11.9747ZM10.5216 2.84515C10.5988 2.81319 10.716 2.78372 10.9567 2.76729C11.2042 2.75041 11.5238 2.75 12 2.75C12.4762 2.75 12.7958 2.75041 13.0432 2.76729C13.284 2.78372 13.4012 2.81319 13.4783 2.84515C13.7846 2.97202 14.028 3.21536 14.1548 3.52165C14.1949 3.61826 14.228 3.76887 14.2414 4.12597C14.271 4.91835 14.68 5.68129 15.4061 6.10048C16.1321 6.51968 16.9974 6.4924 17.6984 6.12188C18.0143 5.9549 18.1614 5.90832 18.265 5.89467C18.5937 5.8514 18.9261 5.94047 19.1891 6.14228C19.2554 6.19312 19.3395 6.27989 19.4741 6.48016C19.6125 6.68603 19.7726 6.9626 20.0107 7.375C20.2488 7.78741 20.4083 8.06438 20.5174 8.28713C20.6235 8.50382 20.6566 8.62007 20.6675 8.70287C20.7108 9.03155 20.6217 9.36397 20.4199 9.62698C20.3562 9.70995 20.2424 9.81399 19.9397 10.0041C19.2684 10.426 18.8122 11.1616 18.8121 11.9999C18.8121 12.8383 19.2683 13.574 19.9397 13.9959C20.2423 14.186 20.3561 14.29 20.4198 14.373C20.6216 14.636 20.7107 14.9684 20.6674 15.2971C20.6565 15.3799 20.6234 15.4961 20.5173 15.7128C20.4082 15.9355 20.2487 16.2125 20.0106 16.6249C19.7725 17.0373 19.6124 17.3139 19.474 17.5198C19.3394 17.72 19.2553 17.8068 19.189 17.8576C18.926 18.0595 18.5936 18.1485 18.2649 18.1053C18.1613 18.0916 18.0142 18.045 17.6983 17.8781C16.9973 17.5075 16.132 17.4803 15.4059 17.8995C14.68 18.3187 14.271 19.0816 14.2414 19.874C14.228 20.2311 14.1949 20.3817 14.1548 20.4784C14.028 20.7846 13.7846 21.028 13.4783 21.1549C13.4012 21.1868 13.284 21.2163 13.0432 21.2327C12.7958 21.2496 12.4762 21.25 12 21.25C11.5238 21.25 11.2042 21.2496 10.9567 21.2327C10.716 21.2163 10.5988 21.1868 10.5216 21.1549C10.2154 21.028 9.97201 20.7846 9.84514 20.4784C9.80512 20.3817 9.77195 20.2311 9.75859 19.874C9.72896 19.0817 9.31997 18.3187 8.5939 17.8995C7.86784 17.4803 7.00262 17.5076 6.30158 17.8781C5.98565 18.0451 5.83863 18.0917 5.73495 18.1053C5.40626 18.1486 5.07385 18.0595 4.81084 17.8577C4.74458 17.8069 4.66045 17.7201 4.52586 17.5198C4.38751 17.314 4.22736 17.0374 3.98926 16.625C3.75115 16.2126 3.59171 15.9356 3.4826 15.7129C3.37646 15.4962 3.34338 15.3799 3.33248 15.2971C3.28921 14.9684 3.37828 14.636 3.5801 14.373C3.64376 14.2901 3.75761 14.186 4.0602 13.9959C4.73158 13.5741 5.18782 12.8384 5.18786 12.0001C5.18791 11.1616 4.73165 10.4259 4.06021 10.004C3.75769 9.81389 3.64385 9.70987 3.58019 9.62691C3.37838 9.3639 3.28931 9.03149 3.33258 8.7028C3.34348 8.62001 3.37656 8.50375 3.4827 8.28707C3.59181 8.06431 3.75125 7.78734 3.98935 7.37493C4.22746 6.96253 4.3876 6.68596 4.52596 6.48009C4.66055 6.27983 4.74468 6.19305 4.81093 6.14222C5.07395 5.9404 5.40636 5.85133 5.73504 5.8946C5.83873 5.90825 5.98576 5.95483 6.30173 6.12184C7.00273 6.49235 7.86791 6.51962 8.59394 6.10045C9.31998 5.68128 9.72896 4.91837 9.75859 4.12602C9.77195 3.76889 9.80512 3.61827 9.84514 3.52165C9.97201 3.21536 10.2154 2.97202 10.5216 2.84515Z" fill="#c8c8c8"/>
    </svg>
    """
    px = QPixmap(38, 38)
    px.fill(Qt.transparent)
    renderer = QSvgRenderer(QByteArray(svg))
    p = QPainter(px)
    renderer.render(p)
    p.end()
    return QIcon(px)


def _create_file_icon():
    svg = b"""
    <svg fill="#c8c8c8" version="1.1" xmlns="http://www.w3.org/2000/svg"
         viewBox="796 796 200 200" enable-background="new 796 796 200 200">
        <g>
            <path d="M976.83,857.4l-56.941-56.941c-2.875-2.875-6.695-4.458-10.762-4.458H825.93c-8.393,0-15.218,6.828-15.218,15.222v169.56
                c0,8.393,6.825,15.219,15.218,15.219h140.14c8.391,0,15.218-6.826,15.218-15.219V868.162
                C981.288,864.099,979.705,860.275,976.83,857.4z M969.676,980.781c0,1.989-1.615,3.609-3.604,3.609H825.931
                c-1.989,0-3.605-1.62-3.605-3.609V811.22c0-1.987,1.616-3.605,3.605-3.605h79.408c2.174,0,3.937,1.763,3.937,3.936v42.937
                c0,7.25,5.876,13.126,13.123,13.126h43.342c1.045,0,2.046,0.414,2.784,1.152c0.737,0.739,1.152,1.74,1.152,2.783L969.676,980.781z"/>
            <path d="M896.116,875.189c-1.452-0.62-2.988-0.934-4.566-0.934c-3.022,0-5.887,1.152-8.067,3.246l-20.942,20.108h-14.479
                c-6.421,0-11.646,5.225-11.646,11.646v23.218c0,6.422,5.224,11.646,11.646,11.646h14.479l20.942,20.108
                c2.18,2.091,5.045,3.243,8.066,3.243c1.578,0,3.112-0.313,4.565-0.932c4.301-1.834,7.079-6.038,7.079-10.714v-69.923
                C903.194,881.227,900.416,877.021,896.116,875.189z M891.261,955.15l-20.539-19.722c-2.18-2.091-5.044-3.243-8.064-3.243H848.35
                v-22.643h14.308c3.02,0,5.884-1.152,8.065-3.245l20.538-19.72V955.15z"/>
            <path d="M915.962,892.81c-2.873,1.613-3.896,5.251-2.281,8.124c3.396,6.048,5.192,12.939,5.192,19.93
                c0,6.989-1.796,13.881-5.192,19.93c-1.613,2.872-0.592,6.511,2.281,8.124c0.923,0.518,1.927,0.765,2.916,0.765
                c2.087,0,4.112-1.097,5.208-3.047c4.396-7.827,6.721-16.74,6.721-25.771c0-9.032-2.324-17.945-6.721-25.773
                C922.471,892.217,918.834,891.195,915.962,892.81z"/>
            <path d="M936.403,879.753c-2.845,1.664-3.802,5.318-2.14,8.163c5.82,9.953,8.897,21.346,8.897,32.948
                c0,11.601-3.077,22.994-8.897,32.947c-1.662,2.845-0.705,6.499,2.14,8.163c0.946,0.553,1.983,0.816,3.007,0.816
                c2.049,0,4.047-1.058,5.156-2.956c6.887-11.778,10.528-25.254,10.528-38.971s-3.642-27.193-10.528-38.972
                C942.902,879.045,939.246,878.086,936.403,879.753z"/>
        </g>
    </svg>
    """
    px = QPixmap(38, 38)
    px.fill(Qt.transparent)
    renderer = QSvgRenderer(QByteArray(svg))
    p = QPainter(px)
    renderer.render(p)
    p.end()
    return QIcon(px)


def _create_clipboard_icon():
    svg = b"""
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M16 4.00195C18.175 4.01406 19.3529 4.11051 20.1213 4.87889C21 5.75757 21 7.17179 21 10.0002V16.0002C21 18.8286 21 20.2429 20.1213 21.1215C19.2426 22.0002 17.8284 22.0002 15 22.0002H9C6.17157 22.0002 4.75736 22.0002 3.87868 21.1215C3 20.2429 3 18.8286 3 16.0002V10.0002C3 7.17179 3 5.75757 3.87868 4.87889C4.64706 4.11051 5.82497 4.01406 8 4.00195" stroke="#c8c8c8" stroke-width="1.5"/>
        <path d="M8 14H16" stroke="#c8c8c8" stroke-width="1.5" stroke-linecap="round"/>
        <path d="M7 10.5H17" stroke="#c8c8c8" stroke-width="1.5" stroke-linecap="round"/>
        <path d="M9 17.5H15" stroke="#c8c8c8" stroke-width="1.5" stroke-linecap="round"/>
        <path d="M8 3.5C8 2.67157 8.67157 2 9.5 2H14.5C15.3284 2 16 2.67157 16 3.5V4.5C16 5.32843 15.3284 6 14.5 6H9.5C8.67157 6 8 5.32843 8 4.5V3.5Z" stroke="#c8c8c8" stroke-width="1.5"/>
    </svg>
    """
    px = QPixmap(38, 38)
    px.fill(Qt.transparent)
    renderer = QSvgRenderer(QByteArray(svg))
    p = QPainter(px)
    renderer.render(p)
    p.end()
    return QIcon(px)


class MainWindow(QMainWindow):
    hotkey_toggle_recording = Signal()

    DEFAULTS = {
        "model": "base.en",
        "device": "cpu",
        "quantization": "float32",
        "task_mode": "transcribe",
        "append_mode": False,
        "clipboard_visible": False,
        "clipboard_always_on_top": True,
        "clipboard_docked": True,
        "file_panel_visible": False,
        "file_panel_docked": True,
    }

    def __init__(self, cuda_available: bool = False):
        super().__init__()

        self.setWindowTitle("Faster Whisper Transcriber")

        self.settings = QSettings("FasterWhisperTranscriber", "Transcriber")

        self.loaded_model_settings = config_manager.get_model_settings()
        self.task_mode = config_manager.get_value("task_mode", "transcribe")
        audio_device_id = self._resolve_audio_device()
        self.controller = TranscriberController(audio_device_id=audio_device_id)
        self.server_manager = ServerManager(self)
        self.supported_quantizations: dict[str, list[str]] = {"cpu": [], "cuda": []}
        self.is_recording = False
        self.cuda_available = cuda_available
        self._clipboard_visible = False
        self._file_panel_visible = False
        self._toggleable_widgets: list[QWidget] = []
        self._model_is_loaded = False
        self._is_loading_model = False
        self._download_total_bytes = 0
        self._server_mode_enabled = bool(config_manager.get_value("server_mode_enabled", False))
        self._server_port = int(config_manager.get_value("server_port", 8765))

        self.clipboard_window = ClipboardSideWindow(None, width=_DEFAULT_CLIPBOARD_WIDTH)
        self.clipboard_window.user_closed.connect(self._on_clipboard_closed)
        self.clipboard_window.docked_changed.connect(
            lambda d: logger.debug(f"Clipboard docked: {d}")
        )
        self.clipboard_window.always_on_top_changed.connect(
            self._on_clipboard_always_on_top_changed
        )

        self.file_panel = FilePanelWindow(None, width=340)
        self.file_panel.user_closed.connect(self._on_file_panel_closed)
        self.file_panel.docked_changed.connect(
            lambda d: logger.debug(f"File panel docked: {d}")
        )
        self.file_panel.transcribe_file_requested.connect(
            self._on_file_panel_transcribe
        )
        self.file_panel.batch_start_requested.connect(self._on_batch_start)
        self.file_panel.batch_stop_requested.connect(self._on_batch_stop)

        self._metrics_collector = MetricsCollector(interval_ms=1000)
        self._metrics_collector.metrics_updated.connect(self._on_metrics_updated)

        self.server_manager.server_started.connect(self._on_server_started)
        self.server_manager.server_stopped.connect(self._on_server_stopped)
        self.server_manager.server_error.connect(self._on_server_error)

        self._build_ui()
        self._setup_connections()

        self._load_quantization_support()

        self.set_widgets_enabled(False)

        self._restore_state()

        self.hotkey_toggle_recording.connect(self._toggle_recording, Qt.QueuedConnection)
        self.global_hotkey = GlobalHotkey(lambda: self.hotkey_toggle_recording.emit())
        self.global_hotkey.start()

        self._sample_timer = QTimer(self)
        self._sample_timer.setInterval(50)
        self._sample_timer.timeout.connect(self._feed_audio_samples)

        self.setAcceptDrops(True)
        if app := QApplication.instance():
            app.installEventFilter(self)

        self._metrics_collector.start()

        if self._server_mode_enabled:
            self._start_server_mode(self._server_port)

        logger.info("MainWindow initialized")

    def _load_quantization_support(self) -> None:
        try:
            CheckQuantizationSupport().update_supported_quantizations()
            config = config_manager.load_config()
            self.supported_quantizations = config.get(
                "supported_quantizations", {"cpu": [], "cuda": []}
            )

            if not self.supported_quantizations.get("cpu"):
                self.supported_quantizations["cpu"] = ["float32"]
        except Exception as e:
            logger.error(f"Failed to load precision support: {e}")
            self.supported_quantizations = {"cpu": ["float32"], "cuda": []}

    def _validate_model(self, model_name: str) -> str:
        valid_models = ModelMetadata.get_all_model_names()
        if model_name in valid_models:
            return model_name
        logger.warning(f"Invalid model '{model_name}', falling back to default")
        return self.DEFAULTS["model"]

    def _validate_device(self, device: str) -> str:
        if device == "cuda" and not self.cuda_available:
            logger.warning("CUDA no longer available, falling back to CPU")
            return "cpu"
        if device in ["cpu", "cuda"]:
            return device
        logger.warning(f"Invalid device '{device}', falling back to default")
        return self.DEFAULTS["device"]

    def _validate_quantization(
        self, quantization: str, model_name: str, device: str
    ) -> str:
        available = ModelMetadata.get_quantization_options(
            model_name, device, self.supported_quantizations
        )
        if quantization in available:
            return quantization
        if available:
            logger.warning(
                f"Precision '{quantization}' not available for "
                f"{model_name}/{device}, using {available[0]}"
            )
            return available[0]
        logger.warning(
            f"No quantizations available for {model_name}/{device}, using float32"
        )
        return "float32"

    def _validate_task_mode(self, task_mode: str, model_name: str) -> str:
        if task_mode == "translate" and not ModelMetadata.supports_translation(
            model_name
        ):
            logger.warning(
                f"Model '{model_name}' doesn't support translation, "
                f"falling back to transcribe"
            )
            return "transcribe"
        if task_mode in ["transcribe", "translate"]:
            return task_mode
        return self.DEFAULTS["task_mode"]

    def _restore_state(self) -> None:
        logger.info("Restoring application state from QSettings")

        geometry = self.settings.value(SETTINGS_GEOMETRY)
        if geometry and isinstance(geometry, QByteArray):
            if not self.restoreGeometry(geometry):
                logger.warning(
                    "Failed to restore main window geometry, using defaults"
                )
                self.resize(_DEFAULT_MAIN_WIDTH, _DEFAULT_MAIN_HEIGHT)
                self._center_on_screen()
        else:
            self.resize(_DEFAULT_MAIN_WIDTH, _DEFAULT_MAIN_HEIGHT)
            self._center_on_screen()

        saved_model = self.settings.value(SETTINGS_MODEL, self.DEFAULTS["model"])
        saved_device = self.settings.value(SETTINGS_DEVICE, self.DEFAULTS["device"])
        saved_quant = self.settings.value(
            SETTINGS_QUANTIZATION, self.DEFAULTS["quantization"]
        )
        saved_task = self.settings.value(
            SETTINGS_TASK_MODE, self.DEFAULTS["task_mode"]
        )

        model = self._validate_model(saved_model)
        device = self._validate_device(saved_device)
        quantization = self._validate_quantization(saved_quant, model, device)
        task_mode = self._validate_task_mode(saved_task, model)

        self.loaded_model_settings = {
            "model_name": model,
            "quantization_type": quantization,
            "device_type": device,
        }

        self.task_mode = task_mode
        self._update_translation_availability(model)

        append_mode = self.settings.value(
            SETTINGS_APPEND_MODE, self.DEFAULTS["append_mode"], type=bool
        )
        self.clipboard_window.set_append_mode(append_mode)

        always_on_top = self.settings.value(
            SETTINGS_CLIPBOARD_ALWAYS_ON_TOP,
            self.DEFAULTS["clipboard_always_on_top"],
            type=bool,
        )
        self.clipboard_window.set_always_on_top(always_on_top)

        clipboard_docked = self.settings.value(
            SETTINGS_CLIPBOARD_DOCKED, self.DEFAULTS["clipboard_docked"], type=bool
        )
        self._clipboard_visible = self.settings.value(
            SETTINGS_CLIPBOARD_VISIBLE,
            self.DEFAULTS["clipboard_visible"],
            type=bool,
        )

        if self._clipboard_visible:
            if clipboard_docked:
                self.clipboard_window.show_docked(
                    self._host_rect_global(), gap=_DOCK_GAP, animate=False
                )
            else:
                clip_geometry = self.settings.value(SETTINGS_CLIPBOARD_GEOMETRY)
                if clip_geometry and isinstance(clip_geometry, QByteArray):
                    self.clipboard_window.set_docked(False)
                    self.clipboard_window.restoreGeometry(clip_geometry)
                    self.clipboard_window.show()
                else:
                    self.clipboard_window.show_docked(
                        self._host_rect_global(), gap=_DOCK_GAP, animate=False
                    )

        self._update_clipboard_button_text()

        # Restore file panel state
        file_panel_docked = self.settings.value(
            SETTINGS_FILE_PANEL_DOCKED, self.DEFAULTS["file_panel_docked"], type=bool
        )
        self._file_panel_visible = self.settings.value(
            SETTINGS_FILE_PANEL_VISIBLE,
            self.DEFAULTS["file_panel_visible"],
            type=bool,
        )

        if self._file_panel_visible:
            if file_panel_docked:
                self.file_panel.show_docked(
                    self._host_rect_global(), gap=_DOCK_GAP, animate=False
                )
            else:
                fp_geometry = self.settings.value(SETTINGS_FILE_PANEL_GEOMETRY)
                if fp_geometry and isinstance(fp_geometry, QByteArray):
                    self.file_panel.set_docked(False)
                    self.file_panel.restoreGeometry(fp_geometry)
                    self.file_panel.show()
                else:
                    self.file_panel.show_docked(
                        self._host_rect_global(), gap=_DOCK_GAP, animate=False
                    )

        self._update_file_panel_button_text()

        # Restore file panel internal settings
        self._restore_file_panel_settings()

        try:
            config_manager.set_model_settings(model, quantization, device)
            config_manager.set_value("task_mode", task_mode)
            config_manager.set_value("clipboard_append_mode", append_mode)
            config_manager.set_value("show_clipboard_window", self._clipboard_visible)
        except Exception as e:
            logger.warning(f"Failed to sync config manager: {e}")

        logger.info(
            f"State restored: model={model}, device={device}, "
            f"quant={quantization}, task={task_mode}"
        )

        self._update_model_status("No model loaded")
        self.controller.update_model(model, quantization, device)

    def _save_state(self) -> None:
        logger.info("Saving application state to QSettings")

        self.settings.setValue(SETTINGS_GEOMETRY, self.saveGeometry())

        self.settings.setValue(
            SETTINGS_MODEL,
            self.loaded_model_settings.get("model_name", self.DEFAULTS["model"]),
        )
        self.settings.setValue(
            SETTINGS_DEVICE,
            self.loaded_model_settings.get("device_type", self.DEFAULTS["device"]),
        )
        self.settings.setValue(
            SETTINGS_QUANTIZATION,
            self.loaded_model_settings.get(
                "quantization_type", self.DEFAULTS["quantization"]
            ),
        )
        self.settings.setValue(SETTINGS_TASK_MODE, self.task_mode)

        self.settings.setValue(
            SETTINGS_APPEND_MODE, self.clipboard_window.is_append_mode()
        )
        self.settings.setValue(SETTINGS_CLIPBOARD_VISIBLE, self._clipboard_visible)
        self.settings.setValue(
            SETTINGS_CLIPBOARD_ALWAYS_ON_TOP,
            self.clipboard_window.is_always_on_top(),
        )
        self.settings.setValue(
            SETTINGS_CLIPBOARD_DOCKED, self.clipboard_window.is_docked()
        )

        if not self.clipboard_window.is_docked():
            self.settings.setValue(
                SETTINGS_CLIPBOARD_GEOMETRY, self.clipboard_window.saveGeometry()
            )

        # Save file panel state
        self.settings.setValue(SETTINGS_FILE_PANEL_VISIBLE, self._file_panel_visible)
        self.settings.setValue(
            SETTINGS_FILE_PANEL_DOCKED, self.file_panel.is_docked()
        )
        if not self.file_panel.is_docked():
            self.settings.setValue(
                SETTINGS_FILE_PANEL_GEOMETRY, self.file_panel.saveGeometry()
            )

        # Save file panel internal settings
        self._save_file_panel_settings()

        self.settings.sync()
        logger.debug("State saved successfully")

    def _save_file_panel_settings(self) -> None:
        state = self.file_panel.get_panel_state()
        self.settings.setValue(SETTINGS_FILE_PANEL_MULTI_MODE, state["multi_mode"])
        self.settings.setValue(SETTINGS_FILE_PANEL_RECURSIVE, state["recursive"])
        self.settings.setValue(SETTINGS_FILE_PANEL_BATCH_SIZE, state["batch_size"])
        self.settings.setValue(SETTINGS_FILE_PANEL_FORMAT, state["format"])
        self.settings.setValue(SETTINGS_FILE_PANEL_OUTPUT_MODE, state["output_mode_index"])
        self.settings.setValue(SETTINGS_FILE_PANEL_CUSTOM_DIR, state["custom_output_dir"])
        self.settings.setValue(SETTINGS_FILE_PANEL_SELECTED_PATH, state["selected_path"])
        self.settings.setValue(
            SETTINGS_FILE_PANEL_EXT_CHECKED,
            json.dumps(self.file_panel.get_ext_checked()),
        )

    def _restore_file_panel_settings(self) -> None:
        state = {
            "multi_mode": self.settings.value(SETTINGS_FILE_PANEL_MULTI_MODE, False, type=bool),
            "recursive": self.settings.value(SETTINGS_FILE_PANEL_RECURSIVE, False, type=bool),
            "batch_size": self.settings.value(SETTINGS_FILE_PANEL_BATCH_SIZE, 16, type=int),
            "format": self.settings.value(SETTINGS_FILE_PANEL_FORMAT, "txt"),
            "output_mode_index": self.settings.value(SETTINGS_FILE_PANEL_OUTPUT_MODE, 0, type=int),
            "custom_output_dir": self.settings.value(SETTINGS_FILE_PANEL_CUSTOM_DIR, ""),
            "selected_path": self.settings.value(SETTINGS_FILE_PANEL_SELECTED_PATH, ""),
        }
        self.file_panel.restore_panel_state(state)

        ext_json = self.settings.value(SETTINGS_FILE_PANEL_EXT_CHECKED, "")
        if ext_json:
            try:
                ext_checked = json.loads(ext_json)
                self.file_panel.set_ext_checked(ext_checked)
            except (json.JSONDecodeError, TypeError):
                pass

    def _center_on_screen(self) -> None:
        if screen := QApplication.primaryScreen():
            screen_geometry = screen.availableGeometry()
            x = (screen_geometry.width() - self.width()) // 2 + screen_geometry.x()
            y = (screen_geometry.height() - self.height()) // 2 + screen_geometry.y()
            self.move(x, y)

    def _on_clipboard_always_on_top_changed(self, value: bool) -> None:
        self.settings.setValue(SETTINGS_CLIPBOARD_ALWAYS_ON_TOP, value)

    def _setup_connections(self) -> None:
        self.clipboard_window.append_mode_changed.connect(
            self._on_append_mode_changed
        )

        self.controller.update_button_signal.connect(self._on_button_text_update)
        self.controller.enable_widgets_signal.connect(self.set_widgets_enabled)
        self.controller.text_ready_signal.connect(self._on_transcription_ready)
        self.controller.result_ready_signal.connect(self._on_result_ready)
        self.controller.model_loaded_signal.connect(self._on_model_loaded_success)
        self.controller.error_occurred.connect(self._show_error_dialog)
        self.controller.transcription_cancelled_signal.connect(
            self._on_transcription_cancelled
        )

        self.controller.model_download_started.connect(
            self._on_model_download_started
        )
        self.controller.model_download_progress.connect(
            self._on_model_download_progress
        )
        self.controller.model_download_finished.connect(
            self._on_model_download_finished
        )
        self.controller.model_download_cancelled.connect(
            self._on_model_download_cancelled
        )
        self.controller.model_loading_started.connect(
            self._on_model_loading_started
        )

        self.controller.batch_progress.connect(self.file_panel.update_batch_progress)
        self.controller.batch_completed.connect(self.file_panel.on_batch_completed)
        self.controller.batch_completed.connect(self._on_batch_finished)
        self.controller.batch_error.connect(self.file_panel.on_batch_error)
        self.controller.batch_error.connect(self._on_batch_finished)

    def _build_ui(self) -> None:
        self.menuBar().setVisible(False)

        central = QWidget(self)
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(5, 5, 5, 5)
        root.setSpacing(4)

        root.addWidget(self._build_main_group(), 1)

        # Metrics row
        self.metrics_label = QLabel("")
        self.metrics_label.setAlignment(Qt.AlignCenter)
        self.metrics_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        root.addWidget(self.metrics_label)

        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setMinimumWidth(100)

        self.download_progress_bar = QProgressBar()
        self.download_progress_bar.setFixedWidth(130)
        self.download_progress_bar.setFixedHeight(16)
        self.download_progress_bar.setRange(0, 10000)
        self.download_progress_bar.setValue(0)
        self.download_progress_bar.setTextVisible(False)
        self.download_progress_bar.setVisible(False)

        self.cancel_download_button = QPushButton("Cancel")
        self.cancel_download_button.setFixedWidth(55)
        self.cancel_download_button.setFixedHeight(18)
        self.cancel_download_button.setVisible(False)
        self.cancel_download_button.clicked.connect(self._cancel_model_download)

        self.statusBar().addWidget(self.model_status_label, 1)
        self.statusBar().addPermanentWidget(self.download_progress_bar)
        self.statusBar().addPermanentWidget(self.cancel_download_button)
        self.statusBar().setSizeGripEnabled(True)

        self.resize(_DEFAULT_MAIN_WIDTH, _DEFAULT_MAIN_HEIGHT)
        self.setMinimumSize(_DEFAULT_MAIN_WIDTH, _DEFAULT_MAIN_HEIGHT)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)

    def _build_main_group(self) -> QGroupBox:
        group = QGroupBox("")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        left_col = QVBoxLayout()
        left_col.setSpacing(8)

        self.settings_button = QPushButton()
        self.settings_button.setFlat(True)
        self.settings_button.setIcon(_create_settings_icon())
        self.settings_button.setIconSize(QSize(28, 28))
        self.settings_button.setFixedSize(32, 32)
        self.settings_button.setToolTip("Settings")
        self.settings_button.clicked.connect(self._open_settings_dialog)
        self._register_toggleable_widget(self.settings_button)
        left_col.addWidget(self.settings_button)

        self.transcribe_file_button = QPushButton()
        self.transcribe_file_button.setFlat(True)
        self.transcribe_file_button.setObjectName("fileButton")
        self.transcribe_file_button.setIcon(_create_file_icon())
        self.transcribe_file_button.setIconSize(QSize(28, 28))
        self.transcribe_file_button.setFixedSize(32, 32)
        self.transcribe_file_button.setToolTip("Show File Panel")
        self.transcribe_file_button.clicked.connect(self._toggle_file_panel)
        self._register_toggleable_widget(self.transcribe_file_button)
        left_col.addWidget(self.transcribe_file_button)

        self.clipboard_button = QPushButton()
        self.clipboard_button.setFlat(True)
        self.clipboard_button.setIcon(_create_clipboard_icon())
        self.clipboard_button.setIconSize(QSize(28, 28))
        self.clipboard_button.setFixedSize(32, 32)
        self.clipboard_button.setToolTip("Show Clipboard")
        self.clipboard_button.clicked.connect(self._toggle_clipboard)
        self._register_toggleable_widget(self.clipboard_button)
        left_col.addWidget(self.clipboard_button)

        left_col.addStretch(1)
        layout.addLayout(left_col)

        self.record_button = WaveformButton()
        self.record_button.setObjectName("recordButton")
        self.record_button.setText("Click to Record")
        self.record_button.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.record_button.setToolTip("Click or press F9 to toggle recording")
        self.record_button.clicked.connect(self._toggle_recording)
        self._register_toggleable_widget(self.record_button)
        layout.addWidget(self.record_button, 1)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding
        )
        self.cancel_button.setToolTip("Cancel the current transcription")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel_transcription)
        layout.addWidget(self.cancel_button)

        return group

    def _host_rect_global(self) -> QRect:
        pos = self.mapToGlobal(self.rect().topLeft())
        return QRect(pos.x(), pos.y(), self.width(), self.height())

    def _save_config(self, key: str, value) -> None:
        try:
            config_manager.set_value(key, value)
        except Exception as e:
            logger.warning(f"Failed to save {key}: {e}")

    def _update_translation_availability(self, model_name: str) -> None:
        if not ModelMetadata.supports_translation(model_name) and self.task_mode == "translate":
            self.task_mode = "transcribe"
            self._save_config("task_mode", self.task_mode)
            self.controller.set_task_mode(self.task_mode)

    def _update_model_status(self, text: str) -> None:
        self.model_status_label.setText(text)

    def _show_current_model_status(self) -> None:
        if self._model_is_loaded:
            name = self.loaded_model_settings.get("model_name", "")
            quant = self.loaded_model_settings.get("quantization_type", "")
            device = self.loaded_model_settings.get("device_type", "")
            server_bit = (
                f"  |  server: on ({self._server_port})"
                if self._server_mode_enabled else ""
            )
            self._update_model_status(f"{name} ({quant} / {device}){server_bit}")
        else:
            self._update_model_status("No model loaded")

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        if num_bytes < 1024:
            return f"{num_bytes} B"
        elif num_bytes < 1024 * 1024:
            return f"{num_bytes / 1024:.1f} KB"
        elif num_bytes < 1024 * 1024 * 1024:
            return f"{num_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"

    # --- Metrics ---

    @Slot(object)
    def _on_metrics_updated(self, metrics) -> None:
        parts = [
            f"CPU: {metrics.cpu_usage:.0f}%",
            f"RAM: {metrics.ram_usage_percent:.0f}%",
        ]
        if metrics.gpu_utilization is not None:
            parts.append(f"GPU: {metrics.gpu_utilization:.0f}%")
        if metrics.vram_usage_percent is not None:
            parts.append(f"VRAM: {metrics.vram_usage_percent:.0f}%")
        if metrics.power_usage_percent is not None:
            parts.append(f"Power: {metrics.power_usage_percent:.0f}%")
        self.metrics_label.setText("  |  ".join(parts))

    # --- Model download/loading ---

    @Slot(str, object)
    def _on_model_download_started(self, model_name: str, total_bytes: int) -> None:
        self._is_loading_model = True
        self._download_total_bytes = total_bytes
        self.download_progress_bar.setValue(0)
        self.download_progress_bar.setVisible(True)
        self.cancel_download_button.setVisible(True)
        total_str = self._format_bytes(total_bytes)
        self._update_model_status(f"Downloading {model_name}... 0 B / {total_str}")

    @Slot(object, object)
    def _on_model_download_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            pct = int(downloaded / total * 10000)
            self.download_progress_bar.setValue(pct)
        dl_str = self._format_bytes(downloaded)
        total_str = self._format_bytes(total)
        self._update_model_status(f"Downloading... {dl_str} / {total_str}")

    @Slot(str)
    def _on_model_download_finished(self, model_name: str) -> None:
        self.download_progress_bar.setVisible(False)
        self.cancel_download_button.setVisible(False)
        self._update_model_status(f"Loading {model_name}...")

    @Slot()
    def _on_model_download_cancelled(self) -> None:
        self._is_loading_model = False
        self.download_progress_bar.setVisible(False)
        self.cancel_download_button.setVisible(False)
        self._show_current_model_status()

    @Slot(str)
    def _on_model_loading_started(self, model_name: str) -> None:
        self._is_loading_model = True
        self.download_progress_bar.setVisible(False)
        self.cancel_download_button.setVisible(False)
        self._update_model_status(f"Loading {model_name}...")

    @Slot()
    def _cancel_model_download(self) -> None:
        self.cancel_download_button.setEnabled(False)
        self.controller.cancel_model_loading()

    # --- Settings ---

    def _is_busy(self) -> bool:
        return (
            self.controller.is_transcribing()
            or self.controller.is_batch_processing()
            or self.server_manager.is_transcription_active()
        )

    @Slot()
    def _open_settings_dialog(self) -> None:
        current_audio = {
            "name": self.settings.value(SETTINGS_AUDIO_DEVICE_NAME, ""),
            "hostapi": self.settings.value(SETTINGS_AUDIO_DEVICE_HOSTAPI, ""),
        }
        whisper_settings = {
            "include_timestamps": config_manager.get_value("include_timestamps", False),
            "beam_size": config_manager.get_value("beam_size", 5),
            "vad_filter": config_manager.get_value("vad_filter", False),
            "condition_on_previous_text": config_manager.get_value("condition_on_previous_text", True),
        }
        server_settings = {
            "server_mode_enabled": self._server_mode_enabled,
            "server_port": self._server_port,
        }
        dlg = SettingsDialog(
            parent=self,
            cuda_available=self.cuda_available,
            supported_quantizations=self.supported_quantizations,
            current_settings=self.loaded_model_settings,
            current_task_mode=self.task_mode,
            current_audio_device=current_audio,
            current_whisper_settings=whisper_settings,
            current_ext_checked=self.file_panel.get_ext_checked(),
            current_server_settings=server_settings,
            is_busy_check=self._is_busy,
        )
        dlg.model_update_requested.connect(self._on_settings_update_requested)
        dlg.audio_device_changed.connect(self._on_audio_device_changed)
        dlg.task_mode_changed.connect(self._on_task_mode_changed)
        dlg.whisper_settings_changed.connect(self._on_whisper_settings_changed)
        dlg.file_types_changed.connect(self._on_file_types_changed)
        dlg.server_mode_changed.connect(self._on_server_mode_changed)
        dlg.exec()

    @Slot(str, str, str)
    def _on_settings_update_requested(
        self, model: str, quant: str, device: str
    ) -> None:
        self.controller.update_model(model, quant, device)

    def _resolve_audio_device(self) -> int | None:
        name = self.settings.value(SETTINGS_AUDIO_DEVICE_NAME, "")
        hostapi = self.settings.value(SETTINGS_AUDIO_DEVICE_HOSTAPI, "")
        if not name:
            return None
        device_id = find_device_id_by_name(name, hostapi)
        if device_id is None:
            logger.warning(f"Saved audio device '{name}' not found, using system default")
        return device_id

    @Slot(str, str)
    def _on_audio_device_changed(self, device_name: str, hostapi: str) -> None:
        self.settings.setValue(SETTINGS_AUDIO_DEVICE_NAME, device_name)
        self.settings.setValue(SETTINGS_AUDIO_DEVICE_HOSTAPI, hostapi)
        self.settings.sync()

        if device_name:
            device_id = find_device_id_by_name(device_name, hostapi)
        else:
            device_id = None

        self.controller.set_audio_device(device_id)
        logger.info(f"Audio device changed to: {device_name or 'System Default'}")

    @Slot(str)
    def _on_button_text_update(self, text: str) -> None:
        self.record_button.setText(text)

        if (
            "Transcribing" in text
            and "Done" not in text
            and "Failed" not in text
            and "Cancelled" not in text
        ):
            self.record_button.set_state(WaveformButton.TRANSCRIBING)
            self.cancel_button.setVisible(True)
            self.cancel_button.setEnabled(True)

    @Slot()
    def _on_clipboard_closed(self) -> None:
        self._clipboard_visible = False
        self._update_clipboard_button_text()
        self._save_config("show_clipboard_window", False)

    @Slot()
    def _on_file_panel_closed(self) -> None:
        self._file_panel_visible = False
        self._update_file_panel_button_text()

    @Slot(bool)
    def _on_append_mode_changed(self, checked: bool) -> None:
        self._save_config("clipboard_append_mode", checked)

    @Slot(str, str)
    def _show_error_dialog(self, title: str, message: str) -> None:
        logger.error(f"Error: {title} - {message}")
        self._is_loading_model = False
        self.download_progress_bar.setVisible(False)
        self.cancel_download_button.setVisible(False)
        self.cancel_download_button.setEnabled(True)
        self._show_current_model_status()
        self.file_panel.on_single_file_done()
        if "model" in title.lower():
            QMessageBox.critical(self, title, message)
        else:
            QMessageBox.warning(self, title, message)

    @Slot(str)
    def _on_task_mode_changed(self, mode: str) -> None:
        self.task_mode = mode
        self._save_config("task_mode", self.task_mode)
        self.controller.set_task_mode(self.task_mode)

    def _on_whisper_settings_changed(self, settings: dict) -> None:
        for key, value in settings.items():
            self._save_config(key, value)
        # Translate include_timestamps to the params the transcription service expects
        whisper_params = {
            "without_timestamps": not settings.get("include_timestamps", False),
            "word_timestamps": False,
            "beam_size": settings.get("beam_size", 5),
            "vad_filter": settings.get("vad_filter", False),
            "condition_on_previous_text": settings.get("condition_on_previous_text", True),
        }
        self.controller.set_whisper_params(whisper_params)

    def _on_file_types_changed(self, ext_checked: dict) -> None:
        self.file_panel.set_ext_checked(ext_checked)

    @Slot(bool, int)
    def _on_server_mode_changed(self, enabled: bool, port: int) -> None:
        self._server_mode_enabled = enabled
        self._server_port = int(port)
        self._save_config("server_mode_enabled", enabled)
        self._save_config("server_port", self._server_port)

        if enabled:
            self._start_server_mode(self._server_port)
        else:
            self.server_manager.stop_server()

    def _start_server_mode(self, port: int) -> None:
        model_name = self.loaded_model_settings.get("model_name", self.DEFAULTS["model"])
        quantization = self.loaded_model_settings.get(
            "quantization_type", self.DEFAULTS["quantization"]
        )
        model_key = f"{model_name} - {quantization}"
        default_settings = ServerTranscriptionSettings(
            model_key=model_key,
            device=self.loaded_model_settings.get("device_type", "cpu"),
            task_mode=self.task_mode,
            language=None,
            output_format=config_manager.get_value("output_format", "txt"),
            include_timestamps=bool(
                config_manager.get_value("include_timestamps", False)
            ),
            word_timestamps=bool(config_manager.get_value("word_timestamps", False)),
            beam_size=int(config_manager.get_value("beam_size", 5)),
            vad_filter=bool(config_manager.get_value("vad_filter", True)),
            condition_on_previous_text=bool(
                config_manager.get_value("condition_on_previous_text", False)
            ),
            batch_size=1,
        )
        self.server_manager.start_server(
            port, self.controller.model_manager, default_settings
        )

    @Slot(int)
    def _on_server_started(self, port: int) -> None:
        logger.info(f"Server started on port {port}")
        self._show_current_model_status()

    @Slot()
    def _on_server_stopped(self) -> None:
        logger.info("Server stopped")
        self._show_current_model_status()

    @Slot(str)
    def _on_server_error(self, message: str) -> None:
        logger.error(f"Server error: {message}")
        self._server_mode_enabled = False
        self._save_config("server_mode_enabled", False)
        self._show_current_model_status()
        QMessageBox.warning(self, "Server Error", message)

    @Slot(str, str, str)
    def _on_model_loaded_success(
        self, model_name: str, quant: str, device: str
    ) -> None:
        self._is_loading_model = False
        self._model_is_loaded = True
        self.loaded_model_settings = {
            "model_name": model_name,
            "quantization_type": quant,
            "device_type": device,
        }
        self._update_translation_availability(model_name)
        self._show_current_model_status()
        self.download_progress_bar.setVisible(False)
        self.cancel_download_button.setVisible(False)
        self.cancel_download_button.setEnabled(True)

    # --- Recording ---

    @Slot()
    def _toggle_recording(self) -> None:
        if self._server_mode_enabled:
            return

        if self.controller.is_transcribing() or self.controller.is_batch_processing():
            return

        if not self.record_button.isEnabled():
            return

        if self.is_recording:
            self._sample_timer.stop()
            self.controller.stop_recording()
            self.is_recording = False
            self.record_button.setText("Processing...")
            update_button_property(self.record_button, "recording", False)
        else:
            if self.controller.start_recording():
                self.is_recording = True
                self.record_button.setText("Recording...")
                self.record_button.set_state(WaveformButton.RECORDING)
                self._sample_timer.start()
                update_button_property(self.record_button, "recording", True)

    @Slot()
    def _feed_audio_samples(self) -> None:
        if self.is_recording:
            samples = self.controller.audio_manager.get_latest_samples()
            if samples is not None:
                self.record_button.update_waveform(samples)

    @Slot()
    def _cancel_transcription(self) -> None:
        self.controller.cancel_transcription()
        self.cancel_button.setEnabled(False)

    @Slot()
    def _on_transcription_cancelled(self) -> None:
        self.record_button.set_state(WaveformButton.IDLE)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setVisible(False)

    def _is_supported_audio_file(self, path: str) -> bool:
        try:
            return Path(path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        except Exception:
            return False

    # --- File panel transcription ---

    @Slot(str, int, str, str, str)
    def _on_file_panel_transcribe(self, file_path: str, batch_size: int,
                                   output_mode: str, output_format: str,
                                   output_dir: str) -> None:
        self._pending_output_mode = output_mode
        self._pending_output_format = output_format
        self._pending_output_dir = output_dir
        self._pending_source_file = file_path

        # Force timestamps on for formats that require them
        if output_format in ("srt", "vtt"):
            self.controller.transcription_service.set_timestamps_override(False)
        else:
            self.controller.transcription_service.set_timestamps_override(None)

        logger.info(
            f"Transcribing: {file_path} (batch={batch_size}, "
            f"mode={output_mode}, fmt={output_format})"
        )
        self.record_button.setText("Transcribing...")
        self.controller.transcribe_file(file_path, batch_size=batch_size)

    @Slot(object)
    def _on_result_ready(self, result) -> None:
        output_mode = getattr(self, "_pending_output_mode", "clipboard")
        output_format = getattr(self, "_pending_output_format", "txt")
        output_dir = getattr(self, "_pending_output_dir", "")
        source_file = getattr(self, "_pending_source_file", "")

        if output_mode == "clipboard":
            return

        if not result.segments:
            logger.warning("No segments available for file output")
            return

        source_path = Path(source_file) if source_file else result.source_file
        if not source_path:
            logger.warning("No source file path for file output")
            return

        out_suffix = f".{output_format}"
        if output_mode == "save_to_custom" and output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_file = out_dir / f"{source_path.stem}{out_suffix}"
        else:
            output_file = source_path.with_suffix(out_suffix)

        try:
            write_output(result, output_file, output_format)
        except Exception as e:
            logger.error(f"Failed to write output file: {e}")
            QMessageBox.warning(
                self, "Output Error", f"Failed to write output file:\n{e}"
            )

    # --- Batch processing ---

    @Slot(list, str, str, int, str)
    def _on_batch_start(self, files, fmt, output_dir, batch_size, task_mode) -> None:
        include_timestamps = config_manager.get_value("include_timestamps", False)
        # Force timestamps on for formats that require them
        if fmt in ("srt", "vtt"):
            include_timestamps = True
        whisper_params = {
            "without_timestamps": not include_timestamps,
            "word_timestamps": False,
            "beam_size": config_manager.get_value("beam_size", 5),
            "vad_filter": config_manager.get_value("vad_filter", False),
            "condition_on_previous_text": config_manager.get_value("condition_on_previous_text", True),
        }
        self.record_button.setText("Transcribing...")
        self.record_button.set_state(WaveformButton.TRANSCRIBING)
        self.record_button.setEnabled(False)
        self.controller.start_batch_processing(
            files=files,
            output_format=fmt,
            output_directory=output_dir if output_dir else None,
            batch_size=batch_size,
            task_mode=task_mode,
            whisper_params=whisper_params,
        )

    @Slot()
    def _on_batch_stop(self) -> None:
        self.controller.stop_batch_processing()
        self.record_button.setText("Click to Record")
        self.record_button.set_state(WaveformButton.IDLE)
        self.record_button.setEnabled(True)

    @Slot(str)
    def _on_batch_finished(self, message: str) -> None:
        self.record_button.setText("Click to Record")
        self.record_button.set_state(WaveformButton.IDLE)
        self.record_button.setEnabled(True)

    # --- Drag and drop ---

    def _transcribe_specific_file(self, file_path: str) -> None:
        if not file_path or not self._is_supported_audio_file(file_path):
            QMessageBox.warning(
                self, "Unsupported File", "Please select a supported audio file."
            )
            return

        self._pending_output_mode = "clipboard"
        self._pending_output_format = "txt"
        self._pending_output_dir = ""
        self._pending_source_file = file_path

        logger.info(f"Transcribing dropped file: {file_path}")
        self.controller.transcribe_file(file_path, batch_size=None)

    # --- Clipboard toggle ---

    @Slot()
    def _toggle_clipboard(self) -> None:
        self._clipboard_visible = not self._clipboard_visible
        self._update_clipboard_button_text()
        self._save_config("show_clipboard_window", self._clipboard_visible)

        host_rect = self._host_rect_global()
        if self._clipboard_visible:
            if self.clipboard_window.is_docked():
                self.clipboard_window.show_docked(host_rect, gap=_DOCK_GAP)
            else:
                self.clipboard_window.show()
                self.clipboard_window.raise_()
        else:
            self.clipboard_window.hide_animated(host_rect, gap=_DOCK_GAP)

    def _update_clipboard_button_text(self) -> None:
        self.clipboard_button.setToolTip(
            "Hide Clipboard" if self._clipboard_visible else "Show Clipboard"
        )

    # --- File panel toggle ---

    @Slot()
    def _toggle_file_panel(self) -> None:
        self._file_panel_visible = not self._file_panel_visible
        self._update_file_panel_button_text()

        host_rect = self._host_rect_global()
        if self._file_panel_visible:
            if self.file_panel.is_docked():
                self.file_panel.show_docked(host_rect, gap=_DOCK_GAP)
            else:
                self.file_panel.show()
                self.file_panel.raise_()
        else:
            self.file_panel.hide_animated(host_rect, gap=_DOCK_GAP)

    def _update_file_panel_button_text(self) -> None:
        self.transcribe_file_button.setToolTip(
            "Hide File Panel" if self._file_panel_visible else "Show File Panel"
        )

    def _register_toggleable_widget(self, widget: QWidget) -> None:
        self._toggleable_widgets.append(widget)

    @Slot(str)
    def _on_transcription_ready(self, text: str) -> None:
        self.record_button.set_state(WaveformButton.IDLE)

        output_mode = getattr(self, "_pending_output_mode", "clipboard")

        if output_mode in ("clipboard", "save_and_clipboard"):
            self.clipboard_window.add_transcription(text)

            if app := QApplication.instance():
                clip_text = (
                    self.clipboard_window.get_full_text()
                    if self.clipboard_window.is_append_mode()
                    else text
                )
                app.clipboard().setText(clip_text)

        if self.is_recording:
            self.is_recording = False
            update_button_property(self.record_button, "recording", False)

        self.file_panel.on_single_file_done()

    @Slot(bool)
    def set_widgets_enabled(self, enabled: bool) -> None:
        for widget in self._toggleable_widgets:
            if hasattr(widget, "setEnabled"):
                widget.setEnabled(enabled)

        if enabled:
            self.cancel_button.setVisible(False)
            self.cancel_button.setEnabled(True)

        if not enabled and self.is_recording:
            self._sample_timer.stop()
            self.controller.stop_recording()
            self.is_recording = False
            update_button_property(self.record_button, "recording", False)

        if enabled and self.record_button.get_state() != WaveformButton.IDLE:
            self.record_button.set_state(WaveformButton.IDLE)

    def _extract_first_supported_drop(self, event) -> str | None:
        if not (md := event.mimeData()) or not md.hasUrls():
            return None
        for url in md.urls():
            if url.isLocalFile() and self._is_supported_audio_file(
                p := url.toLocalFile()
            ):
                return p
        return None

    def eventFilter(self, obj, event):
        if obj is not self and not (
            isinstance(obj, QWidget) and self.isAncestorOf(obj)
        ):
            return super().eventFilter(obj, event)

        et = event.type()
        if et in (QEvent.DragEnter, QEvent.DragMove):
            if self._extract_first_supported_drop(event):
                event.acceptProposedAction()
                return True
        elif et == QEvent.Drop:
            if p := self._extract_first_supported_drop(event):
                event.acceptProposedAction()
                self._transcribe_specific_file(p)
                return True
        return super().eventFilter(obj, event)

    def _sync_side_windows(self) -> None:
        host_rect = self._host_rect_global()

        self.clipboard_window.update_host_rect(host_rect)
        if (
            self.clipboard_window.isVisible()
            and self.clipboard_window.is_docked()
        ):
            self.clipboard_window.reposition_to_host(host_rect, gap=_DOCK_GAP)

        self.file_panel.update_host_rect(host_rect)
        if (
            self.file_panel.isVisible()
            and self.file_panel.is_docked()
        ):
            self.file_panel.reposition_to_host(host_rect, gap=_DOCK_GAP)

    def moveEvent(self, event):
        super().moveEvent(event)
        self._sync_side_windows()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_side_windows()

    def closeEvent(self, event):
        import time as _time
        _t0 = _time.perf_counter()
        logger.info("=== SHUTDOWN DEBUG START ===")

        self._sample_timer.stop()
        self.record_button.stop()
        logger.info(f"[SHUTDOWN] timers stopped: {_time.perf_counter() - _t0:.3f}s")

        _t1 = _time.perf_counter()
        self.server_manager.cleanup()
        logger.info(f"[SHUTDOWN] server_manager.cleanup(): {_time.perf_counter() - _t1:.3f}s")

        _t1 = _time.perf_counter()
        self._metrics_collector.stop()
        logger.info(f"[SHUTDOWN] metrics_collector.stop(): {_time.perf_counter() - _t1:.3f}s")

        if hasattr(self, "global_hotkey"):
            _t1 = _time.perf_counter()
            self.global_hotkey.stop()
            logger.info(f"[SHUTDOWN] global_hotkey.stop(): {_time.perf_counter() - _t1:.3f}s")

        _t1 = _time.perf_counter()
        self.controller.audio_manager.audio_ready.disconnect()
        self.controller.transcription_service.transcription_error.disconnect()
        self.controller.error_occurred.disconnect()
        logger.info(f"[SHUTDOWN] signal disconnects: {_time.perf_counter() - _t1:.3f}s")

        _t1 = _time.perf_counter()
        self._save_state()
        logger.info(f"[SHUTDOWN] _save_state(): {_time.perf_counter() - _t1:.3f}s")

        _t1 = _time.perf_counter()
        self._save_config("show_clipboard_window", self._clipboard_visible)
        config_manager.flush_sync()
        logger.info(f"[SHUTDOWN] config flush: {_time.perf_counter() - _t1:.3f}s")

        _t1 = _time.perf_counter()
        self.clipboard_window.close()
        self.file_panel.close()
        logger.info(f"[SHUTDOWN] side windows close: {_time.perf_counter() - _t1:.3f}s")

        _t1 = _time.perf_counter()
        self.controller.stop_all_threads()
        logger.info(f"[SHUTDOWN] stop_all_threads(): {_time.perf_counter() - _t1:.3f}s")

        _t1 = _time.perf_counter()
        super().closeEvent(event)
        logger.info(f"[SHUTDOWN] super().closeEvent(): {_time.perf_counter() - _t1:.3f}s")
        logger.info(f"=== SHUTDOWN DEBUG END === total: {_time.perf_counter() - _t0:.3f}s")
