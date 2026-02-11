from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QFormLayout,
    QWidget,
    QLabel,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
)

from core.models.metadata import ModelMetadata
from core.models.download import list_local_models, get_local_model_path, get_models_directory
from core.models.loader import _make_repo_string, _extract_model_name_from_repo
from config.manager import config_manager
from gui.styles import update_button_property
from core.logging_config import get_logger

logger = get_logger(__name__)


class SettingsDialog(QDialog):
    model_update_requested = Signal(str, str, str)
    reset_dimensions_requested = Signal()

    def __init__(
        self,
        parent: QWidget | None,
        cuda_available: bool,
        supported_quantizations: dict[str, list[str]],
        current_settings: dict[str, str],
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(380)

        self.cuda_available = cuda_available
        self.supported_quantizations = supported_quantizations
        self.current_settings = dict(current_settings)

        self._build_ui()
        self._setup_connections()
        self._update_models_folder_label()
        self._populate_from_settings()
        self._check_for_changes()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Models folder selection row (at the top)
        folder_row = QHBoxLayout()
        folder_row.setSpacing(10)
        
        self.models_folder_btn = QPushButton("Models Folder...")
        self.models_folder_btn.setToolTip("Choose folder for storing models")
        self.models_folder_btn.clicked.connect(self._on_models_folder_clicked)
        self.models_folder_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        folder_row.addWidget(self.models_folder_btn)
        
        self.models_folder_label = QLabel()
        self.models_folder_label.setStyleSheet("color: gray; font-size: 10pt;")
        folder_row.addWidget(self.models_folder_label, 1)
        folder_row.addStretch()
        
        layout.addLayout(folder_row)

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        self.model_dropdown = QComboBox()
        self.model_dropdown.setToolTip("Choose a Whisper model (✓ = downloaded, ⬇ = needs download)")
        form.addRow("Model", self.model_dropdown)

        self.device_dropdown = QComboBox()
        devices = ["cpu", "cuda"] if self.cuda_available else ["cpu"]
        self.device_dropdown.addItems(devices)
        form.addRow("Device", self.device_dropdown)

        self.quantization_dropdown = QComboBox()
        form.addRow("Precision", self.quantization_dropdown)
        
        # Populate model dropdown after all widgets are created
        self._populate_model_dropdown()

        layout.addLayout(form)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        self.update_btn = QPushButton("Update Settings")
        self.update_btn.setObjectName("updateButton")
        self.update_btn.setEnabled(False)
        self.update_btn.clicked.connect(self._on_update_clicked)
        button_row.addWidget(self.update_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        button_row.addWidget(close_btn)

        layout.addLayout(button_row)

        reset_row = QHBoxLayout()
        reset_row.setSpacing(10)

        reset_btn = QPushButton("Reset Window Dimensions")
        reset_btn.setToolTip("Reset the main window and clipboard window to their default sizes and positions")
        reset_btn.clicked.connect(self._on_reset_dimensions_clicked)
        reset_row.addWidget(reset_btn)

        reset_row.addStretch(1)

        layout.addLayout(reset_row)

    def _setup_connections(self) -> None:
        self.model_dropdown.currentTextChanged.connect(self._update_quantization_options)
        self.device_dropdown.currentTextChanged.connect(self._update_quantization_options)
        self.model_dropdown.currentTextChanged.connect(self._check_for_changes)
        self.device_dropdown.currentTextChanged.connect(self._check_for_changes)
        self.quantization_dropdown.currentTextChanged.connect(self._check_for_changes)
        self.quantization_dropdown.currentTextChanged.connect(self._refresh_model_dropdown)

    def _populate_from_settings(self) -> None:
        model = self.current_settings.get("model_name", "base")
        
        # Find model by itemData (not display text with indicators)
        for i in range(self.model_dropdown.count()):
            if self.model_dropdown.itemData(i) == model:
                self.model_dropdown.setCurrentIndex(i)
                break
        
        self.device_dropdown.setCurrentText(self.current_settings.get("device_type", "cpu"))
        self._update_quantization_options()
        self.quantization_dropdown.setCurrentText(
            self.current_settings.get("quantization_type", "float16")
        )

    def _update_quantization_options(self) -> None:
        model = self._get_current_model_name()
        device = self.device_dropdown.currentText()
        opts = ModelMetadata.get_quantization_options(model, device, self.supported_quantizations)

        self.quantization_dropdown.blockSignals(True)
        current = self.quantization_dropdown.currentText()
        self.quantization_dropdown.clear()
        self.quantization_dropdown.addItems(opts)
        if current in opts:
            self.quantization_dropdown.setCurrentText(current)
        elif opts:
            self.quantization_dropdown.setCurrentText(opts[0])
        self.quantization_dropdown.blockSignals(False)
        
        self._refresh_model_dropdown()  # Update indicators for current quantization
        self._check_for_changes()

    def _check_for_changes(self) -> None:
        current = {
            "model_name": self._get_current_model_name(),
            "quantization_type": self.quantization_dropdown.currentText(),
            "device_type": self.device_dropdown.currentText(),
        }
        has_changes = current != self.current_settings
        self.update_btn.setEnabled(has_changes)
        self.update_btn.setText("Reload Model" if has_changes else "Update Settings")
        update_button_property(self.update_btn, "changed", has_changes)

    def _on_update_clicked(self) -> None:
        model = self._get_current_model_name()
        quant = self.quantization_dropdown.currentText()
        device = self.device_dropdown.currentText()
        self.model_update_requested.emit(model, quant, device)
        self.accept()

    def _get_current_model_name(self) -> str:
        """Get the real model name from dropdown (without ✓/⬇ indicator)."""
        index = self.model_dropdown.currentIndex()
        model_name = self.model_dropdown.itemData(index)
        if model_name is not None:
            return model_name
        # Fallback: strip indicator from display text
        text = self.model_dropdown.currentText()
        if text.startswith(("✓ ", "⬇ ")):
            return text[2:].strip()
        return text

    def _populate_model_dropdown(self) -> None:
        """Populate model dropdown with availability indicators (✓ = downloaded, ⬇ = needs download).
        
        Indicator shows availability for the CURRENTLY SELECTED quantization.
        If no quantization selected (initial load), checks for any quantization.
        """
        local_models = set(list_local_models())
        self.model_dropdown.clear()

        # Get current quantization if available
        current_quant = None
        if self.quantization_dropdown.count() > 0:
            current_quant = self.quantization_dropdown.currentText()

        for model_name in ModelMetadata.get_all_model_names():
            has_model = False

            if current_quant:
                # Check for specific quantization
                repo_id = _make_repo_string(model_name, current_quant)
                local_name = _extract_model_name_from_repo(repo_id)
                has_model = local_name in local_models
            else:
                # Check for any quantization (initial load)
                for quant in ["float32", "float16", "bfloat16", "int8", "int8_float32", "int8_float16", "int8_bfloat16"]:
                    try:
                        repo_id = _make_repo_string(model_name, quant)
                        local_name = _extract_model_name_from_repo(repo_id)
                        if local_name in local_models:
                            has_model = True
                            break
                    except:
                        pass

            display_name = f"✓ {model_name}" if has_model else f"⬇ {model_name}"
            self.model_dropdown.addItem(display_name, model_name)

    def _refresh_model_dropdown(self) -> None:
        """Refresh model dropdown indicators (e.g. after quantization change)."""
        current_model = self._get_current_model_name()
        
        # Block signals to prevent recursion (setCurrentIndex triggers currentTextChanged)
        self.model_dropdown.blockSignals(True)
        self._populate_model_dropdown()

        for i in range(self.model_dropdown.count()):
            if self.model_dropdown.itemData(i) == current_model:
                self.model_dropdown.setCurrentIndex(i)
                break
        
        self.model_dropdown.blockSignals(False)

    def _estimate_model_size(self, model_name: str, quant: str) -> str:
        """Estimate model download size."""
        sizes: dict[str, str] = {
            "tiny": "40 MB",
            "base": "75 MB",
            "small": "250 MB",
            "medium": "800 MB",
            "large-v3": "1.5 GB",
            "large-v3-turbo": "800 MB",
            "distil-whisper-large-v3": "800 MB",
        }
        return sizes.get(model_name, "~1 GB")

    def _on_models_folder_clicked(self) -> None:
        """Handle Models Folder button click."""
        # Get current directory
        current_dir = get_models_directory()
        
        # Show folder selection dialog
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Models Folder",
            str(current_dir),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not folder:
            return  # User cancelled
        
        folder_path = Path(folder)
        
        # Validate folder (check if it's writable)
        if not self._validate_models_folder(folder_path):
            return
        
        # Save to config
        config_manager.set_models_directory(str(folder_path))
        
        # Update label
        self._update_models_folder_label()
        
        # Refresh model dropdown (models might be different in new folder)
        self._refresh_model_dropdown()
        
        logger.info(f"Models folder changed to: {folder_path}")

    def _validate_models_folder(self, folder_path: Path) -> bool:
        """Validate that the selected folder is suitable for storing models."""
        # Check if folder exists or can be created
        if not folder_path.exists():
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created models folder: {folder_path}")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Invalid Folder",
                    f"Cannot create folder:\n{folder_path}\n\nError: {e}"
                )
                return False
        
        # Check if folder is writable
        if not folder_path.is_dir():
            QMessageBox.critical(
                self,
                "Invalid Folder",
                f"Selected path is not a folder:\n{folder_path}"
            )
            return False
        
        # Try to write a test file
        test_file = folder_path / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Folder Not Writable",
                f"Cannot write to folder:\n{folder_path}\n\nError: {e}"
            )
            return False
        
        return True

    def _update_models_folder_label(self) -> None:
        """Update the label showing current models folder."""
        current_dir = get_models_directory()
        custom_dir = config_manager.get_models_directory()
        
        if custom_dir:
            # Show shortened path if too long
            path_str = str(current_dir)
            if len(path_str) > 40:
                path_str = "..." + path_str[-37:]
            self.models_folder_label.setText(f"📁 {path_str}")
            self.models_folder_label.setToolTip(f"Current models folder:\n{current_dir}")
        else:
            self.models_folder_label.setText("📁 Default (models/)")
            self.models_folder_label.setToolTip(f"Using default models folder:\n{current_dir}")

    def _on_reset_dimensions_clicked(self) -> None:
        self.reset_dimensions_requested.emit()