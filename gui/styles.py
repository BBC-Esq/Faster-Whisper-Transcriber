APP_STYLESHEET = """
QWidget {
    font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 10pt;
}

QMainWindow {
    background: #0f1115;
}

QGroupBox {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    margin-top: 9px;
    padding: 8px;
    color: rgba(255, 255, 255, 0.92);
    background: rgba(255, 255, 255, 0.03);
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: rgba(255, 255, 255, 0.85);
}

QLabel {
    color: rgba(255, 255, 255, 0.92);
}

QComboBox, QLineEdit {
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 10px;
    padding: 5px 9px;
    background: rgba(255, 255, 255, 0.05);
    color: rgba(255, 255, 255, 0.95);
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    background: #141821;
    color: rgba(255, 255, 255, 0.95);
    selection-background-color: rgba(89, 160, 255, 0.25);
}

QPushButton {
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 10px;
    padding: 6px 10px;
    background: rgba(255, 255, 255, 0.06);
    color: rgba(255, 255, 255, 0.95);
}

QPushButton:hover {
    background: rgba(255, 255, 255, 0.09);
}

QPushButton:pressed {
    background: rgba(255, 255, 255, 0.07);
}

QPushButton:disabled, QComboBox:disabled, QRadioButton:disabled {
    color: rgba(255, 255, 255, 0.35);
    background: rgba(255, 255, 255, 0.03);
}

QPushButton#recordButton {
    background: rgba(89, 160, 255, 0.16);
    border: 1px solid rgba(89, 160, 255, 0.22);
    font-weight: 600;
}

QPushButton#recordButton:hover {
    background: rgba(89, 160, 255, 0.22);
}

QPushButton#recordButton:pressed {
    background: rgba(89, 160, 255, 0.18);
}

QPushButton#recordButton[recording="true"] {
    background: rgba(34, 197, 94, 0.18);
    border: 1px solid rgba(34, 197, 94, 0.28);
    font-weight: 600;
}

QPushButton#recordButton[recording="true"]:hover {
    background: rgba(34, 197, 94, 0.24);
}

QPushButton#recordButton[recording="true"]:pressed {
    background: rgba(34, 197, 94, 0.20);
}

QPushButton#updateButton[changed="true"] {
    background: rgba(245, 158, 11, 0.18);
    border: 1px solid rgba(245, 158, 11, 0.30);
    font-weight: 700;
}

QPushButton#updateButton[changed="true"]:hover {
    background: rgba(245, 158, 11, 0.24);
}

QPushButton#updateButton[changed="true"]:pressed {
    background: rgba(245, 158, 11, 0.20);
}

QPushButton#clipboardButton {
    background: rgba(255, 255, 255, 0.06);
}

QRadioButton {
    spacing: 8px;
    color: rgba(255, 255, 255, 0.92);
}

QTextEdit {
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 10px;
    padding: 10px;
    background: rgba(255, 255, 255, 0.04);
    color: rgba(255, 255, 255, 0.95);
    selection-background-color: rgba(89, 160, 255, 0.28);
}

QStatusBar {
    background: rgba(255, 255, 255, 0.03);
    color: rgba(255, 255, 255, 0.85);
}

QMessageBox {
    background: #0f1115;
}
"""


def update_button_property(button, prop: str, value: bool) -> None:
    button.setProperty(prop, "true" if value else "false")
    button.style().unpolish(button)
    button.style().polish(button)