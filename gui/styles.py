APP_STYLESHEET = """
QPushButton#recordButton {
    background: rgba(89, 160, 255, 0.16);
    border: 1px solid rgba(89, 160, 255, 0.22);
    border-radius: 9px;
    font-weight: 600;
}

QPushButton#recordButton:hover {
    background: rgba(89, 160, 255, 0.22);
}

QPushButton#recordButton:pressed {
    background: rgba(89, 160, 255, 0.18);
}

QPushButton#copyButton:hover, QPushButton#clearButton:hover,
QPushButton#updateButton:hover, QPushButton#closeButton:hover,
QPushButton#fileStartButton:hover, QPushButton#fileStopButton:hover,
QPushButton#fileDockButton:hover, QPushButton#fileSelectButton:hover,
QPushButton#fileTypesButton:hover {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
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

QStatusBar QLabel {
    color: rgba(34, 197, 94, 0.95);
    font-weight: 600;
}

QProgressBar {
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 3px;
    background: rgba(255, 255, 255, 0.05);
    text-align: center;
    color: rgba(255, 255, 255, 0.9);
    font-size: 8pt;
}

QProgressBar::chunk {
    background: rgba(89, 160, 255, 0.45);
    border-radius: 2px;
}

QMenuBar::item:selected {
    background: rgba(255, 255, 255, 0.08);
}

QMenu::item:selected {
    background: rgba(89, 160, 255, 0.25);
}
"""


def update_button_property(button, prop: str, value: bool) -> None:
    button.setProperty(prop, "true" if value else "false")
    button.style().unpolish(button)
    button.style().polish(button)