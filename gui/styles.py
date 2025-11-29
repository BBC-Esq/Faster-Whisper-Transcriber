RECORD_BUTTON_NORMAL = ""

RECORD_BUTTON_RECORDING = """
QPushButton {
    background-color: #0d4f26;
    color: white;
}
QPushButton:hover {
    background-color: #0a3d1c;
}
QPushButton:pressed {
    background-color: #082b13;
}
"""

UPDATE_BUTTON_CHANGED = """
QPushButton {
    background-color: #B8860B;
    color: white;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #9A7209;
}
QPushButton:pressed {
    background-color: #7D5C07;
}
"""

CLIPBOARD_BUTTON_STYLE = ""


def apply_recording_button_style(button, is_recording: bool) -> None:
    if is_recording:
        button.setStyleSheet(RECORD_BUTTON_RECORDING)
    else:
        button.setStyleSheet(RECORD_BUTTON_NORMAL)


def apply_update_button_style(button, has_changes: bool) -> None:
    if has_changes:
        button.setStyleSheet(UPDATE_BUTTON_CHANGED)
    else:
        button.setStyleSheet("")


def apply_clipboard_button_style(button) -> None:
    button.setStyleSheet(CLIPBOARD_BUTTON_STYLE)