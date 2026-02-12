"""
Reusable Qt widgets for the app UI.

Placed in: myapp/gui/widgets.py
Currently just a placeholder so that the `gui` package is complete; add custom widgets here later.
"""

# Example reusable widget template
if False:
    from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout


    class ExampleWidget(QWidget):
        def __init__(self, text: str, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel(text))
