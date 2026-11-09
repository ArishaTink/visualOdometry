"""Точка входа GUI."""
from __future__ import annotations

import sys

from PySide6 import QtWidgets

from vo_app.main_window import MainWindow
from vo_app.theme import apply_theme


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())