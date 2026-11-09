"""Цвета, тёмная тема Qt и оформление графиков pyqtgraph."""
from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

BG = "#16181d"
PANEL = "#1e2128"
BORDER = "#2c313b"
TEXT = "#e4e7ec"
MUTED = "#8b93a1"
ACCENT = "#4da3ff"

GT_COLOR = "#e4e7ec"
VO_COLOR = "#4da3ff"
ALIGNED_COLOR = "#ffb454"
INLIER_COLOR = "#5fd38d"
OUTLIER_COLOR = "#ff6b6b"

STYLESHEET = f"""
QWidget {{ color: {TEXT}; font-size: 13px; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QGroupBox {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
    margin-top: 14px; padding: 12px 10px 10px 10px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {MUTED}; }}
QPushButton {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; padding: 7px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; }}
QPushButton#primary {{ background: {ACCENT}; color: #0b1220; border: none; font-weight: 600; }}
QPushButton#primary:disabled {{ background: {BORDER}; color: {MUTED}; }}
QPushButton:checked {{ border-color: {ACCENT}; }}
QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {BG}; border: 1px solid {BORDER}; border-radius: 5px; padding: 4px 6px;
}}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; }}
QTabBar::tab {{ background: transparent; padding: 8px 16px; color: {MUTED}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QProgressBar {{ background: {PANEL}; border: none; border-radius: 4px; height: 8px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QStatusBar {{ color: {MUTED}; }}
"""


def apply_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(BG))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(BG))
    pal.setColor(QtGui.QPalette.Text, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor(PANEL))
    pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(ACCENT))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)
    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")


def style_plot(plot: pg.PlotWidget, title: str, x_label: str = "", y_label: str = "") -> None:
    """Единый вид всех графиков."""
    plot.setBackground(PANEL)
    plot.setTitle(title, color=MUTED, size="10pt")
    plot.showGrid(x=True, y=True, alpha=0.15)
    if x_label:
        plot.setLabel("bottom", x_label)
    if y_label:
        plot.setLabel("left", y_label)