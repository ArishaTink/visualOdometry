"""График отношения длины пути VO / GT по окнам (проверка дрейфа масштаба)."""
from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore

from vo.api import EvaluationReport
from vo_app import theme


class ScaleDriftPlot(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        theme.style_plot(self, "Масштаб VO вдоль траектории",
                         "Пройдено по GT, м", "|Δest| / |Δgt|")
        self._curve = self.plot(pen=pg.mkPen(theme.VO_COLOR, width=2),
                                symbol="o", symbolSize=7, symbolBrush=theme.VO_COLOR)
        self._mean_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(theme.MUTED, style=QtCore.Qt.DashLine))
        self.addItem(self._mean_line)
        self._mean_line.hide()

    def show_report(self, report: EvaluationReport) -> None:
        if not report.has_drift_data:
            self.reset()
            return
        self._curve.setData(report.drift_centers, report.drift_ratio)
        self._mean_line.setValue(report.drift_stats["mean"])
        self._mean_line.show()

    def reset(self) -> None:
        self._curve.setData([], [])
        self._mean_line.hide()