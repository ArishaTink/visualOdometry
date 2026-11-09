"""Графики по кадрам: число инлайеров и длина шага."""
from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtWidgets

from vo_app import theme


class StatsPlots(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._inliers_plot = pg.PlotWidget()
        theme.style_plot(self._inliers_plot, "Инлайеры RANSAC", "кадр")
        self._inliers_curve = self._inliers_plot.plot(pen=pg.mkPen(theme.INLIER_COLOR, width=2))

        self._step_plot = pg.PlotWidget()
        theme.style_plot(self._step_plot, "Длина шага", "кадр")
        self._step_curve = self._step_plot.plot(pen=pg.mkPen(theme.ALIGNED_COLOR, width=2))

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._inliers_plot)
        row.addWidget(self._step_plot)

        self._inliers: list[int] = []
        self._steps: list[float] = []

    def append(self, n_inliers: int, step: float) -> None:
        self._inliers.append(n_inliers)
        self._steps.append(step)
        self._inliers_curve.setData(self._inliers)
        self._step_curve.setData(self._steps)

    def reset(self) -> None:
        self._inliers.clear()
        self._steps.clear()
        self._inliers_curve.setData([])
        self._step_curve.setData([])