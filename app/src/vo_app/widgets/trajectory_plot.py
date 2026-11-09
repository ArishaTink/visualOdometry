"""Траектория сверху (X–Z): GT, VO и, после завершения, выровненный VO."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from vo_app import theme


class TrajectoryPlot(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        theme.style_plot(self, "Траектория, вид сверху", "X, м", "Z, м")
        self.setAspectLocked(True)
        self.addLegend(offset=(10, 10))

        self._gt_curve = self.plot(pen=pg.mkPen(theme.GT_COLOR, width=3), name="GT")
        self._vo_curve = self.plot(pen=pg.mkPen(theme.VO_COLOR, width=2), name="VO")
        self._aligned_curve = self.plot(pen=pg.mkPen(theme.ALIGNED_COLOR, width=2), name="VO + Umeyama")

        self._gt: list[np.ndarray] = []
        self._vo: list[np.ndarray] = []

    def append(self, vo_xyz: np.ndarray, gt_xyz: np.ndarray) -> None:
        self._vo.append(vo_xyz)
        self._gt.append(gt_xyz)
        self._set_curve(self._gt_curve, self._gt)
        self._set_curve(self._vo_curve, self._vo)

    def show_aligned(self, est_aligned: np.ndarray) -> None:
        self._aligned_curve.setData(est_aligned[:, 0], est_aligned[:, 2])

    def reset(self) -> None:
        self._gt.clear()
        self._vo.clear()
        for curve in (self._gt_curve, self._vo_curve, self._aligned_curve):
            curve.setData([], [])

    @staticmethod
    def _set_curve(curve: pg.PlotDataItem, points: list[np.ndarray]) -> None:
        arr = np.asarray(points)
        curve.setData(arr[:, 0], arr[:, 2])