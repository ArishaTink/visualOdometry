"""Текущий кадр с треками: зелёные — инлайеры, красные — выбросы RANSAC."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from vo.api import Tracks
from vo_app import theme


class VideoView(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(theme.PANEL)

        self._view = self.addViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self._view.setMouseEnabled(False, False)

        self._image = pg.ImageItem()
        self._flow = pg.PlotCurveItem(pen=pg.mkPen(theme.INLIER_COLOR, width=1), connect="pairs")
        self._outliers = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(theme.OUTLIER_COLOR))
        self._inliers = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(theme.INLIER_COLOR))
        for item in (self._image, self._flow, self._outliers, self._inliers):
            self._view.addItem(item)

        self._first_frame = True

    def show_frame(self, image: np.ndarray, tracks: Tracks | None) -> None:
        self._image.setImage(image, autoLevels=False, levels=(0, 255))
        if self._first_frame:
            self._view.autoRange(padding=0)
            self._first_frame = False
        self._draw_tracks(tracks)

    def clear_view(self) -> None:
        self._first_frame = True
        self._draw_tracks(None)

    def _draw_tracks(self, tracks: Tracks | None) -> None:
        if tracks is None or len(tracks.next_pts) == 0:
            for item in (self._inliers, self._outliers):
                item.setData([], [])
            self._flow.setData([], [])
            return

        good = tracks.inlier_mask
        p1, p2 = tracks.prev_pts[good], tracks.next_pts[good]

        # отрезки «было -> стало»: точки идут парами (p1[0], p2[0], p1[1], p2[1], ...)
        segments = np.empty((2 * len(p1), 2))
        segments[0::2], segments[1::2] = p1, p2
        self._flow.setData(segments[:, 0], segments[:, 1], connect="pairs")

        self._inliers.setData(p2[:, 0], p2[:, 1])
        bad = ~good
        self._outliers.setData(tracks.next_pts[bad, 0], tracks.next_pts[bad, 1])