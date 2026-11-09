"""Панель чисел: прогресс во время работы и итоговые метрики после неё."""
from __future__ import annotations

from PySide6 import QtWidgets

from vo.api import EvaluationReport, FrameResult

_DASH = "—"


class MetricsPanel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Метрики", parent)
        self._form = QtWidgets.QFormLayout(self)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self._form.addRow(self.progress)

        self._frame = self._row("Кадр")
        self._inliers = self._row("Инлайеры")
        self._gt_len = self._row("Длина GT")
        self._ate_raw = self._row("ATE без выравн.")
        self._ate_al = self._row("ATE Umeyama")
        self._end_err = self._row("Ошибка в конце")
        self._scale = self._row("Масштаб Umeyama")
        self.reset()

    def _row(self, caption: str) -> QtWidgets.QLabel:
        value = QtWidgets.QLabel(_DASH)
        self._form.addRow(caption, value)
        return value

    def update_frame(self, r: FrameResult) -> None:
        self.progress.setMaximum(r.total)
        self.progress.setValue(r.index + 1)
        self._frame.setText(f"{r.index + 1} / {r.total}")
        self._inliers.setText(str(r.n_inliers))

    def show_report(self, rep: EvaluationReport) -> None:
        self._gt_len.setText(f"{rep.gt_length:.1f} м")
        self._ate_raw.setText(f"{rep.ate_raw:.2f} м")
        self._ate_al.setText(f"{rep.ate_aligned:.2f} м")
        pct = 100 * rep.end_error_aligned / max(rep.gt_length, 1e-9)
        self._end_err.setText(f"{rep.end_error_aligned:.2f} м ({pct:.2f}%)")
        self._scale.setText(f"{rep.align_scale:.3f}")

    def reset(self) -> None:
        self.progress.setValue(0)
        for label in (self._frame, self._inliers, self._gt_len, self._ate_raw,
                      self._ate_al, self._end_err, self._scale):
            label.setText(_DASH)