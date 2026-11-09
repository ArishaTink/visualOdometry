"""VOSession: один прогон одометрии по последовательности кадр за кадром."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vo.datasets.kitti import KittiSequence
from vo.evaluation.report import EvaluationReport, build_report
from vo.models import FrameResult, Tracks
from vo.monocular_vo import MonocularVO
from vo.params import VOParams


class VOSession:
    def __init__(self, sequence_dir: str | Path, params: VOParams | None = None):
        self.params = params or VOParams()
        self.sequence = KittiSequence(sequence_dir)
        self._vo = MonocularVO(self.sequence.K, **self.params.vo_kwargs())
        self._next = 0

    def __len__(self) -> int:
        return len(self.sequence)

    @property
    def finished(self) -> bool:
        return self._next >= len(self)

    def step(self) -> FrameResult:
        """Обработать следующий кадр."""
        if self.finished:
            raise RuntimeError("Все кадры уже обработаны")

        i = self._next
        scale = self.sequence.gt_step(i) if self.params.gt_scale else None
        image = self.sequence.load_frame(i)
        position = self._vo.process_frame(image, scale=scale)
        self._next += 1
        return self._make_result(i, image, position)

    def report(self) -> EvaluationReport:
        """Итоговые метрики по уже обработанным кадрам."""
        n = self._next
        if n < 3:
            raise RuntimeError("Для отчёта нужно обработать хотя бы 3 кадра")
        est = np.array(self._vo.trajectory)[:n]
        gt = self.sequence.gt_xyz[:n]
        return build_report(est, gt,
                            with_scale=not self.params.gt_scale,
                            window_m=self.params.window_m)

    def _make_result(self, i: int, image: np.ndarray, position: np.ndarray) -> FrameResult:
        vo = self._vo
        tracks = None
        if vo.last_tracks is not None:
            prev_pts, next_pts, mask = vo.last_tracks
            tracks = Tracks(prev_pts.copy(), next_pts.copy(), mask.copy())
        return FrameResult(
            index=i, total=len(self), image=image,
            position=position, gt_position=self.sequence.gt_xyz[i],
            n_inliers=vo.n_inliers[-1] if vo.n_inliers else 0,
            step=vo.scales[-1] if vo.scales else 0.0,
            tracks=tracks,
        )