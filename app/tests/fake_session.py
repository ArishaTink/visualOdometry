"""Заглушки для тестов интерфейса: работают без данных KITTI и без расчётов VO."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vo.api import FrameResult, Tracks, VOParams, build_report


class FakeSession:
    """Тот же интерфейс, что у VOSession: __len__, finished, step, report."""

    def __init__(self, sequence_dir: Path, params: VOParams, n_frames: int = 40):
        self.params = params
        self._n = n_frames
        self._i = 0
        self._rng = np.random.default_rng(0)
        t = np.linspace(0, 3, n_frames)
        self._gt = np.stack([20 * np.sin(t), np.zeros(n_frames), 15 * t * 3], axis=1)
        self._est = self._gt * 0.8 + self._rng.normal(0, 0.2, self._gt.shape)

    def __len__(self) -> int:
        return self._n

    @property
    def finished(self) -> bool:
        return self._i >= self._n

    def step(self) -> FrameResult:
        i = self._i
        self._i += 1
        p1 = self._rng.random((40, 2)) * [380, 110]
        tracks = Tracks(p1, p1 + [2.0, 0.5], self._rng.random(40) > 0.2)
        return FrameResult(
            index=i, total=self._n,
            image=(self._rng.random((120, 400)) * 255).astype(np.float32),
            position=self._est[i], gt_position=self._gt[i],
            n_inliers=int(tracks.inlier_mask.sum()), step=1.0, tracks=tracks)

    def report(self):
        return build_report(self._est, self._gt,
                            with_scale=not self.params.gt_scale,
                            window_m=self.params.window_m)


class MemorySettings:
    """Настройки в памяти, чтобы тесты не писали в реестр."""

    def __init__(self, data_root: Path):
        self._root, self._params, self._geometry = data_root, VOParams(), None

    def data_root(self): return self._root
    def set_data_root(self, p): self._root = Path(p)
    def params(self): return self._params
    def set_params(self, p): self._params = p
    def geometry(self): return self._geometry
    def set_geometry(self, g): self._geometry = g