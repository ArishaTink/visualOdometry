"""Данные, которые ядро отдаёт наружу после обработки кадра."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tracks:
    """Треки между предыдущим и текущим кадром (в пикселях)."""
    prev_pts: np.ndarray        # (M, 2)
    next_pts: np.ndarray        # (M, 2)
    inlier_mask: np.ndarray     # (M,) bool


@dataclass
class FrameResult:
    """Результат обработки одного кадра."""
    index: int
    total: int
    image: np.ndarray           # (H, W) float32, значения 0..255
    position: np.ndarray        # (3,) положение камеры по VO
    gt_position: np.ndarray     # (3,) положение камеры по GT
    n_inliers: int
    step: float                 # длина шага
    tracks: Tracks | None       # None, если на кадре оценка не делалась