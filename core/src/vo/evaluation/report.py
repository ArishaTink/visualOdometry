"""Итоговая оценка качества траектории: выравнивание, ATE, дрейф масштаба."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vo.evaluation.metrics import ate_rmse, scale_drift, umeyama_alignment


@dataclass
class EvaluationReport:
    gt: np.ndarray                    # (N, 3) эталонная траектория
    est_raw: np.ndarray               # (N, 3) траектория VO без выравнивания
    est_aligned: np.ndarray           # (N, 3) после Umeyama
    gt_length: float                  # длина GT-пути, м
    align_scale: float                # масштаб s из Umeyama
    ate_raw: float
    ate_aligned: float
    end_error_raw: float              # ошибка в последнем кадре без выравнивания, м
    end_error_aligned: float          # то же после выравнивания, м
    drift_centers: np.ndarray = field(default_factory=lambda: np.empty(0))
    drift_ratio: np.ndarray = field(default_factory=lambda: np.empty(0))
    drift_stats: dict = field(default_factory=dict)

    @property
    def has_drift_data(self) -> bool:
        return "mean" in self.drift_stats

    @property
    def has_scale_drift(self) -> bool | None:
        """True/False — вывод о дрейфе, None — траектория слишком короткая."""
        if not self.has_drift_data:
            return None
        st = self.drift_stats
        return not (st["cv"] < 0.10 and abs(st["rel_trend"]) < 0.15)


def build_report(est: np.ndarray, gt: np.ndarray,
                 with_scale: bool = True, window_m: float = 100.0) -> EvaluationReport:
    """est и gt — массивы (N, 3) одинаковой длины."""
    gt_length = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))

    s, R, t = umeyama_alignment(est, gt, with_scale=with_scale)
    est_aligned = (s * (R @ est.T)).T + t

    centers, ratio, stats = scale_drift(est, gt, window_m=window_m)

    return EvaluationReport(
        gt=gt, est_raw=est, est_aligned=est_aligned,
        gt_length=gt_length, align_scale=float(s),
        ate_raw=ate_rmse(est, gt), ate_aligned=ate_rmse(est_aligned, gt),
        end_error_raw=float(np.linalg.norm(est[-1] - gt[-1])),
        end_error_aligned=float(np.linalg.norm(est_aligned[-1] - gt[-1])),
        drift_centers=centers, drift_ratio=ratio, drift_stats=stats,
    )