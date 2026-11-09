"""Параметры запуска одометрии. Один объект вместо россыпи аргументов."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VOParams:
    # --- параметры MonocularVO ---
    max_corners: int = 300
    min_tracked: int = 100
    px_tol: float = 1.0
    min_inliers_pose: int = 20
    min_flow_px: float = 1.0
    max_depth: float = 100.0
    min_scale_pts: int = 10
    min_fwd: float = -1.0

    # --- параметры запуска и оценки ---
    gt_scale: bool = False        # брать длину шага из GT вместо оценки самим VO
    window_m: float = 100.0       # окно проверки дрейфа масштаба, м

    def vo_kwargs(self) -> dict:
        """Аргументы для конструктора MonocularVO (без параметров запуска)."""
        kwargs = asdict(self)
        kwargs.pop("gt_scale")
        kwargs.pop("window_m")
        return kwargs