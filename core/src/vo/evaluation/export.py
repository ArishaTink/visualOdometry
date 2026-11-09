"""Сохранение результатов на диск."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vo.evaluation.plotting import save_trajectories_png
from vo.evaluation.report import EvaluationReport


def save_report(report: EvaluationReport, out_dir: str | Path) -> list[Path]:
    """Записывает три файла траекторий и светлый график trajectory.png.
    Возвращает список путей."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "trajectory_gt.txt": report.gt,
        "trajectory_vo_raw.txt": report.est_raw,
        "trajectory_vo.txt": report.est_aligned,
    }
    paths = []
    for name, data in files.items():
        path = out / name
        np.savetxt(path, data)
        paths.append(path)

    png = out / "trajectory.png"
    save_trajectories_png(
        report.gt,
        {"без выравнивания": report.est_raw,
         f"Umeyama Sim(3) (s={report.align_scale:.3f})": report.est_aligned},
        png)
    paths.append(png)
    return paths