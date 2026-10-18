import numpy as np
from evo.core.geometry import umeyama_alignment as evo_umeyama

def ate_rmse(est, gt) -> float:
    return float(np.sqrt(np.mean(np.sum((est - gt) ** 2, axis=1))))

def umeyama_alignment(src, dst, with_scale=True):
    R, t, s = evo_umeyama(src.T, dst.T, with_scale)   # evo ждёт массивы (3, N)
    return s, R, t

def scale_drift(est: np.ndarray, gt: np.ndarray, window_m: float = 100.0):
    """Отношение длины пути VO к длине пути GT по окнам в window_m метров GT.
    Возвращает (центры окон, отношения, словарь статистик)."""
    se = np.linalg.norm(np.diff(est, axis=0), axis=1)
    sg = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    idx = (np.cumsum(sg) // window_m).astype(int)

    sum_e = np.bincount(idx, weights=se)
    sum_g = np.bincount(idx, weights=sg)
    ok = sum_g > 0.5 * window_m                  # отбрасываем короткий хвост
    centers = (np.nonzero(ok)[0] + 0.5) * window_m
    ratio = sum_e[ok] / sum_g[ok]

    stats = {"windows": len(ratio)}
    if len(ratio) >= 3:
        mean = ratio.mean()
        slope = np.polyfit(centers, ratio, 1)[0]
        stats.update(
            global_ratio=se.sum() / sg.sum(),
            mean=mean,
            cv=ratio.std() / mean,                                   # разброс масштаба
            rel_trend=slope * (centers[-1] - centers[0]) / mean,     # изменение за весь путь
            min=ratio.min(), max=ratio.max(),
        )
    return centers, ratio, stats