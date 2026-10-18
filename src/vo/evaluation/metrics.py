import numpy as np
from evo.core.geometry import umeyama_alignment as evo_umeyama

def ate_rmse(est, gt) -> float:
    return float(np.sqrt(np.mean(np.sum((est - gt) ** 2, axis=1))))

def umeyama_alignment(src, dst, with_scale=True):
    R, t, s = evo_umeyama(src.T, dst.T, with_scale)   # evo ждёт массивы (3, N)
    return s, R, t