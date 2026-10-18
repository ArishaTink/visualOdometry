import numpy as np
from .imgproc import image_gradients
from scipy.ndimage import uniform_filter

def shi_tomasi_corners(img, max_corners=200, quality_level=0.01,
                        min_distance=10, block_size=5) -> np.ndarray:
    Ix, Iy = image_gradients(img)

    Sxx = uniform_filter(Ix * Ix, size=block_size)
    Syy = uniform_filter(Iy * Iy, size=block_size)
    Sxy = uniform_filter(Ix * Iy, size=block_size)

    trace = Sxx + Syy
    det = Sxx * Syy - Sxy ** 2

    disc = np.clip((trace / 2) ** 2 - det, 0, None)
    sqrt_disc = np.sqrt(disc)
    l1 = trace / 2 + sqrt_disc
    l2 = trace / 2 - sqrt_disc
    response = np.minimum(l1, l2)
    response[response < 0] = 0.0

    border = block_size
    mask = np.zeros_like(response, dtype=bool)
    mask[border:-border, border:-border] = True
    response[~mask] = 0.0

    thresh = quality_level * response.max()
    ys, xs = np.nonzero(response > thresh)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    scores = response[ys, xs]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    selected = []
    occupied = np.zeros_like(response, dtype=bool)

    for x, y in zip(xs, ys):
        if len(selected) >= max_corners:
            break
        y0, y1 = max(0, y - min_distance), y + min_distance + 1
        x0, x1 = max(0, x - min_distance), x + min_distance + 1
        if occupied[y0:y1, x0:x1].any():
            continue
        selected.append((x, y))
        occupied[y0:y1, x0:x1] = True

    return np.array(selected, dtype=np.float32)