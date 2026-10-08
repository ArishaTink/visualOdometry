from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel, gaussian_filter, uniform_filter
from PIL import Image
from scipy.ndimage import map_coordinates

def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)


def build_pyramid(img: np.ndarray, levels: int) -> list:
    pyramid = [img]
    for _ in range(levels - 1):
        smoothed = gaussian_filter(pyramid[-1], sigma=1.0)
        pyramid.append(smoothed[::2, ::2])
    return pyramid


def image_gradients(img: np.ndarray):
    Ix = sobel(img, axis=1) / 8.0
    Iy = sobel(img, axis=0) / 8.0
    return Ix, Iy

def load_kitti_calib(calib_path: str, cam_id: str = "00") -> np.ndarray:
    key = f"P_rect_{cam_id}:"
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith(key):
                values = np.array(line.split()[1:], dtype=np.float64)
                P = values.reshape(3, 4)
                return P[:, :3]
    raise ValueError(f"Не нашёл {key} в {calib_path}")

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

def lucas_kanade_track(prev_img, next_img, points, win_size=15,
                        pyramid_levels=3, max_iter=10, eps=0.01):

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.array([], dtype=bool)

    prev_pyr = build_pyramid(prev_img, pyramid_levels)
    next_pyr = build_pyramid(next_img, pyramid_levels)

    half_win = win_size // 2
    y_grid, x_grid = np.mgrid[-half_win:half_win + 1, -half_win:half_win + 1]

    N = len(points)
    status = np.ones(N, dtype=bool)
    g = np.zeros((N, 2), dtype=np.float32) # накопленный сдвиг по всем уровням

    for L in range(pyramid_levels - 1, -1, -1):
        if L < pyramid_levels - 1:
            g *= 2.0

        scale = 2.0 ** L
        pts_L = points / scale #координаты точек в уменьшенном изобр

        I_L, J_L = prev_pyr[L], next_pyr[L]
        Ix_L, Iy_L = image_gradients(I_L)

        for i in range(N):
            if not status[i]:
                continue

            px, py = pts_L[i]

            patch_x = px + x_grid
            patch_y = py + y_grid

            if (patch_x.min() < 0 or patch_x.max() >= I_L.shape[1] - 1 or
                patch_y.min() < 0 or patch_y.max() >= I_L.shape[0] - 1):
                if L == 0:
                    status[i] = False
                continue

            coords = np.vstack([patch_y.ravel(), patch_x.ravel()])

            I_patch = map_coordinates(I_L, coords, order=1).reshape(win_size, win_size)
            Ix_patch = map_coordinates(Ix_L, coords, order=1).reshape(win_size, win_size)
            Iy_patch = map_coordinates(Iy_L, coords, order=1).reshape(win_size, win_size)

            Sxx = np.sum(Ix_patch ** 2)
            Syy = np.sum(Iy_patch ** 2)
            Sxy = np.sum(Ix_patch * Iy_patch)
            M = np.array([[Sxx, Sxy], [Sxy, Syy]], dtype=np.float32)
            if np.linalg.det(M) < 1e-6:
                status[i] = False
                continue
            M_inv = np.linalg.inv(M)

            v_L = np.zeros(2, dtype=np.float32) # насколько данный пиксель сдвинулся на данном уровне
            valid = True
            J_patch = I_patch

            for _ in range(max_iter):

                curr_x = patch_x + g[i, 0] + v_L[0] # в g и v изначально нули
                curr_y = patch_y + g[i, 1] + v_L[1]

                if (curr_x.min() < 0 or curr_x.max() >= J_L.shape[1] - 1 or
                    curr_y.min() < 0 or curr_y.max() >= J_L.shape[0] - 1):
                    valid = False
                    break

                curr_coords = np.vstack([curr_y.ravel(), curr_x.ravel()])
                J_patch = map_coordinates(J_L, curr_coords, order=1).reshape(win_size, win_size) # интенсивность второго кадра

                It = J_patch - I_patch # производная по времени - изменение интенсивности между двумя кадрами
                b = -np.array([np.sum(Ix_patch * It), np.sum(Iy_patch * It)], dtype=np.float32)

                eta = M_inv @ b # умножение матрицы на вектор
                v_L += eta

                if np.linalg.norm(eta) < eps:
                    break

            if not valid:
                status[i] = False
                continue

            g[i] += v_L

    tracked_points = points + g.astype(np.float32) # итоговый коорд на втором кадре

    h, w = prev_img.shape
    for i in range(N):
        if status[i]:
            tx, ty = tracked_points[i]
            if tx < 0 or tx >= w - 1 or ty < 0 or ty >= h - 1:
                status[i] = False

    return tracked_points, status

def track_with_fb_check(prev_img, next_img, points, fb_threshold=1.0, **kwargs):
    fwd_pts, fwd_status = lucas_kanade_track(prev_img, next_img, points, **kwargs)
    back_pts, back_status = lucas_kanade_track(next_img, prev_img, fwd_pts, **kwargs)
    fb_error = np.linalg.norm(points - back_pts, axis=1)
    good = fwd_status & back_status & (fb_error < fb_threshold)
    return fwd_pts, good

def normalize_points(pts: np.ndarray):
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    scale = np.sqrt(2) / (mean_dist + 1e-12)
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1],
    ])
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T


def eight_point_algorithm(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    pts1_n, T1 = normalize_points(pts1)
    pts2_n, T2 = normalize_points(pts2)

    x1, y1 = pts1_n[:, 0], pts1_n[:,1]
    x2, y2 = pts2_n[:, 0], pts2_n[:, 1]
    ones = np.ones_like(x1)

    A = np.column_stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1, y1, ones,
    ])

    _, _, Vt = np.linalg.svd(A)
    F_hat = Vt[-1].reshape(3,3)

    U, s, Vt2 = np.linalg.svd(F_hat)
    s[-1] = 0.0
    F = U @ np.diag(s) @ Vt2 #Проверить, что будет если оба сингулярных значения приравнять к 1, а наименьшее к 0

    F = T2.T @ F @ T1
    return F


def sampson_distance(E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:

    ones = np.ones((len(pts1), 1)) #перевод в однородные координаты
    x1 = np.hstack([pts1, ones])
    x2 = np.hstack([pts2, ones])

    Ex1 = (E @ x1.T).T #эпиполярные линии для точек из первого изображения
    Etx2 = (E.T @ x2.T).T #эпиполярные линии для точек из второго изображения

    numerator = np.sum(x2 * Ex1, axis=1) ** 2
    denominator = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    denominator = np.maximum(denominator, 1e-12)

    return numerator / denominator



def estimate_essential_ransac(pts1, pts2, K, px_tol=1.0, threshold=None,
                               max_iters=1000, min_inliers=12, confidence=0.999):
    K_inv = np.linalg.inv(K)
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
    pts1_n = (K_inv @ pts1_h.T).T[:, :2]
    pts2_n = (K_inv @ pts2_h.T).T[:, :2]

    n = len(pts1_n)
    if n < 8:
        return None, None
    
    if threshold is None:
        f = (K[0, 0] + K[1, 1]) / 2.0
        threshold = (px_tol / f) ** 2

    best_inliers = None
    best_E = None
    rng = np.random.default_rng()

    iters = max_iters
    i = 0

    while i < iters:
        i += 1
        idx = rng.choice(n, size=8, replace=False) #без повторов

        try:
            E_candidate = eight_point_algorithm(pts1_n[idx], pts2_n[idx])
        except np.linalg.LinAlgError:
            continue

        d = sampson_distance(E_candidate, pts1_n, pts2_n)
        inliers = d < threshold

        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_E = E_candidate

            w = best_inliers.sum() / n
            if w > 0:
                denom = 1.0 - w ** 8
                if 1e-12 < denom < 1.0 - 1e-12:
                    need = np.log(1.0 - confidence) / np.log(denom)
                    iters = min(max_iters, max(20, int(np.ceil(need))))

    if best_inliers is None or best_inliers.sum() < min_inliers:
        return None, None

    E_final = eight_point_algorithm(pts1_n[best_inliers], pts2_n[best_inliers])

    return E_final, best_inliers
    
def triangulate_point(P1: np.ndarray, P2: np.ndarray, x1, x2) -> np.ndarray:
    A = np.array([
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
        x2[0] * P2[2] - P2[0],
        x2[1] * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def decompose_essential(E: np.ndarray, pts1_norm: np.ndarray,
                         pts2_norm: np.ndarray):
    U, _, Vt = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    R_a = U @ W @ Vt
    R_b = U @ W.T @ Vt
    t = U[:, 2]

    candidates = [(R_a, t), (R_a, -t), (R_b, t), (R_b, -t)]

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])

    n_test = min(len(pts1_norm), 50)
    if n_test == 0:
        R0, t0 = candidates[0]
        return R0, t0 / (np.linalg.norm(t0) + 1e-12)
    idx = np.linspace(0, len(pts1_norm) - 1, n_test).astype(int)

    best_count = -1
    best_R, best_t = candidates[0]

    for R, t_cand in candidates:
        if np.linalg.det(R) < 0:
            R = -R
        P2 = np.hstack([R, t_cand.reshape(3, 1)])

        count = 0
        for i in idx:
            X = triangulate_point(P1, P2, pts1_norm[i], pts2_norm[i])
            z1 = X[2]                       # глубина в первой камере
            z2 = (R @ X + t_cand)[2]
            if z1 > 0 and z2 > 0:
                count += 1

        if count > best_count:
            best_count = count
            best_R, best_t = R, t_cand

        best_t = best_t / (np.linalg.norm(best_t) + 1e-12)
        return best_R, best_t
    

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    raise NotImplementedError("Реализовать разложение E и cheirality-тест")
