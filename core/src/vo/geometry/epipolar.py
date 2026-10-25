import numpy as np

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

def eight_point_algorithm(pts1: np.ndarray, pts2: np.ndarray, essential: bool = False) -> np.ndarray:
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

    if essential:
        U, _, Vt = np.linalg.svd(F)
        F = U @ np.diag([1.0, 1.0, 0.0]) @ Vt

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