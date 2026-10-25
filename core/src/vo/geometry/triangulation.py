import numpy as np

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

def triangulate_depths(R, t, pts1_n, pts2_n):
    """Глубины точек в первой (z1) и второй (z2) камере при ||t|| = 1."""
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])
    n = len(pts1_n)
    z1 = np.empty(n)
    z2 = np.empty(n)
    for i in range(n):
        X = triangulate_point(P1, P2, pts1_n[i], pts2_n[i])
        z1[i] = X[2]
        z2[i] = (R @ X + t)[2]
    return z1, z2