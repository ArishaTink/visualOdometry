import numpy as np
from .triangulation import triangulate_point

def decompose_essential(E: np.ndarray, pts1_norm: np.ndarray,
                         pts2_norm: np.ndarray):
    U, _, Vt = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

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