import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def refine_pose(R, t, pts1_n, pts2_n, f, px_tol=1.0):
    """Уточняет (R, t) минимизацией ошибки Сампсона с робастной функцией потерь.
    pts*_n - нормализованные координаты, f - фокусное расстояние в пикселях."""
    x1 = np.hstack([pts1_n, np.ones((len(pts1_n), 1))])
    x2 = np.hstack([pts2_n, np.ones((len(pts2_n), 1))])

    t0 = t / np.linalg.norm(t)
    # t параметризуем двумя поворотами в касательной плоскости к t0
    # (сферические углы вырождаются именно при движении вперёд)
    a = np.array([1.0, 0, 0]) if abs(t0[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(t0, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(t0, e1)

    def unpack(p):
        Rm = Rotation.from_rotvec(p[:3]).as_matrix()
        tv = t0 + p[3] * e1 + p[4] * e2
        return Rm, tv / np.linalg.norm(tv)

    def resid(p):
        Rm, tv = unpack(p)
        E = _skew(tv) @ Rm
        Ex1 = x1 @ E.T
        Etx2 = x2 @ E
        num = np.sum(x2 * Ex1, axis=1)
        den = np.sqrt(Ex1[:, 0]**2 + Ex1[:, 1]**2 + Etx2[:, 0]**2 + Etx2[:, 1]**2) + 1e-12
        return num / den

    p0 = np.concatenate([Rotation.from_matrix(R).as_rotvec(), [0.0, 0.0]])
    sol = least_squares(resid, p0, loss="cauchy", f_scale=px_tol / f)
    return unpack(sol.x)