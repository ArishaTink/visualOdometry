import numpy as np
import pytest


@pytest.fixture
def K():
    return np.array([[718.856, 0.0, 607.19],
                     [0.0, 718.856, 185.22],
                     [0.0, 0.0, 1.0]])


@pytest.fixture
def make_scene(K):
    """Фабрика синтетической сцены: 3D-точки, пиксели в двух камерах, нормализованные координаты.
    Модель движения: X2 = R @ X1 + t."""
    Kinv = np.linalg.inv(K)

    def to_norm(p):
        return (Kinv @ np.column_stack([p, np.ones(len(p))]).T).T[:, :2]

    def _make(R, t, n=200, noise_px=0.0, seed=0):
        rng = np.random.default_rng(seed)
        X1 = np.column_stack([rng.uniform(-10, 10, n),
                              rng.uniform(-3, 3, n),
                              rng.uniform(5, 50, n)])
        X2 = X1 @ R.T + t

        def proj(X):
            uv = (K @ X.T).T
            return uv[:, :2] / uv[:, 2:3]

        pts1, pts2 = proj(X1), proj(X2)
        if noise_px:
            pts1 = pts1 + rng.normal(0, noise_px, pts1.shape)
            pts2 = pts2 + rng.normal(0, noise_px, pts2.shape)
        return dict(X1=X1, X2=X2, pts1=pts1, pts2=pts2,
                    p1n=to_norm(pts1), p2n=to_norm(pts2), to_norm=to_norm)
    return _make