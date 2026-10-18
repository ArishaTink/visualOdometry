import numpy as np
import pytest

from vo.geometry.epipolar import normalize_points, eight_point_algorithm, sampson_distance
from vo.geometry.ransac import estimate_essential_ransac
from vo.geometry.triangulation import triangulate_depths
from vo.geometry.pose import decompose_essential
from helpers import rot, skew, rot_err_deg, dir_err_deg

POSES = [
    pytest.param(rot(), [0, 0, 1.0], id="forward"),
    pytest.param(rot(yaw=3), [0.1, 0, 1.0], id="yaw+3"),
    pytest.param(rot(yaw=-4, pitch=1), [-0.2, 0.05, 0.9], id="yaw-4"),
    pytest.param(rot(), [1.0, 0, 0.3], id="sideways"),
    pytest.param(rot(yaw=10), [0.0, 0, -0.5], id="backward"),
]


def test_normalize_points():
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], [1241, 376], size=(100, 2))
    pn, T = normalize_points(pts)
    assert np.allclose(pn.mean(axis=0), 0, atol=1e-9)
    assert np.isclose(np.linalg.norm(pn, axis=1).mean(), np.sqrt(2))
    manual = (T @ np.column_stack([pts, np.ones(100)]).T).T[:, :2]
    assert np.allclose(manual, pn)


@pytest.mark.parametrize("n", [8, 200])
@pytest.mark.parametrize("R,t", POSES)
def test_eight_point_exact_data(make_scene, R, t, n):
    t = np.asarray(t, float)
    s = make_scene(R, t, n=n)
    E = eight_point_algorithm(s["p1n"], s["p2n"], essential=True)
    E_true = skew(t) @ R

    # 1) эпиполярное ограничение x2^T E x1 = 0
    x1 = np.column_stack([s["p1n"], np.ones(n)])
    x2 = np.column_stack([s["p2n"], np.ones(n)])
    res = np.abs(np.einsum("ni,ij,nj->n", x2, E / np.linalg.norm(E), x1))
    assert res.max() < 1e-8

    # 2) совпадение с истинной E с точностью до масштаба и знака
    cos = abs(np.sum(E / np.linalg.norm(E) * E_true / np.linalg.norm(E_true)))
    assert cos > 1 - 1e-8

    # 3) сингулярные значения (a, a, 0)
    sv = np.linalg.svd(E, compute_uv=False)
    assert np.isclose(sv[0], sv[1], rtol=1e-6) and sv[2] < 1e-9 * sv[0]


def test_sampson_zero_on_exact_and_scales_with_noise(make_scene):
    R, t = rot(yaw=2), np.array([0.2, 0, 1.0])
    s = make_scene(R, t, n=2000)
    E = skew(t) @ R
    assert sampson_distance(E, s["p1n"], s["p2n"]).max() < 1e-18

    rng = np.random.default_rng(0)
    sigma = 1e-3                                   # ~0.7 px
    d = sampson_distance(E, s["p1n"], s["p2n"] + rng.normal(0, sigma, s["p2n"].shape))
    # возвращается квадрат расстояния; для шума только в pts2 ожидается порядка sigma^2 / 2
    assert 0.2 * sigma**2 < d.mean() < 1.2 * sigma**2


@pytest.mark.parametrize("R,t", POSES)
def test_decompose_essential_recovers_pose(make_scene, R, t):
    t = np.asarray(t, float)
    s = make_scene(R, t)
    R_est, t_est = decompose_essential(skew(t) @ R, s["p1n"], s["p2n"])
    assert rot_err_deg(R_est, R) < 1e-3, "не тот поворот"
    assert np.isclose(np.linalg.norm(t_est), 1.0)
    assert dir_err_deg(t_est, t) < 1e-3, "не то направление t (проверьте знак / cheirality)"


@pytest.mark.parametrize("R,t", POSES)
def test_triangulated_depths_match_truth(make_scene, R, t):
    t = np.asarray(t, float)
    s = make_scene(R, t)
    t_unit = t / np.linalg.norm(t)
    z1, z2 = triangulate_depths(R, t_unit, s["p1n"], s["p2n"])
    # при ||t|| = 1 глубины = истинные / ||t||. На этом держится _estimate_step
    assert np.allclose(z1, s["X1"][:, 2] / np.linalg.norm(t), rtol=1e-6)
    assert np.allclose(z2, s["X2"][:, 2] / np.linalg.norm(t), rtol=1e-6)


def test_ransac_rejects_outliers_and_recovers_pose(K, make_scene):
    R_true, t_true = rot(yaw=2), np.array([0.2, 0, 1.0])
    n = 300
    s = make_scene(R_true, t_true, n=n, noise_px=0.3)
    pts1, pts2 = s["pts1"], s["pts2"].copy()

    rng = np.random.default_rng(1)
    out = rng.choice(n, 90, replace=False)         # 30% выбросов
    pts2[out] = rng.uniform([0, 0], [1241, 376], size=(90, 2))
    is_in = np.ones(n, bool)
    is_in[out] = False

    E, mask = estimate_essential_ransac(pts1, pts2, K, px_tol=1.0)
    assert E is not None
    recall = (mask & is_in).sum() / is_in.sum()
    precision = (mask & is_in).sum() / mask.sum()
    assert recall > 0.85, f"recall={recall:.2f}"
    assert precision > 0.95, f"precision={precision:.2f}"

    R, t = decompose_essential(E, s["to_norm"](pts1[mask]), s["to_norm"](pts2[mask]))
    assert rot_err_deg(R, R_true) < 1.0
    assert dir_err_deg(t, t_true) < 5.0


def test_ransac_too_few_points(K):
    pts = np.random.default_rng(0).uniform(0, 300, size=(5, 2))
    E, mask = estimate_essential_ransac(pts, pts + 1, K)
    assert E is None and mask is None


def test_sampson_threshold_in_pixels(K, make_scene):
    """Порог px_tol должен соответствовать пикселям: 0.5px-смещение проходит, 5px нет."""
    R, t = rot(), np.array([0.0, 0, 1.0])
    s = make_scene(R, t, n=200)
    E = skew(t) @ R
    f = K[0, 0]
    thr = (1.0 / f) ** 2
    shifted_small = s["p2n"] + np.array([0.0, 0.5 / f])
    shifted_big = s["p2n"] + np.array([0.0, 5.0 / f])   # при движении вперёд эпиполярные линии ~радиальные
    assert (sampson_distance(E, s["p1n"], shifted_small) < thr).mean() > 0.9
    assert (sampson_distance(E, s["p1n"], shifted_big) < thr).mean() < 0.3