"""
Проверка всех компонентов main.py.

Часть 1 — автоматические тесты на синтетических данных, где известен правильный
ответ (известный сдвиг, известные R и t и т.д.). Работают без датасета.
Часть 2 — визуальная проверка на реальных кадрах KITTI (запускается, только если
файлы найдены).
"""
import glob
import os
import tempfile
import traceback

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection
from PIL import Image
from scipy.ndimage import gaussian_filter, shift as nd_shift

from main import (
    load_gray, build_pyramid, image_gradients, load_kitti_calib,
    shi_tomasi_corners, lucas_kanade_track, track_with_fb_check,
    normalize_points, eight_point_algorithm, sampson_distance,
    estimate_essential_ransac, triangulate_point, decompose_essential,
)

# пути к реальным данным — поправь под себя
SAMPLE_IMG = "image_00/25a73ae7f3df9a32a3311b965c06e7a8.jpg"
FRAME_0 = "image_00/data/0000000012.png"
FRAME_1 = "image_00/data/0000000013.png"
CALIB_PATH = "calib_cam_to_cam.txt"
FRAME_DIR = "image_00/data"   # все кадры последовательности
START_FRAME = 0               # с какого по счёту кадра начинать

SHOW_PLOTS = True

# =============================================================
# Мини-раннер тестов
# =============================================================

RESULTS = []


def run_test(name, fn):
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
        print(f"[PASS ] {name}")
    except NotImplementedError as e:
        RESULTS.append((name, "SKIP", str(e)))
        print(f"[SKIP ] {name}: {e}")
    except AssertionError as e:
        RESULTS.append((name, "FAIL", str(e)))
        print(f"[FAIL ] {name}: {e}")
    except Exception as e:
        RESULTS.append((name, "ERROR", f"{type(e).__name__}: {e}"))
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)


def print_summary():
    print("\n" + "=" * 60)
    for status in ("PASS", "FAIL", "ERROR", "SKIP"):
        n = sum(1 for r in RESULTS if r[1] == status)
        print(f"{status:6s}: {n}")
    print("=" * 60)


# =============================================================
# Синтетические данные
# =============================================================

def make_texture(h=200, w=260, sigma=2.0, seed=0):
    """Гладкая случайная текстура — хорошо трекается LK."""
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.random((h, w)), sigma)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img.astype(np.float32)


def make_squares():
    """Чёрный фон с двумя белыми квадратами — известны 8 углов."""
    img = np.zeros((120, 200), dtype=np.float32)
    img[30:70, 30:70] = 255.0
    img[40:90, 110:170] = 255.0
    # углы (x, y) — на границе между пикселями, т.е. ±0.5
    true_corners = np.array([
        [29.5, 29.5], [69.5, 29.5], [29.5, 69.5], [69.5, 69.5],
        [109.5, 39.5], [169.5, 39.5], [109.5, 89.5], [169.5, 89.5],
    ])
    return img, true_corners


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def rot_xyz(ax, ay, az):
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# Параметры, похожие на KITTI
K_SYN = np.array([[721.5, 0.0, 609.6],
                  [0.0, 721.5, 172.9],
                  [0.0, 0.0, 1.0]])
IMG_W, IMG_H = 1242, 375
R_SYN = rot_xyz(np.deg2rad(1.0), np.deg2rad(3.0), np.deg2rad(0.5))
T_SYN = np.array([0.2, 0.05, -1.0])  # X2 = R X1 + t
E_SYN = skew(T_SYN) @ R_SYN


def make_two_view(n=200, seed=1):
    """3D-точки перед камерой и их проекции в обе камеры.
    Возвращает X (в системе камеры 1), пиксели в кадрах 1 и 2, нормированные координаты."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.uniform(-10, 10, n * 3),
        rng.uniform(-2, 2, n * 3),
        rng.uniform(5, 40, n * 3),
    ])
    X2 = (R_SYN @ X.T).T + T_SYN

    def project(Xc):
        p = (K_SYN @ Xc.T).T
        return p[:, :2] / p[:, 2:3]

    u1, u2 = project(X), project(X2)
    ok = ((X2[:, 2] > 0.5) &
          (u1[:, 0] >= 0) & (u1[:, 0] < IMG_W) & (u1[:, 1] >= 0) & (u1[:, 1] < IMG_H) &
          (u2[:, 0] >= 0) & (u2[:, 0] < IMG_W) & (u2[:, 1] >= 0) & (u2[:, 1] < IMG_H))
    X, u1, u2 = X[ok][:n], u1[ok][:n], u2[ok][:n]
    n1 = X[:, :2] / X[:, 2:3]
    X2 = (R_SYN @ X.T).T + T_SYN
    n2 = X2[:, :2] / X2[:, 2:3]
    return X, u1, u2, n1, n2


def e_distance(E_est, E_true):
    """Расстояние между E с точностью до масштаба и знака."""
    a = E_est / np.linalg.norm(E_est)
    b = E_true / np.linalg.norm(E_true)
    return min(np.linalg.norm(a - b), np.linalg.norm(a + b))


# =============================================================
# Тесты
# =============================================================

def test_load_gray():
    rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # чисто красный → L = 0.299*255 ≈ 76
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.png")
        Image.fromarray(rgb).save(p)
        img = load_gray(p)
    assert img.shape == (10, 20), f"форма {img.shape}, ожидалась (10, 20)"
    assert img.dtype == np.float32, f"dtype {img.dtype}"
    assert abs(img.mean() - 76) <= 1, f"яркость {img.mean():.1f}, ожидалось ~76"


def test_build_pyramid():
    img = make_texture(200, 260)
    pyr = build_pyramid(img, 4)
    assert len(pyr) == 4, f"уровней {len(pyr)}"
    expected = [(200, 260), (100, 130), (50, 65), (25, 33)]
    for L, (lvl, exp) in enumerate(zip(pyr, expected)):
        assert lvl.shape == exp, f"уровень {L}: {lvl.shape}, ожидалось {exp}"
    assert pyr[0] is img or np.array_equal(pyr[0], img), "уровень 0 должен совпадать с исходником"
    # верхние уровни должны быть сглаженными (меньше дисперсия)
    assert pyr[2].std() < pyr[0].std(), "верхний уровень не сглажен"


def test_image_gradients():
    h, w = 50, 60
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = 3.0 * x + 2.0 * y
    Ix, Iy = image_gradients(img)
    inner = (slice(2, -2), slice(2, -2))
    assert np.allclose(Ix[inner], 3.0, atol=1e-4), f"Ix = {Ix[inner].mean():.4f}, ожидалось 3"
    assert np.allclose(Iy[inner], 2.0, atol=1e-4), f"Iy = {Iy[inner].mean():.4f}, ожидалось 2"


def test_load_kitti_calib():
    P = np.array([[721.5377, 0, 609.5593, 0],
                  [0, 721.5377, 172.854, 0],
                  [0, 0, 1, 0]])
    P1 = P.copy()
    P1[0, 3] = -387.57
    text = (
        "calib_time: 09-Jan-2012 13:57:47\n"
        "S_rect_00: 1.242000e+03 3.750000e+02\n"
        "P_rect_00: " + " ".join(f"{v:.6e}" for v in P.ravel()) + "\n"
        "P_rect_01: " + " ".join(f"{v:.6e}" for v in P1.ravel()) + "\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "calib.txt")
        with open(p, "w") as f:
            f.write(text)
        K = load_kitti_calib(p, "00")
        assert K.shape == (3, 3), f"форма {K.shape}"
        assert np.allclose(K, P[:, :3], atol=1e-3), f"K неверная:\n{K}"
        try:
            load_kitti_calib(p, "07")
            raise AssertionError("для отсутствующей камеры ожидался ValueError")
        except ValueError:
            pass


def test_shi_tomasi_squares():
    img, true_corners = make_squares()
    corners = shi_tomasi_corners(img, max_corners=50, quality_level=0.01,
                                 min_distance=10, block_size=5)
    assert corners.ndim == 2 and corners.shape[1] == 2, f"форма {corners.shape}"
    assert len(corners) >= len(true_corners), \
        f"найдено {len(corners)} углов, ожидалось не меньше {len(true_corners)}"
    # каждый истинный угол найден
    for c in true_corners:
        d = np.linalg.norm(corners - c, axis=1).min()
        assert d < 4, f"угол {c} не найден (ближайший в {d:.1f} px)"
    # лишних точек вдали от углов нет (на прямой границе λ_min = 0)
    for c in corners:
        d = np.linalg.norm(true_corners - c, axis=1).min()
        assert d < 4, f"ложный угол в {c} (до ближайшего истинного {d:.1f} px)"


def test_shi_tomasi_params():
    img = make_texture(200, 260, sigma=1.5)
    c = shi_tomasi_corners(img, max_corners=30, min_distance=10)
    assert len(c) <= 30, f"max_corners нарушен: {len(c)}"
    # min_distance: подавление квадратным окном → по Чебышёву > min_distance
    diff = np.abs(c[:, None, :] - c[None, :, :]).max(axis=2)
    np.fill_diagonal(diff, np.inf)
    assert diff.min() > 10, f"min_distance нарушен: {diff.min()}"
    flat = np.full((50, 50), 100.0, dtype=np.float32)
    assert len(shi_tomasi_corners(flat)) == 0, "на однородном изображении найдены углы"


def test_lucas_kanade_known_shift():
    img0 = make_texture(200, 260)
    dx, dy = 2.3, -1.7  # точки сдвигаются на (+dx, +dy)
    img1 = nd_shift(img0, (dy, dx), order=3, mode="nearest").astype(np.float32)
    pts0 = shi_tomasi_corners(img0, max_corners=60, min_distance=8)
    margin = 30
    keep = ((pts0[:, 0] > margin) & (pts0[:, 0] < 260 - margin) &
            (pts0[:, 1] > margin) & (pts0[:, 1] < 200 - margin))
    pts0 = pts0[keep]
    pts1, status = lucas_kanade_track(img0, img1, pts0)
    assert pts1.shape == pts0.shape, f"форма {pts1.shape}"
    assert status.mean() > 0.9, f"отслежено только {status.mean():.0%} точек"
    err = np.linalg.norm(pts1[status] - (pts0[status] + [dx, dy]), axis=1)
    assert np.median(err) < 0.1, f"медианная ошибка {np.median(err):.3f} px"
    assert np.percentile(err, 90) < 0.3, f"90-й перцентиль ошибки {np.percentile(err, 90):.3f} px"


def test_lucas_kanade_large_shift():
    """Сдвиг больше половины окна — без пирамиды не отследить."""
    img0 = make_texture(200, 260)
    dx, dy = 9.0, 6.0
    img1 = nd_shift(img0, (dy, dx), order=3, mode="nearest").astype(np.float32)
    pts0 = shi_tomasi_corners(img0, max_corners=60, min_distance=8)
    keep = ((pts0[:, 0] > 40) & (pts0[:, 0] < 220) & (pts0[:, 1] > 40) & (pts0[:, 1] < 160))
    pts0 = pts0[keep]
    pts1, status = lucas_kanade_track(img0, img1, pts0, pyramid_levels=3)
    err = np.linalg.norm(pts1[status] - (pts0[status] + [dx, dy]), axis=1)
    assert status.mean() > 0.8, f"отслежено только {status.mean():.0%}"
    assert np.median(err) < 0.2, f"медианная ошибка {np.median(err):.3f} px"


def test_lucas_kanade_empty():
    img = make_texture(50, 50)
    pts, st = lucas_kanade_track(img, img, np.empty((0, 2), dtype=np.float32))
    assert pts.shape == (0, 2) and st.shape == (0,)


def test_fb_check():
    img0 = make_texture(200, 260)
    img1 = nd_shift(img0, (1.0, 2.0), order=3, mode="nearest").astype(np.float32)
    pts0 = shi_tomasi_corners(img0, max_corners=60, min_distance=8)
    keep = ((pts0[:, 0] > 30) & (pts0[:, 0] < 230) & (pts0[:, 1] > 30) & (pts0[:, 1] < 170))
    pts0 = pts0[keep]
    _, good = track_with_fb_check(img0, img1, pts0, fb_threshold=1.0)
    assert good.mean() > 0.9, f"при чистом сдвиге прошло только {good.mean():.0%}"
    # совершенно другое изображение — большинство должно отсеяться
    other = make_texture(200, 260, seed=123)
    _, good_bad = track_with_fb_check(img0, other, pts0, fb_threshold=1.0)
    assert good_bad.mean() < 0.3, f"на несвязанных кадрах прошло {good_bad.mean():.0%}"


def test_normalize_points():
    rng = np.random.default_rng(0)
    pts = rng.uniform([100, 50], [1200, 350], size=(100, 2))
    pn, T = normalize_points(pts)
    assert np.allclose(pn.mean(axis=0), 0, atol=1e-9), f"центр {pn.mean(axis=0)}"
    md = np.linalg.norm(pn, axis=1).mean()
    assert abs(md - np.sqrt(2)) < 1e-9, f"среднее расстояние {md:.6f}, ожидалось √2"
    back = (np.linalg.inv(T) @ np.hstack([pn, np.ones((100, 1))]).T).T[:, :2]
    assert np.allclose(back, pts), "T^-1 не возвращает исходные точки"


def test_eight_point_exact():
    _, _, _, n1, n2 = make_two_view(50)
    E = eight_point_algorithm(n1, n2)
    s = np.linalg.svd(E, compute_uv=False)
    assert s[2] / s[0] < 1e-8, f"E не ранга 2: σ = {s}"
    x1 = np.hstack([n1, np.ones((len(n1), 1))])
    x2 = np.hstack([n2, np.ones((len(n2), 1))])
    res = np.abs(np.sum(x2 * (E @ x1.T).T, axis=1)) / np.linalg.norm(E)
    assert res.max() < 1e-8, f"эпиполярное ограничение нарушено: {res.max():.2e}"
    d = e_distance(E, E_SYN)
    assert d < 1e-6, f"E отличается от истинной: {d:.2e}"
    # у настоящей E два ненулевых сингулярных числа равны
    assert abs(s[0] - s[1]) / s[0] < 1e-6, f"σ1 ≠ σ2: {s}"


def test_eight_point_minimal():
    _, _, _, n1, n2 = make_two_view(8, seed=5)
    E = eight_point_algorithm(n1, n2)
    assert E.shape == (3, 3)
    assert e_distance(E, E_SYN) < 1e-4, f"на 8 точках E неверна: {e_distance(E, E_SYN):.2e}"


def test_eight_point_noise():
    _, u1, u2, _, _ = make_two_view(200, seed=2)
    rng = np.random.default_rng(3)
    u1n = u1 + rng.normal(0, 0.5, u1.shape)
    u2n = u2 + rng.normal(0, 0.5, u2.shape)
    Kinv = np.linalg.inv(K_SYN)
    n1 = (Kinv @ np.hstack([u1n, np.ones((len(u1n), 1))]).T).T[:, :2]
    n2 = (Kinv @ np.hstack([u2n, np.ones((len(u2n), 1))]).T).T[:, :2]
    E = eight_point_algorithm(n1, n2)
    d = e_distance(E, E_SYN)
    assert d < 0.05, f"при шуме 0.5 px E сильно отличается: {d:.3f}"


def test_sampson_distance():
    _, _, _, n1, n2 = make_two_view(50)
    d0 = sampson_distance(E_SYN, n1, n2)
    assert d0.shape == (len(n1),)
    assert d0.max() < 1e-20, f"на точных соответствиях d = {d0.max():.2e}"
    # смещение второй точки вдоль нормали к эпиполярной линии на δ → d ≈ δ²
    x1 = np.hstack([n1, np.ones((len(n1), 1))])
    lines = (E_SYN @ x1.T).T
    normal = lines[:, :2] / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    delta = 1e-3
    d1 = sampson_distance(E_SYN, n1, n2 + delta * normal)
    # знаменатель учитывает обе картинки, поэтому d ≤ δ², но того же порядка
    ratio = d1 / delta ** 2
    assert np.all(ratio > 0.3) and np.all(ratio <= 1.0 + 1e-6), \
        f"d/δ² вне [0.3, 1]: min={ratio.min():.3f}, max={ratio.max():.3f}"
    # квадратичность
    d2 = sampson_distance(E_SYN, n1, n2 + 2 * delta * normal)
    assert np.allclose(d2 / d1, 4.0, rtol=1e-2), "d не растёт квадратично по смещению"
    # инвариантность к масштабу E
    assert np.allclose(sampson_distance(5 * E_SYN, n1, n2 + delta * normal), d1)


def test_ransac_essential():
    _, u1, u2, _, _ = make_two_view(300, seed=4)
    rng = np.random.default_rng(7)
    u1 = u1 + rng.normal(0, 0.3, u1.shape)
    u2 = u2 + rng.normal(0, 0.3, u2.shape)
    n = len(u1)
    is_out = np.zeros(n, dtype=bool)
    is_out[rng.choice(n, int(0.3 * n), replace=False)] = True
    u2[is_out] += rng.uniform(-60, 60, (is_out.sum(), 2))  # 30% выбросов

    E, inl = estimate_essential_ransac(u1, u2, K_SYN, px_tol=1.0)
    assert E is not None, "RANSAC вернул None"
    assert inl.shape == (n,) and inl.dtype == bool, f"маска: {inl.shape}, {inl.dtype}"
    recall = (inl & ~is_out).sum() / (~is_out).sum()
    precision = (inl & ~is_out).sum() / max(inl.sum(), 1)
    assert recall > 0.8, f"найдено только {recall:.0%} истинных инлайеров"
    assert precision > 0.95, f"среди инлайеров {1 - precision:.0%} выбросов"
    assert e_distance(E, E_SYN) < 0.05, f"E отличается от истинной: {e_distance(E, E_SYN):.3f}"


def test_ransac_too_few():
    pts = np.random.default_rng(0).uniform(0, 500, (5, 2))
    E, inl = estimate_essential_ransac(pts, pts, K_SYN)
    assert E is None and inl is None, "для < 8 точек ожидалось (None, None)"


def test_triangulate_point():
    X, _, _, n1, n2 = make_two_view(20, seed=6)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R_SYN, T_SYN.reshape(3, 1)])
    for i in range(len(X)):
        Xh = np.asarray(triangulate_point(P1, P2, n1[i], n2[i]), dtype=np.float64).ravel()
        if Xh.size == 4:
            Xh = Xh[:3] / Xh[3]
        assert Xh.size == 3, f"ожидалась 3D-точка, получено {Xh.size} чисел"
        err = np.linalg.norm(Xh - X[i]) / np.linalg.norm(X[i])
        assert err < 1e-6, f"точка {i}: относительная ошибка {err:.2e}"


def test_decompose_essential():
    _, _, _, n1, n2 = make_two_view(100, seed=8)
    out = decompose_essential(E_SYN, n1, n2)
    R, t = out[0], np.asarray(out[1]).ravel()
    assert R.shape == (3, 3), f"форма R {R.shape}"
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.linalg.det(R) > 0, "R не вращение"
    assert np.allclose(R, R_SYN, atol=1e-6), f"R неверна:\n{R}"
    t_dir = t / np.linalg.norm(t)
    t_true = T_SYN / np.linalg.norm(T_SYN)
    assert np.dot(t_dir, t_true) > 1 - 1e-6, f"направление t неверно: {t_dir} vs {t_true}"
    # cheirality: E и -E дают одно и то же решение
    out2 = decompose_essential(-E_SYN, n1, n2)
    assert np.allclose(out2[0], R_SYN, atol=1e-6), "для -E выбрано другое R"


def run_all_tests():
    tests = [
        ("load_gray", test_load_gray),
        ("build_pyramid", test_build_pyramid),
        ("image_gradients", test_image_gradients),
        ("load_kitti_calib", test_load_kitti_calib),
        ("shi_tomasi: квадраты", test_shi_tomasi_squares),
        ("shi_tomasi: параметры", test_shi_tomasi_params),
        ("lucas_kanade: известный сдвиг", test_lucas_kanade_known_shift),
        ("lucas_kanade: большой сдвиг (пирамида)", test_lucas_kanade_large_shift),
        ("lucas_kanade: пустой вход", test_lucas_kanade_empty),
        ("track_with_fb_check", test_fb_check),
        ("normalize_points", test_normalize_points),
        ("eight_point: точные данные", test_eight_point_exact),
        ("eight_point: 8 точек", test_eight_point_minimal),
        ("eight_point: шум 0.5 px", test_eight_point_noise),
        ("sampson_distance", test_sampson_distance),
        ("estimate_essential_ransac", test_ransac_essential),
        ("estimate_essential_ransac: < 8 точек", test_ransac_too_few),
        ("triangulate_point", test_triangulate_point),
        ("decompose_essential", test_decompose_essential),
    ]
    for name, fn in tests:
        run_test(name, fn)
    print_summary()


# =============================================================
# Визуальная проверка на реальных данных
# =============================================================

def visual_shi_tomasi(path):
    img_gray = load_gray(path)
    corners = shi_tomasi_corners(img_gray, max_corners=200, quality_level=0.01,
                                 min_distance=10, block_size=5)
    print(f"Найдено углов: {len(corners)}")
    print(f"Форма массива: {corners.shape}")

    plt.figure(figsize=(12, 6))
    plt.imshow(img_gray, cmap='gray')
    plt.scatter(corners[:, 0], corners[:, 1], c='red', s=15, marker='x')
    plt.title(f"Shi-Tomasi Corners (всего: {len(corners)})")
    plt.axis('off')
    plt.show()


def visual_tracking_sequence(frame_dir, start=0, max_corners=150, min_tracked=60,
                             fb_threshold=1.0, min_distance=10):
    """Пошаговый просмотр трекинга по всей последовательности.

    Точки трекаются с кадра на кадр (вперёд + FB-проверка). Отбракованные
    теряются; если выживших меньше min_tracked, на текущем кадре добираются
    новые углы Shi-Tomasi вдали от уже существующих точек.

    Управление: «Вперёд» / → / пробел — следующий кадр, «Назад» / ← — предыдущий.
    Уже посчитанные кадры кешируются, поэтому назад листается мгновенно.
    """
    paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))[start:]
    if len(paths) < 2:
        print(f"В {frame_dir} меньше двух кадров — трекинг пропущен")
        return None
    print(f"Кадров в последовательности: {len(paths)}")

    empty = np.empty((0, 2), dtype=np.float32)
    images = {}

    def get_img(k):
        if k not in images:
            images[k] = load_gray(paths[k])
            images.pop(k - 3, None)  # держим в памяти только последние кадры
        return images[k]

    def detect_new(img, existing, n_needed):
        if n_needed <= 0:
            return empty
        cand = shi_tomasi_corners(img, max_corners=max_corners, min_distance=min_distance)
        if len(cand) and len(existing):
            d = np.linalg.norm(cand[:, None, :] - existing[None, :, :], axis=2).min(axis=1)
            cand = cand[d > min_distance]
        return cand[:n_needed].astype(np.float32)

    # steps[k] — всё, что нужно для отрисовки кадра k
    #   pts      — точки на кадре k, которые пойдут в трекинг на кадр k+1
    #   prev/curr — пары соответствий (k-1 → k), прошедшие FB-проверку
    #   rejected — точки кадра k-1, которые отбраковались
    #   new      — добавленные на кадре k углы
    first = shi_tomasi_corners(get_img(0), max_corners=max_corners, min_distance=min_distance)
    steps = [dict(pts=first, prev=empty, curr=empty, rejected=empty, new=first)]

    def compute_until(k):
        while len(steps) <= k:
            j = len(steps)
            prev_pts = steps[j - 1]["pts"]
            nxt, good = track_with_fb_check(get_img(j - 1), get_img(j), prev_pts,
                                            fb_threshold=fb_threshold)
            curr = nxt[good].astype(np.float32)
            new = empty
            if len(curr) < min_tracked:
                new = detect_new(get_img(j), curr, max_corners - len(curr))
            steps.append(dict(pts=np.vstack([curr, new]), prev=prev_pts[good],
                              curr=curr, rejected=prev_pts[~good], new=new))
            print(f"[{j:4d}] {os.path.basename(paths[j])}: отслежено {len(curr)}, "
                  f"отбраковано {(~good).sum()}, добавлено {len(new)}")

    fig, ax = plt.subplots(figsize=(13, 7))
    plt.subplots_adjust(bottom=0.13)
    im = ax.imshow(get_img(0), cmap='gray', vmin=0, vmax=255)
    flow = LineCollection([], colors='yellow', linewidths=1.0)
    ax.add_collection(flow)
    scat_rej = ax.scatter([], [], c='red', s=25, marker='x', label='отбракованы (позиция на k-1)')
    scat_trk = ax.scatter([], [], c='lime', s=18, edgecolors='black', linewidths=0.4,
                          label='отслежены')
    scat_new = ax.scatter([], [], c='cyan', s=18, edgecolors='black', linewidths=0.4,
                          label='новые углы')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.7)
    title = ax.set_title("", fontsize=13)
    ax.axis('off')

    current = [0]

    def show(k):
        compute_until(k)
        s = steps[k]
        im.set_data(get_img(k))
        flow.set_segments(np.stack([s["prev"], s["curr"]], axis=1) if len(s["curr"]) else [])
        scat_trk.set_offsets(s["curr"] if len(s["curr"]) else empty)
        scat_new.set_offsets(s["new"] if len(s["new"]) else empty)
        scat_rej.set_offsets(s["rejected"] if len(s["rejected"]) else empty)
        name = os.path.basename(paths[k])
        if k == 0:
            title.set_text(f"Кадр {k + 1}/{len(paths)} ({name}): найдено углов {len(s['new'])}")
        else:
            title.set_text(f"Кадр {k + 1}/{len(paths)} ({name}): отслежено {len(s['curr'])}, "
                           f"отбраковано {len(s['rejected'])}, добавлено {len(s['new'])}")
        current[0] = k
        fig.canvas.draw_idle()

    def go_next(event=None):
        if current[0] < len(paths) - 1:
            show(current[0] + 1)

    def go_prev(event=None):
        if current[0] > 0:
            show(current[0] - 1)

    def on_key(event):
        if event.key in (' ', 'right'):
            go_next()
        elif event.key == 'left':
            go_prev()

    fig.canvas.mpl_connect('key_press_event', on_key)

    btn_prev = Button(plt.axes([0.30, 0.03, 0.18, 0.06]), '◀ Назад (←)')
    btn_next = Button(plt.axes([0.52, 0.03, 0.18, 0.06]), 'Вперёд (→ / Пробел) ▶')
    btn_prev.on_clicked(go_prev)
    btn_next.on_clicked(go_next)

    show(0)
    plt.show()
    # кнопки возвращаем, чтобы их не удалил сборщик мусора
    return dict(fig=fig, show=show, go_next=go_next, go_prev=go_prev,
                steps=steps, buttons=(btn_prev, btn_next))


def visual_ransac(path0, path1, calib_path):
    img0 = load_gray(path0)
    img1 = load_gray(path1)
    K = load_kitti_calib(calib_path, "00")
    print(f"K =\n{K}")

    pts0 = shi_tomasi_corners(img0, max_corners=500, min_distance=8)
    pts1, good = track_with_fb_check(img0, img1, pts0, fb_threshold=1.0)
    p0, p1 = pts0[good], pts1[good]
    print(f"Соответствий после FB-проверки: {len(p0)}")

    E, inl = estimate_essential_ransac(p0, p1, K, px_tol=1.0)
    if E is None:
        print("RANSAC не нашёл модель")
        return
    print(f"Инлайеров RANSAC: {inl.sum()} из {len(inl)} ({inl.mean():.0%})")
    print(f"Сингулярные числа E: {np.linalg.svd(E, compute_uv=False)}")

    # средняя Sampson-ошибка инлайеров в пикселях
    Kinv = np.linalg.inv(K)
    n0 = (Kinv @ np.hstack([p0, np.ones((len(p0), 1))]).T).T[:, :2]
    n1 = (Kinv @ np.hstack([p1, np.ones((len(p1), 1))]).T).T[:, :2]
    d_px = np.sqrt(sampson_distance(E, n0, n1)) * K[0, 0]
    print(f"Sampson-ошибка инлайеров: медиана {np.median(d_px[inl]):.3f} px")

    try:
        R, t = decompose_essential(E, n0[inl], n1[inl])[:2]
        angles = np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        print(f"Поворот между кадрами: {angles:.2f}°, t = {np.ravel(t)}")
        # для KITTI (машина едет вперёд) t должен быть направлен в основном по -z или +z
    except NotImplementedError as e:
        print(f"decompose_essential пропущен: {e}")

    plt.figure(figsize=(14, 6))
    plt.imshow(img1, cmap='gray')
    for a, b in zip(p0[~inl], p1[~inl]):
        plt.plot([a[0], b[0]], [a[1], b[1]], 'r-', lw=1)
    for a, b in zip(p0[inl], p1[inl]):
        plt.plot([a[0], b[0]], [a[1], b[1]], 'lime', lw=1)
    plt.scatter(p1[inl, 0], p1[inl, 1], c='lime', s=8)
    plt.title(f"Оптический поток: зелёные — инлайеры E ({inl.sum()}), красные — выбросы ({(~inl).sum()})")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    run_all_tests()

    if not SHOW_PLOTS:
        raise SystemExit

    if os.path.exists(SAMPLE_IMG):
        visual_shi_tomasi(SAMPLE_IMG)
    else:
        print(f"Нет {SAMPLE_IMG} — визуализация углов пропущена")

    if os.path.isdir(FRAME_DIR):
        viewer = visual_tracking_sequence(FRAME_DIR, start=START_FRAME)
    else:
        print(f"Нет папки {FRAME_DIR} — визуализация трекинга пропущена")

    if os.path.exists(FRAME_0) and os.path.exists(FRAME_1):
        if os.path.exists(CALIB_PATH):
            visual_ransac(FRAME_0, FRAME_1, CALIB_PATH)
        else:
            print(f"Нет {CALIB_PATH} — проверка RANSAC на реальных данных пропущена")
    else:
        print("Нет кадров KITTI — проверка RANSAC на реальных данных пропущена")