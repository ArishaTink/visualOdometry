import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift

from vo.imgproc import build_pyramid, image_gradients
from vo.features import shi_tomasi_corners
from vo.tracking import lucas_kanade_track, track_with_fb_check


def textured_image(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.normal(size=(h, w)), 4)
    img = (img - img.mean()) / img.std() * 40 + 128
    return img.astype(np.float32)


def grid_points():
    xs, ys = np.meshgrid(np.arange(60, 261, 20), np.arange(60, 181, 20))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)   # (x, y)


def test_pyramid_shapes():
    pyr = build_pyramid(np.zeros((101, 200), np.float32), 3)
    assert [p.shape for p in pyr] == [(101, 200), (51, 100), (26, 50)]


def test_image_gradients_normalisation_and_axes():
    yy, xx = np.mgrid[0:50, 0:60].astype(np.float32)
    img = 2.0 * xx + 3.0 * yy                      # dI/dx = 2, dI/dy = 3
    Ix, Iy = image_gradients(img)
    assert np.allclose(Ix[2:-2, 2:-2], 2.0, atol=1e-4)
    assert np.allclose(Iy[2:-2, 2:-2], 3.0, atol=1e-4)


def test_shi_tomasi_corners_xy_order():
    img = np.zeros((200, 300), np.float32)
    img[60:140, 100:220] = 255.0                   # строки 60..139, столбцы 100..219
    pts = shi_tomasi_corners(img, max_corners=50, min_distance=10)
    corners = np.array([[100, 60], [219, 60], [100, 139], [219, 139]], float)  # (x, y)
    assert len(pts) >= 4
    for c in corners:
        assert np.linalg.norm(pts - c, axis=1).min() < 6, f"нет детекции у угла {c}"
    for p in pts:
        assert np.linalg.norm(corners - p, axis=1).min() < 10, f"лишняя точка {p}"


def test_shi_tomasi_flat_image_gives_nothing():
    pts = shi_tomasi_corners(np.full((100, 100), 128, np.float32))
    assert pts.shape == (0, 2)


@pytest.mark.parametrize("dx,dy", [(0.4, -0.3), (3.0, 2.0), (-6.0, 4.5), (9.0, -7.0)])
def test_lk_recovers_known_shift(dx, dy):
    I0 = textured_image()
    I1 = shift(I0, (dy, dx), order=3, mode="reflect")   # содержимое сдвинуто на (+dx, +dy)
    pts = grid_points()
    out, ok = lucas_kanade_track(I0, I1, pts)
    assert ok.mean() > 0.9, f"треков выжило только {ok.mean():.0%}"
    err = np.linalg.norm(out[ok] - (pts[ok] + [dx, dy]), axis=1)
    assert np.median(err) < 0.1, f"медианная ошибка {np.median(err):.3f}px"
    assert np.percentile(err, 90) < 0.3


def test_lk_rejects_textureless_region():
    flat = np.full((240, 320), 128, np.float32)
    _, ok = lucas_kanade_track(flat, flat, grid_points())
    assert not ok.any()


def test_fb_check_keeps_good_tracks_on_pure_shift():
    I0 = textured_image()
    I1 = shift(I0, (1.0, 2.0), order=3, mode="reflect")
    _, good = track_with_fb_check(I0, I1, grid_points())
    assert good.mean() > 0.9