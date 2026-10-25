import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import vo.monocular_vo as vo_mod
from vo.monocular_vo import MonocularVO
from helpers import rot_err_deg

H, W = 376, 1241


class FakeWorld:
    """Камера едет вперёд со скоростями speeds[k] (шаг k -> k+1) и поворачивает по yaw."""

    def __init__(self, K, n_frames, speeds, yaw_rate_deg=0.5, noise_px=0.0, seed=0):
        rng = np.random.default_rng(seed)
        n = 1500
        self.X = np.column_stack([rng.uniform(-12, 12, n),
                                  rng.uniform(-2, 2, n),
                                  rng.uniform(10, 120, n)])
        self.K, self.noise_px, self.rng = K, noise_px, rng
        self.R, self.c = [], []
        R, c = np.eye(3), np.zeros(3)
        dR = Rotation.from_euler("y", yaw_rate_deg, degrees=True).as_matrix()
        for k in range(n_frames):
            self.R.append(R.copy())
            self.c.append(c.copy())
            if k < n_frames - 1:
                c = c + speeds[k] * (R @ np.array([0.0, 0.0, 1.0]))
                R = R @ dR

    def project(self, k):
        Xc = (self.X - self.c[k]) @ self.R[k]            # R_k^T (X - c_k)
        uv = (self.K @ Xc.T).T
        z = Xc[:, 2]
        with np.errstate(all="ignore"):
            px = uv[:, :2] / uv[:, 2:3]
        vis = (z > 1) & (px[:, 0] >= 0) & (px[:, 0] < W - 1) & (px[:, 1] >= 0) & (px[:, 1] < H - 1)
        return np.where(vis[:, None], px, 1e9), vis


def run_vo(monkeypatch, K, world, n_frames, pass_scale_none=True):
    fid = lambda img: int(img[0, 0])

    def fake_detect(img, max_corners=200, **kw):
        px, vis = world.project(fid(img))
        return px[vis][:max_corners].astype(np.float32)

    def fake_track(prev_img, next_img, points, **kw):
        px_p, _ = world.project(fid(prev_img))
        px_n, vis_n = world.project(fid(next_img))
        d = np.linalg.norm(points[:, None, :] - px_p[None], axis=2)
        ids = d.argmin(1)
        ok = (d[np.arange(len(points)), ids] < 2.0) & vis_n[ids]
        out = px_n[ids].copy()
        if world.noise_px:
            out += world.rng.normal(0, world.noise_px, out.shape)
        return out.astype(np.float32), ok

    monkeypatch.setattr(vo_mod, "shi_tomasi_corners", fake_detect)
    monkeypatch.setattr(vo_mod, "track_with_fb_check", fake_track)

    vo = MonocularVO(K, min_flow_px=0.5, min_fwd=0.8)
    for k in range(n_frames):
        img = np.full((H, W), k, np.float32)
        if pass_scale_none:
            vo.process_frame(img, scale=None)
        else:
            vo.process_frame(img)                 # проверяем значение по умолчанию
    return vo


@pytest.mark.parametrize("noise,pos_tol,rot_tol", [(0.0, 0.05, 0.1), (0.3, 2.5, 3.0)],
                         ids=["noise-free", "noise-0.3px"])
def test_pose_accumulation(monkeypatch, K, noise, pos_tol, rot_tol):
    n = 30
    world = FakeWorld(K, n, speeds=np.full(n - 1, 1.0), noise_px=noise)
    vo = run_vo(monkeypatch, K, world, n)

    est, gt = np.array(vo.trajectory), np.array(world.c)
    assert len(est) == n, "trajectory должна содержать ровно по одной позе на кадр"
    assert len(vo.scales) == len(vo.n_inliers) == n - 1

    err = np.linalg.norm(est - gt, axis=1)

    skipped = [(k, n_i) for k, (s, n_i) in enumerate(zip(vo.scales, vo.n_inliers), start=1) if s == 0.0]
    msg = (f"макс. ошибка позиции {err.max():.3f} м; "
           f"кадров с нулевым шагом: {len(skipped)} (кадр, инлайеры): {skipped}")
    assert err.max() < pos_tol, msg
    assert rot_err_deg(vo.R_pos, world.R[-1]) < rot_tol


@pytest.mark.parametrize("pass_scale_none", [True, False], ids=["scale=None", "default-arg"])
def test_scale_follows_changing_speed(monkeypatch, K, pass_scale_none):
    n = 30
    speeds = 1.0 + 0.02 * np.arange(n - 1)          # 1.00 -> 1.56, изменение 2% за кадр (в пределах clip)
    world = FakeWorld(K, n, speeds=speeds)
    vo = run_vo(monkeypatch, K, world, n, pass_scale_none=pass_scale_none)

    assert np.allclose(vo.scales, speeds, rtol=0.03), \
        f"оценённые шаги {np.round(vo.scales[:6], 3)}... vs истинные {np.round(speeds[:6], 3)}..."