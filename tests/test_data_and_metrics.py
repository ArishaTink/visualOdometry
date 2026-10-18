import glob
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vo.datasets.kitti import load_kitti_calib, gt_camera_poses
from vo.evaluation.metrics import ate_rmse, umeyama_alignment

DATA = Path(__file__).resolve().parents[1] / "data" / "video_1"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="нет data/video_1")


def test_ate_rmse_known_value():
    est = np.zeros((4, 3))
    gt = np.tile([3.0, 4.0, 0.0], (4, 1))
    assert np.isclose(ate_rmse(est, gt), 5.0)


def test_umeyama_recovers_sim3():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(100, 3)) * 10
    R = Rotation.random(random_state=1).as_matrix()
    s_true, t_true = 2.5, np.array([1.0, -2.0, 3.0])
    dst = s_true * (R @ src.T).T + t_true

    s, R_est, t_est = umeyama_alignment(src, dst, with_scale=True)
    assert np.isclose(s, s_true) and np.allclose(R_est, R) and np.allclose(t_est, t_true)
    assert ate_rmse((s * (R_est @ src.T)).T + t_est, dst) < 1e-8


def test_umeyama_without_scale_returns_unit_scale():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3)) * 10
    R = Rotation.random(random_state=2).as_matrix()
    dst = (R @ src.T).T + np.array([5.0, 0.0, -1.0])
    s, R_est, _ = umeyama_alignment(src, dst, with_scale=False)
    assert s == 1.0 and np.allclose(R_est, R)


@needs_data
def test_kitti_calibration_sane():
    K = load_kitti_calib(str(DATA / "calib_cam_to_cam.txt"), "00")
    assert K.shape == (3, 3) and K[2, 2] == 1.0 and K[0, 1] == 0.0
    assert 600 < K[0, 0] < 800 and 600 < K[1, 1] < 800     # fx, fy для KITTI ~ 707..722
    assert 500 < K[0, 2] < 700 and 100 < K[1, 2] < 250     # cx, cy


@needs_data
def test_gt_poses_are_consistent_with_camera_frame():
    T = gt_camera_poses(str(DATA / "oxts" / "data"), str(DATA), "00")
    n_frames = len(glob.glob(str(DATA / "image_00" / "data" / "*.png")))
    assert len(T) == n_frames, f"поз {len(T)}, кадров {n_frames}"

    assert np.allclose(T[0], np.eye(4), atol=1e-9)
    Rs = T[:, :3, :3]
    assert np.allclose(np.einsum("nij,nkj->nik", Rs, Rs), np.eye(3), atol=1e-6)
    assert np.allclose(np.linalg.det(Rs), 1.0, atol=1e-6)

    rel = np.linalg.inv(T[:-1]) @ T[1:]            # движение кадра k+1 в системе кадра k
    d = rel[:, :3, 3]
    step = np.linalg.norm(d, axis=1)
    assert step.max() < 5.0, "скачок GT больше 5 м за кадр"

    moving = step > 0.5
    assert moving.any()
    assert np.median(d[moving, 2]) > 0, "машина должна ехать в +Z камеры"
    assert np.median(np.abs(d[moving, 2]) / step[moving]) > 0.95, \
        "движение не вдоль оси Z камеры: проверьте цепочку калибровок IMU->velo->cam"