import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import os
import glob

def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)

def load_kitti_calib(calib_path: str, cam_id: str = "00") -> np.ndarray:
    key = f"P_rect_{cam_id}:"
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith(key):
                values = np.array(line.split()[1:], dtype=np.float64)
                P = values.reshape(3, 4)
                return P[:, :3]
    raise ValueError(f"Не нашёл {key} в {calib_path}")

def read_calib(path):
    d = {}
    for line in open(path):
        key, val = line.split(":", 1)
        try:
            d[key] = np.array(val.split(), dtype=float)
        except ValueError:
            pass
    return d

def to_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.reshape(R, (3, 3)), t
    return T

def gt_camera_poses(oxts_dir, calib_dir=".", cam_id="00") -> np.ndarray:
    files = sorted(glob.glob(os.path.join(oxts_dir, "*.txt")))
    oxts = np.array([np.loadtxt(f) for f in files])
    lat, lon, alt = oxts[:, 0], oxts[:, 1], oxts[:, 2]

    r = 6378137.0 * np.cos(np.deg2rad(lat[0]))
    T_imu = np.tile(np.eye(4), (len(oxts), 1, 1))
    T_imu[:, :3, :3] = Rotation.from_euler("xyz", oxts[:, 3:6]).as_matrix()
    T_imu[:, 0, 3] = r * np.deg2rad(lon)
    T_imu[:, 1, 3] = r * np.log(np.tan(np.deg2rad(90 + lat) / 2))
    T_imu[:, 2, 3] = alt
    T_imu = np.linalg.inv(T_imu[0]) @ T_imu

    read = lambda name: read_calib(os.path.join(calib_dir, name))
    iv, vc, cc = read("calib_imu_to_velo.txt"), read("calib_velo_to_cam.txt"), read("calib_cam_to_cam.txt")
    T_cam_imu = to_T(cc[f"R_rect_{cam_id}"], np.zeros(3)) @ to_T(vc["R"], vc["T"]) @ to_T(iv["R"], iv["T"])
    return T_cam_imu @ T_imu @ np.linalg.inv(T_cam_imu)