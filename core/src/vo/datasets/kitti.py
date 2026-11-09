import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import os
import glob
from pathlib import Path

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

def find_sequences(root) -> list[Path]:
    """Папки с последовательностями KITTI внутри root (или сам root)."""
    root = Path(root)
    if (root / "image_00").is_dir():
        return [root]
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "image_00").is_dir())
 
class KittiSequence:
    """Калибровка, кадры и GT одной последовательности."""
 
    def __init__(self, directory):
        self.directory = Path(directory)
        self.name = self.directory.name
        self.K = load_kitti_calib(str(self.directory / "calib_cam_to_cam.txt"), cam_id="00")
 
        frames = sorted(glob.glob(str(self.directory / "image_00" / "data" / "*.png")))
        gt = gt_camera_poses(str(self.directory / "oxts" / "data"),
                             str(self.directory), cam_id="00")[:, :3, 3]
 
        self.n_images, self.n_gt = len(frames), len(gt)
        n = min(self.n_images, self.n_gt)
        self.frame_paths = frames[:n]
        self.gt_xyz = gt[:n]
 
    def __len__(self) -> int:
        return len(self.frame_paths)
 
    def load_frame(self, i: int) -> np.ndarray:
        return load_gray(self.frame_paths[i])
 
    def gt_step(self, i: int):
        """Длина шага GT между кадрами i-1 и i (для i == 0 — None)."""
        if i == 0:
            return None
        return float(np.linalg.norm(self.gt_xyz[i] - self.gt_xyz[i - 1]))
 