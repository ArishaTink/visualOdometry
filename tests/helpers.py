import numpy as np
from scipy.spatial.transform import Rotation


def rot(yaw=0.0, pitch=0.0, roll=0.0):
    """Поворот в градусах. Система камеры: x вправо, y вниз, z вперёд."""
    return Rotation.from_euler("yxz", [yaw, pitch, roll], degrees=True).as_matrix()


def skew(t):
    return np.array([[0, -t[2], t[1]],
                     [t[2], 0, -t[0]],
                     [-t[1], t[0], 0]])


def rot_err_deg(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))


def dir_err_deg(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))