from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel, gaussian_filter, uniform_filter
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
from evo.core.geometry import umeyama_alignment as evo_umeyama
import os
import glob

def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)

def build_pyramid(img: np.ndarray, levels: int) -> list:
    pyramid = [img]
    for _ in range(levels - 1):
        smoothed = gaussian_filter(pyramid[-1], sigma=1.0)
        pyramid.append(smoothed[::2, ::2])
    return pyramid

def image_gradients(img: np.ndarray):
    Ix = sobel(img, axis=1) / 8.0
    Iy = sobel(img, axis=0) / 8.0
    return Ix, Iy

def load_kitti_calib(calib_path: str, cam_id: str = "00") -> np.ndarray:
    key = f"P_rect_{cam_id}:"
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith(key):
                values = np.array(line.split()[1:], dtype=np.float64)
                P = values.reshape(3, 4)
                return P[:, :3]
    raise ValueError(f"Не нашёл {key} в {calib_path}")

def shi_tomasi_corners(img, max_corners=200, quality_level=0.01,
                        min_distance=10, block_size=5) -> np.ndarray:
    Ix, Iy = image_gradients(img)

    Sxx = uniform_filter(Ix * Ix, size=block_size)
    Syy = uniform_filter(Iy * Iy, size=block_size)
    Sxy = uniform_filter(Ix * Iy, size=block_size)

    trace = Sxx + Syy
    det = Sxx * Syy - Sxy ** 2

    disc = np.clip((trace / 2) ** 2 - det, 0, None)
    sqrt_disc = np.sqrt(disc)
    l1 = trace / 2 + sqrt_disc
    l2 = trace / 2 - sqrt_disc
    response = np.minimum(l1, l2)
    response[response < 0] = 0.0

    border = block_size
    mask = np.zeros_like(response, dtype=bool)
    mask[border:-border, border:-border] = True
    response[~mask] = 0.0

    thresh = quality_level * response.max()
    ys, xs = np.nonzero(response > thresh)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    scores = response[ys, xs]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    selected = []
    occupied = np.zeros_like(response, dtype=bool)

    for x, y in zip(xs, ys):
        if len(selected) >= max_corners:
            break
        y0, y1 = max(0, y - min_distance), y + min_distance + 1
        x0, x1 = max(0, x - min_distance), x + min_distance + 1
        if occupied[y0:y1, x0:x1].any():
            continue
        selected.append((x, y))
        occupied[y0:y1, x0:x1] = True

    return np.array(selected, dtype=np.float32)

def lucas_kanade_track(prev_img, next_img, points, win_size=15,
                        pyramid_levels=3, max_iter=10, eps=0.01):

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.array([], dtype=bool)

    prev_pyr = build_pyramid(prev_img, pyramid_levels)
    next_pyr = build_pyramid(next_img, pyramid_levels)

    half_win = win_size // 2
    y_grid, x_grid = np.mgrid[-half_win:half_win + 1, -half_win:half_win + 1]

    N = len(points)
    status = np.ones(N, dtype=bool)
    g = np.zeros((N, 2), dtype=np.float32) # накопленный сдвиг по всем уровням

    for L in range(pyramid_levels - 1, -1, -1):
        if L < pyramid_levels - 1:
            g *= 2.0

        scale = 2.0 ** L
        pts_L = points / scale #координаты точек в уменьшенном изобр

        I_L, J_L = prev_pyr[L], next_pyr[L]
        Ix_L, Iy_L = image_gradients(I_L)

        for i in range(N):
            if not status[i]:
                continue

            px, py = pts_L[i]

            patch_x = px + x_grid
            patch_y = py + y_grid

            if (patch_x.min() < 0 or patch_x.max() >= I_L.shape[1] - 1 or
                patch_y.min() < 0 or patch_y.max() >= I_L.shape[0] - 1):
                if L == 0:
                    status[i] = False
                continue

            coords = np.vstack([patch_y.ravel(), patch_x.ravel()])

            I_patch = map_coordinates(I_L, coords, order=1).reshape(win_size, win_size)
            Ix_patch = map_coordinates(Ix_L, coords, order=1).reshape(win_size, win_size)
            Iy_patch = map_coordinates(Iy_L, coords, order=1).reshape(win_size, win_size)

            Sxx = np.sum(Ix_patch ** 2)
            Syy = np.sum(Iy_patch ** 2)
            Sxy = np.sum(Ix_patch * Iy_patch)
            M = np.array([[Sxx, Sxy], [Sxy, Syy]], dtype=np.float32)
            if np.linalg.det(M) < 1e-6:
                status[i] = False
                continue
            M_inv = np.linalg.inv(M)

            v_L = np.zeros(2, dtype=np.float32) # насколько данный пиксель сдвинулся на данном уровне
            valid = True
            J_patch = I_patch

            for _ in range(max_iter):

                curr_x = patch_x + g[i, 0] + v_L[0] # в g и v изначально нули
                curr_y = patch_y + g[i, 1] + v_L[1]

                if (curr_x.min() < 0 or curr_x.max() >= J_L.shape[1] - 1 or
                    curr_y.min() < 0 or curr_y.max() >= J_L.shape[0] - 1):
                    valid = False
                    break

                curr_coords = np.vstack([curr_y.ravel(), curr_x.ravel()])
                J_patch = map_coordinates(J_L, curr_coords, order=1).reshape(win_size, win_size) # интенсивность второго кадра

                It = J_patch - I_patch # производная по времени - изменение интенсивности между двумя кадрами
                b = -np.array([np.sum(Ix_patch * It), np.sum(Iy_patch * It)], dtype=np.float32)

                eta = M_inv @ b # умножение матрицы на вектор
                v_L += eta

                if np.linalg.norm(eta) < eps:
                    break

            if not valid:
                status[i] = False
                continue

            g[i] += v_L

    tracked_points = points + g.astype(np.float32) # итоговый коорд на втором кадре

    h, w = prev_img.shape
    for i in range(N):
        if status[i]:
            tx, ty = tracked_points[i]
            if tx < 0 or tx >= w - 1 or ty < 0 or ty >= h - 1:
                status[i] = False

    return tracked_points, status

def track_with_fb_check(prev_img, next_img, points, fb_threshold=1.0, **kwargs):
    fwd_pts, fwd_status = lucas_kanade_track(prev_img, next_img, points, **kwargs)
    back_pts, back_status = lucas_kanade_track(next_img, prev_img, fwd_pts, **kwargs)
    fb_error = np.linalg.norm(points - back_pts, axis=1)
    good = fwd_status & back_status & (fb_error < fb_threshold)
    return fwd_pts, good

def normalize_points(pts: np.ndarray):
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    scale = np.sqrt(2) / (mean_dist + 1e-12)
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1],
    ])
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T

def eight_point_algorithm(pts1: np.ndarray, pts2: np.ndarray, essential: bool = False) -> np.ndarray:
    pts1_n, T1 = normalize_points(pts1)
    pts2_n, T2 = normalize_points(pts2)

    x1, y1 = pts1_n[:, 0], pts1_n[:,1]
    x2, y2 = pts2_n[:, 0], pts2_n[:, 1]
    ones = np.ones_like(x1)

    A = np.column_stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1, y1, ones,
    ])

    _, _, Vt = np.linalg.svd(A)
    F_hat = Vt[-1].reshape(3,3)

    U, s, Vt2 = np.linalg.svd(F_hat)
    s[-1] = 0.0
    F = U @ np.diag(s) @ Vt2 #Проверить, что будет если оба сингулярных значения приравнять к 1, а наименьшее к 0

    F = T2.T @ F @ T1

    if essential:
        U, _, Vt = np.linalg.svd(F)
        F = U @ np.diag([1.0, 1.0, 0.0]) @ Vt

    return F

def sampson_distance(E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:

    ones = np.ones((len(pts1), 1)) #перевод в однородные координаты
    x1 = np.hstack([pts1, ones])
    x2 = np.hstack([pts2, ones])

    Ex1 = (E @ x1.T).T #эпиполярные линии для точек из первого изображения
    Etx2 = (E.T @ x2.T).T #эпиполярные линии для точек из второго изображения

    numerator = np.sum(x2 * Ex1, axis=1) ** 2
    denominator = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    denominator = np.maximum(denominator, 1e-12)

    return numerator / denominator

def estimate_essential_ransac(pts1, pts2, K, px_tol=1.0, threshold=None,
                               max_iters=1000, min_inliers=12, confidence=0.999):
    K_inv = np.linalg.inv(K)
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
    pts1_n = (K_inv @ pts1_h.T).T[:, :2]
    pts2_n = (K_inv @ pts2_h.T).T[:, :2]

    n = len(pts1_n)
    if n < 8:
        return None, None
    
    if threshold is None:
        f = (K[0, 0] + K[1, 1]) / 2.0
        threshold = (px_tol / f) ** 2

    best_inliers = None
    best_E = None
    rng = np.random.default_rng()

    iters = max_iters
    i = 0

    while i < iters:
        i += 1
        idx = rng.choice(n, size=8, replace=False) #без повторов

        try:
            E_candidate = eight_point_algorithm(pts1_n[idx], pts2_n[idx], essential=True)
        except np.linalg.LinAlgError:
            continue

        d = sampson_distance(E_candidate, pts1_n, pts2_n)
        inliers = d < threshold

        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_E = E_candidate

            w = best_inliers.sum() / n
            if w > 0:
                denom = 1.0 - w ** 8
                if 1e-12 < denom < 1.0 - 1e-12:
                    need = np.log(1.0 - confidence) / np.log(denom)
                    iters = min(max_iters, max(20, int(np.ceil(need))))

    if best_inliers is None or best_inliers.sum() < min_inliers:
        return None, None

    E_final = eight_point_algorithm(pts1_n[best_inliers], pts2_n[best_inliers], essential=True)

    return E_final, best_inliers
    
def triangulate_point(P1: np.ndarray, P2: np.ndarray, x1, x2) -> np.ndarray:
    A = np.array([
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
        x2[0] * P2[2] - P2[0],
        x2[1] * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def triangulate_depths(R, t, pts1_n, pts2_n):
    """Глубины точек в первой (z1) и второй (z2) камере при ||t|| = 1."""
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])
    n = len(pts1_n)
    z1 = np.empty(n)
    z2 = np.empty(n)
    for i in range(n):
        X = triangulate_point(P1, P2, pts1_n[i], pts2_n[i])
        z1[i] = X[2]
        z2[i] = (R @ X + t)[2]
    return z1, z2

def decompose_essential(E: np.ndarray, pts1_norm: np.ndarray,
                         pts2_norm: np.ndarray):
    U, _, Vt = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

    R_a = U @ W @ Vt
    R_b = U @ W.T @ Vt
    t = U[:, 2]

    candidates = [(R_a, t), (R_a, -t), (R_b, t), (R_b, -t)]

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])

    n_test = min(len(pts1_norm), 50)
    if n_test == 0:
        R0, t0 = candidates[0]
        return R0, t0 / (np.linalg.norm(t0) + 1e-12)
    idx = np.linspace(0, len(pts1_norm) - 1, n_test).astype(int)

    best_count = -1
    best_R, best_t = candidates[0]

    for R, t_cand in candidates:
        if np.linalg.det(R) < 0:
            R = -R
        P2 = np.hstack([R, t_cand.reshape(3, 1)])

        count = 0
        for i in idx:
            X = triangulate_point(P1, P2, pts1_norm[i], pts2_norm[i])
            z1 = X[2]                       # глубина в первой камере
            z2 = (R @ X + t_cand)[2]
            if z1 > 0 and z2 > 0:
                count += 1

        if count > best_count:
            best_count = count
            best_R, best_t = R, t_cand

    best_t = best_t / (np.linalg.norm(best_t) + 1e-12)
    return best_R, best_t

class MonocularVO:
    def __init__(self, K, max_corners=300, min_tracked=100,
                 px_tol=1.0, min_inliers_pose=20,
                 min_flow_px=1.0,
                 max_depth=100.0,
                 min_scale_pts=10,
                 min_fwd=0.8):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.max_corners = max_corners
        self.min_tracked = min_tracked
        self.px_tol = px_tol
        self.min_inliers_pose = min_inliers_pose
        self.min_flow_px = min_flow_px
        self.max_depth = max_depth
        self.min_scale_pts = min_scale_pts
        self.min_fwd = min_fwd

        self.R_pos = np.eye(3)
        self.t_pos = np.zeros(3)
        self.trajectory = [self.t_pos.copy()]
        self.rotations = [self.R_pos.copy()]
        self.n_inliers = []
        self.scales = []

        self.step_scale = 1.0

        self.prev_img = None
        self.prev_pts = None
        self.prev_depth = None

    def _to_norm(self, pts):
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        return (self.K_inv @ pts_h.T).T[:, :2]

    def _record(self, n_inl, step):
        self.n_inliers.append(n_inl)
        self.scales.append(step)
        self.trajectory.append(self.t_pos.copy())
        self.rotations.append(self.R_pos.copy())

    def _estimate_step(self, depth_prev, z1):
        """depth_prev — глубины точек в кадре k в единицах VO,
        z1 — те же точки, триангулированные с ||t|| = 1.
        Отношение = длина текущего шага в единицах VO."""
        ok = (np.isfinite(depth_prev) & (depth_prev > 0) & (depth_prev < self.max_depth) &
              np.isfinite(z1) & (z1 > 0))
        if ok.sum() < self.min_scale_pts:
            return self.step_scale                      # нет данных -> постоянная скорость
        step = np.median(depth_prev[ok] / z1[ok])
        return float(np.clip(step, 0.8 * self.step_scale, 1.25 * self.step_scale))

    def _replenish(self, img, pts, depth, min_dist=10):
        new = shi_tomasi_corners(img, self.max_corners)
        if len(pts) and len(new):
            d = np.linalg.norm(new[:, None] - pts[None], axis=2).min(axis=1)
            new = new[d > min_dist]
        new = new[:max(0, self.max_corners - len(pts))]
        return (np.vstack([pts, new]).astype(np.float32),
                np.concatenate([depth, np.full(len(new), np.nan)]))  

    def process_frame(self, img: np.ndarray, scale: float = 1.0) -> np.ndarray:

        if self.prev_img is None:
            self.prev_img = img
            self.prev_pts = shi_tomasi_corners(img, self.max_corners)
            self.prev_depth = np.full(len(self.prev_pts), np.nan)
            return self.t_pos.copy()

        tracked_pts, status = track_with_fb_check(self.prev_img, img, self.prev_pts)      
        pts1 = self.prev_pts[status]
        pts2 = tracked_pts[status]
        depth_prev = self.prev_depth[status]

        if len(pts1) >= 8:
            flow = np.median(np.linalg.norm(pts2 - pts1, axis=1))
            if flow < self.min_flow_px:
                self._record(0, 0.0)
                return self.t_pos.copy()

        E, inlier_mask = estimate_essential_ransac(pts1, pts2, self.K, px_tol=self.px_tol)
        n_inl = 0 if inlier_mask is None else int(inlier_mask.sum())
        if E is not None and n_inl < self.min_inliers_pose:
            E = None

        new_depth = np.full(len(pts2), np.nan)
        step = 0.0

        if E is not None:
            pts1_n = self._to_norm(pts1[inlier_mask])
            pts2_n = self._to_norm(pts2[inlier_mask])
            R, t = decompose_essential(E, pts1_n, pts2_n)

            R_cam = R.T
            t_cam = -R.T @ t
            # Проверка: машина едет в основном вперёд. Иначе направление ненадёжно,
            # пропускаем кадр и ждём большей базы (prev_img не меняем).
            if t_cam[2] < self.min_fwd:
                self._record(n_inl, 0.0)
                return self.t_pos.copy()

            z1, z2 = triangulate_depths(R, t, pts1_n, pts2_n)
            step = self._estimate_step(depth_prev[inlier_mask], z1) if scale is None else scale
            self.step_scale = step

            new_depth[inlier_mask] = np.where(np.isfinite(z2) & (z2 > 0), z2 * step, np.nan)

            self.t_pos = self.t_pos + step * (self.R_pos @ t_cam)
            self.R_pos = self.R_pos @ R_cam

            # выбрасываем выбросы RANSAC (в т.ч. движущиеся объекты), чтобы их не трекать дальше
            pts2, new_depth = pts2[inlier_mask], new_depth[inlier_mask]

        self._record(n_inl, step)

        next_pts, next_depth = pts2, new_depth
        if len(next_pts) < self.min_tracked:
            next_pts, next_depth = self._replenish(img, next_pts, next_depth)

        self.prev_img = img
        self.prev_pts = next_pts
        self.prev_depth = next_depth
        return self.t_pos.copy()

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

def ate_rmse(est, gt) -> float:
    return float(np.sqrt(np.mean(np.sum((est - gt) ** 2, axis=1))))

def umeyama_alignment(src, dst, with_scale=True):
    R, t, s = evo_umeyama(src.T, dst.T, with_scale)   # evo ждёт массивы (3, N)
    return s, R, t

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # папка, где лежит main.py
    VIDEO = "video_1"                                        # переключить на "video_2" для второго видео

    DATA_DIR = os.path.join(BASE_DIR, "data", VIDEO)
    CALIB_DIR = DATA_DIR
    FRAMES = os.path.join(DATA_DIR, "image_00", "data", "*.png")
    OXTS_DIR = os.path.join(DATA_DIR, "oxts", "data")
    USE_GT_SCALE = False                        # True: масштаб берём из GT, False: VO сам оценивает масштаб

    K = load_kitti_calib(os.path.join(CALIB_DIR, "calib_cam_to_cam.txt"), cam_id="00")
    print("K =\n", K)

    frame_paths = sorted(glob.glob(FRAMES))
    print(f"Кадров найдено: {len(frame_paths)}")

    gt_poses = gt_camera_poses(OXTS_DIR, CALIB_DIR, cam_id="00")
    gt_xyz = gt_poses[:, :3, 3]
 
    n = min(len(frame_paths), len(gt_xyz))
    if len(frame_paths) != len(gt_xyz):
        print(f"Внимание: кадров {len(frame_paths)}, GT-поз {len(gt_xyz)} -> берём {n}")

    vo = MonocularVO(K, min_flow_px=2.0, min_fwd=0.85)

    for i in range(n):
        scale = None
        if USE_GT_SCALE and i > 0:
            scale = float(np.linalg.norm(gt_xyz[i] - gt_xyz[i - 1]))
    
        pos = vo.process_frame(load_gray(frame_paths[i]), scale=scale)
        inl = vo.n_inliers[-1] if vo.n_inliers else 0
        print(f"[{i + 1}/{n}] инлайеров={inl:3d}  "
              f"VO=({pos[0]:7.2f}, {pos[2]:7.2f})  GT=({gt_xyz[i, 0]:7.2f}, {gt_xyz[i, 2]:7.2f})")
        if i > 0:
            print(f"      шаг VO={vo.scales[-1]:.3f}  GT={np.linalg.norm(gt_xyz[i]-gt_xyz[i-1]):.3f}")
 
    est = np.array(vo.trajectory)[:n]
    gt = gt_xyz[:n]
    gt_len = np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1))
    print(f"\nДлина GT-траектории: {gt_len:.1f} м\n")

    s, R_al, t_al = umeyama_alignment(est, gt, with_scale=not USE_GT_SCALE)
    est_al = (s * (R_al @ est.T)).T + t_al

    variants = {}
    variants["без выравнивания"] = est
    variants[f"Umeyama Sim(3) (s={s:.3f})"] = est_al

    errors = {}
    for name, e in variants.items():
        err = np.linalg.norm(e - gt, axis=1)
        errors[name] = err
        print(f"{name:32s} ATE={ate_rmse(e, gt):8.3f} м   "
              f"в конце={err[-1]:8.3f} м ({100 * err[-1] / max(gt_len, 1e-9):.2f}% пути)")
        
    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))

        ax1.plot(gt[:, 0], gt[:, 2], "k-", lw=3, label="GT (OXTS)")

        for name, e in variants.items():
            ax1.plot(e[:, 0], e[:, 2], "-", lw=1.5, label=name)

        ax1.scatter(
            [gt[0, 0]], [gt[0, 2]],
            c="red", s=60, label="Старт", zorder=5
        )

        ax1.set_xlabel("X, м")
        ax1.set_ylabel("Z, м")
        ax1.set_title("Траектория, вид сверху")
        ax1.axis("equal")
        ax1.grid(True)
        ax1.legend(fontsize=8)

        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib не установлен -> траектории в trajectory_*.txt")


        