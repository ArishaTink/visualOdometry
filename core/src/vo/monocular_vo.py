import numpy as np
from .features import shi_tomasi_corners
from .tracking import track_with_fb_check
from .geometry import (
    estimate_essential_ransac,
    decompose_essential,
    triangulate_depths,
    refine_pose,
)

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

        self.last_tracks = None

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

    def process_frame(self, img: np.ndarray, scale: float | None = None) -> np.ndarray:

        if self.prev_img is None:
            self.prev_img = img
            self.prev_pts = shi_tomasi_corners(img, self.max_corners)
            self.prev_depth = np.full(len(self.prev_pts), np.nan)
            return self.t_pos.copy()

        self.last_tracks = None

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
            f = (self.K[0, 0] + self.K[1, 1]) / 2
            R, t = refine_pose(R, t, pts1_n, pts2_n, f, px_tol=self.px_tol)   # НОВОЕ

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