from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from vo.monocular_vo import MonocularVO
from vo.datasets.kitti import load_gray, load_kitti_calib, gt_camera_poses
from vo.evaluation.metrics import ate_rmse, umeyama_alignment
from vo.evaluation.plotting import plot_trajectories

ROOT = Path(__file__).resolve().parents[1]   # visualOdometry/


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="video_2")
    p.add_argument("--gt-scale", action="store_true",
                   help="брать масштаб шага из GT вместо оценки самим VO")
    p.add_argument("--verbose", action="store_true", help="печатать каждый кадр")
    p.add_argument("--no-show", action="store_true", help="не открывать окно с графиком")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = ROOT / "data" / args.video
    results_dir = ROOT / "results" / args.video
    results_dir.mkdir(parents=True, exist_ok=True)

    K = load_kitti_calib(str(data_dir / "calib_cam_to_cam.txt"), cam_id="00")
    frame_paths = sorted(glob.glob(str(data_dir / "image_00" / "data" / "*.png")))
    gt_xyz = gt_camera_poses(str(data_dir / "oxts" / "data"), str(data_dir),
                             cam_id="00")[:, :3, 3]

    n = min(len(frame_paths), len(gt_xyz))
    if len(frame_paths) != len(gt_xyz):
        print(f"Внимание: кадров {len(frame_paths)}, GT-поз {len(gt_xyz)} -> берём {n}")

    vo = MonocularVO(K, min_flow_px=2.0, min_fwd=0.85)

    for i in range(n):
        scale = None                                   # None -> VO сам оценивает масштаб
        if args.gt_scale and i > 0:
            scale = float(np.linalg.norm(gt_xyz[i] - gt_xyz[i - 1]))

        pos = vo.process_frame(load_gray(frame_paths[i]), scale=scale)

        if args.verbose:
            inl = vo.n_inliers[-1] if vo.n_inliers else 0
            print(f"[{i + 1}/{n}] инлайеров={inl:3d}  "
                  f"VO=({pos[0]:7.2f}, {pos[2]:7.2f})  "
                  f"GT=({gt_xyz[i, 0]:7.2f}, {gt_xyz[i, 2]:7.2f})")

    est = np.array(vo.trajectory)[:n]
    gt = gt_xyz[:n]
    gt_len = np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1))
    print(f"\nДлина GT-траектории: {gt_len:.1f} м\n")

    # выравнивание Umeyama
    s, R_al, t_al = umeyama_alignment(est, gt, with_scale=not args.gt_scale)
    est_al = (s * (R_al @ est.T)).T + t_al

    variants = {
        "без выравнивания": est,
        f"Umeyama Sim(3) (s={s:.3f})": est_al,
    }

    for name, traj in variants.items():
        err = np.linalg.norm(traj - gt, axis=1)
        print(f"{name:32s} ATE={ate_rmse(traj, gt):8.3f} м   "
              f"в конце={err[-1]:8.3f} м ({100 * err[-1] / max(gt_len, 1e-9):.2f}% пути)")

    # сохраняем результаты
    np.savetxt(results_dir / "trajectory_gt.txt", gt)
    np.savetxt(results_dir / "trajectory_vo_raw.txt", est)
    np.savetxt(results_dir / "trajectory_vo.txt", est_al)

    plot_trajectories(gt, variants,
                      save_path=results_dir / "trajectory.png",
                      show=not args.no_show)


if __name__ == "__main__":
    main()