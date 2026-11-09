"""Консольный запуск одометрии без GUI.

    python scripts/run_kitti.py --video video_3 --no-show
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vo.api import VOParams, VOSession, save_report
from vo.evaluation.plotting import plot_scale_drift, plot_trajectories

ROOT = Path(__file__).resolve().parents[1]   # visualOdometry/


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="video_3")
    p.add_argument("--gt-scale", action="store_true",
                   help="брать масштаб шага из GT вместо оценки самим VO")
    p.add_argument("--verbose", action="store_true", help="печатать каждый кадр")
    p.add_argument("--no-show", action="store_true", help="не открывать окно с графиком")
    p.add_argument("--window", type=float, default=100.0,
                   help="размер окна для проверки дрейфа масштаба, м")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = ROOT / "results" / args.video

    params = VOParams(gt_scale=args.gt_scale, window_m=args.window)
    session = VOSession(ROOT / "data" / args.video, params)

    seq = session.sequence
    if seq.n_images != seq.n_gt:
        print(f"Внимание: кадров {seq.n_images}, GT-поз {seq.n_gt} -> берём {len(seq)}")

    while not session.finished:
        r = session.step()
        if args.verbose:
            print(f"[{r.index + 1}/{r.total}] инлайеров={r.n_inliers:3d}  "
                  f"VO=({r.position[0]:7.2f}, {r.position[2]:7.2f})  "
                  f"GT=({r.gt_position[0]:7.2f}, {r.gt_position[2]:7.2f})")

    rep = session.report()
    print(f"\nДлина GT-траектории: {rep.gt_length:.1f} м\n")

    pct = lambda err: 100 * err / max(rep.gt_length, 1e-9)
    print(f"{'без выравнивания':32s} ATE={rep.ate_raw:8.3f} м   "
          f"в конце={rep.end_error_raw:8.3f} м ({pct(rep.end_error_raw):.2f}% пути)")
    print(f"{f'Umeyama Sim(3) (s={rep.align_scale:.3f})':32s} ATE={rep.ate_aligned:8.3f} м   "
          f"в конце={rep.end_error_aligned:8.3f} м ({pct(rep.end_error_aligned):.2f}% пути)")

    print("\nДрейф масштаба (окно %.0f м):" % args.window)
    if not rep.has_drift_data:
        print("  слишком короткая траектория для оценки")
    else:
        st = rep.drift_stats
        print(f"  глобальное отношение длин: {st['global_ratio']:.3f}")
        print(f"  по окнам: mean={st['mean']:.3f}  min={st['min']:.3f}  max={st['max']:.3f}")
        print(f"  разброс (CV)            : {100 * st['cv']:.1f}%")
        print(f"  тренд за весь путь      : {100 * st['rel_trend']:+.1f}%")
        print("  вывод:", "ЕСТЬ дрейф масштаба" if rep.has_scale_drift else "дрейфа масштаба нет")
        results_dir.mkdir(parents=True, exist_ok=True)
        plot_scale_drift(rep.drift_centers, rep.drift_ratio,
                         save_path=results_dir / "scale_drift.png", show=not args.no_show)

    save_report(rep, results_dir)
    plot_trajectories(rep.gt,
                      {"без выравнивания": rep.est_raw,
                       f"Umeyama Sim(3) (s={rep.align_scale:.3f})": rep.est_aligned},
                      show=not args.no_show)     # trajectory.png уже записан в save_report


if __name__ == "__main__":
    main()