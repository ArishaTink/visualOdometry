from __future__ import annotations

import numpy as np


def plot_trajectories(gt: np.ndarray,
                      variants: dict[str, np.ndarray],
                      title: str = "Траектория, вид сверху",
                      save_path=None,
                      show: bool = True) -> None:
    """gt и значения variants: массивы (N, 3). Рисуем проекцию X-Z."""
    import matplotlib.pyplot as plt   # импорт внутри, чтобы ядро не зависело от matplotlib

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(gt[:, 0], gt[:, 2], "k-", lw=3, label="GT (OXTS)")

    for name, traj in variants.items():
        ax.plot(traj[:, 0], traj[:, 2], "-", lw=1.5, label=name)

    ax.scatter([gt[0, 0]], [gt[0, 2]], c="red", s=60, label="Старт", zorder=5)
    ax.set_xlabel("X, м")
    ax.set_ylabel("Z, м")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)