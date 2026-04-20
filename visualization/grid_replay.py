from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_TRAINED_TRAJ_NAME = "trained_trajectory.pkl"
DEFAULT_RANDOM_TRAJ_NAME = "random_trajectory.pkl"
DEFAULT_SAVE_NAME = "grid_compare_episode0.png"


def load_trajectory(path: str | Path):
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def filter_episode(records, episode_idx: int):
    return [r for r in records if r["episode"] == episode_idx]


def infer_grid_size_from_records(records) -> int:
    max_zone = 0
    for r in records:
        max_zone = max(max_zone, int(r["state"][0]), int(r["next_state"][0]))
    grid_size = int((max_zone + 1) ** 0.5)
    if grid_size * grid_size != max_zone + 1:
        raise ValueError(
            f"Cannot infer square grid size from max zone {max_zone}. "
            f"Please pass grid_size manually."
        )
    return grid_size


def zone_to_xy(zone: int, grid_size: int):
    x = zone // grid_size
    y = zone % grid_size
    return x, y


def _build_path_xy(episode_records, grid_size: int):
    if not episode_records:
        raise ValueError("episode_records is empty")

    path_zones = [int(episode_records[0]["state"][0])]
    for r in episode_records:
        path_zones.append(int(r["next_state"][0]))

    xs = []
    ys = []
    for z in path_zones:
        x, y = zone_to_xy(z, grid_size)
        xs.append(y)   # horizontal axis
        ys.append(x)   # vertical axis
    return xs, ys


def _draw_single_trajectory(ax, episode_records, grid_size: int, title: str):
    # draw grid
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, linewidth=0.8)
        ax.axvline(i - 0.5, linewidth=0.8)

    xs, ys = _build_path_xy(episode_records, grid_size)

    # plot trajectory
    ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label="trajectory")

    # start and end
    ax.scatter(xs[0], ys[0], s=120, marker="s", label="start")
    ax.scatter(xs[-1], ys[-1], s=120, marker="*", label="end")

    # sparse step labels
    for i, (px, py) in enumerate(zip(xs, ys)):
        if i == 0 or i == len(xs) - 1 or i % max(1, len(xs) // 10) == 0:
            ax.text(px + 0.05, py + 0.05, str(i), fontsize=8)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_aspect("equal")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title)
    ax.legend()


def plot_grid_compare(
    trained_episode_records,
    random_episode_records,
    grid_size: int,
    save_path: str | Path | None = None,
    title: str | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    _draw_single_trajectory(
        axes[0],
        trained_episode_records,
        grid_size=grid_size,
        title="Trained Agent",
    )
    _draw_single_trajectory(
        axes[1],
        random_episode_records,
        grid_size=grid_size,
        title="Random Agent",
    )

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"saved figure to: {save_path}")

    # plt.show()
    plt.close()


def run_grid_compare(
    outputs_dir: str | Path,
    episode_idx: int = 0,
    trained_traj_name: str = DEFAULT_TRAINED_TRAJ_NAME,
    random_traj_name: str = DEFAULT_RANDOM_TRAJ_NAME,
    save_name: str = DEFAULT_SAVE_NAME,
):
    outputs_dir = Path(outputs_dir)

    trained_traj_path = outputs_dir / trained_traj_name
    random_traj_path = outputs_dir / random_traj_name
    save_path = outputs_dir / save_name

    print(f"outputs_dir: {outputs_dir}")
    print(f"loading trained trajectory from: {trained_traj_path}")
    print(f"loading random trajectory from: {random_traj_path}")

    trained_records = load_trajectory(trained_traj_path)
    random_records = load_trajectory(random_traj_path)

    trained_episode_records = filter_episode(trained_records, episode_idx)
    random_episode_records = filter_episode(random_records, episode_idx)

    grid_size_trained = infer_grid_size_from_records(trained_records)
    grid_size_random = infer_grid_size_from_records(random_records)

    if grid_size_trained != grid_size_random:
        raise ValueError(
            f"Grid size mismatch: trained={grid_size_trained}, random={grid_size_random}"
        )

    plot_grid_compare(
        trained_episode_records=trained_episode_records,
        random_episode_records=random_episode_records,
        grid_size=grid_size_trained,
        save_path=save_path,
        title=f"{outputs_dir.name} | Episode {episode_idx} | Trained vs Random",
    )


if __name__ == "__main__":
    run_grid_compare(
        outputs_dir="outputs/run_20260419_104953",
        episode_idx=0,
    )