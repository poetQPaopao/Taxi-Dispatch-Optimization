from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


DEFAULT_TRAINED_TRAJ_NAME = "trained_trajectory.pkl"
DEFAULT_RANDOM_TRAJ_NAME = "random_trajectory.pkl"


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
    row = zone // grid_size
    col = zone % grid_size
    return row, col


def build_path_xy(episode_records, grid_size: int):
    if not episode_records:
        raise ValueError("episode_records is empty")

    zones = [int(episode_records[0]["state"][0])]
    for r in episode_records:
        zones.append(int(r["next_state"][0]))

    xs = []
    ys = []
    for z in zones:
        row, col = zone_to_xy(z, grid_size)
        xs.append(col)   # x-axis uses column
        ys.append(row)   # y-axis uses row
    return xs, ys


def draw_grid(ax, grid_size: int):
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, linewidth=0.8)
        ax.axvline(i - 0.5, linewidth=0.8)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_aspect("equal")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")


def get_episode_total_reward(episode_records):
    return sum(float(r["reward"]) for r in episode_records)


def run_grid_animation_compare(
    outputs_dir: str | Path,
    episode_idx: int = 0,
    trained_traj_name: str = DEFAULT_TRAINED_TRAJ_NAME,
    random_traj_name: str = DEFAULT_RANDOM_TRAJ_NAME,
    fps: int = 5,
    interval_ms: int = 250,
    save_gif: bool = True,
    save_mp4: bool = False,
    show_plot: bool = True,
):
    outputs_dir = Path(outputs_dir)

    trained_traj_path = outputs_dir / trained_traj_name
    random_traj_path = outputs_dir / random_traj_name

    trained_records = load_trajectory(trained_traj_path)
    random_records = load_trajectory(random_traj_path)

    trained_episode = filter_episode(trained_records, episode_idx)
    random_episode = filter_episode(random_records, episode_idx)

    if not trained_episode:
        raise ValueError(f"No trained records found for episode {episode_idx}")
    if not random_episode:
        raise ValueError(f"No random records found for episode {episode_idx}")

    trained_grid_size = infer_grid_size_from_records(trained_records)
    random_grid_size = infer_grid_size_from_records(random_records)

    if trained_grid_size != random_grid_size:
        raise ValueError(
            f"Grid size mismatch: trained={trained_grid_size}, random={random_grid_size}"
        )

    grid_size = trained_grid_size

    tx, ty = build_path_xy(trained_episode, grid_size)
    rx, ry = build_path_xy(random_episode, grid_size)

    trained_reward = get_episode_total_reward(trained_episode)
    random_reward = get_episode_total_reward(random_episode)

    n_frames = max(len(tx), len(rx))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"{outputs_dir.name} | Episode {episode_idx} | "
        f"Trained reward={trained_reward:.2f} | Random reward={random_reward:.2f}"
    )

    ax_t, ax_r = axes
    draw_grid(ax_t, grid_size)
    draw_grid(ax_r, grid_size)

    ax_t.set_title("Trained Agent")
    ax_r.set_title("Random Agent")

    # empty artists
    line_t, = ax_t.plot([], [], marker="o", linewidth=2, markersize=4, label="trajectory")
    line_r, = ax_r.plot([], [], marker="o", linewidth=2, markersize=4, label="trajectory")

    start_t = ax_t.scatter([], [], s=120, marker="s", label="start")
    start_r = ax_r.scatter([], [], s=120, marker="s", label="start")

    curr_t = ax_t.scatter([], [], s=140, marker="o", label="current")
    curr_r = ax_r.scatter([], [], s=140, marker="o", label="current")

    end_t = ax_t.scatter([], [], s=140, marker="*", label="end")
    end_r = ax_r.scatter([], [], s=140, marker="*", label="end")

    text_t = ax_t.text(0.02, 1.02, "", transform=ax_t.transAxes, fontsize=10)
    text_r = ax_r.text(0.02, 1.02, "", transform=ax_r.transAxes, fontsize=10)

    ax_t.legend(loc="upper right")
    ax_r.legend(loc="upper right")

    # set fixed start/end
    start_t.set_offsets([[tx[0], ty[0]]])
    start_r.set_offsets([[rx[0], ry[0]]])

    end_t.set_offsets([[tx[-1], ty[-1]]])
    end_r.set_offsets([[rx[-1], ry[-1]]])

    def update(frame_idx):
        # trained
        t_end = min(frame_idx + 1, len(tx))
        line_t.set_data(tx[:t_end], ty[:t_end])
        curr_t.set_offsets([[tx[t_end - 1], ty[t_end - 1]]])
        text_t.set_text(f"step = {t_end - 1}")

        # random
        r_end = min(frame_idx + 1, len(rx))
        line_r.set_data(rx[:r_end], ry[:r_end])
        curr_r.set_offsets([[rx[r_end - 1], ry[r_end - 1]]])
        text_r.set_text(f"step = {r_end - 1}")

        return (
            line_t, line_r,
            start_t, start_r,
            curr_t, curr_r,
            end_t, end_r,
            text_t, text_r,
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    if save_gif:
        gif_path = outputs_dir / f"grid_compare_episode{episode_idx}.gif"
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        print(f"saved gif to: {gif_path}")

    if save_mp4:
        mp4_path = outputs_dir / f"grid_compare_episode{episode_idx}.mp4"
        anim.save(mp4_path, fps=fps)
        print(f"saved mp4 to: {mp4_path}")

    if show_plot:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)