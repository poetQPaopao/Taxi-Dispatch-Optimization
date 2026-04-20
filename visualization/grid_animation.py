from __future__ import annotations
import numpy as np
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection


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
        ax.axhline(i - 0.5, linewidth=0.6, alpha=0.35)
        ax.axvline(i - 0.5, linewidth=0.6, alpha=0.35)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_aspect("equal")
    # ax.set_xlabel("Column")
    # ax.set_ylabel("Row")


def get_episode_total_reward(episode_records):
    return sum(float(r["reward"]) for r in episode_records)


def get_pending_matrix(episode_records, frame_idx: int, grid_size: int):
    '''
    Transform the pending orders information into matrix (for heatmap)
    '''
    if not episode_records:
        raise ValueError("episode_records is empty")

    rec_idx = min(frame_idx, len(episode_records) - 1)
    pending_counts = episode_records[rec_idx].get("pending_counts", [])

    if not pending_counts:
        return np.zeros((grid_size, grid_size), dtype=float)

    if len(pending_counts) != grid_size * grid_size:
        raise ValueError(
            f"pending_counts length {len(pending_counts)} does not match "
            f"grid size {grid_size}x{grid_size}"
        )

    return np.array(pending_counts, dtype=float).reshape(grid_size, grid_size)


def build_fading_segments(xs, ys, end_idx: int, tail_length: int = 5):
    """
    Build recent trajectory segments with fading alpha.
    Only keep the last `tail_length` steps.
    """
    if end_idx <= 0:
        return [], []

    start_idx = max(0, end_idx - tail_length)
    seg_points = []

    for i in range(start_idx, end_idx):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        seg_points.append([(x0, y0), (x1, y1)])

    if not seg_points:
        return [], []

    n = len(seg_points)
    alphas = np.linspace(0.08, 0.85, n)
    return seg_points, alphas


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

    n_frames = max(len(trained_episode), len(random_episode))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"{outputs_dir.name} | Episode {episode_idx}\n"
        f"Demand Heatmap + Taxi Motion | "
        f"Trained reward={trained_reward:.2f} | Random reward={random_reward:.2f}"
    )

    ax_t, ax_r = axes
    draw_grid(ax_t, grid_size)
    draw_grid(ax_r, grid_size)

    ax_t.set_title("Trained Agent")
    ax_r.set_title("Random Agent")


    # heatmap background
    heat_t = ax_t.imshow(
        get_pending_matrix(trained_episode, 0, grid_size),
        extent=(-0.5, grid_size - 0.5, grid_size - 0.5, -0.5),
        interpolation="bilinear",
        alpha=0.55,
        cmap="YlOrRd",
    )

    heat_r = ax_r.imshow(
        get_pending_matrix(random_episode, 0, grid_size),
        extent=(-0.5, grid_size - 0.5, grid_size - 0.5, -0.5),
        interpolation="bilinear",
        alpha=0.55,
        cmap="YlOrRd",
    )

    # fading tail artists
    tail_t = LineCollection([], linewidths=3.0)
    tail_r = LineCollection([], linewidths=3.0)
    ax_t.add_collection(tail_t)
    ax_r.add_collection(tail_r)

    # current taxi point
    curr_t = ax_t.scatter([], [], s=160, marker="o", edgecolors="black", linewidths=0.8, zorder=5)
    curr_r = ax_r.scatter([], [], s=160, marker="o", edgecolors="black", linewidths=0.8, zorder=5)

    # texts
    text_t = ax_t.text(0.02, 0.02, "", transform=ax_t.transAxes, fontsize=10)
    text_r = ax_r.text(0.02, 0.02, "", transform=ax_r.transAxes, fontsize=10)

    # optional colorbars
    fig.colorbar(heat_t, ax=ax_t, fraction=0.046, pad=0.04)
    fig.colorbar(heat_r, ax=ax_r, fraction=0.046, pad=0.04)



    def update(frame_idx):
            tail_length = 8

            # ===== trained =====
            t_end = min(frame_idx, len(tx) - 1)
            t_segments, t_alphas = build_fading_segments(tx, ty, t_end, tail_length=tail_length)

            if t_segments:
                tail_t.set_segments(t_segments)
                tail_t.set_alpha(t_alphas)
            else:
                tail_t.set_segments([])

            curr_t.set_offsets([[tx[t_end], ty[t_end]]])

            t_rec_idx = min(frame_idx, len(trained_episode) - 1)
            t_rec = trained_episode[t_rec_idx]
            t_reward = float(t_rec.get("reward", 0.0))
            t_matched = bool(t_rec.get("matched", False))
            t_pending = get_pending_matrix(trained_episode, frame_idx, grid_size)
            heat_t.set_data(t_pending)

            t_zone = int(t_rec["next_state"][0])
            t_zone_orders = int(t_rec.get("pending_counts", [0] * (grid_size * grid_size))[t_zone])
            t_total_orders = int(sum(t_rec.get("pending_counts", [])))

            text_t.set_text(
                f"step={t_rec_idx} | reward={t_reward:.2f} | matched={t_matched}\n"
                f"current_zone={t_zone} | orders_here={t_zone_orders} | total_pending={t_total_orders}"
            )

            # ===== random =====
            r_end = min(frame_idx, len(rx) - 1)
            r_segments, r_alphas = build_fading_segments(rx, ry, r_end, tail_length=tail_length)

            if r_segments:
                tail_r.set_segments(r_segments)
                tail_r.set_alpha(r_alphas)
            else:
                tail_r.set_segments([])

            curr_r.set_offsets([[rx[r_end], ry[r_end]]])

            r_rec_idx = min(frame_idx, len(random_episode) - 1)
            r_rec = random_episode[r_rec_idx]
            r_reward = float(r_rec.get("reward", 0.0))
            r_matched = bool(r_rec.get("matched", False))
            r_pending = get_pending_matrix(random_episode, frame_idx, grid_size)
            heat_r.set_data(r_pending)

            r_zone = int(r_rec["next_state"][0])
            r_zone_orders = int(r_rec.get("pending_counts", [0] * (grid_size * grid_size))[r_zone])
            r_total_orders = int(sum(r_rec.get("pending_counts", [])))

            text_r.set_text(
                f"step={r_rec_idx} | reward={r_reward:.2f} | matched={r_matched}\n"
                f"current_zone={r_zone} | orders_here={r_zone_orders} | total_pending={r_total_orders}"
            )

            return (
                heat_t, heat_r,
                tail_t, tail_r,
                curr_t, curr_r,
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

    