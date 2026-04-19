from pathlib import Path
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json

def make_run_dir(root="outputs"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_reward_curve(reward_history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not reward_history:
        return

    episodes_list = [row[0] for row in reward_history]
    rewards = [row[1] for row in reward_history]

    plt.figure(figsize=(8, 4))
    plt.plot(episodes_list, rewards, linewidth=1.2, alpha=0.5, label="raw reward")

    # moving average
    window = min(20, len(rewards))
    if window >= 2:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(
            episodes_list[window - 1:],
            smoothed,
            linewidth=2.0,
            label=f"moving average (w={window})",
        )

    plt.title("Training Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)