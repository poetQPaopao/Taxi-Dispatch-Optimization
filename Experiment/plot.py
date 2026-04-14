from pathlib import Path
import matplotlib.pyplot as plt


def plot_reward_curve(results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = [r["episode"] for r in results]
    rewards = [r["episode_reward"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, rewards, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward Curve")
    plt.tight_layout()
    plt.savefig(out_dir / "reward_curve.png")
    plt.close()


def plot_episode_metrics(results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = [r["episode"] for r in results]
    completed = [r["completed_orders"] for r in results]
    illegal = [r["illegal_actions"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, completed, marker="o", label="completed_orders")
    plt.plot(episodes, illegal, marker="o", label="illegal_actions")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.title("Episode Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "episode_metrics.png")
    plt.close()