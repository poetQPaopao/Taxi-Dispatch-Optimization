"""Evaluation entry point. Load a trained model and evaluate it."""
from experiment_runner import run_single_experiment


if __name__ == "__main__":
    result = run_single_experiment(
        mode="eval",
        run_name="eval_debug",
        episodes=200,
        max_steps=200,
        seed=42,
        env_type="grid",
        n=1,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.0,
        load_path="outputs/train_debug/model.pkl",
        evaluate_random=True,
        make_visualizations=True,
        vis_episode_idx=0,
        log_every=20,
        save_checkpoints=False,
        save_best_checkpoint=False,
    )
    s = result["trained_summary_metrics"]
    rs = result.get("random_summary_metrics", {})
    print(f"\nTrained:  reward={s.get('mean_total_reward', '?'):.2f}, "
          f"profit/time={s.get('mean_profit_per_time', '?'):.3f}, "
          f"completion={s.get('mean_completion_rate', '?'):.3f}")
    print(f"Random:   reward={rs.get('mean_total_reward', '?'):.2f}, "
          f"profit/time={rs.get('mean_profit_per_time', '?'):.3f}")
