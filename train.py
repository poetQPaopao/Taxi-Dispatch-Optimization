"""Quick training entry point. For full experiments see run_all_experiments.py."""
from experiment_runner import run_single_experiment


if __name__ == "__main__":
    result = run_single_experiment(
        mode="train",
        run_name="train_debug",
        episodes=1000,
        max_steps=200,
        seed=42,
        env_type="grid",
        n=1,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.2,
        evaluate_random=True,
        log_every=100,
        checkpoint_every=500,
        save_checkpoints=True,
        save_best_checkpoint=True,
    )
    s = result["trained_summary_metrics"]
    print(f"Done. mean_reward={s.get('mean_total_reward', '?'):.2f}, "
          f"profit/time={s.get('mean_profit_per_time', '?'):.3f}, "
          f"completion_rate={s.get('mean_completion_rate', '?'):.3f}")
