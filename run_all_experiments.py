"""
Run all 4 experiments defined in experiments.md sequentially.
Each experiment runs in its own output directory with a clear name.

Usage:
    python run_all_experiments.py          # run all
    python run_all_experiments.py --only A  # run only Experiment A
"""

import sys
import json
from pathlib import Path

from experiment_runner import run_single_experiment

# Ensure Algorithm is importable
sys.path.insert(0, str(Path(__file__).parent / "Algorithm"))


# ============================================================
# Shared defaults
# ============================================================

SHARED = dict(
    episodes=10000,
    max_steps=200,
    seed=42,
    n=1,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.2,
    evaluate_random=True,
    log_every=500,
    checkpoint_every=2000,
    save_checkpoints=True,
    save_best_checkpoint=True,
)


def exp_a_grid_baseline():
    """Grid environment: RL vs Random baseline."""
    return run_single_experiment(
        mode="train",
        run_name="expA_grid_n1",
        env_type="grid",
        **SHARED,
    )


def exp_b_graph_baseline():
    """Graph environment: RL vs Random baseline."""
    return run_single_experiment(
        mode="train",
        run_name="expB_graph_n1",
        env_type="graph",
        env_kwargs={
            "center_coords": (22.894, 113.478),
            "view_radius": 3000,
            "cache_path": "cache/taxi_graph.graphml",
            "meters_per_step": 1000.0,
        },
        **SHARED,
    )


def exp_c_n_sweep():
    """Grid environment: sweep n = [1, 3, 5]."""
    results = []
    for n in [1, 3, 5]:
        r = run_single_experiment(
            mode="train",
            run_name=f"expC_grid_n{n}",
            env_type="grid",
            episodes=5000,
            n=n,
            **{k: v for k, v in SHARED.items() if k not in ("episodes", "n")},
        )
        results.append(r)
    return results


def exp_d_ms_sweep():
    """Graph environment: sweep meters_per_step = [300, 500, 800, 1000]."""
    results = []
    for ms in [300, 500, 800, 1000]:
        r = run_single_experiment(
            mode="train",
            run_name=f"expD_graph_ms{int(ms)}",
            env_type="graph",
            env_kwargs={
                "center_coords": (22.894, 113.478),
                "view_radius": 3000,
                "cache_path": "cache/taxi_graph.graphml",
                "meters_per_step": float(ms),
            },
            episodes=5000,
            **{k: v for k, v in SHARED.items() if k not in ("episodes",)},
        )
        results.append(r)
    return results


EXPERIMENTS = {
    "A": ("Grid RL vs Random", exp_a_grid_baseline),
    "B": ("Graph RL vs Random", exp_b_graph_baseline),
    "C": ("n-step Sweep (Grid)", exp_c_n_sweep),
    "D": ("meters_per_step Sweep (Graph)", exp_d_ms_sweep),
}


def print_all_results(all_results):
    """Print a summary table of all experiments."""
    print("\n" + "=" * 90)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 90)
    for exp_id, (name, _) in EXPERIMENTS.items():
        if exp_id in all_results:
            for r in all_results[exp_id]:
                s = r.get("trained_summary_metrics", {})
                print(f"[{exp_id}] {r['config']['run_name']}")
                print(f"  mean_reward={s.get('mean_total_reward', '?'):.2f}  "
                      f"profit/time={s.get('mean_profit_per_time', '?'):.3f}  "
                      f"completion_rate={s.get('mean_completion_rate', '?'):.3f}  "
                      f"empty_drive={s.get('mean_empty_drive_ratio', '?'):.3f}")
                if r.get("random_summary_metrics"):
                    rs = r["random_summary_metrics"]
                    print(f"  [random] mean_reward={rs.get('mean_total_reward', '?'):.2f}  "
                          f"profit/time={rs.get('mean_profit_per_time', '?'):.3f}")
    print("=" * 90)

    # Save summary
    summary_path = Path("outputs") / "all_experiments_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    simple = {}
    for exp_id in all_results:
        simple[exp_id] = []
        for r in all_results[exp_id]:
            simple[exp_id].append({
                "run_name": r["config"]["run_name"],
                "trained": r.get("trained_summary_metrics", {}),
                "random": r.get("random_summary_metrics", {}),
            })
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(simple, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    only = None
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        if idx + 1 < len(sys.argv):
            only = set(sys.argv[idx + 1].upper().split(","))

    all_results = {}

    for exp_id, (name, func) in EXPERIMENTS.items():
        if only and exp_id not in only:
            continue
        print(f"\n{'='*60}")
        print(f"Running Experiment {exp_id}: {name}")
        print(f"{'='*60}")
        try:
            result = func()
            # Ensure list
            if not isinstance(result, list):
                result = [result]
            all_results[exp_id] = result
        except Exception as e:
            print(f"[ERROR] Experiment {exp_id} failed: {e}")
            import traceback
            traceback.print_exc()

    print_all_results(all_results)
