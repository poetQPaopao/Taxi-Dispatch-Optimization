
from __future__ import annotations

from experiment_runner import run_param_sweep, run_single_experiment


GRAPH_BASE_CONFIG = {
    "episodes": 30000,
    "max_steps": 200,
    "seed": 42,
    "env_type": "graph",
    "env_kwargs": {
        "center_coords": (22.894, 113.478),
        "view_radius": 3000,
        "cache_path": "cache/taxi_graph.graphml",
        "meters_per_step": 2000
    },
    "n": 1,
    "alpha": 0.1,
    "gamma": 0.95,
    "epsilon": 0.2,
    "save_model_name": "nstep_agent.pkl",
    "evaluate_random": True,
    "make_visualizations": False,
    "vis_episode_idx": 49,
}

GRID_BASE_CONFIG = {
    "episodes": 30000,
    "max_steps": 200,
    "seed": 42,
    "env_type": "grid",
    "env_kwargs": {},
    "n": 1,
    "alpha": 0.1,
    "gamma": 0.95,
    "epsilon": 0.2,
    "save_model_name": "nstep_agent.pkl",
    "evaluate_random": True,
    "make_visualizations": False,
    "vis_episode_idx": 99,
}


def run_one_debug_experiment():
    result = run_single_experiment(
        run_name="debug_graph_single",
        **GRAPH_BASE_CONFIG,
    )
    print(result)


def run_graph_sweep():
    # 我主要想看meters per step的影响
    sweep_grid = {
        "env_kwargs": [
            {
                "center_coords": (22.894, 113.478),
                "view_radius": 3000,
                "cache_path": "cache/taxi_graph.graphml",
                "meters_per_step": 300.0,
            },
            {
                "center_coords": (22.894, 113.478),
                "view_radius": 3000,
                "cache_path": "cache/taxi_graph.graphml",
                "meters_per_step": 500.0,
            },
            {
                "center_coords": (22.894, 113.478),
                "view_radius": 3000,
                "cache_path": "cache/taxi_graph.graphml",
                "meters_per_step": 800.0,
            },
            {
                "center_coords": (22.894, 113.478),
                "view_radius": 3000,
                "cache_path": "cache/taxi_graph.graphml",
                "meters_per_step": 1000.0,
            },
        ]
    }


    results = run_param_sweep(
        base_config=GRAPH_BASE_CONFIG,
        sweep_grid=sweep_grid,
        sweep_name="graph_m_per_step_sweep",
    )
    print("graph sweep finished. runs:", len(results))


def run_grid_sweep():
    sweep_grid = {
        "n": [1, 3, 5],
        "alpha": [0.1],
        "epsilon": [0.2],
    }
    results = run_param_sweep(
        base_config=GRID_BASE_CONFIG,
        sweep_grid=sweep_grid,
        sweep_name="grid_n",
    )
    print("grid sweep finished. runs:", len(results))


if __name__ == "__main__":
    # Pick one:
    # run_graph_sweep()
    # run_grid_sweep()
    run_one_debug_experiment()
