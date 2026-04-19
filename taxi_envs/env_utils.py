from __future__ import annotations

from typing import Dict, List

import numpy as np

from .taxi_env import TaxiDispatchEnv
from .graph_taxi_env import GraphTaxiDispatchEnv


def make_env(
	config: Dict | None = None,
	num_taxis: int = 1,
	grid_size: int = 5,
	max_steps: int = 96,
	seed: int | None = None,
) -> TaxiDispatchEnv:
	_ = num_taxis
	if config is None:
		n_zones = grid_size * grid_size
		config = {
			"N_zones": n_zones,
			"episode_length": max_steps,
			"demand_matrix": np.full((n_zones, max_steps), 0.5, dtype=float),
			"destination_distribution": np.full(
				(n_zones, max_steps, n_zones),
				1.0 / n_zones,
				dtype=float,
			),
			"travel_time_matrix": _build_travel_time_matrix(n_zones, grid_size),
		}
	env = TaxiDispatchEnv(config=config)
	if seed is not None:
		env.reset(seed=seed)
	return env


def make_graph_env(
	config: Dict | None = None,
	num_taxis: int = 1,
	max_orders: int = 1,
	max_steps: int = 96,
	seed: int | None = None,
	center_coords: tuple[float, float] = (22.894, 113.478),
	view_radius: int = 3000,
	network_type: str = "all",
	meters_per_step: float = 1000.0,
	cache_path: str | None = None,
) -> GraphTaxiDispatchEnv:
	env = GraphTaxiDispatchEnv(
		config=config,
		max_steps=max_steps,
		center_coords=center_coords,
		view_radius=view_radius,
		network_type=network_type,
		meters_per_step=meters_per_step,
		cache_path=cache_path,
	)
	if seed is not None:
		env.reset(seed=seed)
	return env


def list_valid_dispatches(env: TaxiDispatchEnv) -> List[int]:
	return list(range(env.action_space.n))


def sample_dispatch(env: TaxiDispatchEnv, rng: np.random.Generator | None = None) -> int:
	candidates = list_valid_dispatches(env)
	if not candidates:
		return 0
	if rng is None:
		rng = np.random.default_rng()
	return candidates[int(rng.integers(0, len(candidates)))]


def build_grid_observation(env: TaxiDispatchEnv) -> np.ndarray:
	grid_size = int(round(env.n_zones ** 0.5))
	if grid_size * grid_size != env.n_zones:
		raise ValueError("n_zones must be a perfect square for grid visualization")
	grid = np.zeros((1, grid_size, grid_size), dtype=np.float32)
	x = env.current_zone // grid_size
	y = env.current_zone % grid_size
	grid[0, x, y] = 1.0
	return grid


def _build_travel_time_matrix(n_zones: int, grid_size: int) -> np.ndarray:
	if grid_size * grid_size != n_zones:
		matrix = np.ones((n_zones, n_zones), dtype=int)
		np.fill_diagonal(matrix, 0)
		return matrix
	coords = [(i // grid_size, i % grid_size) for i in range(n_zones)]
	matrix = np.zeros((n_zones, n_zones), dtype=int)
	for i, (x1, y1) in enumerate(coords):
		for j, (x2, y2) in enumerate(coords):
			matrix[i, j] = abs(x1 - x2) + abs(y1 - y2)
	return matrix
