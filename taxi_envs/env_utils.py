from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .taxi_env import TaxiDispatchEnv
from .graph_taxi_env import GraphTaxiDispatchEnv


def make_env(
	num_taxis: int = 5,
	grid_size: int = 10,
	max_steps: int = 200,
	seed: int | None = None,
) -> TaxiDispatchEnv:
	env = TaxiDispatchEnv(num_taxis=num_taxis, grid_size=grid_size, max_steps=max_steps)
	if seed is not None:
		np.random.seed(seed)
	return env


def make_graph_env(
	num_taxis: int = 5,
	max_orders: int = 10,
	max_steps: int = 200,
	seed: int | None = None,
	center_coords: tuple[float, float] = (22.7952, 113.5583),
	view_radius: int = 3000,
	network_type: str = "all",
	meters_per_step: float = 50.0,
	cache_path: str | None = None,
) -> GraphTaxiDispatchEnv:
	env = GraphTaxiDispatchEnv(
		num_taxis=num_taxis,
		max_orders=max_orders,
		max_steps=max_steps,
		center_coords=center_coords,
		view_radius=view_radius,
		network_type=network_type,
		meters_per_step=meters_per_step,
		cache_path=cache_path,
	)
	if seed is not None:
		np.random.seed(seed)
	return env


def list_valid_dispatches(env: TaxiDispatchEnv) -> List[Tuple[int, int]]:
	valid: List[Tuple[int, int]] = []
	for taxi in env.taxis:
		if not taxi.is_free:
			continue
		for order in env.orders:
			if order.picked_up or order.finished:
				continue
			if order.created_time > 0:
				continue
			valid.append((taxi.id, order.id))
	return valid


def sample_dispatch(env: TaxiDispatchEnv, rng: np.random.Generator | None = None) -> Tuple[int, int]:
	candidates = list_valid_dispatches(env)
	if not candidates:
		return 0, 0
	if rng is None:
		rng = np.random.default_rng()
	return candidates[int(rng.integers(0, len(candidates)))]


def build_grid_observation(env: TaxiDispatchEnv) -> np.ndarray:
	grid = np.zeros((3, env.grid_size, env.grid_size), dtype=np.float32)

	for taxi in env.taxis:
		x, y = taxi.loc
		grid[0, x, y] = 1.0

	for order in env.orders:
		if order.picked_up or order.finished:
			continue
		sx, sy = order.start
		grid[1, sx, sy] = 1.0
		waiting_time = max(0, env.current_step - order.created_time)
		grid[2, sx, sy] = float(min(255, waiting_time))

	return grid
