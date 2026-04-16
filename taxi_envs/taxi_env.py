import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple


class Taxi:
    def __init__(self, zone_id: int):
        self.zone_id = zone_id
        self.is_free = True


class Order:
    def __init__(self, origin: int, dest: int, fare: float, time_generated: int):
        self.origin = origin
        self.dest = dest
        self.fare = fare
        self.time_generated = time_generated
    
    def __repr__(self):
        return f"Order(to:{self.dest}, fare:{self.fare:.1f})"


class TaxiDispatchEnv(gym.Env):
    def __init__(self, config: Dict | None = None):
        super().__init__()
        cfg = config or {}

        def _get(name: str, default):
            if isinstance(cfg, dict) and name in cfg:
                return cfg[name]
            if hasattr(cfg, name):
                return getattr(cfg, name)
            return default

        self.n_zones = int(_get("N_zones", _get("num_zones", 25)))
        self.episode_length = int(_get("episode_length", 96))
        self.cost_empty = float(_get("cost_empty", 1.0))
        self.cost_occupied = float(_get("cost_occupied", 0.5))
        self.base_fare = float(_get("base_fare", 5.0))
        self.rate_per_step = float(_get("rate_per_step", 1.0))

        demand_matrix = _get("demand_matrix", None)
        destination_distribution = _get("destination_distribution", None)
        travel_time_matrix = _get("travel_time_matrix", None)

        self.demand_matrix = self._init_demand_matrix(demand_matrix)
        self.destination_distribution = self._init_destination_distribution(
            destination_distribution
        )
        self.travel_time_matrix = self._init_travel_time_matrix(travel_time_matrix)

        self._action_space = spaces.Discrete(self.n_zones)
        self._observation_space = spaces.Tuple(
            (spaces.Discrete(self.n_zones), spaces.Discrete(self.episode_length))
        )

        self.rng = np.random.default_rng()
        self.current_zone = 0
        self.current_time = 0
        self.pending_orders: List[List[Order]] = [[] for _ in range(self.n_zones)]
        self.total_orders = 0
        self.completed_orders = 0

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


    def reset(self, seed: int | None = None, options: Dict | None = None):
        # Gymnasium reset returns (obs, info).
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_time = 0
        self.current_zone = int(self.rng.integers(0, self.n_zones))
        self.total_orders = 0
        self.completed_orders = 0
        self._generate_orders_for_time(self.current_time)
        
        return self._get_state(), {}

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, bool, Dict]:
        if not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an int target_zone")
        if action < 0 or action >= self.n_zones:
            return self._get_state(), -1.0, False, False, {"illegal": True}

        target_zone = int(action)

        travel_time = int(self.travel_time_matrix[self.current_zone, target_zone])
        if target_zone == self.current_zone:
            # Waiting should advance time but not incur empty-move cost.
            travel_time = max(1, travel_time)
            move_reward = 0.0
        else:
            move_reward = -self.cost_empty * travel_time

        next_time = self.current_time + travel_time
        time_elapsed = travel_time

        # Stop if we would cross the episode horizon.
        if next_time >= self.episode_length:
            remaining = self.episode_length - self.current_time
            self.current_zone = target_zone
            self.current_time = self.episode_length - 1
            return self._get_state(), move_reward, False, True, {"time_elapsed": remaining}

        # Advance to the repositioned zone and generate orders for the new time.
        self.current_zone = target_zone
        self.current_time = next_time
        self._generate_orders_for_time(self.current_time)

        matched_order = self._match_order(self.current_zone, self.current_time)
        
        # No match: only movement reward applies.
        if matched_order is None:
            return self._get_state(), move_reward, False, False, {"matched": False, "time_elapsed": time_elapsed}

        # Matched: check whether the trip fits in the remaining horizon.
        trip_time = int(self.travel_time_matrix[self.current_zone, matched_order.dest])
        arrival_time = self.current_time + trip_time
        if arrival_time >= self.episode_length:
            remaining = self.episode_length - self.current_time
            self.current_time = self.episode_length - 1
            return self._get_state(), move_reward, False, True, {
                "matched": True,
                "trip_blocked": True,
                "time_elapsed": remaining,
            }

        time_elapsed += trip_time

        trip_reward = matched_order.fare - self.cost_occupied * trip_time
        self.completed_orders += 1
        self.current_zone = matched_order.dest
        total_reward = move_reward + trip_reward

        # Normal arrival: generate orders for the new time step.
        self.current_time = arrival_time
        self._generate_orders_for_time(self.current_time)
        
        return self._get_state(), total_reward, False, False, {"matched": True, "time_elapsed": time_elapsed}
    
    def render(self, mode: str = "human") -> None:
        if mode != "human":
            return
        pending = [len(orders) for orders in self.pending_orders]
        print(
            f"time={self.current_time} zone={self.current_zone} "
            f"pending_orders={pending}"
        )

    def _get_state(self) -> Tuple[int, int]:
        return self.current_zone, self.current_time

    def _init_demand_matrix(self, demand_matrix):
        if demand_matrix is None:
            return np.full((self.n_zones, self.episode_length), 0.5, dtype=float)
        return np.asarray(demand_matrix, dtype=float)

    def _init_destination_distribution(self, destination_distribution):
        if destination_distribution is None:
            dist = np.full(
                (self.n_zones, self.episode_length, self.n_zones),
                1.0 / self.n_zones,
                dtype=float,
            )
            return dist
        dist = np.asarray(destination_distribution, dtype=float)
        totals = dist.sum(axis=-1, keepdims=True)
        totals[totals == 0.0] = 1.0
        return dist / totals

    def _init_travel_time_matrix(self, travel_time_matrix):
        if travel_time_matrix is not None:
            return np.asarray(travel_time_matrix, dtype=int)

        grid_size = int(round(self.n_zones ** 0.5))
        if grid_size * grid_size == self.n_zones:
            coords = [(i // grid_size, i % grid_size) for i in range(self.n_zones)]
            matrix = np.zeros((self.n_zones, self.n_zones), dtype=int)
            for i, (x1, y1) in enumerate(coords):
                for j, (x2, y2) in enumerate(coords):
                    matrix[i, j] = abs(x1 - x2) + abs(y1 - y2)
            return matrix

        matrix = np.ones((self.n_zones, self.n_zones), dtype=int)
        np.fill_diagonal(matrix, 0)
        return matrix

    def _generate_orders_for_time(self, time_step: int) -> None:
        self.pending_orders = [[] for _ in range(self.n_zones)]
        for zone in range(self.n_zones):
            lam = float(self.demand_matrix[zone, time_step])
            if lam <= 0.0:
                continue
            count = int(self.rng.poisson(lam))
            if count <= 0:
                continue
            self.total_orders += count
            probs = self.destination_distribution[zone, time_step]
            for _ in range(count):
                dest = int(self.rng.choice(self.n_zones, p=probs))
                trip_time = int(self.travel_time_matrix[zone, dest])
                fare = self.base_fare + self.rate_per_step * trip_time
                self.pending_orders[zone].append(Order(zone, dest, fare, time_step))

    def _match_order(self, zone: int, time_step: int) -> Order | None:
        orders = self.pending_orders[zone]
        if not orders:
            return None
        return orders[int(self.rng.integers(0, len(orders)))]