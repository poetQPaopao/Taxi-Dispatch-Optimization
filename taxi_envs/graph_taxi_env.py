import os
from typing import Dict, List, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import osmnx as ox


class GraphOrder:
    def __init__(self, origin: int, dest: int, fare: float, time_generated: int):
        self.origin = origin
        self.dest = dest
        self.fare = fare
        self.time_generated = time_generated
    
    def __repr__(self) -> str:
        return f"GraphOrder(to:{self.dest}, fare:{self.fare:.1f})"


def load_or_create_graph(
    center_coords: Tuple[float, float],
    dist: int,
    network_type: str,
    cache_path: str,
    intersection_tolerance: float,
) -> nx.MultiDiGraph:
    if os.path.exists(cache_path):
        return ox.load_graphml(cache_path)

    graph = ox.graph_from_point(
        center_coords,
        dist=dist,
        custom_filter=(
            '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified"]'
        ),
        simplify=True,
    )
    graph = _largest_component(graph, strongly=True)
    graph = ox.project_graph(graph)
    graph = ox.consolidate_intersections(
        graph,
        tolerance=intersection_tolerance,
        rebuild_graph=True,
        dead_ends=False,
    )
    graph = _largest_component(graph, strongly=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    ox.save_graphml(graph, cache_path)
    return graph


class GraphTaxiDispatchEnv(gym.Env):
    def __init__(
        self,
        config: Dict | None = None,
        max_steps: int = 96,
        center_coords: Tuple[float, float] = (22.7952, 113.5583),
        view_radius: int = 3000,
        network_type: str = "all",
        meters_per_step: float = 50.0,
        cache_path: str | None = None,
        intersection_tolerance: float = 20.0,
    ):
        super().__init__()
        cfg = config or {}

        def _get(name: str, default):
            if isinstance(cfg, dict) and name in cfg:
                return cfg[name]
            if hasattr(cfg, name):
                return getattr(cfg, name)
            return default

        self.center_coords = center_coords
        self.view_radius = view_radius
        self.network_type = network_type
        self.meters_per_step = meters_per_step

        if cache_path is None:
            cache_path = os.path.join("cache", "taxi_graph.graphml")
        self.cache_path = cache_path

        self.graph = load_or_create_graph(
            center_coords=center_coords,
            dist=view_radius,
            network_type=network_type,
            cache_path=cache_path,
            intersection_tolerance=intersection_tolerance,
        )
        self.node_ids = list(self.graph.nodes)
        if not self.node_ids:
            raise RuntimeError("Graph contains no nodes.")
        self.node_ids.sort()
        self.node_to_index = {node_id: idx for idx, node_id in enumerate(self.node_ids)}

        self.n_zones = int(_get("N_zones", _get("num_zones", len(self.node_ids))))
        self.episode_length = int(_get("episode_length", max_steps))
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
        self.travel_time_matrix = (
            np.asarray(travel_time_matrix, dtype=int)
            if travel_time_matrix is not None
            else None
        )

        self._action_space = spaces.Discrete(self.n_zones)
        self._observation_space = spaces.Tuple(
            (spaces.Discrete(self.n_zones), spaces.Discrete(self.episode_length))
        )

        self.rng = np.random.default_rng()
        self.current_zone = 0
        self.current_time = 0
        self.pending_orders: List[List[GraphOrder]] = [[] for _ in range(self.n_zones)]
        self.total_orders = 0
        self.completed_orders = 0
        self._travel_time_cache: Dict[Tuple[int, int], int | None] = {}

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, seed: int | None = None, options=None):
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

    def _shortest_path_length(self, source: int, target: int) -> float | None:
        try:
            return float(nx.shortest_path_length(self.graph, source, target, weight="length"))
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None



    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, bool, Dict]:
        if not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an int target_zone")
        if action < 0 or action >= self.n_zones:
            return self._get_state(), -1.0, False, False, {"illegal": True}

        target_zone = int(action)
        travel_time = self._travel_time(self.current_zone, target_zone)
        if travel_time is None:
            return self._get_state(), -1.0, False, False, {"illegal": True, "no_path": True}

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

        matched_order = self._match_order(self.current_zone)
        if matched_order is None:
            return self._get_state(), move_reward, False, False, {
                "matched": False,
                "time_elapsed": time_elapsed,
            }

        trip_time = self._travel_time(self.current_zone, matched_order.dest)
        if trip_time is None:
            return self._get_state(), move_reward, False, False, {
                "matched": False,
                "time_elapsed": time_elapsed,
                "no_path": True,
            }
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
        self.current_time = arrival_time
        self._generate_orders_for_time(self.current_time)

        total_reward = move_reward + trip_reward
        return self._get_state(), total_reward, False, False, {
            "matched": True,
            "time_elapsed": time_elapsed,
        }

    def render(self):
        print("GraphTaxiDispatchEnv: render not implemented.")
        pass

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
                trip_time = self._travel_time(zone, dest)
                if trip_time is None:
                    continue
                fare = self.base_fare + self.rate_per_step * trip_time
                self.pending_orders[zone].append(GraphOrder(zone, dest, fare, time_step))

    def _match_order(self, zone: int) -> GraphOrder | None:
        orders = self.pending_orders[zone]
        if not orders:
            return None
        return orders[int(self.rng.integers(0, len(orders)))]

    def _travel_time(self, origin_zone: int, dest_zone: int) -> int | None:
        if self.travel_time_matrix is not None:
            return int(self.travel_time_matrix[origin_zone, dest_zone])

        key = (origin_zone, dest_zone)
        if key in self._travel_time_cache:
            return self._travel_time_cache[key]

        if origin_zone >= len(self.node_ids) or dest_zone >= len(self.node_ids):
            self._travel_time_cache[key] = None
            return None

        source_node = self.node_ids[origin_zone]
        target_node = self.node_ids[dest_zone]
        length_m = self._shortest_path_length(source_node, target_node)
        if length_m is None:
            self._travel_time_cache[key] = None
            return None
        steps = int(np.ceil(length_m / self.meters_per_step))
        self._travel_time_cache[key] = steps
        return steps
    

def _largest_component(graph: nx.MultiDiGraph, strongly: bool) -> nx.MultiDiGraph:
    if strongly:
        components = nx.strongly_connected_components(graph)
    else:
        components = nx.connected_components(graph.to_undirected())
    largest = max(components, key=len, default=None)
    if not largest:
        return graph
    return graph.subgraph(largest).copy()
