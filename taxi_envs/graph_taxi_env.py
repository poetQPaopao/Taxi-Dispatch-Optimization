import os
from typing import Dict, List, Tuple, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import osmnx as ox


class GraphTaxi:
    def __init__(self, taxi_id: int, node_id: int):
        self.id = taxi_id
        self.location = node_id
        self.is_free = True
        self.dest = None
        self.dispatched_order = None


class GraphOrder:
    def __init__(self, order_id: int, start_node: int, end_node: int):
        self.id = order_id
        self.start = start_node
        self.end = end_node
        self.created_time = 0
        self.picked_up = False
        self.finished = False


def load_or_create_graph(
    center_coords: Tuple[float, float],
    dist: int,
    network_type: str,
    cache_path: str,
) -> nx.MultiDiGraph:
    if os.path.exists(cache_path):
        return ox.load_graphml(cache_path)

    graph = ox.graph_from_point(center_coords, dist=dist, network_type=network_type)
    graph = ox.project_graph(graph)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    ox.save_graphml(graph, cache_path)
    return graph


class GraphTaxiDispatchEnv(gym.Env):
    def __init__(
        self,
        num_taxis: int = 5,
        max_orders: int = 10,
        max_steps: int = 200,
        center_coords: Tuple[float, float] = (22.7952, 113.5583),
        view_radius: int = 3000,
        network_type: str = "all",
        meters_per_step: float = 50.0,
        cache_path: str | None = None,
    ):
        super().__init__()
        self.num_taxis = num_taxis
        self.max_orders = max_orders
        self.max_steps = max_steps
        self.current_step = 0
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
        )
        self.node_ids = list(self.graph.nodes)
        if not self.node_ids:
            raise RuntimeError("Graph contains no nodes.")
        self.max_node_id = int(max(self.node_ids))

        self.action_space = spaces.Discrete(num_taxis * max_orders)
        self.observation_space = spaces.Dict({
            "taxis": spaces.Sequence(spaces.Dict({
                "id": spaces.Discrete(num_taxis),
                "location": spaces.Box(low=0, high=self.max_node_id, shape=(1,), dtype=np.int64),
                "is_free": spaces.Discrete(2),
            })),
            "orders": spaces.Sequence(spaces.Dict({
                "id": spaces.Discrete(max_orders),
                "pickup": spaces.Box(low=0, high=self.max_node_id, shape=(1,), dtype=np.int64),
                "created_time": spaces.Discrete(max_steps),
            })),
            "current_time": spaces.Discrete(max_steps),
        })

        self.taxis: List[GraphTaxi] = []
        self.orders: List[GraphOrder] = []
        self.reset()

    def _init_state(self) -> None:
        rng = np.random.default_rng()
        self.taxis = [
            GraphTaxi(i, int(rng.choice(self.node_ids)))
            for i in range(self.num_taxis)
        ]
        self.orders = [
            GraphOrder(
                i,
                int(rng.choice(self.node_ids)),
                int(rng.choice(self.node_ids)),
            )
            for i in range(self.max_orders)
        ]

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self._init_state()
        return self._get_observation()

    def _get_observation(self) -> Dict[str, Any]:
        obs = {
            "taxis": [],
            "orders": [],
            "current_time": self.current_step,
        }
        for taxi in self.taxis:
            obs["taxis"].append({
                "id": taxi.id,
                "location": taxi.location,
                "is_free": taxi.is_free,
            })
        for order in self.orders:
            if not order.finished:
                obs["orders"].append({
                    "id": order.id,
                    "pickup": order.start,
                    "created_time": order.created_time,
                })
        return obs

    def _action_to_pair(self, action_int: int, pending_order_ids: List[int]) -> Tuple[int, int] | None:
        taxi_id = action_int // self.max_orders
        slot = action_int % self.max_orders
        if slot >= len(pending_order_ids):
            return None
        return (taxi_id, pending_order_ids[slot])

    def _shortest_path_length(self, source: int, target: int) -> float | None:
        try:
            return float(nx.shortest_path_length(self.graph, source, target, weight="length"))
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

    def step(self, action: int | Tuple[int, int]):
        pending_ids = [o["id"] for o in self._get_observation()["orders"]]
        if isinstance(action, tuple):
            pair = action
        else:
            pair = self._action_to_pair(action, pending_ids)
        if pair is None:
            return self._get_observation(), -1.0, False, {"illegal": True}

        taxi_id, order_id = pair
        self.current_step += 1

        if taxi_id >= self.num_taxis:
            return self._get_observation(), -1.0, False, {"illegal": True}
        taxi = self.taxis[taxi_id]
        order = next((o for o in self.orders if o.id == order_id and not o.finished), None)
        if order is None:
            return self._get_observation(), -1.0, False, {"illegal": True}
        if not taxi.is_free or order.picked_up or order.finished:
            return self._get_observation(), -1.0, False, {"illegal": True}

        time_to_pickup = self._shortest_path_length(taxi.location, order.start)
        trip_distance = self._shortest_path_length(order.start, order.end)
        if time_to_pickup is None or trip_distance is None:
            return self._get_observation(), -1.0, False, {"illegal": True, "no_path": True}

        time_to_pickup_steps = int(np.ceil(time_to_pickup / self.meters_per_step))
        trip_steps = int(np.ceil(trip_distance / self.meters_per_step))
        total_travel_steps = time_to_pickup_steps + trip_steps

        pickup_time = self.current_step + time_to_pickup_steps
        waiting_time = pickup_time - order.created_time
        self.current_step += total_travel_steps

        reward = 20 - (waiting_time + trip_distance) / 1000.0

        taxi.location = order.end
        taxi.is_free = True
        taxi.dispatched_order = None

        order.picked_up = True
        order.finished = True

        done = all(o.finished for o in self.orders)
        if self.current_step >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, {
            "travel_time": time_to_pickup + trip_distance,
            "travel_steps": total_travel_steps,
        }

    def render(self):
        print("GraphTaxiDispatchEnv: render not implemented.")
        pass
