import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, List, Any

class Taxi:
    def __init__(self, id, loc):
        self.id = id
        self.loc = loc
        self.is_free = True
        self.dest = None
        self.dispatched_order = None

    def _move_towards(self, current, target):
        x, y = current
        tx, ty = target
        if x < tx:
            x += 1
        elif x > tx:
            x -= 1
        if y < ty:
            y += 1
        elif y > ty:
            y -= 1
        return [x, y]

class Order:
    def __init__(self, id, start, end):
        self.id = id
        self.start = start
        self.end = end
        self.created_time = 0
        self.picked_up = False
        self.finished = False

class TaxiDispatchEnv(gym.Env):
    def __init__(self, num_taxis=5, grid_size=10, max_orders=10, max_steps=200):
        super().__init__()
        self.num_taxis = num_taxis
        self.grid_size = grid_size
        self.max_orders = max_orders
        self.max_steps = max_steps
        self.current_step = 0

        # Action space: discrete index = taxi_id * max_orders + order_slot
        self.action_space = spaces.Discrete(num_taxis * max_orders)

        # Observation space: dictionary (to match state_encoder)
        self.observation_space = spaces.Dict({
            'taxis': spaces.Sequence(spaces.Dict({
                'id': spaces.Discrete(num_taxis),
                'location': spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=int),
                'is_free': spaces.Discrete(2)
            })),
            'orders': spaces.Sequence(spaces.Dict({
                'id': spaces.Discrete(max_orders),
                'pickup': spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=int),
                'created_time': spaces.Discrete(max_steps)
            })),
            'current_time': spaces.Discrete(max_steps)
        })

        self.taxis = []
        self.orders = []
        self.reset()

    def _init_state(self):
        self.taxis = [
            Taxi(i, [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])
            for i in range(self.num_taxis)
        ]
        self.orders = [
            Order(i,
                  [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)],
                  [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])
            for i in range(self.max_orders)
        ]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self._init_state()
        return self._get_observation()

    def _get_observation(self) -> Dict[str, Any]:
        obs = {
            'taxis': [],
            'orders': [],
            'current_time': self.current_step
        }
        for taxi in self.taxis:
            obs['taxis'].append({
                'id': taxi.id,
                'location': taxi.loc,      # key 'location'
                'is_free': taxi.is_free
            })
        # Only include unfinished orders
        for order in self.orders:
            if not order.finished:
                obs['orders'].append({
                    'id': order.id,
                    'pickup': order.start, # key 'pickup'
                    'created_time': order.created_time
                })
        return obs

    def _action_to_pair(self, action_int: int, pending_order_ids: List[int]) -> Tuple[int, int] | None:
        taxi_id = action_int // self.max_orders
        slot = action_int % self.max_orders
        if slot >= len(pending_order_ids):
            return None
        return (taxi_id, pending_order_ids[slot])

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        # Get current pending order IDs
        pending_ids = [o['id'] for o in self._get_observation()['orders']]
        pair = self._action_to_pair(action, pending_ids)
        if pair is None:
            return self._get_observation(), -1.0, False, {"illegal": True}

        taxi_id, order_id = pair
        self.current_step += 1

        # Validate
        if taxi_id >= self.num_taxis:
            return self._get_observation(), -1.0, False, {"illegal": True}
        taxi = self.taxis[taxi_id]
        order = next((o for o in self.orders if o.id == order_id and not o.finished), None)
        if order is None:
            return self._get_observation(), -1.0, False, {"illegal": True}
        if not taxi.is_free:
            return self._get_observation(), -1.0, False, {"illegal": True}
        if order.picked_up or order.finished:
            return self._get_observation(), -1.0, False, {"illegal": True}

        # Simulate trip
        start_pos = taxi.loc
        pickup_pos = order.start
        dropoff_pos = order.end

        time_to_pickup = manhattan_distance(start_pos, pickup_pos)
        trip_distance = manhattan_distance(pickup_pos, dropoff_pos)
        total_travel_time = time_to_pickup + trip_distance

        pickup_time = self.current_step + time_to_pickup
        waiting_time = pickup_time - order.created_time

        self.current_step += total_travel_time

        # Reward: scale down to keep values reasonable
        reward = 20 - (waiting_time + trip_distance) / 10

        taxi.loc = dropoff_pos
        taxi.is_free = True
        taxi.dispatched_order = None

        order.picked_up = True
        order.finished = True

        done = all(o.finished for o in self.orders)
        if self.current_step >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, {"travel_time": total_travel_time}

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for order in self.orders:
            if not order.finished:
                x, y = order.start
                grid[x][y] = "O"
        for taxi in self.taxis:
            x, y = taxi.loc
            grid[x][y] = "T"
        print("\n".join(" ".join(row) for row in grid))

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])