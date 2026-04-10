import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Taxi():
    def __init__(self, id, loc):
        self.id = id
        self.loc = loc
        self.is_free = True
        self.dest = None
        self.dispatched_order = None

    def _move_towards(self, current, target):
        x, y = current
        target_x, target_y = target
        
        if x < target_x:
            x += 1
        elif x > target_x:
            x -= 1
        
        if y < target_y:
            y += 1
        elif y > target_y:
            y -= 1
            
        return [x, y]

class Order():
    def __init__(self, id, start, end):
        self.id = id
        self.start = start
        self.end = end
        self.created_time = 0
        self.picked_up = False
        self.finished = False

class TaxiDispatchEnv(gym.Env):
    def __init__(self, num_taxis=5, grid_size=10, max_steps=200):
        super().__init__()
        self.num_taxis = num_taxis
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        
        # 1. 动作空间: 为每辆车分配一个目标网格的 (x, y) 坐标
        # 维度是 2 * num_taxis。例如5辆车，输出10个离散值 [x1, y1, x2, y2...]
        self.action_space = spaces.MultiDiscrete([grid_size] * (num_taxis * 2))
        
        # 2. 状态空间: 2D 网格的多通道表示 (类似图像)[taxi_pos, order_start, time_waiting]
        self.observation_space = spaces.Box(
            low=0.0, high=255.0, # 假设等待时间最大映射到255
            shape=(3, grid_size, grid_size), 
            dtype=np.float32
        )
        
        # inner state
        self.taxis = []
        self.orders = []
        self.reset()

    def _init_state(self):
        self.taxis = [
            Taxi(
                i,
                [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)],
            )
            for i in range(self.num_taxis)
        ]
        self.orders = [
            Order(
                i,
                [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)],
                [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)],
            )
            for i in range(10)
        ]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self._init_state()
        return self._get_observation()

    def _get_observation(self):
        obs = {}
        obs['taxis'] = []
        obs['orders'] = []
        obs['current_time'] = self.current_step
        for taxi in self.taxis:
            obs['taxis'].append({'id': taxi.id, 'loc': taxi.loc, 'is_free': taxi.is_free})
        for order in self.orders:
            obs['orders'].append({'id': order.id, 'start': order.start, 'end': order.end, 'created_time': order.created_time})
        return obs
    
    def step(self, action:tuple):
        self.current_step += 1
        reward = 0.0
        taxi_id, order_id = action

        # 1. 验证动作合法性
        if taxi_id >= self.num_taxis or order_id >= len(self.orders):
            return self._get_observation(), -10.0, False
        
        taxi = self.taxis[taxi_id]
        order = self.orders[order_id]
        if not taxi.is_free or order.created_time > 0:
            return self._get_observation(), -5.0, False
        
        # 2. dispatch taxi to order
        taxi.is_free = False
        taxi.dest = order.start
        taxi.dispatched_order = order

        # 3. move taxi towards order start
        for taxi in self.taxis:
            if not taxi.is_free:
                taxi.loc = taxi._move_towards(taxi.loc, taxi.dest)
                if taxi.loc == taxi.dest:
                    if taxi.dispatched_order and taxi.dispatched_order.start == taxi.dest:
                        # pick up passenger
                        order = taxi.dispatched_order
                        order.picked_up = True
                        taxi.dest = taxi.dispatched_order.end
                    elif taxi.dispatched_order and taxi.dispatched_order.end == taxi.dest:
                        # drop off passenger
                        reward += 20.0
                        taxi.is_free = True
                        taxi.dispatched_order = None
        
        # 4. update created time for orders
        for order in self.orders:
            if not order.picked_up:
                waiting_time = self.current_step - order.created_time
                if waiting_time > 10:  # 超过10步未被接单，给予负奖励
                    reward -= 1.0 * (waiting_time - 10)

        # 5. check if episode is done
        done = True
        for order in self.orders:
            if not order.finished:
                done = False
                break
        if self.current_step >= self.max_steps:
            done = True
            
        return self._get_observation(), reward, done

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for order in self.orders:
            if order.picked_up or order.finished:
                continue
            x, y = order.start
            grid[x][y] = "O"

        for taxi in self.taxis:
            x, y = taxi.loc
            grid[x][y] = "T"

        print("\n".join(" ".join(row) for row in grid))