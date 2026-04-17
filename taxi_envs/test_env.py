import os
import sys

import numpy as np

if __package__:
	from .env_utils import make_env, sample_dispatch
else:
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	if root not in sys.path:
		sys.path.insert(0, root)
	from taxi_envs.env_utils import make_env, sample_dispatch


def run_random_agent(episodes: int = 3, max_steps: int = 200, seed: int = 42) -> None:
	rng = np.random.default_rng(seed)
	env = make_env(max_steps=max_steps, seed=seed)
	for ep in range(episodes):
		obs, _info = env.reset(seed=seed)
		total_reward = 0.0
		done = False
		step = 0
		print(env.pending_orders)
		while not done and step < max_steps:
			current_state = obs
			action = sample_dispatch(env, rng=rng)
			obs, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			matched = info.get("matched", False)
			dt = info.get("time_elapsed", 0)
			print(
				f"Ep: {ep} | Step: {step:02d} | State: {current_state} "
				f"-> Action: {action:2d} -> Next: {obs} | Reward: {reward:6.2f} "
				f"| dt: {dt:2d} | matched: {matched}"
			)
			total_reward += reward
			step += 1
		print(f"episode={ep} steps={step} total_reward={total_reward:.2f}")
		
		


if __name__ == "__main__":
	run_random_agent()
