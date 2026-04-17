import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from taxi_envs.env_utils import make_env


ROOT_DIR = os.path.dirname(__file__)
ALGO_DIR = os.path.join(ROOT_DIR, "Algorithm")
if ALGO_DIR not in sys.path:
	sys.path.insert(0, ALGO_DIR)

from nstep_sarsa import NStepSarsaAgent
from state_encoder import StateEncoder


def run_integration_test(
	episodes=3000,
	max_steps=200,
	seed=42,
	n=3,
	alpha=0.1,
	gamma=0.95,
	epsilon=0.2,
	save_path=None,
	load_path=None,
):
	env = make_env(max_steps=max_steps, seed=seed)
	num_zones = env.n_zones

	def _as_raw_state(obs):
		zone, current_time = obs
		return {"zone": int(zone), "current_time": int(current_time)}

	encoder = StateEncoder(num_zones=num_zones, max_time_steps=env.episode_length)
	agent = NStepSarsaAgent(
		state_encoder=encoder,
		num_actions=num_zones,
		n=n,
		alpha=alpha,
		gamma=gamma,
		epsilon=epsilon,
	)
	if load_path:
		agent.load(load_path)
	reward_history = []

	for ep in range(episodes):
		episode_seed = seed + ep if seed is not None else None
		obs, _info = env.reset(seed=seed)
		total_reward = 0.0
		step = 0
		terminated = False
		truncated = False

		agent.start_episode(_as_raw_state(obs))

		while not (terminated or truncated) and step < max_steps:
			current_state = obs
			action = agent.get_current_action()
			obs, reward, terminated, truncated, _info = env.step(action)
			dt = int(_info.get("time_elapsed", 0))
			matched = _info.get("matched", False)
			# print(
			# 	f"Ep: {ep} | Step: {step:02d} | State: {current_state} "
			# 	f"-> Action: {action:2d} -> Next: {obs} | Reward: {reward:6.2f} "
			# 	f"| dt: {dt:2d} | matched: {matched}"
			# )
			total_reward += reward
			agent.step(_as_raw_state(obs), reward, terminated or truncated)
			step += 1
		# env.render()
		print(
			f"episode={ep} steps={step} total_reward={total_reward:.2f} "
			f"epsilon={agent.epsilon:.3f}"
		)
		reward_history.append((ep, total_reward))


	if reward_history:
		episodes_list = [row[0] for row in reward_history]
		rewards = [row[1] for row in reward_history]
		plt.figure(figsize=(8, 4))
		plt.plot(episodes_list, rewards, linewidth=1.5)
		plt.title("Training Reward per Episode")
		plt.xlabel("Episode")
		plt.ylabel("Total Reward")
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.show()

	if save_path:
		agent.save(save_path)
	
	# if agent.q_table.table:
	# 	print("\nQ Table (state, action -> Q)")
	# 	for (state_tuple, action), q_value in agent.q_table.table.items():
	# 		print(f"state={state_tuple} action={action} Q={q_value:.4f}")


if __name__ == "__main__":
	run_integration_test()
