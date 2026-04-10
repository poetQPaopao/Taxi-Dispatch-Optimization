import os
import sys

import numpy as np

from taxi_envs.env_utils import make_env

ROOT_DIR = os.path.dirname(__file__)
ALGO_DIR = os.path.join(ROOT_DIR, "Algorithm")
if ALGO_DIR not in sys.path:
	sys.path.insert(0, ALGO_DIR)

from nstep_sarsa import NStepSarsaAgent
from state_encoder import StateEncoder


def _flatten_loc(loc, grid_size):
	return int(loc[0]) * grid_size + int(loc[1])


def _adapt_observation(obs, grid_size):
	"""Adapt env observation keys to Algorithm.state_encoder expectations."""
	adapted = {
		"taxis": [],
		"orders": [],
		"current_time": obs["current_time"],
	}
	for taxi in obs["taxis"]:
		adapted["taxis"].append(
			{
				"id": taxi["id"],
				"location": _flatten_loc(taxi["loc"], grid_size),
				"is_free": taxi["is_free"],
			}
		)
	for order in obs["orders"]:
		adapted["orders"].append(
			{
				"id": order["id"],
				"pickup": _flatten_loc(order["start"], grid_size),
				"created_time": order["created_time"],
			}
		)
	return adapted


def run_integration_test(
	episodes=3,
	max_steps=200,
	seed=0,
	n=3,
	alpha=0.1,
	gamma=0.95,
	epsilon=0.2,
):
	env = make_env(max_steps=max_steps, seed=seed)
	num_orders = len(env.orders)
	num_zones = env.grid_size * env.grid_size

	encoder = StateEncoder(num_zones=num_zones)
	agent = NStepSarsaAgent(
		state_encoder=encoder,
		num_taxis=env.num_taxis,
		max_orders=num_orders,
		n=n,
		alpha=alpha,
		gamma=gamma,
		epsilon=epsilon,
	)

	for ep in range(episodes):
		obs = env.reset()
		total_reward = 0.0
		step = 0
		done = False

		adapted = _adapt_observation(obs, env.grid_size)
		agent.start_episode(adapted)

		while not done and step < max_steps:
			adapted = _adapt_observation(obs, env.grid_size)
			action_index = agent.get_current_action()
			pending_order_ids = [order["id"] for order in adapted["orders"]]
			action = agent.to_env_action(action_index, pending_order_ids)
			if action is None:
				action = (0, num_orders)

			obs, reward, done = env.step(action)
			total_reward += reward
			next_adapted = _adapt_observation(obs, env.grid_size)
			agent.step(next_adapted, reward, done)
			step += 1
		env.render()
		print(f"episode={ep} steps={step} total_reward={total_reward:.2f}")


if __name__ == "__main__":
	run_integration_test()
