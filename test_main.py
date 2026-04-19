import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from taxi_envs.env_utils import make_env
from utils import make_run_dir, save_pickle, save_reward_curve, save_json
import json

ROOT_DIR = os.path.dirname(__file__)
ALGO_DIR = os.path.join(ROOT_DIR, "Algorithm")
if ALGO_DIR not in sys.path:
	sys.path.insert(0, ALGO_DIR)

from Algorithm.nstep_sarsa import NStepSarsaAgent
from Algorithm.state_encoder import StateEncoder

from visualization.trajectory import TrajectoryRecorder, save_trajectory

from pathlib import Path

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
	config = {
    "episodes": episodes,
    "max_steps": max_steps,
    "seed": seed,
    "n": n,
    "alpha": alpha,
    "gamma": gamma,
    "epsilon": epsilon,
    "save_path": save_path,
    "load_path": load_path,
}

	env = make_env(max_steps=max_steps, seed=seed)
	num_zones = env.n_zones
	run_dir = make_run_dir("outputs")
	print(f"run directory: {run_dir}")


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
	recorder = TrajectoryRecorder()

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

			recorder.add_step(
				episode=ep,
				step=step,
				state=current_state,
				action=action,
				next_state=obs,
				reward=reward,
				info=_info,
				terminated=terminated,
				truncated=truncated,
			)


			# dt = int(_info.get("time_elapsed", 0))
			dt = int(_info.get("time_elapsed", 0))
			matched = _info.get("matched", False)
			# print(
			# 	f"Ep: {ep} | Step: {step:02d} | State: {current_state} "
			# 	f"-> Action: {action:2d} -> Next: {obs} | Reward: {reward:6.2f} "
			# 	f"| dt: {dt:2d} | matched: {matched}"
			# )
			total_reward += reward
			# agent.step(_as_raw_state(obs), reward, terminated or truncated)
			agent.step(_as_raw_state(obs), reward, terminated or truncated, duration=max(dt, 1))
			step += 1
		# env.render()
		print(
			f"episode={ep} steps={step} total_reward={total_reward:.2f} "
			f"epsilon={agent.epsilon:.3f}"
		)
		reward_history.append((ep, total_reward))


	if reward_history:
		save_reward_curve(reward_history, run_dir / "reward_curve.png")

	# always save config and reward history
	save_json(config, run_dir / "config.json")
	save_pickle(reward_history, run_dir / "reward_history.pkl")

	if save_path:
		model_path = run_dir / Path(save_path).name
	else:
		model_path = run_dir / "nstep_agent.pkl"

	agent.save(str(model_path))

	traj_path = run_dir / f"{model_path.stem}_trajectory.pkl"
	save_trajectory(recorder.get_records(), traj_path)

	print(f"model saved to: {model_path}")
	print(f"trajectory saved to: {traj_path}")
	print(f"config saved to: {run_dir / 'config.json'}")
	print(f"reward history saved to: {run_dir / 'reward_history.pkl'}")
	print(f"reward curve saved to: {run_dir / 'reward_curve.png'}")


		


if __name__ == "__main__":
	run_integration_test(episodes=500, save_path='outputs/nstep_agent.pkl')
