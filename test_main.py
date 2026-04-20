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
from baseline import RandomDispatchAgent
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

    run_dir = make_run_dir("outputs")
    print(f"run directory: {run_dir}")

    env = make_env(max_steps=max_steps, seed=seed)
    num_zones = env.n_zones

    def _as_raw_state(obs):
        zone, current_time = obs
        return {"zone": int(zone), "current_time": int(current_time)}

    # ---------------------------
    # trained agent
    # ---------------------------
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

    trained_reward_history = []
    trained_recorder = TrajectoryRecorder()

    for ep in range(episodes):
        episode_seed = seed + ep if seed is not None else None
        obs, _info = env.reset(seed=episode_seed)

        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        agent.start_episode(_as_raw_state(obs))

        while not (terminated or truncated) and step < max_steps:
            current_state = obs
            action = agent.get_current_action()

            obs, reward, terminated, truncated, _info = env.step(action)

            # record current demand snapshot
            _info = dict(_info)
            _info["pending_counts"] = [len(order_list) for order_list in env.pending_orders]

            trained_recorder.add_step(
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

            dt = int(_info.get("time_elapsed", 1))
            total_reward += reward
            agent.step(
                _as_raw_state(obs),
                reward,
                terminated or truncated,
                duration=max(dt, 1),
            )
            step += 1

        print(
            f"[trained] episode={ep} steps={step} total_reward={total_reward:.2f} "
            f"epsilon={agent.epsilon:.3f}"
        )
        trained_reward_history.append((ep, total_reward))

    # ---------------------------
    # random agent
    # ---------------------------
    random_agent = RandomDispatchAgent()
    random_reward_history = []
    random_recorder = TrajectoryRecorder()

    for ep in range(episodes):
        episode_seed = seed + ep if seed is not None else None
        obs, _info = env.reset(seed=episode_seed)

        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and step < max_steps:
            current_state = obs
            action = random_agent.act(env)

            obs, reward, terminated, truncated, _info = env.step(action)
            # record current demand snapshot
            _info = dict(_info)
            _info["pending_counts"] = [len(order_list) for order_list in env.pending_orders]


            random_recorder.add_step(
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

            total_reward += reward
            step += 1

        print(f"[random]  episode={ep} steps={step} total_reward={total_reward:.2f}")
        random_reward_history.append((ep, total_reward))

    # ---------------------------
    # save artifacts
    # ---------------------------
    save_json(config, run_dir / "config.json")

    save_pickle(trained_reward_history, run_dir / "trained_reward_history.pkl")
    save_pickle(random_reward_history, run_dir / "random_reward_history.pkl")

    if trained_reward_history:
        save_reward_curve(trained_reward_history, run_dir / "trained_reward_curve.png")
    if random_reward_history:
        save_reward_curve(random_reward_history, run_dir / "random_reward_curve.png")

    if save_path:
        model_path = run_dir / Path(save_path).name
    else:
        model_path = run_dir / "nstep_agent.pkl"

    agent.save(str(model_path))

    trained_traj_path = run_dir / "trained_trajectory.pkl"
    random_traj_path = run_dir / "random_trajectory.pkl"

    save_trajectory(trained_recorder.get_records(), trained_traj_path)
    save_trajectory(random_recorder.get_records(), random_traj_path)

    print(f"model saved to: {model_path}")
    print(f"trained trajectory saved to: {trained_traj_path}")
    print(f"random trajectory saved to: {random_traj_path}")
    print(f"config saved to: {run_dir / 'config.json'}")
    print(f"trained reward history saved to: {run_dir / 'trained_reward_history.pkl'}")
    print(f"random reward history saved to: {run_dir / 'random_reward_history.pkl'}")
    print(f"trained reward curve saved to: {run_dir / 'trained_reward_curve.png'}")
    print(f"random reward curve saved to: {run_dir / 'random_reward_curve.png'}")

    from visualization.grid_replay import run_grid_compare
    from visualization.grid_animation import run_grid_animation_compare

    run_grid_compare(
        outputs_dir=run_dir,
        episode_idx=60
    )

    run_grid_animation_compare(
        outputs_dir=run_dir,
        episode_idx=60,
        fps=3,
        interval_ms=300,
        save_gif=True,
        save_mp4=False, # 支持视频保存
        show_plot=False,
    )


if __name__ == "__main__":
    run_integration_test(
        episodes=100,
        max_steps=200,
        seed=42,
        save_path="nstep_agent.pkl",
    )
