import os
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from taxi_envs.env_utils import make_graph_env

ROOT_DIR = os.path.dirname(__file__)
ALGO_DIR = os.path.join(ROOT_DIR, "Algorithm")
if ALGO_DIR not in sys.path:
    sys.path.insert(0, ALGO_DIR)

from nstep_sarsa import NStepSarsaAgent
from state_encoder import StateEncoder


@dataclass
class RunResult:
    name: str
    rewards: list[float]


def _as_raw_state(obs: tuple[int, int]) -> dict:
    zone, current_time = obs
    return {"zone": int(zone), "current_time": int(current_time)}


def run_random_agent(
    episodes: int,
    max_steps: int,
    seed: int,
    env_kwargs: dict,
) -> RunResult:
    rng = np.random.default_rng(seed)
    env = make_graph_env(max_steps=max_steps, seed=seed, **env_kwargs)
    rewards = []

    for ep in range(episodes):
        episode_seed = seed + ep if seed is not None else None
        obs, _info = env.reset(seed=episode_seed)
        total_reward = 0.0
        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated) and step < max_steps:
            action = int(rng.integers(0, env.action_space.n))
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            step += 1

        rewards.append(total_reward)

    return RunResult(name="random", rewards=rewards)


def run_sarsa_agent(
    name: str,
    episodes: int,
    max_steps: int,
    seed: int,
    use_smdp: bool,
    env_kwargs: dict,
    n: int = 3,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.1,
) -> RunResult:
    env = make_graph_env(max_steps=max_steps, seed=seed, **env_kwargs)
    encoder = StateEncoder(num_zones=env.n_zones, max_time_steps=env.episode_length)
    agent = NStepSarsaAgent(
        state_encoder=encoder,
        num_actions=env.n_zones,
        n=n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )

    rewards = []
    for ep in range(episodes):
        episode_seed = seed + ep if seed is not None else None
        obs, _info = env.reset(seed=episode_seed)
        total_reward = 0.0
        terminated = False
        truncated = False
        step = 0

        agent.start_episode(_as_raw_state(obs))

        while not (terminated or truncated) and step < max_steps:
            action = agent.get_current_action()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if use_smdp:
                duration = int(info.get("time_elapsed", 1))
            else:
                duration = 1
            agent.step(_as_raw_state(obs), reward, terminated or truncated, duration)
            step += 1

        rewards.append(total_reward)

    return RunResult(name=name, rewards=rewards)


def summarize(results: list[RunResult]) -> None:
    print("\nSummary (mean ± std)")
    for result in results:
        arr = np.asarray(result.rewards, dtype=float)
        print(f"{result.name:10s}: {arr.mean():8.2f} ± {arr.std():6.2f}")


def plot_avg_rewards(results: list[RunResult]) -> None:
    names = [r.name for r in results]
    means = [float(np.mean(r.rewards)) for r in results]

    plt.figure(figsize=(6, 4))
    plt.bar(names, means, color="#4C78A8")
    plt.title("Average Reward by Agent")
    plt.xlabel("Agent")
    plt.ylabel("Average Reward")
    plt.tight_layout()
    plt.show()


def plot_learning_curves(results: list[RunResult], window: int = 5) -> None:
    plt.figure(figsize=(8, 4))
    for result in results:
        rewards = np.asarray(result.rewards, dtype=float)
        if window > 1:
            kernel = np.ones(window) / window
            smoothed = np.convolve(rewards, kernel, mode="valid")
            x = np.arange(len(smoothed)) + window - 1
            plt.plot(x, smoothed, label=f"{result.name} (ma{window})")
        else:
            plt.plot(rewards, label=result.name)

    plt.title("Learning Curve (Episode Reward)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_compare(
    episodes: int = 10000,
    max_steps: int = 96,
    seed: int = 42,
) -> None:
    env_kwargs: dict = {}

    results = []
    results.append(run_random_agent(episodes, max_steps, seed, env_kwargs))
    results.append(
        run_sarsa_agent(
            name="mdp_sarsa",
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            use_smdp=False,
            env_kwargs=env_kwargs,
        )
    )
    results.append(
        run_sarsa_agent(
            name="smdp_sarsa",
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            use_smdp=True,
            env_kwargs=env_kwargs,
        )
    )

    summarize(results)
    plot_avg_rewards(results)
    plot_learning_curves(results)


if __name__ == "__main__":
    run_compare()
