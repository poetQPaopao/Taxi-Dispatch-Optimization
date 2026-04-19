import numpy as np

from taxi_envs.env_utils import make_graph_env


def run_random_agent(episodes: int = 3, max_steps: int = 2000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    env = make_graph_env(max_steps=max_steps, seed=seed)
    print(f"graph zones={env.n_zones}")

    for ep in range(episodes):
        episode_seed = seed + ep if seed is not None else None
        obs, _info = env.reset(seed=episode_seed)
        total_reward = 0.0
        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated) and step < max_steps:
            current_state = obs
            action = int(rng.integers(0, env.action_space.n))
            obs, reward, terminated, truncated, _info = env.step(action)
            matched = _info.get("matched", False)
            dt = _info.get("time_elapsed", 0)
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
