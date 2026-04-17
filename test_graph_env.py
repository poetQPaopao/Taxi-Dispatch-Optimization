import numpy as np

from taxi_envs.env_utils import make_graph_env, sample_dispatch


def run_random_agent(episodes: int = 3, max_steps: int = 50, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    env = make_graph_env(max_steps=max_steps, seed=seed)

    for ep in range(episodes):
        obs, _info = env.reset(seed=seed)
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            action = sample_dispatch(env, rng=rng)
            obs, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

        print(f"episode={ep} steps={step} total_reward={total_reward:.2f}")


if __name__ == "__main__":
    run_random_agent()
