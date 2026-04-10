import numpy as np

from taxi_envs.env_utils import make_env, sample_dispatch


def run_random_agent(episodes: int = 10, max_steps: int = 200, seed: int = 42) -> None:
	rng = np.random.default_rng(seed)
	env = make_env(max_steps=max_steps, seed=seed)

	for ep in range(episodes):
		obs = env._get_observation()
		total_reward = 0.0
		done = False
		step = 0

		while not done and step < max_steps:
			action = sample_dispatch(env, rng=rng)
			obs, reward, done = env.step(action)
			total_reward += reward
			step += 1
		env.render()
		env.reset(seed)

		print(f"episode={ep} steps={step} total_reward={total_reward:.2f}")
		


if __name__ == "__main__":
	run_random_agent()
