import numpy as np

if __package__:
	from .env_utils import make_env, sample_dispatch
else:
	from taxi_envs.env_utils import make_env, sample_dispatch


def run_random_agent(episodes: int = 3, max_steps: int = 50, seed: int = 0) -> None:
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

		print(f"episode={ep} steps={step} total_reward={total_reward:.2f}")


if __name__ == "__main__":
	run_random_agent()
