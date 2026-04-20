from taxi_envs.env_utils import sample_dispatch

class RandomDispatchAgent:
    def act(self, env):
        return sample_dispatch(env)