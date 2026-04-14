from Experiment.adapter import adapt_obs, get_pending_order_ids


def run_one_episode(env, agent, use_learning_agent=True, episode_idx=0):
    obs = env.reset()
    total_reward = 0.0
    step = 0
    illegal_actions = 0
    valid_actions = 0
    step_logs = []

    if use_learning_agent:
        agent_obs = adapt_obs(obs, env.grid_size)
        agent.start_episode(agent_obs)

    done = False
    while not done and step < env.max_steps:
        if use_learning_agent:
            action_idx = agent.get_current_action()
            pending_order_ids = get_pending_order_ids(obs)
            env_action = agent.to_env_action(action_idx, pending_order_ids)

            if env_action is None:
                reward = -10.0
                next_obs = obs
                done = False
                info = {"illegal": True} 
                illegal_actions += 1
                action_repr = {
                    "action_idx": action_idx,
                    "env_action": None,
                    "is_illegal": True,
                }
            else:
                next_obs, reward, done, info = env.step(env_action)
                valid_actions += 1
                action_repr = {
                    "action_idx": action_idx,
                    "env_action": list(env_action),
                    "is_illegal": False,
                }

            next_agent_obs = adapt_obs(next_obs, env.grid_size)
            agent.step(next_agent_obs, reward, done)

        else:
            env_action = agent.act(env)
            next_obs, reward, done, info = env.step(env_action)
            valid_actions += 1
            action_repr = {
                "action_idx": None,
                "env_action": list(env_action) if env_action is not None else None,
                "is_illegal": False,
            }

        step_logs.append({
            "episode": episode_idx,
            "step": step,
            "reward": reward,
            "done": done,
            "current_time": getattr(env, "current_step", step),
            "num_pending_orders": len(get_pending_order_ids(next_obs)),
            "travel_time": info.get("travel_time", None),
            "illegal": info.get("illegal", False),
            **action_repr,
        })

        obs = next_obs
        total_reward += reward
        step += 1

    completed_orders = sum(1 for o in env.orders if getattr(o, "finished", False))
    total_orders = len(env.orders)

    episode_summary = {
        "episode": episode_idx,
        "agent_name": getattr(agent, "name", agent.__class__.__name__),
        "episode_reward": total_reward,
        "steps": step,
        "illegal_actions": illegal_actions,
        "valid_actions": valid_actions,
        "completed_orders": completed_orders,
        "total_orders": total_orders,
    }

    if not done:
        print("[WARNING] Episode truncated by env.max_steps. Env may not terminate naturally.")

    return episode_summary, step_logs


def evaluate(env, agent, episodes=5, use_learning_agent=True):
    results = []
    all_step_logs = []

    for ep in range(episodes):
        summary, step_logs = run_one_episode(
            env=env,
            agent=agent,
            use_learning_agent=use_learning_agent,
            episode_idx=ep,
        )
        results.append(summary)
        all_step_logs.extend(step_logs)

    return results, all_step_logs