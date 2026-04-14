


def get_num_episodes(results):
    return len(results)


def get_mean_reward(results):
    n = len(results)
    if n == 0:
        return 0.0
    return sum(r["episode_reward"] for r in results) / n


def get_mean_steps(results):
    n = len(results)
    if n == 0:
        return 0.0
    return sum(r["steps"] for r in results) / n


def get_mean_illegal_actions(results):
    n = len(results)
    if n == 0:
        return 0.0
    return sum(r["illegal_actions"] for r in results) / n


def get_mean_valid_actions(results):
    n = len(results)
    if n == 0:
        return 0.0
    return sum(r["valid_actions"] for r in results) / n


def get_mean_completed_orders(results):
    n = len(results)
    if n == 0:
        return 0.0
    return sum(r["completed_orders"] for r in results) / n


def get_mean_total_orders(results):
    n = len(results)
    if n == 0:
        return 0.0
    return sum(r["total_orders"] for r in results) / n


def get_completion_rate(results):
    total_completed = sum(r["completed_orders"] for r in results)
    total_orders = sum(r["total_orders"] for r in results)
    if total_orders == 0:
        return 0.0
    return total_completed / total_orders


def summarize_results(results):
    return {
        "num_episodes": get_num_episodes(results),
        "mean_reward": get_mean_reward(results),
        "mean_steps": get_mean_steps(results),
        "mean_illegal_actions": get_mean_illegal_actions(results),
        "mean_valid_actions": get_mean_valid_actions(results),
        "mean_completed_orders": get_mean_completed_orders(results),
        "mean_total_orders": get_mean_total_orders(results),
        "completion_rate": get_completion_rate(results),
    }