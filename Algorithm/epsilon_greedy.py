# epsilon_greedy.py
import random
import numpy as np

def epsilon_greedy(state_tuple, q_table, epsilon, num_actions):
    """Return action index, and also the chosen action's Q-value (for logging)"""
    if random.random() < epsilon:
        action = random.randrange(num_actions)
        q_value = q_table.get(state_tuple, action)
    else:
        q_vals = q_table.get_all_for_state(state_tuple, num_actions)
        # break ties randomly
        max_q = max(q_vals)
        best_actions = [i for i, q in enumerate(q_vals) if q == max_q]
        action = random.choice(best_actions)
        q_value = max_q
    return action, q_value