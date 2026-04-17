# nstep_sarsa.py
# n‑step SARSA agent for SMDP (Semi‑Markov Decision Process).
# Each transition has a duration (real time or steps) which affects discounting.

from collections import deque
import random
from q_table import QTable
from epsilon_greedy import epsilon_greedy

class NStepSarsaAgent:
    """
    n‑step SARSA agent for SMDP with epsilon‑greedy exploration.
    Uses a Q‑table (dictionary) to store state‑action values.
    Memory stores (state_tuple, action, reward, duration) for the last n steps.
    """

    def __init__(self, state_encoder, num_actions: int,
                 n: int = 3, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        Args:
            state_encoder: object with encode(raw_state) -> tuple (hashable)
            num_actions: number of possible actions
            n: number of steps to look ahead (n‑step return)
            alpha: learning rate
            gamma: discount factor (applied to cumulative duration)
            epsilon: initial exploration probability
            epsilon_min: minimum epsilon after decay
            epsilon_decay: multiplicative decay factor per episode
        """
        self.state_encoder = state_encoder
        self.num_actions = num_actions
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = QTable()
        # Memory stores (state, action, reward, duration)
        self.memory = deque(maxlen=n)
        self.last_state = None
        self.last_action = None

    # ----------------------------------------------------------------------
    # Core methods
    # ----------------------------------------------------------------------

    def act(self, raw_state, epsilon_override=None):
        """Choose an action using epsilon‑greedy."""
        state_tuple = self.state_encoder.encode(raw_state)
        eps = epsilon_override if epsilon_override is not None else self.epsilon
        action, _ = epsilon_greedy(state_tuple, self.q_table, eps, self.num_actions)
        return action

    def start_episode(self, raw_state):
        """Call at the beginning of each episode."""
        self.memory.clear()
        self.last_state = self.state_encoder.encode(raw_state)
        self.last_action = self.act(raw_state)

    def get_current_action(self):
        """Return the action to take now."""
        return self.last_action

    def step(self, raw_next_state, reward, done, duration=1):
        """
        Call after each environment step.

        Args:
            raw_next_state: next raw state
            reward: reward received for the transition
            done: whether episode ended
            duration: real time (or steps) taken for this transition (default 1)
        """
        next_state_tuple = self.state_encoder.encode(raw_next_state)

        # Store transition with duration
        self.memory.append((self.last_state, self.last_action, reward, duration))

        if done:
            # Terminal: update all remaining steps using actual returns
            self._update_from_memory(terminal=True)
            self.memory.clear()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.last_state = None
            self.last_action = None
            return

        # Not done: choose next action
        next_action = self.act(raw_next_state)

        # Perform n‑step updates if memory has n steps
        self._update_from_memory(terminal=False,
                                 next_state=next_state_tuple,
                                 next_action=next_action)

        # Prepare for next step
        self.last_state = next_state_tuple
        self.last_action = next_action

    # ----------------------------------------------------------------------
    # Internal n‑step SMDP update logic
    # ----------------------------------------------------------------------

    def _update_from_memory(self, terminal, next_state=None, next_action=None):
        """
        Perform n‑step SMDP updates.
        The discount exponent is the cumulative duration.
        """
        if terminal:
            # Update all steps in memory using actual returns (no bootstrap)
            T = len(self.memory)
            for t in range(T):
                # Compute return from step t to the end
                G = 0.0
                cum_time = 0.0
                for i in range(T - t):
                    _, _, r, dur = self.memory[t + i]
                    G += (self.gamma ** cum_time) * r
                    cum_time += dur
                state_t, action_t, _, _ = self.memory[t]
                current_q = self.q_table.get(state_t, action_t)
                td_error = G - current_q
                self.q_table.update(state_t, action_t, self.alpha * td_error)
        else:
            # Non‑terminal: update the oldest step when memory has exactly n steps
            if len(self.memory) == self.n:
                state_0, action_0, _, _ = self.memory[0]

                # Compute n‑step return with cumulative durations
                G = 0.0
                cum_time = 0.0
                for i in range(self.n):
                    _, _, r, dur = self.memory[i]
                    G += (self.gamma ** cum_time) * r
                    cum_time += dur

                # Add bootstrap term if we have next state/action
                if next_state is not None and next_action is not None:
                    G += (self.gamma ** cum_time) * self.q_table.get(next_state, next_action)

                current_q = self.q_table.get(state_0, action_0)
                td_error = G - current_q
                self.q_table.update(state_0, action_0, self.alpha * td_error)

                # Remove the oldest step (sliding window)
                self.memory.popleft()

    # ----------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------

    def save(self, path):
        """Save Q‑table to file."""
        self.q_table.save(path)

    def load(self, path):
        """Load Q‑table from file."""
        self.q_table.load(path)