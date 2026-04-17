# nstep_sarsa.py
# n‑step SARSA agent for a single‑taxi repositioning problem.
# State = (zone, time), Action = target zone (0 … num_zones-1).

from collections import deque
import random
from q_table import QTable
from epsilon_greedy import epsilon_greedy

class NStepSarsaAgent:
    """
    n‑step SARSA agent with epsilon‑greedy exploration.
    Uses a Q‑table (dictionary) to store state‑action values.
    State is encoded as a tuple (zone, time); action is an integer (target zone).
    """

    def __init__(self, state_encoder, num_actions: int,
                 n: int = 3, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        Args:
            state_encoder: object with encode(raw_state) -> tuple (hashable)
            num_actions: number of possible actions (equals number of zones)
            n: number of steps to look ahead (n‑step return)
            alpha: learning rate (step size)
            gamma: discount factor
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

        # Q‑table stores values for (state_tuple, action)
        self.q_table = QTable()

        # Memory buffer for the current episode:
        # stores (state_tuple, action, reward) for the last n steps.
        self.memory = deque(maxlen=n)

        # Internal state for the current step
        self.last_state = None
        self.last_action = None

    # ----------------------------------------------------------------------
    # Core methods
    # ----------------------------------------------------------------------

    def act(self, raw_state, epsilon_override=None):
        """
        Choose an action (integer target zone) using epsilon‑greedy.

        Args:
            raw_state: raw environment state (dictionary with 'zone', 'current_time')
            epsilon_override: if provided, use this epsilon instead of self.epsilon

        Returns:
            integer action (0 … num_actions-1)
        """
        state_tuple = self.state_encoder.encode(raw_state)
        eps = epsilon_override if epsilon_override is not None else self.epsilon
        action, _ = epsilon_greedy(state_tuple, self.q_table, eps, self.num_actions)
        return action

    def start_episode(self, raw_state):
        """
        Call this at the beginning of each episode.
        Resets memory, encodes the initial state, and chooses the first action.

        Args:
            raw_state: initial raw state from env.reset()
        """
        self.memory.clear()
        self.last_state = self.state_encoder.encode(raw_state)
        self.last_action = self.act(raw_state)

    def get_current_action(self):
        """
        Returns the action that should be taken next.
        """
        return self.last_action

    def step(self, raw_next_state, reward, done):
        """
        Call this after each environment step, providing the outcome of the last action.

        Args:
            raw_next_state: state after executing last_action
            reward: reward received for that transition
            done: whether the episode ended after this step
        """
        next_state_tuple = self.state_encoder.encode(raw_next_state)

        # Store the completed transition
        self.memory.append((self.last_state, self.last_action, reward))

        if done:
            # Episode finished: update all remaining steps using actual returns
            self._update_from_memory(terminal=True)
            self.memory.clear()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.last_state = None
            self.last_action = None
            return

        # Not done: choose next action
        next_action = self.act(raw_next_state)

        # Perform n‑step updates for the oldest step if memory has n steps
        self._update_from_memory(terminal=False,
                                 next_state=next_state_tuple,
                                 next_action=next_action)

        # Prepare for next step
        self.last_state = next_state_tuple
        self.last_action = next_action

    # ----------------------------------------------------------------------
    # Internal n‑step update logic
    # ----------------------------------------------------------------------

    def _update_from_memory(self, terminal, next_state=None, next_action=None):
        """
        Perform n‑step SARSA updates based on the current memory content.

        Args:
            terminal: True if episode ended (no bootstrap), False otherwise.
            next_state: the state after the last step in memory (used for bootstrap)
            next_action: the action taken in next_state (used for bootstrap)
        """
        if terminal:
            # Case 1: Episode ended. Update all steps in memory using actual returns.
            T = len(self.memory)
            for t in range(T):
                # Compute return from step t to the end of the episode
                G = 0.0
                for i in range(T - t):
                    G += (self.gamma ** i) * self.memory[t + i][2]  # reward
                state_t, action_t, _ = self.memory[t]
                current_q = self.q_table.get(state_t, action_t)
                td_error = G - current_q
                self.q_table.update(state_t, action_t, self.alpha * td_error)
        else:
            # Case 2: Non‑terminal. Only update the oldest step when memory length == n.
            if len(self.memory) == self.n:
                state_0, action_0, _ = self.memory[0]

                # Compute the n‑step return: sum of discounted rewards from step 0 to n-1
                G = 0.0
                for i in range(self.n):
                    G += (self.gamma ** i) * self.memory[i][2]

                # Add bootstrap term: gamma^n * Q(next_state, next_action)
                if next_state is not None and next_action is not None:
                    G += (self.gamma ** self.n) * self.q_table.get(next_state, next_action)

                # Update Q(s0, a0)
                current_q = self.q_table.get(state_0, action_0)
                td_error = G - current_q
                self.q_table.update(state_0, action_0, self.alpha * td_error)

                # Remove the oldest step (sliding window)
                self.memory.popleft()

    # ----------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------

    def save(self, path):
        """Save the Q‑table to a file using pickle."""
        self.q_table.save(path)

    def load(self, path):
        """Load the Q‑table from a file."""
        self.q_table.load(path)