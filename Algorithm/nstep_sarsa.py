from collections import deque
import random
from q_table import QTable
from epsilon_greedy import epsilon_greedy


class NStepSarsaAgent:
    """
    n-step SARSA agent with epsilon-greedy exploration.
    Uses a Q-table (dictionary) to store state-action values.
    The agent internally tracks the last state and last action,
    and provides methods to start an episode, get the current action,
    step through the environment, and convert actions to environment format.
    """

    def __init__(self, state_encoder, num_taxis, max_orders, n=3, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Args:
            state_encoder: object with encode(raw_state) -> tuple (hashable state representation)
            num_taxis: fixed number of taxis in the environment
            max_orders: maximum number of orders that can exist at once (used for action mapping)
            n: number of steps to look ahead for n-step return
            alpha: learning rate (step size)
            gamma: discount factor for future rewards
            epsilon: initial exploration probability
            epsilon_min: minimum epsilon after decay
            epsilon_decay: multiplicative decay factor per episode
        """
        self.state_encoder = state_encoder
        self.num_taxis = num_taxis
        self.max_orders = max_orders
        self.num_actions = num_taxis * max_orders  # total possible action indices
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table stores values for (state_tuple, action) pairs
        self.q_table = QTable()

        # Memory buffer for the current episode: stores (state_tuple, action, reward)
        # Maximum length n ensures we never store more than n steps.
        self.memory = deque(maxlen=n)

        # Internal state for the current step: last state and last action taken.
        self.last_state = None
        self.last_action = None

    # --------------------------------------------------------------------------
    # Core agent methods
    # --------------------------------------------------------------------------

    def act(self, raw_state, epsilon_override=None):
        """
        Choose an action (integer index) for the given raw state using epsilon-greedy.
        Args:
            raw_state: the raw environment state (dictionary)
            epsilon_override: if provided, use this epsilon instead of self.epsilon
        Returns:
            integer action index (0 .. num_actions-1)
        """
        # Convert raw state to a hashable tuple
        state_tuple = self.state_encoder.encode(raw_state)
        # Decide which epsilon to use
        eps = epsilon_override if epsilon_override is not None else self.epsilon
        # Call epsilon_greedy (returns action and its Q-value; we ignore the Q-value)
        action, _ = epsilon_greedy(state_tuple, self.q_table, eps, self.num_actions)
        return action

    def start_episode(self, raw_state):
        """
        Call this once at the beginning of each episode.
        Resets memory, encodes the initial state, and chooses the first action.
        Args:
            raw_state: initial raw state from env.reset()
        """
        self.memory.clear()
        self.last_state = self.state_encoder.encode(raw_state)
        self.last_action = self.act(raw_state)  # choose first action

    def get_current_action(self):
        """
        Returns the action (integer index) that should be taken next.
        The agent maintains this action internally after start_episode() and after each step().
        Returns:
            integer action or None if episode not started or already ended.
        """
        return self.last_action

    def to_env_action(self, action_int, pending_order_ids):
        """
        Convert an integer action index to the format expected by the environment.
        The environment expects a tuple (taxi_id, order_id) for a single dispatch.
        Mapping: action_int = taxi_id * max_orders + order_slot
        where order_slot indexes into the list of pending order IDs.
        If the order_slot is out of range (no order at that slot), returns None
        (the environment should then assign a large negative penalty).
        Args:
            action_int: integer action index from the agent
            pending_order_ids: list of order IDs that are still unassigned (in the same order as the state's orders list)
        Returns:
            tuple (taxi_id, order_id) or None if illegal action
        """
        taxi_id = action_int // self.max_orders
        order_slot = action_int % self.max_orders
        if order_slot >= len(pending_order_ids):
            return None  # illegal action: no order at that slot
        order_id = pending_order_ids[order_slot]
        return (taxi_id, order_id)

    def step(self, raw_next_state, reward, done):
        """
        Call this after each environment step, providing the outcome of the last action.
        The agent will:
            - store the completed transition (last_state, last_action, reward)
            - update Q-values using n-step returns (if enough steps are collected or episode ends)
            - choose the next action (if not done) and store it internally
        Args:
            raw_next_state: the state after executing last_action
            reward: reward received for that transition
            done: whether the episode ended after this step
        """
        # Encode the new state (the state after the action)
        next_state_tuple = self.state_encoder.encode(raw_next_state)

        # Store the transition that just completed
        self.memory.append((self.last_state, self.last_action, reward))

        if done:
            # Episode finished: update all remaining steps in memory using only actual rewards
            self._update_from_memory(terminal=True)
            self.memory.clear()
            # Decay epsilon for the next episode
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            # Reset internal state for the next episode
            self.last_state = None
            self.last_action = None
            return

        # Not done: choose the next action using the current policy (same epsilon)
        next_action = self.act(raw_next_state)

        # Perform n-step updates for the oldest step in memory if we have n steps
        self._update_from_memory(terminal=False,
                                 next_state=next_state_tuple,
                                 next_action=next_action)

        # Update internal state for the next call to step()
        self.last_state = next_state_tuple
        self.last_action = next_action

    # --------------------------------------------------------------------------
    # Internal n-step update logic
    # --------------------------------------------------------------------------

    def _update_from_memory(self, terminal, next_state=None, next_action=None):
        """
        Perform n-step SARSA updates based on the current memory content.
        This is an internal helper called by step().
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
                # No bootstrap because episode ended
                state_t, action_t, _ = self.memory[t]
                current_q = self.q_table.get(state_t, action_t)
                td_error = G - current_q
                self.q_table.update(state_t, action_t, self.alpha * td_error)
        else:
            # Case 2: Non-terminal. Only update the oldest step when memory length == n.
            if len(self.memory) == self.n:
                state_0, action_0, _ = self.memory[0]

                # Compute the n-step return: sum of discounted rewards from step 0 to n-1
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

                # Remove the oldest step from memory (sliding window)
                self.memory.popleft()

    # --------------------------------------------------------------------------
    # Persistence
    # --------------------------------------------------------------------------

    def save(self, path):
        """Save the Q-table to a file using pickle."""
        self.q_table.save(path)

    def load(self, path):
        """Load the Q-table from a file."""
        self.q_table.load(path)