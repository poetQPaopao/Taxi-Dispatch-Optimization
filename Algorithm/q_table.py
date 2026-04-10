class QTable:
    def __init__(self):
        self.table = {}  # key: (state_tuple, action) -> value

    def get(self, state_tuple, action):
        """Return Q(s,a), default 0.0"""
        return self.table.get((state_tuple, action), 0.0)

    def update(self, state_tuple, action, delta):
        """Q(s,a) += delta (where delta = alpha * td_error)"""
        key = (state_tuple, action)
        self.table[key] = self.get(state_tuple, action) + delta

    def get_all_for_state(self, state_tuple, num_actions):
        """Return list of Q-values for all actions in this state."""
        return [self.get(state_tuple, a) for a in range(num_actions)]

    def save(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.table, f)

    def load(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            self.table = pickle.load(f)
