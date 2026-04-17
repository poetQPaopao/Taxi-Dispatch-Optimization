# state_encoder.py
# Encodes the raw environment state into a hashable tuple for the Q‑table.

class StateEncoder:
    """
    Encodes a raw state dictionary (zone, current_time) into a tuple (zone, time).
    The state space size is num_zones * max_time_steps.
    """
    def __init__(self, num_zones: int, max_time_steps: int):
        """
        Args:
            num_zones: number of zones (e.g., 25 for a 5x5 grid)
            max_time_steps: maximum time steps in an episode (e.g., 96)
        """
        self.num_zones = num_zones
        self.max_time_steps = max_time_steps

    def encode(self, raw_state: dict) -> tuple:
        """
        Convert raw_state dictionary into a tuple (zone, current_time).

        Args:
            raw_state: dict with keys 'zone' (int) and 'current_time' (int)

        Returns:
            tuple of two ints: (zone, current_time)
        """
        zone = raw_state['zone']
        time = raw_state['current_time']
        return (zone, time)