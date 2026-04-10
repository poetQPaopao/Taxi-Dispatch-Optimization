# state_encoder.py
class StateEncoder:
    def __init__(self, num_zones, max_orders=5, wait_buckets=[0, 5, 10, float('inf')]):
        self.num_zones = num_zones
        self.max_orders = max_orders
        self.wait_buckets = wait_buckets  # e.g., [0,5,10,inf] -> bucket index 0,1,2

    def _bucket_wait_time(self, wait_time):
        for i, bound in enumerate(self.wait_buckets[:-1]):
            if wait_time < bound:
                return i
        return len(self.wait_buckets) - 1  # last bucket

    def encode(self, raw_state):
        """
        raw_state: dict with 'taxis', 'orders', 'current_time'
        returns tuple of ints
        """
        # 1. Taxi info: for each taxi (id, location, is_free)
        taxi_features = []
        for taxi in raw_state['taxis']:
            taxi_features.append(taxi['location'])          # zone id (0..num_zones-1)
            taxi_features.append(1 if taxi['is_free'] else 0)
            
        # 2. Orders: sort by creation time (oldest first), take first max_orders
        orders = sorted(raw_state['orders'], key=lambda o: o['created_time'])
        orders = orders[:self.max_orders]
        order_features = []
        current_time = raw_state['current_time']
        for order in orders:
            wait_time = current_time - order['created_time']
            wait_bucket = self._bucket_wait_time(wait_time)
            order_features.append(order['pickup'])   # pickup zone
            order_features.append(wait_time)       # waiting bucket
        # Pad if fewer orders than max_orders
        while len(order_features) < self.max_orders * 2:
            order_features.append(-1)  # sentinel for no order
        # Combine all into tuple
        state_tuple = tuple(taxi_features + order_features)
        return state_tuple
