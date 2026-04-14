import os


def _get_int(name, default):
    return int(os.getenv(name, default))


def _get_float(name, default):
    return float(os.getenv(name, default))


def _get_str(name, default):
    return os.getenv(name, default)


def load_config():
    config = {
        "agent": _get_str("AGENT", "rl"),              # rl / random / both
        "episodes": _get_int("EPISODES", 5),
        "max_steps": _get_int("MAX_STEPS", 50),
        "seed": _get_int("SEED", 0),

        "num_taxis": _get_int("NUM_TAXIS", 5),
        "grid_size": _get_int("GRID_SIZE", 10),
        "max_orders": _get_int("MAX_ORDERS", 5),

        "n": _get_int("N_STEP", 3),
        "alpha": _get_float("ALPHA", 0.1),
        "gamma": _get_float("GAMMA", 0.95),
        "epsilon": _get_float("EPSILON", 0.2),

        "run_name": _get_str("RUN_NAME", ""),
    }
    return config