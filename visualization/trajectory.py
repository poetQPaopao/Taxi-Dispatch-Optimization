from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import pickle


@dataclass
class TrajectoryStep:
    episode: int
    step: int

    state: tuple[int, int]
    action: int
    next_state: tuple[int, int]

    reward: float
    matched: bool
    illegal: bool

    terminated: bool
    truncated: bool

    time_elapsed: int
    pending_counts: list[int]

    completed_orders: int
    total_orders: int
    empty_time: int
    occupied_time: int
    trip_fare: float


class TrajectoryRecorder:
    """Independent trajectory recorder. Only records what the outer runner loop sees."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add_step(
        self,
        *,
        episode: int,
        step: int,
        state: tuple[int, int],
        action: int,
        next_state: tuple[int, int],
        reward: float,
        info: dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> None:
        record = TrajectoryStep(
            episode=episode,
            step=step,
            state=(int(state[0]), int(state[1])),
            action=int(action),
            next_state=(int(next_state[0]), int(next_state[1])),
            reward=float(reward),
            matched=bool(info.get("matched", False)),
            illegal=bool(info.get("illegal", False)),
            terminated=bool(terminated),
            truncated=bool(truncated),
            time_elapsed=int(info.get("time_elapsed", 1)),
            pending_counts=[int(x) for x in info.get("pending_counts", [])],
            completed_orders=int(info.get("completed_orders", -1)),
            total_orders=int(info.get("total_orders", -1)),
            empty_time=int(info.get("empty_time", -1)),
            occupied_time=int(info.get("occupied_time", -1)),
            trip_fare=float(info.get("fare", 0.0)),
        )
        self.records.append(asdict(record))

    def reset(self) -> None:
        self.records.clear()

    def get_records(self) -> list[dict[str, Any]]:
        return self.records


def save_trajectory(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(records, f)


def load_trajectory(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)
