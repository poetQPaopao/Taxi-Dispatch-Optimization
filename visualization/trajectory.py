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


class TrajectoryRecorder:
    """
    Independent trajectory recorder.

    - Does NOT depend on internal env implementation details
    - Does NOT modify agent/env
    - Only records what the outer runner/test loop already sees
    """

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