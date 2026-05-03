from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


def _safe_mean(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


# ---------------------------------------------------------------------------
# Episode grouping
# ---------------------------------------------------------------------------

def split_records_by_episode(records: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[int(rec["episode"])].append(rec)
    for ep in grouped:
        grouped[ep].sort(key=lambda x: int(x["step"]))
    return dict(sorted(grouped.items()))


# ---------------------------------------------------------------------------
# Per-episode metrics
# ---------------------------------------------------------------------------

def compute_episode_metrics(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute one row per episode with the 8 core metrics."""
    grouped = split_records_by_episode(records)
    rows: List[Dict[str, Any]] = []

    for ep, ep_records in grouped.items():
        total_reward = sum(float(rec["reward"]) for rec in ep_records)
        steps = len(ep_records)
        total_time = sum(max(1, int(rec["time_elapsed"])) for rec in ep_records)

        # Cumulative counters start at 0 after reset — last value is episode total
        completed = int(ep_records[-1].get("completed_orders", 0))
        total_orders_gen = int(ep_records[-1].get("total_orders", 0))
        empty_time = int(ep_records[-1].get("empty_time", 0))
        occupied_time = int(ep_records[-1].get("occupied_time", 0))

        # behavioral
        reposition_count = 0
        stay_count = 0
        fares: List[float] = []
        for rec in ep_records:
            zone = int(rec["state"][0])
            action = int(rec["action"])
            if action == zone:
                stay_count += 1
            else:
                reposition_count += 1
            fare = float(rec.get("trip_fare", 0.0))
            if fare > 0:
                fares.append(fare)

        row = {
            "episode": ep,
            "steps": steps,
            "total_time": total_time,
            # Core
            "total_reward": total_reward,
            "profit_per_time": total_reward / total_time if total_time else 0.0,
            "completed_orders": completed,
            "completion_rate": completed / total_orders_gen if total_orders_gen else 0.0,
            "empty_drive_ratio": empty_time / (empty_time + occupied_time) if (empty_time + occupied_time) else 0.0,
            # Behavioral
            "reposition_rate": reposition_count / steps if steps else 0.0,
            "stay_and_wait_rate": stay_count / steps if steps else 0.0,
            "mean_trip_fare": _safe_mean(fares),
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Summary (aggregated over all episodes)
# ---------------------------------------------------------------------------

def summarize_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    def col(name: str) -> List[float]:
        return [float(r[name]) for r in rows]

    summary = {
        "num_episodes": len(rows),
        # Core
        "mean_total_reward": _safe_mean(col("total_reward")),
        "mean_profit_per_time": _safe_mean(col("profit_per_time")),
        "mean_completed_orders": _safe_mean(col("completed_orders")),
        "mean_completion_rate": _safe_mean(col("completion_rate")),
        "mean_empty_drive_ratio": _safe_mean(col("empty_drive_ratio")),
        # Behavioral
        "mean_reposition_rate": _safe_mean(col("reposition_rate")),
        "mean_stay_and_wait_rate": _safe_mean(col("stay_and_wait_rate")),
        "mean_trip_fare": _safe_mean(col("mean_trip_fare")),
        # Best
        "best_episode_by_reward": int(max(rows, key=lambda r: r["total_reward"])["episode"]),
        "best_reward": float(max(rows, key=lambda r: r["total_reward"])["total_reward"]),
        "best_profit_per_time": float(max(rows, key=lambda r: r["profit_per_time"])["profit_per_time"]),
    }

    # learning gain: second half vs first half
    half = max(1, len(rows) // 2)
    first_half = rows[:half]
    second_half = rows[half:]

    summary["learning_gain_reward"] = (
        _safe_mean([r["total_reward"] for r in second_half]) -
        _safe_mean([r["total_reward"] for r in first_half])
    )
    summary["learning_gain_completion_rate"] = (
        _safe_mean([r["completion_rate"] for r in second_half]) -
        _safe_mean([r["completion_rate"] for r in first_half])
    )
    summary["learning_gain_profit_per_time"] = (
        _safe_mean([r["profit_per_time"] for r in second_half]) -
        _safe_mean([r["profit_per_time"] for r in first_half])
    )

    return summary


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_episode_metrics(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_metrics(summary: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def build_and_save_metrics(
    records: List[Dict[str, Any]],
    output_dir: str | Path,
    prefix: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output_dir = Path(output_dir)
    rows = compute_episode_metrics(records)
    summary = summarize_metrics(rows)
    save_episode_metrics(rows, output_dir / f"{prefix}_episode_metrics.csv")
    save_summary_metrics(summary, output_dir / f"{prefix}_summary_metrics.json")
    return rows, summary
