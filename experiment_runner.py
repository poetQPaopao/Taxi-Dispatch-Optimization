from __future__ import annotations

import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from taxi_envs.env_utils import make_env, make_graph_env
from utils import make_run_dir, save_pickle, save_reward_curve, save_json

ROOT_DIR = os.path.dirname(__file__)
ALGO_DIR = os.path.join(ROOT_DIR, "Algorithm")
if ALGO_DIR not in sys.path:
    sys.path.insert(0, ALGO_DIR)

from Algorithm.nstep_sarsa import NStepSarsaAgent
from Algorithm.state_encoder import StateEncoder
from baseline import RandomDispatchAgent
from Experiment.metrics import build_and_save_metrics
from visualization.trajectory import TrajectoryRecorder, save_trajectory




# -------------------
# Environment Builder
# -------------------
def _build_env(env_type: str, max_steps: int, seed: int | None, env_kwargs: Dict[str, Any] | None = None):
    env_kwargs = dict(env_kwargs or {})

    if env_type == "grid":
        return make_env(max_steps=max_steps, seed=seed, **env_kwargs)

    if env_type == "graph":
        cache_path = env_kwargs.pop("cache_path", "cache/taxi_graph.graphml")
        return make_graph_env(
            max_steps=max_steps,
            seed=seed,
            cache_path=cache_path,
            **env_kwargs,
        )

    raise ValueError(f"Unknown env_type: {env_type}")


def _as_raw_state(obs: Any) -> Dict[str, int]:
    zone, current_time = obs
    return {"zone": int(zone), "current_time": int(current_time)}


def _episode_seed(base_seed: int | None, episode_idx: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + int(episode_idx)


# -------------------
# Env Metrics Helper
# -------------------
def _attach_env_metrics(env, info: Dict[str, Any]) -> Dict[str, Any]:
    info = dict(info)
    if "pending_counts" not in info and hasattr(env, "pending_orders"):
        try:
            info["pending_counts"] = [len(order_list) for order_list in env.pending_orders]
        except TypeError:
            info["pending_counts"] = []
    if "pending_total" not in info and "pending_counts" in info:
        info["pending_total"] = int(sum(info["pending_counts"]))
    if "completed_orders" not in info and hasattr(env, "completed_orders"):
        info["completed_orders"] = int(env.completed_orders)
    if "total_orders" not in info and hasattr(env, "total_orders"):
        info["total_orders"] = int(env.total_orders)
    if "empty_time" not in info and hasattr(env, "empty_time"):
        info["empty_time"] = int(env.empty_time)
    if "occupied_time" not in info and hasattr(env, "occupied_time"):
        info["occupied_time"] = int(env.occupied_time)
    return info


# -------------------
# Checkpoint Helper
# -------------------
def _save_checkpoint(agent, ckpt_dir: Path, episode_idx: int, tag: str | None = None) -> str:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if tag:
        if tag == "best_total_reward":
            path = ckpt_dir / "best.pkl"
        else:
            path = ckpt_dir / f"{tag}.pkl"
    else:
        path = ckpt_dir / f"ep{episode_idx:05d}.pkl"
    agent.save(str(path))
    print(f"[ckpt] saved: {path}")
    return str(path)


# -------------------
# Run Agent (Train/Eval)
# -------------------
def _run_trained_agent(
    env,
    agent,
    episodes: int,
    max_steps: int,
    seed: int | None,
    run_dir: Path,
    save_checkpoints: bool = True,
    checkpoint_every: int = 500,
    checkpoint_episodes: List[int] | None = None,
    save_best_checkpoint: bool = True,
    log_every: int = 50,
    do_learning: bool = True,
):
    reward_history = []
    recorder = TrajectoryRecorder()
    checkpoint_episodes = sorted(set(checkpoint_episodes or []))
    ckpt_dir = run_dir / "checkpoints"
    saved_checkpoints: List[Dict[str, Any]] = []

    best_total_reward = float("-inf")
    best_total_reward_path = ""
    best_total_reward_episode = None

    for ep in range(episodes):
        obs, info = env.reset(seed=_episode_seed(seed, ep))
        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        agent.start_episode(_as_raw_state(obs))

        while not (terminated or truncated) and step < max_steps:
            current_state = obs
            action = agent.get_current_action()
            obs, reward, terminated, truncated, info = env.step(action)

            info = _attach_env_metrics(env, info)

            recorder.add_step(
                episode=ep,
                step=step,
                state=current_state,
                action=action,
                next_state=obs,
                reward=reward,
                info=info,
                terminated=terminated,
                truncated=truncated,
            )

            dt = int(info.get("time_elapsed", 1))
            total_reward += reward

            if do_learning:
                agent.step(
                    _as_raw_state(obs),
                    reward,
                    terminated or truncated,
                    duration=max(dt, 1),
                )
            step += 1

        reward_history.append((ep, total_reward))
        current_ep = ep + 1

        if log_every > 0 and ((ep + 1) % log_every == 0 or ep == 0 or ep == episodes - 1):
            phase = "train" if do_learning else "eval"
            print(
                f"[{phase}] ep={ep + 1}/{episodes} "
                f"reward={total_reward:.2f} "
                f"steps={step} "
                f"epsilon={getattr(agent, 'epsilon', -1):.3f}"
            )

        if do_learning and save_checkpoints:
            should_save_regular = checkpoint_every > 0 and current_ep % checkpoint_every == 0
            should_save_manual = current_ep in checkpoint_episodes
            if should_save_regular or should_save_manual:
                ckpt_path = _save_checkpoint(agent, ckpt_dir, current_ep)
                saved_checkpoints.append(
                    {
                        "episode": current_ep,
                        "path": ckpt_path,
                        "reason": "manual" if should_save_manual and not should_save_regular else "interval",
                        "total_reward": total_reward,
                    }
                )

        if do_learning and save_best_checkpoint and total_reward > best_total_reward:
            best_total_reward = total_reward
            best_total_reward_episode = current_ep
            best_total_reward_path = _save_checkpoint(agent, ckpt_dir, current_ep, tag="best_total_reward")
            print(f"[ckpt] new best reward ckpt: {best_total_reward_path} | reward={total_reward:.2f}")

            saved_checkpoints.append(
                {
                    "episode": current_ep,
                    "path": best_total_reward_path,
                    "reason": "best_total_reward",
                    "total_reward": total_reward,
                }
            )

    checkpoint_index = {
        "checkpoint_every": checkpoint_every,
        "checkpoint_episodes": checkpoint_episodes,
        "best_total_reward": {
            "episode": best_total_reward_episode,
            "path": best_total_reward_path,
            "reward": best_total_reward if best_total_reward > float("-inf") else None,
        },
        "saved_checkpoints": saved_checkpoints,
    }

    if do_learning:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with (ckpt_dir / "checkpoint_index.json").open("w", encoding="utf-8") as f:
            json.dump(checkpoint_index, f, indent=2)

    return reward_history, recorder, checkpoint_index


def _run_random_agent(env, episodes: int, max_steps: int, seed: int | None, log_every: int = 50):
    agent = RandomDispatchAgent()
    reward_history = []
    recorder = TrajectoryRecorder()

    for ep in range(episodes):
        obs, info = env.reset(seed=_episode_seed(seed, ep))
        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and step < max_steps:
            current_state = obs
            action = agent.act(env)
            obs, reward, terminated, truncated, info = env.step(action)

            info = _attach_env_metrics(env, info)

            recorder.add_step(
                episode=ep,
                step=step,
                state=current_state,
                action=action,
                next_state=obs,
                reward=reward,
                info=info,
                terminated=terminated,
                truncated=truncated,
            )

            total_reward += reward
            step += 1

        reward_history.append((ep, total_reward))

        if log_every > 0 and ((ep + 1) % log_every == 0 or ep == 0 or ep == episodes - 1):
            print(
                f"[random] ep={ep + 1}/{episodes} "
                f"reward={total_reward:.2f} "
                f"steps={step}"
            )

    return reward_history, recorder



def _run_visualizations(
    *,
    run_dir: Path,
    env_type: str,
    vis_episode_idx: int,
    env_kwargs: Dict[str, Any],
):
    if env_type == "grid":
        from visualization.grid_replay import run_grid_compare
        from visualization.grid_animation import run_grid_animation_compare

        run_grid_compare(outputs_dir=run_dir, episode_idx=vis_episode_idx)
        run_grid_animation_compare(
            outputs_dir=run_dir,
            episode_idx=vis_episode_idx,
            fps=3,
            interval_ms=300,
            save_gif=True,
            save_mp4=False,
            show_plot=False,
        )
    elif env_type == "graph":
        from visualization.graph_animation import run_graph_animation

        graph_path = env_kwargs.get("cache_path", "cache/taxi_graph.graphml")
        run_graph_animation(
            outputs_dir=run_dir,
            episode_idx=vis_episode_idx,
            mode="trained",
            graph_path=graph_path,
            fps=6,
            record=True,
        )
        run_graph_animation(
            outputs_dir=run_dir,
            episode_idx=vis_episode_idx,
            mode="random",
            graph_path=graph_path,
            fps=6,
        )



# -------------------
# Main Experiment Function
# -------------------
def run_single_experiment(
    *,
    mode: str = "eval",   # "train" or "eval"
    run_name: str | None = None,
    parent_dir: str | Path | None = None,
    episodes: int = 1,
    max_steps: int = 200,
    seed: int | None = 42,
    env_type: str = "grid",
    env_kwargs: Dict[str, Any] | None = None,
    n: int = 1,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.2,
    save_model_name: str = "nstep_agent.pkl",
    load_path: str | None = None,
    evaluate_random: bool = True,
    make_visualizations: bool = False,
    vis_episode_idx: int = 100,
    save_checkpoints: bool = True,
    checkpoint_every: int = 500,
    checkpoint_episodes: List[int] | None = None,
    save_best_checkpoint: bool = True,
    log_every: int = 50,
) -> Dict[str, Any]:
    if mode not in {"train", "eval"}:
        raise ValueError("mode must be 'train' or 'eval'")
    if mode == "eval" and not load_path:
        raise ValueError("In eval mode, load_path must be provided.")

    run_dir = make_run_dir("outputs", name=run_name) if parent_dir is None else Path(parent_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _build_env(env_type=env_type, max_steps=max_steps, seed=seed, env_kwargs=env_kwargs)

    config = {
        "mode": mode,
        "run_name": run_name,
        "parent_dir": str(parent_dir) if parent_dir is not None else None,
        "episodes": episodes,
        "max_steps": max_steps,
        "seed": seed,
        "env_type": env_type,
        "env_kwargs": env_kwargs or {},
        "n": n,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "save_model_name": save_model_name,
        "load_path": load_path,
        "evaluate_random": evaluate_random,
        "make_visualizations": make_visualizations,
        "vis_episode_idx": vis_episode_idx,
        "save_checkpoints": save_checkpoints,
        "checkpoint_every": checkpoint_every,
        "checkpoint_episodes": checkpoint_episodes or [],
        "save_best_checkpoint": save_best_checkpoint,
        "log_every": log_every,
    }
    save_json(config, run_dir / "config.json")

    encoder = StateEncoder(num_zones=env.n_zones, max_time_steps=env.episode_length)
    agent = NStepSarsaAgent(state_encoder=encoder, num_actions=env.n_zones, n=n, alpha=alpha, gamma=gamma, epsilon=epsilon)
    if load_path:
        agent.load(load_path)

    trained_history, trained_recorder, checkpoint_index = _run_trained_agent(
        env=env,
        agent=agent,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        run_dir=run_dir,
        save_checkpoints=save_checkpoints if mode == "train" else False,
        checkpoint_every=checkpoint_every,
        checkpoint_episodes=checkpoint_episodes,
        save_best_checkpoint=save_best_checkpoint if mode == "train" else False,
        log_every=log_every,
        do_learning=(mode == "train"),
    )

    save_pickle(trained_history, run_dir / "trained_reward_history.pkl")
    save_reward_curve(trained_history, run_dir / "trained_reward_curve.png")
    save_trajectory(trained_recorder.get_records(), run_dir / "trained_trajectory.pkl")
    _, trained_metric_summary = build_and_save_metrics(trained_recorder.get_records(), run_dir, prefix="trained")
    
    records = trained_recorder.get_records()
    print("[DEBUG first record]", records[0] if records else None)
    print("[DEBUG second record]", records[1] if len(records) > 1 else None)

    print("[metrics][trained]", trained_metric_summary)

    model_path = run_dir / save_model_name
    if mode == "train":
        agent.save(str(model_path))
    else:
        model_path = Path(load_path)

    random_history = []
    random_metric_summary = {}
    if evaluate_random:
        random_env = _build_env(env_type=env_type, max_steps=max_steps, seed=seed, env_kwargs=env_kwargs)
        random_history, random_recorder = _run_random_agent(
            env=random_env,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            log_every=log_every,
        )
        save_pickle(random_history, run_dir / "random_reward_history.pkl")
        save_reward_curve(random_history, run_dir / "random_reward_curve.png")
        save_trajectory(random_recorder.get_records(), run_dir / "random_trajectory.pkl")
        _, random_metric_summary = build_and_save_metrics(
            random_recorder.get_records(), run_dir, prefix="random"
        )
        print("[metrics][random]", random_metric_summary)

    result = {
        "run_dir": str(run_dir),
        "config": config,
        "trained_mean_reward": float(mean(v for _, v in trained_history)),
        "trained_summary_metrics": trained_metric_summary,
        "random_mean_reward": float(mean(v for _, v in random_history)) if random_history else None,
        "random_summary_metrics": random_metric_summary,
        "model_path": str(model_path),
    }

    if make_visualizations:
        _run_visualizations(
            run_dir=run_dir,
            env_type=env_type,
            vis_episode_idx=vis_episode_idx,
            env_kwargs=env_kwargs or {},
        )

    return result