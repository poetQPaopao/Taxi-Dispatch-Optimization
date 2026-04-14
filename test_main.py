from taxi_envs.env_utils import make_env
from Algorithm.state_encoder import StateEncoder
from Algorithm.nstep_sarsa import NStepSarsaAgent

from Experiment.runner import evaluate
from Experiment.metrics import summarize_results
from Experiment.baseline import RandomDispatchAgent
from Experiment.plot import plot_reward_curve, plot_episode_metrics
from Experiment.utils import make_output_dir, save_json, save_jsonl
from Experiment.config import load_config


def build_rl_agent(env, config):
    encoder = StateEncoder(
        num_zones=env.grid_size * env.grid_size,
        max_orders=config["max_orders"],
    )
    agent = NStepSarsaAgent(
        state_encoder=encoder,
        num_taxis=env.num_taxis,
        max_orders=config["max_orders"],
        n=config["n"],
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=config["epsilon"],
    )
    agent.name = "nstep_sarsa"
    return agent


def run_and_save(env, agent, use_learning_agent, out_dir, tag, config):
    results, step_logs = evaluate(
        env=env,
        agent=agent,
        episodes=config["episodes"],
        use_learning_agent=use_learning_agent,
    )

    summary = summarize_results(results)

    save_json(config, out_dir / f"{tag}_config.json")
    save_json(summary, out_dir / f"{tag}_summary.json")
    save_jsonl(results, out_dir / f"{tag}_episode_results.jsonl")
    save_jsonl(step_logs, out_dir / f"{tag}_step_logs.jsonl")

    tmp_plot_dir = out_dir / f"{tag}_plots_temp"
    plot_reward_curve(results, tmp_plot_dir)
    plot_episode_metrics(results, tmp_plot_dir)

    import shutil
    shutil.move(str(tmp_plot_dir / "reward_curve.png"), str(out_dir / f"{tag}_reward_curve.png"))
    shutil.move(str(tmp_plot_dir / "episode_metrics.png"), str(out_dir / f"{tag}_episode_metrics.png"))
    tmp_plot_dir.rmdir()

    print(f"[{tag}] summary = {summary}")


def main():
    config = load_config()
    out_dir = make_output_dir("outputs", config["run_name"] or None)

    save_json(config, out_dir / "run_config.json")

    if config["agent"] in ["random", "both"]:
        env = make_env(
            num_taxis=config["num_taxis"],
            grid_size=config["grid_size"],
            max_steps=config["max_steps"],
            seed=config["seed"],
        )
        random_agent = RandomDispatchAgent()
        run_and_save(
            env=env,
            agent=random_agent,
            use_learning_agent=False,
            out_dir=out_dir,
            tag="random",
            config=config,
        )

    if config["agent"] in ["rl", "both"]:
        env = make_env(
            num_taxis=config["num_taxis"],
            grid_size=config["grid_size"],
            max_steps=config["max_steps"],
            seed=config["seed"],
        )
        rl_agent = build_rl_agent(env, config)
        run_and_save(
            env=env,
            agent=rl_agent,
            use_learning_agent=True,
            out_dir=out_dir,
            tag="rl",
            config=config,
        )

    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()