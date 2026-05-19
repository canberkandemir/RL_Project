"""Systematic Part 1 experiments: REINFORCE and Actor-Critic on Hopper-v4."""

import time
import random

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from agent import Policy, Agent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moving_average(values, window=10):
    values = np.array(values, dtype=np.float32)

    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window) / window, mode="valid")


def make_experiment_name(algorithm, baseline):
    if algorithm == "reinforce":
        if baseline is None:
            return "reinforce_no_baseline"
        return f"reinforce_baseline_{baseline}"

    if algorithm == "actor_critic":
        return "actor_critic"

    return algorithm


def train_agent(
    algorithm="reinforce",
    baseline=None,
    n_episodes=300,
    render=False,
    seed=42,
):
    set_seed(seed)

    if render:
        env = gym.make("Hopper-v4", render_mode="human")
    else:
        env = gym.make("Hopper-v4")

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    print("\n======================================")
    print("Experiment:", make_experiment_name(algorithm, baseline))
    print("State space:", env.observation_space)
    print("Action space:", env.action_space)
    print("State dimension:", state_space)
    print("Action dimension:", action_space)
    print("Seed:", seed)
    print("======================================")

    policy = Policy(state_space, action_space)
    agent = Agent(policy, algorithm=algorithm, baseline=baseline)

    episode_rewards = []
    start_time = time.time()

    for episode in range(n_episodes):
        state, info = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, action_log_prob = agent.get_action(state)

            action_np = action.detach().cpu().numpy()
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

            next_state, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            agent.store_outcome(state, next_state, action_log_prob, reward, done)

            state = next_state
            total_reward += reward
            steps += 1

            if render:
                env.render()

        loss = agent.update_policy()
        episode_rewards.append(total_reward)

        print(
            f"Episode {episode + 1:03d}/{n_episodes} | "
            f"Reward: {total_reward:8.2f} | "
            f"Steps: {steps:3d} | "
            f"Loss: {loss:8.4f}"
        )

    elapsed_time = time.time() - start_time
    env.close()

    final_avg_10 = float(np.mean(episode_rewards[-10:]))
    final_avg_50 = float(np.mean(episode_rewards[-50:])) if len(episode_rewards) >= 50 else float(np.mean(episode_rewards))
    avg_reward = float(np.mean(episode_rewards))

    print("\nTraining finished.")
    print("Algorithm:", algorithm)
    print("Baseline:", baseline)
    print("Total time:", elapsed_time, "seconds")
    print("Average reward:", avg_reward)
    print("Final average reward last 10:", final_avg_10)
    print("Final average reward last 50:", final_avg_50)

    return {
        "episode_rewards": episode_rewards,
        "elapsed_time": elapsed_time,
        "avg_reward": avg_reward,
        "final_avg_10": final_avg_10,
        "final_avg_50": final_avg_50,
    }


def plot_learning_curves(results_df, experiment_names, output_path):
    window = 10

    plt.figure(figsize=(11, 6))

    for name in experiment_names:
        rewards = results_df[name].values
        ma_rewards = moving_average(rewards, window)

        if len(ma_rewards) == len(rewards):
            x = np.arange(1, len(ma_rewards) + 1)
        else:
            x = np.arange(window, len(rewards) + 1)

        plt.plot(x, ma_rewards, label=name)

    plt.xlabel("Episode")
    plt.ylabel("Episode reward, moving average over 10 episodes")
    plt.title("Part 1: REINFORCE Baselines vs Actor-Critic on Hopper-v4")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    n_episodes = 300
    seed = 42

    experiments = [
        {"algorithm": "reinforce", "baseline": None},
        {"algorithm": "reinforce", "baseline": 10.0},
        {"algorithm": "reinforce", "baseline": 20.0},
        {"algorithm": "reinforce", "baseline": 50.0},
        {"algorithm": "actor_critic", "baseline": None},
    ]

    all_rewards = {}
    summary_rows = []

    for exp in experiments:
        algorithm = exp["algorithm"]
        baseline = exp["baseline"]
        name = make_experiment_name(algorithm, baseline)

        result = train_agent(
            algorithm=algorithm,
            baseline=baseline,
            n_episodes=n_episodes,
            render=False,
            seed=seed,
        )

        all_rewards[name] = result["episode_rewards"]

        summary_rows.append({
            "experiment": name,
            "algorithm": algorithm,
            "baseline": baseline,
            "episodes": n_episodes,
            "seed": seed,
            "avg_reward": result["avg_reward"],
            "final_avg_10": result["final_avg_10"],
            "final_avg_50": result["final_avg_50"],
            "elapsed_time_sec": result["elapsed_time"],
        })

    results_df = pd.DataFrame({
        "episode": np.arange(1, n_episodes + 1)
    })

    for name, rewards in all_rewards.items():
        results_df[name] = rewards

    summary_df = pd.DataFrame(summary_rows)

    results_df.to_csv("part1_results.csv", index=False)
    summary_df.to_csv("part1_summary.csv", index=False)

    plot_learning_curves(
        results_df=results_df,
        experiment_names=list(all_rewards.keys()),
        output_path="part1_comparison_learning_curve.png",
    )

    print("\n========== Part 1 Summary ==========")
    print(summary_df)

    print("\nSaved CSV: part1_results.csv")
    print("Saved CSV: part1_summary.csv")
    print("Saved plot: part1_comparison_learning_curve.png")


if __name__ == "__main__":
    main()
