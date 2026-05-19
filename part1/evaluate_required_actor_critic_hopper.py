"""Evaluate the Part 1 required from-scratch Actor-Critic Hopper policy."""

import argparse

import gymnasium as gym
import numpy as np
import torch

from agent import Agent, Policy


def evaluate(model_path, episodes, seed):
    env = gym.make("Hopper-v4")

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    policy = Policy(state_space, action_space)
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()

    agent = Agent(policy, algorithm="actor_critic", baseline=None)

    returns = []
    lengths = []

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            with torch.no_grad():
                action, _ = agent.get_action(state, evaluation=True)

            action_np = np.clip(
                action.detach().cpu().numpy(),
                env.action_space.low,
                env.action_space.high,
            )

            state, reward, terminated, truncated, info = env.step(action_np)
            total_reward += float(reward)
            steps += 1

        returns.append(total_reward)
        lengths.append(steps)
        print(f"Episode {episode + 1:03d} | return={total_reward:.2f} | steps={steps}")

    env.close()

    returns = np.array(returns, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.float32)

    print("\n=== Required Actor-Critic evaluation ===")
    print("Model:", model_path)
    print("Episodes:", episodes)
    print(f"Mean return: {returns.mean():.2f}")
    print(f"Std return:  {returns.std():.2f}")
    print(f"Min return:  {returns.min():.2f}")
    print(f"Max return:  {returns.max():.2f}")
    print(f"Mean length: {lengths.mean():.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="best_actor_critic_policy.pth")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    evaluate(args.model_path, args.episodes, args.seed)


if __name__ == "__main__":
    main()
