"""Train PPO on Hopper-v4 for a stronger visual locomotion demo."""

import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def main():
    os.makedirs("models", exist_ok=True)

    env = gym.make("Hopper-v4")
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=42,
    )

    # Quick test: 200_000
    # Better visual result: 1_000_000 or more
    total_timesteps = 500_000

    model.learn(total_timesteps=total_timesteps)

    model.save("models/ppo_hopper_500k")

    env.close()

    print("\nTraining finished.")
    print("Saved PPO Hopper model to: models/ppo_hopper_500k.zip")


if __name__ == "__main__":
    main()
