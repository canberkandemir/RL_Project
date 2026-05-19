import argparse
import os

import gymnasium as gym
import panda_gym  # noqa: F401

from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor

from rand_wrapper import RandomizationWrapper


def float_to_tag(value: float) -> str:
    return str(value).replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-type", type=str, default="target", choices=["source", "target"])
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Optional cube-mass randomization strategy for SAC+HER training.",
    )
    parser.add_argument("--mass-min", type=float, default=0.5)
    parser.add_argument("--mass-max", type=float, default=5.0)
    parser.add_argument("--adr-initial-min", type=float, default=0.8)
    parser.add_argument("--adr-initial-max", type=float, default=1.2)
    parser.add_argument("--adr-max-limit", type=float, default=5.0)
    parser.add_argument("--adr-step", type=float, default=0.3)
    parser.add_argument("--adr-success-threshold", type=float, default=0.25)
    parser.add_argument("--adr-window-size", type=int, default=10)
    parser.add_argument("--print-masses", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs("models", exist_ok=True)

    env = gym.make(
        "PandaPush-v3",
        type=args.env_type,
        reward_type="sparse",
    )

    if args.sampling_strategy != "none":
        env = RandomizationWrapper(
            env,
            mode=args.sampling_strategy,
            mass_range=(args.mass_min, args.mass_max),
            adr_initial_range=(args.adr_initial_min, args.adr_initial_max),
            adr_min_limit=args.mass_min,
            adr_max_limit=args.adr_max_limit,
            adr_step=args.adr_step,
            adr_success_threshold=args.adr_success_threshold,
            adr_window_size=args.adr_window_size,
            verbose=args.print_masses,
        )

    env = Monitor(env)

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        learning_rate=1e-3,
        buffer_size=1_000_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.05,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=0,
        seed=args.seed,
    )

    print("\nSAC+HER training configuration")
    print("Environment type:", args.env_type)
    print("Sampling strategy:", args.sampling_strategy)
    print("Timesteps:", args.timesteps)
    print("Seed:", args.seed)

    if args.sampling_strategy == "udr":
        print("UDR mass range:", (args.mass_min, args.mass_max))

    if args.sampling_strategy == "adr":
        print("ADR initial range:", (args.adr_initial_min, args.adr_initial_max))
        print("ADR limits:", (args.mass_min, args.adr_max_limit))
        print("ADR step:", args.adr_step)
        print("ADR success threshold:", args.adr_success_threshold)
        print("ADR window size:", args.adr_window_size)

    model.learn(total_timesteps=args.timesteps)

    if args.sampling_strategy == "none":
        save_path = f"models/sac_her_push_{args.env_type}_{args.timesteps//1000}k_seed{args.seed}"
    elif args.sampling_strategy == "udr":
        save_path = (
            f"models/sac_her_push_udr_{args.env_type}_{args.timesteps//1000}k_"
            f"m{float_to_tag(args.mass_min)}-{float_to_tag(args.mass_max)}_"
            f"seed{args.seed}"
        )
    else:
        save_path = (
            f"models/sac_her_push_adr_{args.env_type}_{args.timesteps//1000}k_"
            f"init{float_to_tag(args.adr_initial_min)}-{float_to_tag(args.adr_initial_max)}_"
            f"lim{float_to_tag(args.mass_min)}-{float_to_tag(args.adr_max_limit)}_"
            f"step{float_to_tag(args.adr_step)}_"
            f"thr{float_to_tag(args.adr_success_threshold)}_"
            f"win{args.adr_window_size}_seed{args.seed}"
        )
    model.save(save_path)

    env.close()

    print("\nTraining finished")
    print("Saved model to:", save_path + ".zip")


if __name__ == "__main__":
    main()
