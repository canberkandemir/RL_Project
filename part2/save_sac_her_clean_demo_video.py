"""Save a centered SAC+HER PandaPush demo as an MP4 video."""

from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401  # required so PandaPush-v3 is registered
from stable_baselines3 import SAC


MODEL_PATH = "models/sac_her_push_target_500k_seed42.zip"
OUTPUT_PATH = Path("results/pandapush_centered_demo_seed280.mp4")

ENV_TYPE = "target"
DEMO_SEED = 280
SUCCESS_DISTANCE_THRESHOLD = 0.005
MAX_EPISODE_STEPS = 60

FPS = 30
INITIAL_HOLD_SECONDS = 2.0
STEP_REPEAT_FRAMES = 12
FINAL_HOLD_SECONDS = 6.0


def make_env():
    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=ENV_TYPE,
        reward_type="sparse",
        render_width=1280,
        render_height=720,
    )
    env.unwrapped.task.distance_threshold = SUCCESS_DISTANCE_THRESHOLD
    return env


def goal_distance(obs):
    achieved = np.array(obs["achieved_goal"], dtype=float)
    desired = np.array(obs["desired_goal"], dtype=float)
    return float(np.linalg.norm(achieved - desired))


def append_frame(frames, env, repeat=1):
    frame = env.render()
    for _ in range(repeat):
        frames.append(frame)


def save_mp4(frames, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()


def main():
    load_env = make_env()
    model = SAC.load(MODEL_PATH, env=load_env)

    env = make_env()
    obs, info = env.reset(seed=DEMO_SEED)
    init_dist = goal_distance(obs)

    frames = []
    append_frame(frames, env, repeat=int(INITIAL_HOLD_SECONDS * FPS))

    terminated = False
    truncated = False
    steps = 0
    final_dist = init_dist
    final_info = {}
    total_reward = 0.0

    while not (terminated or truncated) and steps < MAX_EPISODE_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        total_reward += float(reward)
        final_info = info
        final_dist = goal_distance(obs)

        append_frame(frames, env, repeat=STEP_REPEAT_FRAMES)

    append_frame(frames, env, repeat=int(FINAL_HOLD_SECONDS * FPS))

    save_mp4(frames, OUTPUT_PATH)

    success = bool(final_info.get("is_success", False))
    print("Saved video:", OUTPUT_PATH)
    print("Seed:", DEMO_SEED)
    print("Success:", success)
    print("Steps:", steps)
    print(f"Initial distance: {init_dist:.4f}")
    print(f"Final distance: {final_dist:.4f}")
    print(f"Return: {total_reward:.3f}")

    env.close()
    load_env.close()


if __name__ == "__main__":
    main()
