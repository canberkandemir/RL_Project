"""Render the extra PPO Hopper policy with fast, stable camera tracking."""

import argparse
import time

import gymnasium as gym
import mujoco
from stable_baselines3 import PPO


def set_follow_camera(env):
    try:
        env.render()

        viewer = env.unwrapped.mujoco_renderer.viewer
        model = env.unwrapped.model

        torso_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "torso",
        )

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = torso_id

        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

    except Exception as error:
        print("Could not set tracking camera:", error)


def update_camera_position(env):
    """Keep the camera centered on the Hopper's forward progress."""
    try:
        viewer = env.unwrapped.mujoco_renderer.viewer
        torso_x = float(env.unwrapped.data.qpos[0])
        torso_z = float(env.unwrapped.data.qpos[1])

        viewer.cam.lookat[0] = torso_x
        viewer.cam.lookat[1] = 0.0
        viewer.cam.lookat[2] = max(0.8, torso_z)

    except Exception as error:
        print("Could not update camera position:", error)


def main():
    parser = argparse.ArgumentParser(description="Render the extra PPO Hopper visual demo.")
    parser.add_argument("--model-path", default="models/ppo_hopper_500k")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--sleep-time", type=float, default=0.0)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument(
        "--sim-steps-per-render",
        type=int,
        default=4,
        help="Simulator steps between rendered frames. Higher values make the render faster.",
    )
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.render_every = max(1, args.render_every)
    args.sim_steps_per_render = max(1, args.sim_steps_per_render)

    env = gym.make("Hopper-v4", render_mode="human")
    model = PPO.load(args.model_path)

    print("Loaded model:", args.model_path)
    print("Duration seconds:", args.duration)
    print("Render every:", args.render_every)
    print("Sim steps per render:", args.sim_steps_per_render)
    print("Action scale:", args.action_scale)
    print("Sleep time:", args.sleep_time)

    start_time = time.time()
    episode = 0

    while time.time() - start_time < args.duration:
        obs, info = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        set_follow_camera(env)

        while not (terminated or truncated) and time.time() - start_time < args.duration:
            for _ in range(args.sim_steps_per_render):
                if terminated or truncated:
                    break

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(args.action_scale * action)

                total_reward += float(reward)
                steps += 1

            if steps % args.render_every == 0:
                update_camera_position(env)
                env.render()
                if args.sleep_time > 0:
                    time.sleep(args.sleep_time)

        print(f"Episode {episode} | reward = {total_reward:.2f} | steps = {steps}")
        episode += 1

    env.close()


if __name__ == "__main__":
    main()
