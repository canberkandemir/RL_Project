import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import cv2
import gymnasium as gym
import mujoco
import numpy as np
import torch


OUT = ROOT / "renders"
SIZE = (1280, 720)
FPS = 30

sys.path.insert(0, str(ROOT / "part1"))
sys.path.insert(0, str(ROOT / "part2"))
sys.path.insert(0, str(ROOT / "part2" / "panda-gym"))

import panda_gym  # noqa: E402,F401
from agent import Agent, Policy  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402


def resize(frame):
    return cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA)


def add_label(frame, title, detail=""):
    frame = resize(frame)
    out = frame.copy()

    cv2.rectangle(out, (0, 0), (SIZE[0], 92), (18, 24, 31), thickness=-1)
    cv2.putText(
        out,
        title,
        (28, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if detail:
        cv2.putText(
            out,
            detail,
            (28, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (214, 226, 236),
            1,
            cv2.LINE_AA,
        )

    return out


def title_frame(title, detail=""):
    frame = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    frame[:] = (24, 30, 38)

    cv2.putText(
        frame,
        title,
        (70, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if detail:
        cv2.putText(
            frame,
            detail,
            (70, 385),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (212, 224, 232),
            1,
            cv2.LINE_AA,
        )

    return frame


def open_writer(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        SIZE,
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")

    return writer


def write_rgb(writers, frame):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for writer in writers:
        writer.write(bgr)


def write_title(writers, title, detail=""):
    frame = title_frame(title, detail)
    for _ in range(FPS):
        write_rgb(writers, frame)


def hopper_camera(env):
    try:
        env.render()
        renderer = env.unwrapped.mujoco_renderer
        viewer = renderer._get_viewer("rgb_array")

        torso_id = mujoco.mj_name2id(
            env.unwrapped.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "torso",
        )

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = torso_id
        viewer.cam.distance = 6.2
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -18
    except Exception:
        pass


def update_hopper_camera(env):
    try:
        viewer = env.unwrapped.mujoco_renderer._get_viewer("rgb_array")
        viewer.cam.lookat[0] = float(env.unwrapped.data.qpos[0])
        viewer.cam.lookat[1] = 0.0
        viewer.cam.lookat[2] = 1.2
    except Exception:
        pass


def hopper_frame_detail(env, total_reward, steps, episode):
    x = float(env.unwrapped.data.qpos[0])
    z = float(env.unwrapped.data.qpos[1])
    return f"episode {episode} | step {steps} | reward {total_reward:.1f} | x {x:.2f} | z {z:.2f}"


def render_hopper_actor_critic(writers, preview_frames, seconds):
    title = "Part 1 required: Actor-Critic Hopper"
    detail = "From-scratch Actor-Critic policy"
    write_title(writers, title, detail)

    env = gym.make("Hopper-v4", render_mode="rgb_array")
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    policy = Policy(state_space, action_space)
    policy.load_state_dict(torch.load(ROOT / "part1" / "best_actor_critic_policy.pth", map_location="cpu"))
    policy.eval()
    agent = Agent(policy, algorithm="actor_critic")

    total_frames = seconds * FPS
    frame_count = 0
    episode = 0
    last_frame = None

    while frame_count < total_frames:
        state, _ = env.reset(seed=123 + episode)
        hopper_camera(env)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated) and frame_count < total_frames:
            for _ in range(3):
                if terminated or truncated:
                    break
                with torch.no_grad():
                    action, _ = agent.get_action(state, evaluation=True)
                action = action.detach().cpu().numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1

            update_hopper_camera(env)
            frame = env.render()
            frame = add_label(frame, title, hopper_frame_detail(env, total_reward, steps, episode))
            write_rgb(writers, frame)
            last_frame = frame
            frame_count += 1

        episode += 1

    preview_frames.append((title, last_frame))
    env.close()


def render_hopper_ppo(writers, preview_frames, seconds):
    title = "Part 1 extra: PPO Hopper"
    detail = "Extra stable-baselines3 PPO demo, kept separate from required algorithms"
    write_title(writers, title, detail)

    env = gym.make("Hopper-v4", render_mode="rgb_array")
    model = PPO.load(ROOT / "part1" / "models" / "ppo_hopper_500k.zip")

    total_frames = seconds * FPS
    frame_count = 0
    episode = 0
    last_frame = None

    while frame_count < total_frames:
        obs, _ = env.reset(seed=42 + episode)
        hopper_camera(env)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated) and frame_count < total_frames:
            for _ in range(4):
                if terminated or truncated:
                    break
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1

            update_hopper_camera(env)
            frame = env.render()
            detail = hopper_frame_detail(env, total_reward, steps, episode)
            frame = add_label(frame, title, detail)
            write_rgb(writers, frame)
            last_frame = frame
            frame_count += 1

        episode += 1

    preview_frames.append((title, last_frame))
    env.close()


def make_push_env(env_type):
    return gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=env_type,
        reward_type="dense",
        render_width=SIZE[0],
        render_height=SIZE[1],
    )


def load_sb3_model(model_path, env):
    model_path = str(model_path)
    if "ppo" in model_path:
        return PPO.load(model_path)
    return SAC.load(model_path, env=env)


def goal_distance(obs):
    achieved = np.asarray(obs["achieved_goal"], dtype=float)
    desired = np.asarray(obs["desired_goal"], dtype=float)
    return float(np.linalg.norm(achieved - desired))


def render_push_clip(writers, preview_frames, title, model_path, env_type, seed, max_steps=50):
    detail = f"{env_type} environment | {Path(model_path).name}"
    write_title(writers, title, detail)

    env = make_push_env(env_type)
    model = load_sb3_model(model_path, env)
    obs, info = env.reset(seed=seed)

    initial_dist = goal_distance(obs)
    final_dist = initial_dist
    total_reward = 0.0
    success = False
    terminated = False
    truncated = False
    steps = 0
    last_frame = None

    for _ in range(FPS):
        frame = env.render()
        frame = add_label(frame, title, f"start | goal distance {initial_dist:.3f}")
        write_rgb(writers, frame)
        last_frame = frame

    while not (terminated or truncated) and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        total_reward += float(reward)
        final_dist = goal_distance(obs)
        success = bool(info.get("is_success", False))

        detail = (
            f"step {steps} | reward {total_reward:.2f} | "
            f"goal distance {final_dist:.3f} | success {success}"
        )
        frame = env.render()
        frame = add_label(frame, title, detail)

        for _ in range(5):
            write_rgb(writers, frame)

        last_frame = frame

    finish_detail = (
        f"finish | steps {steps} | return {total_reward:.2f} | "
        f"distance {final_dist:.3f} | success {success}"
    )
    for _ in range(2 * FPS):
        frame = env.render()
        frame = add_label(frame, title, finish_detail)
        write_rgb(writers, frame)
        last_frame = frame

    preview_frames.append((title, last_frame))
    env.close()


def save_preview_sheet(preview_frames, output_path):
    thumbs = []
    thumb_w, thumb_h = 420, 236

    for title, frame in preview_frames:
        if frame is None:
            continue
        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, thumb_h - 46), (thumb_w, thumb_h), (18, 24, 31), thickness=-1)
        cv2.putText(
            thumb,
            title[:42],
            (12, thumb_h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)

    if not thumbs:
        return

    cols = 3
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 245, dtype=np.uint8)

    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        y0 = row * thumb_h
        x0 = col * thumb_w
        sheet[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def render_clip(name, render_func, combined_writer, preview_frames):
    path = OUT / f"{name}.mp4"
    print("Rendering:", path, flush=True)
    writer = open_writer(path)
    render_func([writer, combined_writer], preview_frames)
    writer.release()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hopper-seconds", type=int, default=10)
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    combined_path = OUT / "all_policy_renders.mp4"
    combined_writer = open_writer(combined_path)
    preview_frames = []
    written = []

    written.append(
        render_clip(
            "part1_required_actor_critic_hopper",
            lambda writers, previews: render_hopper_actor_critic(
                writers,
                previews,
                args.hopper_seconds,
            ),
            combined_writer,
            preview_frames,
        )
    )

    written.append(
        render_clip(
            "part1_extra_ppo_hopper",
            lambda writers, previews: render_hopper_ppo(
                writers,
                previews,
                args.hopper_seconds,
            ),
            combined_writer,
            preview_frames,
        )
    )

    push_specs = [
        (
            "part2_ppo_source_to_source",
            "Part 2 PPO: source to source",
            ROOT / "part2" / "models" / "ppo_push_none_source_200k_lr0p0001_g0p95_n1024_b64_fixed_seed42.zip",
            "source",
            9000,
        ),
        (
            "part2_ppo_source_to_target",
            "Part 2 PPO: source to target",
            ROOT / "part2" / "models" / "ppo_push_none_source_200k_lr0p0001_g0p95_n1024_b64_fixed_seed42.zip",
            "target",
            9000,
        ),
        (
            "part2_ppo_target_to_target",
            "Part 2 PPO: target to target",
            ROOT / "part2" / "models" / "ppo_push_none_target_200k_lr0p0001_g0p95_n1024_b64_fixed_seed42.zip",
            "target",
            9000,
        ),
        (
            "part2_ppo_udr_target",
            "Part 2 PPO UDR: source to target",
            ROOT / "part2" / "models" / "ppo_push_udr_source_200k_lr0p0001_g0p95_n1024_b64_m0p5-5p0_seed0.zip",
            "target",
            9000,
        ),
        (
            "part2_ppo_adr_target",
            "Part 2 PPO ADR: source to target",
            ROOT / "part2" / "models" / "ppo_push_adr_source_200k_lr0p0001_g0p95_n1024_b64_init0p8-1p2_lim0p5-5p0_step0p3_thr0p25_win10_seed0.zip",
            "target",
            9000,
        ),
        (
            "part2_sac_her_source_to_source",
            "Part 2 SAC+HER: source to source",
            ROOT / "part2" / "models" / "sac_her_push_source_500k_seed42.zip",
            "source",
            9000,
        ),
        (
            "part2_sac_her_source_to_target",
            "Part 2 SAC+HER: source to target",
            ROOT / "part2" / "models" / "sac_her_push_source_500k_seed42.zip",
            "target",
            9000,
        ),
        (
            "part2_sac_her_target_to_target",
            "Part 2 SAC+HER: target to target",
            ROOT / "part2" / "models" / "sac_her_push_target_500k_seed42.zip",
            "target",
            9000,
        ),
        (
            "part2_sac_her_udr_target",
            "Part 2 SAC+HER UDR: source to target",
            ROOT / "part2" / "models" / "sac_her_push_udr_source_500k_m0p5-5p0_seed42.zip",
            "target",
            9000,
        ),
        (
            "part2_sac_her_adr_target",
            "Part 2 SAC+HER ADR: source to target",
            ROOT / "part2" / "models" / "sac_her_push_adr_source_500k_init0p8-1p2_lim0p5-5p0_step0p3_thr0p25_win10_seed42.zip",
            "target",
            9000,
        ),
    ]

    for name, title, model_path, env_type, seed in push_specs:
        if not model_path.exists():
            print("Skipping missing model:", model_path)
            continue

        written.append(
            render_clip(
                name,
                lambda writers, previews, title=title, model_path=model_path, env_type=env_type, seed=seed: render_push_clip(
                    writers,
                    previews,
                    title,
                    model_path,
                    env_type,
                    seed,
                ),
                combined_writer,
                preview_frames,
            )
        )

    combined_writer.release()
    save_preview_sheet(preview_frames, OUT / "all_policy_renders_preview.png")

    print("\nSaved individual videos:")
    for path in written:
        print(path)

    print("\nSaved combined video:", combined_path)
    print("Saved preview sheet:", OUT / "all_policy_renders_preview.png")


if __name__ == "__main__":
    main()
