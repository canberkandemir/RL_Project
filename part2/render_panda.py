"""
Render a trained PandaPush-v3 policy and save start+finish composite images.
Produces two files: default angle and top-down angle.
Filenames follow the pattern: part2_{model_abbrev}_target_seed{N}_start_finish[_topdown].png

Usage:
    python render_panda.py --model-path models/sac_push_udr_source_200k_....zip \
        --env-type target --seed 53 --out-dir renders
"""

import argparse
import os
import re

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import panda_gym  # registers PandaPush-v3
from stable_baselines3 import PPO, SAC


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(model_path, env=None):
    lower = model_path.lower()
    if "ppo" in lower:
        return PPO.load(model_path)
    if "sac" in lower:
        return SAC.load(model_path, env=env)
    raise ValueError("Cannot infer algorithm — path must contain 'ppo' or 'sac'.")


def set_object_mass(env, mass):
    sim = env.unwrapped.task.sim
    body_id = sim._bodies_idx["object"]
    sim.physics_client.changeDynamics(bodyUniqueId=body_id, linkIndex=-1, mass=float(mass))


def parse_model_name(model_path):
    """
    Extract readable config info and a short file-safe name from the model filename.
    Returns (file_prefix, title_lines).
    """
    stem = os.path.splitext(os.path.basename(model_path))[0]
    # e.g. sac_push_udr_source_200k_lr0p0003_g0p95_buf100000_b256_m0p5-5p0_seed0

    algo   = "SAC" if stem.startswith("sac") else "PPO"
    strat  = ("UDR" if "_udr_" in stem else
              "ADR" if "_adr_" in stem else "None")
    env_tr = "target" if "_none_target_" in stem else "source"
    steps  = re.search(r"_(\d+k)_", stem)
    steps  = steps.group(1).upper() if steps else "?"

    # mass range for UDR
    mass_m = re.search(r"m([\d]+p[\d]+)-([\d]+p[\d]+)", stem)
    mass_str = ""
    if mass_m:
        lo = mass_m.group(1).replace("p", ".")
        hi = mass_m.group(2).replace("p", ".")
        mass_str = f"[{lo}, {hi}] kg"

    # ADR params
    step_m = re.search(r"step([\d]+p[\d]+)", stem)
    thr_m  = re.search(r"thr([\d]+p[\d]+)", stem)
    win_m  = re.search(r"win(\d+)", stem)
    adr_str = ""
    if step_m and thr_m and win_m:
        step_v = step_m.group(1).replace("p", ".")
        thr_v  = thr_m.group(1).replace("p", ".")
        win_v  = win_m.group(1)
        adr_str = f"step={step_v}  thr={thr_v}  win={win_v}"

    # learning rate
    lr_m = re.search(r"lr([\d]+p[\d]+)", stem)
    lr_str = lr_m.group(1).replace("p", ".e-") if lr_m else ""
    # fix: lr0p0003 → 0.0003
    lr_m2 = re.search(r"lr(0p\d+)", stem)
    if lr_m2:
        lr_str = lr_m2.group(1).replace("p", ".")

    # gamma
    g_m = re.search(r"g(0p\d+)", stem)
    g_str = g_m.group(1).replace("p", ".") if g_m else ""

    # build file-safe prefix:  sac_udr_source_200k  (no lr/gamma/buf noise)
    short_parts = [algo.lower()]
    if strat != "None":
        short_parts.append(strat.lower())
    else:
        short_parts.append("none")
    short_parts.append(env_tr)
    if steps != "?":
        short_parts.append(steps.lower())
    if mass_str:
        mlo = mass_m.group(1).replace("p", "p")
        mhi = mass_m.group(2).replace("p", "p")
        short_parts.append(f"m{mlo}-{mhi}")
    if adr_str:
        short_parts.append(f"thr{thr_m.group(1)}")
    file_prefix = "_".join(short_parts)

    # build title lines
    line1 = f"{algo} | Strategy: {strat} | Train: {env_tr} | Timesteps: {steps}"
    line2_parts = []
    if lr_str:
        line2_parts.append(f"lr={lr_str}")
    if g_str:
        line2_parts.append(f"γ={g_str}")
    if mass_str:
        line2_parts.append(f"UDR range: {mass_str}")
    if adr_str:
        line2_parts.append(f"ADR: {adr_str}")
    line2 = "  |  ".join(line2_parts) if line2_parts else ""

    return file_prefix, [line1, line2] if line2 else [line1]


# ── camera ───────────────────────────────────────────────────────────────────

def capture_default(env):
    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None.")
    return np.array(frame, dtype=np.uint8)


def capture_topdown(env, width=640, height=480):
    client = env.unwrapped.task.sim.physics_client

    # Angled top-down: same direction as default but higher & more overhead
    # Shows table + cube + goal clearly
    eye    = [0.0, -0.7, 1.6]
    target = [0.0,  0.2, 0.4]
    up     = [0.0,  0.0, 1.0]

    view_matrix = client.computeViewMatrix(eye, target, up)
    proj_matrix = client.computeProjectionMatrixFOV(
        fov=35, aspect=float(width) / height, nearVal=0.1, farVal=5.0
    )
    _, _, rgba, _, _ = client.getCameraImage(
        width=width, height=height,
        viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=client.ER_TINY_RENDERER,
    )
    return np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


# ── composite ─────────────────────────────────────────────────────────────────

def make_composite(start_frame, end_frame, title_lines,
                   label_start="Start", label_end="Success"):
    h, w = start_frame.shape[:2]
    gap      = 12
    label_h  = 30
    title_h  = 28 * len(title_lines) + 10
    total_w  = w * 2 + gap
    total_h  = title_h + label_h + h

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 20, 20))
    draw   = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = font_title

    # title lines (centred)
    for i, line in enumerate(title_lines):
        draw.text((total_w // 2, 8 + i * 28), line,
                  fill=(220, 220, 220), anchor="mt", font=font_title)

    # frame labels
    y_label = title_h + 5
    draw.text((w // 2,           y_label), label_start,
              fill=(255, 255, 255), anchor="mt", font=font_label)
    draw.text((w + gap + w // 2, y_label), label_end,
              fill=(100, 255, 100) if label_end == "Success" else (255, 255, 255),
              anchor="mt", font=font_label)

    # paste frames
    y_frame = title_h + label_h
    canvas.paste(Image.fromarray(start_frame), (0,       y_frame))
    canvas.paste(Image.fromarray(end_frame),   (w + gap, y_frame))

    return canvas


# ── main render loop ──────────────────────────────────────────────────────────

def run(model_path, env_type, seed, eval_mass, out_dir, n_episodes):
    file_prefix, title_lines = parse_model_name(model_path)

    env   = gym.make("PandaPush-v3", render_mode="rgb_array",
                     type=env_type, reward_type="dense")
    model = load_model(model_path, env=env)

    os.makedirs(out_dir, exist_ok=True)

    best = dict(start_def=None, end_def=None,
                start_top=None, end_top=None, success=False)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        if eval_mass is not None:
            set_object_mass(env, eval_mass)

        s_def = capture_default(env)
        s_top = capture_topdown(env)

        terminated = truncated = False
        e_def, e_top = s_def, s_top
        success = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            e_def = capture_default(env)
            e_top = capture_topdown(env)

        if isinstance(info, dict):
            success = bool(info.get("is_success", False))

        print(f"  ep {ep+1:02d} | success={success}")

        if success and not best["success"]:
            best.update(start_def=s_def, end_def=e_def,
                        start_top=s_top, end_top=e_top, success=True)
            break
        if not best["success"]:
            best.update(start_def=s_def, end_def=e_def,
                        start_top=s_top, end_top=e_top)

    env.close()

    end_label = "Success" if best["success"] else "End"
    base_name = f"part2_{file_prefix}_{env_type}_seed{seed}"

    # default angle
    p_def = os.path.join(out_dir, f"{base_name}_start_finish.png")
    make_composite(best["start_def"], best["end_def"],
                   title_lines, "Start", end_label).save(p_def)
    print(f"  → {p_def}")

    # top-down angle
    p_top = os.path.join(out_dir, f"{base_name}_start_finish_topdown.png")
    make_composite(best["start_top"], best["end_top"],
                   title_lines, "Start", end_label).save(p_top)
    print(f"  → {p_top}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--env-type",   default="target", choices=["source", "target"])
    parser.add_argument("--seed",       type=int,   default=53)
    parser.add_argument("--eval-mass",  type=float, default=None)
    parser.add_argument("--out-dir",    type=str,   default="renders")
    parser.add_argument("--episodes",   type=int,   default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(model_path=args.model_path, env_type=args.env_type,
        seed=args.seed, eval_mass=args.eval_mass,
        out_dir=args.out_dir, n_episodes=args.episodes)
