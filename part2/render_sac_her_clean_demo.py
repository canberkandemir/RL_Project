
"""Render a clean SAC+HER PandaPush visual demo.

This script:
1. Loads the trained SAC+HER target-domain model.
2. Searches for a seed where:
   - the cube starts far enough from the goal,
   - the policy succeeds,
   - the final cube-goal distance is very small.
3. Renders that clean successful episode slowly.
4. Holds the final successful state for a longer time.

This is for visual demonstration only.
The official evaluation should still use eval_sb3.py over many episodes.
"""

import time
import numpy as np

import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC


MODEL_PATH = "models/sac_her_push_target_500k_seed42.zip"
ENV_TYPE = "target"

# The default PandaPush threshold is 0.05 m. That is good for training/eval,
# but it can stop a visual demo when the cube is merely near the target.
SUCCESS_DISTANCE_THRESHOLD = 0.005

# Search settings
PREFERRED_DEMO_SEED = 280
MAX_SEARCH_SEEDS = 1000
MAX_EPISODE_STEPS = 60

# Use full trained action strength.
# Do not reduce this if you want accurate pushing.
ACTION_SCALE = 1.00

# Require visible movement, not a trivial 1-step success.
MIN_INITIAL_DIST = 0.08
MIN_STEPS = 4

# Require clean final placement near the goal center.
FINAL_DIST_MAX = SUCCESS_DISTANCE_THRESHOLD

# Rendering speed during the pushing motion.
# Bigger = slower video.
SLEEP_TIME = 0.12

# Keep final successful state visible.
FINAL_HOLD_SECONDS = 10


def make_env(render=False):
    """Create PandaPush environment."""
    env = gym.make(
        "PandaPush-v3",
        render_mode="human" if render else "rgb_array",
        type=ENV_TYPE,
        reward_type="sparse",
    )
    env.unwrapped.task.distance_threshold = SUCCESS_DISTANCE_THRESHOLD
    return env


def goal_distance(obs):
    """Compute distance between cube/object position and desired goal."""
    achieved = np.array(obs["achieved_goal"], dtype=float)
    desired = np.array(obs["desired_goal"], dtype=float)
    return float(np.linalg.norm(achieved - desired))


def run_episode(model, seed, render=False, env=None):
    """Run one episode and optionally render it."""
    owns_env = env is None
    if env is None:
        env = make_env(render=render)

    obs, info = env.reset(seed=seed)
    init_dist = goal_distance(obs)

    if render:
        print(f"Initial cube-goal distance: {init_dist:.4f}")
        env.render()
        time.sleep(2.0)

    terminated = False
    truncated = False
    steps = 0
    total_reward = 0.0
    final_info = {}
    final_dist = init_dist

    while not (terminated or truncated) and steps < MAX_EPISODE_STEPS:
        action, _ = model.predict(obs, deterministic=True)

        # Full action strength by default.
        # Only change ACTION_SCALE if you intentionally want a slower/weaker policy.
        action = ACTION_SCALE * action

        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        total_reward += float(reward)
        final_info = info
        final_dist = goal_distance(obs)

        if render:
            print(
                f"step={steps:02d} | "
                f"dist={final_dist:.4f} | "
                f"reward={reward} | "
                f"success={info.get('is_success', None)}"
            )
            env.render()
            time.sleep(SLEEP_TIME)

    success = bool(final_info.get("is_success", False))

    if render:
        print(
            f"\nFinished | seed={seed} | success={success} | "
            f"steps={steps} | init_dist={init_dist:.4f} | "
            f"final_dist={final_dist:.4f} | return={total_reward:.3f}"
        )

        print(f"\nHolding final successful position for {FINAL_HOLD_SECONDS} seconds...")

        n_hold_frames = int(FINAL_HOLD_SECONDS / 0.1)
        for _ in range(n_hold_frames):
            env.render()
            time.sleep(0.1)

    if owns_env:
        env.close()

    return success, steps, total_reward, init_dist, final_dist


def main():
    # HER models must be loaded with an environment.
    load_env = make_env(render=False)
    model = SAC.load(MODEL_PATH, env=load_env)

    print("Loaded model:", MODEL_PATH)
    print("Searching for clean visual demo...")
    print("Required conditions:")
    print("  init_dist >=", MIN_INITIAL_DIST)
    print("  final_dist <=", FINAL_DIST_MAX)
    print("  steps >=", MIN_STEPS)
    print("  success_threshold =", SUCCESS_DISTANCE_THRESHOLD)
    print("  action_scale =", ACTION_SCALE)

    success, steps, total_reward, init_dist, final_dist = run_episode(
        model,
        PREFERRED_DEMO_SEED,
        render=False,
    )

    if success and init_dist >= MIN_INITIAL_DIST and steps >= MIN_STEPS and final_dist <= FINAL_DIST_MAX:
        print("\nUsing preferred centered demo seed:", PREFERRED_DEMO_SEED)
        print(
            f"seed={PREFERRED_DEMO_SEED} | steps={steps} | "
            f"init={init_dist:.4f} | final={final_dist:.4f}"
        )
        run_episode(model, PREFERRED_DEMO_SEED, render=True)
        load_env.close()
        return

    best = None
    search_env = make_env(render=False)

    for seed in range(MAX_SEARCH_SEEDS):
        success, steps, total_reward, init_dist, final_dist = run_episode(
            model,
            seed,
            render=False,
            env=search_env,
        )

        if success and init_dist >= MIN_INITIAL_DIST:
            print(
                f"seed={seed:04d} | "
                f"steps={steps:02d} | "
                f"init={init_dist:.4f} | "
                f"final={final_dist:.4f}"
            )

            # Keep best fallback by smallest final distance.
            if best is None or final_dist < best[4]:
                best = (seed, steps, total_reward, init_dist, final_dist)

            if steps >= MIN_STEPS and final_dist <= FINAL_DIST_MAX:
                print("\nFound clean demo seed:", seed)
                search_env.close()
                run_episode(model, seed, render=True)
                load_env.close()
                return

    print("\nNo seed matched the strict condition.")
    search_env.close()

    if best is not None:
        seed, steps, total_reward, init_dist, final_dist = best
        print("Rendering best found:")
        print(
            f"seed={seed} | steps={steps} | "
            f"init={init_dist:.4f} | final={final_dist:.4f}"
        )
        run_episode(model, seed, render=True)
    else:
        print("No successful non-trivial seed found.")

    load_env.close()


if __name__ == "__main__":
    main()
