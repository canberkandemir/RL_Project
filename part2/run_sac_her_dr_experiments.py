import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV = os.environ.copy()
ENV.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".mplconfig"))


def float_to_tag(value: float) -> str:
    return str(value).replace(".", "p")


def run(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=ROOT, env=ENV, check=True)


def model_path(args: argparse.Namespace, strategy: str) -> Path:
    steps = args.timesteps // 1000

    if strategy == "udr":
        name = (
            f"sac_her_push_udr_source_{steps}k_"
            f"m{float_to_tag(args.mass_min)}-{float_to_tag(args.mass_max)}_"
            f"seed{args.seed}.zip"
        )
    elif strategy == "adr":
        name = (
            f"sac_her_push_adr_source_{steps}k_"
            f"init{float_to_tag(args.adr_initial_min)}-"
            f"{float_to_tag(args.adr_initial_max)}_"
            f"lim{float_to_tag(args.mass_min)}-"
            f"{float_to_tag(args.adr_max_limit)}_"
            f"step{float_to_tag(args.adr_step)}_"
            f"thr{float_to_tag(args.adr_success_threshold)}_"
            f"win{args.adr_window_size}_seed{args.seed}.zip"
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return ROOT / "models" / name


def train_command(args: argparse.Namespace, strategy: str) -> list[str]:
    command = [
        sys.executable,
        "train_sac_her_pandapush.py",
        "--env-type",
        "source",
        "--sampling-strategy",
        strategy,
        "--timesteps",
        str(args.timesteps),
        "--seed",
        str(args.seed),
        "--mass-min",
        str(args.mass_min),
        "--mass-max",
        str(args.mass_max),
    ]

    if strategy == "adr":
        command.extend(
            [
                "--adr-initial-min",
                str(args.adr_initial_min),
                "--adr-initial-max",
                str(args.adr_initial_max),
                "--adr-max-limit",
                str(args.adr_max_limit),
                "--adr-step",
                str(args.adr_step),
                "--adr-success-threshold",
                str(args.adr_success_threshold),
                "--adr-window-size",
                str(args.adr_window_size),
            ]
        )

    return command


def eval_command(
    args: argparse.Namespace,
    model: Path,
    experiment_name: str,
    env_type: str,
    eval_mass: float | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "eval_sb3.py",
        "--model-path",
        str(model.relative_to(ROOT)),
        "--episodes",
        str(args.episodes),
        "--env-type",
        env_type,
        "--seed",
        str(args.eval_seed),
        "--save-csv",
        "--experiment-name",
        experiment_name,
    ]

    if eval_mass is not None:
        command.extend(["--eval-mass", str(eval_mass)])

    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate SAC+HER UDR and ADR PandaPush experiments.",
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=9000)
    parser.add_argument("--mass-min", type=float, default=0.5)
    parser.add_argument("--mass-max", type=float, default=5.0)
    parser.add_argument("--adr-initial-min", type=float, default=0.8)
    parser.add_argument("--adr-initial-max", type=float, default=1.2)
    parser.add_argument("--adr-max-limit", type=float, default=5.0)
    parser.add_argument("--adr-step", type=float, default=0.3)
    parser.add_argument("--adr-success-threshold", type=float, default=0.25)
    parser.add_argument("--adr-window-size", type=int, default=10)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even if the expected model file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    strategies = ["udr", "adr"]
    models = {strategy: model_path(args, strategy) for strategy in strategies}

    if not args.skip_training:
        for strategy in strategies:
            if models[strategy].exists() and not args.force_train:
                print(f"Skipping {strategy.upper()} training; found {models[strategy]}")
                continue
            run(train_command(args, strategy))

    if not args.skip_eval:
        for strategy in strategies:
            model = models[strategy]
            if not model.exists():
                raise FileNotFoundError(
                    f"Missing {strategy.upper()} model: {model}. "
                    "Run without --skip-training or lower --timesteps to match "
                    "an existing model filename."
                )

            run(
                eval_command(
                    args,
                    model,
                    f"sac_her_{strategy}_source_{args.episodes}ep",
                    "source",
                )
            )
            run(
                eval_command(
                    args,
                    model,
                    f"sac_her_{strategy}_target_{args.episodes}ep",
                    "target",
                )
            )

            for mass in [1.0, 2.0, 3.0, 4.0, 5.0]:
                run(
                    eval_command(
                        args,
                        model,
                        f"sac_her_{strategy}_mass_{int(mass)}",
                        "source",
                        eval_mass=mass,
                    )
                )

    if not args.skip_plots:
        run([sys.executable, "plot_results.py"])


if __name__ == "__main__":
    main()
