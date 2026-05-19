import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = "results/eval_results.csv"
OUT_DIR = "results"


def clean_label(name):
    if "sac_her_udr" in name:
        return "SAC+HER UDR [0.5, 5.0]"
    if "sac_her_adr" in name:
        return "SAC+HER ADR adaptive"
    if "sac_her_source_to_source" in name:
        return "SAC+HER source->source"
    if "sac_her_source_to_target" in name:
        return "SAC+HER source->target"
    if "sac_her_target_to_target" in name:
        return "SAC+HER target->target"
    if "ppo_none_source_to_source" in name:
        return "PPO source->source"
    if "ppo_none_source_to_target" in name:
        return "PPO source->target"
    if "ppo_none_target_to_target" in name:
        return "PPO target->target"
    if "udr_v2" in name:
        return "PPO UDR [0.8, 3.0]"
    if "udr_v3" in name:
        return "PPO UDR [0.5, 5.0]"
    if "adr_v2" in name:
        return "PPO ADR adaptive"
    return name


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    df["success_rate"] = pd.to_numeric(df["success_rate"], errors="coerce")
    df["mean_return"] = pd.to_numeric(df["mean_return"], errors="coerce")
    df["eval_mass"] = pd.to_numeric(df["eval_mass"], errors="coerce")

    # =========================
    # 1. Target success bar plot
    # =========================

    target_df = df[
        (df["env_type"] == "target") &
        (df["eval_mass"].isna()) &
        (df["experiment_name"].str.contains("udr_v2|udr_v3|adr_v2|sac_her_udr|sac_her_adr", regex=True))
    ].copy()

    target_df["label"] = target_df["experiment_name"].apply(clean_label)

    target_summary = (
        target_df
        .groupby("label", as_index=False)
        .agg(
            success_rate=("success_rate", "mean"),
            mean_return=("mean_return", "mean"),
        )
    )

    print("\n=== Target summary ===")
    print(target_summary)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(target_summary["label"], target_summary["success_rate"] * 100)
    plt.ylabel("Success rate (%)")
    plt.xlabel("Method")
    plt.title("Domain Randomization: Target Success Rate")
    plt.bar_label(bars, fmt="%.1f%%", padding=3)
    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("results/domain_randomization_target_barplot.png", dpi=300)
    plt.close()

    # =========================
    # 2. Mean return bar plot
    # =========================

    plt.figure(figsize=(8, 5))
    plt.bar(target_summary["label"], target_summary["mean_return"])
    plt.ylabel("Mean return")
    plt.xlabel("Method")
    plt.title("Domain Randomization: Target Mean Return")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("results/mean_return_barplot.png", dpi=300)
    plt.close()

    # =========================
    # 3. Mass robustness curve
    # =========================

    mass_df = df[
        (df["eval_mass"].notna()) &
        (df["experiment_name"].str.contains("udr_v2|udr_v3|adr_v2|sac_her_udr|sac_her_adr", regex=True))
    ].copy()

    mass_df["label"] = mass_df["experiment_name"].apply(clean_label)

    mass_summary = (
        mass_df
        .groupby(["label", "eval_mass"], as_index=False)
        .agg(success_rate=("success_rate", "mean"))
    )

    print("\n=== Mass robustness summary ===")
    print(mass_summary)

    plt.figure(figsize=(8, 5))

    for label in mass_summary["label"].unique():
        sub = mass_summary[mass_summary["label"] == label].sort_values("eval_mass")
        plt.plot(
            sub["eval_mass"],
            sub["success_rate"] * 100,
            marker="o",
            label=label,
        )

    plt.xlabel("Object mass (kg)")
    plt.ylabel("Success rate (%)")
    plt.title("Robustness Across Object Masses")
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/mass_robustness_curve.png", dpi=300)
    plt.close()

    # =========================
    # 4. SAC+HER lower/upper bounds
    # =========================

    sac_her_df = df[
        df["experiment_name"].str.contains(
            "sac_her_source_to_source|sac_her_source_to_target|sac_her_target_to_target",
            regex=True,
        )
    ].copy()

    if not sac_her_df.empty:
        sac_her_df["label"] = sac_her_df["experiment_name"].apply(clean_label)

        sac_her_order = [
            "SAC+HER source->source",
            "SAC+HER source->target",
            "SAC+HER target->target",
        ]

        sac_her_summary = (
            sac_her_df
            .groupby("label", as_index=False)
            .agg(
                success_rate=("success_rate", "mean"),
                mean_return=("mean_return", "mean"),
            )
        )
        sac_her_summary["label"] = pd.Categorical(
            sac_her_summary["label"],
            categories=sac_her_order,
            ordered=True,
        )
        sac_her_summary = sac_her_summary.sort_values("label")

        print("\n=== SAC+HER lower/upper-bound summary ===")
        print(sac_her_summary)

        plt.figure(figsize=(8, 5))
        bars = plt.bar(
            sac_her_summary["label"].astype(str),
            sac_her_summary["success_rate"] * 100,
        )
        plt.ylabel("Success rate (%)")
        plt.xlabel("Training -> test configuration")
        plt.title("SAC+HER Lower and Upper Bounds")
        plt.bar_label(bars, fmt="%.1f%%", padding=3)
        plt.ylim(0, 105)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig("results/sac_her_lower_upper_success_barplot.png", dpi=300)
        plt.close()

    print("\nSaved plots:")
    print("results/domain_randomization_target_barplot.png")
    print("results/mean_return_barplot.png")
    print("results/mass_robustness_curve.png")
    print("results/sac_her_lower_upper_success_barplot.png")


if __name__ == "__main__":
    main()
