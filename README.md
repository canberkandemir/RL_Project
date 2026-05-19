# FAIML Reinforcement Learning Project

Course project for FAIML - 01VSDWS.

## Getting Started

```bash
pip install -r requirements.txt
cd part2/panda-gym
pip install -e .
```

## Part 1: Hopper-v4

Required from-scratch experiments:

```bash
cd part1
python test_random_policy.py
python train.py
python evaluate_required_actor_critic_hopper.py
python render_required_actor_critic_hopper.py
```

Extra PPO visual demo, kept separate from the required algorithms:

```bash
cd part1
python train_ppo_hopper.py
python render_ppo_hopper.py
```

Main outputs:

- `part1/part1_summary.csv`
- `part1/part1_comparison_learning_curve.png`
- `part1/best_actor_critic_policy.pth`
- `part1/models/ppo_hopper_500k.zip`

## Part 2: PandaPush-v3

PPO/SAC training with fixed, UDR, or ADR cube mass:

```bash
cd part2
python train_sb3.py --algo ppo --env-type source --sampling-strategy none --timesteps 200000
python train_sb3.py --algo ppo --env-type source --sampling-strategy udr --timesteps 200000
python train_sb3.py --algo ppo --env-type source --sampling-strategy adr --timesteps 200000
python eval_sb3.py --model-path models/MODEL_NAME.zip --env-type target --episodes 50 --save-csv
python plot_results.py
```

Stronger SAC+HER PandaPush policy:

```bash
cd part2
python train_sac_her_pandapush.py --env-type source --timesteps 500000
python train_sac_her_pandapush.py --env-type target --timesteps 500000
python render_sac_her_clean_demo.py
python save_sac_her_clean_demo_video.py
```

Final SAC+HER domain-randomization run:

```bash
cd part2
python run_sac_her_dr_experiments.py
```

Main outputs:

- `part2/results/eval_results.csv`
- `part2/results/domain_randomization_target_barplot.png`
- `part2/results/mean_return_barplot.png`
- `part2/results/mass_robustness_curve.png`
- `part2/results/sac_her_lower_upper_success_barplot.png`
- `part2/results/pandapush_centered_demo_seed280.mp4`

## Report Assets

After experiments finish, collect report-ready tables, plots, and demo frames:

```bash
python make_report_assets.py
```

The PNG files are written to `report_assets/`, including summary tables,
comparison plots, and start/finish frames from the PandaPush demo video.

## Project Structure

```text
FAIML-RL-26/
├── README.md
├── make_report_assets.py
├── requirements.txt
├── part1/
│   ├── agent.py
│   ├── test_random_policy.py
│   ├── train.py
│   ├── evaluate_required_actor_critic_hopper.py
│   ├── render_required_actor_critic_hopper.py
│   └── render_ppo_hopper.py
└── part2/
    ├── eval_sb3.py
    ├── plot_results.py
    ├── rand_wrapper.py
    ├── render_sac_her_clean_demo.py
    ├── run_sac_her_dr_experiments.py
    ├── save_sac_her_clean_demo_video.py
    ├── train_sac_her_pandapush.py
    ├── train_sb3.py
    └── panda-gym/
```
