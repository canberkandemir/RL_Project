# From Simulation to Reality: Policy Gradient Methods and Domain Randomization for Reinforcement Learning in Robotics

Course project for FAIML - 01VSDWS.

## Getting Started

```bash
# On macOS (Apple Silicon), install pybullet via conda first:
conda install -c conda-forge pybullet

pip install -r requirements.txt
cd part2/panda-gym
pip install -e .
```

## Part 1: Hopper-v4

Required from-scratch experiments only:

```bash
cd part1
python test_random_policy.py
python train.py                       # all 5 experiments: REINFORCE (4 baselines) + vanilla Actor-Critic
python evaluate_required_actor_critic_hopper.py
python render_required_actor_critic_hopper.py
```

`python train.py` trains all five configurations for 300 episodes (seed 42),
produces the comparison learning curve (Figure 2), and saves the vanilla
Actor-Critic policy used for evaluation and rendering.

Main outputs:

- `part1/part1_summary.csv`
- `part1/part1_comparison_learning_curve.png`
- `part1/best_actor_critic_policy.pth`

## Part 2: PandaPush-v3

Required PPO/SAC training with fixed cube mass (lower/upper bounds):

```bash
cd part2
python train_sb3.py --algo ppo --env-type source --sampling-strategy none --timesteps 200000
python train_sb3.py --algo ppo --env-type target --sampling-strategy none --timesteps 200000
python train_sb3.py --algo sac --env-type source --sampling-strategy none --timesteps 200000
python train_sb3.py --algo sac --env-type target --sampling-strategy none --timesteps 200000
python train_sb3.py --algo sac --env-type target --sampling-strategy none --timesteps 120000 --load-model-path models/sac_push_none_target_260k_lr0p0003_g0p95_buf100000_b256_fixed_seed42.zip --run-name sac_push_none_target_380k_lr0p0003_g0p95_buf100000_b256_fixed_seed42 --quiet
```

Evaluate lower/upper bound models (source→source, source→target, target→target):

```bash
cd part2
python eval_sb3.py --model-path models/ppo_push_none_source_200k_lr0p0001_g0p95_n1024_b64_fixed_seed42.zip --env-type source --episodes 50 --save-csv --experiment-name ppo_none_source_to_source_50ep
python eval_sb3.py --model-path models/ppo_push_none_source_200k_lr0p0001_g0p95_n1024_b64_fixed_seed42.zip --env-type target --episodes 50 --save-csv --experiment-name ppo_none_source_to_target_50ep
python eval_sb3.py --model-path models/ppo_push_none_target_200k_lr0p0001_g0p95_n1024_b64_fixed_seed42.zip --env-type target --episodes 50 --save-csv --experiment-name ppo_none_target_to_target_50ep
python eval_sb3.py --model-path models/sac_push_none_source_200k_lr0p0003_g0p95_buf100000_b256_fixed_seed42.zip --env-type source --episodes 50 --save-csv --experiment-name sac_none_source_to_source_50ep
python eval_sb3.py --model-path models/sac_push_none_source_200k_lr0p0003_g0p95_buf100000_b256_fixed_seed42.zip --env-type target --episodes 50 --save-csv --experiment-name sac_none_source_to_target_50ep
python eval_sb3.py --model-path models/sac_push_none_target_380k_lr0p0003_g0p95_buf100000_b256_fixed_seed42.zip --env-type target --episodes 50 --save-csv --experiment-name sac_none_target_to_target_50ep
```

Domain randomization experiments (UDR and ADR):

```bash
cd part2
python train_sb3.py --algo ppo --env-type source --sampling-strategy udr --timesteps 200000
python train_sb3.py --algo ppo --env-type source --sampling-strategy adr --timesteps 200000
python train_sb3.py --algo sac --env-type source --sampling-strategy udr --timesteps 200000 --mass-min 0.5 --mass-max 5.0
python train_sb3.py --algo sac --env-type source --sampling-strategy adr --timesteps 200000 --mass-min 0.5 --adr-max-limit 5.0 --adr-initial-min 0.8 --adr-initial-max 1.2 --adr-step 0.1 --adr-success-threshold 0.65 --adr-window-size 30
python train_sb3.py --algo sac --env-type source --sampling-strategy udr --timesteps 500000 --mass-min 0.5 --mass-max 5.0 --seed 0 --learning-rate 3e-4 --gamma 0.95
python train_sb3.py --algo sac --env-type source --sampling-strategy adr --timesteps 500000 --mass-min 0.5 --adr-max-limit 5.0 --adr-initial-min 0.8 --adr-initial-max 1.2 --adr-step 0.1 --adr-success-threshold 0.65 --adr-window-size 30
```

Evaluate DR models on target domain + mass robustness sweep:

```bash
cd part2
for MODEL in \
    models/ppo_push_udr_source_200k_lr0p0001_g0p95_n1024_b64_m0p5-5p0_seed0.zip \
    models/ppo_push_udr_source_200k_lr0p0001_g0p95_n1024_b64_m0p8-3p0_seed0.zip \
    models/ppo_push_adr_source_200k_lr0p0001_g0p95_n1024_b64_init0p8-1p2_lim0p5-5p0_step0p3_thr0p25_win10_seed0.zip \
    models/sac_push_udr_source_200k_lr0p0003_g0p95_buf100000_b256_m0p5-5p0_seed0.zip \
    models/sac_push_adr_source_200k_lr0p0003_g0p95_buf100000_b256_init0p8-1p2_lim0p5-5p0_step0p1_thr0p65_win30_seed0.zip \
    models/sac_push_udr_source_500k_lr0p0003_g0p95_buf100000_b256_m0p5-5p0_seed0.zip \
    models/sac_push_adr_source_500k_lr0p0003_g0p95_buf100000_b256_init0p8-1p2_lim0p5-5p0_step0p1_thr0p65_win30_seed0.zip
do
    NAME=$(basename $MODEL .zip)
    python eval_sb3.py --model-path $MODEL --env-type target --episodes 50 --seed 9000 --save-csv --experiment-name ${NAME}_target
    for MASS in 1.0 2.0 3.0 4.0 5.0; do
        python eval_sb3.py --model-path $MODEL --env-type source --eval-mass $MASS --episodes 50 --seed 9000 --save-csv --experiment-name ${NAME}_mass_${MASS}
    done
done
python plot_results.py
```

Optional: render start+finish images for any trained model:

```bash
cd part2
python render_panda.py --model-path models/MODEL_NAME.zip --env-type target --seed 53 --out-dir ../renders
```

Produces two images per model: default angle and zoomed top-down angle, each with a title showing the full training configuration.

Main outputs:

- `part2/results/eval_results.csv`
- `part2/results/domain_randomization_target_barplot.png`
- `part2/results/mean_return_barplot.png`
- `part2/results/mass_robustness_curve.png`
- `part2/results/lower_upper_success_barplot.png`
- `renders/part2_*_start_finish.png` (default angle)
- `renders/part2_*_start_finish_topdown.png` (top-down angle)

## Report Assets

After all experiments finish, collect report-ready tables, plots, and render images:

```bash
cd ..   # back to project root
python make_report_assets.py
```

The PNG files are written to `report_assets/`, including summary tables,
comparison plots, and the required Part 2 result plots.

## Project Structure

```text
FAIML-RL-26/
├── README.md
├── make_report_assets.py
├── requirements.txt
├── renders/                          # start+finish render images (generated)
├── report_assets/                    # report-ready tables, plots, renders (generated)
├── part1/
│   ├── agent.py
│   ├── test_random_policy.py
│   ├── train.py
│   ├── evaluate_required_actor_critic_hopper.py
│   ├── render_required_actor_critic_hopper.py
│   └── render_trained_hopper.py
└── part2/
    ├── eval_sb3.py
    ├── plot_results.py
    ├── rand_wrapper.py
    ├── render_panda.py
    ├── train_sb3.py
    └── panda-gym/
```
