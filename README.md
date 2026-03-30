# Training and Deployment Safety in On-Policy and Off-Policy RL

**Course:** CSI5340 / ELG5214 — Reinforcement Learning & Deep RL  
**Team:** Farina Salman, Rachna Sunilkumar Deshpande, Abdulaziz Al-Tayar, Harsh Patel

## Research Question

> In which environmental conditions will PPO have more stable and safer behavior compared to SAC for both training and deployment, when both have the same compute budgets and evaluation protocols?

## Project Structure

```
├── src/                            # Source code
│   ├── config.py                   # Experiment configurations
│   ├── utils.py                    # Utilities (seeding, GPU, callbacks)
│   ├── metrics.py                  # Custom safety metrics
│   ├── train_mujoco.py             # MuJoCo training (Hopper, HalfCheetah)
│   ├── train_safety.py             # Safety-Gym training
│   ├── evaluate.py                 # Evaluation & metric collection
│   └── run_all_experiments.py      # Master orchestrator
├── analysis/
│   └── generate_plots.py           # Visualization generation
├── results/                        # Experiment outputs
├── plots/                          # Generated figures
├── presentation/                   # Presentation slides
└── README.md
```

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Smoke test (single short run)
CUDA_VISIBLE_DEVICES=1 python src/train_mujoco.py --env Hopper-v4 --algo ppo --seed 0 --total-timesteps 10000

# Run all experiments (HP search + final runs + evaluation)
CUDA_VISIBLE_DEVICES=1 python src/run_all_experiments.py

# Or skip HP search and use defaults
CUDA_VISIBLE_DEVICES=1 python src/run_all_experiments.py --skip-hp

# Evaluate only
python src/evaluate.py --all

# Generate plots
python analysis/generate_plots.py
```

## Environments

| Environment | Type | Purpose |
|---|---|---|
| Hopper-v4 | MuJoCo | Training stability, convergence smoothness |
| HalfCheetah-v4 | MuJoCo | Training stability, convergence smoothness |
| SafetyPointGoal1-v0 | Safety-Gym | Constraint adherence, safety evaluation |

## Metrics

- **Episodic Return**: Standard reward performance
- **Action Smoothness**: L2 norm of consecutive action differences
- **Policy Entropy**: Entropy of policy distribution
- **Constraint Violations**: Per-episode cost (Safety-Gym)
- **Feasibility Rate**: % episodes with zero violations
- **Recovery Ratio**: Performance after state perturbation
- **Gradient Norms**: Training stability indicator
- **Q-value Variance**: Critic uncertainty (SAC only)

## Reproducibility

- 10 independent seeds per condition
- All seeds, hyperparameters, and configs logged as JSON
- Deterministic seeding across NumPy, PyTorch, and environments
- Hardware: Shared GPU server (GPU index 1)
