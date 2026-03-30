"""
Experiment configuration for PPO vs SAC safety comparison.

Hyperparameter search ranges justified by:
- Engstrom et al. (2020): Implementation Matters in Deep RL
- Andrychowicz et al. (2021): What Matters in On-Policy RL
- Haarnoja et al. (2018): SAC original paper defaults
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from itertools import product


# ── GPU Configuration ─────────────────────────────────────────────────────────
# Prefer GPU 1 (as instructed), use both if available
PREFERRED_GPU = 1
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(PREFERRED_GPU))

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


@dataclass
class EnvConfig:
    """Environment configuration."""
    name: str               # Gymnasium env id
    category: str           # 'mujoco' or 'safety'
    total_timesteps: int    # Total training steps
    eval_freq: int          # Evaluate every N steps
    n_eval_episodes: int    # Episodes per evaluation
    n_envs_ppo: int         # Vectorized envs for PPO
    n_envs_sac: int         # Vectorized envs for SAC (typically 1)


@dataclass
class AlgoConfig:
    """Algorithm hyperparameter configuration."""
    algo: str                          # 'ppo' or 'sac'
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 256
    policy_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "net_arch": [256, 256],
    })
    # PPO-specific
    n_steps: int = 2048                # Steps per rollout (PPO)
    n_epochs: int = 10                 # PPO epochs per update
    clip_range: float = 0.2            # PPO clip parameter
    ent_coef: float = 0.0              # Entropy coefficient
    max_grad_norm: float = 0.5         # Gradient clipping
    gae_lambda: float = 0.95           # GAE lambda
    # SAC-specific
    buffer_size: int = 1_000_000       # Replay buffer size
    learning_starts: int = 10_000      # Random exploration steps
    tau: float = 0.005                 # Soft update coefficient
    train_freq: int = 1                # Update every N steps
    gradient_steps: int = 1            # Gradient steps per update


# ── Environment Configurations ────────────────────────────────────────────────
ENVIRONMENTS = {
    "Hopper-v4": EnvConfig(
        name="Hopper-v4",
        category="mujoco",
        total_timesteps=1_000_000,
        eval_freq=10_000,
        n_eval_episodes=20,
        n_envs_ppo=8,
        n_envs_sac=8,
    ),
    "HalfCheetah-v4": EnvConfig(
        name="HalfCheetah-v4",
        category="mujoco",
        total_timesteps=1_000_000,
        eval_freq=10_000,
        n_eval_episodes=20,
        n_envs_ppo=8,
        n_envs_sac=8,
    ),
    "SafetyPointGoal1-v0": EnvConfig(
        name="SafetyPointGoal1-v0",
        category="safety",
        total_timesteps=1_000_000,
        eval_freq=10_000,
        n_eval_episodes=20,
        n_envs_ppo=8,
        n_envs_sac=8,
    ),
}


# ── Hyperparameter Search Space ───────────────────────────────────────────────
# Justified by Engstrom et al. (2020) and Andrychowicz et al. (2021)
HYPERPARAM_SEARCH = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "gamma": [0.99, 0.995],
    "batch_size": [256, 1024],
}

# Number of seeds for final experiments
N_SEEDS = 10
SEED_LIST = list(range(N_SEEDS))

# Reduced budget for hyperparameter search (fewer seeds, fewer steps)
HP_SEARCH_SEEDS = 3
HP_SEARCH_TIMESTEPS_FRACTION = 0.3  # Use 30% of total timesteps for HP search


def get_hp_configs(algo: str) -> List[AlgoConfig]:
    """Generate all hyperparameter configurations for grid search."""
    configs = []
    for lr, gamma, bs in product(
        HYPERPARAM_SEARCH["learning_rate"],
        HYPERPARAM_SEARCH["gamma"],
        HYPERPARAM_SEARCH["batch_size"],
    ):
        config = AlgoConfig(
            algo=algo,
            learning_rate=lr,
            gamma=gamma,
            batch_size=bs,
        )
        configs.append(config)
    return configs


def get_result_dir(env_name: str, algo: str, seed: int,
                   hp_tag: str = "best") -> str:
    """Get the result directory for a specific experiment run."""
    env_config = ENVIRONMENTS[env_name]
    env_slug = env_name.lower().replace("-", "_")
    path = os.path.join(RESULTS_DIR, env_config.category, env_slug,
                        algo, hp_tag, f"seed_{seed}")
    os.makedirs(path, exist_ok=True)
    return path


def hp_tag(lr: float, gamma: float, bs: int) -> str:
    """Create a human-readable tag for a hyperparameter config."""
    return f"lr{lr}_g{gamma}_bs{bs}"
