"""
Training script for PPO and SAC on MuJoCo environments (Hopper-v4, HalfCheetah-v4).

Usage:
    python src/train_mujoco.py --env Hopper-v4 --algo ppo --seed 0
    python src/train_mujoco.py --env HalfCheetah-v4 --algo sac --seed 0
    python src/train_mujoco.py --env Hopper-v4 --algo ppo --seed 0 --total-timesteps 10000  # smoke test
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    ENVIRONMENTS, PREFERRED_GPU, get_result_dir, hp_tag, AlgoConfig
)
from src.utils import (
    setup_gpu, set_seed, setup_logging, make_vec_env, MetricsCallback, save_config
)

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback


def train(env_name: str, algo: str, seed: int,
          learning_rate: float = 3e-4,
          gamma: float = 0.99,
          batch_size: int = 256,
          total_timesteps: int = None,
          tag: str = "best"):
    """
    Train PPO or SAC on a MuJoCo environment.

    Args:
        env_name: Gymnasium environment ID
        algo: 'ppo' or 'sac'
        seed: Random seed
        learning_rate: Learning rate
        gamma: Discount factor
        batch_size: Batch size
        total_timesteps: Override total timesteps (None = use config default)
        tag: Hyperparameter tag for result directory
    """
    env_config = ENVIRONMENTS[env_name]
    if total_timesteps is None:
        total_timesteps = env_config.total_timesteps

    result_dir = get_result_dir(env_name, algo, seed, tag)
    log_dir = os.path.join(result_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    setup_logging(os.path.join(result_dir, "training.log"))
    logging.info(f"{'='*60}")
    logging.info(f"Training {algo.upper()} on {env_name} | Seed {seed}")
    logging.info(f"LR={learning_rate}, γ={gamma}, BS={batch_size}")
    logging.info(f"Total timesteps: {total_timesteps}")
    logging.info(f"Result dir: {result_dir}")
    logging.info(f"{'='*60}")

    # Setup
    device = setup_gpu(PREFERRED_GPU)
    set_seed(seed)

    # Create environments
    n_envs = env_config.n_envs_ppo if algo == "ppo" else env_config.n_envs_sac
    train_env = make_vec_env(env_name, n_envs, seed, log_dir=log_dir)

    # Create eval environment
    eval_env = make_vec_env(env_name, 1, seed + 1000,
                           log_dir=os.path.join(log_dir, "eval"))

    # Policy kwargs (MLP 2x256)
    policy_kwargs = dict(net_arch=[256, 256])

    # Common kwargs
    common_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        gamma=gamma,
        seed=seed,
        verbose=0,
        device="cuda",
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(result_dir, "tb_logs"),
    )

    # Algorithm-specific setup
    if algo == "ppo":
        model = PPO(
            **common_kwargs,
            n_steps=2048,
            batch_size=batch_size,
            n_epochs=10,
            clip_range=0.2,
            ent_coef=0.0,
            max_grad_norm=0.5,
            gae_lambda=0.95,
        )
    elif algo == "sac":
        model = SAC(
            **common_kwargs,
            batch_size=batch_size,
            buffer_size=1_000_000,
            learning_starts=10_000,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Callbacks
    metrics_cb = MetricsCallback(log_dir=log_dir, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=result_dir,
        log_path=os.path.join(result_dir, "eval_logs"),
        eval_freq=max(env_config.eval_freq // n_envs, 1),
        n_eval_episodes=env_config.n_eval_episodes,
        deterministic=True,
        verbose=0,
    )

    # Save experiment config
    exp_config = {
        "env_name": env_name,
        "algo": algo,
        "seed": seed,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "batch_size": batch_size,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "policy_kwargs": policy_kwargs,
        "timestamp": datetime.now().isoformat(),
    }
    save_config(exp_config, os.path.join(result_dir, "config.json"))

    # Train
    logging.info("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[metrics_cb, eval_cb],
        progress_bar=False,
    )

    # Save final model
    model.save(os.path.join(result_dir, "final_model"))
    logging.info(f"Training complete. Model saved to {result_dir}")

    # Cleanup
    train_env.close()
    eval_env.close()

    return result_dir


def main():
    parser = argparse.ArgumentParser(description="Train PPO/SAC on MuJoCo")
    parser.add_argument("--env", type=str, required=True,
                        choices=["Hopper-v4", "HalfCheetah-v4"])
    parser.add_argument("--algo", type=str, required=True,
                        choices=["ppo", "sac"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--tag", type=str, default="best")

    args = parser.parse_args()

    train(
        env_name=args.env,
        algo=args.algo,
        seed=args.seed,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        total_timesteps=args.total_timesteps,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
