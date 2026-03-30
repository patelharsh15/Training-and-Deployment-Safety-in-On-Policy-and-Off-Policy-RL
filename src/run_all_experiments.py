"""
Master script to run all experiments: hyperparameter search + full multi-seed runs.

Usage:
    python src/run_all_experiments.py                    # Run everything
    python src/run_all_experiments.py --hp-search-only   # Only hyperparameter search
    python src/run_all_experiments.py --final-only       # Only final runs (assumes HP search done)
    python src/run_all_experiments.py --env Hopper-v4    # Single environment
    python src/run_all_experiments.py --evaluate-only    # Only evaluate existing models
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from typing import Dict, List, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    ENVIRONMENTS, RESULTS_DIR, N_SEEDS, HP_SEARCH_SEEDS,
    HP_SEARCH_TIMESTEPS_FRACTION, get_hp_configs, hp_tag, get_result_dir
)
from src.utils import setup_logging

import numpy as np


def run_training_cmd(env_name: str, algo: str, seed: int,
                     lr: float, gamma: float, bs: int,
                     total_timesteps: int, tag: str,
                     is_safety: bool = False):
    """Run a single training process."""
    script = "src/train_safety.py" if is_safety else "src/train_mujoco.py"
    cmd = [
        sys.executable, script,
        "--env", env_name,
        "--algo", algo,
        "--seed", str(seed),
        "--lr", str(lr),
        "--gamma", str(gamma),
        "--batch-size", str(bs),
        "--total-timesteps", str(total_timesteps),
        "--tag", tag,
    ]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logging.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Training failed: {result.stderr[-500:]}")
    return result.returncode


def run_hp_search(env_name: str, algo: str):
    """
    Run hyperparameter search for a given env/algo combination.
    Uses reduced seeds and timesteps.
    """
    env_config = ENVIRONMENTS[env_name]
    is_safety = env_config.category == "safety"
    hp_timesteps = int(env_config.total_timesteps * HP_SEARCH_TIMESTEPS_FRACTION)
    configs = get_hp_configs(algo)

    logging.info(f"\n{'='*60}")
    logging.info(f"HP Search: {algo.upper()} on {env_name}")
    logging.info(f"Configs: {len(configs)}, Seeds: {HP_SEARCH_SEEDS}, Steps: {hp_timesteps}")
    logging.info(f"{'='*60}\n")

    for config in configs:
        tag = hp_tag(config.learning_rate, config.gamma, config.batch_size)
        for seed in range(HP_SEARCH_SEEDS):
            run_training_cmd(
                env_name, algo, seed,
                config.learning_rate, config.gamma, config.batch_size,
                hp_timesteps, tag, is_safety
            )


def select_best_hp(env_name: str, algo: str) -> Tuple[float, float, int]:
    """Select best hyperparameters based on mean eval return across seeds."""
    env_config = ENVIRONMENTS[env_name]
    env_slug = env_name.lower().replace("-", "_")
    configs = get_hp_configs(algo)
    best_return = -np.inf
    best_config = (3e-4, 0.99, 256)  # Defaults

    for config in configs:
        tag = hp_tag(config.learning_rate, config.gamma, config.batch_size)
        returns = []

        for seed in range(HP_SEARCH_SEEDS):
            result_dir = get_result_dir(env_name, algo, seed, tag)
            eval_file = os.path.join(result_dir, "eval_logs", "evaluations.npz")

            if os.path.exists(eval_file):
                data = np.load(eval_file)
                mean_returns = data["results"].mean(axis=1)
                returns.append(mean_returns[-1] if len(mean_returns) > 0 else -np.inf)

        if returns:
            mean_return = np.mean(returns)
            logging.info(f"  {tag}: mean_return={mean_return:.2f} (±{np.std(returns):.2f})")
            if mean_return > best_return:
                best_return = mean_return
                best_config = (config.learning_rate, config.gamma, config.batch_size)

    logging.info(f"  Best config: lr={best_config[0]}, γ={best_config[1]}, bs={best_config[2]}")
    logging.info(f"  Best return: {best_return:.2f}")

    # Save selection
    selection = {
        "env_name": env_name,
        "algo": algo,
        "best_lr": best_config[0],
        "best_gamma": best_config[1],
        "best_batch_size": best_config[2],
        "best_mean_return": float(best_return),
        "timestamp": datetime.now().isoformat(),
    }
    save_path = os.path.join(
        RESULTS_DIR, env_config.category, env_slug, algo, "best_hp.json"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(selection, f, indent=2)

    return best_config


def run_final_experiments(env_name: str, algo: str,
                         lr: float, gamma: float, bs: int):
    """Run final experiments with best hyperparameters across all seeds."""
    env_config = ENVIRONMENTS[env_name]
    is_safety = env_config.category == "safety"

    logging.info(f"\n{'='*60}")
    logging.info(f"Final Runs: {algo.upper()} on {env_name}")
    logging.info(f"lr={lr}, γ={gamma}, bs={bs}, Seeds: {N_SEEDS}")
    logging.info(f"{'='*60}\n")

    for seed in range(N_SEEDS):
        run_training_cmd(
            env_name, algo, seed,
            lr, gamma, bs,
            env_config.total_timesteps, "best", is_safety
        )


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--env", type=str, default=None,
                        help="Run only for this environment")
    parser.add_argument("--algo", type=str, default=None,
                        choices=["ppo", "sac"])
    parser.add_argument("--hp-search-only", action="store_true")
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--skip-hp", action="store_true",
                        help="Skip HP search, use defaults")

    args = parser.parse_args()

    setup_logging(os.path.join(RESULTS_DIR, "experiment_log.txt"))

    envs = [args.env] if args.env else list(ENVIRONMENTS.keys())
    algos = [args.algo] if args.algo else ["ppo", "sac"]

    if args.evaluate_only:
        from src.evaluate import evaluate_all
        evaluate_all()
        return

    for env_name in envs:
        for algo in algos:
            if not args.final_only:
                if args.skip_hp:
                    logging.info(f"Skipping HP search, using defaults for {algo}/{env_name}")
                else:
                    # Phase 1: Hyperparameter search
                    run_hp_search(env_name, algo)

            if not args.hp_search_only:
                # Phase 2: Select best HP (or use defaults)
                if args.skip_hp:
                    best_lr, best_gamma, best_bs = 3e-4, 0.99, 256
                else:
                    best_lr, best_gamma, best_bs = select_best_hp(env_name, algo)

                # Phase 3: Final runs with best HP
                run_final_experiments(env_name, algo,
                                     best_lr, best_gamma, best_bs)

    logging.info("\n" + "="*60)
    logging.info("All experiments complete!")
    logging.info("="*60)

    # Auto-evaluate
    if not args.hp_search_only:
        logging.info("Starting evaluation of all models...")
        from src.evaluate import evaluate_all
        evaluate_all()


if __name__ == "__main__":
    main()
