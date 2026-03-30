"""
Evaluation script: loads trained models and runs comprehensive evaluation.

Usage:
    python src/evaluate.py --env Hopper-v4 --algo ppo --seed 0
    python src/evaluate.py --env Hopper-v4 --algo ppo --seed 0 --tag lr3e-4_g0.99_bs256
    python src/evaluate.py --all  # Evaluate all completed experiments
"""

import os
import sys
import json
import argparse
import logging
import glob
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ENVIRONMENTS, RESULTS_DIR, get_result_dir
from src.metrics import full_evaluation
from src.utils import setup_logging, set_seed

from stable_baselines3 import PPO, SAC
import numpy as np


ALGO_MAP = {"ppo": PPO, "sac": SAC}


def evaluate_single(env_name: str, algo: str, seed: int,
                    tag: str = "best",
                    n_episodes: int = 100,
                    model_name: str = "best_model") -> Optional[dict]:
    """Evaluate a single trained model."""
    result_dir = get_result_dir(env_name, algo, seed, tag)
    model_path = os.path.join(result_dir, model_name)

    if not os.path.exists(model_path + ".zip"):
        # Try final_model
        model_path = os.path.join(result_dir, "final_model")
        if not os.path.exists(model_path + ".zip"):
            logging.warning(f"No model found at {result_dir}")
            return None

    env_config = ENVIRONMENTS[env_name]
    is_safety = env_config.category == "safety"

    logging.info(f"Evaluating {algo.upper()} on {env_name} | Seed {seed} | Tag: {tag}")

    set_seed(seed)
    AlgoClass = ALGO_MAP[algo]
    model = AlgoClass.load(model_path, device="cuda")

    metrics = full_evaluation(
        model, env_name,
        n_episodes=n_episodes,
        is_safety=is_safety,
        perturbation_episodes=20,
    )

    # Save evaluation results
    eval_path = os.path.join(result_dir, "eval_results.json")
    # Convert any non-serializable types
    save_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (list, float, int, str, type(None))):
            save_metrics[k] = v
        elif isinstance(v, np.floating):
            save_metrics[k] = float(v)
        else:
            save_metrics[k] = str(v)

    with open(eval_path, "w") as f:
        json.dump(save_metrics, f, indent=2)

    logging.info(f"  Return: {metrics['mean_return']:.2f} ± {metrics['se_return']:.2f}")
    logging.info(f"  Smoothness: {metrics['mean_action_smoothness']:.4f}")
    logging.info(f"  Recovery: {metrics['mean_recovery_ratio']:.4f}")
    if is_safety:
        logging.info(f"  Mean Cost: {metrics['mean_cost']:.2f}")
        logging.info(f"  Feasibility: {metrics['feasibility_rate']:.2%}")

    return metrics


def evaluate_random_baseline(env_name: str, n_episodes: int = 100) -> dict:
    """Evaluate random policy baseline."""
    import gymnasium as gym
    env_config = ENVIRONMENTS[env_name]
    is_safety = env_config.category == "safety"

    if is_safety:
        import safety_gymnasium
        env = safety_gymnasium.make(env_name)
    else:
        env = gym.make(env_name)

    returns, lengths, costs, smoothness_list = [], [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_cost = 0.0
        actions = []
        while not done:
            action = env.action_space.sample()
            actions.append(action)
            result = env.step(action)
            if len(result) == 6:  # Safety-Gym
                obs, reward, cost, terminated, truncated, info = result
                ep_cost += cost
            else:
                obs, reward, terminated, truncated, info = result
                ep_cost += info.get("cost", 0.0)
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)
        lengths.append(len(actions))
        costs.append(ep_cost)

        # Action smoothness
        if len(actions) > 1:
            actions_arr = np.array(actions)
            diffs = np.diff(actions_arr, axis=0)
            sm = float(np.mean(np.linalg.norm(diffs, axis=1)))
            smoothness_list.append(sm)

    env.close()

    result = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "se_return": float(np.std(returns) / np.sqrt(len(returns))),
        "mean_length": float(np.mean(lengths)),
        "mean_action_smoothness": float(np.mean(smoothness_list)) if smoothness_list else 0.0,
        "mean_cost": float(np.mean(costs)),
        "feasibility_rate": sum(1 for c in costs if c <= 0) / len(costs),
        "per_episode_returns": returns,
        "per_episode_costs": costs,
        "n_episodes": n_episodes,
    }

    # Save
    env_slug = env_name.lower().replace("-", "_")
    save_dir = os.path.join(RESULTS_DIR, env_config.category, env_slug, "random")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "eval_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    logging.info(f"Random baseline on {env_name}: Return={result['mean_return']:.2f}")
    return result


def evaluate_all():
    """Find and evaluate all completed training runs."""
    setup_logging()
    for env_name, env_config in ENVIRONMENTS.items():
        # Random baseline
        logging.info(f"\n{'='*60}\nRandom baseline: {env_name}\n{'='*60}")
        evaluate_random_baseline(env_name)

        # Trained models
        for algo in ["ppo", "sac"]:
            for seed in range(10):
                result_dir = get_result_dir(env_name, algo, seed)
                model_path = os.path.join(result_dir, "best_model.zip")
                final_path = os.path.join(result_dir, "final_model.zip")
                if os.path.exists(model_path) or os.path.exists(final_path):
                    evaluate_single(env_name, algo, seed)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None, choices=["ppo", "sac"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="best")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all completed experiments")
    parser.add_argument("--random-only", action="store_true",
                        help="Only evaluate random baselines")

    args = parser.parse_args()

    setup_logging()

    if args.all:
        evaluate_all()
    elif args.random_only:
        for env_name in ENVIRONMENTS:
            evaluate_random_baseline(env_name, args.n_episodes)
    elif args.env and args.algo:
        evaluate_single(args.env, args.algo, args.seed, args.tag, args.n_episodes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
