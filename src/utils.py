"""
Utility functions: seeding, GPU setup, logging, and custom callbacks.
"""

import os
import json
import random
import logging
from typing import Optional, Dict, List, Any

import numpy as np
import torch
import gymnasium as gym

class SafetyToGymWrapper(gym.Wrapper):
    """
    Converts safety-gymnasium's native 6-value step() to standard gym 5-value step().
    Stable-Baselines3 crashes if step() returns 6 values. We safely pack the custom
    `cost` value into the `info` dictionary where our custom metrics tracker can find it.
    """
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info['cost'] = cost
        return obs, reward, terminated, truncated, info

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


def setup_gpu(preferred_gpu: int = 1) -> torch.device:
    """Set up GPU device, preferring the specified GPU index."""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if preferred_gpu < n_gpus:
            device = torch.device(f"cuda:{preferred_gpu}")
        else:
            device = torch.device("cuda:0")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logging.warning("No GPU available, using CPU")
    return device


def set_seed(seed: int):
    """Set deterministic seeds across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For reproducibility (may impact performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    """Set up logging to console and optionally to file."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def make_env(env_name: str, seed: int, rank: int = 0,
             log_dir: Optional[str] = None, is_safety: bool = False):
    """Create a function that returns a seeded, monitored environment."""
    def _init():
        if is_safety:
            import safety_gymnasium
            env = safety_gymnasium.make(env_name)
            env = SafetyToGymWrapper(env)
        else:
            env = gym.make(env_name)
        env.reset(seed=seed + rank)
        if log_dir:
            env = Monitor(env, os.path.join(log_dir, f"rank_{rank}"))
        return env
    return _init


def make_vec_env(env_name: str, n_envs: int, seed: int,
                 log_dir: Optional[str] = None,
                 is_safety: bool = False):
    """Create vectorized environments (SubprocVecEnv for n>1)."""
    env_fns = [
        make_env(env_name, seed, rank=i, log_dir=log_dir, is_safety=is_safety)
        for i in range(n_envs)
    ]
    if n_envs > 1:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


class MetricsCallback(BaseCallback):
    """
    Custom callback to log detailed training metrics:
    - Episode returns
    - Action smoothness (L2 norm of consecutive action differences)
    - Gradient norms (tracked per update)
    - Policy entropy (from info dict)
    """

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_actions: List[List[np.ndarray]] = []
        self.action_smoothness: List[float] = []
        self.gradient_norms: List[float] = []
        self.timesteps_log: List[int] = []
        self.episode_costs: List[float] = []  # For safety envs
        self._current_actions: List[np.ndarray] = []
        self._current_cost: float = 0.0

    def _on_step(self) -> bool:
        # Collect actions for smoothness calculation
        if self.locals.get("actions") is not None:
            actions = self.locals["actions"]
            if isinstance(actions, np.ndarray):
                self._current_actions.append(actions.copy().flatten())

        # Check for episode completion
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_returns.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps_log.append(self.num_timesteps)

                # Compute action smoothness for completed episode
                if len(self._current_actions) > 1:
                    actions_arr = np.array(self._current_actions)
                    diffs = np.diff(actions_arr, axis=0)
                    smoothness = np.mean(np.linalg.norm(diffs, axis=1))
                    self.action_smoothness.append(float(smoothness))
                else:
                    self.action_smoothness.append(0.0)
                self._current_actions = []

                # Track costs for safety environments
                if "cost" in info:
                    self.episode_costs.append(info["cost"])
                    self._current_cost = 0.0

        return True

    def _on_rollout_end(self) -> None:
        """Track gradient norms after each update."""
        if hasattr(self.model, "policy"):
            total_norm = 0.0
            n_params = 0
            for p in self.model.policy.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
                    n_params += 1
            if n_params > 0:
                total_norm = total_norm ** 0.5
                self.gradient_norms.append(total_norm)

    def _on_training_end(self) -> None:
        """Save all collected metrics to JSON."""
        metrics = {
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "action_smoothness": self.action_smoothness,
            "gradient_norms": self.gradient_norms,
            "timesteps": self.timesteps_log,
            "episode_costs": self.episode_costs,
        }
        filepath = os.path.join(self.log_dir, "training_metrics.json")
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
        if self.verbose:
            logging.info(f"Saved training metrics to {filepath}")


def save_config(config: Dict[str, Any], filepath: str):
    """Save experiment configuration to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2, default=str)
