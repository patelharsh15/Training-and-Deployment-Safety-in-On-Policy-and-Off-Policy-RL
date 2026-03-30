"""
Custom safety and stability metrics for PPO vs SAC comparison.

Metrics implemented:
1. Action Smoothness: L2 norm of consecutive action differences
2. Policy Entropy: Entropy of the policy distribution
3. Constraint Violations: Episode cost count (Safety-Gym)
4. Feasibility Rate: % episodes with zero violations
5. Recovery Metric: Performance after state perturbation
6. Q-value Variance: (SAC) Variance of Q-function estimates
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import gymnasium as gym


def compute_action_smoothness(actions: np.ndarray) -> float:
    """
    Compute action smoothness as mean L2 norm of consecutive action differences.
    Lower values = smoother policy.

    Args:
        actions: Array of shape (T, action_dim) for one episode
    Returns:
        Mean L2 norm of action differences
    """
    if len(actions) < 2:
        return 0.0
    diffs = np.diff(actions, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    return float(np.mean(norms))


def compute_action_magnitude(actions: np.ndarray) -> float:
    """Mean L2 norm of actions (measures how aggressive the policy is)."""
    return float(np.mean(np.linalg.norm(actions, axis=1)))


def compute_policy_entropy(model, observations: np.ndarray) -> float:
    """
    Compute the mean entropy of the policy distribution over observations.

    Args:
        model: SB3 model (PPO or SAC)
        observations: Array of observations to evaluate
    Returns:
        Mean entropy value
    """
    obs_tensor = torch.FloatTensor(observations).to(model.device)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        entropy = dist.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(dim=-1)
    return float(entropy.mean().cpu().numpy())


def compute_q_value_variance(model, observations: np.ndarray,
                             actions: np.ndarray) -> Optional[float]:
    """
    Compute variance of Q-value estimates (SAC only, has two Q-networks).

    Args:
        model: SB3 SAC model
        observations: Observation array
        actions: Action array
    Returns:
        Mean variance of Q1 vs Q2 estimates, or None if not SAC
    """
    if not hasattr(model, "critic"):
        return None

    obs_tensor = torch.FloatTensor(observations).to(model.device)
    act_tensor = torch.FloatTensor(actions).to(model.device)

    with torch.no_grad():
        q1, q2 = model.critic(obs_tensor, act_tensor)
        variance = ((q1 - q2) ** 2).mean()
    return float(variance.cpu().numpy())


def _step_env(env, action, is_safety: bool = False):
    """Step environment, handling both standard (5-val) and safety-gym (6-val) returns."""
    result = env.step(action)
    if len(result) == 6:  # Safety-Gym: obs, reward, cost, terminated, truncated, info
        obs, reward, cost, terminated, truncated, info = result
        info["cost"] = cost
    else:  # Standard Gym: obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = result
    return obs, reward, terminated, truncated, info


def evaluate_episode(model, env, deterministic: bool = True,
                     is_safety: bool = False) -> Dict:
    """
    Run one episode and collect detailed metrics.

    Returns dict with: return, length, actions, observations, costs, smoothness
    """
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    total_cost = 0.0
    actions_list = []
    observations_list = []
    step_count = 0

    while not done:
        observations_list.append(obs.copy())
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = _step_env(env, action, is_safety)
        done = terminated or truncated
        total_reward += reward
        total_cost += info.get("cost", 0.0)
        actions_list.append(action.copy())
        step_count += 1

    actions_arr = np.array(actions_list)
    observations_arr = np.array(observations_list)

    return {
        "return": total_reward,
        "length": step_count,
        "cost": total_cost,
        "actions": actions_arr,
        "observations": observations_arr,
        "action_smoothness": compute_action_smoothness(actions_arr),
        "action_magnitude": compute_action_magnitude(actions_arr),
    }


def evaluate_with_perturbation(model, env, perturbation_scale: float = 0.1,
                               perturbation_step: int = 50,
                               deterministic: bool = True,
                               is_safety: bool = False) -> Dict:
    """
    Run episode and inject state perturbation mid-episode to measure recovery.

    Args:
        model: Trained SB3 model
        env: Gymnasium environment
        perturbation_scale: Scale of additive Gaussian noise
        perturbation_step: Step at which to inject perturbation
    Returns:
        Dict with pre/post perturbation rewards and recovery metric
    """
    obs, info = env.reset()
    done = False
    pre_rewards = []
    post_rewards = []
    step_count = 0
    perturbed = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = _step_env(env, action, is_safety)
        done = terminated or truncated
        step_count += 1

        if step_count < perturbation_step:
            pre_rewards.append(reward)
        else:
            if not perturbed:
                # Inject noise into observation (simulating state perturbation)
                noise = np.random.normal(0, perturbation_scale, obs.shape)
                obs = obs + noise
                perturbed = True
            post_rewards.append(reward)

    pre_mean = np.mean(pre_rewards) if pre_rewards else 0.0
    post_mean = np.mean(post_rewards) if post_rewards else 0.0

    # Recovery ratio: how well does post-perturbation performance compare to pre
    recovery = post_mean / (pre_mean + 1e-8) if pre_mean != 0 else 1.0

    return {
        "pre_perturbation_return": float(np.sum(pre_rewards)),
        "post_perturbation_return": float(np.sum(post_rewards)),
        "pre_perturbation_mean_reward": float(pre_mean),
        "post_perturbation_mean_reward": float(post_mean),
        "recovery_ratio": float(recovery),
        "total_return": float(np.sum(pre_rewards) + np.sum(post_rewards)),
        "episode_length": step_count,
    }


def compute_feasibility_rate(episode_costs: List[float],
                             threshold: float = 0.0) -> float:
    """
    Compute % of episodes with cost <= threshold.

    Args:
        episode_costs: List of per-episode cumulative costs
        threshold: Maximum acceptable cost
    Returns:
        Fraction of feasible episodes [0, 1]
    """
    if not episode_costs:
        return 1.0
    feasible = sum(1 for c in episode_costs if c <= threshold)
    return feasible / len(episode_costs)


def full_evaluation(model, env_name: str, n_episodes: int = 100,
                    is_safety: bool = False,
                    perturbation_episodes: int = 20) -> Dict:
    """
    Run comprehensive evaluation collecting all metrics.

    Args:
        model: Trained SB3 model
        env_name: Environment name
        n_episodes: Number of evaluation episodes
        is_safety: Whether this is a safety environment
        perturbation_episodes: Number of perturbation test episodes
    Returns:
        Dict with all aggregated metrics
    """
    if is_safety:
        import safety_gymnasium
        env = safety_gymnasium.make(env_name)
    else:
        env = gym.make(env_name)

    # Standard evaluation
    episode_results = []
    all_observations = []
    all_actions = []

    for ep in range(n_episodes):
        result = evaluate_episode(model, env, deterministic=True, is_safety=is_safety)
        episode_results.append(result)
        # Collect subset of obs/actions for entropy and Q-value computation
        if ep < 10:
            all_observations.append(result["observations"][:50])
            all_actions.append(result["actions"][:50])

    # Perturbation evaluation
    perturbation_results = []
    for _ in range(perturbation_episodes):
        p_result = evaluate_with_perturbation(model, env, is_safety=is_safety)
        perturbation_results.append(p_result)

    # Aggregate standard metrics
    returns = [r["return"] for r in episode_results]
    lengths = [r["length"] for r in episode_results]
    smoothness = [r["action_smoothness"] for r in episode_results]
    magnitudes = [r["action_magnitude"] for r in episode_results]
    costs = [r["cost"] for r in episode_results]

    # Compute policy entropy
    if all_observations:
        obs_concat = np.concatenate(all_observations, axis=0)
        try:
            entropy = compute_policy_entropy(model, obs_concat)
        except Exception:
            entropy = None
    else:
        entropy = None

    # Compute Q-value variance (SAC only)
    q_var = None
    if all_observations and all_actions:
        obs_concat = np.concatenate(all_observations, axis=0)
        act_concat = np.concatenate(all_actions, axis=0)
        min_len = min(len(obs_concat), len(act_concat))
        try:
            q_var = compute_q_value_variance(
                model, obs_concat[:min_len], act_concat[:min_len]
            )
        except Exception:
            q_var = None

    # Aggregate perturbation metrics
    recovery_ratios = [r["recovery_ratio"] for r in perturbation_results]

    metrics = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "se_return": float(np.std(returns) / np.sqrt(len(returns))),
        "mean_length": float(np.mean(lengths)),
        "mean_action_smoothness": float(np.mean(smoothness)),
        "std_action_smoothness": float(np.std(smoothness)),
        "mean_action_magnitude": float(np.mean(magnitudes)),
        "mean_cost": float(np.mean(costs)),
        "total_cost": float(np.sum(costs)),
        "feasibility_rate": compute_feasibility_rate(costs),
        "policy_entropy": entropy,
        "q_value_variance": q_var,
        "mean_recovery_ratio": float(np.mean(recovery_ratios)),
        "std_recovery_ratio": float(np.std(recovery_ratios)),
        "n_episodes": n_episodes,
        "per_episode_returns": returns,
        "per_episode_smoothness": smoothness,
        "per_episode_costs": costs,
    }

    env.close()
    return metrics
