"""
Generate all plots for the PPO vs SAC safety comparison study.

Produces:
1. Learning curves (episodic return vs steps) with seed traces
2. Action smoothness comparison
3. Constraint violation curves (Safety-Gym)
4. Sample efficiency comparison
5. Reward-safety tradeoff scatter plots
6. Seed variance box plots
7. Policy entropy evolution
8. Perturbation recovery comparison
9. Gradient norm evolution
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ENVIRONMENTS, RESULTS_DIR, PLOTS_DIR, N_SEEDS

# ── Style Setup ───────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "font.family": "serif",
})

COLORS = {"ppo": "#2196F3", "sac": "#FF5722", "random": "#9E9E9E"}
LABELS = {"ppo": "PPO", "sac": "SAC", "random": "Random"}


def load_training_metrics(env_name: str, algo: str, seed: int,
                          tag: str = "best") -> Optional[dict]:
    """Load training metrics JSON for a specific run."""
    from src.config import get_result_dir
    result_dir = get_result_dir(env_name, algo, seed, tag)
    path = os.path.join(result_dir, "logs", "training_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_eval_results(env_name: str, algo: str, seed: int,
                      tag: str = "best") -> Optional[dict]:
    """Load evaluation results JSON."""
    from src.config import get_result_dir
    result_dir = get_result_dir(env_name, algo, seed, tag)
    path = os.path.join(result_dir, "eval_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_eval_log(env_name: str, algo: str, seed: int,
                  tag: str = "best") -> Optional[dict]:
    """Load SB3 evaluation log (evaluations.npz)."""
    from src.config import get_result_dir
    result_dir = get_result_dir(env_name, algo, seed, tag)
    path = os.path.join(result_dir, "eval_logs", "evaluations.npz")
    if os.path.exists(path):
        data = np.load(path)
        return {
            "timesteps": data["timesteps"],
            "results": data["results"],  # (n_evals, n_episodes)
            "ep_lengths": data["ep_lengths"],
        }
    return None


def smooth(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply simple moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


# ── Plot 1: Learning Curves ──────────────────────────────────────────────────
def plot_learning_curves(env_name: str, save_dir: str):
    """Plot episodic return vs training steps with all seed traces."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for algo in ["ppo", "sac"]:
        all_timesteps = []
        all_returns = []

        for seed in range(N_SEEDS):
            data = load_eval_log(env_name, algo, seed)
            if data is None:
                continue

            timesteps = data["timesteps"]
            mean_returns = data["results"].mean(axis=1)

            # Plot individual seed trace (faint)
            ax.plot(timesteps, mean_returns,
                    color=COLORS[algo], alpha=0.15, linewidth=0.8)

            all_timesteps.append(timesteps)
            all_returns.append(mean_returns)

        # Plot bold mean curve
        if all_returns:
            # Interpolate to common x-axis
            min_len = min(len(r) for r in all_returns)
            common_returns = np.array([r[:min_len] for r in all_returns])
            common_ts = all_timesteps[0][:min_len]
            mean_curve = common_returns.mean(axis=0)
            se_curve = common_returns.std(axis=0) / np.sqrt(len(all_returns))

            ax.plot(common_ts, mean_curve,
                    color=COLORS[algo], linewidth=2.5, label=LABELS[algo])
            ax.fill_between(common_ts, mean_curve - se_curve, mean_curve + se_curve,
                           color=COLORS[algo], alpha=0.2)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Average Episodic Return")
    ax.set_title(f"Learning Curves: PPO vs SAC on {env_name}")
    ax.legend(fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"learning_curves_{env_name.lower().replace('-','_')}.png"))
    plt.close()


# ── Plot 2: Action Smoothness ────────────────────────────────────────────────
def plot_action_smoothness(env_name: str, save_dir: str):
    """Box plot comparing action smoothness between PPO and SAC."""
    fig, ax = plt.subplots(figsize=(8, 6))
    data_dict = {}

    for algo in ["ppo", "sac"]:
        smoothness_values = []
        for seed in range(N_SEEDS):
            result = load_eval_results(env_name, algo, seed)
            if result and "per_episode_smoothness" in result:
                smoothness_values.extend(result["per_episode_smoothness"])
            elif result:
                smoothness_values.append(result["mean_action_smoothness"])
        data_dict[LABELS[algo]] = smoothness_values

    if data_dict:
        positions = list(range(len(data_dict)))
        bp = ax.boxplot(data_dict.values(), labels=data_dict.keys(),
                       patch_artist=True, showmeans=True)
        for i, (patch, algo) in enumerate(zip(bp["boxes"], ["ppo", "sac"])):
            patch.set_facecolor(COLORS[algo])
            patch.set_alpha(0.5)

    ax.set_ylabel("Action Smoothness (lower = smoother)")
    ax.set_title(f"Action Smoothness: PPO vs SAC on {env_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"action_smoothness_{env_name.lower().replace('-','_')}.png"))
    plt.close()


# ── Plot 3: Constraint Violations (Safety-Gym only) ─────────────────────────
def plot_constraint_violations(env_name: str, save_dir: str):
    """Plot constraint violations/costs during training (Safety-Gym)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: violations over training
    ax = axes[0]
    for algo in ["ppo", "sac"]:
        all_costs = []
        for seed in range(N_SEEDS):
            metrics = load_training_metrics(env_name, algo, seed)
            if metrics and metrics.get("episode_costs"):
                all_costs.append(metrics["episode_costs"])

        if all_costs:
            min_len = min(len(c) for c in all_costs)
            costs_arr = np.array([c[:min_len] for c in all_costs])
            mean_costs = costs_arr.mean(axis=0)
            smoothed = smooth(mean_costs, window=20)
            ax.plot(smoothed, color=COLORS[algo], linewidth=2, label=LABELS[algo])

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Cost")
    ax.set_title(f"Constraint Violations During Training")
    ax.legend()

    # Right: feasibility rate comparison
    ax = axes[1]
    feas_data = {}
    for algo in ["ppo", "sac"]:
        rates = []
        for seed in range(N_SEEDS):
            result = load_eval_results(env_name, algo, seed)
            if result and "feasibility_rate" in result:
                rates.append(result["feasibility_rate"])
        if rates:
            feas_data[LABELS[algo]] = rates

    if feas_data:
        x = range(len(feas_data))
        means = [np.mean(v) for v in feas_data.values()]
        stds = [np.std(v) for v in feas_data.values()]
        bars = ax.bar(x, means, yerr=stds,
                     color=[COLORS[a] for a in ["ppo", "sac"]][:len(feas_data)],
                     alpha=0.7, capsize=5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(feas_data.keys())
    ax.set_ylabel("Feasibility Rate")
    ax.set_title("Feasibility Rate (% episodes with 0 violations)")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"constraints_{env_name.lower().replace('-','_')}.png"))
    plt.close()


# ── Plot 4: Sample Efficiency ────────────────────────────────────────────────
def plot_sample_efficiency(save_dir: str):
    """Bar chart: steps to reach return threshold across environments."""
    fig, ax = plt.subplots(figsize=(12, 6))

    env_list = [e for e in ENVIRONMENTS if ENVIRONMENTS[e].category == "mujoco"]
    thresholds = {"Hopper-v4": 1500, "HalfCheetah-v4": 3000}

    x = np.arange(len(env_list))
    width = 0.35

    for i, algo in enumerate(["ppo", "sac"]):
        steps_list = []
        for env_name in env_list:
            threshold = thresholds.get(env_name, 1000)
            seed_steps = []
            for seed in range(N_SEEDS):
                data = load_eval_log(env_name, algo, seed)
                if data is not None:
                    mean_returns = data["results"].mean(axis=1)
                    reached = np.where(mean_returns >= threshold)[0]
                    if len(reached) > 0:
                        seed_steps.append(data["timesteps"][reached[0]])
            steps_list.append(np.mean(seed_steps) if seed_steps else np.nan)

        ax.bar(x + i * width, steps_list, width,
               label=LABELS[algo], color=COLORS[algo], alpha=0.7)

    ax.set_xlabel("Environment")
    ax.set_ylabel("Steps to Threshold")
    ax.set_title("Sample Efficiency: Steps to Reach Return Threshold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(env_list)
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sample_efficiency.png"))
    plt.close()


# ── Plot 5: Reward-Safety Tradeoff ───────────────────────────────────────────
def plot_reward_safety_tradeoff(env_name: str, save_dir: str):
    """Scatter: return vs constraint violations for Safety-Gym."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for algo in ["ppo", "sac"]:
        returns_all = []
        costs_all = []
        for seed in range(N_SEEDS):
            result = load_eval_results(env_name, algo, seed)
            if result:
                if "per_episode_returns" in result and "per_episode_costs" in result:
                    returns_all.extend(result["per_episode_returns"])
                    costs_all.extend(result["per_episode_costs"])
                else:
                    returns_all.append(result["mean_return"])
                    costs_all.append(result["mean_cost"])

        if returns_all:
            ax.scatter(costs_all, returns_all,
                      c=COLORS[algo], alpha=0.3, s=20, label=LABELS[algo])
            # Add mean marker
            ax.scatter(np.mean(costs_all), np.mean(returns_all),
                      c=COLORS[algo], s=200, marker="*", edgecolors="black",
                      linewidths=1.5, zorder=5)

    ax.set_xlabel("Episode Cost (Constraint Violations)")
    ax.set_ylabel("Episode Return")
    ax.set_title(f"Reward-Safety Tradeoff on {env_name}")
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"reward_safety_{env_name.lower().replace('-','_')}.png"))
    plt.close()


# ── Plot 6: Seed Variance (Training Stability) ──────────────────────────────
def plot_seed_variance(env_name: str, save_dir: str):
    """Box plot of final returns across seeds to show training stability."""
    fig, ax = plt.subplots(figsize=(8, 6))
    data = {}

    for algo in ["ppo", "sac", "random"]:
        returns = []
        for seed in range(N_SEEDS):
            result = load_eval_results(env_name, algo, seed)
            if result:
                returns.append(result["mean_return"])

        # Random baseline
        if algo == "random":
            env_config = ENVIRONMENTS[env_name]
            env_slug = env_name.lower().replace("-", "_")
            rp = os.path.join(RESULTS_DIR, env_config.category, env_slug,
                             "random", "eval_results.json")
            if os.path.exists(rp):
                with open(rp) as f:
                    rdata = json.load(f)
                returns = rdata.get("per_episode_returns", [rdata["mean_return"]])

        if returns:
            data[LABELS[algo]] = returns

    if data:
        bp = ax.boxplot(data.values(), labels=data.keys(),
                       patch_artist=True, showmeans=True)
        colors_list = [COLORS.get(a, "#9E9E9E") for a in
                      ["ppo", "sac", "random"][:len(data)]]
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

    ax.set_ylabel("Mean Return")
    ax.set_title(f"Training Stability: Return Distribution on {env_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"seed_variance_{env_name.lower().replace('-','_')}.png"))
    plt.close()


# ── Plot 7: Policy Entropy Evolution ─────────────────────────────────────────
def plot_entropy_evolution(env_name: str, save_dir: str):
    """Plot entropy of policy distribution during training."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in ["ppo", "sac"]:
        for seed in range(N_SEEDS):
            metrics = load_training_metrics(env_name, algo, seed)
            if metrics and "gradient_norms" in metrics:
                gn = metrics["gradient_norms"]
                if gn:
                    smoothed = smooth(np.array(gn), window=5)
                    ax.plot(smoothed, color=COLORS[algo], alpha=0.2, linewidth=0.8)

        # Plot mean
        all_gn = []
        for seed in range(N_SEEDS):
            metrics = load_training_metrics(env_name, algo, seed)
            if metrics and metrics.get("gradient_norms"):
                all_gn.append(metrics["gradient_norms"])

        if all_gn:
            min_len = min(len(g) for g in all_gn)
            gn_arr = np.array([g[:min_len] for g in all_gn])
            mean_gn = gn_arr.mean(axis=0)
            smoothed = smooth(mean_gn, window=5)
            ax.plot(smoothed, color=COLORS[algo], linewidth=2.5, label=f"{LABELS[algo]}")

    ax.set_xlabel("Update Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"Gradient Norm Evolution on {env_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"gradient_norms_{env_name.lower().replace('-','_')}.png"))
    plt.close()


# ── Plot 8: Perturbation Recovery ────────────────────────────────────────────
def plot_recovery(save_dir: str):
    """Bar chart: perturbation recovery ratio across environments."""
    fig, ax = plt.subplots(figsize=(10, 6))

    env_list = list(ENVIRONMENTS.keys())
    x = np.arange(len(env_list))
    width = 0.35

    for i, algo in enumerate(["ppo", "sac"]):
        recovery_list = []
        err_list = []
        for env_name in env_list:
            recoveries = []
            for seed in range(N_SEEDS):
                result = load_eval_results(env_name, algo, seed)
                if result and "mean_recovery_ratio" in result:
                    recoveries.append(result["mean_recovery_ratio"])
            recovery_list.append(np.mean(recoveries) if recoveries else 0)
            err_list.append(np.std(recoveries) / np.sqrt(len(recoveries))
                           if len(recoveries) > 1 else 0)

        ax.bar(x + i * width, recovery_list, width, yerr=err_list,
               label=LABELS[algo], color=COLORS[algo], alpha=0.7, capsize=5)

    ax.set_xlabel("Environment")
    ax.set_ylabel("Recovery Ratio (post/pre perturbation)")
    ax.set_title("Deployment Robustness: Recovery After State Perturbation")
    env_short = [e.replace("-v4", "").replace("-v0", "") for e in env_list]
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(env_short, rotation=15)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect recovery")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "recovery_comparison.png"))
    plt.close()


# ── Plot 9: Comprehensive Summary Figure ─────────────────────────────────────
def plot_summary_table(save_dir: str):
    """Create a visual summary table of all metrics."""
    import pandas as pd

    rows = []
    for env_name in ENVIRONMENTS:
        for algo in ["ppo", "sac"]:
            returns, smoothness_vals, costs, recovery_vals = [], [], [], []
            for seed in range(N_SEEDS):
                result = load_eval_results(env_name, algo, seed)
                if result:
                    returns.append(result["mean_return"])
                    smoothness_vals.append(result["mean_action_smoothness"])
                    costs.append(result.get("mean_cost", 0))
                    recovery_vals.append(result.get("mean_recovery_ratio", 0))

            if returns:
                rows.append({
                    "Environment": env_name,
                    "Algorithm": LABELS[algo],
                    "Return": f"{np.mean(returns):.1f} ± {np.std(returns)/np.sqrt(len(returns)):.1f}",
                    "Smoothness": f"{np.mean(smoothness_vals):.4f}",
                    "Cost": f"{np.mean(costs):.2f}",
                    "Recovery": f"{np.mean(recovery_vals):.3f}",
                })

    if rows:
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(14, max(3, len(rows) * 0.6 + 1)))
        ax.axis("off")
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Color header
        for j in range(len(df.columns)):
            table[(0, j)].set_facecolor("#4CAF50")
            table[(0, j)].set_text_props(color="white", fontweight="bold")

        plt.title("Summary: PPO vs SAC Safety Comparison", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "summary_table.png"))
        plt.close()

        # Also save as CSV
        df.to_csv(os.path.join(save_dir, "summary_table.csv"), index=False)


def generate_all_plots():
    """Generate all plots."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Generating plots in {PLOTS_DIR}...")

    for env_name in ENVIRONMENTS:
        print(f"\n  Processing {env_name}...")
        plot_learning_curves(env_name, PLOTS_DIR)
        plot_action_smoothness(env_name, PLOTS_DIR)
        plot_seed_variance(env_name, PLOTS_DIR)
        plot_entropy_evolution(env_name, PLOTS_DIR)

        if ENVIRONMENTS[env_name].category == "safety":
            plot_constraint_violations(env_name, PLOTS_DIR)
            plot_reward_safety_tradeoff(env_name, PLOTS_DIR)

    plot_sample_efficiency(PLOTS_DIR)
    plot_recovery(PLOTS_DIR)
    plot_summary_table(PLOTS_DIR)
    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    generate_all_plots()
