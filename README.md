# Training and Deployment Safety in On-Policy and Off-Policy RL

**Course:** CSI5340 / ELG5214 — Reinforcement Learning & Deep RL  
**Team:** Farina Salman, Rachna Sunilkumar Deshpande, Abdulaziz Al-Tayar, Harsh Patel

---

## 🌟 Overview for Beginners: What is this project?

In Reinforcement Learning (RL), an AI "agent" learns how to behave in an environment through trial and error, similar to how a dog learns tricks via treats. Over millions of attempts, the AI learns to map what it sees (States) to what it should do (Actions) to maximize its score (Rewards).

This research project compares the two most powerful, industry-standard AI algorithms in the world today:
1. **PPO (Proximal Policy Optimization):** The algorithm OpenAI uses to train ChatGPT. It learns strictly from its *current* actions and deletes its memory quickly. This makes it very stable and "safe," but it requires massive amounts of data to learn.
2. **SAC (Soft Actor-Critic):** The algorithm most commonly used to train physical robots. It remembers everything in a massive "Replay Buffer" and constantly studies its past mistakes. This makes it learn very fast with less data, but it can be mathematically unstable and prone to erratic physical twitching.

### The Ultimate Question
> **If we drop both of these AI algorithms into a physics simulator with the exact same budget, which one will learn to control a robot safer and smoother?**

---

## 🎮 The Environments (The Tests)

We are testing the algorithms across three distinct physical challenges (virtual video games):

### 1. `Hopper-v4` (The Balancing Act)
* **The Robot:** A 2D, one-legged stick-figure robot.
* **The Goal:** Learn to balance and hop forward as fast as possible.
* **The Challenge:** It is incredibly unstable. If it leans too far, it instantly falls over and "dies." The AI must learn perfect motor timing to stay upright. We use this to test **Basic Training Stability**.

### 2. `HalfCheetah-v4` (The Sprint)
* **The Robot:** A 2D, two-legged robot.
* **The Goal:** Coordinate front and back legs to sprint forward as fast as physically possible.
* **The Challenge:** Unlike the Hopper, the Cheetah cannot fall over and die. The entire challenge is pure coordination. We use this to test **Convergence Speed (who learns faster)**.

### 3. `SafetyPointGoal1-v0` (The Minefield)
* **The Robot:** A 3D point robot navigating a walled arena.
* **The Goal:** Drive to green "Goal" circles. When a goal is reached, a new one spawns.
* **The Challenge (Safety):** The arena is covered in red "Hazard" circles. Touching a hazard triggers a "Cost" penalty. The AI must learn to navigate to the goals without touching the hazards. We use this to test **Deployment Safety and Constraint Adherence**.

---

## 📊 How We Measure Success (The Metrics)

Instead of just looking at who got the highest score, we custom-built "Safety Metrics" to look under the hood:

- **Action Smoothness:** Are the robot's motors moving fluidly, or twitching violently? (Measured by the difference between consecutive actions).
- **Constraint Violations:** How many times did the AI touch a red hazard in the Safety environment?
- **Policy Entropy:** How "random" is the AI behaving?
- **Recovery Ratio:** If we artificially push the robot (perturbation), does it immediately crash, or does the AI know how to recover?
- **Gradient Norms:** How stable is the underlying math in the neural network during training?

---

## 🚀 Quick Start Guide

Want to run these massive experiments yourself? 

```bash
# 1. Activate the Python environment
source venv/bin/activate

# 2. Run a 10-second Smoke Test to verify GPU access
CUDA_VISIBLE_DEVICES=1 python src/train_mujoco.py --env Hopper-v4 --algo ppo --seed 0 --total-timesteps 10000

# 3. Run the FULL 60-Seed Master Pipeline (Takes ~3 Days on RTX 3070)
# This will automatically train PPO and SAC across all 3 environments, 10 times each!
CUDA_VISIBLE_DEVICES=1 python src/run_all_experiments.py --skip-hp

# 4. Generate all the graphs and statistical tables
python analysis/generate_plots.py
```

---

## 📁 Project Structure

```
├── src/                            # The Brains
│   ├── config.py                   # Rules of the experiment (hyperparameters, 8-core CPU limits)
│   ├── utils.py                    # Utilities (seeding, logging)
│   ├── metrics.py                  # Our custom Safety/Smoothness calculators
│   ├── train_mujoco.py             # Script that drops AI into Hopper & Cheetah
│   ├── train_safety.py             # Script that drops AI into the Minefield
│   ├── evaluate.py                 # Script that tests the AIs after they are fully trained
│   └── run_all_experiments.py      # The Master Robot that runs all 60 tests automatically
├── analysis/
│   └── generate_plots.py           # Draws the final graphs
├── results/                        # Where the AI brains (.zip files) are saved
├── plots/                          # Generated graphs (PNG/PDF)
├── presentation/                   # HTML slides for final university presentation
├── theory_guide.md                 # Deep-dive theoretical explanation of the thesis
└── README.md
```

## 🔬 Scientific Reproducibility

This project is built to strict academic standards:
- **10 Independent Seeds:** Every experiment is run 10 separate times from scratch to prove the results aren't just "lucky".
- **Absolute Determinism:** NumPy, PyTorch, and Gymnasium seeds are strictly locked.
- **Hardware Profile:** Parallelized 8-core `SubprocVecEnv` simulation batched to an NVIDIA RTX 3070 GPU.
