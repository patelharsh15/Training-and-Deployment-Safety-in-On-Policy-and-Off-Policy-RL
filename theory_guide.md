# Project Guide: Theory & Code Execution

This document perfectly breaks down the theoretical concepts behind your research project and exactly how they map to the code we wrote. You can use this as a direct foundation for writing your final report and presenting your slides.

---

## 1. The Big Picture: What are we actually testing?
In Reinforcement Learning (RL), an AI "agent" learns to map **States** (what it sees) to **Actions** (what it does) to maximize **Rewards**. 

Your research compares the two most popular, state-of-the-art RL algorithms in the world today:
1. **PPO (Proximal Policy Optimization):** The algorithm OpenAI uses to train ChatGPT.
2. **SAC (Soft Actor-Critic):** The industry standard for training physical robots.

We are testing them on **Standard environments** (MuJoCo Hopper/Cheetah, where the only goal is to run fast) and **Safety environments** (Safety-Gymnasium, where there are physical constraints/hazards the robot must avoid).

---

## 2. The Core Theory: On-Policy vs Off-Policy

The fundamental difference between PPO and SAC is *how they treat their memories*.

### PPO: On-Policy (The "Live in the Moment" Learner)
* **How it works:** PPO learns strictly from what it is doing *right now*. It plays the game for a few thousand steps, pauses to calculate how to update its brain (gradient descent), and then **completely deletes that data**. 
* **Pros:** Extremely stable. It doesn't accidentally learn bad habits from old data, which is why it's great for "Safety".
* **Cons:** Very "sample inefficient." Because it throws data away, it requires millions of interactions to learn anything.
* **In our Code:** This is why PPO is so fast on the clock. It runs 8 parallel environments, collects data, does a quick math update, and wipes the slate clean (~17 minutes per 1 Million steps).

### SAC: Off-Policy (The "Obsessive Studier" Learner)
* **How it works:** SAC remembers everything. Every single step it takes is saved into a massive database called a **Replay Buffer**. Even while playing new games, it constantly reaches back into this buffer to study thousands of old memories to extract more learning from them. SAC also features "Entropy Maximization"—it actively rewards itself for trying weird, random actions, encouraging massive exploration.
* **Pros:** Extremely "sample efficient." It can learn complex tasks with far fewer physical interactions than PPO.
* **Cons:** Computationally incredibly heavy. Studying the past over and over requires massive GPU mathematics. It can also be unstable (sometimes old memories teach it the wrong lesson for its current stage of learning).
* **In our Code:** This is why SAC originally took 6 hours! It was pausing the game *every single step* to study old memories. We sped it up by telling it to wait 64 steps before studying.

---

## 3. How the Code Works (The Pipeline)

We built an enterprise-grade pipeline to test these theories. Here is what every file does:

#### `src/config.py` (The Blueprint)
Defines the rules of the experiment. It dictates that every run gets exactly 1,000,000 steps, specifies the learning rates, and assigns 8 CPU cores to run physics simulations concurrently.

#### `src/train_mujoco.py` & `src/train_safety.py` (The Gym)
These scripts spawn the physics engines. They initialize either a PPO or SAC brain from `stable-baselines3` and drop it into the environment. As the agent trains, this script constantly saves the `best_model.zip` whenever the agent beats its previous high score.

#### `src/metrics.py` (The Microscope)
This is where the real "Research" happens. Standard RL just tracks "Reward". We track **Safety**:
1. **Action Smoothness:** Are the robot's motors twitching violently, or moving fluidly? PPO is usually smoother; SAC tends to twitch because of its random exploration (Entropy).
2. **Constraint Violations:** How many times did it step on a hazard in Safety-Gym?
3. **Recovery:** If we physically push the robot (perturbation), does the algorithm know how to recover, or does it rely too heavily on perfect memory?

#### `src/run_all_experiments.py` (The Master Orchestrator)
Because RL is highly reliant on random luck (seeds), you cannot run an algorithm just once. You must run it 10 times with 10 different starting points (seeds) to prove it wasn't just a lucky run.
This script automatically loops 60 times (3 envs × 2 algorithms × 10 seeds). It securely tracks what is finished, skips completed runs, and automatically feeds the next job to your GPU so you can sleep while it works.

---

## 4. What will the Results show? (Hypothesis)
When your 60 runs finish and we generate the plots, here is what RL theory expects we will see:

1. **Learning Curves:** SAC will likely spike up and learn how to walk much faster (in fewer env steps) than PPO because it studies past memories.
2. **Safety/Violations:** PPO will likely commit far fewer safety violations. SAC's built-in "Entropy" (randomness) forces it to touch hazards just to see what happens.
3. **Smoothness:** PPO's actions will look smoother. SAC will look mathematically erratic.

By proving this empirically across 60 rigorous tests, your project effectively demonstrates the tradeoff between "Learning Speed" (SAC) and "Deployment Safety" (PPO).
