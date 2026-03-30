import gymnasium
import safety_gymnasium
import torch
from stable_baselines3 import PPO

print(f"PyTorch version: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

env1 = gymnasium.make("Hopper-v4")
env1.reset()
env1.step(env1.action_space.sample())
print("Hopper-v4 OK")

env2 = safety_gymnasium.make("SafetyPointGoal1-v0")
env2.reset()
env2.step(env2.action_space.sample())
print("SafetyPointGoal1-v0 OK")
