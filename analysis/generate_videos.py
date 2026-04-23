import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import imageio
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import get_result_dir

def record_video(env_name, algo, seed, output_path, is_safety=False):
    print(f"Recording {algo.upper()} on {env_name} (Seed {seed})...")
    result_dir = get_result_dir(env_name, algo, seed, "best")
    model_path = os.path.join(result_dir, "final_model.zip")
    
    if not os.path.exists(model_path):
        model_path = os.path.join(result_dir, "best_model.zip")
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Skipping.")
            return

    if algo == "ppo":
        model = PPO.load(model_path)
    else:
        model = SAC.load(model_path)

    if is_safety:
        import safety_gymnasium
        from src.utils import SafetyToGymWrapper
        # Use rgb_array to capture pixels
        env = safety_gymnasium.make(env_name, render_mode="rgb_array", camera_name="track")
        env = SafetyToGymWrapper(env)
    else:
        env = gym.make(env_name, render_mode="rgb_array")
        
    obs, info = env.reset(seed=seed)
    frames = []
    
    try:
        # Capture first frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        done = False
        step = 0
        while not done and step < 1000:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            step += 1
            
        if not frames:
            print(f"Empty frames list for {env_name}. Renderer might be broken.")
            return
            
        # Save video
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Using imageio to save high quality mp4
        imageio.mimsave(output_path, frames, fps=30, macro_block_size=None)
        print(f"Saved video to {output_path} ({step} frames)")
        
    except Exception as e:
        print(f"Error rendering {env_name}: {e}")
    finally:
        env.close()

def find_best_seed(env_name, algo):
    best_seed = 0
    best_return = -np.inf
    # Search up to Seed 12 (covers standard 10-seed and expanded 12-seed runs)
    for seed in range(13): 
        res_dir = get_result_dir(env_name, algo, seed, "best")
        eval_path = os.path.join(res_dir, "eval_logs", "evaluations.npz")
        if os.path.exists(eval_path):
            data = np.load(eval_path)
            mean_returns = data["results"].mean(axis=1)
            if len(mean_returns) > 0 and mean_returns[-1] > best_return:
                best_return = mean_returns[-1]
                best_seed = seed
    return best_seed

if __name__ == "__main__":
    vid_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots", "videos")
    os.makedirs(vid_dir, exist_ok=True)
    
    # Generate for Hopper and HalfCheetah
    for env_name in ["Hopper-v4", "HalfCheetah-v4"]:
        for algo in ["ppo", "sac"]:
            best_seed = find_best_seed(env_name, algo)
            out_path = os.path.join(vid_dir, f"{env_name.lower()}_{algo.lower()}.mp4")
            record_video(env_name, algo, best_seed, out_path, is_safety=False)
            
    # For safety env
    for env_name in ["SafetyPointGoal1-v0"]:
        for algo in ["ppo", "sac"]:
            best_seed = find_best_seed(env_name, algo)
            out_path = os.path.join(vid_dir, f"{env_name.lower()}_{algo.lower()}.mp4")
            record_video(env_name, algo, best_seed, out_path, is_safety=True)
