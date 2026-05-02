import os
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import cv2

class ScoreOverlayWrapper(gym.Wrapper):
    """
    A custom wrapper that uses OpenCV to draw the current score directly onto 
    the video frames before they are saved.
    """
    def __init__(self, env):
        super().__init__(env)
        self.current_score = 0
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_score = info.get("score", 0)
        return obs, reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        self.current_score = 0
        return self.env.reset(**kwargs)
        
    def render(self):
        frame = self.env.render()
        if frame is not None:
            frame = frame.copy()
            cv2.putText(
                frame, 
                f"Score: {self.current_score}", 
                (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
        return frame


def train_model(name, net_arch, total_timesteps, eval_freq_per_env):
    """
    Trains a PPO model with the given network architecture and step limits,
    outputting perfectly organized logs, models, and EXACTLY 20 videos.
    """
    LOG_DIR = f"./logs/{name}/"
    MODEL_DIR = f"./models/{name}/"
    VIDEO_DIR = f"./videos/{name}/"

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    num_cpu = 4
    train_env = make_vec_env(
        "FlappyBird-v0", 
        n_envs=num_cpu, 
        vec_env_cls=SubprocVecEnv,
        monitor_dir=LOG_DIR
    )

    eval_env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    eval_env = ScoreOverlayWrapper(eval_env)
    eval_env = gym.wrappers.RecordVideo(
        eval_env, 
        video_folder=VIDEO_DIR, 
        name_prefix=f"flappy_agent_{name}",
        episode_trigger=lambda x: True 
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=eval_freq_per_env,
        deterministic=True,        
        render=False,              
        n_eval_episodes=1          
    )

    policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch))

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cpu"
    )

    print(f"\n--- Starting Training for {name.upper()} model ---")
    print(f"Architecture: {net_arch}, Total Steps: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    model.save(f"{MODEL_DIR}/ppo_flappy_bird_{name}_final")
    print(f"--- Completed Training for {name.upper()} model ---\n")


def main():
    # 1. SMALL MODEL (16 neurons per layer)
    # We train it for 250,000 steps. 
    # To get 20 videos: 250k / 20 = 12,500 total steps per eval.
    # 12,500 total / 4 cpu environments = 3125 eval_freq_per_env
    train_model("small", [16, 16], 250000, 3125)

    # 2. MEDIUM MODEL (64 neurons per layer)
    # We train it for 500,000 steps. 
    # To get 20 videos: 500k / 20 = 25,000 total steps per eval.
    # 25,000 total / 4 cpu environments = 6250 eval_freq_per_env
    train_model("medium", [64, 64], 500000, 6250)

    # 3. LARGE MODEL (256 neurons per layer - the max size)
    # We train it for 1,000,000 steps. 
    # To get 20 videos: 1M / 20 = 50,000 total steps per eval.
    # 50,000 total / 4 cpu environments = 12500 eval_freq_per_env
    train_model("large", [256, 256], 1000000, 12500)

if __name__ == "__main__":
    main()
