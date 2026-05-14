import argparse
import os
from datetime import datetime
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import cv2

NET_ARCH = {
    "small":  [16, 16],
    "medium": [64, 64],
    "large":  [256, 256],
}

DEFAULT_STEPS = {
    "small":  250_000,
    "medium": 500_000,
    "large":  1_000_000,
}

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


def train_model(name, net_arch, total_timesteps, timestamp):
    num_cpu = 4
    eval_freq_per_env = max(1, total_timesteps // (20 * num_cpu))

    LOG_DIR = f"./logs/{name}/{timestamp}/"
    MODEL_DIR = f"./models/{name}/{timestamp}/"
    VIDEO_DIR = f"./videos/{name}/{timestamp}/"

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

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
    parser = argparse.ArgumentParser(description="Train a PPO agent on Flappy Bird")
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        help="Model size to train (default: train all three)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Total training timesteps (default: size-specific preset)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.size:
        sizes = [args.size]
    else:
        sizes = ["small", "medium", "large"]

    for size in sizes:
        steps = args.steps if args.steps is not None else DEFAULT_STEPS[size]
        train_model(size, NET_ARCH[size], steps, timestamp)

if __name__ == "__main__":
    main()
