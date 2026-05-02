import time
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO

# Load the final "Large" model
model_path = "./models/large/ppo_flappy_bird_large_final"
print(f"Loading model from {model_path}...")
model = PPO.load(model_path)

# Create the environment with a visible window (human render mode)
env = gym.make("FlappyBird-v0", render_mode="human")

print("Starting inference! Press Ctrl+C in the terminal to stop.")
obs, info = env.reset()

while True:
    # Let the AI predict the best action
    # deterministic=True means the AI uses its absolute best move (no random guessing)
    action, _states = model.predict(obs, deterministic=True)
    
    # Execute the move
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render the game window
    env.render()
    
    # Slow down the game slightly so it's watchable for a human
    time.sleep(1/30) 

    # If the bird dies, reset the game
    if terminated or truncated:
        print(f"Game Over! Final Score: {info.get('score', 0)}")
        obs, info = env.reset()
