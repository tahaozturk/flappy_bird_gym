# Flappy Bird AI Agent

This project contains a Reinforcement Learning agent trained to play Flappy Bird using Proximal Policy Optimization (PPO) from the `stable-baselines3` library. It was built for a YouTube video demonstrating how AI learns to play games!

## Setup and Installation

1. It is recommended to use a Python virtual environment.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Use

### 1. Training the Models
The `train.py` script trains three different sizes of neural networks (Small, Medium, and Large) to play the game. It will automatically save model checkpoints, TensorBoard logs, and gameplay videos during the training process.

```bash
python train.py
```

### 2. Watching the AI Play
The `enjoy.py` script loads the fully trained "Large" model and renders the game on your screen so you can watch the AI play in real-time.

```bash
python enjoy.py
```

### 3. Processing Videos (Optional)
The `process_videos.py` script takes the raw gameplay clips generated during training, adds a text overlay (showing the model size and training step), speeds them up, and stitches them together into a final master video.

```bash
python process_videos.py
```
