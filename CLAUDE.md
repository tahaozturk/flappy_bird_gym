# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (Python 3.10.12 required)
pip install -r requirements.txt

# Train all three models sequentially with default steps
python train.py

# Train a specific model size with a custom step count
python train.py --size medium --steps 2000000
python train.py --size large --steps 500000

# Watch the trained large model play live
python enjoy.py

# Post-process recorded videos with overlays
python process_videos.py
```

No build step, test suite, or linter is configured.

## Architecture

This is a reinforcement learning project that trains a PPO agent (via Stable Baselines 3) to play Flappy Bird (via `flappy-bird-gymnasium`), then records and post-processes gameplay videos.

### Pipeline

**train.py** — core training script

- Trains three MLP-policy PPO models sequentially: Small `[16,16]`, Medium `[64,64]`, Large `[256,256]` neurons.
- Each model uses 4 parallel training environments (`SubprocVecEnv`) and a separate evaluation environment wrapped with `RecordVideo`.
- `ScoreOverlayWrapper` (custom `gym.Wrapper` at the top of `train.py`) intercepts `render()` calls and uses OpenCV to burn the current game score into each frame before it is recorded.
- Evaluation frequency is computed so that exactly 20 video clips are captured per model: `eval_freq = (total_timesteps / 20) / num_envs`.
- Outputs are written into timestamped subfolders to avoid overwriting previous runs: `models/{size}/{timestamp}/`, `logs/{size}/{timestamp}/`, `videos/{size}/{timestamp}/`.
- `--size` selects which model to train; omitting it trains all three. `--steps` overrides the per-size default.

**enjoy.py** — inference/playback

- Loads `models/large/ppo_flappy_bird_large_final` and renders live gameplay in a Pygame window at ~30 FPS.

**process_videos.py** — post-processing

- Reads `.meta.json` sidecar files alongside each recorded `.mp4` to recover the training step at which the clip was recorded.
- Speeds each clip up 2×, overlays `"Model: {size}\nStep: {step}"` text (semi-transparent background, bottom-left), and concatenates clips chronologically.
- Outputs one master video per model size to `edited_videos/`.

### Key hyperparameters (PPO)

| Model  | Net arch  | Total steps | Videos |
|--------|-----------|-------------|--------|
| Small  | [16, 16]  | 250 000     | 20     |
| Medium | [64, 64]  | 500 000     | 20     |
| Large  | [256, 256]| 1 000 000   | 20     |

Shared: `lr=3e-4`, `batch_size=64`, `n_steps=1024`, `n_epochs=10`, `gamma=0.99`.
