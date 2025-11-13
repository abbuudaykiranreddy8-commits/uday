# Project 3 — DQN for Breakout

## Overview
This repository contains a Deep Q-Network (DQN) implementation for playing Atari Breakout, developed as part of DS551/CS525 Fall 2025.

## Files Included
- `agent_dqn.py` - Core Agent_DQN class with required API methods
- `dqn_model.py` - PyTorch CNN model (DeepMind architecture)
- `train_script.py` - Training loop with environment setup and plotting
- `evaluate_and_screenshot.py` - Evaluation on 100 episodes
- `job.sh` - SLURM submission script for Turing cluster
- `report.md` - Project report template
- `README.md` - This file
- `submission_instructions.txt` - TA evaluation checklist

## How to Run

### 1) Setup (Turing / Local)
Create conda environment (python 3.11) and install dependencies:

```bash
conda create -n myenv python=3.11 -y
conda activate myenv
pip install torch torchvision torchaudio
pip install gymnasium[atari,accept-rom-license]==0.29.0 autorom[accept-rom-license] numpy matplotlib opencv-python-headless moviepy tqdm
python -m autorom && AutoROM -y
```

### 2) Quick Local Test (few frames)
```bash
python train_script.py --frames 200000 --save quick_test.pth
```

### 3) Full Training (recommended on GPU / Turing)
Edit `job.sh` if needed, then submit:
```bash
sbatch job.sh
```

### 4) Evaluate
```bash
python evaluate_and_screenshot.py --model my_dqn_breakout.pth --episodes 100
```

Save terminal output screenshot showing the printed average reward.

### 5) Submission
Create zip with:
- agent_dqn.py
- dqn_model.py
- my_dqn_breakout.pth
- report.pdf
- README.md
- training_curve.png

Name it: `FirstName_LastName_hw3.zip`

## Model Architecture
- DeepMind-like CNN architecture
- Conv(32,8,4) -> Conv(64,4,2) -> Conv(64,3,1) -> FC512 -> Output

## Hyperparameters
- Learning rate: 1.5e-4
- Optimizer: Adam
- Batch size: 32
- Replay buffer size: 10,000
- Gamma: 0.99
- Epsilon schedule: 1.0 → 0.025 over 1M frames

## Target Goal
Average reward ≥ 40 over 100 evaluation episodes

## References
- DeepMind DQN Paper: Nature 2015
- Gymnasium Documentation
- Assignment Specifications
