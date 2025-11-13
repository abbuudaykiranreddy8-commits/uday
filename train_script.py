# train_script.py
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

from agent_dqn import Agent_DQN

def make_env(env_id='BreakoutNoFrameskip-v4'):
    env = gym.make(env_id)
    env = AtariPreprocessing(env, grayscale_obs=True, frame_skip=1, screen_size=84, scale_obs=True)
    env = FrameStack(env, num_stack=4)
    return env

def plot_training_curve(episode_rewards, out='training_curve.png'):
    # episode_rewards: list of episode total rewards per episode
    window = 30
    avgs = []
    for i in range(len(episode_rewards)):
        start = max(0, i - window + 1)
        avgs.append(np.mean(episode_rewards[start:i+1]))
    plt.figure(figsize=(10,5))
    plt.plot(avgs, label=f'{window}-episode avg')
    plt.xlabel('Episode')
    plt.ylabel('Average reward (moving)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out)
    print(f"Saved training curve to {out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--frames', type=int, default=2_000_000)
    parser.add_argument('--save', default='my_dqn_breakout.pth')
    parser.add_argument('--log_interval', type=int, default=5000)
    args = parser.parse_args()

    env = make_env(args.env)
    agent = Agent_DQN(env, args)

    print("Starting training...")
    episode_rewards = agent.train(num_frames=args.frames, save_path=args.save, log_interval=args.log_interval)

    # Save training curve
    plot_training_curve(episode_rewards, out='training_curve.png')

if __name__ == '__main__':
    main()
