# evaluate_and_screenshot.py
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from agent_dqn import Agent_DQN
import os

def make_env(env_id='BreakoutNoFrameskip-v4'):
    env = gym.make(env_id)
    env = AtariPreprocessing(env, grayscale_obs=True, frame_skip=1, screen_size=84, scale_obs=True)
    env = FrameStack(env, num_stack=4)
    return env

def evaluate(model_path, n_episodes=100, env_id='BreakoutNoFrameskip-v4'):
    env = make_env(env_id)
    dummy_args = type('x', (), {})()
    agent = Agent_DQN(env, dummy_args)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=agent.device)
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.policy.to(agent.device).eval()

    rewards = []
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            a = agent.make_action(s, test=True)
            s, r, done, info = env.step(a)
            ep_r += r
        rewards.append(ep_r)
        print(f"Episode {ep+1}/{n_episodes} -> {ep_r:.2f}")
    avg = float(np.mean(rewards))
    std = float(np.std(rewards))
    print(f"Average reward over {n_episodes} episodes: {avg:.2f} ± {std:.2f}")
    return rewards

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='my_dqn_breakout.pth')
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()
    evaluate(args.model, n_episodes=args.episodes)
