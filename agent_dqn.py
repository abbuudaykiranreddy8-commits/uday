# agent_dqn.py
import torch
import random
import numpy as np
from collections import deque
import torch.nn.functional as F
import torch.optim as optim
from dqn_model import DQN

class Agent_DQN(object):
    """
    Required API for Project 3.
    Implements exactly these functions (do not change signatures):
    - __init__(self, env, args)
    - init_game_setting(self)
    - make_action(self, state, test)
    - train(self)
    - push(self)
    - replay_buffer(self)
    """
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Hyperparameters (from slides / recommended) ---
        self.gamma = 0.99
        self.lr = 1.5e-4
        self.batch_size = 32
        self.replay_size = 10000
        self.start_train = 5000
        self.target_update_freq = 5000
        self.train_freq = 4
        
        # epsilon schedule
        self.eps_start = 1.0
        self.eps_final = 0.025
        self.eps_decay_steps = 1_000_000
        
        # replay buffer (deque)
        self.buffer = deque(maxlen=self.replay_size)
        
        # action space
        self.num_actions = self.env.action_space.n
        
        # networks
        self.policy = DQN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.target = DQN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # bookkeeping
        self.total_steps = 0
        self.episode_rewards = []
        self.lives_history = []
        self.losses = []

    def init_game_setting(self):
        """Called before training/testing to initialize any settings."""
        pass

    def make_action(self, state, test=False):
        """
        state: stacked frames (numpy array or torch tensor), shape (4,84,84)
        test: if True, run deterministic (eps=0)
        Returns: action (int)
        """
        if isinstance(state, np.ndarray):
            s = torch.tensor(state, dtype=torch.float32)
        else:
            s = state
        s = s.unsqueeze(0).to(self.device)
        
        eps = 0.0 if test else self._epsilon()
        if random.random() < eps:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q = self.policy(s)
            return int(q.argmax(1).item())

    def _epsilon(self):
        frac = min(self.total_steps / float(self.eps_decay_steps), 1.0)
        return self.eps_start + frac * (self.eps_final - self.eps_start)

    def push(self, state, action, reward, next_state, done):
        """Add transition to replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def replay_buffer(self):
        """Sample a minibatch from the deque and convert to tensors."""
        batch = random.sample(self.buffer, self.batch_size)
        s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)
        
        s = torch.stack([torch.tensor(x, dtype=torch.float32) for x in s_batch]).to(self.device)
        ns = torch.stack([torch.tensor(x, dtype=torch.float32) for x in ns_batch]).to(self.device)
        a = torch.tensor(a_batch, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(r_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        d = torch.tensor(d_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        return s, a, r, ns, d

    def train(self, num_frames=2_000_000, save_path='my_dqn_breakout.pth', log_interval=1000):
        """Training loop."""
        obs = self.env.reset()
        episode_reward = 0.0
        episode_count = 0
        recent_lives = deque(maxlen=30)
        
        while self.total_steps < num_frames:
            self.total_steps += 1
            action = self.make_action(obs, test=False)
            next_obs, reward, done, info = self.env.step(action)
            
            self.push(obs, action, reward, next_obs, float(done))
            episode_reward += reward
            
            if isinstance(info, dict) and 'lives' in info:
                recent_lives.append(info['lives'])
            
            if len(self.buffer) >= self.start_train and (self.total_steps % self.train_freq == 0):
                s, a, r, ns, d = self.replay_buffer()
                q_values = self.policy(s).gather(1, a)
                with torch.no_grad():
                    next_q = self.target(ns).max(1)[0].unsqueeze(1)
                    target = r + (1.0 - d) * self.gamma * next_q
                loss = F.smooth_l1_loss(q_values, target)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                self.losses.append(loss.item())
            
            if self.total_steps % self.target_update_freq == 0:
                self.target.load_state_dict(self.policy.state_dict())
            
            if done:
                self.episode_rewards.append(episode_reward)
                episode_count += 1
                episode_reward = 0.0
                obs = self.env.reset()
            else:
                obs = next_obs
            
            if self.total_steps % log_interval == 0:
                avg100 = float(np.mean(self.episode_rewards[-100:])) if len(self.episode_rewards) >= 1 else 0.0
                print(f"[Steps {self.total_steps}] Episodes: {episode_count} | Avg100: {avg100:.2f} | Eps: {self._epsilon():.3f}")
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'target_state_dict': self.target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps
        }, save_path)
        print(f"Training finished. Model saved to {save_path}")
        return self.episode_rewards
