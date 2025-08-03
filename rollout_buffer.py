import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, buffer_size, obs_dim, gamma=0.99, lam=0.95):
        """
        Buffer for PPO rollout storage (discrete action space).

        Args:
            buffer_size (int): max number of steps in buffer
            obs_dim (tuple): observation shape (e.g. (4,))
            gamma (float): discount factor
            lam (float): GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0

        self.obs_buf = np.zeros((buffer_size, *obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)  # discrete actions as ints
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

    def add(self, obs, action, reward, done, log_prob, value):
        assert self.ptr < self.buffer_size, "Buffer overflow!"
        self.obs_buf[self.ptr] = obs
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, done):
        adv = 0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            adv = delta + self.gamma * self.lam * next_non_terminal * adv
            self.advantages[t] = adv

        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get(self):
        adv_mean = np.mean(self.advantages[:self.ptr])
        adv_std = np.std(self.advantages[:self.ptr]) + 1e-8
        self.advantages[:self.ptr] = (self.advantages[:self.ptr] - adv_mean) / adv_std

        batch = {
            'states': torch.tensor(self.obs_buf[:self.ptr], dtype=torch.float32),
            'actions': torch.tensor(self.actions[:self.ptr], dtype=torch.int64),
            'log_probs': torch.tensor(self.log_probs[:self.ptr], dtype=torch.float32),
            'returns': torch.tensor(self.returns[:self.ptr], dtype=torch.float32),
            'advantages': torch.tensor(self.advantages[:self.ptr], dtype=torch.float32),
        }
        return batch

    def reset(self):
        self.ptr = 0

    def is_full(self):
        return self.ptr >= self.buffer_size
