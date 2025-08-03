import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        # Actor network outputs logits for categorical distribution
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic network outputs scalar state value
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        """
        Given a batch of states, returns:
          - logits: raw action scores (before softmax)
          - value: state-value estimates (shape [batch_size, 1])
        """
        logits = self.actor(state)
        value = self.critic(state)
        return logits, value

    def get_action_probs(self, state):
        """
        Given states, returns action probabilities.
        Note: Applies softmax over logits.
        """
        logits = self.actor(state)
        return F.softmax(logits, dim=-1)

    def get_value(self, state):
        """
        Returns scalar value estimates for states.
        """
        return self.critic(state).squeeze(-1)  # squeeze last dim for convenience


