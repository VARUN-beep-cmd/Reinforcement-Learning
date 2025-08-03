def compute_gae(rollout_buffer, gamma=0.99, lam=0.95):
    """
    Compute GAE advantages and discounted returns for the rollout buffer.

    Args:
        rollout_buffer: Object containing rewards, dones, values arrays/lists.
        gamma: Discount factor.
        lam: GAE lambda parameter.

    This function updates rollout_buffer.advantages and rollout_buffer.returns in-place.
    """
    advantages = np.zeros_like(rollout_buffer.rewards, dtype=np.float32)
    returns = np.zeros_like(rollout_buffer.rewards, dtype=np.float32)
    gae = 0
    next_value = 0

    for t in reversed(range(len(rollout_buffer.rewards))):
        mask = 1.0 - rollout_buffer.dones[t]  # 0 if done else 1
        delta = rollout_buffer.rewards[t] + gamma * next_value * mask - rollout_buffer.values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        returns[t] = advantages[t] + rollout_buffer.values[t]
        next_value = rollout_buffer.values[t]

    rollout_buffer.advantages = advantages
    rollout_buffer.returns = returns
