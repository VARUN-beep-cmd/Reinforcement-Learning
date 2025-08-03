import torch
import torch.nn.functional as F
from typing import Dict

def ppo_update(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_epsilon: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    epochs: int = 4,
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Perform PPO update on the model using a batch dict of rollout data.

    Args:
        model: ActorCritic model returning (logits, values).
        batch: Dict with keys: 'states', 'actions', 'log_probs', 'returns', 'advantages'.
               Each value is a torch tensor on the correct device.
        optimizer: Torch optimizer for model parameters.
        device: Device to run computations on (cpu or cuda).
        clip_epsilon: Clipping epsilon for PPO objective.
        vf_coef: Weight for value loss.
        ent_coef: Weight for entropy bonus.
        epochs: Number of passes over dataset.
        batch_size: Minibatch size.

    Returns:
        Dict with average losses: 'actor_loss', 'critic_loss', 'entropy'.
    """

    # Move batch tensors to the device (for safety)
    states = batch['states'].to(device)
    actions = batch['actions'].to(device)
    old_log_probs = batch['log_probs'].to(device)
    returns = batch['returns'].to(device)
    advantages = batch['advantages'].to(device)

    dataset_size = len(states)

    actor_losses = []
    critic_losses = []
    entropies = []

    for _ in range(epochs):
        indices = torch.randperm(dataset_size)

        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_idx = indices[start:end]

            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]

            # Forward pass: get logits and values
            logits, values = model(batch_states)
            dist = torch.distributions.Categorical(logits=logits)

            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - batch_old_log_probs)

            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(values.squeeze(), batch_returns)

            loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.item())

    avg_losses = {
        'actor_loss': sum(actor_losses) / len(actor_losses),
        'critic_loss': sum(critic_losses) / len(critic_losses),
        'entropy': sum(entropies) / len(entropies),
    }

    return avg_losses
