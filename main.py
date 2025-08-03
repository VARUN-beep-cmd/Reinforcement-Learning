import gym
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.optim as optim
from actor_critic import ActorCritic  # Your actor-critic model
from rollout_buffer import RolloutBuffer
from ppo_update import ppo_update  # PPO update function you already have

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create environment
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape
    action_dim = env.action_space.n
    buffer_size = 2048

    print("Environment initialized:")
    print(f" - Observation dimension: {obs_dim}")
    print(f" - Action dimension: {action_dim}")

    # Initialize policy and buffer
    policy = ActorCritic(obs_dim[0], action_dim).to(device)
    buffer = RolloutBuffer(buffer_size, obs_dim)

    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    num_iterations = 100
    all_avg_rewards = []

    for iteration in range(num_iterations):
        step = 0
        done = False
        ep_reward = 0
        episode_rewards = []

        obs = env.reset()
        if isinstance(obs, tuple):  # gymnasium compatibility
            obs = obs[0]

        while step < buffer_size:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            action_probs = policy.get_action_probs(obs_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = policy.get_value(obs_tensor)

            step_result = env.step(action.item())
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = step_result

            buffer.add(obs, action.item(), reward, done, log_prob.item(), value.item())

            ep_reward += reward
            obs = next_obs
            step += 1

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                done = False

        last_value = policy.get_value(torch.FloatTensor(obs).unsqueeze(0).to(device)).item()
        buffer.compute_returns_and_advantages(last_value, done)

        batch = buffer.get()
        losses = ppo_update(policy, batch, optimizer, device)

        buffer.reset()

        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        all_avg_rewards.append(avg_reward)
        print(f"Iteration {iteration + 1}/{num_iterations} - Average Reward: {avg_reward:.2f}")

    print("Training completed.")

    # # Demo rendering after training
    # demo_env = gym.make("CartPole-v1", render_mode="human")
    # for ep in range(5):
    #     obs = demo_env.reset()
    #     if isinstance(obs, tuple):
    #         obs = obs[0]
    #     done = False
    #     while not done:
    #         demo_env.render()
    #         time.sleep(0.02)
    #         obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    #         action_probs = policy.get_action_probs(obs_tensor)
    #         action = torch.argmax(action_probs, dim=-1)
    #         step_result = demo_env.step(action.item())
    #         if len(step_result) == 5:
    #             obs, reward, terminated, truncated, _ = step_result
    #             done = terminated or truncated
    #         else:
    #             obs, reward, done, _ = step_result

    # demo_env.close()
    # time.sleep(1)  # allow window to close cleanly

    # Plot rewards
    plt.plot(all_avg_rewards)
    plt.xlabel("Training Iteration")
    plt.ylabel("Average Episode Reward")
    plt.title("Average Rewards Over PPO Training Iterations")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
