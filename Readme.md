Proximal Policy Optimization (PPO) Agent for CartPole-v1

This project implements Proximal Policy Optimization (PPO)
a powerful actor-critic algorithm in reinforcement learning (RL), to solve the CartPole-v1 environment from OpenAI Gymnasium. 
PPO is known for its reliability and simplicity, and is widely used in academic and industry settings.

The aim is to train an agent that can balance a pole on a moving cart using discrete actions. 
This repository provides a clean, readable, and modular PPO implementation suitable for learning, demonstration, 
and further extension.


Objectives

1. Implement PPO from scratch using PyTorch

2. Build a custom Actor-Critic network architecture

3. Store experiences using a RolloutBuffer

4. Update the policy using the clipped surrogate objective

5. Visualize training performance

6. Render trained agent behavior


Environment

Environment: CartPole-v1

Frameworks: PyTorch, Gymnasium, NumPy, Matplotlib

Observation Space: Box(4,) — Continuous state representation

Action Space: Discrete(2) — Move cart left or right

Goal: Keep the pole upright for as long as possible (max episode length: 500 steps).


Project Structure

RL_project/
├── main.py             # Training script
├── actor_critic.py     # Actor-Critic model
├── rollout_buffer.py   # Buffer for storing rollouts and computing advantages
├── ppo_update.py       # PPO algorithm with clipped objective
├── README.md           # Project documentation



Key Concepts

1. Actor-Critic Architecture

Shared feature extractor
Actor head: Outputs action probabilities
Critic head: Outputs state value estimate


2. Rollout Buffer

Stores observations, actions, rewards, dones, log-probs, and values
Computes Generalized Advantage Estimation (GAE)
Normalizes advantages for stability

3. PPO Loss Function

The PPO loss is composed of three parts:
Policy Loss with clipped ratio
Value Loss (MSE between predicted value and 'return')
Entropy Bonus (to encourage exploration)



How to Run

pip install torch gymnasium numpy matplotlib -> To install the dependency

python main.py -> To train the agent
You will see printed logs per iteration




Visualization

At the end of training, the program will automatically plot the average reward per iteration:

X-axis: Training Iteration
Y-axis: Average Episode Reward
This shows the agent's performance improvement over time.

Demo: Rendering the Trained Agent
After training, a few episodes are rendered using the trained policy to visualize behavior.
You can modify the number of demo episodes or adjust delay via time.sleep().
To close the PyGame window, simply click the close (X) icon or press Esc if supported.


Author:
VARUN HUCHCHANNAVAR
Final Year Undergraduate Student, Dept of Mechanical Engineering, Indian Institute of Technology Kharagpur

This project is open source and free to use for educational and research purposes.
