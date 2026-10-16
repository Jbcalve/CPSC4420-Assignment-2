import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

def train_q_learning(
    env_name="Taxi-v4",
    n_episodes=1000,
    max_steps=200,
    alpha=0.1,       # Learning rate
    gamma=0.99,      # Discount factor
    epsilon=1.0,     # Initial exploration rate
    epsilon_decay=0.995,
    min_epsilon=0.01
):
    env = gym.make(env_name)
    
    # Initialize Q-table with zeros
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.zeros((state_size, action_size))
    
    rewards_per_episode = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_episode_reward = 0
        
        for step in range(max_steps):
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore
            else:
                action = np.argmax(q_table[state]) # Exploit
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Q-learning update rule
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state, best_next_action] * (not terminated)
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error
            
            state = next_state
            total_episode_reward += reward
            
            if terminated or truncated:
                break
                
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_per_episode.append(total_episode_reward)
        
    env.close()
    return q_table, rewards_per_episode

# Example: Investigate different hyperparameters
learning_rates = [0.1, 0.5, 0.9]
results = {}

for lr in learning_rates:
    print(f"Training with learning rate (alpha) = {lr}...")
    _, rewards = train_q_learning(alpha=lr)
    results[lr] = rewards

# Plotting the results to analyze hyperparameter behavior
plt.figure(figsize=(10, 6))
for lr, rewards in results.items():
    # Smooth rewards for cleaner visualization (moving average)
    smoothed_rewards = np.convolve(rewards, np.ones(100)/100, mode='valid')
    plt.plot(smoothed_rewards, label=f"Alpha = {lr}")

plt.xlabel("Episodes")
plt.ylabel("Total Reward (100-episode moving average)")
plt.title("Q-Learning Performance with Different Learning Rates on Taxi-v4")
plt.legend()
plt.grid(True)
plt.savefig("learning_rate_test.png")
plt.show()