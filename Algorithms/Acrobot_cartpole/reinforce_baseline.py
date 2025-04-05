
# MDP
import numpy as np
from collections import defaultdict
import gym

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """Neural network to approximate the policy."""
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        # self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, output_dim)
        # self.fc4 = nn.Linear(8, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        x = self.softmax(self.fc3(x))
        return x


class ValueNetwork(nn.Module):
    """Neural network to approximate the value function."""
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        # self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(128, 1)
        # self.fc4 = nn.Linear(4,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        x = self.fc3(x)
        return x


class ReinforceAgent():
    """REINFORCE with baseline agent for training in a custom environment."""
    def __init__(self, env, state_size,action_size, lr_policy, lr_value):
        self.env = env
        self.gamma = 0.99

        # Initialize policy and value networks
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.value_network = ValueNetwork(state_size)

        # Optimizers
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=lr_value)

    def generate_episode(self,state):
        """Generate an episode (states, actions, rewards) using the current policy."""
        episode = []

        t=0
        terminated=False
        while not terminated :
            # Get action probabilities and sample an action
            t=t+1
            state_tensor = torch.FloatTensor(state)
            action_probs = self.policy_network(state_tensor)
            action = np.random.choice(self.env.action_space.n, p=action_probs.detach().numpy())
            next_state,reward,terminated,truncate = self.env.step(action)
            terminated=terminated or truncate
            episode.append((state, action , reward))
            state = next_state

        return episode

    def update(self,episode):
        """Perform policy and value updates based on the generated episode."""
        T = len(episode)
        for t in range(T):
            # Calculate discounted return G_t
            G_t = sum([self.gamma ** (i-t) * episode[i][2] for i in range(t, T)])
            G_t_tensor = torch.tensor([G_t]).float().unsqueeze(0)

            # Convert state to tensor and get value estimate
            state_tensor = torch.FloatTensor(episode[t][0]).unsqueeze(0)
            value_estimate = self.value_network(state_tensor)
            delta = G_t_tensor - value_estimate

            # Update value network
            self.optimizer_value.zero_grad()
            mse_loss_fn=nn.MSELoss()
            value_loss =mse_loss_fn(value_estimate, G_t_tensor)  # MSE loss
            value_loss.backward()
            self.optimizer_value.step()

            # Update policy network
            self.optimizer_policy.zero_grad()
            action_probs = self.policy_network(state_tensor)
            log_prob_action = torch.log(action_probs[0, episode[t][1]])
            # policy_loss = -(self.env.gamma**t)*log_prob_action * delta.detach()
            policy_loss=-log_prob_action*delta.detach()
            policy_loss.backward()
            self.optimizer_policy.step()

    def train(self, N_episodes):
        """Train the agent over multiple episodes."""
        return_per_episode = []
        total_steps = 0
        for episode_i in range(N_episodes):
            state,info = self.env.reset(return_info=True)
            episode = self.generate_episode(state)
            self.update(episode)
            self.env.close()
            print(episode)
            total_return= sum(step[2] for step in episode)
            return_per_episode.append(total_return)
            if episode_i % 10 == 0:
                print(f"Episode {episode_i}: Total Return = {total_return:.2f}")

        return return_per_episode


def plot_rewards(rewards, title="Training Performance", save_path=None):
    """Plot the training rewards."""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Total Return")
    plt.xlabel("Episodes")
    plt.ylabel("Total Return")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_avg_rewards(all_rewards, title="Training Performance (Average of 5 Runs)", save_path=None):
    """Plot the average and standard deviation of rewards over multiple runs."""
    avg_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    # Compute average of the last 100 episodes
    avg_last_100_rewards = np.mean(avg_rewards[-30:])

    plt.figure(figsize=(11, 7))

    # Plot average rewards
    plt.plot(avg_rewards, label="Average Return", color='blue')

    # Fill between for standard deviation
    plt.fill_between(range(len(avg_rewards)),
                     avg_rewards - std_rewards,
                     avg_rewards + std_rewards,
                     color='blue', alpha=0.2, label="Std Dev")

    # Add horizontal dashed line for last 100 episodes average
    # plt.axhline(y=10, color="red", linestyle="--", label=f"Avg over Last 100 Episodes: {avg_last_100_rewards:.2f}")
    plt.axhline(y=10, color="red", linestyle="--", label="Return = 10")

    # Annotate the value of the dashed line on the plot
    plt.text(len(avg_rewards) - 1, avg_last_100_rewards - 88, f"Last 20 Ep Avg:{avg_last_100_rewards:.2f}",
             color="green", fontsize=9, ha="right", va="bottom", backgroundcolor="white")

    # Add labels, title, and legend
    plt.xlabel("Episodes")
    plt.ylabel("Total Return")
    plt.title(title)
    plt.legend()
    # plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Training setup
def main():

    env =  gym.make('Acrobot-v1') # env = gym.make('CartPole-v1')

    state_size = env.observation_space.shape[0]
    action_size =  env.action_space.n
    lr_policy = 1e-4
    lr_value = 1e-2
    N_episodes = 1000
    num_runs=3

    all_rewards = []

    for run in range(num_runs):
        print(f"Starting Run {run + 1}")
        agent = ReinforceAgent(env, state_size, action_size, lr_policy, lr_value)
        rewards = agent.train(N_episodes)
        all_rewards.append(rewards)

    # Convert to NumPy array for easier calculations
    return np.array(all_rewards)

all_rewards = main()
plot_avg_rewards(all_rewards, title="REINFORCE with Baseline for Acrobot-v1 (5 Runs)")

