
# MDP
import numpy as np
from collections import defaultdict
import gym

# prompt: generate for one-step actor critic\

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """Neural network to approximate the policy."""
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        # self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


class ValueNetwork(nn.Module):
    """Neural network to approximate the value function."""
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        # self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x

class OneStepActorCriticAgent():
    def __init__(self, env, state_size, action_size, lr_policy, lr_value):

        self.env = env
        self.gamma =0.99

        self.policy_network = PolicyNetwork(state_size, action_size)
        self.value_network = ValueNetwork(state_size)

        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=lr_value)

    def train(self, N_episodes):
        return_per_episode = []

        for episode_i in range(N_episodes):
            state,info = self.env.reset(return_info=True)####Change to self.env.reset() if not working
            total_return = 0
            I=1
            t=0
            terminated = False
            while not terminated :
                t=t+1
                state_tensor = torch.FloatTensor(state)
                action_probs = self.policy_network(state_tensor)
                action = np.random.choice(self.env.action_space.n, p=action_probs.detach().numpy())
                next_state,reward,terminated,truncate = self.env.step(action)###Add info after truncate if not working
                terminated=terminated or truncate
                total_return += reward

                # One-step Actor-Critic update
                next_state_tensor = torch.FloatTensor(next_state)
                value_estimate = self.value_network(state_tensor).unsqueeze(0)
                next_value_estimate = self.value_network(next_state_tensor).unsqueeze(0)
                if terminated:
                    next_value_estimate = torch.tensor([0]).float().unsqueeze(0)
                delta = reward + self.gamma * next_value_estimate - value_estimate

                self.optimizer_value.zero_grad()
                vloss = nn.MSELoss()
                td_target = reward + self.gamma * next_value_estimate
                value_loss = vloss(value_estimate, td_target.detach())
                # value_loss = vloss(value_estimate,next_value_estimate)
                value_loss.backward()
                self.optimizer_value.step()

                self.optimizer_policy.zero_grad()
                # print(action_idx,action_probs)
                log_prob_action = torch.log(action_probs[action])
                policy_loss = -I*log_prob_action * delta.detach()
                policy_loss.backward()
                self.optimizer_policy.step()

                state = next_state
                I=self.gamma*I


            return_per_episode.append(total_return)
            if episode_i % 10 == 0:
                print(f"Episode {episode_i}: Total Return = {total_return:.2f}")
        return return_per_episode

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
        agent = OneStepActorCriticAgent(env, state_size, action_size, lr_policy, lr_value)
        rewards = agent.train(N_episodes)
        all_rewards.append(rewards)

    # Convert to NumPy array for easier calculations
    return np.array(all_rewards)

all_rewards = main()
plot_avg_rewards(all_rewards, title="1-Step Actor Critic for Acrobot-v1 (5 Runs)")

