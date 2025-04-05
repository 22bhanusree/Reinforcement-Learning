
# MDP
import numpy as np
from collections import defaultdict

class CatMonsterWorld:
    def __init__(self):
        # States
        self.S_0 = (0, 0)
        self.S_infinity = [(4, 4)]
        self.furniture = [(2, 1), (2, 2), (2, 3), (3, 2)]
        self.monsters = [(0, 3), (4, 1)]
        self.states = [(r, c) for r in range(5) for c in range(5) if (r, c) not in self.furniture]

        # Actions
        self.actions = ["U", "D", "L", "R"]
        self.correct = {
            "U": (-1, 0),
            "D": (1, 0),
            "L": (0, -1),
            "R": (0, 1)
        }
        self.action_symbols = {
            'U': '↑',
            'D': '↓',
            'L': '←',
            'R': '→'
        }

        # Reward
        self.R_step = -0.05
        self.R_monster = -8
        self.R_food = 10

        # Transition probabilities
        self.move_correct_prob = 0.70
        self.move_left_prob = 0.12
        self.move_right_prob = 0.12
        self.no_move_prob = 0.06

        # Discount Factor
        self.gamma = 0.925


    def is_terminal(self, state):
        return state in self.S_infinity

    def is_forbidden(self, state):
        return state in self.furniture

    def is_monster(self, state):
        return state in self.monsters

    def in_walls(self, state):
        r, c = state
        return 0 <= r < 5 and 0 <= c < 5

    def transition_probabilities(self,state, action):
        """Return the state transition probabilities of a given state and action."""
        r, c = state
        correct_move = self.correct[action]
        transitions = []

        # Correct direction
        next_state = (r + correct_move[0], c + correct_move[1])
        if self.in_walls(next_state) and not self.is_forbidden(next_state):
            transitions.append((self.move_correct_prob, next_state))
        else:
            transitions.append((self.move_correct_prob, state))

        # Move left and right to current action direction
        if action =="U":
            left_move = self.correct["L"]
            right_move = self.correct["R"]
        elif action == "D":
            left_move = self.correct["R"]
            right_move = self.correct["L"]
        elif action == "L":
            left_move = self.correct["D"]
            right_move = self.correct["U"]
        else:
            left_move = self.correct["U"]
            right_move = self.correct["D"]


        next_left_state = (r + left_move[0], c + left_move[1])
        next_right_state = (r + right_move[0], c + right_move[1])

        if self.in_walls(next_left_state) and not self.is_forbidden(next_left_state):
            transitions.append((self.move_left_prob, next_left_state))
        else:
            transitions.append((self.move_left_prob, state))

        if self.in_walls(next_right_state) and not self.is_forbidden(next_right_state):
            transitions.append((self.move_right_prob, next_right_state))
        else:
            transitions.append((self.move_right_prob, state))

        # No move
        transitions.append((self.no_move_prob, state))

        return transitions

    def Reward(self,next_state):
        if next_state in self.S_infinity:
            return self.R_food
        elif next_state in self.monsters:
            return self.R_monster
        return self.R_step

    def d0_start_state(self):
        """Initialize a random start state excluding forbidden states."""
        while True:
            state = (np.random.randint(0, 5), np.random.randint(0, 5))
            if not self.is_forbidden(state):
                return state

    def choose_next_state(self, prob_next_states):
        """Choose the next state based on the given transition probabilities."""
        state_prob_map = defaultdict(float)

        for prob ,state in prob_next_states:
            state_prob_map[state] += prob

        unique_states = list(state_prob_map.keys())
        combined_probs = list(state_prob_map.values())

        # Map unique states to string representations
        state_to_str_map = {state: f's_{i+1}' for i, state in enumerate(unique_states)}
        str_to_state_map = {v: k for k, v in state_to_str_map.items()}

        # Randomly select a string
        selected_state_str = np.random.choice(list(state_to_str_map.values()), p=combined_probs)
        selected_state = str_to_state_map[selected_state_str]

        return selected_state

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
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.value_network = ValueNetwork(state_size)
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=lr_value)

    def train(self, N_episodes):
        return_per_episode = []

        for episode_i in range(N_episodes):
            state = self.env.S_0
            total_return = 0
            I=1
            t=0
            while not self.env.is_terminal(state) and t<2000:
                t=t+1
                state_tensor = torch.FloatTensor(state)
                action_probs = self.policy_network(state_tensor)
                action_idx = np.random.choice(len(self.env.actions), p=action_probs.detach().numpy())
                action = self.env.actions[action_idx]

                prob_next_states = self.env.transition_probabilities(state, action)
                next_state = self.env.choose_next_state(prob_next_states)
                reward = self.env.Reward(next_state)
                total_return += reward



                # One-step Actor-Critic update
                next_state_tensor = torch.FloatTensor(next_state)
                value_estimate = self.value_network(state_tensor).unsqueeze(0)
                next_value_estimate = self.value_network(next_state_tensor).unsqueeze(0)
                if self.env.is_terminal(next_state):
                    next_value_estimate = torch.tensor([0]).float().unsqueeze(0)
                delta = reward + self.env.gamma * next_value_estimate - value_estimate

                self.optimizer_value.zero_grad()
                vloss = nn.MSELoss()
                td_target = reward + self.env.gamma * next_value_estimate
                value_loss = vloss(value_estimate, td_target.detach())
                # value_loss = vloss(value_estimate,next_value_estimate)
                value_loss.backward()
                self.optimizer_value.step()

                self.optimizer_policy.zero_grad()
                # print(action_idx,action_probs)
                log_prob_action = torch.log(action_probs[action_idx])
                policy_loss = -I*log_prob_action * delta.detach()
                policy_loss.backward()
                self.optimizer_policy.step()

                state = next_state
                I=self.env.gamma*I


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
    avg_last_100_rewards = np.mean(avg_rewards[-20:])

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
    plt.text(len(avg_rewards) - 1, avg_last_100_rewards + 25, f"Last 20 Ep Avg:{avg_last_100_rewards:.2f}",
             color="green", fontsize=9, ha="right", va="bottom", backgroundcolor="white")

    # Add labels, title, and legend
    plt.xlabel("Episodes")
    plt.ylabel("Total Return")
    plt.title(title)
    # plt.legend()
    # plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Training setup
def main():
    # Environment and training parameters
    # env = CatMonsterWorld()
    state_size = 2  # State is a tuple (row, column)
    action_size = 4 # 4 Actions ,left , right , up , down
    # hidden_size = 128
    lr_policy = 1e-5
    lr_value = 1e-2
    N_episodes = 1500
    num_runs=2

    all_rewards = []

    for run in range(num_runs):
        print(f"Starting Run {run + 1}")
        env = CatMonsterWorld()
        agent = OneStepActorCriticAgent(env, state_size, action_size, lr_policy, lr_value)
        rewards = agent.train(N_episodes)
        all_rewards.append(rewards)

    # Convert to NumPy array for easier calculations
    return np.array(all_rewards)

all_rewards = main()
plot_avg_rewards(all_rewards, title="One-Step Actor-Critic for CatMonsterWorld (5 Runs)")

