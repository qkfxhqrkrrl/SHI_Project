#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


class REINFORCE:
    def __init__(self, policy_network):
        self.policy_network = policy_network
        self.optimizer = optim.Adam(policy_network.parameters(), lr=1e-2)
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update_policy(self):
        # print(len(self.rewards))
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            # print(R, end=" ")
            returns.insert(0, R)
        # print()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.saved_log_probs = []
        self.rewards = []
        
def main():
    # env = ... # Initialize your environment here
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = PolicyNetwork(state_size, action_size)
    agent = REINFORCE(policy_net)

    for episode in range(1000): # Run for a number of episodes
        state = env.reset()[0]
        # print(state)
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = agent.select_action(state)
            state, reward, done, _, _ = env.step(action)
            # print(reward, end=" ")
            agent.rewards.append(reward)
            if done:
                # print(f"Done after {t}")
                break
        # print()
        agent.update_policy()
        if episode % 50 == 0:
            print(f"Episode {episode} complete")

if __name__ == "__main__":
    main()

# %%
