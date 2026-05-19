import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0

    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_value(x_critic)

        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device="cpu", algorithm="reinforce", baseline=None):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.algorithm = algorithm
        self.baseline = baseline

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        if self.algorithm == "reinforce":
            returns = discount_rewards(rewards, self.gamma)

            if self.baseline is not None:
                advantages = returns - self.baseline
            else:
                advantages = returns

            policy_loss = -(action_log_probs * advantages.detach()).mean()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            return policy_loss.item()

        elif self.algorithm == "actor_critic":
            _, state_values = self.policy(states)
            state_values = state_values.squeeze(-1)

            with torch.no_grad():
                _, next_state_values = self.policy(next_states)
                next_state_values = next_state_values.squeeze(-1)
                targets = rewards + self.gamma * next_state_values * (1 - done)

            advantages = targets - state_values

            actor_loss = -(action_log_probs * advantages.detach()).mean()
            critic_loss = F.mse_loss(state_values, targets.detach())

            total_loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            return total_loss.item()

        else:
            raise ValueError("Unknown algorithm. Use 'reinforce' or 'actor_critic'.")

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:
            return normal_dist.mean, None

        else:
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
