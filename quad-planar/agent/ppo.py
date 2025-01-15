import torch
import torch.nn as nn


class network(nn.Module):
    def __init__(self):

        self.device = torch.device("cuda")

        self.policy = nn.Sequential(
            nn.Linear(5, 126),
            nn.ReLU(),
            nn.Linear(126, 126),
            nn.ReLU(),
        )

        self.means = nn.Sequential(nn.Linear(126, 2))
        # as opposed to the REINFORCE impl, which gets stds as a layer out of the policy network
        # OpenAI implements PPO with a state-independent log_std
        self.log_stds = nn.Parameter(torch.zeros(2))

        # critic output
        self.critic = nn.Linear(126, 1)

        self.policy.to(self.device)

    def forward(self, state):
        features = self.policy(state)

        mean = self.means(features)
        std = torch.exp(self.log_stds)
        critic_feedback = self.critic(features)

        return (mean, std, critic_feedback)
