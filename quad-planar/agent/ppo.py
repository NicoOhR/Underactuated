import torch
import torch.nn as nn
import torch.optim


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


class PPO:
    def __init__(self, env, policy_class, **hyperparameters):
        self.env = env
        self.obs = env.observation_space.shape[0]
        self.act = env.action_space.shape[0]

        self.net = network()

        params = [
            {"params": self.net.policy.parameters()},
            {"params": self.net.means.parameters()},
            {"params": self.net.critic.parameters()},
            {"params": [self.net.log_stds], "weight_decay": 0.0},
        ]

        self.actor_optim = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
