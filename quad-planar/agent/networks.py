from __future__ import annotations
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

import gymnasium as gym


class Policy_Network(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_layer1 = 16
        hidden_layer2 = 32

        self.policy = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_layer1),
            nn.ReLU(),
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.ReLU(),
        )

        self.mean_layer = nn.Sequential(nn.Linear(hidden_layer2, action_space_dims))

        self.stddev_layer = nn.Sequential(nn.Linear(hidden_layer2, action_space_dims))

        self.policy.to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.policy(x.float().to(self.device))
        means = self.mean_layer(shared)
        stddevs = torch.log(1 + torch.exp(self.stddev_layer(shared)))
        return means, stddevs


class REINFORCE:
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.eps = 1e-6
        self.probs = []
        self.rewards = []
        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        self.net.to(self.device)

    def sample_action(self, state: np.ndarray):
        tstate = torch.from_numpy(state).to(self.device)
        means, stddevs = self.net(tstate)
        cov = torch.diag(stddevs**2)
        dist = MultivariateNormal(means, cov)
        action = dist.sample()
        prob = dist.log_prob(action)

        f1, f2 = action.tolist()

        self.probs.append(prob)

        return [f1, f2]

    def update(self):
        running_g = 0
        gs = []

        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs).to(self.device)
        log_probs = torch.stack(self.probs)
        log_prob_mean = log_probs.mean()

        loss = -torch.sum(log_prob_mean * deltas)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []

    def save(self, fname):
        torch.save(self.net.state_dict(), fname)
