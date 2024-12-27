from __future__ import annotations
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym


class Policy_Network(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):

        super().__init__()

        hidden_layer1 = 16
        hidden_layer2 = 32

        self.shared_network = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_layer1),
            nn.ReLU(),
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.ReLU(),
        )

        self.policy_mean_network = nn.Sequential(
            nn.Linear(hidden_layer2, action_space_dims)
        )

        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_layer2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_network(x.float())
        action_means = self.policy_mean_network(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )
        return action_means, action_stddevs


class REINFORCE:
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.eps = 1e-6
        self.probs = []
        self.rewards = []
        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        tstate = torch.from_numpy(state)
        action_means, action_stddevs = self.net(tstate)

        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action
    
    def update(self):
        running_g = 0
        gs = []
        
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

            deltas = torch.stack(self.probs)
            log_probs = torch.stack(self.probs)
            log_prob_mean = log_probs.mean()

            loss = -torch.sum(log_prob_mean * deltas)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.probs = []
            self.rewards = []


