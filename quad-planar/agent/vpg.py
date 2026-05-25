import numpy as np
from torch.distributions import MultivariateNormal
import torch
import torch.nn as nn
import torch.optim


class network(nn.Module):
    device: torch.device
    policy: nn.Sequential
    log_std: nn.Parameter
    value: nn.Sequential

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.device = torch.device("cuda")
        # theta: s in R6 -> a in [0,1]^2
        self.policy = nn.Sequential(
            nn.Linear(6, 126),
            nn.ReLU(),
            nn.Linear(126, 126),
            nn.ReLU(),
            nn.Linear(126, 2),
        )
        self.log_std = nn.Parameter(torch.full((2,), -2.3))

        # phi: s in R6 -> value in R
        self.value = nn.Sequential(
            nn.Linear(6, 126),
            nn.ReLU(),
            nn.Linear(126, 126),
            nn.ReLU(),
            nn.Linear(126, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[MultivariateNormal, torch.Tensor]:
        mu: torch.Tensor = self.policy.forward(obs)
        std: torch.Tensor = torch.exp(self.log_std)
        cov: torch.Tensor = torch.diag(std * std)
        dist: MultivariateNormal = MultivariateNormal(mu, cov)
        v: torch.Tensor = self.value(obs).squeeze(-1)
        return (dist, v)


class VPG:
    net: network
    policy_opt: torch.optim.AdamW
    value_opt: torch.optim.AdamW

    def __init__(self, env) -> None:
        self.env = env
        self.net = network()

        self.policy_opt = torch.optim.AdamW(
            list(self.net.policy.parameters()) + [self.net.log_std]
        )
        self.value_opt = torch.optim.AdamW(self.net.value.parameters())

    def action(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        given an observation, returns the action given by the the theta network
        as well as the rest of the agent supplied batch entry
        """
        dist: MultivariateNormal
        value: torch.Tensor
        dist, value = self.net.forward(obs)
        action: torch.Tensor = dist.sample()
        # print(action)
        return action, value, dist.log_prob(action)

    def update(
        self, batch: tuple[list, list[torch.Tensor], list[float]]
    ) -> tuple[float, float]:
        """
        from the batch of rewards, compute one increment of the training loop.
        Batch is a tuple of lists of equal size representing the values at each
        time step
        """
        S_list: list[torch.Tensor] = []
        A_list: list[torch.Tensor] = []
        RTG_list: list[torch.Tensor] = []
        s, a, r = batch
        S_list.append(torch.as_tensor(np.array(s)))
        A_list.append(torch.stack(a))
        R: torch.Tensor = torch.as_tensor(r)
        rtg: torch.Tensor = torch.zeros_like(R)
        tot: float = 0.0
        for t in reversed(range(R.shape[0])):
            tot = R[t] + 0.75 * tot
            rtg[t] = tot
        RTG_list.append(rtg)
        S_Batch: torch.Tensor = torch.cat(S_list, dim=0).float()
        A_Batch: torch.Tensor = torch.cat(A_list, dim=0)
        RTG_Batch: torch.Tensor = torch.cat(RTG_list, dim=0)
        dist: MultivariateNormal
        V: torch.Tensor
        dist, V = self.net(S_Batch)

        advantage: torch.Tensor = (RTG_Batch - V).detach()
        policy_loss: torch.Tensor = -(dist.log_prob(A_Batch) * advantage).mean()
        value_loss: torch.Tensor = (V - RTG_Batch).pow(2).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_opt.step()

        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()
        return policy_loss.item(), value_loss.item()
