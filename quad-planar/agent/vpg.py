import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.optim


class network(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.device = torch.device("cuda")
        # theta: s in R5 -> a in [0,1]^2
        self.policy = nn.Sequential(
            nn.Linear(5, 126),
            nn.ReLU(),
            nn.Linear(126, 126),
            nn.ReLU(),
            nn.Linear(126, 2),
        )
        self.log_std = nn.Parameter(torch.ones((2, 1)) * torch.log(torch.tensor(0.75)))
        # phi: s in R5 -> value in R
        self.value = nn.Sequential(
            nn.Linear(5, 126),
            nn.ReLU(),
            nn.Linear(126, 126),
            nn.ReLU(),
            nn.Linear(126, 1),
        )

    def forward(self, obs: torch.Tensor):
        mu = self.policy.forward(obs)
        dist = Normal(mu, torch.exp(self.log_std))
        v = self.value.forward(obs).squeeze()
        return (dist, v)


class VPG:
    def __init__(self, env):
        self.env = env
        self.net = network()

        self.policy_opt = torch.optim.AdamW(self.net.policy.parameters())
        self.value_opt = torch.optim.AdamW(self.net.value.parameters())

    def action(self, obs):
        """
        given an observation, returns the action given by the the theta network
        as well as the rest of the agent supplied batch entry
        """
        dist, value = self.net.forward(obs)
        action = dist.sample()
        return action, value, dist.log_prob(action)

    def update(self, batch):
        """
        from the batch of rewards, compute one increment of the training loop.
        Batch is a tuple of lists of equal size representing the values at each
        time step
        """
        policy_loss = torch.zeros
        value_loss = 0
        S_list, A_list, RTG_list = [], [], []
        for i in range(len(batch)):
            s, a, r = batch[i]
            S_list.append(torch.as_tensor(s))
            A_list.append(torch.as_tensor(a))
            R = torch.as_tensor(r)
            rtg = torch.zeros_like(R)
            tot = 0.0
            # add discount later
            for t in range(R.shape[0]):
                tot += R[t]
                rtg[t] = tot
            RTG_list.append(rtg)
        S_Batch = torch.cat(S_list, dim=0)
        A_Batch = torch.cat(A_list, dim=0)
        RTG_Batch = torch.cat(RTG_list, dim=0)
        dist, V = self.net(S_Batch)

        advantage = RTG_Batch - V
        policy_loss = -(dist.log_prob(S_Batch) * advantage).mean()
        value_loss = (V - RTG_Batch).pow(2).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()
        return policy_loss.item(), value_loss.item()
