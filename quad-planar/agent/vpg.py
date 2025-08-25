import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.optim

class network(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.device = torch.device("cuda")
        #theta: s in R5 -> a in [0,1]^2
        self.policy = nn.Sequential(
            nn.Linear(5, 126),
            nn.ReLU(),
            nn.Linear(126, 126),
            nn.ReLU(),
            nn.Linear(126, 2)
        )
        self.log_std = nn.Parameter(torch.ones((2,1)) * torch.log(torch.tensor(0.75)))
        #phi: s in R5 -> value in R
        self.value = nn.Sequential(
            nn.Linear(5, 126),
            nn.ReLU(),
            nn.Linear(126, 126),
            nn.ReLU(),
            nn.Linear(126, 1)
        )

    def forward(self, obs: torch.Tensor):
        mu = self.policy.forward(obs)
        dist = Normal(mu, torch.exp(self.log_std))
        v = self.value.forward(obs).squeeze()
        return (dist, v)

class VPG:
    def __init__(self, env, policy): 
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
        s, a, r, v, log_prob = batch #this might be better as a data class
        r_to_go  = [sum(r[i:]) for i in range(1, len(r))] #undiscounted
        advantage = [r_i - v_i for r_i, v_i  in zip(r_to_go, v)] #switch to GAE


        
