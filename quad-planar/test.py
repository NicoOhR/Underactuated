from environment.render import QuadRender
from agent.networks import REINFORCE
import gymnasium as gym
import torch
import sys

env = gym.make("QuadEnv-v0", render_mode="human")
if env.observation_space.shape:
    obs_space_dims = env.observation_space.shape[0]
else:
    sys.exit(-1)

action_space_dims = 4
model = REINFORCE(obs_space_dims, action_space_dims)
model.net.load_state_dict(torch.load("trained/agent_episode_99000.pt"))

done = False
obs, info = env.reset()
while not done:
    action = model.sample_action(obs)
    obs, reward, terminated, truncated, info = env.step(int(action))
    done = terminated or truncated
