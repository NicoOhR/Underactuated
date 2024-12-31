from env.render import QuadRender
from agent.networks import REINFORCE
import gymnasium as gym
import torch
import sys

env = gym.make("QuadEnv-v0", render_mode="human")
if env.observation_space.shape:
    obs_space_dims = env.observation_space.shape[0]
else:
    sys.exit(-1)
if env.action_space.shape:
    action_space_dims = env.action_space.shape[0]
else:
    sys.exit(-1)

model = REINFORCE(obs_space_dims, action_space_dims)
model.net.load_state_dict(torch.load("trained/agent_episode_49000.pt"))

done = False
obs, info = env.reset()
while not done:
    action = model.sample_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
