from env.render import QuadRender
from stable_baselines3 import PPO
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

model = PPO.load("ppo_quad")

done = False
obs, info = env.reset()
while not done:
    action, _ = model.predict(obs)
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)
