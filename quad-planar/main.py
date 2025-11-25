from stable_baselines3.common import vec_env

# from agent.networks import REINFORCE
from agent import vpg
from tqdm import tqdm
from env.environment import QuadEnv
from env.quadcopter import Quad2d
from setproctitle import setproctitle
import gymnasium as gym
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def main_baseline():
    env = gym.make("QuadEnv-v0")
    model = PPO("MlpPolicy", env, device="cpu", verbose=1)
    model.learn(total_timesteps=int(2e6))
    model.save("ppo_quad_cpu")
    del model


def main():
    env = gym.make("QuadEnv-v0")
    model = vpg.VPG(env)
    while True:
        rewards, actions, states = [], [], []
        state, _ = env.reset()
        while True:
            action, value, log_probs = model.action(torch.tensor(state).float())
            print(action.shape)
            obs, reward, terminated, _, info = env.step(action)
            rewards.append(reward)
            states.append(obs)
            actions.append(action)
            state = obs

            if terminated:
                break
        batch = (states, actions, rewards)
        model.update(batch)


if __name__ == "__main__":
    setproctitle("training_process")
    try:
        main()
    except KeyboardInterrupt:
        print("exiting application")
