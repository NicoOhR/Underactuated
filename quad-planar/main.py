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
    for i in tqdm(range(int(2e6))):
        rewards, actions, states = [], [], []
        state, _ = env.reset()
        while True:
            action, value, log_probs = model.action(torch.tensor(state).float())
            obs, reward, terminated, _, info = env.step(action)
            rewards.append(reward)
            states.append(obs)
            actions.append(action)
            state = obs

            if terminated:
                break
        # single trajectory
        batch = (states, actions, rewards)
        model.update(batch)

    state, _ = env.reset()

    while True:
        action, _, _ = model.action(torch.tensor(state).float())
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
        state = obs


if __name__ == "__main__":
    setproctitle("training_process")
    try:
        main()
    except KeyboardInterrupt:
        print("exiting application")
