from stable_baselines3.common import vec_env

# from agent.networks import REINFORCE
from agent import vpg
from tqdm import tqdm
from env.environment import QuadEnv
from env.environment import QuadRender
from env.quadcopter import Quad2d
from setproctitle import setproctitle
import gymnasium as gym
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import random
import sys
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
    episode_returns = []
    for i in tqdm(range(int(5e5))):
        rewards, actions, states = [], [], []
        state, _ = env.reset()
        while True:
            action, value, log_probs = model.action(torch.tensor(state).float())
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = obs

            if terminated or truncated:
                break
        # single trajectory
        batch = (states, actions, rewards)
        model.update(batch)
        episode_returns.append(float(np.sum(rewards)))
        if (i + 1) % 1000 == 0:
            print(f"Episode {i + 1}: return {episode_returns[-1]:.3f}")

    torch.save(model.net.state_dict(), "vpg_model.pkl")

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
