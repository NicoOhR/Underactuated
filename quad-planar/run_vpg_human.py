import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from agent import vpg
from tqdm import tqdm
from env.environment import QuadEnv
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


from agent.vpg import VPG


def main():
    env = gym.make("QuadEnv-v0", render_mode="human")
    model = VPG(env)
    state_dict = torch.load("vpg_model.pkl", map_location="cpu")
    model.net.load_state_dict(state_dict)
    model.net.eval()

    state, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            action, _, _ = model.action(torch.tensor(state).float())
        obs, _, terminated, _, _ = env.step(action)
        done = terminated
        state = obs

    plt.ioff()
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
