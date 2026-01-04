import gymnasium as gym
import matplotlib.pyplot as plt
import torch
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

    input("Loaded model and env. Press Enter to start rendering...")

    state, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            action, _, _ = model.action(torch.tensor(state).float())
        obs, _, terminated, _, _ = env.step(action)
        done = terminated
        state = obs

    # Keep the render window open after the episode ends.
    plt.ioff()
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
