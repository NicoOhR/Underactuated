from agent.networks import REINFORCE
from tqdm import tqdm
from env.environment import QuadEnv
from env.quadcopter import Quadcopter2d
from setproctitle import setproctitle
import gymnasium as gym
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import random
import sys
import matplotlib.pyplot as plt


def main():
    env = gym.make("QuadEnv-v0")
    wrapped = gym.wrappers.RecordEpisodeStatistics(env, 50)
    total_episodes = int(0.5e5)
    if env.observation_space.shape:
        obs_space_dims = env.observation_space.shape[0]
    else:
        print("failed to get observation dimensions")
        return -1

    action_space_dims = 4

    reward_over_seeds = []

    for seed in [1]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        agent = REINFORCE(obs_space_dims, action_space_dims)
        reward_over_episodes = []
        for episode in tqdm(range(total_episodes)):
            obs, info = wrapped.reset(seed=seed)
            done = False
            while not done:
                action = agent.sample_action(obs)
                obs, reward, terminated, truncated, info = wrapped.step(int(action))
                agent.rewards.append(reward)
                done = terminated or truncated

            reward_over_episodes.append(wrapped.return_queue[-1])
            agent.update()
            if episode % 1000 == 0:
                avg_reward = int(np.mean(wrapped.return_queue))
                print("Episode:", episode, "Avg:", avg_reward)
                agent.save(f"trained/agent_episode_{episode}.pt")

            reward_over_seeds.append(reward_over_episodes)

    #     rewards_to_plot = reward_over_seeds
    #     df1 = pd.DataFrame(rewards_to_plot).melt()
    #     df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    #     sns.set_theme(style="darkgrid", context="talk", palette="rainbow")
    #     sns.lineplot(x="episodes", y="reward", data=df1).set(
    #         title="REINFORCE for planar quadcopter"
    #     )
    #     plt.show()
    return 0


if __name__ == "__main__":
    setproctitle("training_process")
    try:
        main()
    except KeyboardInterrupt:
        print("Program Interrupted")
