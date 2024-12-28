from typing import Optional
import time
import numpy as np
import gymnasium as gym
from .quadcopter import Quadcopter2d
from .render import QuadRender
import matplotlib.pyplot as plt


class QuadEnv(gym.Env):
    def __init__(self, render_mode=None):
        metadata = {"render_modes": ["human"]}
        super(QuadEnv, self).__init__()
        """
        action space:
            * left propeller active, right propeller active, neither, both
        """
        self.action_space = gym.spaces.Discrete(4)
        """
         observation: * Encoders: u1/u2
             * IMU: acc_x, acc_y, omega
             * R^5: [0.0,0.0, unbounded, unbounded, unbounded] [50, 50, unbounded, unbounded, unbounded]
        """
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float64,
        )
        self.quad = Quadcopter2d()
        self.render_mode = render_mode
        if render_mode == "human":
            self._init_render()

    def reset(self, **kwargs):
        super().reset(**kwargs)
        state, reward = self._get_obs_info()
        info = {"reward": reward}
        self.quad.reset()
        return (np.array(state, dtype=np.float64), info)

    def _get_obs_info(self):
        u1, u2, acc_x, acc_y, acc_ang, _ = self.quad.get_agent_state()
        reward = -1 * (acc_x + acc_y + acc_ang) / 3
        return ([u1, u2, acc_x, acc_y, acc_ang], reward)

    def step(self, action):
        # print(action)
        direction = self.quad.current_action[action]
        self.quad.set_input(direction)
        self.quad.update()
        obs, reward = self._get_obs_info()
        info = {"reward": reward}
        terminated = self.quad.crash()
        if self.render_mode == "human":
            frame = int(self.quad.time_alive / self.renderer.dt)
            self.renderer.render(frame)

        return np.array(obs), reward, terminated, False, info

    def _init_render(self):
        self.renderer = QuadRender(self.quad)

    def close(self):
        if self.renderer:
            plt.close(self.renderer.fig)


if __name__ == "__main__":
    test_env = QuadEnv(render_mode="human")
    obs, info = test_env.reset()
    done = False
    while not done:
        obs, reward, terminated, truncated, info = test_env.step(action=3)
        test_env.render()
        done = terminated or truncated
    test_env.close()
