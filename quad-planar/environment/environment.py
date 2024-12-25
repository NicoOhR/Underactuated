from typing import Optional
import time
import numpy as np
import gymnasium as gym
from quadcopter import Quadcopter2d
from render import QuadRender
import matplotlib.pyplot as plt


class QuadEnv(gym.Env):
    def __init__(self, render_mode=None):
        metadata = {"render_modes": ["human"], "render_fps": 5}
        super(QuadEnv, self).__init__()
        """
        action space:
            * left propeller active, right propeller active, neither, both
        """
        self._action_space = gym.spaces.Discrete(4)
        """
         observation: 
             * Encoders: u1/u2
             * IMU: acc_x, acc_y, omega
             * R^5: [0.0,0.0, unbounded, unbounded, unbounded] [50, 50, unbounded, unbounded, unbounded]
        """
        self._observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, np.inf, np.inf, np.inf]),
            high=np.array([50.0, 50.0, np.inf, np.inf, np.inf]),
            dtype=np.float32,
        )
        self.quad = Quadcopter2d()
        self.renderer = QuadRender(self.quad)
        self.render_mode = render_mode

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.quad.reset()
        return self._get_obs_info()

    def _get_obs_info(self):
        u1, u2, acc_x, acc_y, acc_ang, time = self.quad.get_agent_state()
        reward = -1 * (acc_x + acc_y + acc_ang) / 3 + (time / 10)
        return [u1, u2, acc_x, acc_y, acc_ang], reward

    def step(self, action):
        direction = self.quad.current_action[action]
        self.quad.set_input(direction)
        self.quad.update()
        obs, reward = self._get_obs_info()
        terminated = self.quad.crash()
        return obs, reward, terminated, False, reward

    def render(self):
        if self.render_mode == "human":
            frame = int(self.quad.time_alive / self.renderer.dt)
            self.renderer.render(frame)
            time.sleep(self.renderer.dt)

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
