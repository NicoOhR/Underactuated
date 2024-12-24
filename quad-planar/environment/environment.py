from typing import Optional
import numpy as np
import gymnasium as gym
from quadcopter import Quadcopter2d
from render import QuadRender


class QuadEnv(gym.Env):
    def __init__(self):
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

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.quad.reset()
        return self._get_obs_info()

    def _get_obs_info(self):
        u1, u2, acc_x, acc_y, acc_ang, time = self.quad.get_agent_state()
        reward = -1 * (acc_x + acc_y + acc_ang) / 3 + (time / 10)
        return [u1, u2, acc_x, acc_y, acc_ang], reward

    def step(self, action):
        obs, reward = self._get_obs_info()
        direction = self.quad.current_action[action]
        terminated = self.quad.crash()
        return obs, reward, terminated, False, reward
