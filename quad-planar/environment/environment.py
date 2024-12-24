from typing import Optional
import numpy as np
import gymnasium as gym


class WorldEnv(gym.Env):
    def __init__(self):
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
