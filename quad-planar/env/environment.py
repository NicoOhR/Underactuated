import math
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from env.quadcopter import Quad2d
from env.render import QuadRender


class QuadEnv(gym.Env):
    def __init__(self, render_mode=None):
        metadata = {"render_modes": ["human"]}
        super(QuadEnv, self).__init__()
        self.action_space = gym.spaces.Box(
            low=0.0, high=5.0, shape=(2,), dtype=np.float64
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float64,
        )
        self.quad = Quad2d()
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
        vx, vy, omega, acc_x, acc_y, alpha = self.quad.dynamics(0, self.quad.y)
        x, y, theta, vx, vy, omega = self.quad.y
        reward = max(0, 1 - math.sqrt((1 - x) ** 2 + (1 - y) ** 2))
        if self.quad.crash():
            reward -= 1
        return ([vx, vy, omega, acc_x, acc_y, alpha], reward)

    def step(self, action):
        f1, f2 = action
        self.quad.set_input(f1, f2)
        self.quad.update()
        obs, reward = self._get_obs_info()
        info = {"reward": reward}
        terminated = self.quad.crash()
        if math.isclose(self.quad.t, 10.0):
            truncated = True
        else:
            truncated = False
        if self.render_mode == "human":
            frame = int(self.quad.t / self.renderer.dt)
            self.renderer.render(frame)

        return np.array(obs), reward, terminated, truncated, info

    def _init_render(self):
        self.renderer = QuadRender(self.quad)

    def close(self):
        if self.renderer:
            plt.close(self.renderer.fig)


if __name__ == "__main__":

    def main():
        test_env = QuadEnv(render_mode="human")
        obs, info = test_env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = test_env.step(action=(3, 5))
            done = terminated or truncated
        test_env.close()

    try:
        main()
    except KeyboardInterrupt:
        print("exit early")
