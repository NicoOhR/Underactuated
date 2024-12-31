import math
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


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
        vx, vy, acc_x, acc_y, acc_ang, _ = self.quad.dynamics(0, self.quad.y)
        reward = -math.sqrt(acc_x**2 + acc_y**2 + acc_ang**2) - math.sqrt(vx**2 + vy**2)
        return ([vx, vy, acc_x, acc_y, acc_ang], reward)

    def step(self, action):
        f1, f2 = action
        self.quad.set_input(f1, f2)
        self.quad.update()
        obs, reward = self._get_obs_info()
        info = {"reward": reward}
        terminated = self.quad.crash()

        if self.render_mode == "human":
            frame = int(self.quad.t / self.renderer.dt)
            self.renderer.render(frame)

        return np.array(obs), reward, terminated, False, info

    def _init_render(self):
        self.renderer = QuadRender(self.quad)

    def close(self):
        if self.renderer:
            plt.close(self.renderer.fig)


if __name__ == "__main__":
    from env.quadcopter import Quad2d
    from env.render import QuadRender

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
        print("Program Interrupted")
