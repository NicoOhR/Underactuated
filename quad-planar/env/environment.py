import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import gymnasium as gym
from env.quadcopter import Quad2d
from env.render import QuadRender


class QuadEnv(gym.Env):
    metadata: dict[str, list[str]] = {"render_modes": ["human"]}
    action_space: gym.spaces.Box
    observation_space: gym.spaces.Box
    quad: Quad2d
    render_mode: str | None
    renderer: QuadRender

    def __init__(self, render_mode: str | None = None) -> None:
        super(QuadEnv, self).__init__()
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float64
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

    def reset(self, **kwargs) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
        super().reset(**kwargs)
        self.quad.reset()
        state, reward = self._get_obs_info()
        info: dict[str, float] = {"reward": reward}
        return (np.array(state, dtype=np.float64), info)

    def _get_obs_info(self) -> tuple[list[float], float]:
        x, y, theta, vx, vy, omega = self.quad.y
        reward: float = 1 - math.sqrt((1 - x) ** 2 + (1 - y) ** 2)
        if self.quad.crash():
            self.quad.t = 0
            reward -= 1
        return ([x, y, theta, vx, vy, omega], reward)

    def step(
        self, action: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], float, bool, bool, dict[str, float]]:
        action = np.clip(action, 0, 1)
        f1, f2 = action
        self.quad.set_input(f1, f2)
        self.quad.update()
        obs, reward = self._get_obs_info()
        info: dict[str, float] = {"reward": reward}
        terminated: bool = self.quad.crash()
        truncated: bool
        if math.isclose(self.quad.t, 10.0):
            self.quad.t = 0
            truncated = True
        else:
            truncated = False
        if self.render_mode == "human":
            frame: int = int(self.quad.t / self.renderer.dt)
            self.renderer.render(frame)

        return np.array(obs), reward, terminated, truncated, info

    def _init_render(self) -> None:
        self.renderer = QuadRender(self.quad)

    def close(self) -> None:
        if self.renderer:
            plt.close(self.renderer.fig)


if __name__ == "__main__":

    def main() -> None:
        test_env = QuadEnv(render_mode="human")
        obs, info = test_env.reset()
        done: bool = False
        while not done:
            obs, reward, terminated, truncated, info = test_env.step(action=(3, 5))
            done = terminated or truncated
        test_env.close()

    try:
        main()
    except KeyboardInterrupt:
        print("exit early")
