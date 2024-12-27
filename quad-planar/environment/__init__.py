from gymnasium.envs.registration import register
from .quadcopter import Quadcopter2d

register(
    id="QuadEnv-v0",
    entry_point="environment.environment:QuadEnv",
    kwargs={"render_mode": "human"},
    max_episode_steps=100,
)
