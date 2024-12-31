from gymnasium.envs.registration import register
from .quadcopter import Quad2d

register(
    id="QuadEnv-v0",
    entry_point="env.environment:QuadEnv",
    kwargs=None,
    max_episode_steps=100,
)
