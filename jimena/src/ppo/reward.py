from __future__ import annotations

import gymnasium as gym
import numpy as np


def walker2d_reward(
    obs: np.ndarray,
    action,
    next_obs: np.ndarray,
    terminated: bool,
    truncated: bool,
    info: dict,
    env_reward: float,
    env: gym.Env,
) -> float:
    data = env.unwrapped.data

    z = float(data.qpos[1])      # torso height
    ang = float(data.qpos[2])    # torso pitch
    z_vel = float(data.qvel[1])  # vertical velocity

    # Small shaping only
    posture_penalty = -0.10 * abs(ang)
    bounce_penalty = -0.05 * abs(z_vel)

    # Very mild penalty only if it is clearly too low
    low_height_penalty = -0.20 if z < 0.80 else 0.0

    # Do not add a big explicit fall penalty here
    reward = (
        float(env_reward)
        + posture_penalty
        + bounce_penalty
        + low_height_penalty
    )

    return float(reward)


walker2d_default_reward = walker2d_reward