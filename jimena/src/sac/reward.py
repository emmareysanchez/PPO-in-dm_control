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

    z = float(data.qpos[1])
    ang = float(data.qpos[2])
    z_vel = float(data.qvel[1])

    posture_penalty = -0.05 * abs(ang)
    low_height_penalty = -0.10 if z < 0.75 else 0.0

    reward = float(env_reward) + posture_penalty + low_height_penalty
    return float(reward)

walker2d_default_reward = walker2d_reward