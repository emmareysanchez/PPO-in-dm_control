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
    x_vel = float(data.qvel[0])   # velocidad hacia delante
    action = np.asarray(action, dtype=np.float32)

    forward_bonus = 0.20 * np.tanh(x_vel)
    posture_penalty = -0.03 * abs(ang)
    low_height_penalty = -0.05 if z < 0.80 else 0.0
    action_penalty = -0.001 * float(np.square(action).mean())

    reward = (
        float(env_reward)
        + forward_bonus
        + posture_penalty
        + low_height_penalty
        + action_penalty
    )

    info["reward/base"] = float(env_reward)
    info["reward/forward_bonus"] = float(forward_bonus)
    info["reward/posture_penalty"] = float(posture_penalty)
    info["reward/low_height_penalty"] = float(low_height_penalty)
    info["reward/action_penalty"] = float(action_penalty)

    return float(reward)


walker2d_default_reward = walker2d_reward