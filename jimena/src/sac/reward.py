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
    reward_forward = float(info.get("reward_forward", 0.0))
    reward_survive = float(info.get("reward_survive", 0.0))
    reward_ctrl = float(info.get("reward_ctrl", 0.0))

    data = env.unwrapped.data
    z = float(data.qpos[1])
    ang = float(data.qpos[2])

    fall_penalty = -3.0 if (terminated and not truncated) else 0.0
    posture_penalty = -0.3 * abs(ang)
    low_height_penalty = -0.5 if z < 0.95 else 0.0

    reward = (
        1.2 * reward_forward
        + 1.2 * reward_survive
        + 0.5 * reward_ctrl
        + posture_penalty
        + low_height_penalty
        + fall_penalty
    )

    return float(reward)


walker2d_default_reward = walker2d_reward