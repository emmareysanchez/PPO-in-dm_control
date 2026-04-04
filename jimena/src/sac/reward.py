from __future__ import annotations

import numpy as np
import gymnasium as gym


def walker2d_reward(
    obs,
    action,
    next_obs,
    terminated,
    truncated,
    info,
    env_reward,
    env,
) -> float:

    data = env.unwrapped.data
    ang  = float(data.qpos[2])  # torso angle
    z    = float(data.qpos[1])  # height

    fall_penalty       = -5.0 if (terminated and not truncated) else 0.0
    posture_penalty    = -0.2 * abs(ang)
    low_height_penalty = -1.0 if z < 1.0 else 0.0

    reward = (
        env_reward              # recompensa original de MuJoCo, ya balanceada
        + posture_penalty
        + low_height_penalty
        + fall_penalty
    )

    return float(np.clip(reward, -10.0, 10.0))


walker2d_default_reward = walker2d_reward