from __future__ import annotations

import gymnasium as gym
import numpy as np


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
    reward_forward = float(info.get("reward_forward", 0.0))
    reward_survive = float(info.get("reward_survive", 0.0))
    reward_ctrl = float(info.get("reward_ctrl", 0.0))

    fall_penalty = -3.0 if (terminated and not truncated) else 0.0

    reward = (
        2.3 * reward_forward
        + 0.8 * reward_survive
        + 0.05 * reward_ctrl
        + fall_penalty
    )
    return float(reward)


walker2d_default_reward = walker2d_reward