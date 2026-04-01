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
    reward_ctrl = float(info.get("reward_ctrl", 0.0))
    reward_survive = float(info.get("reward_survive", 0.0))

    return (
    1.5 * reward_forward
    + 1.0 * reward_survive
    + 0.1 * reward_ctrl
)


walker2d_default_reward = walker2d_reward