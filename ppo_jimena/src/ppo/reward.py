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

    torso_y = float(info.get("torso_y", 1.25))
    head_y = float(info.get("head_y", 1.45))

    fall_penalty = -10.0 if (terminated and not truncated) else 0.0
    low_torso_penalty = -2.0 if torso_y < 0.95 else 0.0
    low_head_penalty = -1.0 if head_y < 1.15 else 0.0

    reward = (
        3.0 * reward_forward
        + 0.5 * reward_survive
        + 0.5 * reward_ctrl
        + fall_penalty
        + low_torso_penalty
        + low_head_penalty
    )

    return float(reward)


walker2d_default_reward = walker2d_reward
