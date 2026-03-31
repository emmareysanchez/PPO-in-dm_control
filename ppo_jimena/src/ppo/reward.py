from __future__ import annotations

import gymnasium as gym
import numpy as np


def walker2d_reward(
    obs: np.ndarray,
    action: int,
    next_obs: np.ndarray,
    terminated: bool,
    truncated: bool,
    info: dict,
    env_reward: float,
    env: gym.Env,
) -> float:
    # Use the environment reward directly: it already sums
    # reward_survive + reward_forward + reward_ctrl across all
    # action_repeat substeps, which is what we want.
    return float(env_reward)


walker2d_default_reward = walker2d_reward