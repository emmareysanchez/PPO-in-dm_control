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

    torso_y = float(info.get("torso_y", 0.0))
    head_y = float(info.get("head_y", 0.0))

    # Strong penalty for falling
    fall_penalty = -20.0 if (terminated and not truncated) else 0.0

    # Penalize crouched / folded policies
    torso_posture_penalty = -3.0 * max(0.0, 1.0 - torso_y)
    head_posture_penalty = -1.5 * max(0.0, 1.35 - head_y)

    return (
        1.5 * reward_forward
        + 0.5 * reward_survive
        + 0.1 * reward_ctrl
        + torso_posture_penalty
        + head_posture_penalty
        + fall_penalty
    )


walker2d_default_reward = walker2d_reward