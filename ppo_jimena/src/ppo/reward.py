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

    # Penalización fuerte si se cae
    fall_penalty = -10.0 if (terminated and not truncated) else 0.0

    reward = (
        3.0 * reward_forward      # avanzar es lo más importante
        + 0.5 * reward_survive    # mantenerse vivo ayuda pero menos
        + 0.5 * reward_ctrl       # ya es negativo normalmente
        + fall_penalty
    )

    return float(reward)


walker2d_default_reward = walker2d_reward