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

    # Penalizar caer (terminated sin truncated = caída)
    fall_penalty = -20.0 if (terminated and not truncated) else 0.0

    return (
        3.0 * reward_forward      # positivo si avanza, negativo si retrocede
        + 0.5 * reward_survive    # sobrevivir importa menos que avanzar
        + 0.5 * reward_ctrl       # ya negativo, penaliza torque
        + fall_penalty            # penalización explícita por caer
    )


walker2d_default_reward = walker2d_reward