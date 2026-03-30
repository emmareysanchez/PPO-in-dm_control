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
    # Usamos directamente la recompensa acumulada del entorno, que ya incluye
    # reward_survive + reward_forward + reward_ctrl sumados sobre todos los
    # substeps del action_repeat. Recomponerla desde info solo captura el
    # último substep y sesga el entrenamiento.
    return float(env_reward)


# alias por compatibilidad
walker2d_default_reward = walker2d_reward