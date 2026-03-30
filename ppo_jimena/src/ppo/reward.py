from __future__ import annotations

import gymnasium as gym
import numpy as np


def _first_float(info: dict, keys: list[str]) -> float | None:
    for k in keys:
        if k in info:
            return float(info[k])
    return None


def _body_xy(info: dict, body: str) -> tuple[float, float] | None:
    xk = f"{body}_x"
    yk = f"{body}_y"
    if xk in info and yk in info:
        return float(info[xk]), float(info[yk])
    return None


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
    healthy_reward = float(info.get("reward_survive", 0.0))
    forward_reward = float(info.get("reward_forward", 0.0))
    ctrl_cost = float(info.get("reward_ctrl", 0.0))  # ya viene negativo

    r = 0.0
    r += 1.0 * healthy_reward   # suficiente señal para aprender a no caer
    r += 1.0 * forward_reward   # prioriza avance pero sin ser agresivo
    r += 1.0 * ctrl_cost        # penaliza torque excesivo (ya negativo)

    return float(r)


# alias por compatibilidad
walker2d_default_reward = walker2d_reward