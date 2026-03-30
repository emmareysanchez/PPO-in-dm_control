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
    ctrl_cost = float(info.get("reward_ctrl", 0.0))

    x_velocity = float(info.get("x_velocity", 0.0))
    torso_dx = _first_float(info, ["torso_dx"])
    if torso_dx is not None:
        x_velocity = torso_dx

    fr = _body_xy(info, "foot_right")
    fl = _body_xy(info, "foot_left")

    feet_dist = 0.0
    if fr is not None and fl is not None:
        dx = fr[0] - fl[0]
        dy = fr[1] - fl[1]
        feet_dist = float(np.sqrt(dx * dx + dy * dy))

    foot_right_dx = float(info.get("foot_right_dx", 0.0))
    foot_left_dx = float(info.get("foot_left_dx", 0.0))
    torso_dx_val = float(info.get("torso_dx", x_velocity))
    feet_max_speed = max(foot_right_dx, foot_left_dx) - max(0.0, torso_dx_val)

    heel_right_y = float(info.get("heel_right_y", 0.0))
    heel_left_y = float(info.get("heel_left_y", 0.0))
    feet_height_reward = 0.25 * min(heel_right_y, heel_left_y)

    r = 0.0
    r += healthy_reward * 1.2
    r += forward_reward * 0.5
    r -= ctrl_cost

    # optional shaping
    # r += feet_height_reward
    # r += 0.02 * np.clip(x_velocity, -30.0, 30.0)
    # r += 0.30 * (np.clip(feet_dist, 0.0, 2.0) - 0.5)
    # r += 0.02 * np.clip(feet_max_speed, -30.0, 30.0)

    if not (terminated or truncated):
        r += 0.05

    return float(r)


# alias por compatibilidad
walker2d_default_reward = walker2d_reward