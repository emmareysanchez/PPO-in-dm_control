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

    data = env.unwrapped.data
    z = float(data.qpos[1])      # torso height
    ang = float(data.qpos[2])    # torso pitch
    vz = float(data.qvel[1])     # vertical velocity

    fall_penalty = -10.0 if (terminated and not truncated) else 0.0

    # Penaliza saltar/rebotar
    vertical_penalty = -0.5 * abs(vz)

    # Penaliza posturas muy inclinadas
    angle_penalty = -1.0 * max(0.0, abs(ang) - 0.4)

    # Penaliza bajar demasiado el torso
    low_height_penalty = -2.0 if z < 0.95 else 0.0

    reward = (
        2.5 * reward_forward
        + 0.5 * reward_survive
        + 0.5 * reward_ctrl
        + vertical_penalty
        + angle_penalty
        + low_height_penalty
        + fall_penalty
    )

    return float(reward)


walker2d_default_reward = walker2d_reward