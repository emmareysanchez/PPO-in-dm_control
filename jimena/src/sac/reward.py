from __future__ import annotations

import gymnasium as gym
import numpy as np


def walker2d_reward(
    obs,
    action,
    next_obs,
    terminated,
    truncated,
    info,
    env_reward,
    env,
) -> float:
    reward_forward = float(info.get("reward_forward", 0.0))
    reward_ctrl = float(info.get("reward_ctrl", 0.0))

    data = env.unwrapped.data
    ang = float(data.qpos[2])   # torso angle
    z = float(data.qpos[1])     # height

    fall_penalty = -5.0 if (terminated and not truncated) else 0.0
    posture_penalty = -1.0 * abs(ang)
    low_height_penalty = -1.0 if z < 1.0 else 0.0

    print(f"[DEBUG info keys] {list(info.keys())}")
    print(f"[DEBUG] forward={reward_forward:.3f}, ctrl={reward_ctrl:.3f}")

    reward = (
        2.0 * reward_forward
        + 0.02 * reward_ctrl
        + posture_penalty
        + low_height_penalty
        + fall_penalty
    )
    return float(reward)


walker2d_default_reward = walker2d_reward