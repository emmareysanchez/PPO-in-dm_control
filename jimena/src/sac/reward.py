from __future__ import annotations

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
    return float(env_reward)


walker2d_default_reward = walker2d_reward