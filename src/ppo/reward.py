from __future__ import annotations

from typing import Any


def environment_reward(timestep: Any) -> float:
    """Return the reward currently produced by the environment.

    The original implementation used the reward emitted by dm_control without
    any extra shaping. This helper makes that choice explicit.

    Args:
        timestep: dm_control timestep returned by ``env.step``.

    Returns:
        Scalar environment reward.
    """
    reward: float = float(timestep.reward or 0.0)
    return reward
