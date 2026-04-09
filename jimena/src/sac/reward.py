from __future__ import annotations

import gymnasium as gym
import numpy as np


class RewardShaping(gym.Wrapper):
    """
    Reward shaping para Walker2d-v5.
    Replica la lógica de ProgressWithSafetyShapingNew del código de referencia:

      shaped = survive * alive_weight + forward * speed_weight + ctrl
             - height_penalty - angle_penalty - smooth_penalty
    """

    def __init__(
        self,
        env: gym.Env,
        z_ref: float = 1.10,
        angle_ref: float = 0.7,
        w_z: float = 0.7,
        w_ang: float = 0.3,
        w_smooth: float = 0.0,
        alive_weight: float = 1.25,
        speed_weight: float = 0.75,
    ) -> None:
        super().__init__(env)
        self.z_ref = float(z_ref)
        self.angle_ref = float(angle_ref)
        self.w_z = float(w_z)
        self.w_ang = float(w_ang)
        self.w_smooth = float(w_smooth)
        self.alive_weight = float(alive_weight)
        self.speed_weight = float(speed_weight)
        self._prev_action: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        forward = float(info.get("reward_forward", 0.0)) * self.speed_weight
        survive = float(info.get("reward_survive", 0.0)) * self.alive_weight
        ctrl    = float(info.get("reward_ctrl",    0.0))

        shaped = survive + forward + ctrl

        data = self.env.unwrapped.data
        z   = float(data.qpos[1])
        ang = float(data.qpos[2])

        z_pen   = self.w_z   * max(0.0, self.z_ref - z)
        ang_pen = self.w_ang * max(0.0, abs(ang) - self.angle_ref)
        shaped -= z_pen + ang_pen

        info["debug/reward_forward"] = forward
        info["debug/reward_survive"] = survive
        info["debug/reward_ctrl"]    = ctrl
        info["debug/height_pen"]     = z_pen
        info["debug/angle_pen"]      = ang_pen

        if self.w_smooth > 0.0:
            a = np.asarray(action, dtype=np.float32)
            if self._prev_action is not None:
                smooth_pen = self.w_smooth * float(np.sum((a - self._prev_action) ** 2))
                shaped -= smooth_pen
                info["debug/smooth_pen"] = smooth_pen
            else:
                info["debug/smooth_pen"] = 0.0
            self._prev_action = a
        else:
            info["debug/smooth_pen"] = 0.0

        return obs, float(shaped), bool(terminated), bool(truncated), info


# ---------------------------------------------------------------------------
# Functional interface — kept for backwards compatibility with any code
# that still calls walker2d_reward() directly.
# ---------------------------------------------------------------------------

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
    """
    Functional shaping used when reward needs to be computed outside a wrapper.
    Reads directly from env.unwrapped.data so it works even without the wrapper.
    """
    data = env.unwrapped.data
    z   = float(data.qpos[1])
    ang = float(data.qpos[2])

    forward = float(info.get("reward_forward", 0.0)) * 0.75
    survive = float(info.get("reward_survive", 0.0)) * 1.25
    ctrl    = float(info.get("reward_ctrl",    0.0))

    z_pen   = 0.7 * max(0.0, 1.10 - z)
    ang_pen = 0.3 * max(0.0, abs(ang) - 0.7)

    return forward + survive + ctrl - z_pen - ang_pen


walker2d_default_reward = walker2d_reward