from __future__ import annotations

import gymnasium as gym
import numpy as np


class WalkerRewardShaping(gym.Wrapper):
    """
    Reward shaping para Walker2d-v5.
    Repesa los componentes nativos y penaliza posturas inestables.
    """

    def __init__(
        self,
        env: gym.Env,
        forward_scale: float = 0.75,
        survival_scale: float = 1.25,
        min_height: float = 1.10,
        height_penalty_scale: float = 0.7,
        max_tilt: float = 0.7,
        tilt_penalty_scale: float = 0.3,
    ):
        super().__init__(env)
        self.forward_scale = float(forward_scale)
        self.survival_scale = float(survival_scale)
        self.min_height = float(min_height)
        self.height_penalty_scale = float(height_penalty_scale)
        self.max_tilt = float(max_tilt)
        self.tilt_penalty_scale = float(tilt_penalty_scale)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        forward  = float(info.get("reward_forward", 0.0)) * self.forward_scale
        survival = float(info.get("reward_survive", 0.0)) * self.survival_scale
        ctrl     = float(info.get("reward_ctrl",    0.0))

        torso_height = float(self.env.unwrapped.data.qpos[1])
        torso_tilt   = float(self.env.unwrapped.data.qpos[2])

        height_penalty = self.height_penalty_scale * max(0.0, self.min_height - torso_height)
        tilt_penalty   = self.tilt_penalty_scale   * max(0.0, abs(torso_tilt) - self.max_tilt)

        shaped = forward + survival + ctrl - height_penalty - tilt_penalty

        info["shaping/forward"]        = forward
        info["shaping/survival"]       = survival
        info["shaping/ctrl"]           = ctrl
        info["shaping/height_penalty"] = height_penalty
        info["shaping/tilt_penalty"]   = tilt_penalty

        return obs, float(shaped), bool(terminated), bool(truncated), info
