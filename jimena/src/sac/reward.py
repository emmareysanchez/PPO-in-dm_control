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
    data = env.unwrapped.data

    z = float(data.qpos[1])
    ang = float(data.qpos[2])
    x_vel = float(data.qvel[0])
    action = np.asarray(action, dtype=np.float32)

    forward_bonus = 0.20 * np.tanh(x_vel)
    posture_penalty = -0.03 * abs(ang)
    low_height_penalty = -0.05 if z < 0.80 else 0.0
    action_penalty = -0.001 * float(np.square(action).mean())

    reward = (
        float(env_reward)
        + forward_bonus
        + posture_penalty
        + low_height_penalty
        + action_penalty
    )

    info["reward/base"] = float(env_reward)
    info["reward/shaped"] = float(reward)
    info["reward/forward_bonus"] = float(forward_bonus)
    info["reward/posture_penalty"] = float(posture_penalty)
    info["reward/low_height_penalty"] = float(low_height_penalty)
    info["reward/action_penalty"] = float(action_penalty)

    return float(reward)


walker2d_default_reward = walker2d_reward


class WalkerRewardShaping(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_fn=walker2d_default_reward):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        next_obs, env_reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = self.reward_fn(
            obs=self._last_obs,
            action=action,
            next_obs=next_obs,
            terminated=terminated,
            truncated=truncated,
            info=info,
            env_reward=env_reward,
            env=self.env,
        )

        self._last_obs = next_obs
        return next_obs, float(shaped_reward), terminated, truncated, info