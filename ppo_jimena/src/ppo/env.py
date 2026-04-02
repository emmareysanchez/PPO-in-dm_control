from __future__ import annotations

from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo, TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from ppo_jimena.src.ppo.reward import walker2d_reward


@dataclass
class EnvSpec:
    env_id: str
    frame_stack: int
    action_repeat: int
    time_limit: int
    obs_h: int
    obs_w: int
    grayscale: bool = False
    action_prototypes: list[list[float]] | None = None


class PixelObservationWrapper(gym.Wrapper):
    """Replace state observations with rendered pixels."""

    def __init__(
        self,
        env: gym.Env,
        height: int = 84,
        width: int = 84,
        grayscale: bool = False,
    ) -> None:
        super().__init__(env=env)
        self.height = int(height)
        self.width = int(width)
        self.grayscale = bool(grayscale)

        channels = 1 if self.grayscale else 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(channels, self.height, self.width),
            dtype=np.uint8,
        )

    def _get_obs(self) -> np.ndarray:
        frame = self.unwrapped.render()
        if frame is None:
            raise RuntimeError("env.render() returned None. Use render_mode='rgb_array'.")

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return frame[None, :, :].astype(np.uint8)

        return np.transpose(frame, (2, 0, 1)).astype(np.uint8)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        _, info = self.env.reset(**kwargs)
        return self._get_obs(), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        _, reward, terminated, truncated, info = self.env.step(action)
        next_obs = self._get_obs()
        
        reward = walker2d_reward(
            obs=None,
            action=action,
            next_obs=next_obs,
            terminated=terminated,
            truncated=truncated,
            info=info,
            env_reward=float(reward),
            env=self.env,
        )
        
        return next_obs, float(reward), bool(terminated), bool(truncated), info


class FrameStack(gym.Wrapper):
    """Stack the k most recent frames on the channel axis."""

    def __init__(self, env: gym.Env, k: int) -> None:
        super().__init__(env=env)
        self.k = int(k)
        self.frames: list[np.ndarray] | None = None
        c, h, w = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(int(c) * self.k, int(h), int(w)),
            dtype=np.uint8,
        )

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs.copy() for _ in range(self.k)]
        return self._get_obs(), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert self.frames is not None
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def _get_obs(self) -> np.ndarray:
        assert self.frames is not None
        return np.concatenate(self.frames, axis=0)


def _build_env_stack(spec: EnvSpec, seed: int) -> gym.Env:
    env = gym.make(spec.env_id, render_mode="rgb_array")
    env.reset(seed=int(seed))
    env.action_space.seed(int(seed))

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = PixelObservationWrapper(
        env=env,
        height=int(spec.obs_h),
        width=int(spec.obs_w),
        grayscale=bool(spec.grayscale),
    )

    if int(spec.frame_stack) > 1:
        env = FrameStack(env=env, k=int(spec.frame_stack))

    env = TimeLimit(env=env, max_episode_steps=int(spec.time_limit))
    return env


def _make_env_fn(spec: EnvSpec, seed: int):
    def _thunk():
        return _build_env_stack(spec=spec, seed=seed)

    return _thunk


def make_train_env(spec: EnvSpec, seed: int, n_envs: int = 4):
    if int(n_envs) <= 1:
        env = DummyVecEnv([_make_env_fn(spec=spec, seed=seed)])
    else:
        env_fns = [_make_env_fn(spec=spec, seed=seed + i) for i in range(int(n_envs))]
        env = SubprocVecEnv(env_fns)

    env = VecMonitor(env)
    return env


def make_eval_env(spec: EnvSpec, seed: int, video_dir: str) -> gym.Env:
    env = _build_env_stack(spec=spec, seed=seed)
    env = RecordVideo(
        env=env,
        video_folder=video_dir,
        episode_trigger=lambda ep: True,
        name_prefix="eval",
        disable_logger=True,
    )
    return env
