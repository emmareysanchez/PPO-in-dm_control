from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo, TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from jimena.src.sac.reward import RewardShaping


@dataclass
class EnvSpec:
    env_id: str
    frame_stack: int
    action_repeat: int
    time_limit: int
    obs_h: int
    obs_w: int
    grayscale: bool = False
    reward_shaping: bool = True
    terminate_when_unhealthy: bool = True
    healthy_z_range: tuple[float, float] = field(default_factory=lambda: (0.8, 2.0))
    action_prototypes: list[list[float]] | None = None


class PixelStackWrapper(gym.Wrapper):
    """
    Render -> resize -> (optional grayscale) -> stack K frames on channel axis.
    Output shape: (C*K, H, W)  where C=1 (grayscale) or C=3 (RGB).

    Action repeat is handled here so reward accumulation happens
    before the frame is captured — matches the working reference implementation.
    """

    def __init__(
        self,
        env: gym.Env,
        k: int = 4,
        height: int = 84,
        width: int = 84,
        grayscale: bool = False,
        action_repeat: int = 1,
    ) -> None:
        super().__init__(env=env)
        self.k = int(k)
        self.height = int(height)
        self.width = int(width)
        self.grayscale = bool(grayscale)
        self.action_repeat = int(action_repeat)

        self._frames: deque[np.ndarray] = deque(maxlen=self.k)

        c = 1 if self.grayscale else 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(c * self.k, self.height, self.width),
            dtype=np.uint8,
        )

    def _get_frame(self) -> np.ndarray:
        frame = self.unwrapped.render()
        if frame is None:
            raise RuntimeError("env.render() returned None. Use render_mode='rgb_array'.")
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return frame[None, :, :].astype(np.uint8)           # (1, H, W)
        return np.transpose(frame, (2, 0, 1)).astype(np.uint8)  # (3, H, W)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        _, info = self.env.reset(**kwargs)
        frame = self._get_frame()
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(frame)
        return self._get_obs(), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.action_repeat):
            _, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        self._frames.append(self._get_frame())
        return self._get_obs(), total_reward, bool(terminated), bool(truncated), info


# ---------------------------------------------------------------------------
# Environment stack
# ---------------------------------------------------------------------------

def _build_env_stack(spec: EnvSpec, seed: int) -> gym.Env:
    env = gym.make(
        spec.env_id,
        render_mode="rgb_array",
        terminate_when_unhealthy=spec.terminate_when_unhealthy,
        healthy_z_range=spec.healthy_z_range,
    )
    env.reset(seed=int(seed))
    env.action_space.seed(int(seed))

    # 1. Statistics — must be BEFORE reward shaping so logged rewards are shaped
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # 2. Reward shaping — independent wrapper applied BEFORE pixels,
    #    matching the working reference (ProgressWithSafetyShapingNew)
    if spec.reward_shaping:
        env = RewardShaping(env)

    # 3. Pixels + frame stack + action repeat — unified wrapper
    env = PixelStackWrapper(
        env=env,
        k=int(spec.frame_stack),
        height=int(spec.obs_h),
        width=int(spec.obs_w),
        grayscale=bool(spec.grayscale),
        action_repeat=int(spec.action_repeat),
    )

    # 4. Hard episode length cap
    env = TimeLimit(env=env, max_episode_steps=int(spec.time_limit))
    return env


def _make_env_fn(spec: EnvSpec, seed: int) -> Callable[[], gym.Env]:
    def _thunk() -> gym.Env:
        return _build_env_stack(spec=spec, seed=seed)
    return _thunk


def make_train_env(spec: EnvSpec, seed: int, n_envs: int = 1):
    """
    SAC is off-policy — a single environment is standard.
    n_envs > 1 is supported but uncommon.
    """
    if int(n_envs) <= 1:
        env = DummyVecEnv([_make_env_fn(spec=spec, seed=seed)])
    else:
        env_fns = [_make_env_fn(spec=spec, seed=seed + i) for i in range(int(n_envs))]
        env = SubprocVecEnv(env_fns)
    return VecMonitor(env)


def make_eval_env(spec: EnvSpec, seed: int, video_dir: str) -> gym.Env:
    env = _build_env_stack(spec=spec, seed=seed)
    return RecordVideo(
        env=env,
        video_folder=video_dir,
        episode_trigger=lambda ep: True,
        name_prefix="eval",
        disable_logger=True,
    )