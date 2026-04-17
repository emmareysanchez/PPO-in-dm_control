from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo, TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from jimena.src.sac.reward import WalkerRewardShaping


@dataclass
class EnvSpec:
    env_id: str = "Walker2d-v5"
    frame_stack: int = 3
    action_repeat: int = 1
    time_limit: int = 1000
    obs_h: int = 64
    obs_w: int = 64
    grayscale: bool = False
    reward_shaping: bool = True
    terminate_when_unhealthy: bool = True
    healthy_z_range: tuple[float, float] = field(default_factory=lambda: (0.8, 2.0))


def resolve_env_spec(cfg: dict[str, Any]) -> EnvSpec:
    env_cfg = cfg["environment"]
    hz = env_cfg.get("healthy_z_range", [0.8, 2.0])
    return EnvSpec(
        env_id=str(env_cfg.get("env_id", "Walker2d-v5")),
        frame_stack=int(env_cfg.get("frame_stack", 3)),
        action_repeat=int(env_cfg.get("action_repeat", 1)),
        time_limit=int(env_cfg.get("time_limit", 1000)),
        obs_h=int(env_cfg.get("obs_h", 64)),
        obs_w=int(env_cfg.get("obs_w", 64)),
        grayscale=bool(env_cfg.get("grayscale", True)),
        reward_shaping=bool(env_cfg.get("reward_shaping", True)),
        terminate_when_unhealthy=bool(env_cfg.get("terminate_when_unhealthy", True)),
        healthy_z_range=(float(hz[0]), float(hz[1])),
    )

class PixelStackWrapper(gym.Wrapper):
    """
    Render -> resize -> stack K RGB frames on channel axis.
    Output shape: (3*K, H, W), dtype uint8, range [0, 255].
    """

    def __init__(
        self,
        env: gym.Env,
        k: int = 3,
        height: int = 64,
        width: int = 64,
        grayscale: bool = False,   # lo dejamos por compatibilidad, pero no lo usaremos
        action_repeat: int = 1,
    ) -> None:
        super().__init__(env)
        self.k = int(k)
        self.height = int(height)
        self.width = int(width)
        self.action_repeat = int(action_repeat)

        self._frames: deque[np.ndarray] = deque(maxlen=self.k)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3 * self.k, self.height, self.width),
            dtype=np.uint8,
        )

    def _get_frame(self) -> np.ndarray:
        frame = self.unwrapped.render()
        if frame is None:
            raise RuntimeError(
                "env.render() returned None. Use render_mode='rgb_array'."
            )

        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        frame = self._get_frame()
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(frame)
        return self._get_obs(), info

    def step(self, action):
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


def build_env(spec: EnvSpec, seed: int) -> gym.Env:
    env = gym.make(
        spec.env_id,
        render_mode="rgb_array",
        terminate_when_unhealthy=spec.terminate_when_unhealthy,
        healthy_z_range=spec.healthy_z_range,
    )
    env.reset(seed=int(seed))
    env.action_space.seed(int(seed))

    env = gym.wrappers.RecordEpisodeStatistics(env)

    if spec.reward_shaping:
        env = WalkerRewardShaping(env)

    env = PixelStackWrapper(
        env=env,
        k=spec.frame_stack,
        height=spec.obs_h,
        width=spec.obs_w,
        grayscale=spec.grayscale,
        action_repeat=spec.action_repeat,
    )
    # env = TimeLimit(env, max_episode_steps=spec.time_limit)
    return env


def make_train_env(spec: EnvSpec, seed: int):
    def _make():
        return build_env(spec=spec, seed=seed)

    return VecMonitor(DummyVecEnv([_make]))


def make_eval_env(spec: EnvSpec, seed: int, video_dir: str) -> gym.Env:
    env = build_env(spec=spec, seed=seed)
    return RecordVideo(
        env=env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True,
        name_prefix="eval",
        disable_logger=True,
    )
