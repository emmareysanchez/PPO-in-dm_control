from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from dm_control import suite

from src.ppo.reward import environment_reward


@dataclass
class StepResult:
    """Container for one environment transition."""

    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, float]
    frame: np.ndarray


class Environment:
    """Visual wrapper around a dm_control environment."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the environment wrapper.

        Args:
            config: Full experiment configuration.
        """
        env_config: dict[str, Any] = config["environment"]
        experiment_config: dict[str, Any] = config["experiment"]

        self.domain: str = str(env_config["domain"])
        self.task: str = str(env_config["task"])
        self.height: int = int(env_config["observation_height"])
        self.width: int = int(env_config["observation_width"])
        self.frame_stack: int = int(env_config["frame_stack"])
        self.action_repeat: int = int(env_config.get("action_repeat", 1))
        self.camera_id: int = int(env_config.get("camera_id", 0))
        self.grayscale: bool = bool(env_config.get("grayscale"))
        self.max_episode_steps: int = int(env_config.get("max_episode_steps", 1000))
        self.seed: int = int(experiment_config["seed"])

        self.random_state: np.random.RandomState = np.random.RandomState(self.seed)
        self.env: Any = suite.load(
            domain_name=self.domain,
            task_name=self.task,
            task_kwargs={"random": self.random_state},
        )
        self.action_spec: Any = self.env.action_spec()
        self.action_dim: int = int(self.action_spec.shape[0])
        self.observation_channels: int = 1 if self.grayscale else 3
        self.input_channels: int = self.observation_channels * self.frame_stack
        self.frame_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.frame_stack,
        )
        self.episode_step: int = 0
        self.episode_reward: float = 0.0
        self.start_x: float = 0.0
        self.current_x: float = 0.0

    def _render_frame(self) -> np.ndarray:
        """Render the current frame.

        Returns:
            Normalized frame with shape [C, H, W].
        """
        frame: np.ndarray = self.env.physics.render(
            height=self.height,
            width=self.width,
            camera_id=self.camera_id,
        )
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = frame.astype(np.float32) / 255.0
            frame = frame[None, :, :]
        else:
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))
        return frame

    def _stacked_observation(self) -> np.ndarray:
        """Build the stacked observation tensor.

        Returns:
            Stacked observation with shape [C * frame_stack, H, W].
        """
        return np.concatenate(list(self.frame_buffer), axis=0)

    def _root_x_position(self) -> float:
        """Estimate the horizontal root position.

        Returns:
            Estimated x position.
        """
        return float(self.env.physics.data.qpos[0])

    def reset(self) -> tuple[np.ndarray, dict[str, float]]:
        """Reset the environment.

        Returns:
            Initial stacked observation and reset info.
        """
        self.env.reset()
        self.episode_step = 0
        self.episode_reward = 0.0
        self.start_x = self._root_x_position()
        self.current_x = self.start_x

        frame: np.ndarray = self._render_frame()
        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(frame.copy())

        observation: np.ndarray = self._stacked_observation()
        info: dict[str, float] = {
            "episode_step": 0.0,
            "episode_reward": 0.0,
            "episode_distance": 0.0,
            "episode_avg_speed": 0.0,
        }
        return observation, info

    def step(self, action: np.ndarray) -> StepResult:
        """Apply one action with optional action repeat.

        Args:
            action: Continuous action vector.

        Returns:
            StepResult object.
        """
        clipped_action: np.ndarray = np.clip(
            action.astype(np.float32),
            self.action_spec.minimum,
            self.action_spec.maximum,
        )

        cumulative_reward: float = 0.0
        done: bool = False
        for _ in range(self.action_repeat):
            timestep: Any = self.env.step(clipped_action)
            reward: float = environment_reward(timestep=timestep)
            cumulative_reward += reward
            self.episode_step += 1
            done = bool(timestep.last()) or self.episode_step >= self.max_episode_steps
            if done:
                break

        self.episode_reward += cumulative_reward
        self.current_x = self._root_x_position()
        episode_distance: float = self.current_x - self.start_x
        episode_avg_speed: float = episode_distance / max(self.episode_step, 1)

        frame: np.ndarray = self._render_frame()
        self.frame_buffer.append(frame)
        observation: np.ndarray = self._stacked_observation()
        info: dict[str, float] = {
            "episode_step": float(self.episode_step),
            "episode_reward": float(self.episode_reward),
            "episode_distance": float(episode_distance),
            "episode_avg_speed": float(episode_avg_speed),
        }
        return StepResult(
            observation=observation,
            reward=float(cumulative_reward),
            done=done,
            info=info,
            frame=frame,
        )
