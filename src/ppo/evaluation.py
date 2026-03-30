from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from src.ppo.environment import Environment


@dataclass
class EvaluationResult:
    """Aggregated evaluation metrics."""

    mean_reward: float
    mean_length: float
    mean_distance: float
    mean_avg_speed: float


class Evaluator:
    """Evaluation helper for PPO runs."""

    def __init__(
        self,
        config: dict[str, Any],
        videos_dir: Path,
    ) -> None:
        """Initialize the evaluator.

        Args:
            config: Full experiment configuration.
            videos_dir: Root directory for evaluation videos of this run.
        """
        self.config: dict[str, Any] = config
        self.videos_dir: Path = videos_dir
        self.episodes: int = int(config["evaluation"]["episodes"])
        self.video_fps: int = int(config["evaluation"].get("video_fps", 30))
        self.video_macro_block_size: int = int(
            config["evaluation"].get("video_macro_block_size", 16)
        )

    def _deterministic_action(
        self,
        actor: torch.nn.Module,
        observation: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        """Compute a deterministic action using the policy mean.

        Args:
            actor: Policy network.
            observation: Stacked observation.
            device: Torch device.

        Returns:
            Continuous action vector.
        """
        observation_tensor: torch.Tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            mu, _ = actor(observation_tensor)
        return mu.squeeze(0).cpu().numpy().astype(np.float32)

    def _evaluation_video_dir(self, global_step: int) -> Path:
        """Create the directory for one evaluation phase.

        Args:
            global_step: Training step at which evaluation is executed.

        Returns:
            Directory path for the evaluation videos of that step.
        """
        evaluation_dir: Path = self.videos_dir / f"{global_step:010d}"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        return evaluation_dir

    def _pad_frame_for_video(self, frame: np.ndarray) -> np.ndarray:
        """Pad a frame so its spatial dimensions are video-codec friendly.

        The training pipeline can operate on arbitrary frame sizes, but many
        video codecs expect height and width to be divisible by a macro block
        size, commonly 16. Padding prevents the encoder from applying an
        implicit resize.

        Args:
            frame: RGB frame with shape [H, W, 3].

        Returns:
            RGB frame padded on the bottom and right when needed.
        """
        macro_block_size: int = self.video_macro_block_size
        if macro_block_size <= 1:
            return frame

        height: int = int(frame.shape[0])
        width: int = int(frame.shape[1])
        target_height: int = (
            (height + macro_block_size - 1) // macro_block_size
        ) * macro_block_size
        target_width: int = (
            (width + macro_block_size - 1) // macro_block_size
        ) * macro_block_size

        pad_bottom: int = target_height - height
        pad_right: int = target_width - width
        if pad_bottom == 0 and pad_right == 0:
            return frame

        padded_frame: np.ndarray = np.pad(
            frame,
            pad_width=((0, pad_bottom), (0, pad_right), (0, 0)),
            mode="edge",
        )
        return padded_frame

    def _save_video(self, frames: list[np.ndarray], output_path: Path) -> None:
        """Save an evaluation video.

        Args:
            frames: Sequence of RGB frames with shape [H, W, 3].
            output_path: Destination path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_frames: list[np.ndarray] = [
            self._pad_frame_for_video(frame=frame) for frame in frames
        ]
        imageio.mimsave(
            output_path,
            processed_frames,
            fps=self.video_fps,
            macro_block_size=self.video_macro_block_size,
        )

    def evaluate(
        self,
        actor: torch.nn.Module,
        device: torch.device,
        global_step: int,
    ) -> EvaluationResult:
        """Run evaluation episodes and save videos.

        Args:
            actor: Policy network.
            device: Torch device.
            global_step: Current training step.

        Returns:
            Aggregated evaluation metrics.
        """
        eval_env: Environment = Environment(config=self.config)
        evaluation_dir: Path = self._evaluation_video_dir(global_step=global_step)

        rewards: list[float] = []
        lengths: list[float] = []
        distances: list[float] = []
        avg_speeds: list[float] = []

        for episode_idx in range(self.episodes):
            observation, _ = eval_env.reset()
            done: bool = False
            frames: list[np.ndarray] = []

            while not done:
                rgb_frame: np.ndarray = eval_env.env.physics.render(
                    height=eval_env.height,
                    width=eval_env.width,
                    camera_id=eval_env.camera_id,
                )
                frames.append(rgb_frame)
                action: np.ndarray = self._deterministic_action(
                    actor=actor,
                    observation=observation,
                    device=device,
                )
                step_result = eval_env.step(action=action)
                observation = step_result.observation
                done = step_result.done

            rewards.append(step_result.info["episode_reward"])
            lengths.append(step_result.info["episode_step"])
            distances.append(step_result.info["episode_distance"])
            avg_speeds.append(step_result.info["episode_avg_speed"])

            video_path: Path = evaluation_dir / f"episode_{episode_idx + 1:02d}.mp4"
            self._save_video(
                frames=frames,
                output_path=video_path,
            )

        return EvaluationResult(
            mean_reward=float(np.mean(rewards)),
            mean_length=float(np.mean(lengths)),
            mean_distance=float(np.mean(distances)),
            mean_avg_speed=float(np.mean(avg_speeds)),
        )
