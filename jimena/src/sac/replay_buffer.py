from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class StackedReplayBuffer(ReplayBuffer):
    """
    Replay buffer that stores single frames and reconstructs stacked
    observations on-the-fly at sample time.

    Memory saving vs standard ReplayBuffer:
        standard : buffer_size × K × C × H × W
        this     : buffer_size × C × H × W        (K× smaller)

    where K = frame_stack, C = channels per frame (1 or 3).

    The environment must still wrap observations in a PixelStackWrapper so
    that SB3 knows the stacked observation shape. We override add() to
    un-stack before storing and sample() to re-stack before returning.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Box,
        action_space: spaces.Space,
        frame_stack: int,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        **kwargs,
    ) -> None:
        self.frame_stack = int(frame_stack)

        # Build a single-frame observation space for internal storage
        stacked_shape = observation_space.shape          # (C*K, H, W)
        c_total, h, w = stacked_shape
        assert c_total % self.frame_stack == 0, (
            f"obs channels ({c_total}) must be divisible by frame_stack ({self.frame_stack})"
        )
        self.c_per_frame = c_total // self.frame_stack   # channels of one frame
        single_frame_space = spaces.Box(
            low=0, high=255,
            shape=(self.c_per_frame, h, w),
            dtype=np.uint8,
        )

        super().__init__(
            buffer_size=buffer_size,
            observation_space=single_frame_space,   # store single frames
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            **kwargs,
        )

        # Keep a reference to the real observation space for sampling
        self.stacked_obs_space = observation_space

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unstack(self, obs: np.ndarray) -> np.ndarray:
        """(n_envs, C*K, H, W) -> last frame -> (n_envs, C, H, W)"""
        return obs[:, -self.c_per_frame:, :, :]

    def _stack_from_buffer(self, indices: np.ndarray, use_next: bool = False) -> np.ndarray:
        """
        Reconstruct stacked observations for a batch of indices.
        Returns (batch, C*K, H, W).
        """
        batch_size = len(indices)
        c, h, w = self.c_per_frame, self.observations.shape[-2], self.observations.shape[-1]
        stacked = np.zeros((batch_size, c * self.frame_stack, h, w), dtype=np.uint8)

        for k in range(self.frame_stack):
            offset = self.frame_stack - 1 - k       # 0 = oldest, K-1 = newest
            past_indices = (indices - offset) % self.buffer_size

            # Don't look across episode boundaries: if the episode resets
            # between past_idx and current idx, repeat the oldest valid frame
            if use_next:
                frames = self.next_observations[past_indices, 0]   # (batch, C, H, W)
            else:
                frames = self.observations[past_indices, 0]        # (batch, C, H, W)

            stacked[:, k * c:(k + 1) * c] = frames

        return stacked

    # ------------------------------------------------------------------
    # Override add — un-stack before storing
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict],
    ) -> None:
        # Store only the most recent frame
        super().add(
            self._unstack(obs),
            self._unstack(next_obs),
            action,
            reward,
            done,
            infos,
        )

    # ------------------------------------------------------------------
    # Override sample — re-stack at sample time
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> ReplayBufferSamples:
        indices = np.random.randint(0, self.buffer_size if self.full else self.pos, size=batch_size)

        obs      = self._stack_from_buffer(indices, use_next=False)
        next_obs = self._stack_from_buffer(indices, use_next=True)

        # Normalise to [0, 1] as SB3 CnnPolicy expects
        obs_t      = torch.as_tensor(obs,      dtype=torch.float32, device=self.device) / 255.0
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device) / 255.0

        actions  = torch.as_tensor(self.actions[indices,  0], device=self.device)
        rewards  = torch.as_tensor(self.rewards[indices,  0], device=self.device)
        dones    = torch.as_tensor(self.dones[indices,    0], device=self.device)

        return ReplayBufferSamples(
            observations=obs_t,
            actions=actions,
            next_observations=next_obs_t,
            dones=dones,
            rewards=rewards,
        )