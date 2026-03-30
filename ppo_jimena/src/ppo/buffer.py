from __future__ import annotations

import torch


class RolloutBuffer:
    def __init__(
        self,
        size: int,
        obs_shape: tuple[int, ...],
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.size = int(size)
        self.device = device
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)

        self.obs = torch.zeros((self.size, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros(self.size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(self.size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(self.size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(self.size, dtype=torch.float32, device=device)
        self.values = torch.zeros(self.size, dtype=torch.float32, device=device)

        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if self.ptr >= self.size:
            raise RuntimeError("RolloutBuffer is full. Call reset() after update().")

        self.obs[self.ptr] = obs
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob.detach().view(())
        self.values[self.ptr] = value.detach().view(())
        self.ptr += 1

    def compute_returns_advantages(self, last_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(self.rewards)
        last_adv = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        last_val = last_value.detach().view(())

        for t in reversed(range(self.ptr)):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * last_val * mask - self.values[t]
            last_adv = delta + self.gamma * self.gae_lambda * mask * last_adv
            advantages[t] = last_adv
            last_val = self.values[t]

        returns = advantages[: self.ptr] + self.values[: self.ptr]
        return advantages[: self.ptr], returns

    def reset(self) -> None:
        self.ptr = 0