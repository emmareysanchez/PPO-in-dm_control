from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], hidden_dim: int) -> None:
        super().__init__()
        c, h, w = obs_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flat = self.conv(dummy).reshape(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(n_flat, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class ActorNet(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = ConvEncoder(obs_shape, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.head(x)


class CriticNet(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], hidden_dim: int) -> None:
        super().__init__()
        self.encoder = ConvEncoder(obs_shape, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.head(x).squeeze(-1)


class PPOAgent:
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        device: torch.device | str,
        actor_lr: float = 5e-5,
        critic_lr: float = 1e-4,
        gamma: float = 0.99,
        lambd: float = 0.95,
        k_epochs: int = 10,
        eps_clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 256,
        minibatch_size: int | None = None,
        **_: object,
    ) -> None:
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.lambd = float(lambd)
        self.k_epochs = int(k_epochs)
        self.eps_clip = float(eps_clip)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.minibatch_size = minibatch_size

        self.actor = ActorNet(obs_shape, n_actions, hidden_dim).to(self.device)
        self.critic = CriticNet(obs_shape, hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(critic_lr))

    def act(self, obs: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        obs_b = obs.unsqueeze(0)
        logits = self.actor(obs_b)
        value = self.critic(obs_b)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze(0), value.squeeze(0)

    def act_deterministic(self, obs: torch.Tensor) -> int:
        obs_b = obs.unsqueeze(0)
        logits = self.actor(obs_b)
        return int(torch.argmax(logits, dim=-1).item())

    def update(
        self,
        buffer,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        obs = buffer.obs[: buffer.ptr]
        actions = buffer.actions[: buffer.ptr]
        old_log_probs = buffer.log_probs[: buffer.ptr]
        advantages = advantages.detach()
        returns = returns.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = obs.size(0)
        mb_size = batch_size if self.minibatch_size is None else min(self.minibatch_size, batch_size)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        for _ in range(self.k_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, mb_size):
                mb_idx = indices[start : start + mb_size]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                logits = self.actor(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(mb_obs)
                value_loss = F.mse_loss(values, mb_returns)

                self.actor_optimizer.zero_grad()
                actor_loss_total = policy_loss - self.entropy_coef * entropy
                actor_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss_total = self.value_coef * value_loss
                critic_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        return {
            "policy_loss": float(sum(policy_losses) / max(1, len(policy_losses))),
            "value_loss": float(sum(value_losses) / max(1, len(value_losses))),
            "entropy": float(sum(entropies) / max(1, len(entropies))),
        }

    def save(self, path: str | bytes | "os.PathLike[str]") -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | bytes | "os.PathLike[str]", map_location: torch.device | str | None = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "actor_optimizer" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        if "critic_optimizer" in ckpt:
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])