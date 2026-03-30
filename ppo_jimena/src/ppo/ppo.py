from __future__ import annotations

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


class ActorCriticNet(nn.Module):
    """
    FIX: Actor and critic now share a single CNN encoder.

    Previously, ActorNet and CriticNet each had their own independent
    ConvEncoder, doubling the number of parameters and forcing both heads
    to learn separate visual representations from scratch. This is
    inefficient and hurts stability on pixel-based environments.

    The shared encoder is updated by both the policy and value losses,
    which leads to richer and faster-converging representations.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(obs_shape, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def actor(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor_head(self.encoder(x))

    def critic(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.encoder(x)).squeeze(-1)


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

        # FIX: single shared network instead of two separate ones
        self.ac_net = ActorCriticNet(obs_shape, n_actions, hidden_dim).to(self.device)

        # Two separate optimizers are kept so actor_lr != critic_lr is respected,
        # but they share the encoder — encoder gradients accumulate from both losses.
        self.actor_optimizer = torch.optim.Adam(
            list(self.ac_net.encoder.parameters()) + list(self.ac_net.actor_head.parameters()),
            lr=float(actor_lr),
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.ac_net.encoder.parameters()) + list(self.ac_net.critic_head.parameters()),
            lr=float(critic_lr),
        )

    def act(self, obs: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        obs_b = obs.unsqueeze(0)
        logits, value = self.ac_net(obs_b)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze(0), value.squeeze(0)

    def act_deterministic(self, obs: torch.Tensor) -> int:
        obs_b = obs.unsqueeze(0)
        logits, _ = self.ac_net(obs_b)
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

                logits, values = self.ac_net(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)

                # Actor step — also updates encoder via shared params
                self.actor_optimizer.zero_grad()
                actor_loss_total = policy_loss - self.entropy_coef * entropy
                actor_loss_total.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic step — re-uses encoder features already computed above
                self.critic_optimizer.zero_grad()
                # Recompute values after actor step so gradients are fresh
                _, values_fresh = self.ac_net(mb_obs)
                value_loss_fresh = F.mse_loss(values_fresh, mb_returns)
                critic_loss_total = self.value_coef * value_loss_fresh
                critic_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
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
                "ac_net": self.ac_net.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | bytes | "os.PathLike[str]", map_location: torch.device | str | None = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device)

        # Support both old checkpoint format (separate actor/critic) and new (ac_net)
        if "ac_net" in ckpt:
            self.ac_net.load_state_dict(ckpt["ac_net"])
        else:
            # Legacy: load actor encoder + head and critic head separately
            actor_sd = ckpt.get("actor", {})
            critic_sd = ckpt.get("critic", {})
            merged: dict = {}
            for k, v in actor_sd.items():
                if k.startswith("encoder."):
                    merged[k.replace("encoder.", "encoder.")] = v
                elif k.startswith("head."):
                    merged[k.replace("head.", "actor_head.")] = v
            for k, v in critic_sd.items():
                if k.startswith("head."):
                    merged[k.replace("head.", "critic_head.")] = v
            self.ac_net.load_state_dict(merged, strict=False)

        if "actor_optimizer" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        if "critic_optimizer" in ckpt:
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])