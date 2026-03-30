from __future__ import annotations

import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """Convolutional encoder for stacked visual observations."""

    def __init__(
        self,
        input_channels: int,
        observation_height: int,
        observation_width: int,
    ) -> None:
        """Initialize the encoder.

        Args:
            input_channels: Number of stacked input channels.
            observation_height: Input frame height.
            observation_width: Input frame width.
        """
        super().__init__()
        self.conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample: torch.Tensor = torch.zeros(
                1,
                input_channels,
                observation_height,
                observation_width,
            )
            self.feature_dim: int = int(self.conv(sample).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an observation batch.

        Args:
            x: Batch of observations with shape [B, C, H, W].

        Returns:
            Encoded features.
        """
        return self.conv(x)


class PolicyNetwork(nn.Module):
    """Gaussian policy network for continuous control."""

    def __init__(
        self,
        input_channels: int,
        action_dim: int,
        observation_height: int,
        observation_width: int,
        hidden_dim: int,
    ) -> None:
        """Initialize the policy network.

        Args:
            input_channels: Number of stacked input channels.
            action_dim: Action vector dimension.
            observation_height: Input frame height.
            observation_width: Input frame width.
            hidden_dim: Hidden layer dimension after the encoder.
        """
        super().__init__()
        self.encoder: VisualEncoder = VisualEncoder(
            input_channels=input_channels,
            observation_height=observation_height,
            observation_width=observation_width,
        )
        self.backbone: nn.Sequential = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head: nn.Linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_head: nn.Linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Gaussian parameters.

        Args:
            x: Batch of observations with shape [B, C, H, W].

        Returns:
            Mean and standard deviation tensors.
        """
        features: torch.Tensor = self.encoder(x)
        hidden: torch.Tensor = self.backbone(features)
        mu: torch.Tensor = self.mu_head(hidden)
        log_std: torch.Tensor = self.log_std_head(hidden)
        log_std = torch.clamp(log_std, min=-20.0, max=2.0)
        std: torch.Tensor = torch.exp(log_std)
        return mu, std


class ValueNetwork(nn.Module):
    """Value function network."""

    def __init__(
        self,
        input_channels: int,
        observation_height: int,
        observation_width: int,
        hidden_dim: int,
    ) -> None:
        """Initialize the value network.

        Args:
            input_channels: Number of stacked input channels.
            observation_height: Input frame height.
            observation_width: Input frame width.
            hidden_dim: Hidden layer dimension after the encoder.
        """
        super().__init__()
        self.encoder: VisualEncoder = VisualEncoder(
            input_channels=input_channels,
            observation_height=observation_height,
            observation_width=observation_width,
        )
        self.head: nn.Sequential = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state values.

        Args:
            x: Batch of observations with shape [B, C, H, W].

        Returns:
            Value estimates with shape [B, 1].
        """
        features: torch.Tensor = self.encoder(x)
        return self.head(features)
