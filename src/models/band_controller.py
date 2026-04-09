"""Tiny band-gain controller model."""

import torch
import torch.nn as nn


class BandController(nn.Module):
    """Predict bounded per-band gain controls from waveform input."""

    def __init__(
        self,
        num_bands: int,
        hidden_channels: int = 16,
        max_gain_db: float = 6.0,
        min_gain_db: float = -6.0,
    ) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.max_gain_db = float(max_gain_db)
        self.min_gain_db = float(min_gain_db)

        self.features = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=9, padding=4),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=9, padding=4),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_channels, self.num_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-band linear gains with shape [N, num_bands]."""
        feat = self.features(x).squeeze(-1)
        gain_unit = torch.tanh(self.head(feat))
        center = 0.5 * (self.max_gain_db + self.min_gain_db)
        span = 0.5 * (self.max_gain_db - self.min_gain_db)
        gain_db = center + span * gain_unit
        gain_lin = torch.pow(10.0, gain_db / 20.0)
        return gain_lin
