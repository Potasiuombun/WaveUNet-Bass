"""Minimal DDSP-style controller network producing interpretable controls."""

from typing import Dict

import torch
import torch.nn as nn


class DDSPController(nn.Module):
    """Predict per-band gains, spectral tilt, and optional smooth envelope."""

    def __init__(
        self,
        num_bands: int,
        hidden_channels: int = 16,
        max_gain_db: float = 6.0,
        min_gain_db: float = -6.0,
        max_tilt_db: float = 3.0,
        envelope_enabled: bool = True,
        min_envelope: float = 0.7,
        max_envelope: float = 1.3,
    ) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.max_gain_db = float(max_gain_db)
        self.min_gain_db = float(min_gain_db)
        self.max_tilt_db = float(max_tilt_db)
        self.envelope_enabled = bool(envelope_enabled)
        self.min_envelope = float(min_envelope)
        self.max_envelope = float(max_envelope)

        self.backbone = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=9, padding=4),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=9, padding=4),
            nn.LeakyReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.band_head = nn.Linear(hidden_channels, self.num_bands)
        self.tilt_head = nn.Linear(hidden_channels, 1)

        if self.envelope_enabled:
            self.envelope_head = nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=17, padding=8),
                nn.LeakyReLU(),
                nn.Conv1d(hidden_channels, 1, kernel_size=1),
            )
        else:
            self.envelope_head = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict interpretable control tensors for DDSP-style processing."""
        feat = self.backbone(x)
        pooled = self.global_pool(feat).squeeze(-1)

        gain_unit = torch.tanh(self.band_head(pooled))
        gain_center = 0.5 * (self.max_gain_db + self.min_gain_db)
        gain_span = 0.5 * (self.max_gain_db - self.min_gain_db)
        band_gain_db = gain_center + gain_span * gain_unit

        tilt = torch.tanh(self.tilt_head(pooled)) * self.max_tilt_db

        controls: Dict[str, torch.Tensor] = {
            "band_gain_db": band_gain_db,
            "spectral_tilt_db": tilt,
        }

        if self.envelope_head is not None:
            env_unit = torch.sigmoid(self.envelope_head(feat))
            env = self.min_envelope + (self.max_envelope - self.min_envelope) * env_unit
            controls["gain_envelope"] = env

        return controls
