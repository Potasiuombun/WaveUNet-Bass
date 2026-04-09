"""Tiny residual model for DSP+neural hybrid enhancement."""

from typing import Optional

import torch
import torch.nn as nn


class TinyResidualModel(nn.Module):
    """Small 1D CNN that predicts a bounded residual waveform.

    The model is intentionally lightweight for stable, fast experiments where
    the deterministic DSP baseline performs the main enhancement and this model
    only predicts a small corrective residual.
    """

    def __init__(
        self,
        hidden_channels: int = 16,
        num_layers: int = 4,
        kernel_size: int = 7,
        activation: str = "leaky_relu",
        max_residual: float = 0.2,
    ) -> None:
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.max_residual = float(max_residual)

        if activation == "relu":
            act_factory = nn.ReLU
        elif activation == "gelu":
            act_factory = nn.GELU
        else:
            act_factory = nn.LeakyReLU

        padding = kernel_size // 2
        layers: list[nn.Module] = []
        in_ch = 1
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(in_ch, hidden_channels, kernel_size=kernel_size, padding=padding))
            layers.append(act_factory())
            in_ch = hidden_channels
        layers.append(nn.Conv1d(in_ch, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict bounded residual for input waveform.

        Args:
            x: Input waveform tensor with shape [B, 1, T].

        Returns:
            Residual tensor in range [-max_residual, +max_residual].
        """
        raw = self.net(x)
        return torch.tanh(raw) * self.max_residual


def apply_residual_with_peak_safety(
    dsp_output: torch.Tensor,
    residual: torch.Tensor,
    peak_limit: float = 0.99,
) -> torch.Tensor:
    """Combine DSP output and neural residual, then apply peak-safe scaling."""
    out = dsp_output + residual
    peak = torch.amax(torch.abs(out), dim=-1, keepdim=True)
    scale = torch.clamp(float(peak_limit) / (peak + 1e-12), max=1.0)
    return out * scale
