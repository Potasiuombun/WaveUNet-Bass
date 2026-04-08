"""Perceptual loss wrappers.

This module keeps detectability support optional so baseline training remains
fully functional when libdetectability is not installed.
"""

from typing import Optional

import torch
import torch.nn as nn


class DetectabilityLossWrapper(nn.Module):
    """Optional wrapper around libdetectability loss.

    When disabled, returns a zero tensor on the correct device.
    When enabled, it attempts to import libdetectability lazily and computes
    a scalar perceptual loss.
    """

    def __init__(
        self,
        enabled: bool,
        sample_rate: int = 48000,
        frame_size: int = 1024,
    ):
        """Initialize detectability loss wrapper.

        Args:
            enabled: Whether detectability loss should be active.
            sample_rate: Sampling rate for perceptual model.
            frame_size: Frame size expected by detectability model.

        Raises:
            RuntimeError: If enabled is True and libdetectability is unavailable.
        """
        super().__init__()
        self.enabled = bool(enabled)
        self.sample_rate = int(sample_rate)
        self.frame_size = int(frame_size)
        self._impl: Optional[nn.Module] = None

        if self.enabled:
            self._impl = self._build_impl()

    def _build_impl(self) -> nn.Module:
        """Build detectability implementation from optional dependency."""
        try:
            from libdetectability import DetectabilityLoss  # type: ignore
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "Detectability is enabled but libdetectability is not installed. "
                "Install it or set finetune.detectability_enabled=false."
            ) from exc

        # libdetectability uses sampling_rate (not sample_rate) in current releases.
        return DetectabilityLoss(sampling_rate=self.sample_rate, frame_size=self.frame_size)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute detectability loss or return zero when disabled."""
        if not self.enabled:
            return torch.zeros((), device=output.device, dtype=output.dtype)
        assert self._impl is not None
        # libdetectability expects [batch, time] tensors.
        if output.ndim == 3 and output.shape[1] == 1:
            output = output.squeeze(1)
        if target.ndim == 3 and target.shape[1] == 1:
            target = target.squeeze(1)
        self._impl = self._impl.to(output.device)
        return self._impl(output, target)
