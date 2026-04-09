"""Fast deterministic DSP loudness baseline.

This module intentionally avoids iterative per-frame optimization. The processor
uses a deterministic gain strategy with optional one-pole gain smoothing,
peak-safe scaling, and optional static compression.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch


def _db_to_linear(db: torch.Tensor) -> torch.Tensor:
    """Convert dB to linear gain."""
    return torch.pow(torch.tensor(10.0, device=db.device, dtype=db.dtype), db / 20.0)


def _linear_to_db(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert linear amplitude to dB with epsilon protection."""
    return 20.0 * torch.log10(torch.clamp(x, min=eps))


@dataclass
class SmoothingConfig:
    """Configuration for one-pole gain smoothing."""

    enabled: bool = True
    alpha: float = 0.9


@dataclass
class CompressionConfig:
    """Configuration for simple static compression."""

    enabled: bool = False
    threshold_db: float = -6.0
    ratio: float = 2.0


@dataclass
class FastDSPConfig:
    """Configuration for fast deterministic DSP processing."""

    target_level_db: float = -14.0
    max_gain_db: float = 12.0
    min_gain_db: float = -12.0
    peak_limit: float = 0.99
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)


class FastDSPBaseline:
    """Deterministic DSP loudness enhancement baseline.

    The processing chain is:
    1) Optional static compression.
    2) Frame-level gain toward target RMS level in dB.
    3) Optional one-pole smoothing over per-frame gain values.
    4) Peak-safe output scaling to keep max abs sample <= peak_limit.

    Notes:
    - This class is deterministic and contains no optimization loops.
    - Smoothing is stateful across calls unless :meth:`reset` is called.
    """

    def __init__(self, config: Dict, device: str = "cpu") -> None:
        """Initialize processor from config dictionary.

        Args:
            config: DSP configuration dictionary.
            device: Processing device.
        """
        smooth_cfg = config.get("smoothing", {})
        comp_cfg = config.get("compression", {})

        self.cfg = FastDSPConfig(
            target_level_db=float(config.get("target_level_db", -14.0)),
            max_gain_db=float(config.get("max_gain_db", 12.0)),
            min_gain_db=float(config.get("min_gain_db", -12.0)),
            peak_limit=float(config.get("peak_limit", 0.99)),
            smoothing=SmoothingConfig(
                enabled=bool(smooth_cfg.get("enabled", True)),
                alpha=float(smooth_cfg.get("alpha", 0.9)),
            ),
            compression=CompressionConfig(
                enabled=bool(comp_cfg.get("enabled", False)),
                threshold_db=float(comp_cfg.get("threshold_db", -6.0)),
                ratio=float(comp_cfg.get("ratio", 2.0)),
            ),
        )

        self.device = device
        self._prev_gain_linear: Optional[float] = None

    def reset(self) -> None:
        """Reset smoothing state between independent runs."""
        self._prev_gain_linear = None

    def _normalize_shape(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        """Normalize waveform shape to [B, 1, T] and return original shape."""
        original_shape = tuple(x.shape)
        if x.ndim == 1:
            return x.unsqueeze(0).unsqueeze(0), original_shape
        if x.ndim == 2:
            return x.unsqueeze(1), original_shape
        if x.ndim == 3:
            return x, original_shape
        raise ValueError(f"Unsupported waveform shape: {x.shape}")

    def _restore_shape(self, x: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Restore output tensor shape to match input rank."""
        if len(original_shape) == 1:
            return x.squeeze(0).squeeze(0)
        if len(original_shape) == 2:
            return x.squeeze(1)
        return x

    def _apply_compression(self, x: torch.Tensor) -> torch.Tensor:
        """Apply simple static compression in the dB domain."""
        if not self.cfg.compression.enabled:
            return x

        ratio = max(1.0, self.cfg.compression.ratio)
        threshold_db = self.cfg.compression.threshold_db

        mag = torch.clamp(torch.abs(x), min=1e-8)
        x_db = _linear_to_db(mag)
        over_db = torch.relu(x_db - threshold_db)

        # Hard-knee static compression above threshold.
        reduction_db = over_db * (1.0 - 1.0 / ratio)
        compressed_db = x_db - reduction_db
        compressed_mag = _db_to_linear(compressed_db)
        return torch.sign(x) * compressed_mag

    def _smooth_gains(self, gains_linear: torch.Tensor) -> torch.Tensor:
        """Apply one-pole smoothing over batch-ordered gain values."""
        if not self.cfg.smoothing.enabled:
            return gains_linear

        alpha = float(min(max(self.cfg.smoothing.alpha, 0.0), 0.9999))
        gains_1d = gains_linear.view(-1)
        smoothed = torch.empty_like(gains_1d)

        prev = gains_1d[0].item() if self._prev_gain_linear is None else self._prev_gain_linear
        for idx, current in enumerate(gains_1d):
            prev = alpha * prev + (1.0 - alpha) * current.item()
            smoothed[idx] = prev

        self._prev_gain_linear = float(prev)
        return smoothed.view_as(gains_linear)

    def process_batch(self, waveforms: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Process waveforms with deterministic DSP operations.

        Args:
            waveforms: Input tensor with shape [B, 1, T], [B, T], or [T].

        Returns:
            Tuple of:
            - processed waveforms with the same rank as input
            - batch stats for logging
        """
        x_in = waveforms.to(self.device)
        x, original_shape = self._normalize_shape(x_in)

        x_comp = self._apply_compression(x)

        rms = torch.sqrt(torch.mean(torch.square(x_comp), dim=2, keepdim=True) + 1e-12)
        current_level_db = _linear_to_db(rms)

        target = torch.tensor(self.cfg.target_level_db, device=x.device, dtype=x.dtype)
        desired_gain_db = torch.clamp(
            target - current_level_db,
            min=self.cfg.min_gain_db,
            max=self.cfg.max_gain_db,
        )

        gain_linear = _db_to_linear(desired_gain_db)
        gain_linear = self._smooth_gains(gain_linear)
        x_gain = x_comp * gain_linear

        peak = torch.max(torch.abs(x_gain), dim=2, keepdim=True)[0]
        peak_scale = torch.clamp(self.cfg.peak_limit / (peak + 1e-12), max=1.0)
        x_out = x_gain * peak_scale

        out_stats = {
            "mean_input_level_db": float(current_level_db.mean().item()),
            "mean_gain_db": float(desired_gain_db.mean().item()),
            "mean_peak_scale": float(peak_scale.mean().item()),
            "max_output_peak": float(torch.max(torch.abs(x_out)).item()),
        }

        return self._restore_shape(x_out, original_shape), out_stats
