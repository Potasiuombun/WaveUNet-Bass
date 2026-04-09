"""Fixed-band deterministic analysis/synthesis helpers for lightweight controllers."""

from dataclasses import dataclass
from typing import List, Sequence

import torch


@dataclass
class BandSpec:
    """Frequency band specification in Hz."""

    low_hz: float
    high_hz: float


class FixedBandSplitter:
    """Deterministic FFT-mask band splitter for waveform batches.

    Uses fixed frequency masks on rFFT bins and overlap-free additive
    reconstruction by summing masked spectra and running inverse rFFT.
    """

    def __init__(self, sample_rate: int, bands_hz: Sequence[Sequence[float]]) -> None:
        self.sample_rate = int(sample_rate)
        self.bands = [BandSpec(float(b[0]), float(b[1])) for b in bands_hz]
        if len(self.bands) < 2:
            raise ValueError("bands_hz must define at least 2 bands")

    def _build_masks(self, n_fft: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate).to(device=device, dtype=dtype)
        masks: List[torch.Tensor] = []
        for idx, band in enumerate(self.bands):
            if idx == len(self.bands) - 1:
                m = (freqs >= band.low_hz) & (freqs <= band.high_hz)
            else:
                m = (freqs >= band.low_hz) & (freqs < band.high_hz)
            masks.append(m.to(dtype=dtype))
        return torch.stack(masks, dim=0)  # [BANDS, F]

    def analyze(self, x: torch.Tensor) -> torch.Tensor:
        """Split waveform into fixed bands.

        Args:
            x: Waveform tensor [N, 1, T].

        Returns:
            Band waveforms [N, BANDS, T].
        """
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected input shape [N,1,T], got {tuple(x.shape)}")

        n = x.shape[-1]
        spec = torch.fft.rfft(x.squeeze(1), dim=-1)  # [N, F]
        masks = self._build_masks(n_fft=n, device=spec.device, dtype=x.dtype)
        band_specs = spec.unsqueeze(1) * masks.unsqueeze(0)  # [N, BANDS, F]
        bands = torch.fft.irfft(band_specs, n=n, dim=-1)
        return bands

    def synthesize(self, bands: torch.Tensor) -> torch.Tensor:
        """Recombine band waveforms into a single waveform [N,1,T]."""
        if bands.ndim != 3:
            raise ValueError(f"Expected bands shape [N,BANDS,T], got {tuple(bands.shape)}")
        return bands.sum(dim=1, keepdim=True)

    def apply_band_gains(self, bands: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
        """Apply per-band linear gains.

        Args:
            bands: [N, BANDS, T]
            gains: [N, BANDS] linear gain values
        """
        if gains.ndim != 2:
            raise ValueError(f"Expected gains [N,BANDS], got {tuple(gains.shape)}")
        return bands * gains.unsqueeze(-1)


def peak_safe(x: torch.Tensor, peak_limit: float = 0.99) -> torch.Tensor:
    """Apply peak-safe scaling to keep |x| <= peak_limit."""
    peak = torch.amax(torch.abs(x), dim=-1, keepdim=True)
    scale = torch.clamp(float(peak_limit) / (peak + 1e-12), max=1.0)
    return x * scale
