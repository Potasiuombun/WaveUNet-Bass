"""DDSP-style interpretable control application utilities."""

from typing import Dict

import torch

from .bandsplit import FixedBandSplitter, peak_safe


def _db_to_lin(db: torch.Tensor) -> torch.Tensor:
    return torch.pow(torch.tensor(10.0, device=db.device, dtype=db.dtype), db / 20.0)


def apply_ddsp_controls(
    x: torch.Tensor,
    splitter: FixedBandSplitter,
    controls: Dict[str, torch.Tensor],
    peak_limit: float,
) -> torch.Tensor:
    """Apply interpretable controls to waveform.

    Expected controls:
    - band_gain_db: [N, BANDS]
    - spectral_tilt_db: [N, 1] (optional)
    - gain_envelope: [N, 1, T] (optional)
    """
    bands = splitter.analyze(x)
    band_gain_db = controls["band_gain_db"]
    n_bands = band_gain_db.shape[1]

    # Optional spectral tilt applied around center band index.
    tilt_db = controls.get("spectral_tilt_db")
    if tilt_db is not None:
        idx = torch.linspace(-1.0, 1.0, steps=n_bands, device=x.device, dtype=x.dtype).unsqueeze(0)
        band_gain_db = band_gain_db + tilt_db * idx

    band_gain_lin = _db_to_lin(band_gain_db)
    y_bands = splitter.apply_band_gains(bands, band_gain_lin)
    y = splitter.synthesize(y_bands)

    env = controls.get("gain_envelope")
    if env is not None:
        y = y * env

    return peak_safe(y, peak_limit=peak_limit)
