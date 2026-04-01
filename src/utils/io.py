"""File I/O utilities for audio and data."""
import os
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torchaudio


def load_wav(path: str, sr: int = 48000) -> Tuple[torch.Tensor, int]:
    """Load WAV file as mono waveform.
    
    Args:
        path: Path to WAV file.
        sr: Target sample rate. The file must already be at this rate.
        
    Returns:
        Tuple of (waveform, sample_rate) where waveform is shape [1, T].
        
    Raises:
        RuntimeError: If audio sample rate doesn't match `sr`.
    """
    waveform, sample_rate = torchaudio.load(path)
    
    if sample_rate != sr:
        raise RuntimeError(
            f"Expected {sr} Hz, got {sample_rate} Hz in {path}. "
            "Resampling is not supported; ensure all audio is pre-processed."
        )
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform, sample_rate


def load_npy(path: str) -> np.ndarray:
    """Load NPY array.
    
    Args:
        path: Path to NPY file.
        
    Returns:
        NumPy array. Expected shape [T] or [1, T].
    """
    arr = np.load(path)
    return arr


def save_wav(waveform: torch.Tensor, sr: int, path: str) -> None:
    """Save waveform to WAV file.
    
    Args:
        waveform: Tensor of shape [1, T] or [T].
        sr: Sample rate.
        path: Output path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    # Ensure shape [1, T]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    torchaudio.save(path, waveform.cpu(), sr)


def ensure_parent_dir(path: str) -> str:
    """Ensure parent directory exists.
    
    Args:
        path: File path.
        
    Returns:
        The same path.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path
