"""Visualization utilities."""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_waveform(
    waveform: torch.Tensor,
    sr: int = 48000,
    title: str = "",
    ax: plt.Axes = None
) -> plt.Axes:
    """Plot a single waveform.
    
    Args:
        waveform: Waveform tensor [1, T] or [T].
        sr: Sample rate.
        title: Plot title.
        ax: Matplotlib axes (will create new if None).
        
    Returns:
        Matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    waveform = waveform.squeeze(0).cpu().numpy()
    time = np.arange(len(waveform)) / sr
    
    ax.plot(time, waveform, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_waveforms_comparison(
    input_wf: torch.Tensor,
    target_wf: torch.Tensor,
    output_wf: torch.Tensor,
    sr: int = 48000,
    title: str = "",
    save_path: str = None
) -> None:
    """Plot input/target/output waveforms side by side.
    
    Args:
        input_wf: Input waveform.
        target_wf: Target waveform.
        output_wf: Output waveform.
        sr: Sample rate.
        title: Figure title.
        save_path: Optional path to save figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    plot_waveform(input_wf, sr, "Input", axes[0])
    plot_waveform(target_wf, sr, "Target", axes[1])
    plot_waveform(output_wf, sr, "Output", axes[2])
    
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    return fig


def plot_spectrogram(
    waveform: torch.Tensor,
    sr: int = 48000,
    n_fft: int = 2048,
    title: str = "",
    ax: plt.Axes = None,
    cmap: str = "viridis"
) -> plt.Axes:
    """Plot magnitude spectrogram.
    
    Args:
        waveform: Waveform tensor [1, T] or [T].
        sr: Sample rate.
        n_fft: FFT size.
        title: Plot title.
        ax: Matplotlib axes (will create new if None).
        cmap: Colormap.
        
    Returns:
        Matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    waveform = waveform.squeeze(0).cpu().numpy()
    
    stft = np.abs(np.fft.rfft(waveform, n=n_fft, axis=-1) if waveform.ndim == 1 
                   else np.fft.rfft(waveform, n=n_fft, axis=-1))
    
    # Better approach using torch.stft
    wf_torch = torch.from_numpy(waveform).float()
    S = torch.stft(wf_torch, n_fft=n_fft, return_complex=True)
    S_mag = torch.abs(S).cpu().numpy()
    
    im = ax.imshow(10 * np.log10(S_mag + 1e-10), aspect='auto', origin='lower', cmap=cmap)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    
    return ax
