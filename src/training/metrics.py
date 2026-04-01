"""Evaluation metrics for audio."""
import torch
import numpy as np


def nmse(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Normalised Mean Squared Error.
    
    Args:
        output: Predicted waveform [batch, 1, T] or [1, T].
        target: Target waveform [batch, 1, T] or [1, T].
        
    Returns:
        NMSE scalar.
    """
    mse = torch.mean((output - target) ** 2)
    normalizer = torch.mean(target ** 2) + 1e-8
    return float((mse / normalizer).cpu())


def mae(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Absolute Error.
    
    Args:
        output: Predicted waveform.
        target: Target waveform.
        
    Returns:
        MAE scalar.
    """
    return float(torch.mean(torch.abs(output - target)).cpu())


def peak_reduction(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute peak reduction between output and target.
    
    Peak reduction = (1 - max(abs(output)) / max(abs(target))) * 100%
    
    Args:
        output: Predicted waveform.
        target: Target waveform.
        
    Returns:
        Peak reduction in percentage.
    """
    output_peak = torch.max(torch.abs(output))
    target_peak = torch.max(torch.abs(target))
    
    if target_peak == 0:
        return 0.0
    
    reduction = (1 - output_peak / target_peak) * 100
    return float(reduction.cpu())


def crest_factor(waveform: torch.Tensor) -> float:
    """Compute crest factor of waveform.
    
    Crest factor = peak / RMS
    
    Args:
        waveform: Input waveform.
        
    Returns:
        Crest factor scalar.
    """
    peak = torch.max(torch.abs(waveform))
    rms = torch.sqrt(torch.mean(waveform ** 2))
    
    if rms == 0:
        return 0.0
    
    cf = peak / (rms + 1e-10)
    return float(cf.cpu())


def loudness_proxy(waveform: torch.Tensor) -> float:
    """Compute a loudness proxy (RMS).
    
    Args:
        waveform: Input waveform.
        
    Returns:
        RMS loudness in linear scale.
    """
    rms = torch.sqrt(torch.mean(waveform ** 2))
    return float(rms.cpu())
