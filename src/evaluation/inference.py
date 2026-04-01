"""Inference utilities."""
import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path

from ..utils.io import load_wav, save_wav
from ..data.preprocessing import frame_audio, unframe_audio, normalize_audio, denormalize_audio
from ..data.naming import extract_track_id


@torch.no_grad()
def infer_waveform(
    model: nn.Module,
    waveform: torch.Tensor,
    frame_size: int = 1024,
    hop_size: int = 512,
    device: str = "cpu",
    return_frames: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run inference on a waveform using overlap-add.
    
    Args:
        model: Inference model.
        waveform: Input waveform [1, T] or [T].
        frame_size: Frame size.
        hop_size: Hop size.
        device: Device to run on.
        return_frames: If True, also return frame-level predictions.
        
    Returns:
        Tuple of (output_waveform, metadata).
        output_waveform has same shape as input.
        metadata is a dict with inference info.
    """
    model.eval()
    
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # [1, T]
    
    original_length = waveform.shape[-1]
    
    # Frame the input
    frames = frame_audio(waveform, frame_size, hop_size, pad=False)  # [num_frames, 1, frame_size]
    
    # Infer on each frame
    output_frames = []
    with torch.no_grad():
        for i in range(0, len(frames), 32):  # Process in batches to save memory
            batch_frames = frames[i : i + 32].to(device)
            batch_output = model(batch_frames)
            output_frames.append(batch_output.cpu())
    
    output_frames = torch.cat(output_frames, dim=0)
    
    # Unframe using overlap-add
    output = unframe_audio(output_frames, hop_size, original_length)
    
    # Reshape to [1, T]
    output = output.unsqueeze(0)
    
    metadata = {
        "original_length": original_length,
        "num_frames": len(frames),
        "frame_size": frame_size,
        "hop_size": hop_size
    }
    
    return output, metadata


@torch.no_grad()
def infer_file(
    model: nn.Module,
    input_path: str,
    output_path: str,
    frame_size: int = 1024,
    hop_size: int = 512,
    device: str = "cpu",
    sr: int = 48000
) -> None:
    """Run inference on a WAV file and save output.
    
    Args:
        model: Inference model.
        input_path: Path to input WAV file.
        output_path: Path to save output WAV file.
        frame_size: Frame size.
        hop_size: Hop size.
        device: Device to run on.
        sr: Sample rate.
    """
    # Load input
    print(f"Loading: {input_path}")
    waveform, sr_loaded = load_wav(input_path, sr=sr)
    
    # Infer
    print(f"Inferencing...")
    output, metadata = infer_waveform(
        model, waveform,
        frame_size=frame_size,
        hop_size=hop_size,
        device=device
    )
    
    # Save output
    print(f"Saving: {output_path}")
    save_wav(output.squeeze(0), sr, output_path)
    
    print(f"Done! Metadata: {metadata}")
