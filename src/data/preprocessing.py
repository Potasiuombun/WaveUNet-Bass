"""Audio preprocessing: framing, normalization, filtering."""
from typing import Tuple, Optional, Literal
import numpy as np
import torch


def frame_audio(
    waveform: torch.Tensor,
    frame_size: int,
    hop_size: int,
    pad: bool = False
) -> torch.Tensor:
    """Frame audio waveform.
    
    Args:
        waveform: Tensor of shape [1, T] or [T].
        frame_size: Number of samples per frame.
        hop_size: Number of samples to hop.
        pad: If True, pad last frame to frame_size.
        
    Returns:
        Tensor of shape [num_frames, 1, frame_size].
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # [1, T]
    
    waveform = waveform.squeeze(0)  # [T]
    T = waveform.shape[0]
    
    frames = []
    for start in range(0, T - frame_size + 1, hop_size):
        frame = waveform[start : start + frame_size]
        frames.append(frame)
    
    # Handle last frame
    if pad and T >= frame_size:
        last_start = ((T - frame_size) // hop_size) * hop_size + hop_size
        if last_start + frame_size <= T:
            pass  # Already handled in the loop
        elif last_start < T:
            # Pad last frame
            last_frame = waveform[last_start:]
            last_frame = torch.nn.functional.pad(
                last_frame, (0, frame_size - len(last_frame))
            )
            frames.append(last_frame)
    
    if not frames:
        raise ValueError(
            f"No complete frames in audio of length {T} "
            f"with frame_size={frame_size}, hop_size={hop_size}"
        )
    
    frames = torch.stack(frames)  # [num_frames, frame_size]
    return frames.unsqueeze(1)  # [num_frames, 1, frame_size]


def unframe_audio(
    frames: torch.Tensor,
    hop_size: int,
    original_length: Optional[int] = None
) -> torch.Tensor:
    """Reconstruct audio from frames using overlap-add.
    
    Args:
        frames: Tensor of shape [num_frames, 1, frame_size].
        hop_size: Hop size used during framing.
        original_length: If provided, truncate output to this length.
        
    Returns:
        Tensor of shape [T].
    """
    frames = frames.squeeze(1)  # [num_frames, frame_size]
    num_frames, frame_size = frames.shape
    
    output_length = (num_frames - 1) * hop_size + frame_size
    output = torch.zeros(output_length, dtype=frames.dtype, device=frames.device)
    
    for i, frame in enumerate(frames):
        start = i * hop_size
        output[start : start + frame_size] += frame
    
    # Window correction (simple, assumes Hann-like behavior)
    # For perfect reconstruction with 50% overlap, divide by overlap count
    for i in range(output_length):
        count = 0
        for j in range(num_frames):
            start = j * hop_size
            if start <= i < start + frame_size:
                count += 1
        if count > 0:
            output[i] /= count
    
    if original_length is not None:
        output = output[:original_length]
    
    return output


def normalize_audio(
    waveform: torch.Tensor,
    mode: Literal["none", "peak_per_track", "rms_per_track"] = "none"
) -> Tuple[torch.Tensor, dict]:
    """Normalize audio waveform.
    
    Args:
        waveform: Tensor of shape [T].
        mode: Normalization mode.
        
    Returns:
        Tuple of (normalized_waveform, metadata).
        metadata contains normalization parameters for denormalization.
    """
    metadata = {"mode": mode}
    
    if mode == "none":
        return waveform, metadata
    
    elif mode == "peak_per_track":
        peak = torch.max(torch.abs(waveform))
        if peak > 0:
            normalized = waveform / (peak + 1e-10)
            metadata["peak"] = float(peak)
        else:
            normalized = waveform
            metadata["peak"] = 1.0
        return normalized, metadata
    
    elif mode == "rms_per_track":
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            normalized = waveform / (rms + 1e-10)
            metadata["rms"] = float(rms)
        else:
            normalized = waveform
            metadata["rms"] = 1.0
        return normalized, metadata
    
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def denormalize_audio(
    waveform: torch.Tensor,
    metadata: dict
) -> torch.Tensor:
    """Denormalize audio using metadata from normalize_audio.
    
    Args:
        waveform: Normalized waveform.
        metadata: Metadata dict from normalize_audio.
        
    Returns:
        Denormalized waveform.
    """
    mode = metadata.get("mode", "none")
    
    if mode == "none":
        return waveform
    elif mode == "peak_per_track":
        return waveform * metadata.get("peak", 1.0)
    elif mode == "rms_per_track":
        return waveform * metadata.get("rms", 1.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def filter_frames_by_threshold(
    input_frames: torch.Tensor,
    threshold: float = 0.0
) -> torch.Tensor:
    """Filter frames by threshold on input amplitude.
    
    Args:
        input_frames: Tensor of shape [num_frames, 1, frame_size].
        threshold: Keep frame only if max(abs(input_frame)) >= threshold.
        
    Returns:
        Boolean mask of shape [num_frames].
    """
    max_vals = torch.max(torch.abs(input_frames), dim=2)[0].squeeze(1)
    mask = max_vals >= threshold
    return mask
