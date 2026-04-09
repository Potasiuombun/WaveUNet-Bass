#!/usr/bin/env python3
"""Run inference on audio file using Stage 1 model."""

import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse
from typing import Tuple
import yaml
import torch.nn as nn

# Import model factory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models.factory import create_model


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda") -> nn.Module:
    """Load model from checkpoint using config."""
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create model from config
    model = create_model(config["model"], device=device)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    return model


def load_and_preprocess_audio(audio_path: str, sr: int = 44100) -> Tuple[np.ndarray, np.ndarray]:
    """Load audio and apply CLI-like normalization.
    
    Returns:
        (normalized_waveform, original_waveform)
    """
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    
    # Store original
    y_orig = y.copy()
    
    # CLI normalization: peak normalize whole track
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / (peak + 1e-10)
    
    return y, y_orig


def frame_audio(x: np.ndarray, frame_size: int = 1024, hop_size: int = 512) -> Tuple[np.ndarray, int]:
    """Frame audio and return frames + number of samples before last frame."""
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop_size):
        frames.append(x[start : start + frame_size])
    
    # Return frames, and original length for later reconstruction
    return np.array(frames, dtype=np.float32), len(x)


def reconstruct_audio_from_frames(
    frames: np.ndarray,
    hop_size: int = 512,
    frame_size: int = 1024,
    original_length: int = None
) -> np.ndarray:
    """Reconstruct audio from overlapping frames using overlap-add."""
    
    if len(frames) == 0:
        return np.array([], dtype=np.float32)
    
    # Estimate output length
    n_frames = frames.shape[0]
    estimated_length = (n_frames - 1) * hop_size + frame_size
    
    if original_length:
        estimated_length = max(estimated_length, original_length)
    
    output = np.zeros(estimated_length, dtype=np.float32)
    window = np.hanning(frame_size)
    
    for i, frame in enumerate(frames):
        start = i * hop_size
        output[start : start + frame_size] += frame * window
    
    # Normalize by window overlap (avoid division by zero)
    window_sum = np.zeros(estimated_length, dtype=np.float32)
    for i in range(len(frames)):
        start = i * hop_size
        window_sum[start : start + frame_size] += window
    
    window_sum[window_sum == 0] = 1  # Avoid division by zero
    output = output / window_sum
    
    # Trim to original length
    if original_length:
        output = output[:original_length]
    
    return output


def run_inference(
    audio_path: str,
    checkpoint_path: str,
    config_path: str,
    output_dir: str = "inference_outputs",
    device: str = "cuda",
):
    """Run inference on audio file."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("STAGE 1 INFERENCE")
    print(f"{'='*70}")
    
    # Load model
    print(f"\n📦 Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, config_path, device=device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {model_params:,}")
    
    # Load audio
    print(f"\n🎵 Loading audio from {audio_path}...")
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"   ❌ File not found!")
        return
    
    y_norm, y_orig = load_and_preprocess_audio(str(audio_path))
    print(f"   Duration: {len(y_norm) / 44100:.2f}s ({len(y_norm)} samples)")
    print(f"   Peak (original): {np.max(np.abs(y_orig)):.4f}")
    print(f"   Peak (normalized): {np.max(np.abs(y_norm)):.4f}")
    
    # Frame audio
    print(f"\n🔲 Framing audio (frame_size=1024, hop_size=512)...")
    frames, orig_len = frame_audio(y_norm, frame_size=1024, hop_size=512)
    print(f"   Total frames: {len(frames)}")
    print(f"   Original length: {orig_len}")
    
    # Run inference
    print(f"\n⚡ Running inference...")
    with torch.no_grad():
        output_frames = []
        
        for i in range(0, len(frames), 32):  # Process in batches of 32
            batch_frames = frames[i : i + 32]
            batch_tensor = torch.from_numpy(batch_frames).unsqueeze(1).to(device)  # [batch, 1, frame_size]
            
            batch_output = model(batch_tensor)
            batch_output = batch_output.squeeze(1).cpu().numpy()  # [batch, frame_size]
            output_frames.extend(batch_output)
        
        output_frames = np.array(output_frames, dtype=np.float32)
    
    print(f"   Output frames: {output_frames.shape}")
    
    # Reconstruct audio from frames using overlap-add
    print(f"\n🔧 Reconstructing audio from frames...")
    y_recon = reconstruct_audio_from_frames(output_frames, hop_size=512, original_length=orig_len)
    print(f"   Reconstructed length: {len(y_recon)} samples ({len(y_recon) / 44100:.2f}s)")
    print(f"   Reconstructed peak: {np.max(np.abs(y_recon)):.4f}")
    
    # Compute metrics
    print(f"\n📊 Metrics:")
    mae = np.mean(np.abs(y_recon - y_norm))
    mse = np.mean((y_recon - y_norm) ** 2)
    nmse = mse / (np.mean(y_norm ** 2) + 1e-8)
    print(f"   MAE: {mae:.6f}")
    print(f"   MSE: {mse:.6f}")
    print(f"   NMSE: {nmse:.6f}")
    
    # Save outputs
    print(f"\n💾 Saving outputs...")
    
    # Save normalized input
    input_path = output_dir / "input_normalized.wav"
    sf.write(str(input_path), y_norm, 44100)
    print(f"   ✓ {input_path}")
    
    # Save reconstruction
    recon_path = output_dir / "output_reconstructed.wav"
    sf.write(str(recon_path), y_recon, 44100)
    print(f"   ✓ {recon_path}")
    
    # Save error signal
    error = y_recon - y_norm
    error_path = output_dir / "error.wav"
    sf.write(str(error_path), error, 44100)
    print(f"   ✓ {error_path}")
    
    # Save original
    orig_path = output_dir / "original.wav"
    sf.write(str(orig_path), y_orig, 44100)
    print(f"   ✓ {orig_path}")
    
    print(f"\n{'='*70}")
    print(f"✨ Inference complete! Check {output_dir}/ for audio files")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 1 inference on audio")
    parser.add_argument(
        "--audio",
        default="/home/tudor/Documents/MasterThesis/fma_small_curated_20260408/fma_small/000/000890.mp3",
        help="Path to input audio file"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/stage1_fma_curated_cli_norm/best.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        default="configs/stage1_fma_curated_cli_norm.yaml",
        help="Path to model config"
    )
    parser.add_argument(
        "--output-dir",
        default="inference_outputs",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    run_inference(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )
