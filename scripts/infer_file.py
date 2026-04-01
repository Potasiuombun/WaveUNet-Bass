#!/usr/bin/env python3
"""Inference script for Wave-U-Net baseline."""
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.waveunet import ResidualWaveUNet
from src.evaluation.inference import infer_file
from src.training.checkpointing import Checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run inference on audio file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Input WAV file")
    parser.add_argument("--output", required=True, help="Output WAV file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--sr", type=int, default=48000)
    args = parser.parse_args()
    
    device = args.device
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = ResidualWaveUNet(depth=8, base_channels=16)
    
    state = torch.load(args.checkpoint, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    
    model = model.to(device)
    model.eval()
    print("Model loaded!")
    
    # Run inference
    infer_file(
        model=model,
        input_path=args.input,
        output_path=args.output,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        device=device,
        sr=args.sr
    )
    
    print("✅ Inference completed!")


if __name__ == "__main__":
    main()
