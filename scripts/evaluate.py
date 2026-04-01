#!/usr/bin/env python3
"""Evaluation script for Wave-U-Net baseline."""
import argparse
import sys
from pathlib import Path
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.waveunet import ResidualWaveUNet
from src.data.dataset import create_dataloaders
from src.evaluation.evaluate import evaluate_dataset
from src.utils.seed import set_seed


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wave-U-Net baseline")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Config file")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = args.device or config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}")
    
    # Set seed
    set_seed(config.get("seed", 42))
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = ResidualWaveUNet(
        depth=config["model"]["depth"],
        base_channels=config["model"]["base_channels"]
    )
    
    state = torch.load(args.checkpoint, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    
    model = model.to(device)
    print("Model loaded!")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        data_dir=config["data"]["data_dir"],
        input_pattern=config["data"]["input_pattern"],
        target_pattern=config["data"]["target_pattern"],
        frame_size=config["data"]["frame_size"],
        hop_size=config["data"]["hop_size"],
        threshold=config["data"]["threshold"],
        normalization=config["data"]["normalization"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 2),
        seed=config.get("seed", 42),
        sr=config["data"]["sr"]
    )
    
    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    metrics = evaluate_dataset(model, dataloaders[args.split], device=device)
    
    # Print results
    print("\n" + "="*50)
    print(f"Evaluation Results ({args.split})")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:30s}: {value:.6f}")
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
