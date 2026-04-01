#!/usr/bin/env python3
"""Training script for Wave-U-Net loudness enhancement baseline."""
import argparse
import sys
from pathlib import Path
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.data.dataset import create_dataloaders
from src.data.serialized import load_serialized_dataset, get_dataset_info
from src.models.waveunet import create_waveunet
from src.losses.combined import create_combined_loss
from src.training.engine import Trainer
from src.utils.logging import CSVLogger


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _normalize_config(config: dict) -> dict:
    """Normalize config to support both old (training/logging) and new (train/eval) formats.

    Maps new key names to the canonical internal names used throughout this script,
    so either config format can be loaded without changing the rest of the code.
    """
    config = dict(config)  # shallow copy

    # train: → training: (new format uses 'train', old used 'training')
    if "train" in config and "training" not in config:
        config["training"] = dict(config["train"])
    config.setdefault("training", {})

    t = config["training"]
    # epochs → num_epochs
    if "epochs" in t and "num_epochs" not in t:
        t["num_epochs"] = t["epochs"]
    # grad_clip → gradient_clip
    if "grad_clip" in t and "gradient_clip" not in t:
        t["gradient_clip"] = t["grad_clip"]
    t.setdefault("num_workers", 2)
    t.setdefault("checkpoint_dir", "checkpoints")
    t.setdefault("gradient_clip", 1.0)

    # model normalization
    if "model" in config:
        m = config["model"]
        # name → type
        if "name" in m and "type" not in m:
            m["type"] = m["name"]
        # output_head: identity → output_activation: null
        if "output_head" in m and "output_activation" not in m:
            head = m["output_head"]
            m["output_activation"] = None if head == "identity" else head
        m.setdefault("activation", "leaky_relu")
        m.setdefault("output_activation", None)

    # logging fallback — derive paths from experiment_name if not present
    if "logging" not in config:
        exp = config.get("experiment_name", "experiment")
        config["logging"] = {
            "csv_log":  f"logs/{exp}_metrics.csv",
            "json_log": f"logs/{exp}_metrics.jsonl",
        }

    return config


def create_optimizer(model, config: dict):
    """Create optimizer based on config."""
    opt_config = config["training"]
    opt_name = opt_config.get("optimizer", "adamw").lower()
    lr = opt_config.get("lr", 1e-4)
    weight_decay = opt_config.get("weight_decay", 1e-6)
    
    if opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def create_scheduler(optimizer, config: dict):
    """Create learning rate scheduler."""
    sched_name = config["training"].get("scheduler")
    if sched_name is None:
        return None
    
    sched_name = sched_name.lower()
    if sched_name == "step":
        step_size = config["training"].get("scheduler_step_size", 10)
        gamma = config["training"].get("scheduler_gamma", 0.5)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name == "cosine":
        num_epochs = config["training"].get("num_epochs", 100)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Train Wave-U-Net baseline")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Config file path")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    
    # Data source: either from raw files or serialized dataset
    data_group = parser.add_mutually_exclusive_group(required=False)
    data_group.add_argument(
        "--data-root",
        default=None,
        help="Root directory with raw audio files (overrides config)"
    )
    data_group.add_argument(
        "--dataset-file",
        default=None,
        help="Path to serialized dataset (.pkl or .parquet)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config = _normalize_config(config)
    print(f"Loaded config from {args.config}")
    
    # Device
    device = args.device or config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Set seed: {seed}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    if args.dataset_file:
        # Load from serialized dataset
        print(f"Loading from serialized dataset: {args.dataset_file}")
        
        # Get dataset info
        try:
            info = get_dataset_info(args.dataset_file)
            print(f"\nDataset Info:")
            print(f"  Type: {'LEGACY (input/target only)' if info['is_legacy'] else 'RICH (with metadata)'}")
            print(f"  Total frames: {info['num_frames']}")
            print(f"  Unique tracks: {info['num_unique_tracks']}")
            print(f"  Has split column: {info['has_split_column']}")
            print(f"  Available split type: {info['split_type']}")
            
            if info['is_legacy']:
                print(f"  ⚠️  NOTE: Legacy format detected (no track metadata).")
                print(f"     Grouped split not available. Row-level splitting will be used.")
                print(f"     For leak-free train/val/test splits, regenerate from raw files")
                print(f"     using: python scripts/prepare_dataset.py --data-root <dir> --output dataset.pkl")
        except Exception as e:
            print(f"  Could not read dataset info: {e}")
        
        # Load dataloaders
        dataloaders = {
            "train": load_serialized_dataset(
                args.dataset_file,
                split="train",
                batch_size=config["training"]["batch_size"],
                num_workers=config["training"].get("num_workers", 2),
                validate=True,
            ),
            "val": load_serialized_dataset(
                args.dataset_file,
                split="val",
                batch_size=config["training"]["batch_size"],
                num_workers=config["training"].get("num_workers", 2),
                validate=False,
            ),
            "test": load_serialized_dataset(
                args.dataset_file,
                split="test",
                batch_size=config["training"]["batch_size"],
                num_workers=config["training"].get("num_workers", 2),
                validate=False,
            ),
        }
    else:
        # Load from raw files
        data_root = args.data_root or config["data"]["data_dir"]
        print(f"Loading from raw files: {data_root}")
        dataloaders = create_dataloaders(
            data_dir=data_root,
            input_pattern=config["data"]["input_pattern"],
            target_pattern=config["data"]["target_pattern"],
            frame_size=config["data"]["frame_size"],
            hop_size=config["data"]["hop_size"],
            threshold=config["data"]["threshold"],
            normalization=config["data"]["normalization"],
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"].get("num_workers", 2),
            seed=seed,
            sr=config["data"]["sr"]
        )
    
    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Val batches: {len(dataloaders['val'])}")
    print(f"Test batches: {len(dataloaders['test'])}")
    
    # Create model
    print("\nCreating model...")
    model = create_waveunet(
        depth=config["model"]["depth"],
        base_channels=config["model"]["base_channels"],
        kernel_size=config["model"]["kernel_size"],
        activation=config["model"]["activation"],
        output_activation=config["model"]["output_activation"],
        max_scale=config["model"].get("max_scale"),
        device=device
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Create loss
    print("\nCreating loss...")
    criterion = create_combined_loss(
        l1_weight=config["loss"]["l1_weight"],
        nmse_weight=config["loss"]["nmse_weight"],
        mrstft_weight=config["loss"]["mrstft_weight"]
    )
    
    # Create optimizer
    print("\nCreating optimizer...")
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        gradient_clip=config["training"].get("gradient_clip", 1.0),
        amp=(device == "cuda")
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        num_epochs=config["training"]["num_epochs"],
        checkpoint_dir=config["training"]["checkpoint_dir"],
        patience=config["training"]["early_stopping_patience"],
        csv_log_path=config["logging"].get("csv_log"),
        json_log_path=config["logging"].get("json_log")
    )
    
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
