#!/usr/bin/env python3
"""Training script for Wave-U-Net loudness enhancement baseline."""
import argparse
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.data.dataset import create_dataloaders
from src.data.serialized import load_serialized_dataset, get_dataset_info
from src.models.factory import create_model
from src.losses.combined import create_combined_loss
from src.training.engine import Trainer


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


def _resolve_device(requested_device: Optional[str]) -> str:
    """Resolve runtime device with safe CUDA fallback.

    If no device is specified, defaults to CUDA when available, else CPU.
    """
    if requested_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"

    requested = requested_device.lower()
    if requested == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    if requested not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device '{requested_device}'. Use 'cuda' or 'cpu'.")
    return requested


def _split_frame_counts(dataloaders: Dict[str, DataLoader]) -> Dict[str, int]:
    """Return number of frames per split from dataloader datasets."""
    return {split: len(loader.dataset) for split, loader in dataloaders.items()}


def _resolve_existing_path(path_str: Optional[str], config_path: str) -> Optional[str]:
    """Resolve a path that may be relative to CWD or config directory."""
    if not path_str:
        return None

    candidate = Path(path_str)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    if candidate.exists():
        return str(candidate.resolve())

    config_dir = Path(config_path).resolve().parent
    rel_to_config = config_dir / path_str
    if rel_to_config.exists():
        return str(rel_to_config.resolve())

    return path_str


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON payload to disk with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_latest_train_pointer(payload: Dict[str, Any]) -> Path:
    """Write a pointer file to make the latest training artifacts obvious."""
    pointer_path = Path("logs") / "LATEST_TRAIN.json"
    _write_json(pointer_path, payload)
    return pointer_path


def _save_config_snapshot(config: Dict[str, Any], config_path: str, experiment_name: str) -> Path:
    """Save a resolved config snapshot for reproducibility."""
    snapshot_dir = Path("logs") / experiment_name
    snapshot_path = snapshot_dir / "config_snapshot.yaml"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    with open(snapshot_path, "w", encoding="utf-8") as handle:
        handle.write(f"# source_config: {config_path}\n")
        yaml.safe_dump(config, handle, sort_keys=False)

    return snapshot_path


def _ensure_dataset_sidecar_summary(
    dataset_file: Optional[str],
    split_counts: Dict[str, int],
    dataset_info: Optional[Dict[str, Any]],
) -> Optional[Path]:
    """Create dataset sidecar summary JSON if missing.

    The summary is written next to the serialized dataset as:
    <dataset_stem>.summary.json
    """
    if not dataset_file:
        return None

    dataset_path = Path(dataset_file)
    sidecar_path = dataset_path.with_suffix(".summary.json")

    if sidecar_path.exists():
        return sidecar_path

    payload: Dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "split_counts": split_counts,
    }
    if dataset_info is not None:
        payload.update(
            {
                "format": "legacy" if dataset_info.get("is_legacy") else "rich",
                "num_frames": dataset_info.get("num_frames"),
                "num_unique_tracks": dataset_info.get("num_unique_tracks"),
                "split_type": dataset_info.get("split_type"),
                "columns": dataset_info.get("columns", []),
            }
        )

    _write_json(sidecar_path, payload)
    print(f"Dataset summary written: {sidecar_path}")
    return sidecar_path


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
    device = _resolve_device(args.device)
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Set seed: {seed}")
    
    # Resolve data mode
    dataset_file = _resolve_existing_path(
        args.dataset_file or config.get("data", {}).get("dataset_file"),
        args.config,
    )
    data_root = _resolve_existing_path(
        args.data_root or config.get("data", {}).get("data_dir"),
        args.config,
    )

    data_mode = "serialized" if dataset_file else "raw"

    print("\nStartup:")
    print(f"  Config path: {args.config}")
    print(f"  Dataset mode: {data_mode}")
    print(f"  Dataset path: {dataset_file if dataset_file else '<none>'}")
    print(f"  Data root: {data_root if data_root else '<none>'}")
    print(f"  Device: {device}")
    print(f"  Batch size: {config['training']['batch_size']}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    dataset_info: Optional[Dict[str, Any]] = None
    if dataset_file:
        # Load from serialized dataset
        print(f"Loading from serialized dataset: {dataset_file}")
        
        # Get dataset info
        try:
            info = get_dataset_info(dataset_file)
            dataset_info = info
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
                dataset_file,
                split="train",
                batch_size=config["training"]["batch_size"],
                num_workers=config["training"].get("num_workers", 2),
                validate=True,
            ),
            "val": load_serialized_dataset(
                dataset_file,
                split="val",
                batch_size=config["training"]["batch_size"],
                num_workers=config["training"].get("num_workers", 2),
                validate=False,
            ),
            "test": load_serialized_dataset(
                dataset_file,
                split="test",
                batch_size=config["training"]["batch_size"],
                num_workers=config["training"].get("num_workers", 2),
                validate=False,
            ),
        }
    else:
        # Load from raw files
        if not data_root:
            raise ValueError(
                "Raw mode requires a data directory. Provide --data-root or set data.data_dir in config."
            )
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

    split_counts = _split_frame_counts(dataloaders)
    print("\nSplit frame counts:")
    print(f"  train: {split_counts['train']}")
    print(f"  val:   {split_counts['val']}")
    print(f"  test:  {split_counts['test']}")
    
    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Val batches: {len(dataloaders['val'])}")
    print(f"Test batches: {len(dataloaders['test'])}")

    experiment_name = str(config.get("experiment_name", "experiment"))
    model_name = str(config.get("model", {}).get("type", config.get("model", {}).get("name", "waveunet")))
    print(f"Model name: {model_name}")
    print(
        "Loss weights: "
        f"l1={config['loss']['l1_weight']}, "
        f"nmse={config['loss']['nmse_weight']}, "
        f"mrstft={config['loss']['mrstft_weight']}"
    )

    config_snapshot_path = _save_config_snapshot(config, args.config, experiment_name)
    dataset_sidecar_path = _ensure_dataset_sidecar_summary(dataset_file, split_counts, dataset_info)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config["model"], device=device)
    
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

    run_summary_path = Path("logs") / experiment_name / "run_summary.json"
    run_summary = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config_path": args.config,
        "config_snapshot": str(config_snapshot_path),
        "dataset_mode": data_mode,
        "dataset_path": dataset_file,
        "dataset_sidecar_summary": str(dataset_sidecar_path) if dataset_sidecar_path else None,
        "device": device,
        "batch_size": int(config["training"]["batch_size"]),
        "split_counts": split_counts,
        "model_name": model_name,
        "loss_weights": {
            "l1": float(config["loss"]["l1_weight"]),
            "nmse": float(config["loss"]["nmse_weight"]),
            "mrstft": float(config["loss"]["mrstft_weight"]),
        },
        "num_epochs": int(config["training"]["num_epochs"]),
        "checkpoint_dir": config["training"].get("checkpoint_dir", "checkpoints"),
        "csv_log": config["logging"].get("csv_log"),
        "json_log": config["logging"].get("json_log"),
    }
    _write_json(run_summary_path, run_summary)
    print(f"Run summary written: {run_summary_path}")

    latest_pointer_path = _write_latest_train_pointer(
        {
            "created_at": run_summary["created_at"],
            "experiment_name": experiment_name,
            "run_summary": str(run_summary_path),
            "config_snapshot": str(config_snapshot_path),
            "csv_log": run_summary.get("csv_log"),
            "json_log": run_summary.get("json_log"),
        }
    )
    print(f"Latest train pointer: {latest_pointer_path}")
    
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
