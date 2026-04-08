#!/usr/bin/env python3
"""Overfit a model on a single fixed batch for sanity checking.

This script repeatedly optimizes the exact same mini-batch from a serialized
train split and records loss evolution.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.serialized import load_serialized_dataset
from src.losses.combined import create_combined_loss
from src.models.waveunet import create_waveunet
from src.utils.seed import set_seed


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config from disk."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize old/new config keys used by training scripts."""
    cfg = dict(config)

    if "train" in cfg and "training" not in cfg:
        cfg["training"] = dict(cfg["train"])
    cfg.setdefault("training", {})

    training = cfg["training"]
    if "epochs" in training and "num_epochs" not in training:
        training["num_epochs"] = training["epochs"]
    if "grad_clip" in training and "gradient_clip" not in training:
        training["gradient_clip"] = training["grad_clip"]
    training.setdefault("batch_size", 16)
    training.setdefault("lr", 1e-4)
    training.setdefault("weight_decay", 1e-4)
    training.setdefault("optimizer", "adamw")

    cfg.setdefault("data", {})

    if "model" in cfg:
        model = cfg["model"]
        if "name" in model and "type" not in model:
            model["type"] = model["name"]
        if "output_head" in model and "output_activation" not in model:
            model["output_activation"] = None if model["output_head"] == "identity" else model["output_head"]
        model.setdefault("activation", "leaky_relu")
        model.setdefault("output_activation", None)

    return cfg


def resolve_device(requested_device: Optional[str]) -> str:
    """Resolve compute device with CUDA fallback behavior."""
    if requested_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"

    device = requested_device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device '{requested_device}'. Use 'cuda' or 'cpu'.")
    return device


def resolve_existing_path(path_str: Optional[str], config_path: str) -> Optional[str]:
    """Resolve path from absolute, CWD-relative, or config-relative inputs."""
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


def create_optimizer(model: torch.nn.Module, training_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer for overfit run."""
    opt_name = str(training_cfg.get("optimizer", "adamw")).lower()
    lr = float(training_cfg.get("lr", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer '{opt_name}'")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    use_scheduler: bool,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Create an optional scheduler for the overfit run.

    By default, scheduler is disabled for cleaner memorization dynamics.
    """
    if not use_scheduler:
        return None

    step_size = max(1, total_steps // 3)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)


def create_loss_from_mode(config: Dict[str, Any], loss_mode: str):
    """Create loss function and objective key from selected loss mode."""
    mode = loss_mode.lower()
    if mode == "l1":
        return create_combined_loss(l1_weight=1.0, nmse_weight=0.0, mrstft_weight=0.0), "loss_l1"
    if mode == "nmse":
        return create_combined_loss(l1_weight=0.0, nmse_weight=1.0, mrstft_weight=0.0), "loss_nmse"
    if mode == "combined":
        return (
            create_combined_loss(
                l1_weight=float(config["loss"]["l1_weight"]),
                nmse_weight=float(config["loss"]["nmse_weight"]),
                mrstft_weight=float(config["loss"]["mrstft_weight"]),
            ),
            "loss_total",
        )
    raise ValueError(f"Unsupported loss mode '{loss_mode}'. Use l1, nmse, or combined.")


def maybe_save_plot(log_rows: List[Dict[str, float]], output_path: Path) -> None:
    """Save a simple step-vs-loss plot if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Skipping plot: matplotlib is not available.")
        return

    steps = [row["step"] for row in log_rows]
    losses = [row["loss_total"] for row in log_rows]

    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses)
    plt.xlabel("step")
    plt.ylabel("total_loss")
    plt.title("Overfit Batch Loss")
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def summarize_losses(log_rows: List[Dict[str, float]], objective_key: str) -> Tuple[float, float, float, float]:
    """Compute start, best, final, and relative reduction (%) for objective."""
    start_loss = log_rows[0][objective_key]
    best_loss = min(row[objective_key] for row in log_rows)
    final_loss = log_rows[-1][objective_key]
    if start_loss == 0.0:
        relative_reduction_pct = 0.0
    else:
        relative_reduction_pct = 100.0 * (start_loss - best_loss) / max(start_loss, 1e-12)
    return start_loss, best_loss, final_loss, relative_reduction_pct


def write_latest_pointer(output_dir: Path, payload: Dict[str, Any]) -> Path:
    """Write a latest-run pointer JSON next to overfit artifacts."""
    pointer_path = output_dir / "LATEST_OVERFIT.json"
    with open(pointer_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return pointer_path


def main() -> None:
    """Run fixed-batch overfit sanity training."""
    parser = argparse.ArgumentParser(description="Overfit on a single fixed training batch")
    parser.add_argument("--config", default="configs/baseline_small.yaml", help="Path to config file")
    parser.add_argument("--dataset-file", default=None, help="Serialized dataset path (.pkl/.parquet)")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for fixed batch")
    parser.add_argument("--steps", type=int, default=300, help="Training steps on the same batch")
    parser.add_argument(
        "--target-loss",
        type=float,
        default=None,
        help="Stop early when objective loss <= target value",
    )
    parser.add_argument("--print-every", type=int, default=25, help="Print metrics every N steps")
    parser.add_argument(
        "--loss-mode",
        default="l1",
        choices=["l1", "nmse", "combined"],
        help="Objective used for overfit sanity check (default: l1)",
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--disable-dropout",
        action="store_true",
        help="Force dropout-like regularization off for overfit sanity (best effort).",
    )
    parser.add_argument(
        "--no-scheduler",
        dest="no_scheduler",
        action="store_true",
        default=True,
        help="Disable scheduler (default behavior for stable memorization).",
    )
    parser.add_argument(
        "--use-scheduler",
        dest="no_scheduler",
        action="store_false",
        help="Enable a simple StepLR schedule.",
    )
    parser.add_argument("--output-dir", default="logs/overfit_batch", help="Directory for logs and plot")
    args = parser.parse_args()

    config = normalize_config(load_config(args.config))
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = resolve_device(args.device)
    dataset_path = resolve_existing_path(
        args.dataset_file or config.get("data", {}).get("dataset_file"),
        args.config,
    )
    if not dataset_path:
        raise ValueError("Missing dataset file. Provide --dataset-file or set data.dataset_file in config.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.disable_dropout:
        config.setdefault("model", {})
        config["model"]["dropout"] = 0.0

    print("Startup:")
    print(f"  Config path: {args.config}")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Steps: {args.steps}")
    print(f"  Target loss: {args.target_loss if args.target_loss is not None else '<disabled>'}")
    print(f"  Loss mode: {args.loss_mode}")
    print(f"  Scheduler: {'off' if args.no_scheduler else 'on'}")
    if args.disable_dropout:
        print("  Dropout override: requested (set to 0.0 in config context)")

    train_loader = load_serialized_dataset(
        dataset_path,
        split="train",
        batch_size=args.batch_size,
        num_workers=0,
        validate=False,
    )

    batch = next(iter(train_loader))
    fixed_inputs = batch["input"].to(device)
    fixed_targets = batch["target"].to(device)

    model = create_waveunet(
        depth=int(config["model"]["depth"]),
        base_channels=int(config["model"]["base_channels"]),
        kernel_size=int(config["model"]["kernel_size"]),
        activation=str(config["model"].get("activation", "leaky_relu")),
        output_activation=config["model"].get("output_activation"),
        max_scale=config["model"].get("max_scale"),
        device=device,
    )

    criterion, objective_key = create_loss_from_mode(config, args.loss_mode)

    if args.disable_dropout:
        # Baseline ResidualWaveUNet currently has no explicit dropout layers.
        # Keep this pass for compatibility with future variants.
        dropped = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
                dropped += 1
        if dropped == 0:
            print("  Note: model has no nn.Dropout modules; disable-dropout is a no-op for this baseline.")

    optimizer = create_optimizer(model, config["training"])
    if args.lr is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
    scheduler = create_scheduler(optimizer, use_scheduler=(not args.no_scheduler), total_steps=args.steps)

    log_rows: List[Dict[str, float]] = []

    print("\nOverfitting on one fixed batch...")
    model.train()
    target_reached_step: Optional[int] = None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        outputs = model(fixed_inputs)
        loss_dict = criterion(outputs, fixed_targets)
        loss = loss_dict["total"]

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        row: Dict[str, float] = {
            "step": float(step),
            "loss_total": float(loss.item()),
            "loss_l1": float(loss_dict.get("l1", torch.tensor(0.0)).item()),
            "loss_nmse": float(loss_dict.get("nmse", torch.tensor(0.0)).item()),
            "loss_mrstft": float(loss_dict.get("mrstft", torch.tensor(0.0)).item()),
            "lr": float(current_lr),
        }
        row["loss_objective"] = float(row[objective_key])
        row["min_loss_so_far"] = float(
            row["loss_objective"]
            if step == 1
            else min(log_rows[-1]["min_loss_so_far"], row["loss_objective"])
        )
        log_rows.append(row)

        if step == 1 or step % args.print_every == 0 or step == args.steps:
            print(
                f"step={step:4d} "
                f"obj={row['loss_objective']:.6f} "
                f"best={row['min_loss_so_far']:.6f} "
                f"total={row['loss_total']:.6f} "
                f"nmse={row['loss_nmse']:.6f} "
                f"l1={row['loss_l1']:.6f} "
                f"mrstft={row['loss_mrstft']:.6f}"
            )

        if args.target_loss is not None and row["loss_objective"] <= args.target_loss:
            target_reached_step = step
            print(
                f"Target reached at step {step}: objective={row['loss_objective']:.6f} <= {args.target_loss:.6f}"
            )
            break

    start_loss, best_loss, final_loss, relative_reduction_pct = summarize_losses(log_rows, "loss_objective")

    csv_path = output_dir / "overfit_batch_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "lr",
                "loss_total",
                "loss_l1",
                "loss_nmse",
                "loss_mrstft",
                "loss_objective",
                "min_loss_so_far",
            ],
        )
        writer.writeheader()
        writer.writerows(log_rows)

    json_path = output_dir / "overfit_batch_log.jsonl"
    with open(json_path, "w", encoding="utf-8") as handle:
        for row in log_rows:
            handle.write(json.dumps(row) + "\n")

    plot_path = output_dir / "overfit_batch_plot.png"
    maybe_save_plot(log_rows, plot_path)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "objective": args.loss_mode,
        "steps": args.steps,
        "steps_executed": len(log_rows),
        "batch_size": args.batch_size,
        "device": device,
        "scheduler": "off" if args.no_scheduler else "on",
        "target_loss": args.target_loss,
        "target_reached": target_reached_step is not None,
        "target_reached_step": target_reached_step,
        "start_loss": start_loss,
        "best_loss": best_loss,
        "final_loss": final_loss,
        "relative_reduction_pct": relative_reduction_pct,
        "dataset_path": dataset_path,
    }
    summary_path = output_dir / "overfit_batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    latest_pointer_path = write_latest_pointer(
        output_dir,
        {
            "timestamp_utc": summary["timestamp_utc"],
            "summary_json": str(summary_path),
            "csv_log": str(csv_path),
            "json_log": str(json_path),
            "plot": str(plot_path),
        },
    )

    print("\nDone.")
    print("Overfit summary:")
    print(f"  Start loss:           {start_loss:.6f}")
    print(f"  Best loss:            {best_loss:.6f}")
    print(f"  Final loss:           {final_loss:.6f}")
    print(f"  Relative reduction:   {relative_reduction_pct:.2f}%")
    print(f"  CSV log:              {csv_path}")
    print(f"  JSON log:             {json_path}")
    print(f"  Plot:                 {plot_path}")
    print(f"  Summary JSON:         {summary_path}")
    print(f"  Latest pointer:       {latest_pointer_path}")


if __name__ == "__main__":
    main()
