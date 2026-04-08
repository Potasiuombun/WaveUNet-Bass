#!/usr/bin/env python3
"""Stage-2 detectability fine-tuning for WaveUNet baseline.

This script fine-tunes a stage-1 checkpoint using a mixed objective that keeps
reconstruction active and adds optional detectability with a small weight.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.serialized import get_dataset_info, load_serialized_dataset
from src.losses.perceptual import DetectabilityLossWrapper
from src.losses.reconstruction import L1Loss, NMSELoss
from src.losses.spectral import MultiResolutionSTFTLoss
from src.models.factory import create_model
from src.training.callbacks import EarlyStopper
from src.training.checkpointing import Checkpoint
from src.training.metrics import loudness_proxy, mae, nmse
from src.utils.logging import CSVLogger, JSONLogger
from src.utils.seed import set_seed


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config from disk."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config keys to a stable schema for stage-2 fine-tuning."""
    cfg = dict(config)
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("finetune", {})

    if "name" in cfg["model"] and "type" not in cfg["model"]:
        cfg["model"]["type"] = cfg["model"]["name"]
    if "output_head" in cfg["model"] and "output_activation" not in cfg["model"]:
        head = cfg["model"]["output_head"]
        cfg["model"]["output_activation"] = None if head == "identity" else head
    cfg["model"].setdefault("activation", "leaky_relu")
    cfg["model"].setdefault("output_activation", None)

    loss_cfg = cfg["loss"]
    loss_cfg.setdefault("alpha_reconstruction", 1.0)
    loss_cfg.setdefault("gamma_mrstft", 1.0)
    loss_cfg.setdefault("beta_detectability", 0.02)
    loss_cfg.setdefault("delta_peak", 0.0)
    loss_cfg.setdefault("beta_kl", 0.0)
    loss_cfg.setdefault("l1_weight", 1.0)
    loss_cfg.setdefault("nmse_weight", 1.0)
    loss_cfg.setdefault("peak_limit", 1.0)
    loss_cfg.setdefault("mrstft_fft_sizes", [1024, 512, 256])
    loss_cfg.setdefault("mrstft_hop_sizes", [512, 256, 128])

    finetune_cfg = cfg["finetune"]
    finetune_cfg.setdefault("batch_size", 16)
    finetune_cfg.setdefault("epochs", 10)
    finetune_cfg.setdefault("optimizer", "adamw")
    finetune_cfg.setdefault("lr", 5e-5)
    finetune_cfg.setdefault("weight_decay", 1e-4)
    finetune_cfg.setdefault("gradient_clip", 1.0)
    finetune_cfg.setdefault("early_stopping_patience", 5)
    finetune_cfg.setdefault("early_stopping_monitor", "val_total")
    finetune_cfg.setdefault("freeze_policy", "discriminative_lr")
    finetune_cfg.setdefault("encoder_lr_scale", 0.2)
    finetune_cfg.setdefault("decoder_lr_scale", 1.0)
    finetune_cfg.setdefault("detectability_enabled", False)
    finetune_cfg.setdefault("save_dir", "checkpoints/detectability")

    if "logging" not in cfg:
        exp = cfg.get("experiment_name", "detectability_finetune")
        cfg["logging"] = {
            "csv_log": f"logs/{exp}_metrics.csv",
            "json_log": f"logs/{exp}_metrics.jsonl",
        }

    return cfg


def resolve_device(requested_device: Optional[str]) -> str:
    """Resolve runtime device with safe CUDA fallback."""
    if requested_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"

    device = requested_device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but unavailable. Falling back to CPU.")
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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def apply_freeze_policy(model: torch.nn.Module, policy: str) -> None:
    """Apply fine-tuning freeze policy to model parameters.

    Supported policies:
    - none
    - encoder
    - early_encoder_only
    - discriminative_lr (no freezing; handled in optimizer groups)
    """
    policy_name = policy.lower()

    for p in model.parameters():
        p.requires_grad = True

    if policy_name == "none":
        return

    if not hasattr(model, "encoder"):
        if policy_name in {"encoder", "early_encoder_only"}:
            print("⚠️  Model has no encoder attribute; skipping requested freeze policy.")
        return

    encoder = getattr(model, "encoder")

    if policy_name == "encoder":
        for p in encoder.parameters():
            p.requires_grad = False
        return

    if policy_name == "early_encoder_only":
        if isinstance(encoder, torch.nn.ModuleList):
            cutoff = max(1, len(encoder) // 2)
            for idx, module in enumerate(encoder):
                if idx < cutoff:
                    for p in module.parameters():
                        p.requires_grad = False
        else:
            modules = list(encoder.children())
            cutoff = max(1, len(modules) // 2)
            for idx, module in enumerate(modules):
                if idx < cutoff:
                    for p in module.parameters():
                        p.requires_grad = False
        return

    if policy_name == "discriminative_lr":
        return

    raise ValueError(
        "Unsupported freeze_policy. Use one of: none, encoder, early_encoder_only, discriminative_lr"
    )


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build optimizer with optional discriminative learning rates."""
    finetune_cfg = cfg["finetune"]
    base_lr = float(finetune_cfg["lr"])
    weight_decay = float(finetune_cfg["weight_decay"])
    opt_name = str(finetune_cfg["optimizer"]).lower()
    policy = str(finetune_cfg["freeze_policy"]).lower()

    if policy == "discriminative_lr" and hasattr(model, "encoder"):
        encoder_params: List[torch.nn.Parameter] = []
        non_encoder_params: List[torch.nn.Parameter] = []

        encoder_param_ids = {id(p) for p in model.encoder.parameters()}
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in encoder_param_ids:
                encoder_params.append(p)
            else:
                non_encoder_params.append(p)

        param_groups: List[Dict[str, Any]] = []
        if encoder_params:
            param_groups.append(
                {
                    "params": encoder_params,
                    "lr": base_lr * float(finetune_cfg["encoder_lr_scale"]),
                    "weight_decay": weight_decay,
                }
            )
        if non_encoder_params:
            param_groups.append(
                {
                    "params": non_encoder_params,
                    "lr": base_lr * float(finetune_cfg["decoder_lr_scale"]),
                    "weight_decay": weight_decay,
                }
            )

        if not param_groups:
            raise ValueError("No trainable parameters found after freeze policy.")

        if opt_name == "adamw":
            return torch.optim.AdamW(param_groups)
        if opt_name == "adam":
            return torch.optim.Adam(param_groups)
        if opt_name == "sgd":
            return torch.optim.SGD(param_groups, momentum=0.9)
        raise ValueError(f"Unknown optimizer '{opt_name}'")

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found after freeze policy.")

    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{opt_name}'")


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: str, device: str) -> Dict[str, Any]:
    """Load model weights from checkpoint file."""
    state = torch.load(checkpoint_path, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"], strict=True)
        return state

    # fallback for weights-only files
    model.load_state_dict(state, strict=True)
    return {"epoch": 0, "metrics": {}}


def compute_peak_violation(output: torch.Tensor, limit: float) -> torch.Tensor:
    """Compute peak violation penalty over samples exceeding abs(limit)."""
    excess = torch.relu(torch.abs(output) - limit)
    return torch.mean(excess)


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute mean KL divergence for Gaussian latent posterior."""
    return torch.mean(-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()))


def unpack_model_output(model_output: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Unpack model output across plain and VAE-style models."""
    if isinstance(model_output, dict):
        recon = model_output.get("recon")
        if recon is None:
            raise KeyError("Model output dict must contain 'recon' for stage-2 fine-tuning.")
        return recon, model_output.get("mu"), model_output.get("logvar")

    if isinstance(model_output, torch.Tensor):
        return model_output, None, None

    raise TypeError("Unsupported model output type for fine-tuning.")


def step_losses(
    model_output: Any,
    target: torch.Tensor,
    l1_loss: L1Loss,
    nmse_loss: NMSELoss,
    mrstft_loss: MultiResolutionSTFTLoss,
    detectability_loss: DetectabilityLossWrapper,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute stage-2 mixed loss terms and weighted total."""
    loss_cfg = cfg["loss"]

    output, mu, logvar = unpack_model_output(model_output)

    recon_l1 = l1_loss(output, target)
    recon_nmse = nmse_loss(output, target)
    reconstruction = (
        float(loss_cfg["l1_weight"]) * recon_l1
        + float(loss_cfg["nmse_weight"]) * recon_nmse
    )

    mrstft = mrstft_loss(output, target)
    detectability = detectability_loss(output, target)
    peak_penalty = compute_peak_violation(output, float(loss_cfg["peak_limit"]))
    kl = (
        compute_kl_divergence(mu, logvar)
        if (mu is not None and logvar is not None)
        else torch.zeros((), device=output.device, dtype=output.dtype)
    )

    total = (
        float(loss_cfg["alpha_reconstruction"]) * reconstruction
        + float(loss_cfg["gamma_mrstft"]) * mrstft
        + float(loss_cfg["beta_detectability"]) * detectability
        + float(loss_cfg["delta_peak"]) * peak_penalty
        + float(loss_cfg["beta_kl"]) * kl
    )

    return output, {
        "total": total,
        "reconstruction": reconstruction,
        "recon_l1": recon_l1,
        "recon_nmse": recon_nmse,
        "mrstft": mrstft,
        "detectability": detectability,
        "peak_penalty": peak_penalty,
        "kl": kl,
    }


def to_float_dict(values: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Convert tensor metrics dict to float metrics dict."""
    out: Dict[str, float] = {}
    for key, val in values.items():
        out[key] = float(val.detach().cpu().item())
    return out


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    cfg: Dict[str, Any],
    l1_loss: L1Loss,
    nmse_loss: NMSELoss,
    mrstft_loss: MultiResolutionSTFTLoss,
    detectability_loss: DetectabilityLossWrapper,
) -> Dict[str, float]:
    """Train for one fine-tune epoch."""
    model.train()
    grad_clip = float(cfg["finetune"]["gradient_clip"])

    sums: Dict[str, float] = {
        "train_total": 0.0,
        "train_reconstruction": 0.0,
        "train_mrstft": 0.0,
        "train_detectability": 0.0,
        "train_peak_penalty": 0.0,
        "train_kl": 0.0,
        "train_nmse": 0.0,
        "train_mae": 0.0,
        "train_loudness_proxy": 0.0,
        "train_peak_violation": 0.0,
    }
    n_samples = 0

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        bs = inputs.shape[0]

        optimizer.zero_grad(set_to_none=True)
        model_outputs = model(inputs)

        outputs, loss_terms = step_losses(
            model_outputs,
            targets,
            l1_loss,
            nmse_loss,
            mrstft_loss,
            detectability_loss,
            cfg,
        )
        total_loss = loss_terms["total"]

        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        scalars = to_float_dict(loss_terms)
        sums["train_total"] += scalars["total"] * bs
        sums["train_reconstruction"] += scalars["reconstruction"] * bs
        sums["train_mrstft"] += scalars["mrstft"] * bs
        sums["train_detectability"] += scalars["detectability"] * bs
        sums["train_peak_penalty"] += scalars["peak_penalty"] * bs
        sums["train_kl"] += scalars["kl"] * bs
        sums["train_nmse"] += nmse(outputs, targets) * bs
        sums["train_mae"] += mae(outputs, targets) * bs
        sums["train_loudness_proxy"] += loudness_proxy(outputs) * bs
        sums["train_peak_violation"] += float(compute_peak_violation(outputs, float(cfg["loss"]["peak_limit"])).item()) * bs
        n_samples += bs

    return {k: (v / max(1, n_samples)) for k, v in sums.items()}


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    cfg: Dict[str, Any],
    l1_loss: L1Loss,
    nmse_loss: NMSELoss,
    mrstft_loss: MultiResolutionSTFTLoss,
    detectability_loss: DetectabilityLossWrapper,
) -> Dict[str, float]:
    """Validate for one fine-tune epoch."""
    model.eval()

    sums: Dict[str, float] = {
        "val_total": 0.0,
        "val_reconstruction": 0.0,
        "val_mrstft": 0.0,
        "val_detectability": 0.0,
        "val_peak_penalty": 0.0,
        "val_kl": 0.0,
        "val_nmse": 0.0,
        "val_mae": 0.0,
        "val_loudness_proxy": 0.0,
        "val_peak_violation": 0.0,
    }
    n_samples = 0

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        bs = inputs.shape[0]

        model_outputs = model(inputs)
        outputs, loss_terms = step_losses(
            model_outputs,
            targets,
            l1_loss,
            nmse_loss,
            mrstft_loss,
            detectability_loss,
            cfg,
        )

        scalars = to_float_dict(loss_terms)
        sums["val_total"] += scalars["total"] * bs
        sums["val_reconstruction"] += scalars["reconstruction"] * bs
        sums["val_mrstft"] += scalars["mrstft"] * bs
        sums["val_detectability"] += scalars["detectability"] * bs
        sums["val_peak_penalty"] += scalars["peak_penalty"] * bs
        sums["val_kl"] += scalars["kl"] * bs
        sums["val_nmse"] += nmse(outputs, targets) * bs
        sums["val_mae"] += mae(outputs, targets) * bs
        sums["val_loudness_proxy"] += loudness_proxy(outputs) * bs
        sums["val_peak_violation"] += float(compute_peak_violation(outputs, float(cfg["loss"]["peak_limit"])).item()) * bs
        n_samples += bs

    return {k: (v / max(1, n_samples)) for k, v in sums.items()}


def main() -> None:
    """Run stage-2 detectability fine-tuning."""
    parser = argparse.ArgumentParser(description="Stage-2 detectability fine-tuning")
    parser.add_argument("--config", default="configs/detectability_finetune.yaml")
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--checkpoint", default=None, help="Stage-1 checkpoint path")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = normalize_config(load_config(args.config))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = resolve_device(args.device)
    dataset_file = resolve_existing_path(
        args.dataset_file or cfg.get("data", {}).get("dataset_file"),
        args.config,
    )
    if not dataset_file:
        raise ValueError("Missing dataset path. Provide --dataset-file or data.dataset_file in config.")

    checkpoint_path = resolve_existing_path(
        args.checkpoint or cfg.get("finetune", {}).get("checkpoint"),
        args.config,
    )
    if not checkpoint_path:
        raise ValueError("Missing stage-1 checkpoint. Provide --checkpoint or finetune.checkpoint in config.")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    info = get_dataset_info(dataset_file)
    print("Startup:")
    print(f"  Config path: {args.config}")
    print(f"  Dataset path: {dataset_file}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"  Batch size: {cfg['finetune']['batch_size']}")
    print(f"  Freeze policy: {cfg['finetune']['freeze_policy']}")
    print(
        "  Loss weights: "
        f"alpha_reconstruction={cfg['loss']['alpha_reconstruction']}, "
        f"gamma_mrstft={cfg['loss']['gamma_mrstft']}, "
        f"beta_detectability={cfg['loss']['beta_detectability']}, "
        f"delta_peak={cfg['loss']['delta_peak']}, "
        f"beta_kl={cfg['loss']['beta_kl']}"
    )
    print(f"  Detectability enabled: {cfg['finetune']['detectability_enabled']}")
    print(f"  Dataset split type: {info['split_type']}")

    train_loader = load_serialized_dataset(
        dataset_file,
        split="train",
        batch_size=int(cfg["finetune"]["batch_size"]),
        num_workers=2,
        validate=False,
    )
    val_loader = load_serialized_dataset(
        dataset_file,
        split="val",
        batch_size=int(cfg["finetune"]["batch_size"]),
        num_workers=2,
        validate=False,
    )

    split_counts = {
        "train": len(train_loader.dataset),
        "val": len(val_loader.dataset),
    }
    print(f"  Train frames: {split_counts['train']}")
    print(f"  Val frames: {split_counts['val']}")

    model = create_model(cfg["model"], device=device)
    load_meta = load_checkpoint_weights(model, checkpoint_path, device)

    apply_freeze_policy(model, str(cfg["finetune"]["freeze_policy"]))
    optimizer = build_optimizer(model, cfg)

    l1_loss = L1Loss()
    nmse_loss = NMSELoss()
    mrstft_loss = MultiResolutionSTFTLoss(
        fft_sizes=list(cfg["loss"]["mrstft_fft_sizes"]),
        hop_sizes=list(cfg["loss"]["mrstft_hop_sizes"]),
        window="hann",
    )
    detectability_loss = DetectabilityLossWrapper(
        enabled=bool(cfg["finetune"]["detectability_enabled"]),
        sample_rate=int(cfg.get("data", {}).get("sr", 48000)),
        frame_size=int(cfg.get("data", {}).get("frame_size", 1024)),
    )

    csv_logger = CSVLogger(cfg["logging"]["csv_log"])
    json_logger = JSONLogger(cfg["logging"]["json_log"])

    exp_name = str(cfg.get("experiment_name", "detectability_finetune"))
    save_dir = Path(cfg["finetune"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path("logs") / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = run_dir / "config_snapshot.yaml"
    with open(config_snapshot, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    monitor = str(cfg["finetune"]["early_stopping_monitor"])
    stopper = EarlyStopper(patience=int(cfg["finetune"]["early_stopping_patience"]))

    num_epochs = int(cfg["finetune"]["epochs"])
    start_time = time.time()
    best_monitor_value = float("inf")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg,
            l1_loss,
            nmse_loss,
            mrstft_loss,
            detectability_loss,
        )
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            device,
            cfg,
            l1_loss,
            nmse_loss,
            mrstft_loss,
            detectability_loss,
        )

        epoch_time = time.time() - epoch_start
        metrics = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "lr_group0": float(optimizer.param_groups[0]["lr"]),
            "time_sec": float(epoch_time),
        }

        csv_logger.log(metrics)
        json_logger.log(metrics)

        monitor_value = float(metrics.get(monitor, metrics.get("val_total", float("inf"))))
        is_best = monitor_value < best_monitor_value
        if is_best:
            best_monitor_value = monitor_value
            Checkpoint.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                filepath=str(save_dir / "best_stage2.pth"),
                scheduler=None,
            )

        Checkpoint.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            filepath=str(save_dir / "latest_stage2.pth"),
            scheduler=None,
        )

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"train_total={metrics['train_total']:.6f} "
            f"val_total={metrics['val_total']:.6f} "
            f"val_nmse={metrics['val_nmse']:.6f} "
            f"val_detectability={metrics['val_detectability']:.6f}"
        )

        if stopper.step(monitor_value, epoch):
            print(
                f"Early stopping at epoch {epoch} "
                f"(best {monitor}={stopper.best_loss:.6f} at epoch {stopper.best_epoch})"
            )
            break

    run_summary = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config_path": args.config,
        "dataset_path": dataset_file,
        "checkpoint_path": checkpoint_path,
        "loaded_checkpoint_epoch": int(load_meta.get("epoch", 0)),
        "device": device,
        "freeze_policy": str(cfg["finetune"]["freeze_policy"]),
        "detectability_enabled": bool(cfg["finetune"]["detectability_enabled"]),
        "monitor": monitor,
        "best_monitor_value": float(best_monitor_value),
        "split_counts": split_counts,
        "csv_log": cfg["logging"]["csv_log"],
        "json_log": cfg["logging"]["json_log"],
        "checkpoint_best": str(save_dir / "best_stage2.pth"),
        "checkpoint_latest": str(save_dir / "latest_stage2.pth"),
        "runtime_sec": float(time.time() - start_time),
    }
    write_json(run_dir / "run_summary.json", run_summary)
    print(f"Run summary written: {run_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
